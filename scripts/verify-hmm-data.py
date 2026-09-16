"""Regenerate and verify the recorded data of the hidden Markov models lesson.

Two separate jobs, both read-only unless given `--write`:

1.  **The supervised tagging study.** The declared protocol of
    `docs/teaching/drafts/hidden-markov-models-hmm/lesson.md` section 8 is
    recomputed from the extract this lesson serves: the coarse three-way label
    map, a 146-symbol vocabulary from training words occurring at least twice,
    two smoothing strengths, four decoder configurations, every development
    prediction and every per-token belief. Every value is matched against the
    content packet's `calculated-inputs.json` before
    `src/learn/data/hmm-data.js` is rebuilt. Without `--write` the rebuilt
    module text must be byte-identical to the file already on disk.

2.  **The packet's own trust root.** `calculated-inputs.json` is what
    `scripts/verify-hmm-models.mjs` checks the browser models against, so it
    cannot be the only witness for itself. Every scalar leaf of it is
    re-derived here through an implementation independent of the packet's
    author program:

      * the author program works in log space throughout, with
        `np.logaddexp.reduce`; this file works by RETAINED SCALING, normalising
        each forward column and dividing the backward pass by the same factors.
      * the forward evidence and the best path are additionally checked against
        exhaustive enumeration of all 16 paths, which uses no recursion at all.
      * the smoothing formulas of section 8 are applied to integer counts as
        the manuscript writes them, rather than by accumulating onto a
        pre-seeded float array as the author program does.
      * `hmm-experiments.py` is never imported.

    The declared stochastic protocols - the seed-71 sampling of twelve
    length-30 recordings and the Dirichlet starts of seeds 3, 7, 19 - are
    reproduced through the same NumPy Generator calls, because a different call
    order would produce different DATA rather than a different route to the same
    answer. The EM arithmetic applied to that data is independent.

Coverage is measured as ASSERTED LEAF PATHS of the packet, not as a
hand-maintained tally, and whatever is not covered is named in the evidence.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-hmm-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-hmm-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/hidden-markov-models-hmm"
PACKET_INPUTS = PACKET / "calculated-inputs.json"
PACKET_SEQUENCES = PACKET / "ewt-sequences.json"
ASSET_DIR = ROOT / "public/learn-assets/hmm"
ASSET_SEQUENCES = ASSET_DIR / "ewt-sequences.json"
MODULE = ROOT / "src/learn/data/hmm-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/hmm-data.json"

EXPECTED_SEQUENCES_SHA = "e801e665c4e2a6fa002e04ebb52b00b4fbc1421c7f029494565c0ecbaa016a4d"
EXPECTED_SEQUENCES_BYTES = 89597

#: Declared protocol constants, all from the packet's provenance record.
COARSE_NAMES = ["Noun", "Verb", "Other"]
ORIGINAL_GROUPS = [["NOUN", "PROPN"], ["VERB", "AUX"], ["every other original UPOS tag"]]
MINIMUM_WORD_COUNT = 2
SMOOTHINGS = [0.1, 1.0]
DATA_SEED = 71
SEQUENCE_COUNT = 12
SEQUENCE_LENGTH = 30
FIT_SEEDS = [3, 7, 19]
FIT_STEPS = 40
UNIFORM_STEPS = 3
MISSING = -1
NL = chr(10)

checks: dict[str, int] = {}
failures: list[str] = []


def record(name: str) -> None:
    checks[name] = checks.get(name, 0) + 1


def expect(condition: bool, label: str) -> None:
    record(label.split(" - ")[0])
    if not condition:
        failures.append(label)


def close(actual, expected, label: str, tolerance: float = 1e-12) -> None:
    actual = float(actual)
    expected = float(expected)
    expect(abs(actual - expected) <= tolerance * max(1.0, abs(expected)),
           f"{label} - {actual!r} versus {expected!r}")


def close_all(actual, expected, label: str, tolerance: float = 1e-12) -> None:
    flat_a = np.asarray(actual, dtype=float).ravel()
    flat_b = np.asarray(expected, dtype=float).ravel()
    expect(flat_a.shape == flat_b.shape, f"{label} - shape {flat_a.shape} versus {flat_b.shape}")
    if flat_a.shape != flat_b.shape:
        return
    worst = float(np.max(np.abs(flat_a - flat_b) / np.maximum(1.0, np.abs(flat_b)))) if flat_a.size else 0.0
    expect(worst <= tolerance, f"{label} - worst relative gap {worst:.3e} over {flat_a.size} values")


# ------------------------------------------------- an independent implementation

def local_likelihood(emission: np.ndarray, value: int) -> np.ndarray:
    """A missing report marginalises over every symbol, giving likelihood one."""
    if value == MISSING:
        return np.ones(emission.shape[0])
    return emission[:, value].astype(float)


def forward_scaled(start, transition, emission, observations):
    """Filtering with every scale factor retained. No logarithm is taken here."""
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    factors = []
    filtered = []
    belief = start.copy()
    for time, value in enumerate(observations):
        local = local_likelihood(emission, value)
        predicted = belief if time == 0 else belief @ transition
        mass = predicted * local
        factor = float(mass.sum())
        if factor == 0.0:
            factors.append(0.0)
            return {"factors": factors, "filtered": np.asarray(filtered), "impossible": True}
        belief = mass / factor
        factors.append(factor)
        filtered.append(belief.copy())
    return {"factors": factors, "filtered": np.asarray(filtered), "impossible": False}


def raw_trellis(start, transition, emission, observations, operator):
    """Unscaled joint masses, the numbers the manuscript prints and a figure draws.

    `operator` is `sum` for the forward recursion and `max` for Viterbi; the
    predecessor a Viterbi cell stores is chosen on `previous * transition`,
    BEFORE the shared destination emission, and the first index owns a tie.
    """
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    states = len(start)
    cells = np.zeros((len(observations), states))
    predecessor = np.zeros((len(observations), states), dtype=int)
    cells[0] = start * local_likelihood(emission, observations[0])
    for time in range(1, len(observations)):
        local = local_likelihood(emission, observations[time])
        for destination in range(states):
            carried = cells[time - 1] * transition[:, destination]
            if operator == "sum":
                cells[time, destination] = carried.sum() * local[destination]
            else:
                winner = 0
                for origin in range(1, states):
                    if carried[origin] > carried[winner]:
                        winner = origin
                predecessor[time, destination] = winner
                cells[time, destination] = carried[winner] * local[destination]
    return cells, predecessor


def infer(start, transition, emission, observations):
    """Complete inference by retained scaling, with no log-space anywhere."""
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    states = len(start)
    length = len(observations)
    scaled = forward_scaled(start, transition, emission, observations)
    if scaled["impossible"]:
        return {"impossible": True, "evidence": 0.0, "log_evidence": -np.inf}
    factors = scaled["factors"]
    filtered = scaled["filtered"]
    backward = np.zeros((length, states))
    backward[length - 1] = 1.0
    for time in range(length - 2, -1, -1):
        local = local_likelihood(emission, observations[time + 1])
        backward[time] = (transition * local * backward[time + 1]).sum(axis=1) / factors[time + 1]
    smoothed = filtered * backward
    pair = np.zeros((max(length - 1, 0), states, states))
    for time in range(length - 1):
        local = local_likelihood(emission, observations[time + 1])
        pair[time] = (filtered[time][:, None] * transition * local * backward[time + 1]) / factors[time + 1]
    best, predecessor = raw_trellis(start, transition, emission, observations, "max")
    path = np.zeros(length, dtype=int)
    path[-1] = int(np.argmax(best[-1]))
    for time in range(length - 2, -1, -1):
        path[time] = predecessor[time + 1, path[time + 1]]
    evidence = float(np.prod(factors))
    joint = float(best[-1, path[-1]])
    # The raw joint masses can underflow to zero while the scaled log evidence
    # stays finite. That is the section 7 distinction, not an error, so the
    # ratio is reported as undefined rather than computed as zero over zero.
    underflowed = evidence == 0.0
    return {
        "impossible": False,
        "factors": factors,
        "filtered": filtered,
        "backward_scaled": backward,
        "smoothed": smoothed,
        "pair": pair,
        "path": path,
        "path_joint": joint,
        "path_posterior": None if underflowed else joint / evidence,
        "marginal_modes": np.argmax(smoothed, axis=1),
        "evidence": evidence,
        "underflowed": underflowed,
        "log_evidence": float(np.log(factors).sum()),
        "viterbi_cells": best,
        "predecessor": predecessor,
    }


def path_joint(start, transition, emission, path, observations) -> float:
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    value = start[path[0]] * local_likelihood(emission, observations[0])[path[0]]
    for time in range(1, len(path)):
        value *= transition[path[time - 1], path[time]]
        value *= local_likelihood(emission, observations[time])[path[time]]
    return float(value)


def enumerate_paths(start, transition, emission, observations):
    """All N**T paths with their joint probabilities. No recursion, no trellis."""
    states = len(start)
    entries = [(list(path), path_joint(start, transition, emission, path, observations))
               for path in product(range(states), repeat=len(observations))]
    return entries


def viterbi_log_route(start, transition, emission, observations):
    """Viterbi in log space with `np.argmax`, the AUTHOR PROGRAM'S OWN arithmetic.

    This is deliberately not an independent route, and it is used only for the
    real tagger. Three of the forty development sentences have TWO paths of
    exactly equal joint probability, so which one a decoder reports is decided
    by its arithmetic rather than by the model. Reproducing the declared
    program's arithmetic is the only way to reproduce which member of the tie
    it recorded. The independent guarantee is supplied separately by
    `exact_optimum`, which proves in exact rational arithmetic that every stored
    path really is a maximiser, and reports what the alternative admissible tie
    rules would have scored.
    """
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    states = len(start)
    with np.errstate(divide="ignore"):
        log_transition = np.log(transition)
        local = np.array([np.log(local_likelihood(emission, value)) for value in observations])
    best = np.empty_like(local)
    predecessor = np.zeros(local.shape, dtype=int)
    best[0] = np.log(start) + local[0]
    for time in range(1, len(observations)):
        candidates = best[time - 1][:, None] + log_transition
        predecessor[time] = np.argmax(candidates, axis=0)
        best[time] = candidates[predecessor[time], np.arange(states)] + local[time]
    path = np.zeros(len(observations), dtype=int)
    path[-1] = int(np.argmax(best[-1]))
    for time in range(len(observations) - 2, -1, -1):
        path[time] = predecessor[time + 1, path[time + 1]]
    return path.tolist()


def exact_optimum(start, transition, emission, observations):
    """The exact optimal joint mass and the exact NUMBER of optimal paths.

    Everything is a `Fraction`, so an equality here is an equality and not a
    tolerance. `ways` counts how many distinct paths attain the optimum, which
    is what turns "these two happened to come out the same" into a proved tie.
    """
    states = len(start)
    best = [[start[i] * emission[i][observations[0]] for i in range(states)]]
    ways = [[1] * states]
    for time in range(1, len(observations)):
        row, counted = [], []
        for destination in range(states):
            carried = [best[time - 1][origin] * transition[origin][destination] for origin in range(states)]
            top = max(carried)
            row.append(top * emission[destination][observations[time]])
            counted.append(sum(ways[time - 1][origin] for origin in range(states) if carried[origin] == top))
        best.append(row)
        ways.append(counted)
    top = max(best[-1])
    return top, sum(ways[-1][i] for i in range(states) if best[-1][i] == top)


def exact_joint(start, transition, emission, path, observations):
    """One path's joint probability, in exact rational arithmetic."""
    value = start[path[0]] * emission[path[0]][observations[0]]
    for time in range(1, len(path)):
        value *= transition[path[time - 1]][path[time]] * emission[path[time]][observations[time]]
    return value


def exact_viterbi_path(start, transition, emission, observations, tie: str):
    """An exact-arithmetic Viterbi whose tie rule is stated rather than inherited."""
    states = len(start)
    best = [[start[i] * emission[i][observations[0]] for i in range(states)]]
    predecessor = [[None] * states]
    pick = (lambda winners: winners[0]) if tie == "first" else (lambda winners: winners[-1])
    for time in range(1, len(observations)):
        row, chosen = [], []
        for destination in range(states):
            carried = [best[time - 1][origin] * transition[origin][destination] for origin in range(states)]
            top = max(carried)
            chosen.append(pick([o for o in range(states) if carried[o] == top]))
            row.append(top * emission[destination][observations[time]])
        best.append(row)
        predecessor.append(chosen)
    top = max(best[-1])
    path = [0] * len(observations)
    path[-1] = pick([i for i in range(states) if best[-1][i] == top])
    for time in range(len(observations) - 2, -1, -1):
        path[time] = predecessor[time + 1][path[time + 1]]
    return path


def expected_counts(start, transition, emission, sequences):
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    initial = np.zeros_like(start)
    edge = np.zeros_like(transition)
    symbol = np.zeros_like(emission)
    log_likelihood = 0.0
    for observations in sequences:
        result = infer(start, transition, emission, observations)
        initial += result["smoothed"][0]
        if len(observations) > 1:
            edge += result["pair"].sum(axis=0)
        for time, value in enumerate(observations):
            if value != MISSING:
                symbol[:, value] += result["smoothed"][time]
        log_likelihood += result["log_evidence"]
    return initial, edge, symbol, log_likelihood


def normalize_counts(counts: np.ndarray, previous: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total, out=previous.copy(), where=total > 0)


def em_step(start, transition, emission, sequences):
    initial, edge, symbol, _ = expected_counts(start, transition, emission, sequences)
    return (initial / initial.sum(),
            normalize_counts(edge, np.asarray(transition, dtype=float)),
            normalize_counts(symbol, np.asarray(emission, dtype=float)))


def sample_sequences(start, transition, emission, count, length, seed):
    """The DECLARED sampling protocol, reproduced call for call.

    A different call order would draw different data, so this is reproduction of
    a stated recipe rather than an independent route. Everything computed FROM
    the sampled data below is independent.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    sequences = []
    for _ in range(count):
        state = rng.choice(len(start), p=start)
        observations = []
        for time in range(length):
            if time:
                state = rng.choice(len(start), p=transition[state])
            observations.append(int(rng.choice(emission.shape[1], p=emission[state])))
        sequences.append(observations)
    return sequences


def fit_em(sequences, states, symbols, seed, steps, uniform=False):
    rng = np.random.default_rng(seed)
    model = (rng.dirichlet(np.ones(states)),
             rng.dirichlet(np.ones(states), size=states),
             rng.dirichlet(np.ones(symbols), size=states))
    if uniform:
        model = (np.ones(states) / states,
                 np.ones((states, states)) / states,
                 np.ones((states, symbols)) / symbols)
    checkpoints = [tuple(part.copy() for part in model)]
    history = [expected_counts(*model, sequences)[-1]]
    for _ in range(steps):
        model = em_step(*model, sequences)
        checkpoints.append(tuple(np.asarray(part).copy() for part in model))
        history.append(expected_counts(*model, sequences)[-1])
    return {"history": history, "checkpoints": checkpoints, "model": model}


# ---------------------------------------------------------------- leaf coverage

def leaf_paths(node, prefix=""):
    """Every scalar leaf of a nested structure, as a dotted path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}[{index}]")
    else:
        yield prefix


def resolve(packet, path: str):
    """Follow a dotted/bracketed path into the packet, or raise."""
    node = packet
    token = ""
    index = 0
    while index < len(path):
        character = path[index]
        if character == ".":
            if token:
                node = node[token]
                token = ""
            index += 1
        elif character == "[":
            if token:
                node = node[token]
                token = ""
            end = path.index("]", index)
            node = node[int(path[index + 1:end])]
            index = end + 1
        else:
            token += character
            index += 1
    if token:
        node = node[token]
    return node


class Coverage:
    """Leaf paths a re-derivation genuinely pins, recorded as it goes."""

    def __init__(self, packet):
        self.packet = packet
        self.covered: set[str] = set()
        self.total = set(leaf_paths(packet))

    def mark(self, *paths: str) -> None:
        for path in paths:
            try:
                node = resolve(self.packet, path)
            except (KeyError, IndexError, TypeError) as error:
                raise AssertionError(f"coverage claimed for a path that is not in the packet: {path}") from error
            for leaf in leaf_paths(node, path):
                if leaf not in self.total:
                    raise AssertionError(f"coverage claimed for a leaf outside the packet: {leaf}")
                self.covered.add(leaf)


# ------------------------------------------------------- mechanisms trust root

def rederive_mechanisms(packet, cover: Coverage) -> dict:
    """Every mechanism value, from the declared model and reports alone."""
    block = packet["mechanisms"]
    start = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    observations = [0, 1, 0, 2]
    close_all(start, block["model"]["start"], "trust root - declared initial row", 0.0)
    close_all(transition, block["model"]["transition"], "trust root - declared transition matrix", 0.0)
    close_all(emission, block["model"]["emission"], "trust root - declared emission matrix", 0.0)
    expect(block["observations"] == observations, "trust root - the declared four reports")
    cover.mark("mechanisms.model", "mechanisms.observations")

    # Every probability row is a distribution, checked before anything uses it.
    for index, row in enumerate(transition):
        close(row.sum(), 1.0, f"trust root - transition row {index} sums to one", 1e-15)
    for index, row in enumerate(emission):
        close(row.sum(), 1.0, f"trust root - emission row {index} sums to one", 1e-15)
    close(start.sum(), 1.0, "trust root - the initial row sums to one", 1e-15)

    main = infer(start, transition, emission, observations)
    packet_main = block["main"]
    close_all(np.log(raw_trellis(start, transition, emission, observations, "sum")[0]),
              packet_main["forward_log"], "trust root - forward masses", 1e-12)
    close_all(main["filtered"], packet_main["filtered"], "trust root - filtered rows", 1e-12)
    close_all(main["smoothed"], packet_main["smoothed"], "trust root - smoothed rows", 1e-11)
    close_all(main["pair"], packet_main["pair"], "trust root - pair posteriors", 1e-11)
    close_all(np.log(main["viterbi_cells"]), packet_main["viterbi_log"], "trust root - Viterbi cells", 1e-12)
    expect(main["predecessor"].tolist() == packet_main["predecessor"],
           "trust root - stored Viterbi predecessors")
    expect(main["path"].tolist() == packet_main["path"], "trust root - the decoded best path")
    expect(main["marginal_modes"].tolist() == packet_main["marginal_modes"],
           "trust root - the pointwise marginal modes")
    close(main["log_evidence"], packet_main["log_evidence"], "trust root - log evidence", 1e-13)
    close(main["path_joint"], packet_main["path_joint"], "trust root - best path joint mass", 1e-13)
    close(main["path_posterior"], packet_main["path_posterior"], "trust root - best path posterior", 1e-13)
    # The scaled backward row recovers the packet's unscaled beta exactly.
    tail = np.array([float(np.prod(main["factors"][time + 1:])) for time in range(len(observations))])
    close_all(np.log(main["backward_scaled"] * tail[:, None]), packet_main["backward_log"],
              "trust root - backward likelihoods", 1e-12)
    cover.mark("mechanisms.main")

    # Third route: exhaustive enumeration, with no recursion at all.
    enumerated = enumerate_paths(start, transition, emission, observations)
    expect(len(enumerated) == 16, "trust root - sixteen paths exist for four binary time steps")
    close(sum(value for _, value in enumerated), main["evidence"],
          "trust root - enumerated paths sum to the forward evidence", 1e-13)
    close(max(value for _, value in enumerated), main["path_joint"],
          "trust root - the largest enumerated path is Viterbi's", 1e-13)
    for index, (states_path, value) in enumerate(enumerated):
        expect(block["enumerated_paths"][index]["states"] == states_path,
               f"trust root - enumerated path {index} order")
        close(value, block["enumerated_paths"][index]["joint"], f"trust root - enumerated path {index} mass", 1e-13)
    cover.mark("mechanisms.enumerated_paths")

    # Pair margins, the identity that makes a count-flow diagram interpretable.
    for time in range(len(observations) - 1):
        close_all(main["pair"][time].sum(axis=1), main["smoothed"][time],
                  f"trust root - pair row margin at {time}", 1e-11)
        close_all(main["pair"][time].sum(axis=0), main["smoothed"][time + 1],
                  f"trust root - pair column margin at {time}", 1e-11)
        close(main["pair"][time].sum(), 1.0, f"trust root - pair block {time} sums to one", 1e-11)

    changed = infer(start, transition, emission, [0, 1, 0, 0])
    close_all(changed["filtered"][:3], main["filtered"][:3],
              "trust root - a changed future leaves the filtered prefix alone", 0.0)
    close_all(changed["filtered"], block["changed_future"]["filtered"], "trust root - changed-future filtering", 1e-12)
    close_all(changed["smoothed"], block["changed_future"]["smoothed"], "trust root - changed-future smoothing", 1e-11)
    close_all(changed["pair"], block["changed_future"]["pair"], "trust root - changed-future pairs", 1e-11)
    close_all(np.log(raw_trellis(start, transition, emission, [0, 1, 0, 0], "sum")[0]),
              block["changed_future"]["forward_log"], "trust root - changed-future forward", 1e-12)
    changed_tail = np.array([float(np.prod(changed["factors"][time + 1:])) for time in range(4)])
    close_all(np.log(changed["backward_scaled"] * changed_tail[:, None]),
              block["changed_future"]["backward_log"], "trust root - changed-future backward", 1e-12)
    close_all(np.log(changed["viterbi_cells"]), block["changed_future"]["viterbi_log"],
              "trust root - changed-future Viterbi", 1e-12)
    expect(changed["predecessor"].tolist() == block["changed_future"]["predecessor"],
           "trust root - changed-future predecessors")
    expect(changed["path"].tolist() == block["changed_future"]["path"] == [1, 1, 1, 1],
           "trust root - a corrected final report makes the whole best path Sunny")
    expect(changed["marginal_modes"].tolist() == block["changed_future"]["marginal_modes"],
           "trust root - changed-future modes")
    close(changed["log_evidence"], block["changed_future"]["log_evidence"], "trust root - changed-future evidence", 1e-13)
    close(changed["path_joint"], block["changed_future"]["path_joint"], "trust root - changed-future path joint", 1e-13)
    close(changed["path_posterior"], block["changed_future"]["path_posterior"],
          "trust root - changed-future path posterior", 1e-13)
    cover.mark("mechanisms.changed_future")

    changed_emission = emission.copy()
    changed_emission[0] = [0.2, 0.3, 0.5]
    altered = infer(start, transition, changed_emission, observations)
    close(altered["path_joint"], main["path_joint"],
          "trust root - a changed Rainy row leaves the best path's joint mass alone", 1e-15)
    close(altered["path_posterior"], 9 / 35, "trust root - but its posterior becomes 9/35", 1e-13)
    _mark_inference(altered, block["changed_emission"], start, transition, changed_emission,
                    observations, "changed-emission")
    cover.mark("mechanisms.changed_emission")

    scaled = forward_scaled(start, transition, emission, observations)
    close_all(scaled["factors"], block["scaled"]["factors"], "trust root - retained scale factors", 1e-13)
    close_all(scaled["filtered"][-1], block["scaled"]["filtered_last"], "trust root - final scaled belief", 1e-12)
    close(float(np.log(scaled["factors"]).sum()), block["scaled"]["log_evidence"],
          "trust root - scaling and log space agree on the evidence", 1e-13)
    close(float(np.prod(scaled["factors"])), 0.00933936,
          "trust root - the four scale factors multiply to the stated evidence", 1e-13)
    cover.mark("mechanisms.scaled")

    missing = infer(start, transition, emission, [0, MISSING, 0, 2])
    close(missing["smoothed"][3][0], 0.802780059665355,
          "trust root - a retained missing report leaves final Rainy at .802780", 1e-12)
    _mark_inference(missing, block["missing_middle"], start, transition, emission,
                    [0, MISSING, 0, 2], "missing-middle")
    cover.mark("mechanisms.missing_middle")

    deleted = infer(start, transition, emission, [0, 0, 2])
    close(deleted["smoothed"][2][0], 0.7953204876130554,
          "trust root - deleting that step instead gives .795320", 1e-12)
    expect(abs(missing["smoothed"][3][0] - deleted["smoothed"][2][0]) > 1e-4,
           "trust root - the two answers really differ, so the contrast is not a rounding artefact")
    _mark_inference(deleted, block["deleted_middle"], start, transition, emission, [0, 0, 2], "deleted-middle")
    cover.mark("mechanisms.deleted_middle")

    # The three-state constrained graph of section 5.
    constrained_start = np.array([0.4, 0.35, 0.25])
    constrained_transition = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    certain = np.ones((3, 1))
    illegal = infer(constrained_start, constrained_transition, certain, [0, 0])
    close_all(illegal["smoothed"][0], [0.4, 0.35, 0.25], "trust root - first-time marginals", 1e-13)
    close_all(illegal["smoothed"][1], [0.6, 0.2, 0.2], "trust root - second-time marginals", 1e-13)
    expect(illegal["marginal_modes"].tolist() == [0, 0], "trust root - the pointwise modes are A then A")
    expect(illegal["path"].tolist() == [1, 0], "trust root - Viterbi returns B then A")
    close(illegal["path_joint"], 0.35, "trust root - whose joint probability is .35", 1e-13)
    close(path_joint(constrained_start, constrained_transition, certain, [0, 0], [0, 0]), 0.0,
          "trust root - while A to A has joint probability exactly zero", 0.0)
    close(float(illegal["smoothed"][0][0] + illegal["smoothed"][1][0]), 1.0,
          "trust root - the pointwise modes expect 1.0 correct positions", 1e-13)
    close(float(illegal["smoothed"][0][1] + illegal["smoothed"][1][0]), 0.95,
          "trust root - and B to A expects .95", 1e-13)
    close(illegal["evidence"], 1.0, "trust root - a certain single symbol carries no evidence", 1e-13)
    # The row sums of the joint mass table are the declared initial probabilities.
    joint_table = np.array([[0.0, 0.20, 0.20], [0.35, 0.0, 0.0], [0.25, 0.0, 0.0]])
    close_all(joint_table.sum(axis=1), constrained_start,
              "trust root - the joint table's row sums are the initial probabilities", 1e-13)
    for index, row in enumerate(joint_table):
        if row.sum() > 0:
            close_all(row / row.sum(), constrained_transition[index],
                      f"trust root - row {index} normalises to its transition row", 1e-13)
    _mark_inference(illegal, block["illegal_modes"], constrained_start, constrained_transition,
                    certain, [0, 0], "illegal-modes")
    cover.mark("mechanisms.illegal_modes")

    changed_prior = infer(np.array([0.2, 0.55, 0.25]), constrained_transition, certain, [0, 0])
    expect(changed_prior["marginal_modes"].tolist() == [1, 0] == changed_prior["path"].tolist(),
           "trust root - under the changed prior both decoders choose B then A")
    close(changed_prior["path_joint"], 0.55, "trust root - with joint probability .55", 1e-13)
    close(changed_prior["evidence"], 1.0, "trust root - and total evidence still exactly one", 1e-13)
    _mark_inference(changed_prior, block["changed_prior_modes"], np.array([0.2, 0.55, 0.25]),
                    constrained_transition, certain, [0, 0], "changed-prior")
    cover.mark("mechanisms.changed_prior_modes")

    # Relabelling the two weather states consistently is an exact null.
    order = [1, 0]
    permuted = infer(start[order], transition[order][:, order], emission[order], observations)
    close(permuted["log_evidence"], main["log_evidence"],
          "trust root - a consistent state permutation leaves the evidence alone", 1e-14)
    close(permuted["path_joint"], main["path_joint"],
          "trust root - and leaves the best path's mass alone", 1e-14)
    expect(permuted["path"].tolist() == [order.index(state) for state in main["path"].tolist()],
           "trust root - while relabelling the path indices")
    _mark_inference(permuted, block["permuted"], start[order], transition[order][:, order],
                    emission[order], observations, "permuted")
    cover.mark("mechanisms.permuted")

    # Independent recordings against one concatenation.
    split = [[0, 1], [0, 2]]
    split_counts = expected_counts(start, transition, emission, split)
    for index, name in enumerate(["initial", "edge", "symbol"]):
        close_all(split_counts[index], block["split_counts"][index],
                  f"trust root - split {name} counts", 1e-11)
    close(split_counts[3], block["split_counts"][3], "trust root - split training log likelihood", 1e-12)
    close(float(split_counts[0].sum()), 2.0, "trust root - two recordings contribute two starts", 1e-11)
    close(float(split_counts[1].sum()), 2.0, "trust root - and two within-recording transitions", 1e-11)
    close(float(split_counts[2].sum()), 4.0, "trust root - and four emissions", 1e-11)
    cover.mark("mechanisms.split_counts")

    joined_counts = expected_counts(start, transition, emission, [observations])
    for index, name in enumerate(["initial", "edge", "symbol"]):
        close_all(joined_counts[index], block["joined_counts"][index],
                  f"trust root - joined {name} counts", 1e-11)
    close(joined_counts[3], block["joined_counts"][3], "trust root - joined training log likelihood", 1e-12)
    close(float(joined_counts[0].sum()), 1.0, "trust root - removing the boundary leaves one start", 1e-11)
    close(float(joined_counts[1].sum()), 3.0, "trust root - and three transitions", 1e-11)
    expect(float(np.abs(joined_counts[1] - split_counts[1]).max()) > 1e-3,
           "trust root - the edge counts genuinely change, so a boundary is not a stored constant")
    cover.mark("mechanisms.joined_counts")

    updated = em_step(start, transition, emission, split)
    close_all(updated[0], block["split_update"]["start"], "trust root - updated initial row", 1e-11)
    close_all(updated[1], block["split_update"]["transition"], "trust root - updated transition rows", 1e-11)
    close_all(updated[2], block["split_update"]["emission"], "trust root - updated emission rows", 1e-11)
    after = expected_counts(*updated, split)[-1]
    close(after, block["split_update"]["log_likelihood"], "trust root - log likelihood after one update", 1e-11)
    expect(after > split_counts[3], "trust root - one exact update did not decrease the objective")
    for index, row in enumerate(updated[1]):
        close(float(row.sum()), 1.0, f"trust root - updated transition row {index} is a distribution", 1e-13)
    for index, row in enumerate(updated[2]):
        close(float(row.sum()), 1.0, f"trust root - updated emission row {index} is a distribution", 1e-13)
    cover.mark("mechanisms.split_update")

    # Declared nulls the author recorded as booleans.
    doubled = em_step(start, transition, emission, split + split)
    for index, name in enumerate(["initial", "transition", "emission"]):
        close_all(doubled[index], updated[index],
                  f"trust root - duplicating the dataset leaves the updated {name} row alone", 1e-11)
    doubled_counts = expected_counts(start, transition, emission, split + split)
    for index, name in enumerate(["initial", "edge", "symbol"]):
        close_all(doubled_counts[index], 2 * np.asarray(split_counts[index]),
                  f"trust root - and doubles the expected {name} counts", 1e-11)
    singleton = em_step(start, transition, emission, [[0]])
    expect(np.array_equal(singleton[1], transition),
           "trust root - a length-one recording has no transition, so its rows are retained")
    # The stored booleans are READ, not merely resolved. `Coverage.mark` only
    # checks that a path exists, so marking these three without asserting them
    # counted them as re-derived while nothing compared them to anything:
    # flipping one to false in the packet left the run green at 99.99 %.
    for name in ("duplicated_dataset_parameter_null", "duplicated_dataset_counts_double",
                 "length_one_transition_rows_retained"):
        expect(packet["additional_author_checks"][name] is True,
               f"trust root - the packet records {name} as true, which is what was re-derived above")
    cover.mark("additional_author_checks.duplicated_dataset_parameter_null",
               "additional_author_checks.duplicated_dataset_counts_double",
               "additional_author_checks.length_one_transition_rows_retained")

    # Numerical scale.
    rare_emission = np.tile([0.01, 0.99], (2, 1))
    rare_observations = [0] * 400
    rare = infer(start, transition, rare_emission, rare_observations)
    close(rare["log_evidence"], block["underflow"]["log_evidence"],
          "trust root - four hundred rare reports have a finite log evidence", 1e-12)
    close(rare["log_evidence"], 400 * np.log(0.01), "trust root - which is exactly 400 log .01", 1e-12)
    close(float(np.log(rare["factors"]).sum()), block["underflow"]["scaled_log_evidence"],
          "trust root - and scaling agrees with log space on it", 1e-12)
    close(0.01 ** 400, block["underflow"]["ordinary_probability"], "trust root - the ordinary float underflows to zero", 0.0)
    expect(0.01 ** 400 == 0.0, "trust root - .01 to the 400th really is zero in float64")
    close(0.3 ** 100, block["underflow"]["representable_counterexample"],
          "trust root - while .3 to the 100th stays representable", 1e-15)
    expect(0.3 ** 100 > 0.0, "trust root - so the underflow claim needs the right exponent")
    cover.mark("mechanisms.underflow")

    impossible = infer(start, transition, np.tile([1.0, 0.0], (2, 1)), [1])
    expect(impossible["impossible"] and impossible["evidence"] == 0.0,
           "trust root - a symbol no state can emit gives exactly zero evidence and no posterior")
    expect(block["impossible"] == "Impossible observation sequence under this model.",
           "trust root - the recorded refusal message")
    cover.mark("mechanisms.impossible")

    for index, entry in enumerate(block["duration"]):
        stay = entry["stay"]
        bars = [stay ** (d - 1) * (1 - stay) for d in range(1, 9)]
        close_all(bars, entry["probabilities_d1_to_8"], f"trust root - duration bars at a={stay}", 1e-13)
        close(1 / (1 - stay), entry["mean"], f"trust root - mean dwell time at a={stay}", 1e-13)
        close(sum(bars) + stay ** 8, 1.0, f"trust root - bars plus tail conserve mass at a={stay}", 1e-13)
        # The mean is also the expectation of the geometric series, by a second route.
        close(sum(d * stay ** (d - 1) * (1 - stay) for d in range(1, 4000)), entry["mean"],
              f"trust root - and the series expectation agrees at a={stay}", 1e-6)
    cover.mark("mechanisms.duration")

    # Manuscript values that have no packet key of their own but must still hold.
    filtered_after_walk = infer(start, transition, emission, [0])["filtered"][0]
    close_all(filtered_after_walk, [0.2, 0.8], "manuscript - filtering after one Walk", 1e-14)
    close_all(filtered_after_walk @ transition, [0.46, 0.54], "manuscript - the next-state prediction", 1e-14)
    close(float(filtered_after_walk @ transition @ emission[:, 2]), 0.284,
          "manuscript - the predicted probability of Clean next", 1e-14)
    close(float(filtered_after_walk @ transition @ emission[:, 1]), 0.346,
          "manuscript - and of Shop next", 1e-14)
    close(float(transition[1] @ emission[:, 1]), 0.34,
          "manuscript - collapsing the belief to Sunny first gives .34 instead", 1e-14)
    close(0.8 ** 2 * 0.2, 0.128, "manuscript - practice 7's duration-three probability", 1e-15)
    close(1 / (1 - 0.8), 5.0, "manuscript - and its mean duration of five", 1e-15)
    expect((2 - 1) + 2 * (2 - 1) + 2 * (3 - 1) == 7,
           "manuscript - the two-state three-symbol model has seven free parameters")
    return {"main": main, "split_counts": split_counts, "updated": updated}


def _mark_inference(result, packet_block, start, transition, emission, observations, label):
    """Match one complete inference record against its packet block."""
    close_all(np.log(raw_trellis(start, transition, emission, observations, "sum")[0]),
              packet_block["forward_log"], f"trust root - {label} forward", 1e-12)
    tail = np.array([float(np.prod(result["factors"][time + 1:])) for time in range(len(observations))])
    close_all(np.log(result["backward_scaled"] * tail[:, None]), packet_block["backward_log"],
              f"trust root - {label} backward", 1e-11)
    close_all(result["filtered"], packet_block["filtered"], f"trust root - {label} filtered", 1e-12)
    close_all(result["smoothed"], packet_block["smoothed"], f"trust root - {label} smoothed", 1e-11)
    if len(observations) > 1:
        close_all(result["pair"], packet_block["pair"], f"trust root - {label} pairs", 1e-11)
    close_all(np.log(result["viterbi_cells"]), packet_block["viterbi_log"], f"trust root - {label} Viterbi", 1e-12)
    expect(result["predecessor"].tolist() == packet_block["predecessor"], f"trust root - {label} predecessors")
    expect(result["path"].tolist() == packet_block["path"], f"trust root - {label} path")
    expect(result["marginal_modes"].tolist() == packet_block["marginal_modes"], f"trust root - {label} modes")
    close(result["log_evidence"], packet_block["log_evidence"], f"trust root - {label} log evidence", 1e-12)
    close(result["path_joint"], packet_block["path_joint"], f"trust root - {label} path joint", 1e-12)
    close(result["path_posterior"], packet_block["path_posterior"], f"trust root - {label} path posterior", 1e-12)


# --------------------------------------------------------------- EM trust root

def rederive_em(packet, cover: Coverage) -> dict:
    block = packet["em"]
    start = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    sequences = sample_sequences(start, transition, emission, SEQUENCE_COUNT, SEQUENCE_LENGTH, DATA_SEED)
    expect(sequences == block["sequences"], "trust root - the seed-71 recordings reproduce exactly")
    expect(len(sequences) == SEQUENCE_COUNT and all(len(row) == SEQUENCE_LENGTH for row in sequences),
           "trust root - twelve recordings of thirty reports each")
    cover.mark("em.sequences")

    generating = expected_counts(start, transition, emission, sequences)[-1]
    close(generating, block["true_model_log_likelihood"],
          "trust root - the generating model's score on this sample", 1e-10)
    cover.mark("em.true_model_log_likelihood")

    fits = []
    for index, seed in enumerate(FIT_SEEDS):
        fit = fit_em(sequences, 2, 3, seed, FIT_STEPS)
        recorded = block["fits"][index]
        expect(recorded["seed"] == seed, f"trust root - fit {index} is seed {seed}")
        expect(len(recorded["log_likelihood"]) == FIT_STEPS + 1,
               f"trust root - seed {seed} records the initial model and forty updates")
        close_all(fit["history"], recorded["log_likelihood"], f"trust root - seed {seed} objective history", 1e-8)
        close_all(fit["model"][0], recorded["start"], f"trust root - seed {seed} fitted initial row", 1e-7)
        close_all(fit["model"][1], recorded["transition"], f"trust root - seed {seed} fitted transitions", 1e-7)
        close_all(fit["model"][2], recorded["emission"], f"trust root - seed {seed} fitted emissions", 1e-7)
        differences = np.diff(fit["history"])
        expect(float(differences.min()) >= -1e-8,
               f"trust root - seed {seed} never decreased the exact objective")
        cover.mark(f"em.fits[{index}]")
        fits.append(fit)

    expect(fits[0]["history"][-1] > block["true_model_log_likelihood"],
           "trust root - a fitted model can score above the generating parameters on this finite sample")

    uniform = fit_em(sequences, 2, 3, 0, UNIFORM_STEPS, uniform=True)
    close_all(uniform["history"], block["uniform_start"]["log_likelihood"],
              "trust root - the uniform start's objective history", 1e-9)
    close_all(uniform["model"][2], block["uniform_start"]["emission"],
              "trust root - its two emission rows stay identical", 1e-10)
    close_all(uniform["model"][2][0], uniform["model"][2][1],
              "trust root - to each other, exactly", 1e-15)
    frequencies = np.bincount(np.array(sequences).ravel(), minlength=3) / (SEQUENCE_COUNT * SEQUENCE_LENGTH)
    close_all(uniform["model"][2][0], frequencies,
              "trust root - and equal the empirical symbol frequencies", 1e-10)
    close_all(frequencies, [1 / 3, 0.3527777777777778, 0.3138888888888889],
              "trust root - which are the stated .333333, .352778 and .313889", 1e-12)
    close_all(uniform["model"][0], block["uniform_start"]["start"], "trust root - its initial row", 1e-10)
    close_all(uniform["model"][1], block["uniform_start"]["transition"], "trust root - its transition rows", 1e-10)
    cover.mark("em.uniform_start")
    return {"sequences": sequences, "fits": fits, "uniform": uniform,
            "generating": generating, "frequencies": frequencies}


# ----------------------------------------------------- the supervised tagger

def coarse_of(upos: str) -> int:
    if upos in {"NOUN", "PROPN"}:
        return 0
    if upos in {"VERB", "AUX"}:
        return 1
    return 2


def build_tagger(records, smoothing: float):
    """The declared supervised fit, computed TWICE by two different arithmetic paths.

    `formula` applies (C + alpha) / (sum C + K alpha) to exact integer counts.
    `accumulated` reproduces the author program, which adds ones onto arrays
    pre-seeded with the smoothing constant. The two agree to about one unit in
    the last place, and the difference is asserted rather than assumed.

    The published model is the accumulated one, for a reason worth stating: at
    smoothing 0.1 the two routes differ by 1.1e-16 in a single emission entry,
    and that one bit is enough to swap which of two EXACTLY equally probable
    paths a decoder reports on development sentences 1 and 2 - moving the
    recorded token count between 267 and 268. Since both paths are true
    maximisers, neither route is more correct; using the declared program's
    arithmetic is what makes the published table reproducible, and
    `exact_tie_audit` supplies the guarantee that does not depend on arithmetic
    at all.
    """
    train = [row for row in records if row["split"] == "train"]
    development = [row for row in records if row["split"] == "dev"]
    frequency = Counter(word.lower() for row in train for word in row["tokens"])
    vocabulary = ["<UNKNOWN>"] + sorted(word for word, count in frequency.items()
                                        if count >= MINIMUM_WORD_COUNT)
    index = {word: number for number, word in enumerate(vocabulary)}

    def encode(words):
        return [index.get(word.lower(), 0) for word in words]

    initial_counts = np.zeros(3, dtype=np.int64)
    transition_counts = np.zeros((3, 3), dtype=np.int64)
    emission_counts = np.zeros((3, len(vocabulary)), dtype=np.int64)
    occupancy_counts = np.zeros(3, dtype=np.int64)
    for row in train:
        tags = [coarse_of(value) for value in row["upos"]]
        initial_counts[tags[0]] += 1
        for previous, following in zip(tags, tags[1:]):
            transition_counts[previous, following] += 1
        for tag, symbol in zip(tags, encode(row["tokens"])):
            emission_counts[tag, symbol] += 1
            occupancy_counts[tag] += 1

    formula = {
        "start": (initial_counts + smoothing) / (initial_counts.sum() + 3 * smoothing),
        "transition": ((transition_counts + smoothing)
                       / (transition_counts.sum(axis=1, keepdims=True) + 3 * smoothing)),
        "emission": ((emission_counts + smoothing)
                     / (emission_counts.sum(axis=1, keepdims=True) + len(vocabulary) * smoothing)),
        "occupancy": (occupancy_counts + smoothing) / (occupancy_counts.sum() + 3 * smoothing),
    }

    # The declared program's own accumulation order, for the published model.
    initial = np.full(3, smoothing)
    transition = np.full((3, 3), smoothing)
    emission = np.full((3, len(vocabulary)), smoothing)
    occupancy = np.full(3, smoothing)
    for row in train:
        tags = [coarse_of(value) for value in row["upos"]]
        initial[tags[0]] += 1
        for previous, following in zip(tags, tags[1:]):
            transition[previous, following] += 1
        for tag, symbol in zip(tags, encode(row["tokens"])):
            emission[tag, symbol] += 1
            occupancy[tag] += 1
    accumulated = {
        "start": initial / initial.sum(),
        "transition": transition / transition.sum(axis=1, keepdims=True),
        "emission": emission / emission.sum(axis=1, keepdims=True),
        "occupancy": occupancy / occupancy.sum(),
    }
    for name in ("start", "transition", "emission", "occupancy"):
        gap = float(np.max(np.abs(formula[name] - accumulated[name])))
        expect(gap <= 4e-16,
               f"two arithmetic routes agree on the {name} table at smoothing {smoothing} "
               f"(largest absolute gap {gap:.3e})")
    return {
        "train": train, "development": development, "vocabulary": vocabulary, "encode": encode,
        "formula": formula, "accumulated": accumulated,
        "start": accumulated["start"], "transition": accumulated["transition"],
        "emission": accumulated["emission"], "occupancy": accumulated["occupancy"],
        "integer_counts": {"initial": initial_counts, "transition": transition_counts,
                           "emission": emission_counts, "occupancy": occupancy_counts},
    }


def decode(fit, method: str):
    rows = []
    for row in fit["development"]:
        symbols = fit["encode"](row["tokens"])
        truth = [coarse_of(value) for value in row["upos"]]
        if method == "hmm":
            # The smoothed marginals come from this file's own scaled route; the
            # decoded path reproduces the declared program's log-space argmax,
            # because three sentences have exactly tied optima and the tie is
            # settled by arithmetic. `exact_tie_audit` supplies the independent
            # guarantee that whatever is reported is a true maximiser.
            result = infer(fit["start"], fit["transition"], fit["emission"], symbols)
            prediction = viterbi_log_route(fit["start"], fit["transition"], fit["emission"], symbols)
            beliefs = result["smoothed"]
        else:
            mass = fit["emission"][:, symbols].T * fit["occupancy"]
            beliefs = mass / mass.sum(axis=1, keepdims=True)
            prediction = [int(np.argmax(belief)) for belief in beliefs]
        rows.append({
            "id": row["id"], "tokens": row["tokens"], "upos": row["upos"], "symbols": symbols,
            "truth": truth, "predicted": prediction, "beliefs": np.asarray(beliefs),
            "correct": int(sum(1 for a, b in zip(truth, prediction) if a == b)),
        })
    return rows


def exact_tie_audit(records, smoothing: float, stored_paths) -> dict:
    """Prove, in exact rational arithmetic, what the float decoders can only suggest.

    Three claims are established here and none of them uses a tolerance:

      1. every stored path attains the exact optimal joint probability;
      2. exactly how many paths attain it, sentence by sentence;
      3. what the two extreme admissible tie rules would have scored, so the
         published token count can be reported as one member of a band rather
         than as a robust measurement.

    The structural cause is worth naming: each tied sentence contains adjacent
    tokens that both map to the unknown symbol, so swapping their two states
    permutes the same multiset of transition and emission factors and leaves the
    product identical. It is the lesson's own point about unknown-word collapse,
    showing up in the arithmetic.
    """
    from fractions import Fraction

    alpha = Fraction(smoothing).limit_denominator(10)
    train = [row for row in records if row["split"] == "train"]
    development = [row for row in records if row["split"] == "dev"]
    frequency = Counter(word.lower() for row in train for word in row["tokens"])
    vocabulary = ["<UNKNOWN>"] + sorted(word for word, count in frequency.items()
                                        if count >= MINIMUM_WORD_COUNT)
    index = {word: number for number, word in enumerate(vocabulary)}
    width = len(vocabulary)
    initial = [0] * 3
    transition_counts = [[0] * 3 for _ in range(3)]
    emission_counts = [[0] * width for _ in range(3)]
    for row in train:
        tags = [coarse_of(value) for value in row["upos"]]
        initial[tags[0]] += 1
        for previous, following in zip(tags, tags[1:]):
            transition_counts[previous][following] += 1
        for tag, word in zip(tags, row["tokens"]):
            emission_counts[tag][index.get(word.lower(), 0)] += 1
    start = [(initial[i] + alpha) / (sum(initial) + 3 * alpha) for i in range(3)]
    transition = [[(transition_counts[i][j] + alpha) / (sum(transition_counts[i]) + 3 * alpha)
                   for j in range(3)] for i in range(3)]
    emission = [[(emission_counts[i][v] + alpha) / (sum(emission_counts[i]) + width * alpha)
                 for v in range(width)] for i in range(3)]

    tied, totals = [], {"first": 0, "last": 0, "stored": 0}
    for position, row in enumerate(development):
        symbols = [index.get(word.lower(), 0) for word in row["tokens"]]
        truth = [coarse_of(value) for value in row["upos"]]
        optimum, optimal_paths = exact_optimum(start, transition, emission, symbols)
        stored = stored_paths[position]
        expect(exact_joint(start, transition, emission, stored, symbols) == optimum,
               f"exact audit - the stored path of development sentence {position} at smoothing "
               f"{smoothing} attains the exact optimum")
        first = exact_viterbi_path(start, transition, emission, symbols, "first")
        last = exact_viterbi_path(start, transition, emission, symbols, "last")
        for name, path in (("first", first), ("last", last)):
            expect(exact_joint(start, transition, emission, path, symbols) == optimum,
                   f"exact audit - the {name}-index path of sentence {position} is also a maximiser")
        totals["first"] += sum(1 for a, b in zip(truth, first) if a == b)
        totals["last"] += sum(1 for a, b in zip(truth, last) if a == b)
        totals["stored"] += sum(1 for a, b in zip(truth, stored) if a == b)
        if optimal_paths > 1:
            unknown_pairs = [[i, i + 1] for i in range(len(symbols) - 1)
                             if symbols[i] == 0 and symbols[i + 1] == 0]
            expect(bool(unknown_pairs),
                   f"exact audit - the tie in sentence {position} sits on adjacent unknown words")
            tied.append({"sentence": position, "id": row["id"], "optimalPaths": optimal_paths,
                         "adjacentUnknownPairs": unknown_pairs})
    return {"tiedSentences": tied, "tokenTotals": totals, "optimalPathCounts": [entry["optimalPaths"] for entry in tied]}


def rederive_real_tagging(packet, records, cover: Coverage) -> dict:
    block = packet["real_tagging"]
    expect(block["labels"] == ["NOUN", "VERB", "OTHER"], "trust root - the three coarse label names")
    cover.mark("real_tagging.labels")

    configurations = []
    fits = {}
    for smoothing in SMOOTHINGS:
        fit = build_tagger(records, smoothing)
        fits[smoothing] = fit
        for method in ["lexical", "hmm"]:
            configurations.append({"method": method, "smoothing": smoothing, "fit": fit,
                                   "rows": decode(fit, method)})
    expect(len(configurations) == 4, "trust root - two fits give four decoder configurations")

    reference = fits[SMOOTHINGS[0]]
    expect(reference["vocabulary"] == block["vocabulary"],
           "trust root - the 146-symbol vocabulary reproduces exactly")
    expect(len(reference["vocabulary"]) == 146, "trust root - and holds 146 symbols")
    expect(reference["vocabulary"][0] == "<UNKNOWN>", "trust root - whose first entry is the unknown category")
    expect(reference["vocabulary"][1:] == sorted(reference["vocabulary"][1:]),
           "trust root - with the remaining words in sorted order")
    development_words = {word.lower() for row in reference["development"] for word in row["tokens"]}
    training_words = set(reference["vocabulary"][1:])
    expect(training_words - development_words != set() and training_words <= {
        word.lower() for row in reference["train"] for word in row["tokens"]},
        "trust root - every emission symbol comes from training words, never from development ones")
    cover.mark("real_tagging.vocabulary")

    expect([row["id"] for row in reference["train"]] == block["train_ids"],
           "trust root - the 120 training sentence ids")
    expect([row["id"] for row in reference["development"]] == block["development_ids"],
           "trust root - the 40 development sentence ids")
    expect(len(set(block["train_ids"]) | set(block["development_ids"])) == 160,
           "trust root - and all 160 of them are distinct")
    cover.mark("real_tagging.train_ids", "real_tagging.development_ids")

    train_tokens = sum(len(row["tokens"]) for row in reference["train"])
    expect(train_tokens == block["train_tokens"] == 1188, "trust root - 1,188 training tokens")
    cover.mark("real_tagging.train_tokens")

    unknown = sum(1 for row in reference["development"] for symbol in reference["encode"](row["tokens"])
                  if symbol == 0)
    expect(unknown == block["unknown_development_tokens"] == 140,
           "trust root - 140 development tokens map to the unknown symbol")
    cover.mark("real_tagging.unknown_development_tokens")

    majority_counts = Counter(coarse_of(tag) for row in reference["train"] for tag in row["upos"])
    majority = majority_counts.most_common(1)[0][0]
    expect(majority == block["majority_label"] == 2, "trust root - the majority training label is Other")
    majority_correct = sum(1 for row in reference["development"] for tag in row["upos"]
                           if coarse_of(tag) == majority)
    expect(majority_correct == block["majority_development_correct"] == 216,
           "trust root - a constant Other prediction gets 216 of 341 development tokens")
    cover.mark("real_tagging.majority_label", "real_tagging.majority_development_correct")

    stated = [(266, 8), (268, 9), (266, 8), (270, 9)]
    for index, configuration in enumerate(configurations):
        recorded = block["configurations"][index]
        expect(recorded["method"] == configuration["method"], f"trust root - configuration {index} method")
        close(recorded["smoothing"], configuration["smoothing"], f"trust root - configuration {index} smoothing", 0.0)
        fit = configuration["fit"]
        close_all(fit["start"], recorded["model"]["start"], f"trust root - configuration {index} initial row", 1e-12)
        close_all(fit["transition"], recorded["model"]["transition"],
                  f"trust root - configuration {index} transition rows", 1e-12)
        close_all(fit["emission"], recorded["model"]["emission"],
                  f"trust root - configuration {index} emission rows", 1e-12)
        close_all(fit["occupancy"], recorded["model"]["occupancy"],
                  f"trust root - configuration {index} lexical prior", 1e-12)
        for name, row in [("initial", fit["start"])] + [("transition", r) for r in fit["transition"]] \
                + [("emission", r) for r in fit["emission"]] + [("occupancy", fit["occupancy"])]:
            close(float(np.sum(row)), 1.0, f"trust root - configuration {index} {name} row is a distribution", 1e-12)
        correct = sum(row["correct"] for row in configuration["rows"])
        tokens = sum(len(row["tokens"]) for row in configuration["rows"])
        sentences = sum(1 for row in configuration["rows"] if row["correct"] == len(row["tokens"]))
        expect(correct == recorded["correct"] == stated[index][0],
               f"trust root - configuration {index} gets {stated[index][0]} tokens")
        expect(tokens == recorded["tokens"] == 341, f"trust root - out of 341")
        expect(sentences == recorded["sentences_correct"] == stated[index][1],
               f"trust root - and {stated[index][1]} whole sentences")
        for position, row in enumerate(configuration["rows"]):
            stored = recorded["rows"][position]
            expect(row["id"] == stored["id"], f"trust root - configuration {index} row {position} id")
            expect(row["tokens"] == stored["tokens"], f"trust root - configuration {index} row {position} tokens")
            expect(row["symbols"] == stored["symbols"], f"trust root - configuration {index} row {position} symbols")
            expect(row["truth"] == stored["truth"], f"trust root - configuration {index} row {position} truth")
            expect(row["predicted"] == stored["predicted"],
                   f"trust root - configuration {index} row {position} prediction")
            close_all(row["beliefs"], stored["beliefs"],
                      f"trust root - configuration {index} row {position} beliefs", 1e-10)
            close_all(row["beliefs"].sum(axis=1), np.ones(len(row["tokens"])),
                      f"trust root - configuration {index} row {position} beliefs are distributions", 1e-12)
        cover.mark(f"real_tagging.configurations[{index}]")

    # The declared selection rule, applied to the four recorded scores.
    scores = [(block["configurations"][index]["correct"],
               block["configurations"][index]["method"] == "lexical",
               -block["configurations"][index]["smoothing"]) for index in range(4)]
    chosen = max(range(4), key=lambda i: scores[i])
    expect(chosen == block["selected_configuration_index"] == 3,
           "trust root - the declared tie rule selects the HMM at smoothing 1.0")
    # The tie rule must actually be exercisable: the two lexical rows really tie.
    expect(block["configurations"][0]["correct"] == block["configurations"][2]["correct"],
           "trust root - the two lexical configurations tie, so the tie rule is not decorative")
    expect(max(entry[0] for entry in scores) == 270,
           "trust root - and the winning token count is the unique maximum")
    cover.mark("real_tagging.selected_configuration_index")
    expect(block["reserved_test_scored"] is False, "trust root - the reserved split is not scored")
    cover.mark("real_tagging.reserved_test_scored")

    # Repairs and breaks between the selected HMM and its matching lexical fit.
    lexical_rows = configurations[2]["rows"]
    hmm_rows = configurations[3]["rows"]
    repairs, breaks = [], []
    for position, (left, right) in enumerate(zip(lexical_rows, hmm_rows)):
        for token_index, (truth, lexical, hmm) in enumerate(zip(left["truth"], left["predicted"], right["predicted"])):
            entry = {"sentence": position, "position": token_index, "token": left["tokens"][token_index],
                     "upos": left["upos"][token_index], "truth": truth, "lexical": lexical, "hmm": hmm}
            if lexical != truth and hmm == truth:
                repairs.append(entry)
            elif lexical == truth and hmm != truth:
                breaks.append(entry)
    expect(len(repairs) == 10, "trust root - the selected HMM repairs ten lexical decisions")
    expect(len(breaks) == 6, "trust root - and breaks six")
    expect(len(repairs) - len(breaks) == 270 - 266 == 4, "trust root - for a net gain of four tokens")

    # The two specimens the manuscript names.
    nina = hmm_rows[27]
    expect(nina["tokens"] == ["Dear", "Nina", ","], "trust root - development index 27 is 'Dear Nina ,'")
    expect(lexical_rows[27]["predicted"] == [0, 0, 2], "trust root - whose lexical prediction is Noun Noun Other")
    expect(nina["predicted"] == [2, 0, 2] == nina["truth"],
           "trust root - while the HMM's Other Noun Other matches the coarse reference")
    article = hmm_rows[3]
    expect(article["tokens"][3] == "article", "trust root - development index 3 holds the word 'article'")
    expect(lexical_rows[3]["predicted"][3] == article["truth"][3] == 0,
           "trust root - which the lexical rule tags correctly as Noun")
    expect(article["predicted"][3] == 2, "trust root - and the HMM breaks into Other")

    # Exact-arithmetic audit of the two HMM configurations.
    audits = {}
    for smoothing, index in ((0.1, 1), (1.0, 3)):
        audit = exact_tie_audit(records, smoothing,
                                [row["predicted"] for row in block["configurations"][index]["rows"]])
        expect(audit["tokenTotals"]["stored"] == block["configurations"][index]["correct"],
               f"exact audit - the stored paths at smoothing {smoothing} score the recorded token count")
        expect(len(audit["tiedSentences"]) == 3,
               f"exact audit - three development sentences have more than one optimal path at smoothing {smoothing}")
        expect([entry["sentence"] for entry in audit["tiedSentences"]] == [1, 2, 39],
               "exact audit - and they are development sentences 1, 2 and 39")
        expect(all(entry["optimalPaths"] == 2 for entry in audit["tiedSentences"]),
               "exact audit - each of them has exactly two optimal paths")
        low = min(audit["tokenTotals"].values())
        high = max(audit["tokenTotals"].values())
        expect(low < audit["tokenTotals"]["stored"] <= high,
               f"exact audit - the recorded count at smoothing {smoothing} sits inside the tie band "
               f"{low} to {high}, so the band is not a restatement of the same number")
        audits[smoothing] = audit
    # The published comparison survives the whole band: the HMM beats its matching
    # lexical baseline under every admissible tie rule, and the selected
    # configuration is the same one. That is the claim worth making.
    lexical_high = block["configurations"][2]["correct"]
    for rule in ("first", "last", "stored"):
        expect(audits[1.0]["tokenTotals"][rule] > lexical_high,
               f"exact audit - the HMM beats the lexical baseline under the {rule} tie rule too")
        expect(audits[1.0]["tokenTotals"][rule] > audits[0.1]["tokenTotals"][rule],
               f"exact audit - and smoothing 1.0 beats 0.1 under the {rule} tie rule too")

    return {"configurations": configurations, "repairs": repairs, "breaks": breaks,
            "vocabulary": reference["vocabulary"], "reference": reference,
            "majority": majority, "majority_correct": majority_correct, "unknown": unknown,
            "train_tokens": train_tokens, "audits": audits}


# --------------------------------------------------------------- module output

def trim(value, places: int):
    return round(float(value), places)


def compact_json(node, indent: int = 0) -> str:
    """Indented JSON whose innermost all-scalar lists stay on one line.

    A 146-entry emission row printed one number per line turns a generated module
    into a quarter of a megabyte of whitespace. Structure still reads down the
    page; only the numbers read across it.
    """
    pad = " " * indent
    inner = " " * (indent + 2)
    if isinstance(node, dict):
        if not node:
            return "{}"
        items = [inner + json.dumps(key, ensure_ascii=False) + ": " + compact_json(value, indent + 2)
                 for key, value in node.items()]
        return "{" + NL + ("," + NL).join(items) + NL + pad + "}"
    if isinstance(node, list):
        if not node:
            return "[]"
        if all(value is None or isinstance(value, (int, float, str, bool)) for value in node):
            return json.dumps(node, ensure_ascii=False)
        items = [inner + compact_json(value, indent + 2) for value in node]
        return "[" + NL + ("," + NL).join(items) + NL + pad + "]"
    return json.dumps(node, ensure_ascii=False)


def build_module(records, tagging, em, sequences_sha: str) -> str:
    reference = tagging["reference"]
    development = reference["development"]
    reserved = [row for row in records if row["split"] == "test"]
    vocabulary = tagging["vocabulary"]

    provenance = {
        "name": "Universal Dependencies English Web Treebank",
        "release": "r2.16",
        "published": "May 2025",
        "page": "https://universaldependencies.org/treebanks/en_ewt/index.html",
        "repository": "https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.16",
        "readme": "https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md",
        "citation": ("Silveira, Dozat, de Marneffe, Bowman, Connor, Bauer and Manning, "
                     "A Gold Standard Dependency Corpus for English, LREC 2014"),
        "license": "CC BY-SA 4.0",
        "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
        "file": "/learn-assets/hmm/ewt-sequences.json",
        "attribution": "/learn-assets/hmm/ATTRIBUTION.txt",
        "program": "/learn-assets/hmm/hmm-experiments.py",
        "optionalProgram": "/learn-assets/hmm/hmmlearn-examples.py",
        "bytes": EXPECTED_SEQUENCES_BYTES,
        "sha256": sequences_sha,
        "sentences": {"train": 120, "development": 40, "reserved": 40},
        "tokens": {"train": tagging["train_tokens"], "development": 341,
                   "reserved": sum(len(row["tokens"]) for row in reserved)},
        "distinctSentenceIds": len({row["id"] for row in records}),
        "lengthRange": [3, 15],
        "selection": ("the first sentences of length 3 to 15 in each official split, which is a "
                      "deliberately biased instructional extract rather than a representative sample"),
        "sourceFiles": [
            {"name": "en_ewt-ud-train.conllu",
             "sha256": "3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b"},
            {"name": "en_ewt-ud-dev.conllu",
             "sha256": "531a54ff90d6ab12201c5a50c3e78e6ddac4de69abc4bce5d275d3cd29efe2b6"},
            {"name": "en_ewt-ud-test.conllu",
             "sha256": "e266e515a0a7547657ed3d90d9ba46487d6bd251f27ad4269d4e8a427c8555cd"},
        ],
        "reservedScored": False,
    }

    coarse = [{"name": COARSE_NAMES[index], "originalTags": ORIGINAL_GROUPS[index]} for index in range(3)]

    sentences = []
    for position, row in enumerate(development):
        symbols = reference["encode"](row["tokens"])
        sentences.append({
            "index": position,
            "id": row["id"],
            "tokens": row["tokens"],
            "upos": row["upos"],
            "truth": [coarse_of(value) for value in row["upos"]],
            "symbols": symbols,
            "unknownPositions": [index for index, symbol in enumerate(symbols) if symbol == 0],
        })

    # The two smoothing strengths give TWO fitted models, each shared by two
    # decoders. Storing the 146-column emission table once per fit rather than
    # once per configuration is not only smaller: it is what makes "the same
    # fitted counts, two decision rules" true in the data as well as the prose.
    fitted = []
    for smoothing in SMOOTHINGS:
        fit = next(entry["fit"] for entry in tagging["configurations"]
                   if entry["smoothing"] == smoothing)
        # Probability rows are stored at FULL precision, never rounded.
        # Rounding a 146-column emission row to nine places accumulates about
        # 4e-9 of error, which is enough to fail the browser model's row-sum
        # guard on first paint. Only quantities that are never fed back into a
        # model - beliefs and objective values - are rounded for size.
        fitted.append({
            "smoothing": smoothing,
            "start": [float(value) for value in fit["start"]],
            "transition": [[float(value) for value in row] for row in fit["transition"]],
            "emission": [[float(value) for value in row] for row in fit["emission"]],
            "occupancy": [float(value) for value in fit["occupancy"]],
        })

    configurations = []
    for entry in tagging["configurations"]:
        configurations.append({
            "key": f"{entry['method']}-{entry['smoothing']:g}",
            "method": entry["method"],
            "label": ("Lexical baseline" if entry["method"] == "lexical" else "HMM, Viterbi path"),
            "smoothing": entry["smoothing"],
            "fitIndex": SMOOTHINGS.index(entry["smoothing"]),
            "correct": sum(row["correct"] for row in entry["rows"]),
            "tokens": sum(len(row["tokens"]) for row in entry["rows"]),
            "sentencesCorrect": sum(1 for row in entry["rows"] if row["correct"] == len(row["tokens"])),
            "rows": [{
                "predicted": row["predicted"],
                "correct": row["correct"],
                # One flat row of 3T numbers per sentence, read three at a time.
                # Nesting each token's triple on its own line cost 12 KB of
                # indentation across the four configurations.
                "beliefs": [trim(value, 6) for belief in row["beliefs"] for value in belief],
            } for row in entry["rows"]],
        })

    fits = []
    for index, seed in enumerate(FIT_SEEDS):
        fit = em["fits"][index]
        checkpoint_names = [("initial", 0), ("afterFirstUpdate", 1), ("final", FIT_STEPS)]
        fits.append({
            "seed": seed,
            "logLikelihood": [trim(value, 6) for value in fit["history"]],
            "checkpoints": [{
                "name": name,
                "update": step,
                "logLikelihood": trim(fit["history"][step], 6),
                "start": [float(value) for value in fit["checkpoints"][step][0]],
                "transition": [[float(value) for value in row] for row in fit["checkpoints"][step][1]],
                "emission": [[float(value) for value in row] for row in fit["checkpoints"][step][2]],
            } for name, step in checkpoint_names],
        })

    uniform = em["uniform"]
    uniform_record = {
        "logLikelihood": [trim(value, 6) for value in uniform["history"]],
        "start": [float(value) for value in uniform["model"][0]],
        "transition": [[float(value) for value in row] for row in uniform["model"][1]],
        "emission": [[float(value) for value in row] for row in uniform["model"][2]],
        "empiricalSymbolFrequencies": [float(value) for value in em["frequencies"]],
    }

    body = {
        "provenance": provenance,
        "coarseLabels": coarse,
        "vocabulary": vocabulary,
        "developmentSentences": sentences,
        "fittedModels": fitted,
        "configurations": configurations,
        "selectedConfigurationIndex": 3,
        "matchedLexicalIndex": 2,
        "majority": {"state": tagging["majority"], "name": COARSE_NAMES[tagging["majority"]],
                     "correct": tagging["majority_correct"], "tokens": 341},
        "unknownDevelopmentTokens": tagging["unknown"],
        "decisionChanges": {"repairs": tagging["repairs"], "breaks": tagging["breaks"]},
        "tieAudit": {
            "note": ("Proved in exact rational arithmetic, not in floating point. A tied sentence "
                     "holds adjacent tokens that both map to the unknown symbol, so exchanging their "
                     "two states permutes the same factors and leaves the product identical."),
            "configurations": [{
                "smoothing": smoothing,
                "tiedSentences": tagging["audits"][smoothing]["tiedSentences"],
                "tokenTotals": tagging["audits"][smoothing]["tokenTotals"],
            } for smoothing in SMOOTHINGS],
        },
        "emTrack": {
            "dataSeed": DATA_SEED,
            "recordings": SEQUENCE_COUNT,
            "length": SEQUENCE_LENGTH,
            "sequences": em["sequences"],
            "generatingLogLikelihood": trim(em["generating"], 6),
            "fits": fits,
            "uniformStart": uniform_record,
        },
    }

    header = (
        "/** Recorded English Web Treebank results and EM histories for the hidden" + NL
        + " * Markov models lesson." + NL
        + " *" + NL
        + " * Generated by scripts/verify-hmm-data.py, which recomputes the whole declared" + NL
        + " * supervised protocol from the extract this lesson serves - coarse label map," + NL
        + " * vocabulary, smoothed counts, both decoders, every development prediction and" + NL
        + " * every per-token belief - through an implementation independent of the content" + NL
        + " * packet's author program, matches every value against that packet, and" + NL
        + " * re-derives the packet's own calculated inputs leaf by leaf. Do not edit by hand." + NL
        + " *" + NL
        + " * Source: Universal Dependencies English Web Treebank r2.16, published May 2025," + NL
        + " * annotations licensed CC BY-SA 4.0. This lesson serves its own copy of the" + NL
        + " * unchanged extract at /learn-assets/hmm/ewt-sequences.json" + NL
        + f" * ({EXPECTED_SEQUENCES_BYTES} bytes, SHA-256 {sequences_sha})." + NL
        + " * The full attribution, licence and extraction record is served beside it." + NL
        + " *" + NL
        + " * Protocol: NOUN and PROPN become Noun, VERB and AUX become Verb, every other" + NL
        + " * original tag becomes Other. Training words are lowercased and retained when they" + NL
        + " * occur at least twice, giving 146 emission symbols including one unknown category;" + NL
        + " * development words never change that vocabulary, and 140 of the 341 development" + NL
        + " * tokens map to unknown. Initial, transition and emission probabilities are" + NL
        + " * smoothed counts at strength 0.1 or 1.0, giving two fits and four decoder" + NL
        + " * configurations. Development token accuracy selects among them; ties prefer the" + NL
        + " * lexical decoder and then the smaller strength. The 40 reserved sentences are" + NL
        + " * never predicted or scored." + NL
        + " *" + NL
        + " * `emTrack` holds measured calculations rather than real data: twelve length-30" + NL
        + " * recordings sampled from the constructed two-state model with seed 71, and the" + NL
        + " * exact-EM objective after every update for three Dirichlet starts and one uniform" + NL
        + " * start. The browser reads these; it never runs EM while a reader scrolls." + NL
        + " *" + NL
        + " * Two fitted models serve four configurations: `configurations[i].fitIndex`" + NL
        + " * names the entry of `fittedModels` a decoder used, because the lexical rule" + NL
        + " * and the Viterbi path read the SAME fitted counts. Each sentence's beliefs" + NL
        + " * are one flat row of 3T numbers, read three at a time in Noun, Verb, Other" + NL
        + " * order. Beliefs carry six decimal places, the precision the page prints;" + NL
        + " * probability rows keep twelve and the emission table nine." + NL
        + " *" + NL
        + " * `tieAudit` records something the float decoders cannot establish about" + NL
        + " * themselves: in exact rational arithmetic, three of the forty development" + NL
        + " * sentences have TWO paths of exactly equal joint probability, so the" + NL
        + " * recorded token counts are one member of a band. The stored paths are" + NL
        + " * proved to be exact maximisers, and the band is reported rather than hidden." + NL
        + " */" + NL + NL
    )
    parts = [header]
    for name, value in body.items():
        parts.append(f"export const {name} = " + compact_json(value) + ";" + NL + NL)
    return "".join(parts).rstrip(NL) + NL


# ------------------------------------------------------------------------ main

def main() -> None:
    write = "--write" in sys.argv
    packet = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))
    cover = Coverage(packet)

    # The served copy is the packet's file, byte for byte.
    asset_bytes = ASSET_SEQUENCES.read_bytes()
    packet_bytes = PACKET_SEQUENCES.read_bytes()
    sequences_sha = hashlib.sha256(asset_bytes).hexdigest()
    expect(asset_bytes == packet_bytes, "the served extract is the packet's file byte for byte")
    expect(sequences_sha == EXPECTED_SEQUENCES_SHA, f"its SHA-256 is the recorded one ({sequences_sha})")
    expect(len(asset_bytes) == EXPECTED_SEQUENCES_BYTES, "and its byte count is the recorded one")
    for name in ["ATTRIBUTION.txt", "hmm-experiments.py", "hmmlearn-examples.py"]:
        expect((ASSET_DIR / name).exists(), f"the lesson serves {name} beside its data")

    records = json.loads(ASSET_SEQUENCES.read_text(encoding="utf-8"))
    expect(len(records) == 200, "the extract holds 200 sentences")
    expect(len({row["id"] for row in records}) == 200, "with 200 distinct sentence ids")
    counts = Counter(row["split"] for row in records)
    expect(counts == {"train": 120, "dev": 40, "test": 40}, f"split sizes are 120/40/40, not {dict(counts)}")
    for row in records:
        expect(len(row["tokens"]) == len(row["upos"]),
               f"sentence {row['id']} has one tag per token")
        expect(3 <= len(row["tokens"]) <= 15, f"sentence {row['id']} is within the declared length range")

    mechanisms = rederive_mechanisms(packet, cover)
    em = rederive_em(packet, cover)
    tagging = rederive_real_tagging(packet, records, cover)

    # The one leaf that is a statement about the author's process, not a number.
    not_covered = sorted(cover.total - cover.covered)
    coverage = len(cover.covered) / len(cover.total)

    text = build_module(records, tagging, em, sequences_sha)

    # Every probability row that the browser will feed back into a model must
    # still be a distribution AFTER serialisation. Rounding a 146-column
    # emission row to nine places accumulated 4e-9 and failed the browser's own
    # row-sum guard on first paint; this check is what makes that impossible to
    # reintroduce silently.
    for index, fit in enumerate(json.loads(
            text.split("export const fittedModels = ")[1].split(";" + NL)[0])):
        for name in ("start", "occupancy"):
            close(sum(fit[name]), 1.0, f"serialised fit {index} {name} row is a distribution", 1e-12)
        for position, row in enumerate(fit["transition"]):
            close(sum(row), 1.0, f"serialised fit {index} transition row {position} is a distribution", 1e-12)
        for position, row in enumerate(fit["emission"]):
            close(sum(row), 1.0, f"serialised fit {index} emission row {position} is a distribution", 1e-12)
    for index, fit in enumerate(json.loads(
            text.split("export const emTrack = ")[1].split(";" + NL)[0])["fits"]):
        for checkpoint in fit["checkpoints"]:
            close(sum(checkpoint["start"]), 1.0,
                  f"serialised EM fit {index} checkpoint {checkpoint['name']} initial row", 1e-12)
            for position, row in enumerate(checkpoint["transition"]):
                close(sum(row), 1.0, f"serialised EM fit {index} {checkpoint['name']} transition {position}", 1e-12)
            for position, row in enumerate(checkpoint["emission"]):
                close(sum(row), 1.0, f"serialised EM fit {index} {checkpoint['name']} emission {position}", 1e-12)
    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        raise SystemExit(f"{len(failures)} of {sum(checks.values())} checks failed.")
    if write:
        MODULE.write_text(text, encoding="utf-8", newline=NL)
    elif not MODULE.exists() or MODULE.read_text(encoding="utf-8") != text:
        raise SystemExit("src/learn/data/hmm-data.js is stale or missing; rerun with --write.")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "author numerical verification of the served data and the packet's trust root",
        "module": "src/learn/data/hmm-data.js",
        "moduleHash": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "verifier": "scripts/verify-hmm-data.py",
        "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packetInputs": str(PACKET_INPUTS.relative_to(ROOT)).replace(chr(92), "/"),
        "packetInputsHash": hashlib.sha256(PACKET_INPUTS.read_bytes()).hexdigest(),
        "servedExtract": {"path": "/learn-assets/hmm/ewt-sequences.json",
                          "bytes": EXPECTED_SEQUENCES_BYTES, "sha256": sequences_sha},
        "checks": {"groups": len(checks), "total": sum(checks.values())},
        "trustRoot": {
            "scalarLeaves": len(cover.total),
            "leavesReDerived": len(cover.covered),
            "coverage": round(coverage, 6),
            "notCovered": not_covered,
            "notCoveredReason": (
                "additional_author_checks.native_example_ast_parsed_not_executed records the state of "
                "the author's own process - that hmmlearn was absent during writing - rather than a "
                "calculation. Phase two installed hmmlearn 0.3.3 in an isolated environment and executed "
                "that program for real; scripts/verify-hmm-examples.py records the result, which "
                "supersedes this leaf rather than re-deriving it."
            ) if not_covered else "every scalar leaf of the packet is re-derived here",
        },
        "independence": [
            "Inference here uses retained scaling; the packet's author program uses log space throughout.",
            "Forward evidence and the best path are additionally checked against exhaustive enumeration "
            "of all 16 paths, which uses no recurrence.",
            "Section 8's smoothing is applied to integer counts through the manuscript's formulas, while "
            "the author program accumulates onto pre-seeded float arrays.",
            "hmm-experiments.py is never imported by this file.",
            "The seed-71 sampling and the Dirichlet starts are reproduced call for call, because a "
            "different call order would produce different data rather than a different route.",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "limits": [
            "No network access; the extract this lesson serves is the only data read.",
            "The 40 reserved sentences are read only to count them, never predicted or scored.",
            "These are development results on a small, deliberately short-sentence extract under one "
            "coarse three-way label map; they are not held-out estimates or a claim about taggers.",
            "Rendering, interaction and visual layout are verified separately.",
        ],
        "passed": True,
    }, indent=2) + NL, encoding="utf-8", newline=NL)
    print(f"PASS: {sum(checks.values()):,} data checks across {len(checks)} groups; "
          f"{len(cover.covered):,} of {len(cover.total):,} trust-root scalar leaves re-derived "
          f"({coverage:.2%}); module {'written' if write else 'current'}.")


if __name__ == "__main__":
    main()
