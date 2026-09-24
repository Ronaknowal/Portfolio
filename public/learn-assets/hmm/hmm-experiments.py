"""Complete offline teaching calculations: categorical HMM inference, EM and tagging."""
from collections import Counter
from itertools import product
from pathlib import Path
import json
import numpy as np


def log_values(values):
    with np.errstate(divide="ignore"):
        return np.log(values)


def infer(start, transition, emission, observations):
    """Return joint evidence, filtered/smoothed beliefs and exact best-path decoding.

    Rows of transition/emission sum to one. Observation -1 is missing at a
    retained time step: sum over its categories, giving emission likelihood one.
    """
    observations = np.asarray(observations, dtype=int)
    if len(observations) == 0:
        raise ValueError("This teaching function requires at least one time step.")
    states = len(start)
    log_transition = log_values(transition)
    local = np.array([
        np.zeros(states) if value == -1 else log_values(emission[:, value])
        for value in observations
    ])
    forward = np.empty_like(local)
    forward[0] = log_values(start) + local[0]
    for time in range(1, len(observations)):
        forward[time] = np.logaddexp.reduce(
            forward[time - 1, :, None] + log_transition, axis=0
        ) + local[time]
    log_evidence = np.logaddexp.reduce(forward[-1])
    if not np.isfinite(log_evidence):
        raise ValueError("Impossible observation sequence under this model.")
    backward = np.zeros_like(local)
    for time in range(len(observations) - 2, -1, -1):
        backward[time] = np.logaddexp.reduce(
            log_transition + local[time + 1] + backward[time + 1], axis=1
        )
    smoothed = np.exp(forward + backward - log_evidence)
    filtered = np.exp(
        forward - np.logaddexp.reduce(forward, axis=1, keepdims=True)
    )
    pair = np.empty((len(observations) - 1, states, states))
    for time in range(len(observations) - 1):
        pair[time] = np.exp(
            forward[time, :, None] + log_transition
            + local[time + 1] + backward[time + 1] - log_evidence
        )
    best = np.empty_like(local)
    predecessor = np.zeros(local.shape, dtype=int)
    best[0] = forward[0]
    for time in range(1, len(observations)):
        candidates = best[time - 1, :, None] + log_transition
        predecessor[time] = np.argmax(candidates, axis=0)
        best[time] = candidates[predecessor[time], np.arange(states)] + local[time]
    path = np.zeros(len(observations), dtype=int)
    path[-1] = np.argmax(best[-1])
    for time in range(len(observations) - 2, -1, -1):
        path[time] = predecessor[time + 1, path[time + 1]]
    return {
        "log_evidence": log_evidence, "forward_log": forward,
        "backward_log": backward, "filtered": filtered, "smoothed": smoothed,
        "pair": pair, "viterbi_log": best, "predecessor": predecessor,
        "path": path, "path_joint": np.exp(best[-1, path[-1]]),
        "path_posterior": np.exp(best[-1, path[-1]] - log_evidence),
        "marginal_modes": np.argmax(smoothed, axis=1),
    }


def forward_scaled(start, transition, emission, observations):
    belief = start.copy()
    factors = []
    for time, value in enumerate(observations):
        prediction = belief if time == 0 else belief @ transition
        mass = prediction if value == -1 else prediction * emission[:, value]
        factor = mass.sum()
        if factor == 0:
            raise ValueError("Impossible observation sequence under this model.")
        belief = mass / factor
        factors.append(factor)
    return {"filtered_last": belief, "factors": factors,
            "log_evidence": np.log(factors).sum()}


def expected_counts(model, sequences):
    start, transition, emission = model
    initial = np.zeros_like(start)
    edge = np.zeros_like(transition)
    symbol = np.zeros_like(emission)
    log_likelihood = 0.0
    for observations in sequences:
        result = infer(*model, observations)
        initial += result["smoothed"][0]
        edge += result["pair"].sum(axis=0)
        for time, value in enumerate(observations):
            if value != -1:
                symbol[:, value] += result["smoothed"][time]
        log_likelihood += result["log_evidence"]
    return initial, edge, symbol, log_likelihood


def normalize_counts(counts, previous):
    """An unvisited row is unidentified; retain it instead of inventing counts."""
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total, out=previous.copy(), where=total > 0)


def em_step(model, sequences):
    initial, edge, symbol, _ = expected_counts(model, sequences)
    return (
        initial / initial.sum(),
        normalize_counts(edge, model[1]),
        normalize_counts(symbol, model[2]),
    )


def fit_em(sequences, states, symbols, seed, steps=40, uniform=False):
    rng = np.random.default_rng(seed)
    model = (
        rng.dirichlet(np.ones(states)),
        rng.dirichlet(np.ones(states), size=states),
        rng.dirichlet(np.ones(symbols), size=states),
    )
    if uniform:
        model = (np.ones(states) / states, np.ones((states, states)) / states,
                 np.ones((states, symbols)) / symbols)
    history = [expected_counts(model, sequences)[-1]]
    for _ in range(steps):
        model = em_step(model, sequences)
        history.append(expected_counts(model, sequences)[-1])
    if np.min(np.diff(history)) < -1e-8:
        raise ArithmeticError("The executed exact-EM likelihood decreased.")
    return {"start": model[0], "transition": model[1], "emission": model[2],
            "log_likelihood": history}


def sample_sequences(model, count, length, seed):
    rng = np.random.default_rng(seed)
    start, transition, emission = model
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


def mechanism_examples():
    model = (np.array([.6, .4]), np.array([[.7, .3], [.4, .6]]),
             np.array([[.1, .4, .5], [.6, .3, .1]]))
    observations = [0, 1, 0, 2]
    main = infer(*model, observations)
    paths = []
    for path in product(range(2), repeat=4):
        probability = model[0][path[0]] * model[2][path[0], observations[0]]
        for time in range(1, 4):
            probability *= model[1][path[time - 1], path[time]]
            probability *= model[2][path[time], observations[time]]
        paths.append({"states": path, "joint": probability})
    assert np.isclose(sum(item["joint"] for item in paths),
                      np.exp(main["log_evidence"]))
    assert np.isclose(max(item["joint"] for item in paths), main["path_joint"])
    for time, pair in enumerate(main["pair"]):
        assert np.allclose(pair.sum(axis=1), main["smoothed"][time])
        assert np.allclose(pair.sum(axis=0), main["smoothed"][time + 1])
    future = infer(*model, [0, 1, 0, 0])
    assert np.array_equal(main["filtered"][:3], future["filtered"][:3])
    changed_emission = model[2].copy()
    changed_emission[0] = [.2, .3, .5]
    constrained = (np.array([.4, .35, .25]),
                   np.array([[0., .5, .5], [1., 0., 0.], [1., 0., 0.]]),
                   np.ones((3, 1)))
    split = [[0, 1], [0, 2]]
    counts_split = expected_counts(model, split)
    counts_joined = expected_counts(model, [observations])
    updated = em_step(model, split)
    duplicated = em_step(model, split + split)
    assert all(np.allclose(left, right) for left, right in zip(updated, duplicated))
    doubled_counts = expected_counts(model, split + split)
    assert all(np.allclose(right, 2 * left)
               for left, right in zip(counts_split, doubled_counts))
    single = em_step(model, [[0]])
    assert np.array_equal(single[1], model[1])
    permutation = np.array([1, 0])
    permuted = (model[0][permutation],
                model[1][permutation][:, permutation], model[2][permutation])
    rare = (model[0], model[1], np.tile([.01, .99], (2, 1)))
    rare_observations = [0] * 400
    tiny_log = infer(*rare, rare_observations)["log_evidence"]
    tiny_scaled = forward_scaled(*rare, rare_observations)
    assert np.isclose(tiny_log, tiny_scaled["log_evidence"])
    try:
        infer(model[0], model[1], np.tile([1., 0.], (2, 1)), [1])
        raise AssertionError("Expected impossible evidence.")
    except ValueError as error:
        impossible = str(error)
    return {
        "model": {"start": model[0], "transition": model[1], "emission": model[2]},
        "observations": observations, "main": main, "enumerated_paths": paths,
        "changed_future": future,
        "changed_emission": infer(model[0], model[1], changed_emission, observations),
        "scaled": forward_scaled(*model, observations),
        "missing_middle": infer(*model, [0, -1, 0, 2]),
        "deleted_middle": infer(*model, [0, 0, 2]),
        "illegal_modes": infer(*constrained, [0, 0]),
        "changed_prior_modes": infer(np.array([.2, .55, .25]),
                                     constrained[1], constrained[2], [0, 0]),
        "split_counts": counts_split, "joined_counts": counts_joined,
        "split_update": {"start": updated[0], "transition": updated[1],
                         "emission": updated[2],
                         "log_likelihood": expected_counts(updated, split)[-1]},
        "permuted": infer(*permuted, observations),
        "underflow": {"ordinary_probability": float(.01 ** 400),
                      "log_evidence": tiny_log,
                      "scaled_log_evidence": tiny_scaled["log_evidence"],
                      "representable_counterexample": .3 ** 100},
        "impossible": impossible,
        "duration": [{"stay": stay, "mean": 1 / (1 - stay),
                      "probabilities_d1_to_8": [stay ** (d - 1) * (1 - stay)
                                               for d in range(1, 9)]}
                     for stay in [0., .7, .95]],
    }


def real_tagging(directory):
    records = json.loads((directory / "ewt-sequences.json").read_text(encoding="utf-8"))
    # The retained extract is a flat list; split membership is preserved from EWT.
    train = [row for row in records if row["split"] == "train"]
    development = [row for row in records if row["split"] == "dev"]
    labels = ["NOUN", "VERB", "OTHER"]
    def coarse(upos):
        return 0 if upos in {"NOUN", "PROPN"} else 1 if upos in {"VERB", "AUX"} else 2
    frequency = Counter(word.lower() for row in train for word in row["tokens"])
    vocabulary = ["<UNKNOWN>"] + sorted(word for word, count in frequency.items() if count >= 2)
    index = {word: number for number, word in enumerate(vocabulary)}
    def encode(words):
        return [index.get(word.lower(), 0) for word in words]
    configurations = []
    for smoothing in [.1, 1.]:
        initial = np.full(3, smoothing)
        transition = np.full((3, 3), smoothing)
        emission = np.full((3, len(vocabulary)), smoothing)
        occupancy = np.full(3, smoothing)
        for row in train:
            tags = [coarse(value) for value in row["upos"]]
            initial[tags[0]] += 1
            for previous, following in zip(tags, tags[1:]):
                transition[previous, following] += 1
            for tag, symbol in zip(tags, encode(row["tokens"])):
                emission[tag, symbol] += 1
                occupancy[tag] += 1
        model = (initial / initial.sum(), transition / transition.sum(axis=1, keepdims=True),
                 emission / emission.sum(axis=1, keepdims=True))
        prior = occupancy / occupancy.sum()
        for method in ["lexical", "hmm"]:
            rows = []
            for row in development:
                symbols = encode(row["tokens"])
                truth = np.array([coarse(value) for value in row["upos"]])
                if method == "hmm":
                    result = infer(*model, symbols)
                    prediction = result["path"]
                    beliefs = result["smoothed"]
                else:
                    mass = model[2][:, symbols].T * prior
                    beliefs = mass / mass.sum(axis=1, keepdims=True)
                    prediction = beliefs.argmax(axis=1)
                rows.append({"id": row["id"], "tokens": row["tokens"], "symbols": symbols,
                             "truth": truth, "predicted": prediction, "beliefs": beliefs,
                             "correct": int((truth == prediction).sum())})
            configurations.append({
                "method": method, "smoothing": smoothing, "model": {
                    "start": model[0], "transition": model[1], "emission": model[2],
                    "occupancy": prior},
                "correct": sum(row["correct"] for row in rows),
                "tokens": sum(len(row["tokens"]) for row in rows),
                "sentences_correct": sum(row["correct"] == len(row["tokens"]) for row in rows),
                "rows": rows,
            })
    # Predeclared ties favor lexical, then smaller smoothing.
    chosen = max(range(len(configurations)), key=lambda i: (
        configurations[i]["correct"], configurations[i]["method"] == "lexical",
        -configurations[i]["smoothing"]))
    majority = Counter(coarse(tag) for row in train for tag in row["upos"]).most_common(1)[0][0]
    return {"labels": labels, "vocabulary": vocabulary, "train_ids": [row["id"] for row in train],
            "development_ids": [row["id"] for row in development],
            "train_tokens": sum(len(row["tokens"]) for row in train),
            "unknown_development_tokens": sum(symbol == 0 for row in development for symbol in encode(row["tokens"])),
            "majority_label": majority,
            "majority_development_correct": sum(coarse(tag) == majority for row in development for tag in row["upos"]),
            "configurations": configurations, "selected_configuration_index": chosen,
            "reserved_test_scored": False}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    directory = Path(__file__).resolve().parent
    mechanisms = mechanism_examples()
    data = mechanisms["model"]
    true_model = (np.asarray(data["start"]), np.asarray(data["transition"]),
                  np.asarray(data["emission"]))
    sequences = sample_sequences(true_model, count=12, length=30, seed=71)
    fits = [dict(seed=seed, **fit_em(sequences, 2, 3, seed)) for seed in [3, 7, 19]]
    uniform = fit_em(sequences, 2, 3, seed=0, steps=3, uniform=True)
    result = {"mechanisms": mechanisms, "em": {
        "sequences": sequences, "true_model_log_likelihood": expected_counts(true_model, sequences)[-1],
        "fits": fits, "uniform_start": uniform},
        "real_tagging": real_tagging(directory)}
    (directory / "calculated-inputs.json").write_text(
        json.dumps(result, default=serializable, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print("Recorded exact mechanisms, four EM fits, and four supervised tagging configurations.")


if __name__ == "__main__":
    main()
