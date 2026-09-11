"""Bounded independent Queueing35 review; no production files are written.

Uses conditional customer stages, exact interval arithmetic, residual survival
integrals, and a relative-workload simulation oracle. Run from the repo root.
"""

import argparse
import contextlib
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="scratch/queueing-independent-review/results.json")
args = parser.parse_args()
failures = []
counts = {}


def check(condition, label):
    if not condition:
        failures.append(label)


def close(actual, expected, label, relative=3e-12, absolute=1e-14):
    check(math.isclose(actual, float(expected), rel_tol=relative, abs_tol=absolute),
          f"{label}: actual={actual!r}; expected={float(expected)!r}")


def node_json(source):
    return json.loads(subprocess.check_output(
        ["node", "--input-type=module", "-e", source], cwd=ROOT, text=True,
        encoding="utf-8"))


examples = node_json("import {queueingExamples} from './src/learn/data/queueing-examples.js'; console.log(JSON.stringify(queueingExamples));")
namespaces = {}
for example in examples:
    stdout = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stdout):
        exec(compile(example["code"], example["id"], "exec"), namespace)
    check(stdout.getvalue().rstrip("\n") == example["expected"], f"stdout {example['id']}")
    namespaces[example["id"]] = namespace
counts["actual_complete_programs"] = len(examples)

archive = json.loads((ROOT / "docs/teaching/evidence/queueing-original-content.json").read_text())
check(hashlib.sha256(archive["fullSource"].encode()).hexdigest() == archive["sha256"], "original full source hash")
for block in archive["blocks"]:
    field = "code" if block["language"] == "python" else "expected"
    check(any(row[field] == block["text"] for row in examples), f"preserved original {field}")
counts["byte_conserved_original_blocks"] = len(archive["blocks"])

fixtures = node_json("""
import * as q from './src/learn/data/queueing-models.js';
const finite=[];
for(const [arrival,service] of [[3,7],[7,7],[13,7],[1e-6,1e6],[1e6,1e-6]])
  for(const capacity of [1,2,5,20]) finite.push({arrival,service,capacity,state:q.finiteBufferState(arrival,service,capacity)});
const tails=[];
for(const [arrival,service] of [[2,5],[9,10],[1e-6,2e-6],[999999,1e6]])
  for(const probability of [.001,.1,.6,.99,.9999]) tails.push({arrival,service,probability,state:q.mm1State(arrival,service,probability,7)});
const pooling=[];
for(const servers of [1,2,5,12]) for(const load of [.07,.55,.97]) pooling.push({servers,load,state:q.pooledQueueState(load*servers*3,3,servers)});
const traceJobs=[{id:'x',arrival:0,service:.5},{id:'y',arrival:.25,service:0},{id:'z',arrival:.25,service:1.25},{id:'late',arrival:4,service:.25}];
const windows=[];
for(const boundary of ['system','queue']) for(const horizon of [.125,.25,.5,1,1.75,4,4.25,6]) windows.push({boundary,horizon,state:q.occupancyWindow(traceJobs,horizon,boundary)});
console.log(JSON.stringify({finite,tails,pooling,traceJobs,windows}));
""")

# Given admission and arrival-seen n<K, the customer needs n+1 exponential
# stages. Compute that conditional population explicitly, without Little's law.
for row in fixtures["finite"]:
    arrival, service, capacity = row["arrival"], row["service"], row["capacity"]
    state = row["state"]
    rate_ratio = Fraction(str(arrival)) / Fraction(str(service))
    state_weights = [rate_ratio**n for n in range(capacity + 1)]
    admitted_weights = state_weights[:-1]
    conditional = [value / sum(admitted_weights) for value in admitted_weights]
    mean_total = sum((n + 1) * value for n, value in enumerate(conditional)) / Fraction(str(service))
    mean_wait = sum(n * value for n, value in enumerate(conditional)) / Fraction(str(service))
    close(state["meanTotal"], mean_total, "admitted customer stages total")
    close(state["meanWait"], mean_wait, "admitted customer stages wait")
    admitted = Fraction(str(arrival)) * sum(admitted_weights) / sum(state_weights)
    close(state["admittedRate"], admitted, "admitted throughput")
    native = namespaces["finite-capacity"]["finite_queue"](arrival, service, capacity)
    close(native[3], mean_total, "native conditional stages total")
    close(native[4], mean_wait, "native conditional stages wait")
counts["admitted_customer_stage_cases"] = len(fixtures["finite"])

for row in fixtures["tails"]:
    arrival, service, probability, state = (row[key] for key in ("arrival", "service", "probability", "state"))
    # Invert the survival into the requested probability and respect the zero atom.
    gap = service - arrival
    close(-math.expm1(-gap * state["totalQuantile"]), probability, "total quantile inversion")
    if probability <= state["idle"]:
        check(state["waitQuantile"] == 0, "queue inverse at zero atom")
    else:
        close(1 - (arrival / service) * math.exp(-gap * state["waitQuantile"]), probability, "queue quantile inversion")
    close(sum(state["probabilities"]) + state["tail"], 1, "visible state mass plus overflow")
    native = namespaces["tails"]["stationary_mm1"](arrival, service, probability)
    close(native["queue_quantile"], state["waitQuantile"], "paired queue native")
    close(native["total_quantile"], state["totalQuantile"], "paired total native")
counts["tail_inverse_and_atom_cases"] = len(fixtures["tails"])

# Blocking recursion is a different normalization from the displayed factorial
# weights. Converting Erlang B to C checks the pooled waiting probability.
for row in fixtures["pooling"]:
    servers, load, state = row["servers"], row["load"], row["state"]
    offered = load * servers
    blocking = 1.0
    for slots in range(1, servers + 1):
        blocking = offered * blocking / (slots + offered * blocking)
    wait_probability = blocking / (1 - load + load * blocking)
    close(state["waitProbability"], wait_probability, "Erlang B to C")
    native = namespaces["pooling"]["erlang_c"](offered * 3, 3, servers)
    close(native[0], wait_probability, "native Erlang B to C")
    close(state["meanTotal"] - state["meanWait"], Fraction(1, 3), "pooled service accounting")
counts["pooled_normalization_cases"] = len(fixtures["pooling"])

# Exact rational event rectangles, including zero service and a tied arrival.
arrival_times = [Fraction(0), Fraction(1, 4), Fraction(1, 4), Fraction(4)]
durations = [Fraction(1, 2), Fraction(0), Fraction(5, 4), Fraction(1, 4)]
starts, departures = [], []
available = Fraction(0)
for arrival, duration in zip(arrival_times, durations):
    starts.append(max(arrival, available))
    departures.append(starts[-1] + duration)
    available = departures[-1]
for row in fixtures["windows"]:
    horizon = Fraction(str(row["horizon"]))
    finishes = departures if row["boundary"] == "system" else starts
    events = sorted({Fraction(0), horizon, *[x for x in arrival_times + finishes if 0 <= x <= horizon]})
    rectangle_area = sum((right - left) * sum(a <= (left + right) / 2 < f for a, f in zip(arrival_times, finishes))
                         for left, right in zip(events, events[1:]))
    complete = sum(f - a for a, f in zip(arrival_times, finishes) if f <= horizon)
    close(row["state"]["clippedArea"], rectangle_area, "rational clipped area")
    close(row["state"]["pendingArea"], rectangle_area - complete, "rational pending correction")
counts["rational_censored_windows"] = len(fixtures["windows"])

# Direct integration of the equilibrium residual survival function, including a
# zero-duration atom. The same moments enter the declared repeated-vacation model.
mixtures = [[(.2, 0), (.5, .125), (.3, .75)], [(.75, .04), (.25, .28)], [(1, .2)]]
for atoms in mixtures:
    mean = sum(p * duration for p, duration in atoms)
    endpoints = sorted({0, *[duration for _, duration in atoms]})
    survival = lambda t: sum(p * max(duration - t, 0) for p, duration in atoms) / mean
    residual = sum(quad(survival, left, right, epsabs=1e-13)[0] for left, right in zip(endpoints, endpoints[1:]))
    for load in [.1, .5, .9]:
        arrival = load / mean
        native = namespaces["mixtures"]["mixture_metrics"](atoms, arrival)
        close(native[3], residual, "integrated busy residual")
        close(native[4], load * residual, "idle weighted residual")
        close(native[5] * (1 - load), load * residual, "residual workload fixed point")
        vacation = namespaces["vacations"]["repeated_vacation_wait"](arrival, mean, native[1], mean, native[1])
        close(vacation[1], residual, "vacation residual survival")
        close(vacation[2], residual / (1 - load), "same service and vacation distribution")
counts["residual_survival_and_vacation_cases"] = 9

for minimum in [.001, .125, 2]:
    for multiplier in [1, 1.5, 7, 100]:
        cap = minimum * multiplier
        mean, second = namespaces["heavy-tail"]["truncated_pareto_moments"](cap, minimum)
        survival = lambda t: (minimum / t) ** 1.5
        close(mean, minimum + quad(survival, minimum, cap)[0], "truncated survival first moment")
        close(second, minimum**2 + quad(lambda t: 2*t*survival(t), minimum, cap)[0], "truncated survival second moment")
counts["changed_pareto_survival_cases"] = 12

# Complementary relative-workload oracle: preserve the same seeded draw order,
# but never create or subtract a large absolute timestamp.
simulation_cases = []
for arrival, service, warmup, measured, seed in [(1e-6, 1e6, 100000, 100, 17), (1e-6, 1e6, 10000, 1000, 19), (3, 7, 333, 777, 23), (97, 100, 333, 777, 31)]:
    rng = random.Random(seed)
    workload = 0.0
    waits, totals, services = [], [], []
    for index in range(warmup + measured):
        gap = rng.expovariate(arrival)
        duration = rng.expovariate(service)
        waiting = max(0.0, workload - gap)
        workload = waiting + duration
        if index >= warmup:
            waits.append(waiting)
            totals.append(workload)
            services.append(duration)
    expected = statistics.mean(waits), statistics.mean(totals)
    try:
        actual = namespaces["simulation"]["simulate_fcfs"](arrival, service, seed, warmup=warmup, measured=measured)
        close(actual[0], expected[0], "simulation relative waiting", relative=2e-10, absolute=1e-18)
        close(actual[1], expected[1], "simulation relative total", relative=2e-10, absolute=1e-18)
        disposition = "returned"
    except ValueError as error:
        actual, disposition = None, f"explicit arithmetic rejection: {error}"
        check(arrival == 1e-6 and service == 1e6, "ordinary simulation must remain accepted")
    simulation_cases.append(dict(arrival=arrival, service=service, warmup=warmup, measured=measured, seed=seed,
                                 actual=actual, expected=expected, sampledService=statistics.mean(services), disposition=disposition))
counts["same_draw_relative_workload_cases"] = len(simulation_cases)

freeze = json.loads((ROOT / "docs/teaching/evidence/queueing-author-review.json").read_text())
source_hashes = []
for item in freeze["production"]:
    data = (ROOT / item["path"]).read_bytes()
    source_hashes.append({"path": item["path"], "sha256": hashlib.sha256(data).hexdigest(), "authorFrozenSha256": item["sha256"]})
result = dict(at=datetime.now(timezone.utc).isoformat(), counts=counts, failures=failures,
              simulationCases=simulation_cases, production=source_hashes,
              authorFreeze=freeze["authorFrozenAt"], scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output = ROOT / args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(output), "counts": counts, "failures": failures, "simulation": simulation_cases}, indent=2))
raise SystemExit(bool(failures))
