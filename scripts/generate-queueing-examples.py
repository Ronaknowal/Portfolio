"""Generate complete examples, preserving original programs and executing stdout."""
import ast
import contextlib
import io
import json
from pathlib import Path

import black

ROOT = Path(__file__).resolve().parents[1]
archive = json.loads((ROOT / "docs/teaching/evidence/queueing-original-content.json").read_text(encoding="utf-8"))
original = [item["text"] for item in archive["blocks"] if item["language"] == "python"]
examples = []


def add(identity, title, question, code, note, preserve=False):
    if not preserve:
        before = ast.dump(ast.parse(code), include_attributes=False)
        code = black.format_str(code, mode=black.Mode(line_length=88)).strip()
        assert ast.dump(ast.parse(code), include_attributes=False) == before
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, identity + ".py", "exec"), {})
    examples.append(dict(id=identity, title=title, question=question, language="python", code=code, expected=output.getvalue().strip(), note=note))


add("trace", "Follow jobs and account for an unfinished window",
    "At time 5, which job is still present, and why does completed-only residence omit part of the occupancy area?", r'''
from math import isfinite


def trace_jobs(jobs):
    if len(jobs) > 32:
        raise ValueError("Use at most 32 jobs")
    available = 0
    previous_arrival = 0
    seen = set()
    trace = []
    for name, arrival, service in jobs:
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("Each job needs a distinct nonempty text ID")
        if not isfinite(arrival) or not previous_arrival <= arrival <= 1e6:
            raise ValueError("Use finite nonnegative arrivals in listed order")
        if not isfinite(service) or not 0 <= service <= 1e4:
            raise ValueError("Use finite nonnegative bounded service durations")
        seen.add(name)
        previous_arrival = arrival
        start = max(arrival, available)
        departure = start + service
        if service > 0 and departure == start:
            raise ValueError("Positive service is too small at this time origin; rescale")
        trace.append((name, arrival, start, departure))
        available = departure
    return trace


def window_metrics(trace, horizon):
    if not isfinite(horizon) or not 1e-6 <= horizon <= 1e6:
        raise ValueError("A measurement window needs a finite positive duration")
    area = sum(max(0, min(departure, horizon) - arrival)
               for _, arrival, _, departure in trace)
    completed = [(arrival, departure) for _, arrival, _, departure in trace
                 if departure <= horizon]
    complete_sum = sum(departure - arrival for arrival, departure in completed)
    return area, complete_sum, area / horizon


jobs = [("A", 0, 3), ("B", 1, 1), ("C", 2, 2), ("D", 6, 1)]
trace = trace_jobs(jobs)  # Already in arrival order; tied arrivals keep this order.
for name, arrival, start, departure in trace:
    print(name, "start", start, "depart", departure,
          "wait", start - arrival, "total", departure - arrival)
for horizon in [5, 7]:
    area, complete_sum, mean_number = window_metrics(trace, horizon)
    print("horizon", horizon, "area", area, "completed residence", complete_sum,
          "mean number", round(mean_number, 6))
print("complete-cohort Little product:", round((4 / 7) * (11 / 4), 6))
''', "This is a deterministic trace, not a steady-state estimate. Occupancy uses [arrival, departure); unfinished residence is clipped at the observation horizon.")

add("original-mm1", "Calculate the original 8-per-second example",
    "Why can a job spend 0.5 seconds in a system whose mean service lasts only 0.1 seconds?",
    original[0], "The preserved program uses stationary M/M/1 means. Its inputs satisfy 0 < arrival rate < service rate.", preserve=True)

add("tails", "Keep the waiting atom and total-time tail separate",
    "Can a queue-wait percentile be zero? At load 0.9, how do the queue and total p99 differ?", r'''
from math import exp, isfinite, log1p


def stationary_mm1(arrival, service, probability=0.99):
    if not all(isfinite(x) for x in [arrival, service, probability]):
        raise ValueError("Inputs must be finite")
    if not 0 < arrival < service or not 0 < probability < 1:
        raise ValueError("Need 0 < arrival < service and 0 < probability < 1")
    gap = service - arrival
    rho = arrival / service
    idle = gap / service
    queue_quantile = (0.0 if probability <= idle else
                      log1p((probability - idle) / (1 - probability)) / gap)
    total_quantile = -log1p(-probability) / gap
    result = {"idle": idle, "mean_total": 1 / gap, "mean_wait": rho / gap,
              "queue_quantile": queue_quantile, "total_quantile": total_quantile}
    if not all(isfinite(x) for x in result.values()):
        raise ValueError("Arithmetic range exceeded; rescale the rates")
    return result


result = stationary_mm1(0.9, 1)
for name, value in result.items():
    print(name, round(value, 6))
time = 10
print("P(total > 10):", round(exp(-(1 - 0.9) * time), 6))
print("P(queue > 10):", round(0.9 * exp(-(1 - 0.9) * time), 6))
print("queue median at load 0.2:", stationary_mm1(0.2, 1, 0.5)["queue_quantile"])
try:
    stationary_mm1(1, 1)
except ValueError as error:
    print(error)
''', "These are quantiles of the stationary FCFS model, not worst-case durations. The queue's mass at zero is part of its distribution.")

add("original-mg1", "Retain the original general-service calculation",
    "At the same mean service duration, why does standard deviation 0.05 seconds change mean waiting?",
    original[1], "The preserved P–K example needs Poisson arrivals, independent iid service, FCFS, utilization below one and a finite second moment.", preserve=True)

add("mixtures", "Compare job counts with occupied service time",
    "If long jobs are only 5% of requests, can they occupy more than half the busy time?", r'''
from math import isclose, isfinite


def mixture_metrics(atoms, arrival):
    if not isfinite(arrival) or not 1e-6 <= arrival <= 1e6:
        raise ValueError("Use a finite positive supported rate")
    if not 1 <= len(atoms) <= 32 or any(not isfinite(p) or not 0 <= p <= 1 or
        not isfinite(duration) or not (duration == 0 or 1e-6 <= duration <= 1e4)
        for p, duration in atoms):
        raise ValueError("Use bounded nonnegative probabilities and durations")
    if not isclose(sum(p for p, _ in atoms), 1, abs_tol=1e-12):
        raise ValueError("Probabilities must sum to one")
    mean = sum(p * duration for p, duration in atoms)
    second = sum(p * duration**2 for p, duration in atoms)
    rho = arrival * mean
    if mean <= 0 or rho >= 1:
        raise ValueError("Need positive mean service and load below one")
    busy_shares = [p * duration / mean for p, duration in atoms]
    residual_busy = second / (2 * mean)
    residual_time = arrival * second / 2
    wait = residual_time / (1 - rho)
    return mean, second, busy_shares, residual_busy, residual_time, wait


for name, atoms in [
    ("constant", [(1.0, 0.1)]),
    ("two sizes", [(0.5, 0.05), (0.5, 0.15)]),
    ("rare long", [(0.95, 0.05), (0.05, 1.05)]),
]:
    mean, second, shares, residual_busy, residual_time, wait = mixture_metrics(atoms, 8)
    print(name, "mean", round(mean, 6), "second", round(second, 6))
    print("busy-time shares:", [round(x, 6) for x in shares])
    print("residual if busy / at random time:", round(residual_busy, 6), round(residual_time, 6))
    print("mean wait / total:", round(wait, 6), round(wait + mean, 6))
''', "The busy-time shares are length-biased probabilities. Residual at a random time is zero while idle, so it differs from the residual conditional on a busy server.")

add("pooling", "Compare a central queue, split queues and one faster server",
    "Keep total capacity fixed at 10 jobs/second. Why need the three response times still differ?", r'''
from math import isfinite


def erlang_c(arrival, service, servers):
    if type(servers) is not int or not 1 <= servers <= 12:
        raise ValueError("Use one to twelve identical servers")
    if not all(isfinite(x) and 1e-6 <= x <= 1e6 for x in [arrival, service]):
        raise ValueError("Rates must lie in the supported positive range")
    gap = servers * service - arrival
    if gap <= 0:
        raise ValueError("Arrival rate must be below total capacity")
    offered_load = arrival / service
    weights = [1.0]
    for n in range(1, servers + 1):
        weights.append(weights[-1] * offered_load / n)
    tail_weight = weights[-1] * servers * service / gap
    normalizer = sum(weights[:-1]) + tail_weight
    chance_wait = tail_weight / normalizer
    return chance_wait, chance_wait / gap, chance_wait / gap + 1 / service


arrival, service, servers = 8, 5, 2
chance, wait, total = erlang_c(arrival, service, servers)
print("pooled probability of waiting:", round(chance, 6))
print("pooled mean wait / total:", round(wait, 6), round(total, 6))
print("uniform independent split, mean total:", round(1 / (service - arrival / servers), 6))
print("one server twice as fast, mean total:", round(1 / (servers * service - arrival), 6))
print("one-server check:", tuple(round(x, 6) for x in erlang_c(8, 10, 1)))
''', "Assumptions: central FCFS queue, independent Poisson input and iid exponential work at identical servers. Splitting means independent uniform routing. A faster server changes each job's mean service duration.")

add("finite-capacity", "Use admitted throughput in a finite buffer",
    "Can a finite queue have a stationary distribution when offered load exceeds one? What is lost?", r'''
from math import isfinite


def finite_queue(arrival, service, capacity):
    if type(capacity) is not int or not 1 <= capacity <= 20:
        raise ValueError("Capacity includes service; use one to twenty slots")
    if not all(isfinite(x) and 1e-6 <= x <= 1e6 for x in [arrival, service]):
        raise ValueError("Rates must lie in the supported positive range")
    rho = arrival / service
    weights = [rho**n for n in range(capacity + 1)]
    probabilities = [w / sum(weights) for w in weights]
    admitted = arrival * sum(probabilities[:-1])
    mean_number = sum(n * p for n, p in enumerate(probabilities))
    mean_queue = sum(max(0, n - 1) * p for n, p in enumerate(probabilities))
    return probabilities, admitted, mean_number, mean_number / admitted, mean_queue / admitted


for arrival in [8, 10, 12]:
    probabilities, admitted, count, total, wait = finite_queue(arrival, 10, 3)
    print("offered", arrival, "probabilities", [round(x, 6) for x in probabilities])
    print("drop / admitted rate:", round(probabilities[-1], 6), round(admitted, 6))
    print("L / W / Wq:", round(count, 6), round(total, 6), round(wait, 6))
print("one-slot queue wait:", finite_queue(12, 10, 1)[-1])
''', "M/M/1/K has K total slots, including the in-service job. Its finite stationary law does not mean offered work is all served. The rho=1 case is handled without dividing by 1-rho.")

add("simulation", "Simulate complete customer cohorts with independent replications",
    "Does one finite observed mean prove the formula? Why retain the completions of measured arrivals?", r'''
import random
import statistics
from math import isfinite


def simulate_fcfs(arrival_rate, service_rate, seed, warmup=5000, measured=50000):
    if (not all(isfinite(x) and 1e-6 <= x <= 1e6 for x in [arrival_rate, service_rate])
        or not arrival_rate < service_rate or type(warmup) is not int
        or type(measured) is not int or not 0 <= warmup <= 1000000
        or not 2 <= measured <= 1000000):
        raise ValueError("Use stable positive rates and a nonempty measured cohort")
    rng = random.Random(seed)
    previous_total = 0.0
    waits, totals = [], []
    for index in range(warmup + measured):
        gap = rng.expovariate(arrival_rate)
        service = rng.expovariate(service_rate)
        # Lindley's workload recursion avoids subtracting large calendar times.
        wait = max(0.0, previous_total - gap)
        total = wait + service
        previous_total = total
        if index >= warmup:
            waits.append(wait)
            totals.append(total)
    return statistics.mean(waits), statistics.mean(totals)


results = [simulate_fcfs(8, 10, seed) for seed in range(101, 109)]
means = [total for wait, total in results]
print("total-time means by independent run:", [round(x, 6) for x in means])
print("mean across runs:", round(statistics.mean(means), 6))
print("estimated Monte Carlo standard error:", round(statistics.stdev(means) / len(means)**0.5, 6))
print("stationary theoretical total mean:", 0.5)
print("theory minus finite estimate:", round(0.5 - statistics.mean(means), 6))
''', "This executes a seeded model, not a load test. Lindley's workload recursion tracks remaining durations instead of subtracting large calendar times, which can erase a short service duration after a very long idle period. Burn-in reduces an initially empty transient but does not certify its removal. A measured arrival's eventual completion is retained; jobs within a run are correlated, so the standard error uses independent run means.")

add("heavy-tail", "A stable queue with no finite mean wait",
    "When a Pareto service time has finite mean but infinite second moment, what do increasing truncations reveal?", r'''
from math import isfinite, sqrt


def truncated_pareto_moments(cap, minimum=1 / 30):
    # P(S > t) = (minimum/t)^(3/2) for t >= minimum.
    # For Y=min(S, cap), integrate P(Y>t) and 2t P(Y>t).
    if not all(isfinite(x) and 1e-6 <= x <= 1e6 for x in [cap, minimum]) or cap < minimum:
        raise ValueError("Use a finite cap at least as large as the positive minimum")
    mean = 3 * minimum - 2 * minimum**1.5 / sqrt(cap)
    second = 4 * minimum**1.5 * sqrt(cap) - 3 * minimum**2
    return mean, second


arrival = 8
for cap in [1, 10, 100, 1000]:
    mean, second = truncated_pareto_moments(cap)
    rho = arrival * mean
    wait = arrival * second / (2 * (1 - rho))
    print("cap", cap, "mean", round(mean, 6), "load", round(rho, 6), "mean wait", round(wait, 6))
print("Untruncated mean service = 0.1; load = 0.8; second moment and mean wait are infinite.")
''', "The increasing finite lower bounds support the explained monotone-coupling argument. A finite maximum in a simulation is not a bound on the original distribution's support or a proof that its mean waiting time is finite.")

add("vacations", "Separate congestion from waiting for a polling window",
    "If an idle server repeatedly takes independent 0.4-second vacations, can delay remain as arrivals become rare?", r'''
from math import isfinite


def repeated_vacation_wait(arrival, service_mean, service_second, vacation_mean, vacation_second):
    if not all(isfinite(x) for x in [arrival, service_mean, service_second, vacation_mean, vacation_second]):
        raise ValueError("All inputs must be finite")
    if arrival <= 0 or service_mean <= 0 or vacation_mean <= 0:
        raise ValueError("Rates and means must be positive")
    if service_second < service_mean**2 or vacation_second < vacation_mean**2:
        raise ValueError("A second moment cannot be smaller than the squared mean")
    rho = arrival * service_mean
    if rho >= 1:
        raise ValueError("The service load must be below one")
    congestion = arrival * service_second / (2 * (1 - rho))
    vacation_residual = vacation_second / (2 * vacation_mean)
    if not all(isfinite(x) for x in [congestion, vacation_residual, congestion + vacation_residual]):
        raise ValueError("Arithmetic range exceeded; rescale the inputs")
    return congestion, vacation_residual, congestion + vacation_residual


for arrival in [8, 1, 0.01]:
    terms = repeated_vacation_wait(arrival, 0.1, 0.1**2, 0.4, 0.4**2)
    print("arrival", arrival, "congestion / vacation / total wait:", tuple(round(x, 6) for x in terms))
''', "This formula assumes repeated iid independent vacations whenever the system is empty, exhaustive FCFS service when work is found, and the usual M/G/1 input assumptions. A one-time or load-dependent setup delay is a different model.")

destination = ROOT / "src/learn/data/queueing-examples.js"
destination.write_text("// Complete Python examples; stdout captured by generate-queueing-examples.py.\nexport const queueingExamples = " + json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
out = ROOT / "scratch/queueing-authoring"
out.mkdir(parents=True, exist_ok=True)
(out / "executed-programs.json").write_text(json.dumps({"programs": len(examples), "outputs": {x["id"]: x["expected"] for x in examples}}, indent=2), encoding="utf-8")
print(f"Executed {len(examples)} complete programs; both original code blocks preserved.")
