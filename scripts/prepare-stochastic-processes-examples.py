"""Prepare complete learner programs and verified stdout."""
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap

directory = Path("scratch/stochastic-processes-native")
directory.mkdir(parents=True, exist_ok=True)
examples = {}


def add(key, title, question, code):
    examples[key] = {"title": title, "question": question,
                     "code": textwrap.dedent(code).strip() + "\n"}


original = Path("scratch/stochastic-processes-design/original-lesson.jsx").read_text(encoding="utf-8")
tick = chr(96)
original_codes = re.findall('<CodeBlock language="python">\\{' + tick + '(.*?)' + tick + '\\}</CodeBlock>', original, re.S)
add("weather", "The original five-day weather calculation",
    "Starting sunny, what is the chance of each state after five transitions?", original_codes[0])
add("arrivalProbability", "The original no-arrival calculation",
    "At 2.5 arrivals per minute, what is the probability of no arrival in three minutes?", original_codes[1])
add("brownianOriginal", "The original five-step Brownian sample",
    "Does a negative endpoint contradict a process whose mean is zero?", original_codes[2])

add("pathLaws", "Same marginals, different temporal laws",
    "Can the same fair marginal at every time hide radically different sequences?", '''
    from fractions import Fraction
    from itertools import product


    def law(kind, length):
        paths = list(product([0, 1], repeat=length))
        if kind == "frozen":
            paths = [p for p in paths if len(set(p)) == 1]
        if kind == "alternating":
            paths = [p for p in paths if all(p[i] != p[i-1] for i in range(1, length))]
        mass = Fraction(1, len(paths))
        return [(path, mass) for path in paths]


    for kind in ["fresh", "frozen", "alternating"]:
        paths = law(kind, 4)
        marginals = [sum(mass * path[t] for path, mass in paths) for t in range(4)]
        same = sum(mass for path, mass in paths if path[0] == path[1])
        print(kind, "paths:", len(paths), "P(X_t=1):", [str(p) for p in marginals])
        print("P(next equals current):", same)

    # Hidden cycle A -> B -> C -> A; report 0 for A/B and 1 for C.
    histories = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
    for previous in [1, 0]:
        compatible = [path for path in histories if path[:2] == (previous, 0)]
        chance = Fraction(sum(path[2] for path in compatible), len(compatible))
        print("P(next=1 | previous=" + str(previous) + ", current=0):", chance)
    ''')

add("markov", "Propagate, fit and distinguish equilibrium",
    "Does slowing both switching probabilities change the equilibrium or only the approach to it?", '''
    from fractions import Fraction as F


    def propagate(matrix, initial, steps):
        if type(steps) is not int or steps < 0:
            raise ValueError("Use a nonnegative integer step count.")
        if not matrix or len(matrix) != len(initial):
            raise ValueError("Shapes must match.")
        n = len(initial)
        if any(len(row) != n or any(x < 0 for x in row) or sum(row) != 1 for row in matrix):
            raise ValueError("Each row must be a probability vector.")
        if any(x < 0 for x in initial) or sum(initial) != 1 or steps < 0:
            raise ValueError("Invalid initial law or step count.")
        result = list(initial)
        for _ in range(steps):
            result = [sum(result[i] * matrix[i][j] for i in range(n)) for j in range(n)]
        return result


    def fit_transitions(trajectories, states):
        if type(states) is not int or states < 1 or not trajectories:
            raise ValueError("Use a positive state count and separate trajectories.")
        counts = [[0] * states for _ in range(states)]
        for path in trajectories:
            if not path or any(type(x) is not int or not 0 <= x < states for x in path):
                raise ValueError("Use nonempty paths with valid states.")
            for before, after in zip(path, path[1:]):
                counts[before][after] += 1
        fitted = [None if sum(row) == 0 else [F(x, sum(row)) for x in row] for row in counts]
        return counts, fitted


    for a, b in [(F(1, 5), F(3, 10)), (F(1, 50), F(3, 100)), (F(1), F(1))]:
        matrix = [[1-a, a], [b, 1-b]]
        stationary = [b/(a+b), a/(a+b)]
        print("switch:", str(a), str(b), "stationary:", [str(x) for x in stationary])
        print("sunny after 5:", round(float(propagate(matrix, [F(1), F(0)], 5)[0]), 6))
    matrix = [[F(1), F(0)], [F(0), F(1)]]
    print("identity retains initial:", propagate(matrix, [F(1, 4), F(3, 4)], 10))
    counts, fitted = fit_transitions([[0, 0, 1, 0, 1], [1, 1, 0], [2]], 3)
    print("departure counts:", counts)
    print("fitted rows:", [None if row is None else [str(x) for x in row] for row in fitted])
    ''')

add("absorption", "Solve a finite first-passage problem exactly",
    "From reserve two, how does an upward probability of one quarter change success and stopping time?", '''
    from fractions import Fraction as F


    def solve(matrix, rhs):
        a = [[F(x) for x in row] + [F(value)] for row, value in zip(matrix, rhs)]
        for column in range(len(a)):
            pivot = next((r for r in range(column, len(a)) if a[r][column]), None)
            if pivot is None:
                raise ValueError("Singular system.")
            a[column], a[pivot] = a[pivot], a[column]
            divisor = a[column][column]
            a[column] = [x / divisor for x in a[column]]
            for row in range(len(a)):
                if row != column:
                    coefficient = a[row][column]
                    a[row] = [x - coefficient*y for x, y in zip(a[row], a[column])]
        return [row[-1] for row in a]


    def reserve(boundary, upward):
        if type(boundary) is not int or not 2 <= boundary <= 12 or not 0 < upward < 1:
            raise ValueError("Use boundary 2..12 and 0 < p < 1.")
        n = boundary-1
        system = [[F(i == j) for j in range(n)] for i in range(n)]
        for i in range(n):
            if i > 0:
                system[i][i-1] = -(1-upward)
            if i+1 < n:
                system[i][i+1] = -upward
        rhs = [F(0)] * n
        rhs[-1] = upward
        h = [F(0)] + solve(system, rhs) + [F(1)]
        times = [F(0)] + solve(system, [F(1)] * n) + [F(0)]
        return h, times


    for p in [F(1, 2), F(1, 4)]:
        h, times = reserve(4, p)
        print("p =", p, "success probabilities:", [str(x) for x in h])
        print("mean steps to either boundary:", [str(x) for x in times])
    mass = [F(0), F(0), F(1), F(0), F(0)]
    for step in range(1, 5):
        following = [mass[0], F(0), F(0), F(0), mass[4]]
        for state in range(1, 4):
            following[state-1] += mass[state]/2
            following[state+1] += mass[state]/2
        mass = following
    print("after 4 steps: lower, still moving, upper =", mass[0], sum(mass[1:4]), mass[4])
    ''')

add("arrivals", "Simulate event times and route the same arrivals",
    "How do event epochs become interval counts, and what is still unknown at the observation cutoff?", '''
    import math
    import random


    def event_times(rate, horizon, seed, cap=10000):
        if not math.isfinite(rate) or rate < 0 or not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("Use finite nonnegative rate and positive horizon.")
        if rate == 0:
            return []
        rng = random.Random(seed)
        time, events = 0.0, []
        for _ in range(cap):
            duration = -math.log1p(-rng.random()) / rate
            if duration <= 0 or time + duration <= time:
                raise ArithmeticError("The next positive wait cannot be represented.")
            time += duration
            if time > horizon:
                return events
            events.append(time)
        raise RuntimeError("Event cap reached; do not treat an incomplete path as complete.")


    events = event_times(2.5, 3, 17)
    print("event times:", [round(t, 4) for t in events])
    print("counts at 1, 2, 3 minutes:", [sum(t <= end for t in events) for end in [1, 2, 3]])
    print("count in (1, 2]:", sum(1 < t <= 2 for t in events))
    rng = random.Random(93)
    marks = [int(rng.random() >= 0.4) for _ in events]
    print("routed counts:", [marks.count(i) for i in [0, 1]])
    print("last observed unfinished gap:", round(3 - (events[-1] if events else 0), 4))
    mean, p, m, n = 3.0, 0.4, 1, 2
    conditional = math.comb(m+n, m) * p**m * (1-p)**n
    joint = math.exp(-mean) * mean**(m+n) / math.factorial(m+n) * conditional
    product = (math.exp(-mean*p) * (mean*p)**m / math.factorial(m)
               * math.exp(-mean*(1-p)) * (mean*(1-p))**n / math.factorial(n))
    print("conditional / joint / product:", *(round(x, 6) for x in [conditional, joint, product]))
    print("P(third event by time 1):", round(1 - math.exp(-2.5)*sum(2.5**k/math.factorial(k) for k in range(3)), 6))
    ''')

add("intensity", "Turn cumulative intensity back into event time",
    "Do the first and last one-minute windows have the same arrival law when their rates differ?", '''
    import math


    def intensity(t, first_rate=1.0, second_rate=4.0, change=2.0):
        return first_rate * min(t, change) + second_rate * max(0, t-change)


    def event_clock(unit_times, first_rate, second_rate, change, horizon):
        if any(not math.isfinite(x) or x < 0 for x in [first_rate, second_rate, change, horizon]):
            raise ValueError("Clock parameters must be finite and nonnegative.")
        if change > horizon or any(not math.isfinite(x) or x <= 0 for x in unit_times):
            raise ValueError("Use positive unit-clock event times and an in-window change.")
        if any(b <= a for a, b in zip(unit_times, unit_times[1:])):
            raise ValueError("Unit-clock times must increase.")
        result = []
        for u in unit_times:
            if u > intensity(horizon, first_rate, second_rate, change):
                break
            first_area = first_rate * change
            if first_rate > 0 and u <= first_area:
                result.append(u / first_rate)
            elif second_rate > 0:
                result.append(change + (u-first_area)/second_rate)
        return result


    clock = [0.2, 1.1, 2.2, 3.0, 5.8, 6.5]
    print("events by minute 3:", event_clock(clock, 1, 4, 2, 3))
    print("same unit events, first rate zero:", event_clock(clock, 0, 4, 2, 3))
    for start, end in [(0, 1), (2, 3), (0, 3)]:
        mean = intensity(end)-intensity(start)
        print((start, end), "mean:", mean, "P(no arrival):", round(math.exp(-mean), 6))
    ''')

add("jumpClock", "Count time exposure instead of only state visits",
    "A device alternates On and Off at jumps. Why can it still spend 80% of its time On?", '''
    import math
    import random


    def simulate(alpha, beta, horizon, seed, cap=10000):
        if any(not math.isfinite(x) or x <= 0 for x in [alpha, beta, horizon]):
            raise ValueError("Use positive finite rates and horizon.")
        rng = random.Random(seed)
        time, state = 0.0, 0
        exposure, departures = [0.0, 0.0], [0, 0]
        for _ in range(cap):
            duration = -math.log1p(-rng.random()) / [alpha, beta][state]
            if duration <= 0 or time + duration <= time:
                raise ArithmeticError("The next positive hold cannot be represented.")
            available = min(duration, horizon-time)
            exposure[state] += available
            time += available
            if time >= horizon:
                return exposure, departures
            departures[state] += 1
            state = 1-state
        raise RuntimeError("Jump cap reached; observation window is incomplete.")


    alpha, beta, horizon = 0.5, 2.0, 20.0
    pi_on = beta/(alpha+beta)
    p_on = pi_on+(1-pi_on)*math.exp(-(alpha+beta)*horizon)
    expected_on_time = pi_on*horizon+(1-pi_on)*(-math.expm1(-(alpha+beta)*horizon))/(alpha+beta)
    exposure, departures = simulate(alpha, beta, horizon, 31)
    print("stationary On probability:", pi_on)
    print("P(On at horizon), expected On exposure:", round(p_on, 6), round(expected_on_time, 6))
    print("one-path On/Off exposure:", [round(t, 6) for t in exposure])
    print("departures from each state:", departures)
    print("one-path On time fraction:", round(exposure[0]/horizon, 6))
    print("Euler row at dt=1 from Off:", [beta, 1-beta], "(invalid probability row)")
    ''')

add("brownianGrid", "Scale increments and inspect the same path twice",
    "What changes when you observe a finer grid, and which endpoint must remain identical?", '''
    import math
    import random


    def brownian_path(normals, horizon, drift=0.0, scale=1.0):
        if not normals or any(not math.isfinite(z) for z in normals):
            raise ValueError("Provide finite normal draws.")
        if not math.isfinite(horizon) or horizon <= 0 or not math.isfinite(scale) or scale <= 0 or not math.isfinite(drift):
            raise ValueError("Positive finite horizon and scale required.")
        dt = horizon/len(normals)
        path = [0.0]
        for z in normals:
            path.append(path[-1]+drift*dt+scale*math.sqrt(dt)*z)
        return path


    rng = random.Random(19)
    normals = [rng.gauss(0, 1) for _ in range(16)]
    fine = brownian_path(normals, 2, drift=0.3, scale=0.7)
    for stride in [4, 2, 1]:
        path = fine[::stride]
        dt = 2/(len(path)-1)
        increments = [b-a for a, b in zip(path, path[1:])]
        q = sum(x*x for x in increments)
        expected_q = 0.7**2*2+0.3**2*2*dt
        print("steps:", len(increments), "endpoint:", round(path[-1], 6),
              "observed Q:", round(q, 6), "E[Q]:", round(expected_q, 6))
    times = [0.5, 1.0, 2.0]
    print("model covariance:")
    for s in times:
        print([round(0.7**2*min(s, t), 6) for t in times])
    print("terminal mean / variance:", 0.3*2, round(0.7**2*2, 6))
    ''')

add("bridgeVariation", "Calculate hidden-between-sample uncertainty",
    "If both endpoints are zero, can the continuous path still cross one between them?", '''
    import math
    from itertools import product
    from fractions import Fraction as F


    def bridge(left, right, duration, scale, fraction, barrier):
        if any(not math.isfinite(value) for value in [left, right, duration, scale, fraction, barrier]):
            raise ValueError("All bridge parameters must be finite.")
        if duration <= 0 or scale <= 0 or not 0 <= fraction <= 1:
            raise ValueError("Use positive duration/scale and fraction in [0,1].")
        mean = left+(right-left)*fraction
        variance = scale**2*duration*fraction*(1-fraction)
        log_crossing = 0 if barrier <= max(left, right) else -2*(barrier-left)*(barrier-right)/(scale**2*duration)
        return mean, variance, log_crossing


    for arguments in [(0, 0, 1, 1, 0.5, 1), (0.2, -0.1, 0.5, 0.8, 0.25, 0.7)]:
        mean, variance, log_crossing = bridge(*arguments)
        print("conditional mean / variance / crossing:", *(round(x, 6) for x in [mean, variance, math.exp(log_crossing)]))
    for n in [4, 16, 64]:
        print("standard BM, T=2, steps", n, "E[Q] =", 2, "Var(Q) =", round(8/n, 6))
    stopped = []
    for increments in product([-1, 1], repeat=4):
        position = 0
        for step in increments:
            position += step
            if abs(position) == 2:
                break
        stopped.append(position)
    print("bounded stopped mean:", F(sum(stopped), len(stopped)))
    print("P(max W on [0,1] >= 1):", round(math.erfc(1/math.sqrt(2)), 6))
    ''')

add("diagnosis", "Check a process beyond its marginal histogram",
    "Which temporal diagnostic catches a model failure that matching event totals miss?", '''
    import math
    from statistics import mean, pvariance


    def fitted_rate(events, horizon):
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("Positive finite exposure required.")
        if any(not math.isfinite(t) or not 0 < t <= horizon for t in events):
            raise ValueError("Events must fall in the observed window.")
        if any(b <= a for a, b in zip(events, events[1:])):
            raise ValueError("This simple-event trace must be strictly ordered.")
        return len(events)/horizon


    train = [0.2, 0.7, 1.4, 1.8, 2.3, 2.9]
    regular = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    clustered = [0.1, 0.11, 0.12, 2.8, 2.81, 2.82]
    rate = fitted_rate(train, 3)
    print("training rate / forecast next-minute mean:", rate, rate)
    print("no event next minute:", round(math.exp(-rate), 6))
    for name, events in [("regular", regular), ("clustered", clustered)]:
        fitted_rate(events, 3)
        bins = [sum(i < t <= i+1 for t in events) for i in range(3)]
        print(name, "same total:", len(events), "one-minute counts:", bins)
    print("three bins are too few to establish a Poisson model")
    times = [0, 0.25, 1.0, 2.0]
    positions = [0, 0.2, -0.1, 0.6]
    standardized = [(b-a)/math.sqrt(t-s) for s, t, a, b in
                    zip(times, times[1:], positions, positions[1:])]
    print("normalized increments:", [round(x, 6) for x in standardized])
    print("descriptive mean / variance:", round(mean(standardized), 6), round(pvariance(standardized), 6))
    print("unknown drift, dependence and observation noise need separate checks")
    ''')

records = []
for key, example in examples.items():
    file = directory / (key + ".py")
    file.write_text(example["code"], encoding="utf-8")
    result = subprocess.run([sys.executable, str(file)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(key + ": " + result.stderr)
    example["expected"] = result.stdout.strip()
    records.append({"key": key, "file": str(file), "exitCode": result.returncode,
                    "stdout": result.stdout.strip()})

Path("src/learn/data/stochastic-processes-examples.js").write_text(
    "export const stochasticProcessesExamples = " +
    json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
report = {"executedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "python": sys.version, "programs": records}
(directory / "program-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("PASS:", len(records), "complete native programs executed; output stored with source.")
