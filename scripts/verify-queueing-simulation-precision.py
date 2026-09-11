"""Verify actual displayed simulation against exact arithmetic on identical random draws."""
import contextlib
import hashlib
import io
import json
import random
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'scratch/queueing-simulation-precision'
OUT.mkdir(parents=True, exist_ok=True)
node = """
import {queueingExamples as current} from './src/learn/data/queueing-examples.js';
import {queueingExamples as previous} from './scratch/queueing-authoring/pre-simulation-precision-examples.js';
process.stdout.write(JSON.stringify({current,previous}));
"""
payload = json.loads(subprocess.run(['node', '--input-type=module', '-e', node], cwd=ROOT,
                                    text=True, capture_output=True, check=True).stdout)
previous = {item['id']: item for item in payload['previous']}
examples = {item['id']: item for item in payload['current']}
assert set(previous) == set(examples)
for key, item in examples.items():
    assert item['expected'] == previous[key]['expected'], key
    if key != 'simulation':
        assert item == previous[key], key
namespace = {}
with contextlib.redirect_stdout(io.StringIO()) as stdout:
    exec(compile(examples['simulation']['code'], 'displayed-simulation.py', 'exec'), namespace)
assert stdout.getvalue().strip() == examples['simulation']['expected']

results = []
for arrival_rate, service_rate, seed, warmup, measured in [
    (1e-6, 1e6, 17, 100000, 100),
    (1e-6, 1e6, 19, 10000, 1000),
    (4, 7, 223, 100, 750),
    (9.9, 10, 251, 100, 750),
    (1e5, 1e6, 271, 200, 500),
    (1e-6, 2e-6, 293, 200, 500),
]:
    rng = random.Random(seed)
    arrival = available = Fraction(0)
    waits = totals = services = Fraction(0)
    for index in range(warmup+measured):
        arrival += Fraction.from_float(rng.expovariate(arrival_rate))
        service = Fraction.from_float(rng.expovariate(service_rate))
        start = max(arrival, available)
        available = start+service
        if index >= warmup:
            waits += start-arrival
            totals += available-arrival
            services += service
    expected = [float(waits/measured), float(totals/measured)]
    actual = namespace['simulate_fcfs'](arrival_rate, service_rate, seed, warmup, measured)
    for found, reference in zip(actual, expected):
        assert abs(found-reference) <= 1e-12*max(abs(reference), 1e-10), (actual, expected)
    assert totals == waits+services
    results.append({'arrivalRate': arrival_rate, 'serviceRate': service_rate, 'seed': seed,
                    'warmup': warmup, 'measured': measured, 'actual': actual,
                    'exactCalendarArithmeticMeans': expected,
                    'meanSampledService': float(services/measured)})
record = {'at': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'allTenPrintedOutputsUnchanged': True, 'otherNineExamplesUnchanged': True,
          'oracle': 'Exact Fraction calendar arrival/start/finish arithmetic using identical sampled floats; production uses relative workload recurrence.',
          'cases': results,
          'exampleSha256': hashlib.sha256((ROOT/'src/learn/data/queueing-examples.js').read_bytes()).hexdigest()}
(OUT/'results.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
