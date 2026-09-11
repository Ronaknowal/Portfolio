import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 180
data = json.loads(Path('scratch/pde-wave-boundary/fixtures.json').read_text())
environment = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(data['example']['code'], environment)
assert output.getvalue().rstrip() == data['example']['expected']

def primitive(z):
    return z-z**3+mp.mpf(3)*z**5/5-z**7/7

def reference(x, time, velocity, speed=1):
    x, time, velocity, speed = map(mp.mpf, (x, time, velocity, speed))
    first, last = x-speed*time, x+speed*time
    def bump(z):
        return ((1-z)*(1+z))**3 if abs(z)<1 else mp.mpf(0)
    shape = (bump(first)+bump(last))/2
    low, high = max(mp.mpf(-1), first), min(mp.mpf(1), last)
    moving = velocity*(primitive(high)-primitive(low))/(2*speed) if high>low else mp.mpf(0)
    return shape, moving, shape+moving

largest_relative = 0.0
comparisons = 0
def check(actual, expected):
    global largest_relative, comparisons
    if expected == 0:
        assert actual == 0, (actual, expected)
    else:
        relative = abs(mp.mpf(actual)/expected-1)
        assert relative < mp.mpf('5e-13'), (actual, expected, relative)
        largest_relative = max(largest_relative, float(relative))
    comparisons += 1

for case in data['cases']:
    x, time, velocity = case['x'], case['time'], case['velocity']
    expected = reference(x, time, velocity)
    actual = case['actual']
    check(actual['rightMoving']+actual['leftMoving'], expected[0])
    check(actual['velocityContribution'], expected[1])
    check(actual['total'], expected[2])
    for native, exact in zip(environment['wave'](x, time, velocity), expected):
        check(native, exact)
    assert actual['velocityContribution'] >= 0 and actual['total'] >= 0
for speed in [.25, .7, 2, 3]:
    for x in [-1, -.3, 0, 1]:
        for time in [1e-12, .2]:
            for actual, exact in zip(environment['wave'](x, time, -.4, speed), reference(x, time, -.4, speed)):
                check(actual, exact)
sources = ['src/learn/data/pde-models.js', 'src/learn/data/pde-examples.js']
record = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'cases': len(data['cases']),
          'comparisons': comparisons, 'maximumRelativeError': largest_relative,
          'oracle': '180-digit independent clipped antiderivative and exact input-float conversion; original primitive cancellation is intentionally absent at high precision.',
          'sourceHashes': {source: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources}}
Path('docs/teaching/evidence/pde-wave-boundary-verification.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
