import contextlib
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import mpmath as mp

mp.mp.dps = 70
directory = Path('scratch/geometry-trigonometry-independent')
data = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
worst = 0.0
def close(actual, expected):
    global worst
    error = abs(float(actual) - float(expected))
    worst = max(worst, error)
    assert error <= 2e-11 * max(1, abs(float(expected))), (actual, str(expected), error)

for row in data['frames']:
    px, py, ox, oy, degrees, mode = row['inputs']
    theta = mp.pi * degrees / 180
    rotation = mp.matrix([[mp.cos(theta), -mp.sin(theta)], [mp.sin(theta), mp.cos(theta)]])
    offset = mp.matrix([px - ox, py - oy])
    local = mp.lu_solve(rotation, offset)
    active = mp.matrix([ox, oy]) + rotation * offset
    for index in range(2):
        close(row['result']['local'][index], local[index])
        close(row['result']['rotated'][index], active[index])
        close(row['result']['reconstructed'][index], [px, py][index])
for row in data['circles']:
    theta = mp.pi * row['degrees'] / 180
    for key, expected in [('cosine', mp.cos(theta)), ('sine', mp.sin(theta)), ('tangent', mp.tan(theta))]:
        close(row[key], expected)

contexts = {}
for example in data['examples']:
    context, stdout = {}, io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(example['code'], example['id'], 'exec'), context)
    assert stdout.getvalue().strip() == example['expected'].strip()
    contexts[example['id']] = context

ssa_cases = [(1, 2), (1, 3), (500, 1000), (501, 1000), (999, 1000), (1000, 1000),
             (1000, 999), (999, 999), (7, 13), (5, 8), (1, 1), (1000, 1)]
for a, b in ssa_cases:
    actual = contexts['triangles']['ssa_thirty'](a, b)
    height = mp.mpf(b) / 2
    center = mp.mpf(b) * mp.sqrt(3) / 2
    expected_count = 0 if a < height else 1 if a == height or a >= b else 2
    assert len(actual) == expected_count
    for c, angle_b, angle_c in actual:
        close((mp.mpf(c) - center) ** 2 + height ** 2, a * a)
        close(mp.mpf(angle_b) + angle_c + 30, 180)
        assert c > 0 and angle_b > 0 and angle_c > 0
        # Every side/angle pair shares the same sine-law scale.
        close(mp.mpf(a) / mp.sin(mp.pi / 6), mp.mpf(b) / mp.sin(mp.pi * mp.mpf(angle_b) / 180))

arc_cases = 0
for radius in [0.001, 0.125, 2.5, 1234.5]:
    for degrees in [0.25, 37.5, 195.25, 360]:
        angle, length, area = contexts['arcs']['sector'](radius, degrees)
        exact = mp.pi * mp.mpf(degrees) / 180
        close(angle, exact); close(length, radius * exact); close(area, mp.mpf(radius) ** 2 * exact / 2)
        arc_cases += 1
screen_cases = 0
for point in [(-7.5, 4.25), (0, 0), (2.125, -8.5)]:
    for origin in [(0, 0), (101.5, 240.25)]:
        for scales in [(3.5, 17.25), (30, 0.5), (1, 1)]:
            pixel = contexts['screen']['screen'](point, origin, scales)
            recovered = contexts['screen']['world'](pixel, origin, scales)
            close(recovered[0], point[0]); close(recovered[1], point[1])
            screen_cases += 1
link_cases = 0
for lengths in [(0.125, 2.75), (7.5, 3.25), (2, 2)]:
    for shoulder in [-127.25, 0, 83.75]:
        for elbow in [-180, -37.5, 0, 91.25, 180]:
            result = contexts['links']['endpoint'](*lengths, shoulder, elbow)
            # Geometric complex rotation of a locally assembled elbow replaces two world-angle components.
            local = lengths[0] + lengths[1] * mp.exp(1j * mp.pi * mp.mpf(elbow) / 180)
            expected = local * mp.exp(1j * mp.pi * mp.mpf(shoulder) / 180)
            close(result[0], expected.real); close(result[1], expected.imag)
            close(math.hypot(*result) ** 2, lengths[0] ** 2 + lengths[1] ** 2 + 2 * lengths[0] * lengths[1] * mp.cos(mp.pi * mp.mpf(elbow) / 180))
            link_cases += 1
result = {'reviewedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'counts': {'frames': len(data['frames']), 'circles': len(data['circles']), 'programs': len(contexts),
                     'ssaBoundaryCases': len(ssa_cases), 'offGridNativeArcs': arc_cases,
                     'changedScreenMaps': screen_cases, 'changedLinkConfigurations': link_cases},
          'maximumAbsoluteDifference': worst, 'precision': 'mpmath 70 decimal digits',
          'limits': 'Finite implementation checks accompany an independent complete proof/source read; no all-input floating-point bound is asserted.'}
(directory / 'native-results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
