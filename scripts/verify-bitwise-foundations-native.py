"""Check displayed programs against sets, decimal counts and binary strings."""
import contextlib
import hashlib
import io
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

directory = Path('scratch/bitwise-author')
payload = json.loads((directory / 'payload.json').read_text(encoding='utf-8'))
namespaces = {}
program_results = []
for group in ('originalExamples', 'bitwiseExamples'):
    for name, example in payload[group].items():
        output = io.StringIO()
        namespace = {'__name__': '__main__'}
        with contextlib.redirect_stdout(output):
            exec(compile(example['code'], f'{group}/{name}.py', 'exec'), namespace)
        assert output.getvalue().strip() == example['output'].strip(), (group, name, output.getvalue())
        namespaces[f'{group}/{name}'] = namespace
        program_results.append({'group': group, 'name': name, 'stdout': output.getvalue().strip(),
                                'codeSha256': hashlib.sha256(example['code'].encode()).hexdigest(), 'passed': True})

bitsets = namespaces['bitwiseExamples/bitsets']
words = namespaces['bitwiseExamples/words']
parity = namespaces['bitwiseExamples/parity']
population = namespaces['bitwiseExamples/population']
two = namespaces['bitwiseExamples/twoSingletons']

for state in payload['wordPayload']:
    width, unsigned, shift = state['width'], state['unsigned'], state['shift']
    bits = format(unsigned, f'0{width}b')
    signed = sum(int(bit) * (-(1 << (width - 1)) if index == 0 else 1 << (width - index - 1))
                 for index, bit in enumerate(bits))
    assert words['decode_signed'](unsigned, width) == state['signed'] == signed
    assert words['encode_signed'](signed, width) == unsigned
    assert unsigned >> shift == state['logical']
    assert signed >> shift == state['arithmetic']
    assert ((unsigned << shift) & ((1 << width) - 1)) == state['leftWord']

for trace in payload['sparsePayload']:
    value = trace['value']
    assert population['population_count'](value) == trace['population'] == bin(value).count('1')
    for state in trace['states'][:-1]:
        before = state['current']
        assert before & (before - 1) == state['next']
        assert len(set(bitsets['unpack'](before, 8)) - set(bitsets['unpack'](state['next'], 8))) == 1

for trace in payload['parityPayload']:
    values = trace['values']
    assert parity['xor_fold'](values) == trace['states'][-1]['accumulator']
    assert parity['audit_promise'](values) == trace['oneSingletonPromise']
    counts = Counter(values)
    if trace['twoSingletonPromise']:
        assert sorted(two['two_singletons'](values)) == sorted(value for value, count in counts.items() if count == 1)

for value in range(65536):
    assert population['population_count'](value) == format(value, 'b').count('1')
    assert population['is_power_of_two'](value) == (value in {1 << bit for bit in range(16)})
for value in [-1, -8, -(1 << 100)]:
    assert not population['is_power_of_two'](value)
for width in (1, 4, 8, 16, 32, 100, 1024):
    for signed in (-(1 << (width - 1)), -1, 0, (1 << (width - 1)) - 1):
        word = words['encode_signed'](signed, width)
        assert words['decode_signed'](word, width) == signed

rng = random.Random(110926)
singleton_cases = 0
for _ in range(800):
    choices = rng.sample(range(-1000, 1001), 8)
    singleton = choices[0]
    values = [singleton] + [value for value in choices[2:] for _ in range(2)]
    rng.shuffle(values)
    assert parity['xor_fold'](values) == singleton
    assert parity['audit_promise'](values)
    values.append(choices[1])
    rng.shuffle(values)
    snapshot = list(values)
    assert set(two['two_singletons'](values)) == set(choices[:2])
    assert values == snapshot
    singleton_cases += 1

posting_cases = 0
for width in range(1, 9):
    full = bitsets['full_mask'](width)
    for _ in range(60):
        first = {i for i in range(width) if rng.randrange(2)}
        second = {i for i in range(width) if rng.randrange(2)}
        left = bitsets['pack'](sorted(first) * 2, width)
        right = bitsets['pack'](second, width)
        for value, expected in [(left & right, first & second), (left | right, first | second),
                                (left ^ right, first ^ second), (left & ~right, first - second),
                                ((~left) & full, set(range(width)) - first)]:
            assert bitsets['unpack'](value, width) == sorted(expected)
            assert population['population_count'](value) == len(expected)
            posting_cases += 1

bad_calls = [lambda: bitsets['full_mask'](0), lambda: bitsets['full_mask'](1025),
             lambda: bitsets['pack']([6], 6), lambda: bitsets['pack']([True], 6),
             lambda: bitsets['unpack'](-1, 8), lambda: bitsets['unpack'](256, 8),
             lambda: words['encode_signed'](128, 8), lambda: words['encode_signed'](-129, 8),
             lambda: words['decode_signed'](True, 8), lambda: population['population_count'](-1),
             lambda: population['population_count'](True), lambda: population['is_power_of_two'](2.0),
             lambda: parity['xor_fold']([]), lambda: parity['xor_fold']([True]),
             lambda: two['two_singletons'](iter([1, 2])), lambda: two['two_singletons']([1, 1])]
for call in bad_calls:
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid native input was accepted.')

changed = {'setMasks': {'A': 26, 'B': 41, 'intersection': 26 & 41, 'union': 26 | 41,
                        'xor': 26 ^ 41, 'complementA': (~26) & 63},
           'signedWord': {'unsigned': 11, 'signed': words['decode_signed'](11, 4),
                          'logical': 11 >> 1, 'arithmetic': -5 >> 1, 'boundedLeft': (11 << 1) & 15},
           'invalidParity': parity['xor_fold']([5, 5, 5, 9, 9]),
           'report': {'intersection': bitsets['unpack'](11 & 13, 4), 'xor': bitsets['unpack'](11 ^ 13, 4)}}
assert changed['setMasks'] == dict(A=26, B=41, intersection=8, union=59, xor=51, complementA=37)
assert changed['signedWord'] == dict(unsigned=11, signed=-5, logical=5, arithmetic=-3, boundedLeft=6)
assert changed['report'] == dict(intersection=[0, 3], xor=[1, 2])
for source in payload['sources']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'sources': payload['sources'],
          'modelChecks': payload['checks'], 'originalTeachingSubtrees': payload['originalTeachingSubtrees'],
          'programs': program_results, 'nativeAdditionalChecks': {'populationValues': 65536, 'signedSingletonPairs': singleton_cases,
          'postingComparisons': posting_cases, 'invalidInputs': len(bad_calls)}, 'changedPractice': changed,
          'limits': 'Finite exhaustive checks and changed native cases support the general proofs; no judge submissions, performance benchmark or user study is claimed.'}
(directory / 'native-results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Passed all {len(program_results)} displayed programs, signed/native changes, set/word/parity oracles and original teaching conservation.')
