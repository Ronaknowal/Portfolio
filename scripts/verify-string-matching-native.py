from pathlib import Path
from itertools import product
import contextlib
import io
import json
import random
import unicodedata

examples = json.loads(Path('scratch/string-matching-verification/examples.json').read_text(encoding='utf-8'))
programs = {}
for key, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'], key, 'exec'), namespace)
    programs[key] = namespace

def words(alphabet, maximum):
    return [''.join(letters) for width in range(maximum + 1) for letters in product(alphabet, repeat=width)]

def matches(text, pattern):
    return [start for start in range(len(text) - len(pattern) + 1)
            if text[start:start + len(pattern)] == pattern]

counts = dict(search_cases=0, prefix_cases=0, stream_partitions=0,
              structure_cases=0, palindrome_cases=0, dna_cases=0,
              substring_queries=0, polynomial_pairs=0)
small = words('ab', 7)
patterns = words('ab', 4)
for text in small + ['a🙂a🙂a', 'e\u0301é', 'adbaad']:
    for pattern in patterns + ['🙂a', 'é']:
        expected = matches(text, pattern)
        assert programs['baseline']['naive_all'](text, pattern)[0] == expected
        assert programs['kmp']['find_all'](text, pattern) == expected
        assert programs['z']['z_find_all'](text, pattern) == expected
        for base, modulus in [(3, 7), (31, 1_000_000_007)]:
            starts, audit = programs['rolling']['rolling_find_all'](text, pattern, base, modulus)
            assert starts == expected
            for start, exact, checks in audit:
                assert exact == (start in expected)
                assert 1 <= checks <= len(pattern)
        counts['search_cases'] += 1

for text in words('abc', 6) + ['🙂a🙂', 'aabaaab', 'aabaaac']:
    expected = [max([length for length in range(position + 1)
                     if text[:length] == text[position + 1 - length:position + 1]])
                for position in range(len(text))]
    assert programs['prefix']['prefix_function'](text) == expected
    border_lengths = [length for length in range(len(text) - 1, 0, -1)
                      if text[:length] == text[-length:]]
    assert programs['structure']['border_lengths'](text) == border_lengths
    if text:
        period = next(shift for shift in range(1, len(text) + 1)
                      if all(text[index] == text[index + shift] for index in range(len(text) - shift)))
        tiles = any(text == text[:width] * (len(text) // width)
                    for width in range(1, len(text)) if len(text) % width == 0)
        assert programs['structure']['repetition'](text) == (period, tiles)
    else:
        assert programs['structure']['repetition'](text) == (None, False)
    actual = programs['palindrome']['shortest_prepend_palindrome'](text)
    expected_palindrome = next(text[len(text) - added:][::-1] + text
                              for added in range(len(text) + 1)
                              if (text[len(text) - added:][::-1] + text) == (text[len(text) - added:][::-1] + text)[::-1])
    assert actual == expected_palindrome
    counts['prefix_cases'] += 1
    counts['structure_cases'] += 1
    counts['palindrome_cases'] += 1

for text in words('ab', 5):
    for pattern in words('ab', 3):
        for cut_mask in range(2 ** max(0, len(text) - 1)):
            chunks = ['']
            for position, symbol in enumerate(text):
                chunks[-1] += symbol
                if cut_mask & (1 << position):
                    chunks.append('')
            chunks.insert(1, '')
            matcher = programs['stream']['StreamMatcher'](pattern)
            actual, consumed = [], ''
            for chunk in chunks:
                consumed += chunk
                actual.extend(matcher.feed(chunk))
                assert actual == matches(consumed, pattern)
            assert actual == matches(text, pattern)
            counts['stream_partitions'] += 1

for text in words('ACGT', 5):
    for width in range(1, 7):
        seen, expected = set(), []
        for start in range(len(text) - width + 1):
            word = text[start:start + width]
            if word in seen and word not in expected:
                expected.append(word)
            seen.add(word)
        assert programs['dna']['repeated_dna'](text, width) == expected
        counts['dna_cases'] += 1
for text, width in [('ACN', 2), ('ACGT', 0), ('A', -1), ('A', True), ('A', 1.5)]:
    try:
        programs['dna']['repeated_dna'](text, width)
        raise AssertionError('invalid DNA accepted')
    except ValueError:
        pass

for text in words('ab', 6) + ['🙂a🙂é']:
    for base, modulus in [(3, 7), (31, 1_000_000_007)]:
        index = programs['substring']['PrefixFingerprint'](text, base, modulus)
        for left in range(len(text) + 1):
            for right in range(left, len(text) + 1):
                part = text[left:right]
                direct = sum(ord(symbol) * base ** (len(part) - position - 1)
                             for position, symbol in enumerate(part)) % modulus
                assert index.query(left, right) == (len(part), direct)
                counts['substring_queries'] += 1
        for left, right in [(-1, 0), (1, 0), (0, len(text) + 1)]:
            try:
                index.query(left, right)
                raise AssertionError('invalid interval accepted')
            except ValueError:
                pass

# Independent enumeration of the stated random-base experiment, not the UI's tiny-Q example.
prime = 17
for width in range(1, 4):
    strings = list(product(range(3), repeat=width))
    for first in strings:
        for second in strings:
            if first == second:
                continue
            collisions = 0
            for base in range(2, prime):
                difference = sum((left - right) * base ** (width - index - 1)
                                 for index, (left, right) in enumerate(zip(first, second)))
                collisions += difference % prime == 0
            assert collisions <= width - 1
            counts['polynomial_pairs'] += 1

assert programs['prefix']['prefix_function']('aabaaab') == [0, 1, 0, 1, 2, 2, 3]
assert programs['prefix']['prefix_function']('aabaaac') == [0, 1, 0, 1, 2, 2, 0]
assert programs['palindrome']['shortest_prepend_palindrome']('cabca') == 'acbacabca'
raw = 'A🙂e\u0301'
assert raw[2:4].encode('utf-8') == raw.encode('utf-8')[5:8]
assert len(raw) == 4 and len(raw.encode('utf-16-le')) // 2 == 5 and len(raw.encode('utf-8')) == 8
assert unicodedata.normalize('NFC', raw).find('é') == 2
assert programs['z']['z_function']('aabcaabxaaaz') == [0, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0]
assert programs['rolling']['fingerprint']('xab', 3, 101) == 55
assert programs['rolling']['fingerprint']('x', 3, 101) == 19
assert (19 * 3 ** 2) % 101 == 70
assert (55 - 70) % 101 == programs['rolling']['fingerprint']('ab', 3, 101) == 86
print(json.dumps(counts))
