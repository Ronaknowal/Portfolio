"""Run beside string_matching_mechanisms.py. Standard library only."""
from unicodedata import normalize
from string_matching_mechanisms import find_all, StreamMatcher


def builtin_occurrences(text, pattern):
    """All overlapping code-point offsets; empty pattern matches every boundary."""
    if not pattern:
        return list(range(len(text) + 1))
    found, start = [], 0
    while True:
        position = text.find(pattern, start)
        if position < 0:
            return found
        found.append(position)
        start = position + 1  # Advancing by len(pattern) would omit overlaps.


def main():
    for text, pattern in [('AAAAA', 'AAA'), ('mississippi', 'issi'),
                          ('a🙂a🙂a', 'a🙂a'), ('abc', ''), ('', 'x')]:
        actual = builtin_occurrences(text, pattern)
        assert actual == find_all(text, pattern)
        print(ascii(text), ascii(pattern), actual)
    raw = 'e\u0301-é'
    normalized = normalize('NFC', raw)
    print('raw / NFC lengths:', len(raw), len(normalized))
    print('NFC offsets:', builtin_occurrences(normalized, normalize('NFC', 'e\u0301')))
    chunks = ['AB', '', 'AB', 'ABA']
    matcher = StreamMatcher('ABA')
    found = []
    for chunk in chunks:
        found.extend(matcher.feed(chunk))
    assert found == builtin_occurrences(''.join(chunks), 'ABA')
    print('streamed offsets:', found)


if __name__ == '__main__':
    main()
