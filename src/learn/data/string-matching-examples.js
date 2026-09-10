export const stringMatchingExamples = {
  baseline: {
    title: "A correct baseline and an explicit work count",
    code: String.raw`def naive_all(text, pattern):
    starts = []
    comparisons = 0
    for start in range(len(text) - len(pattern) + 1):
        for offset in range(len(pattern)):
            comparisons += 1
            if text[start + offset] != pattern[offset]:
                break
        else:
            starts.append(start)
    return starts, comparisons


print(naive_all("ababa", "aba"))
print(naive_all("aaaaaaaa", "aaab"))
print(naive_all("ab", ""))
print(naive_all("ab", "abcd"))`,
    expected: String.raw`([0, 2], 7)
([], 20)
([0, 1, 2], 0)
([], 0)`
  },
  prefix: {
    title: "Build prefix lengths, not text positions",
    code: String.raw`def prefix_function(sequence):
    prefix = [0] * len(sequence)
    for position in range(1, len(sequence)):
        matched = prefix[position - 1]
        while matched > 0 and sequence[position] != sequence[matched]:
            matched = prefix[matched - 1]
        if sequence[position] == sequence[matched]:
            matched += 1
        prefix[position] = matched
    return prefix


for pattern in ["ababaca", "aaaa", "abc", ""]:
    print(repr(pattern), prefix_function(pattern))`,
    expected: String.raw`'ababaca' [0, 0, 1, 2, 3, 0, 1]
'aaaa' [0, 1, 2, 3]
'abc' [0, 0, 0]
'' []`
  },
  kmp: {
    title: "Find every exact occurrence, including overlaps",
    code: String.raw`def prefix_function(sequence):
    prefix = [0] * len(sequence)
    for position in range(1, len(sequence)):
        matched = prefix[position - 1]
        while matched > 0 and sequence[position] != sequence[matched]:
            matched = prefix[matched - 1]
        if sequence[position] == sequence[matched]:
            matched += 1
        prefix[position] = matched
    return prefix


def find_all(text, pattern):
    if not pattern:
        return list(range(len(text) + 1))
    prefix = prefix_function(pattern)
    matched = 0
    starts = []
    for position, symbol in enumerate(text):
        while matched > 0 and symbol != pattern[matched]:
            matched = prefix[matched - 1]
        if symbol == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            starts.append(position - len(pattern) + 1)
            matched = prefix[matched - 1]
    return starts


for text, pattern in [("abababacaba", "ababaca"),
                      ("aaaaa", "aaa"), ("ab", ""),
                      ("", "a"), ("a🙂a🙂a", "🙂a")]:
    print(repr(text), repr(pattern), find_all(text, pattern))`,
    expected: String.raw`'abababacaba' 'ababaca' [2]
'aaaaa' 'aaa' [0, 1, 2]
'ab' '' [0, 1, 2]
'' 'a' []
'a🙂a🙂a' '🙂a' [1, 3]`
  },
  stream: {
    title: "Keep prefix state when a chunk ends",
    code: String.raw`def prefix_function(sequence):
    prefix = [0] * len(sequence)
    for position in range(1, len(sequence)):
        matched = prefix[position - 1]
        while matched > 0 and sequence[position] != sequence[matched]:
            matched = prefix[matched - 1]
        if sequence[position] == sequence[matched]:
            matched += 1
        prefix[position] = matched
    return prefix


class StreamMatcher:
    def __init__(self, pattern):
        self.pattern = pattern
        self.prefix = prefix_function(pattern)
        self.matched = 0
        self.offset = 0
        self.initial_boundary_emitted = False

    def feed(self, chunk):
        starts = []
        if not self.pattern and not self.initial_boundary_emitted:
            starts.append(0)
            self.initial_boundary_emitted = True
        for symbol in chunk:
            if not self.pattern:
                self.offset += 1
                starts.append(self.offset)
                continue
            while self.matched > 0 and symbol != self.pattern[self.matched]:
                self.matched = self.prefix[self.matched - 1]
            if symbol == self.pattern[self.matched]:
                self.matched += 1
            self.offset += 1
            if self.matched == len(self.pattern):
                starts.append(self.offset - len(self.pattern))
                self.matched = self.prefix[self.matched - 1]
        return starts


matcher = StreamMatcher("aba")
for chunk in ["xxa", "", "b", "aba"]:
    print(repr(chunk), matcher.feed(chunk),
          "offset", matcher.offset, "q", matcher.matched)
empty = StreamMatcher("")
print("empty feeds", empty.feed(""), empty.feed("ab"), empty.feed(""))`,
    expected: String.raw`'xxa' [] offset 3 q 1
'' [] offset 3 q 1
'b' [] offset 4 q 2
'aba' [2, 4] offset 7 q 1
empty feeds [0] [1, 2] []`
  },
  structure: {
    title: "List borders and distinguish a period from whole copies",
    code: String.raw`def prefix_function(sequence):
    prefix = [0] * len(sequence)
    for position in range(1, len(sequence)):
        matched = prefix[position - 1]
        while matched > 0 and sequence[position] != sequence[matched]:
            matched = prefix[matched - 1]
        if sequence[position] == sequence[matched]:
            matched += 1
        prefix[position] = matched
    return prefix


def border_lengths(text):
    if not text:
        return []
    prefix = prefix_function(text)
    result = []
    length = prefix[-1]
    while length > 0:
        result.append(length)
        length = prefix[length - 1]
    return result


def repetition(text):
    if not text:
        return None, False
    period = len(text) - prefix_function(text)[-1]
    tiles = period < len(text) and len(text) % period == 0
    return period, tiles


for text in ["ababab", "ababa", "aaaa", "abc", ""]:
    print(repr(text), "borders", border_lengths(text),
          "period, tiles", repetition(text))`,
    expected: String.raw`'ababab' borders [4, 2] period, tiles (2, True)
'ababa' borders [3, 1] period, tiles (2, False)
'aaaa' borders [3, 2, 1] period, tiles (1, True)
'abc' borders [] period, tiles (3, False)
'' borders [] period, tiles (None, False)`
  },
  rolling: {
    title: "Roll a fingerprint, then verify the actual symbols",
    code: String.raw`def fingerprint(text, base, modulus):
    value = 0
    for symbol in text:
        value = (value * base + ord(symbol)) % modulus
    return value


def rolling_find_all(text, pattern, base=31, modulus=1_000_000_007):
    if not 2 <= base < modulus:
        raise ValueError("Require 2 <= base < modulus")
    width = len(pattern)
    if width == 0:
        return list(range(len(text) + 1)), []
    if width > len(text):
        return [], []
    high_power = pow(base, width - 1, modulus)
    target = fingerprint(pattern, base, modulus)
    value = fingerprint(text[:width], base, modulus)
    starts, audit = [], []
    for start in range(len(text) - width + 1):
        if value == target:
            exact = True
            checks = 0
            for offset in range(width):
                checks += 1
                if text[start + offset] != pattern[offset]:
                    exact = False
                    break
            audit.append((start, exact, checks))
            if exact:
                starts.append(start)
        if start + width < len(text):
            value = ((value - ord(text[start]) * high_power) * base
                     + ord(text[start + width])) % modulus
    return starts, audit


print("small-hash collision", fingerprint("ad", 3, 7),
      fingerprint("ba", 3, 7))
print("starts, candidate checks", rolling_find_all("adbaad", "ba", 3, 7))
print("many true hits", rolling_find_all("aaaaa", "aaa", 3, 7))`,
    expected: String.raw`small-hash collision 6 6
starts, candidate checks ([2], [(0, False, 1), (1, False, 1), (2, True, 2), (4, False, 1)])
many true hits ([0, 1, 2], [(0, True, 3), (1, True, 3), (2, True, 3)])`
  },
  substring: {
    title: "Cancel a prefix when comparing substring fingerprints",
    code: String.raw`class PrefixFingerprint:
    def __init__(self, text, base=31, modulus=1_000_000_007):
        if not 2 <= base < modulus:
            raise ValueError("Require 2 <= base < modulus")
        self.modulus = modulus
        self.prefix = [0]
        self.powers = [1]
        for symbol in text:
            self.prefix.append((self.prefix[-1] * base + ord(symbol)) % modulus)
            self.powers.append(self.powers[-1] * base % modulus)

    def query(self, left, right):
        if not 0 <= left <= right < len(self.prefix):
            raise ValueError("Require 0 <= left <= right <= text length")
        value = (self.prefix[right]
                 - self.prefix[left] * self.powers[right - left]) % self.modulus
        return right - left, value


text = "abxxab"
index = PrefixFingerprint(text)
print(index.query(0, 2), index.query(4, 6))
print("exact", text[0:2] == text[4:6])
collision = PrefixFingerprint("adba", 3, 7)
print("same signature", collision.query(0, 2) == collision.query(2, 4))
print("same contents", "ad" == "ba")`,
    expected: String.raw`(2, 3105) (2, 3105)
exact True
same signature True
same contents False`
  },
  unicode: {
    title: "Name the offset space before normalizing text",
    code: String.raw`import unicodedata

text = "A🙂e\u0301"
print("code points", len(text))
print("UTF-16 units", len(text.encode("utf-16-le")) // 2)
print("UTF-8 bytes", len(text.encode("utf-8")))
print("code-point names", [f"U+{ord(symbol):04X}" for symbol in text])
composed = unicodedata.normalize("NFC", text)
print("normalized code points", len(composed))
print("raw / NFC match", text.find("é"), composed.find("é"))
print("casefold expands", len("Straße"), len("Straße".casefold()))
print("normalize across seam",
      unicodedata.normalize("NFC", "e" + "\u0301") == "é",
      unicodedata.normalize("NFC", "e") + unicodedata.normalize("NFC", "\u0301") == "é")`,
    expected: String.raw`code points 4
UTF-16 units 5
UTF-8 bytes 8
code-point names ['U+0041', 'U+1F642', 'U+0065', 'U+0301']
normalized code points 3
raw / NFC match -1 2
casefold expands 6 7
normalize across seam True False`
  },
  dna: {
    title: "Use a collision-free code for fixed-length DNA words",
    code: String.raw`def repeated_dna(text, width=10):
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("Require a positive integer width")
    digits = {"A": 0, "C": 1, "G": 2, "T": 3}
    if any(symbol not in digits for symbol in text):
        raise ValueError("This exact alphabet is A, C, G, T")
    if width > len(text):
        return []
    suffix_modulus = 4 ** (width - 1)
    code = 0
    seen, emitted, result = set(), set(), []
    for position, symbol in enumerate(text):
        code = (code % suffix_modulus) * 4 + digits[symbol]
        if position + 1 < width:
            continue
        if code in seen and code not in emitted:
            result.append(text[position - width + 1:position + 1])
            emitted.add(code)
        seen.add(code)
    return result


print(repeated_dna("ACGACGTTACG", 3))
print(repeated_dna("AAAAA", 3))
print(repeated_dna("ACGT", 10))`,
    expected: String.raw`['ACG']
['AAA']
[]`
  },
  palindrome: {
    title: "Turn a palindromic-prefix question into a border",
    code: String.raw`def prefix_function(sequence):
    prefix = [0] * len(sequence)
    for position in range(1, len(sequence)):
        matched = prefix[position - 1]
        while matched > 0 and sequence[position] != sequence[matched]:
            matched = prefix[matched - 1]
        if sequence[position] == sequence[matched]:
            matched += 1
        prefix[position] = matched
    return prefix


def shortest_prepend_palindrome(text):
    if not text:
        return ""
    separator = object()  # Cannot equal any string symbol.
    combined = list(text) + [separator] + list(reversed(text))
    retained = prefix_function(combined)[-1]
    return text[retained:][::-1] + text


for text in ["abac", "abcd", "aacecaaa", "", "a#a"]:
    print(repr(text), repr(shortest_prepend_palindrome(text)))`,
    expected: String.raw`'abac' 'cabac'
'abcd' 'dcbabcd'
'aacecaaa' 'aaacecaaa'
'' ''
'a#a' 'a#a'`
  },
  z: {
    title: "Search with the Z array and a half-open evidence box",
    code: String.raw`def z_function(sequence):
    values = [0] * len(sequence)  # Convention: Z[0] = 0.
    left = right = 0
    for position in range(1, len(sequence)):
        if position < right:
            values[position] = min(right - position, values[position - left])
        while (position + values[position] < len(sequence)
               and sequence[values[position]] == sequence[position + values[position]]):
            values[position] += 1
        if position + values[position] > right:
            left = position
            right = position + values[position]
    return values


def z_find_all(text, pattern):
    if not pattern:
        return list(range(len(text) + 1))
    separator = object()
    combined = list(pattern) + [separator] + list(text)
    values = z_function(combined)
    offset = len(pattern) + 1
    return [position - offset for position in range(offset, len(combined))
            if values[position] == len(pattern)]


print(z_function("aabcaabxaaaz"))
print(z_find_all("ababa", "aba"))
print(z_find_all("###", "##"))
print(z_find_all("ab", ""))`,
    expected: String.raw`[0, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0]
[0, 2]
[0, 1]
[0, 1, 2]`
  }
};
