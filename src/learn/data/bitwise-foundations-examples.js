// Complete standalone Python 3.12+ programs; exact stdout is verified natively.
export const bitwiseExamples = {
  bitsets: {
    filename: "bitmap_search.py",
    code: `def full_mask(width):
    if type(width) is not int or not 1 <= width <= 1024:
        raise ValueError("width must be an integer from 1 to 1024")
    return (1 << width) - 1


def pack(members, width):
    full_mask(width)
    mask = 0
    for member in members:
        if type(member) is not int or not 0 <= member < width:
            raise ValueError("member outside the universe")
        mask |= 1 << member  # Adding twice still records presence once.
    return mask


def unpack(mask, width):
    full = full_mask(width)
    if type(mask) is not int or not 0 <= mask <= full:
        raise ValueError("mask outside the universe")
    return [i for i in range(width) if mask & (1 << i)]


width = 6
a = pack([0, 2, 5, 2], width)
b = pack([1, 2, 4], width)
for name, mask in [
    ("A", a), ("B", b), ("intersection", a & b),
    ("union", a | b), ("A minus B", a & ~b),
    ("symmetric difference", a ^ b), ("complement A", (~a) & full_mask(width)),
]:
    print(name, format(mask, "06b"), mask, unpack(mask, width))
print("add bit 2 again:", a | (1 << 2), "arithmetic addition:", a + (1 << 2))
print("clear absent bit 1:", a & ~(1 << 1), "arithmetic subtraction:", a - (1 << 1))

documents = ["quiet red fox", "blue owl", "red owl", "red red fox owl"]
postings = {}
ordinary = {}
for document_id, text in enumerate(documents):
    for word in text.split():
        postings[word] = postings.get(word, 0) | (1 << document_id)
        ordinary.setdefault(word, set()).add(document_id)
for terms in [("red", "owl"), ("fox", "owl"), ("red", "missing")]:
    mask = postings.get(terms[0], 0) & postings.get(terms[1], 0)
    result = unpack(mask, len(documents))
    expected = sorted(ordinary.get(terms[0], set()) & ordinary.get(terms[1], set()))
    assert result == expected
    print(" AND ".join(terms), "->", result, "count", mask.bit_count())`,
    output: `A 100101 37 [0, 2, 5]
B 010110 22 [1, 2, 4]
intersection 000100 4 [2]
union 110111 55 [0, 1, 2, 4, 5]
A minus B 100001 33 [0, 5]
symmetric difference 110011 51 [0, 1, 4, 5]
complement A 011010 26 [1, 3, 4]
add bit 2 again: 37 arithmetic addition: 41
clear absent bit 1: 37 arithmetic subtraction: 35
red AND owl -> [2, 3] count 2
fox AND owl -> [3] count 1
red AND missing -> [] count 0`,
  },
  words: {
    filename: "word_interpretation.py",
    code: `def checked_width(width):
    if type(width) is not int or not 1 <= width <= 1024:
        raise ValueError("width must be an integer from 1 to 1024")
    return 1 << width


def encode_signed(value, width):
    modulus = checked_width(width)
    if type(value) is not int or not -modulus // 2 <= value < modulus // 2:
        raise ValueError("signed value does not fit")
    return value % modulus


def decode_signed(word, width):
    modulus = checked_width(width)
    if type(word) is not int or not 0 <= word < modulus:
        raise ValueError("word does not fit")
    return word if word < modulus // 2 else word - modulus


for value, width, shift in [(-10, 8, 2), (-3, 4, 1), (5, 4, 2)]:
    word = encode_signed(value, width)
    full = (1 << width) - 1
    print("pattern", format(word, f"0{width}b"), "unsigned", word, "signed", decode_signed(word, width))
    print("right", shift, "logical", word >> shift, "arithmetic", value >> shift)
    print("left", shift, "bounded", (word << shift) & full, "unbounded", word << shift)
print("Python complement:", ~5, "8-bit complement:", (~5) & 255)
print("negative magnitude popcount:", (-1).bit_count(), "8-bit word popcount:", ((-1) & 255).bit_count())
try:
    encode_signed(128, 8)
except ValueError as error:
    print("rejected:", error)
try:
    3 >> -1
except ValueError:
    print("rejected: negative shift")`,
    output: `pattern 11110110 unsigned 246 signed -10
right 2 logical 61 arithmetic -3
left 2 bounded 216 unbounded 984
pattern 1101 unsigned 13 signed -3
right 1 logical 6 arithmetic -2
left 1 bounded 10 unbounded 26
pattern 0101 unsigned 5 signed 5
right 2 logical 1 arithmetic 1
left 2 bounded 4 unbounded 20
Python complement: -6 8-bit complement: 250
negative magnitude popcount: 1 8-bit word popcount: 8
rejected: signed value does not fit
rejected: negative shift`,
  },
  parity: {
    filename: "xor_parity.py",
    code: `from collections import Counter


def xor_fold(values):
    # The fold is meaningful for any nonempty integer sequence.
    # Interpreting it as a singleton requires the separate multiplicity promise.
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("use a nonempty reusable sequence")
    accumulator = 0
    for value in values:
        if type(value) is not int:
            raise ValueError("events must be integers")
        accumulator ^= value
    return accumulator


def audit_promise(values):
    # Diagnostic O(k) storage, NOT part of the constant-word fold.
    counts = Counter(values)
    return sum(count == 1 for count in counts.values()) == 1 and all(
        count in (1, 2) for count in counts.values()
    )


examples = [[12, 5, 12, 9, 5], [8, 0, 8], [-7, 4, 4],
            [6, 6, 6, 2, 2], [1, 2, 3]]
for values in examples:
    print(values, "-> XOR", xor_fold(values), "paired-plus-one?", audit_promise(values))
accumulator = 0
for event in examples[0]:
    before = accumulator
    accumulator ^= event
    print(f"{before:04b} XOR {event:04b} = {accumulator:04b}")`,
    output: `[12, 5, 12, 9, 5] -> XOR 9 paired-plus-one? True
[8, 0, 8] -> XOR 0 paired-plus-one? True
[-7, 4, 4] -> XOR -7 paired-plus-one? True
[6, 6, 6, 2, 2] -> XOR 6 paired-plus-one? False
[1, 2, 3] -> XOR 0 paired-plus-one? False
0000 XOR 1100 = 1100
1100 XOR 0101 = 1001
1001 XOR 1100 = 0101
0101 XOR 1001 = 1100
1100 XOR 0101 = 1001`,
  },
  population: {
    filename: "population_count.py",
    code: `def nonnegative_integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("use a nonnegative integer or explicitly mask a word first")


def population_count(value):
    nonnegative_integer(value)
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count


def is_power_of_two(value):
    if type(value) is not int:
        raise ValueError("value must be an integer")
    return value > 0 and (value & (value - 1)) == 0


def differing_bits(left, right):
    nonnegative_integer(left)
    nonnegative_integer(right)
    return population_count(left ^ right)


value = 180
while value:
    after = value & (value - 1)
    print(f"{value:08b} AND {value - 1:08b} = {after:08b}")
    value = after
for value in [0, 1, 128, 180, 255, 1 << 100]:
    print(value, "ones", population_count(value), "power of two?", is_power_of_two(value))
    assert population_count(value) == value.bit_count()
print("45 vs 39 differing positions:", differing_bits(45, 39))
print("zero and negative powers:", is_power_of_two(0), is_power_of_two(-8))
try:
    population_count(-1)
except ValueError as error:
    print("rejected:", error)`,
    output: `10110100 AND 10110011 = 10110000
10110000 AND 10101111 = 10100000
10100000 AND 10011111 = 10000000
10000000 AND 01111111 = 00000000
0 ones 0 power of two? False
1 ones 1 power of two? True
128 ones 1 power of two? True
180 ones 4 power of two? False
255 ones 8 power of two? False
1267650600228229401496703205376 ones 1 power of two? True
45 vs 39 differing positions: 2
zero and negative powers: False False
rejected: use a nonnegative integer or explicitly mask a word first`,
  },
  twoSingletons: {
    filename: "two_singletons.py",
    code: `def two_singletons(values):
    # Promise: exactly two distinct singletons; every other distinct value paired.
    # Two passes require a reusable sequence. This does not validate multiplicities.
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("use a reusable sequence with at least two integers")
    total = 0
    for value in values:
        if type(value) is not int:
            raise ValueError("values must be integers")
        total ^= value
    if total == 0:
        raise ValueError("zero XOR contradicts two distinct singletons")
    separating_bit = total & -total
    without_bit = 0
    with_bit = 0
    for value in values:
        if value & separating_bit:
            with_bit ^= value
        else:
            without_bit ^= value
    return without_bit, with_bit


for values in [[4, 9, 4, 12, 9, 7], [-4, 0, 9, 9], [0, 1], [-7, 2, -7, 5, 2, -3]]:
    answer = two_singletons(values)
    print(values, "->", sorted(answer))  # Sort two outputs only for display.
values = [4, 9, 4, 12, 9, 7]
total = 0
for value in values:
    total ^= value
bit = total & -total
print("total XOR", total, "separating weight", bit)
print("bit absent:", [value for value in values if not value & bit])
print("bit present:", [value for value in values if value & bit])
try:
    two_singletons([2, 2])
except ValueError as error:
    print("rejected:", error)
# A nonzero total alone does not establish the promise.
print("invalid [1, 2, 3, 4] still computes:", two_singletons([1, 2, 3, 4]))`,
    output: `[4, 9, 4, 12, 9, 7] -> [7, 12]
[-4, 0, 9, 9] -> [-4, 0]
[0, 1] -> [0, 1]
[-7, 2, -7, 5, 2, -3] -> [-3, 5]
total XOR 11 separating weight 1
bit absent: [4, 4, 12]
bit present: [9, 9, 7]
rejected: zero XOR contradicts two distinct singletons
invalid [1, 2, 3, 4] still computes: (0, 4)`,
  }
};
