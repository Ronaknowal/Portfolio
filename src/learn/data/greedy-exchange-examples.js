export const greedyExchangeExamples = {
  coins: {
    title: 'Largest-first coins can be feasible but wasteful, or get stuck',
    code: `def largest_first(denominations, amount):
    if type(amount) is not int or amount < 0:
        raise ValueError("amount must be a nonnegative integer")
    if any(type(coin) is not int or coin <= 0 for coin in denominations):
        raise ValueError("coin values must be positive integers")
    chosen = []
    for coin in sorted(set(denominations), reverse=True):
        count, amount = divmod(amount, coin)
        if count:
            chosen.append((coin, count))
    return chosen, amount

print(largest_first([1, 3, 4], 6))
print("better: two coins of value 3")
print(largest_first([3, 4], 6))
print("still feasible: 3 + 3 = 6")
print(largest_first([1, 3, 4], 0))`,
    expected: '([(4, 1), (1, 2)], 0)\nbetter: two coins of value 3\n([(4, 1)], 2)\nstill feasible: 3 + 3 = 6\n([], 0)',
  },
  intervals: {
    title: 'Select the maximum number of compatible half-open requests',
    code: `def select_appointments(requests):
    # Each record is (unique_id, start, finish); touching endpoints fit.
    if len({row[0] for row in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    if any(type(start) is not int or type(finish) is not int or start >= finish
           for _, start, finish in requests):
        raise ValueError("use integer start < finish")
    selected = []
    last_finish = None
    for identity, start, finish in sorted(requests, key=lambda row: (row[2], row[1], row[0])):
        if last_finish is None or start >= last_finish:
            selected.append(identity)
            last_finish = finish
    return selected

requests = [("A", 0, 6), ("B", 1, 3), ("C", 3, 5),
            ("D", 5, 7), ("E", 6, 9), ("F", 7, 9)]
print(select_appointments(requests))
print("removals:", len(requests) - len(select_appointments(requests)))
print(select_appointments([("X", -3, -1), ("Y", -1, 1)]))
print(select_appointments([]))`,
    expected: "['B', 'C', 'D', 'F']\nremovals: 2\n['X', 'Y']\n[]",
  },
  rooms: {
    title: 'Schedule every request while reusing the earliest available room',
    code: `from heapq import heappop, heappush

def minimum_rooms(requests):
    if len({row[0] for row in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    if any(type(start) is not int or type(finish) is not int or start >= finish
           for _, start, finish in requests):
        raise ValueError("use integer start < finish")
    rooms = []
    available = []  # one (last_finish, room_id) per allocated room
    for identity, start, finish in sorted(requests, key=lambda row: (row[1], row[2], row[0])):
        if available and available[0][0] <= start:
            _, room = heappop(available)
        else:
            room = len(rooms)
            rooms.append([])
        rooms[room].append(identity)
        heappush(available, (finish, room))
    return rooms

requests = [("A", 0, 6), ("B", 1, 3), ("C", 3, 5),
            ("D", 5, 7), ("E", 6, 9), ("F", 7, 9)]
print(minimum_rooms(requests))
print(minimum_rooms([("X", 1, 2), ("Y", 2, 3)]))
print(minimum_rooms([]))`,
    expected: "[['A', 'E'], ['B', 'C', 'D', 'F']]\n[['X', 'Y']]\n[]",
  },
  deadlines: {
    title: 'Earliest deadlines minimize the maximum signed lateness',
    code: `def deadline_schedule(jobs):
    # Every (id, processing_time, deadline) is available at time zero.
    if len({row[0] for row in jobs}) != len(jobs):
        raise ValueError("job IDs must be unique")
    if any(type(processing) is not int or processing <= 0 or type(deadline) is not int
           for _, processing, deadline in jobs):
        raise ValueError("use positive integer durations and integer deadlines")
    time = 0
    schedule = []
    for identity, processing, deadline in sorted(jobs, key=lambda row: (row[2], row[0])):
        start = time
        time += processing
        schedule.append((identity, start, time, time - deadline))
    maximum_lateness = max((row[3] for row in schedule), default=None)
    return schedule, maximum_lateness

jobs = [("A", 4, 9), ("B", 3, 5), ("C", 2, 7)]
schedule, maximum = deadline_schedule(jobs)
print(schedule)
print("maximum lateness:", maximum)
print("maximum tardiness:", max(0, maximum))
print(deadline_schedule([("early", 2, 10)]))
print(deadline_schedule([]))`,
    expected: "[('B', 0, 3, -2), ('C', 3, 5, -2), ('A', 5, 9, 0)]\nmaximum lateness: 0\nmaximum tardiness: 0\n([('early', 0, 2, -8)], -8)\n([], None)",
  },
  fractional: {
    title: 'Fill capacity by exact value density, then change divisibility',
    code: `from fractions import Fraction

def density_allocation(items, capacity, divisible=True):
    # Each record is (unique_id, positive_weight, nonnegative_value).
    if type(capacity) is not int or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")
    if len({row[0] for row in items}) != len(items):
        raise ValueError("item IDs must be unique")
    if any(type(weight) is not int or weight <= 0 or type(value) is not int or value < 0
           for _, weight, value in items):
        raise ValueError("use positive weights and nonnegative integer values")
    remaining = capacity
    selected = []
    total = Fraction(0)
    ordered = sorted(items, key=lambda row: (-Fraction(row[2], row[1]), row[0]))
    for identity, weight, value in ordered:
        amount = min(remaining, weight) if divisible else (weight if weight <= remaining else 0)
        if amount:
            fraction = Fraction(amount, weight)
            selected.append((identity, fraction))
            total += value * fraction
            remaining -= amount
    return selected, total, remaining

items = [("A", 10, 60), ("B", 20, 100), ("C", 30, 120)]
for divisible in [True, False]:
    selected, total, unused = density_allocation(items, 50, divisible)
    print("fractional" if divisible else "whole", [(identity, str(part)) for identity, part in selected],
          "value", total, "unused", unused)
print("better whole selection B+C: weight", 20 + 30, "value", 100 + 120)
print(density_allocation(items, 0)[1])`,
    expected: "fractional [('A', '1'), ('B', '1'), ('C', '2/3')] value 240 unused 0\nwhole [('A', '1'), ('B', '1')] value 160 unused 20\nbetter whole selection B+C: weight 50 value 220\n0",
  },
  huffman: {
    title: 'Build a Huffman code and decode a complete message',
    code: `from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import count

@dataclass
class Node:
    frequency: int
    symbol: str | None = None
    left: "Node | None" = None
    right: "Node | None" = None

def huffman(frequencies):
    if any(not isinstance(symbol, str) or len(symbol) != 1 for symbol in frequencies):
        raise ValueError("this text example uses one-character symbols")
    if any(type(frequency) is not int or frequency <= 0 for frequency in frequencies.values()):
        raise ValueError("frequencies must be positive integers")
    serial = count()
    heap = [(frequency, next(serial), Node(frequency, symbol))
            for symbol, frequency in sorted(frequencies.items())]
    heapify(heap)
    merges = []
    while len(heap) > 1:
        left_weight, _, left = heappop(heap)
        right_weight, _, right = heappop(heap)
        parent = Node(left_weight + right_weight, left=left, right=right)
        merges.append(parent.frequency)
        heappush(heap, (parent.frequency, next(serial), parent))
    root = heap[0][2] if heap else None
    codes = {}
    def visit(node, path):
        if node.symbol is not None:
            codes[node.symbol] = path
        else:
            visit(node.left, path + "0")
            visit(node.right, path + "1")
    if root is not None:
        visit(root, "")
    return root, codes, merges

def decode(bits, root, symbol_count):
    if type(symbol_count) is not int or symbol_count < 0 or any(bit not in "01" for bit in bits):
        raise ValueError("use binary text and a nonnegative symbol count")
    if root is None:
        if bits or symbol_count:
            raise ValueError("empty alphabet cannot decode a message")
        return ""
    if root.symbol is not None:
        if bits:
            raise ValueError("single-symbol empty code has no payload bits")
        return root.symbol * symbol_count
    output = []
    node = root
    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            output.append(node.symbol)
            node = root
    if node is not root or len(output) != symbol_count:
        raise ValueError("incomplete codeword or wrong symbol count")
    return "".join(output)

frequencies = {"A": 2, "B": 3, "C": 7, "D": 9}
root, codes, merges = huffman(frequencies)
print("codes:", sorted(codes.items()))
print("merge costs:", merges, "sum:", sum(merges))
print("weighted bits:", sum(frequencies[symbol] * len(code) for symbol, code in codes.items()))
message = "DAC"
bits = "".join(codes[symbol] for symbol in message)
print(message, bits, decode(bits, root, len(message)))
single, single_codes, _ = huffman({"A": 4})
print("one symbol:", repr(single_codes["A"]), decode("", single, 4))`,
    expected: "codes: [('A', '100'), ('B', '101'), ('C', '11'), ('D', '0')]\nmerge costs: [5, 12, 21] sum: 38\nweighted bits: 38\nDAC 010011 DAC\none symbol: '' AAAA",
  },
  reachability: {
    title: 'A reachable prefix summarizes all possible forward paths',
    code: `def can_reach_last(jump_limits):
    if any(type(value) is not int or value < 0 for value in jump_limits):
        raise ValueError("jump limits must be nonnegative integers")
    if not jump_limits:
        return False  # this API has no start or destination for empty input
    reach = 0
    for index, maximum_jump in enumerate(jump_limits):
        if index > reach:
            return False
        reach = max(reach, index + maximum_jump)
        if reach >= len(jump_limits) - 1:
            return True
    return False

for values in [[2, 3, 0, 0, 1, 0], [3, 2, 1, 0, 4], [0], []]:
    print(values, can_reach_last(values))`,
    expected: '[2, 3, 0, 0, 1, 0] True\n[3, 2, 1, 0, 4] False\n[0] True\n[] False',
  },
  stabbing: {
    title: 'Exercise solution: cover every closed interval with fewest points',
    code: `def covering_points(intervals):
    if any(type(start) is not int or type(finish) is not int or start > finish
           for start, finish in intervals):
        raise ValueError("use integer closed intervals with start <= finish")
    points = []
    for start, finish in sorted(intervals, key=lambda row: (row[1], row[0])):
        if not points or start > points[-1]:
            points.append(finish)
    return points

print(covering_points([(1, 4), (3, 6), (6, 8), (7, 9)]))
print(covering_points([(1, 2), (2, 3)]))
print(covering_points([(5, 5)]))
print(covering_points([]))`,
    expected: '[4, 8]\n[2]\n[5]\n[]',
  },
  matching: {
    title: 'Exercise solution: use the smallest adequate supply',
    code: `def match_thresholds(requirements, supplies):
    if any(type(value) is not int or value < 0 for value in requirements + supplies):
        raise ValueError("use nonnegative integer requirements and supplies")
    people = sorted(enumerate(requirements), key=lambda row: (row[1], row[0]))
    available = sorted(enumerate(supplies), key=lambda row: (row[1], row[0]))
    selected = []
    person = 0
    for supply_id, amount in available:
        if person == len(people):
            break
        person_id, need = people[person]
        if amount >= need:
            selected.append((person_id, supply_id))
            person += 1
    return selected

requirements = [4, 2, 5]
supplies = [3, 5]
print(match_thresholds(requirements, supplies))
print(match_thresholds([1, 1], [1]))
print(match_thresholds([], [2]))`,
    expected: '[(1, 0), (0, 1)]\n[(0, 0)]\n[]',
  },
};
