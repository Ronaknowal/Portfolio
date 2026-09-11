"""Build complete topic-owned examples from actual Python stdout."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import black

ROOT = Path(__file__).resolve().parents[1]
NODES = '''class Node:
    def __init__(self, label, value):
        self.label = label
        self.value = value
        self.next = None

def make_chain(values, entry=-1):
    if type(entry) is not int or not -1 <= entry < len(values):
        raise ValueError("entry must be -1 or a valid node index")
    nodes = [Node(f"n{i}", value) for i, value in enumerate(values)]
    for left, right in zip(nodes, nodes[1:]):
        left.next = right
    if nodes and entry >= 0:
        nodes[-1].next = nodes[entry]
    return (nodes[0] if nodes else None), nodes

def cycle_entry(head):
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None

'''
cycle = NODES + '''def entry_by_seen(head):
    seen = set()
    current = head
    while current is not None:
        if current in seen:
            return current
        seen.add(current)
        current = current.next
    return None

def cycle_size(entry):
    # The argument is an entry returned by cycle_entry, or None.
    if entry is None:
        return 0
    length = 1
    current = entry.next
    while current is not entry:
        length += 1
        current = current.next
    return length

for size, join in [(7, 2), (7, 4), (1, 0), (1, -1), (0, -1)]:
    head, nodes = make_chain([7] * size, join)
    old_links = tuple(node.next for node in nodes)
    entry = cycle_entry(head)
    print("nodes / tail target:", size, join)
    print("entry:", entry.label if entry else None, "cycle size:", cycle_size(entry))
    print("same object as visited method:", entry is entry_by_seen(head))
    print("links unchanged:", all(node.next is old for node, old in zip(nodes, old_links)))
'''
middle = NODES + '''def middle(head, first=False):
    # Precondition for this small internal helper: an acyclic chain.
    if head is None:
        return None
    slow = fast = head
    if first:
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next
    else:
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
    return slow

def split_left_heavy(head):
    if cycle_entry(head) is not None:
        raise ValueError("split requires an acyclic chain; links were not changed")
    boundary = middle(head, first=True)
    if boundary is None:
        return None, None
    right = boundary.next
    boundary.next = None
    return head, right

def labels(head):
    result = []
    while head is not None:
        result.append(head.label)
        head = head.next
    return result

for size in [0, 1, 4, 5, 6]:
    head, nodes = make_chain([7] * size)
    first = middle(head, first=True)
    second = middle(head)
    left, right = split_left_heavy(head)
    print("size:", size, "first / second:", first.label if first else None, second.label if second else None)
    print("returned chains:", labels(left), labels(right))
head, nodes = make_chain([7, 7, 7], 1)
before = tuple(node.next for node in nodes)
try:
    split_left_heavy(head)
except ValueError as error:
    print(error)
print("rejected split preserved links:", all(node.next is old for node, old in zip(nodes, before)))
'''
greater = '''def next_distances(values, inclusive=False):
    if type(inclusive) is not bool or any(type(value) is not int for value in values):
        raise ValueError("use integer readings and a boolean comparison policy")
    answer = [0] * len(values)
    stack = []
    pushes = pops = 0
    for current, value in enumerate(values):
        while stack and (value >= values[stack[-1]] if inclusive else value > values[stack[-1]]):
            previous = stack.pop()
            pops += 1
            answer[previous] = current - previous
        stack.append(current)
        pushes += 1
    return answer, pushes, pops

for readings in [[6, 6, 4, 7, 5, 8], [5, 5, 6], [9, 8, 7, 6, 10], []]:
    answer, pushes, pops = next_distances(readings)
    print("readings:", readings)
    print("strict distances:", answer, "pushes / pops:", pushes, pops)
print("greater-or-equal distances:", next_distances([5, 5, 6], inclusive=True)[0])
'''
histogram = '''def smaller_boundaries(heights):
    if any(type(height) is not int or height < 0 for height in heights):
        raise ValueError("use nonnegative integer heights and unit-width bars")
    n = len(heights)
    left = [-1] * n
    right = [n] * n
    stack = []
    for current in range(n):
        while stack and heights[stack[-1]] >= heights[current]:
            stack.pop()
        if stack:
            left[current] = stack[-1]
        stack.append(current)
    stack = []
    for current in range(n - 1, -1, -1):
        while stack and heights[stack[-1]] >= heights[current]:
            stack.pop()
        if stack:
            right[current] = stack[-1]
        stack.append(current)
    return left, right

def largest_rectangle(heights):
    left, right = smaller_boundaries(heights)
    best_area = 0
    witness = None
    for index, height in enumerate(heights):
        width = right[index] - left[index] - 1
        area = height * width
        if area > best_area:
            best_area = area
            witness = (left[index] + 1, right[index], height)
    # Witness is (start, exclusive end, height); zero optimum has no positive witness.
    return best_area, witness

for heights in [[2, 2, 1, 4, 4, 3], [3, 1, 3, 3, 2], [2, 2], [0, 0], []]:
    left, right = smaller_boundaries(heights)
    area, witness = largest_rectangle(heights)
    print("heights:", heights)
    print("left / right:", left, right)
    print("area / witness:", area, witness)
'''

collections = {
    "linked-traversal-examples.js": ("linkedTraversalExamples", {
        "cycle": ("cycle_entry.py", "Seven equal-valued nodes end at n2. Which object is the entry, and what survives when the tail instead points to n4?", cycle),
        "middle": ("middle_split.py", "For lengths four and five, distinguish both middle queries from the left-heavy split. Which one link is changed?", middle),
    }),
    "monotonic-stack-examples.js": ("monotonicStackExamples", {
        "greater": ("future_distances.py", "Which earlier indices does value 7 settle in [6,6,4,7,5,8]? Do equal values answer a strictly-greater query?", greater),
        "histogram": ("histogram_boundaries.py", "Why do both equal height-4 bars see the same width, yet height 3 supplies the larger rectangle?", histogram),
    }),
}
for filename, (export, programs) in collections.items():
    examples = {}
    for key, (program_name, question, code) in programs.items():
        formatted = black.format_str(code, mode=black.Mode())
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(formatted))
        result = subprocess.run([sys.executable, "-c", formatted], text=True, encoding="utf-8", capture_output=True, check=True)
        assert not result.stderr
        examples[key] = {"filename": program_name, "question": question, "code": formatted, "output": result.stdout.rstrip()}
    target = ROOT / "src/learn/data" / filename
    target.write_text(f"// Complete executed Python programs. Generated by scripts/generate-linked-traversal-extension.py.\nexport const {export} = " + json.dumps(examples, indent=2, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(filename, list(examples))
