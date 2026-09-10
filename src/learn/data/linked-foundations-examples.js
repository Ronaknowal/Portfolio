const nodes=`class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

def build(values):
    head = None
    for value in reversed(values):
        head = Node(value, head)
    return head

def values(head):
    result = []
    seen = set()
    while head is not None:
        if id(head) in seen:
            raise ValueError("cycle detected")
        seen.add(id(head))
        result.append(head.value)
        head = head.next
    return result

`;
export const linkedExamples={
  reverse:{code:nodes+`def reverse(head):
    previous = None
    current = head
    while current is not None:
        following = current.next
        current.next = previous
        previous = current
        current = following
    return previous

head = build([4, 7, 9])
old_head = head
head = reverse(head)
print("reversed:", values(head))
print("old head is now last:", old_head.next is None)
print("empty:", values(reverse(None)))
print("singleton:", values(reverse(build([4]))))`,output:`reversed: [9, 7, 4]
old head is now last: True
empty: []
singleton: [4]`},
  remove:{code:nodes+`def remove_first(head, target):
    sentinel = Node(None, head)
    previous = sentinel
    current = head
    while current is not None:
        if current.value == target:
            previous.next = current.next
            return sentinel.next
        previous = current
        current = current.next
    return sentinel.next

head = build([4, 7, 7, 9])
head = remove_first(head, 7)
print("first 7 removed:", values(head))
head = remove_first(head, 4)
print("head removed:", values(head))
head = remove_first(head, 99)
print("absent target:", values(head))
print("empty:", values(remove_first(None, 7)))`,output:`first 7 removed: [4, 7, 9]
head removed: [7, 9]
absent target: [7, 9]
empty: []`},
  brackets:{code:`def balanced(text):
    stack = []
    opening = "([{"
    partner = {")": "(", "]": "[", "}": "{"}
    for char in text:
        if char in opening:
            stack.append(char)
        elif char in partner:
            if not stack or stack.pop() != partner[char]:
                return False
    return not stack

for text in ["([])", "([)]", ")", "(()", "", "[x]"]:
    print(repr(text), balanced(text))`,output:`'([])' True
'([)]' False
')' False
'(()' False
'' True
'[x]' True`},
  ring:{code:`class RingQueue:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.data = [None] * capacity
        self.head = 0
        self.size = 0

    def put(self, value):
        if self.size == len(self.data):
            raise OverflowError("queue full")
        tail = (self.head + self.size) % len(self.data)
        self.data[tail] = value
        self.size += 1

    def get(self):
        if self.size == 0:
            raise IndexError("queue empty")
        value = self.data[self.head]
        self.data[self.head] = None
        self.head = (self.head + 1) % len(self.data)
        self.size -= 1
        return value

queue = RingQueue(3)
for job in ["A", "B", "C"]:
    queue.put(job)
print("served:", queue.get())
queue.put("D")  # Reuses physical slot 0.
print("physical:", queue.data, "head:", queue.head)
try:
    queue.put("E")
except OverflowError as error:
    print(error)
print("remaining FIFO:", [queue.get(), queue.get(), queue.get()])
try:
    queue.get()
except IndexError as error:
    print(error)`,output:`served: A
physical: ['D', 'B', 'C'] head: 1
queue full
remaining FIFO: ['B', 'C', 'D']
queue empty`},
  lru:{code:`from collections import OrderedDict

cache = OrderedDict()
capacity = 2
source = {"a": 10, "b": 20, "c": 30}

def read(key):
    if key in cache:
        cache.move_to_end(key)
        print("hit", key)
        return cache[key]
    value = source[key]
    cache[key] = value
    evicted = None
    if len(cache) > capacity:
        evicted, _ = cache.popitem(last=False)
    print("miss", key, "evicted", evicted)
    return value

for key in ["a", "b", "a", "c"]:
    read(key)
print("least -> most recent:", list(cache))`,output:`miss a evicted None
miss b evicted None
hit a
miss c evicted b
least -> most recent: ['a', 'c']`},
  window:{code:`from collections import deque

def recent_means(readings, width):
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive integer")
    window = deque()
    total = 0
    result = []
    for value in readings:
        if len(window) == width:
            total -= window.popleft()
        window.append(value)
        total += value
        result.append(total / len(window))
    return result

print("last three:", [round(x, 3) for x in recent_means([2, 4, 8, 10], 3)])
print("last two:", [round(x, 3) for x in recent_means([2, 4, 8, 10], 2)])
print("empty:", recent_means([], 3))`,output:`last three: [2.0, 3.0, 4.667, 7.333]
last two: [2.0, 3.0, 6.0, 9.0]
empty: []`},
};
