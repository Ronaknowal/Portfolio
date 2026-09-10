const heapImplementation = `class MinHeap:
    def __init__(self, values=()):
        self.items = list(values)  # own the list; do not mutate caller input
        for parent in range(len(self.items) // 2 - 1, -1, -1):
            self._sift_down(parent)

    def _sift_down(self, parent):
        items = self.items
        while 2 * parent + 1 < len(items):
            child = 2 * parent + 1
            right = child + 1
            if right < len(items) and items[right] < items[child]:
                child = right
            if items[parent] <= items[child]:
                break
            items[parent], items[child] = items[child], items[parent]
            parent = child

    def push(self, value):
        items = self.items
        items.append(value)
        child = len(items) - 1
        while child > 0:
            parent = (child - 1) // 2
            if items[parent] <= items[child]:
                break
            items[parent], items[child] = items[child], items[parent]
            child = parent

    def pop(self):
        if not self.items:
            raise IndexError("empty heap")
        minimum = self.items[0]
        last = self.items.pop()
        if self.items:
            self.items[0] = last
            self._sift_down(0)
        return minimum
`;

const trieImplementation = `class TrieNode:
    def __init__(self):
        self.children = {}
        self.terminal = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for character in word:
            if character not in node.children:
                node.children[character] = TrieNode()
            node = node.children[character]
        node.terminal = True

    def walk(self, text):
        node = self.root
        for character in text:
            node = node.children.get(character)
            if node is None:
                return None
        return node

    def contains(self, word):
        node = self.walk(word)
        return node is not None and node.terminal

    def starts_with(self, prefix):
        # Empty prefix denotes the root, even in an empty trie.
        return self.walk(prefix) is not None

    def delete(self, word):
        node = self.root
        path = []
        for character in word:
            child = node.children.get(character)
            if child is None:
                return False
            path.append((node, character, child))
            node = child
        if not node.terminal:
            return False
        node.terminal = False
        for parent, character, child in reversed(path):
            if child.terminal or child.children:
                break
            del parent.children[character]
        return True

    def suggestions(self, prefix, limit=3):
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        node = self.walk(prefix)
        if node is None or limit == 0:
            return []
        answer = []
        letters = list(prefix)
        def collect(current):
            if current.terminal:
                answer.append("".join(letters))
                if len(answer) == limit:
                    return True
            for character in sorted(current.children):
                letters.append(character)
                done = collect(current.children[character])
                letters.pop()
                if done:
                    return True
            return False
        collect(node)
        return answer
`;

export const heapTrieExamples = {
  heapOperations: {
    title: 'Build, insert and remove using the same array',
    code: `${heapImplementation}
values = [7, 2, 9, 1, 5]
heap = MinHeap(values)
print("built:", heap.items)
heap.push(0)
print("pushed:", heap.items)
print("removed:", heap.pop())
print("remaining:", heap.items)
print("ordered:", [heap.pop() for _ in range(len(heap.items))])
print("original:", values)
try:
    heap.pop()
except IndexError as error:
    print(error)
`,
    expected: 'built: [1, 2, 9, 7, 5]\npushed: [0, 2, 1, 7, 5, 9]\nremoved: 0\nremaining: [1, 2, 9, 7, 5]\nordered: [1, 2, 5, 7, 9]\noriginal: [7, 2, 9, 1, 5]\nempty heap',
  },
  libraryOperations: {
    title: 'Use heapq, and distinguish two combined operations',
    code: `import heapq

first = [3, 5, 8]
second = first.copy()
print("pushpop:", heapq.heappushpop(first, 1), first)
print("replace:", heapq.heapreplace(second, 1), second)
largest_first = [-value for value in [4, 9, 1, 9]]
heapq.heapify(largest_first)
print("maximums:", [-heapq.heappop(largest_first)
                    for _ in range(len(largest_first))])
empty = []
print("empty pushpop:", heapq.heappushpop(empty, 6), empty)
`,
    expected: 'pushpop: 1 [3, 5, 8]\nreplace: 3 [1, 5, 8]\nmaximums: [9, 9, 4, 1]\nempty pushpop: 6 []',
  },
  topK: {
    title: 'Maintain the largest k occurrences of a stream',
    code: `import heapq

def top_k_states(values, k):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    retained = []
    for value in values:
        if len(retained) < k:
            heapq.heappush(retained, value)
        else:
            heapq.heappushpop(retained, value)
        kth = retained[0] if len(retained) == k else None
        # Sorting is for displaying the state, not maintaining the heap.
        yield sorted(retained, reverse=True), kth

for retained, kth in top_k_states([5, 1, 9, 3, 9, 2], 3):
    print(retained, "kth:", kth)
`,
    expected: '[5] kth: None\n[5, 1] kth: None\n[9, 5, 1] kth: 1\n[9, 5, 3] kth: 3\n[9, 9, 5] kth: 5\n[9, 9, 5] kth: 5',
  },
  stablePriorities: {
    title: 'Keep equal priorities stable and make updates explicit',
    code: `import heapq
from itertools import count

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.live = {}
        self.tickets = count()

    def put(self, job, priority):
        # Job is a hashable identifier; replacing it counts as a new arrival.
        entry = (priority, next(self.tickets), job)
        self.live[job] = entry
        heapq.heappush(self.heap, entry)

    def cancel(self, job):
        return self.live.pop(job, None) is not None

    def pop(self):
        while self.heap:
            entry = heapq.heappop(self.heap)
            priority, ticket, job = entry
            if self.live.get(job) is entry:
                del self.live[job]
                return job, priority
        raise KeyError("no live jobs")

    def compact(self):
        self.heap = list(self.live.values())
        heapq.heapify(self.heap)

queue = PriorityQueue()
queue.put("report", 4)
queue.put("paint", 1)
queue.put("save", 1)
queue.put("report", 0)
print("cancelled:", queue.cancel("paint"))
print(queue.pop())
print(queue.pop())
queue.compact()
print("stored entries:", len(queue.heap))
queue.put("first", 2)
queue.put("second", 2)
print("ties:", queue.pop()[0], queue.pop()[0])
`,
    expected: "cancelled: True\n('report', 0)\n('save', 1)\nstored entries: 0\nties: first second",
  },
  mergeStreams: {
    title: 'Merge several sorted sources through a small frontier',
    code: `import heapq

def merge_sorted(sources):
    iterators = [iter(source) for source in sources]
    frontier = []
    for source_id, iterator in enumerate(iterators):
        try:
            frontier.append((next(iterator), source_id))
        except StopIteration:
            pass
    heapq.heapify(frontier)
    while frontier:
        value, source_id = heapq.heappop(frontier)
        yield value
        try:
            following = next(iterators[source_id])
        except StopIteration:
            continue
        heapq.heappush(frontier, (following, source_id))

sources = [[1, 4, 9], [], [1, 3, 10], [2, 8]]
print(list(merge_sorted(sources)))
print(list(merge_sorted([])))
`,
    expected: '[1, 1, 2, 3, 4, 8, 9, 10]\n[]',
  },
  eventSchedule: {
    title: 'A tiny event simulation can schedule its own next event',
    code: `import heapq
from itertools import count

events = []
tickets = count()
now = 0

def schedule(time, name, followups=()):
    if time < now:
        raise ValueError("cannot schedule in the simulated past")
    heapq.heappush(events, (time, next(tickets), name, followups))

schedule(4, "archive")
schedule(1, "sample", ((2, "sample complete"),))
schedule(1, "heartbeat")
while events:
    now, ticket, name, followups = heapq.heappop(events)
    print(now, name)
    for delay, following in followups:
        schedule(now + delay, following)
`,
    expected: '1 sample\n1 heartbeat\n3 sample complete\n4 archive',
  },
  trieOperations: {
    title: 'Store words, follow prefixes and delete without losing extensions',
    code: `${trieImplementation}
trie = Trie()
for word in ["car", "cart", "cat", "dog", "car"]:
    trie.insert(word)
print("ca exact / prefix:", trie.contains("ca"), trie.starts_with("ca"))
print("car suggestions:", trie.suggestions("car"))
print("delete car:", trie.delete("car"))
print("car exact / prefix:", trie.contains("car"), trie.starts_with("car"))
print("cart remains:", trie.contains("cart"))
print("delete absent:", trie.delete("cab"))
print("ca suggestions:", trie.suggestions("ca"))
trie.insert("")
print("empty word:", trie.contains(""))
trie.delete("")
print("empty word / prefix:", trie.contains(""), trie.starts_with(""))
`,
    expected: "ca exact / prefix: False True\ncar suggestions: ['car', 'cart']\ndelete car: True\ncar exact / prefix: False True\ncart remains: True\ndelete absent: False\nca suggestions: ['cart', 'cat']\nempty word: True\nempty word / prefix: False True",
  },
  prefixPolicies: {
    title: 'The same character path supports shortest and longest matches',
    code: `${trieImplementation}
def matching_prefix(trie, text, longest=False):
    node = trie.root
    best_length = 0 if node.terminal else None
    if best_length is not None and not longest:
        return ""
    for length, character in enumerate(text, start=1):
        node = node.children.get(character)
        if node is None:
            break
        if node.terminal:
            best_length = length
            if not longest:
                break
    return None if best_length is None else text[:best_length]

trie = Trie()
for key in ["a", "app", "apple", "map"]:
    trie.insert(key)
print("shortest:", matching_prefix(trie, "applesauce"))
print("longest:", matching_prefix(trie, "applesauce", longest=True))
print("no match:", matching_prefix(trie, "banana"))
print("suggestions:", trie.suggestions("app", 3))
`,
    expected: "shortest: a\nlongest: apple\nno match: None\nsuggestions: ['app', 'apple']",
  },
  smallestPractice: {
    title: 'Solution: retain the smallest k occurrences',
    code: `import heapq

def smallest_k(values, k):
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if k == 0:
        return []
    negative_heap = []
    for value in values:
        if len(negative_heap) < k:
            heapq.heappush(negative_heap, -value)
        else:
            heapq.heappushpop(negative_heap, -value)
    return sorted(-value for value in negative_heap)

print(smallest_k([5, 1, 9, 3, 9, 2], 3))
print(smallest_k([2, 2, -1], 4))
print(smallest_k([5, 1], 0))
assert smallest_k([], 2) == []
`,
    expected: '[1, 2, 3]\n[-1, 2, 2]\n[]',
  },
  heapSort: {
    title: 'Optional: sort in place by shrinking a max-heap',
    code: `def heap_sort(values):
    def sink(parent, end):
        # Heap occupies [0, end); the sorted suffix must not be touched.
        while 2 * parent + 1 < end:
            child = 2 * parent + 1
            if child + 1 < end and values[child + 1] > values[child]:
                child += 1
            if values[parent] >= values[child]:
                return
            values[parent], values[child] = values[child], values[parent]
            parent = child
    for parent in range(len(values) // 2 - 1, -1, -1):
        sink(parent, len(values))
    for end in range(len(values) - 1, 0, -1):
        values[0], values[end] = values[end], values[0]
        sink(0, end)

values = [7, 2, 9, 1, 5, 2]
heap_sort(values)
print(values)
empty = []
heap_sort(empty)
print(empty)
`,
    expected: '[1, 2, 2, 5, 7, 9]\n[]',
  },
};
