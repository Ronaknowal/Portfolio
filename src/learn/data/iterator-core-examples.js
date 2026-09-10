// Independently runnable iteratorCore examples; verified code/output pairs.
export const iteratorCoreExamples = {
iterProtocol: {
    code: `values = [18, 21, 24]
cursor = iter(values)
print(next(cursor))
print(list(cursor))
print(list(cursor))
print(next(cursor, "finished"))
print(list(values))
print(iter(cursor) is cursor)
try:
    next(cursor)
except StopIteration:
    print("StopIteration")`,
    output: `18
[21, 24]
[]
finished
[18, 21, 24]
True
StopIteration`,
  },
iterYield: {
    code: `def readings():
    print("start")
    yield 18
    print("resume")
    yield 24
    print("finish")

stream = readings()
print("created")
print(next(stream))
print(next(stream))
print(next(stream, "done"))
print(next(stream, "done"))`,
    output: `created
start
18
resume
24
finish
done
done`,
  },
iterExpressions: {
    code: `def double(value):
    print("work", value)
    return value * 2

eager = [double(x) for x in [1, 2]]
lazy = (double(x) for x in [1, 2])
print("ready")
print(next(lazy))
print(list(lazy))
print(list(lazy))
print(eager)`,
    output: `work 1
work 2
ready
work 1
2
work 2
[4]
[]
[2, 4]`,
  },
iterPipeline: {
    code: `from io import StringIO

def non_empty_lines(lines):
    for line in lines:
        text = line.strip()
        if text:
            yield text

def numeric_readings(lines):
    for text in non_empty_lines(lines):
        yield float(text)

with StringIO("18\\n\\n24\\n30\\n") as file:
    stream = numeric_readings(file)
    total = 0.0
    count = 0
    for value in stream:
        total += value
        count += 1
    print(count, total / count if count else None)
print(file.closed)`,
    output: `3 24.0
True`,
  },
iterBatches: {
    code: `from itertools import islice

def batches(iterable, size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    cursor = iter(iterable)
    while True:
        batch = tuple(islice(cursor, size))
        if not batch:
            return
        yield batch

print(list(batches(range(7), 3)))
print(list(batches([], 3)))
try:
    list(batches(range(3), 0))
except ValueError as error:
    print(error)`,
    output: `[(0, 1, 2), (3, 4, 5), (6,)]
[]
size must be a positive integer`,
  },
iterTools: {
    code: `from itertools import chain, count, islice, pairwise

print(list(islice(count(10, 2), 4)))
print(list(chain([1, 2], [3])))
print(list(pairwise([18, 21, 24])))
print(list(map(abs, [-2, 3])))
print(list(filter(lambda x: x > 0, [-2, 0, 3])))
print(list(enumerate(["a", "b"], start=1)))
print(list(zip(["a", "b"], [1])))
try:
    list(zip(["a", "b"], [1], strict=True))
except ValueError:
    print("length mismatch")`,
    output: `[10, 12, 14, 16]
[1, 2, 3]
[(18, 21), (21, 24)]
[2, 3]
[3]
[(1, 'a'), (2, 'b')]
[('a', 1)]
length mismatch`,
  },
iterCustom: {
    code: `class Countdown:
    def __init__(self, start):
        self.remaining = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining <= 0:
            raise StopIteration
        value = self.remaining
        self.remaining -= 1
        return value

class CountdownSource:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return Countdown(self.start)

source = CountdownSource(3)
print(list(source), list(source))
cursor = iter(source)
print(list(cursor), list(cursor))`,
    output: `[3, 2, 1] [3, 2, 1]
[3, 2, 1] []`,
  },
iterDelegate: {
    code: `def flatten_one_level(groups):
    for group in groups:
        yield from group

print(list(flatten_one_level([[1, 2], [], [3]])))

def managed():
    try:
        yield 1
        yield 2
    finally:
        print("cleaned up")

stream = managed()
print(next(stream))
stream.close()`,
    output: `[1, 2, 3]
1
cleaned up`,
  },
iterSend: {
    code: `def accumulator():
    total = 0
    while True:
        value = yield total
        if value is None:
            return total
        total += value

stream = accumulator()
print(next(stream))
print(stream.send(3))
print(stream.send(4))
try:
    stream.send(None)
except StopIteration as end:
    print("returned", end.value)`,
    output: `0
3
7
returned 7`,
  },
iterPractice: {
    code: `def take_until_missing(values):
    for value in values:
        if value is None:
            return
        yield value

print(list(take_until_missing([0, 18, None, 24])))
print(list(take_until_missing([])))

def running_mean(values):
    total = 0.0
    for count, value in enumerate(values, start=1):
        total += value
        yield total / count

print(list(running_mean([18, 24, 30])))
assert list(running_mean([])) == []`,
    output: `[0, 18]
[]
[18.0, 21.0, 24.0]`,
  }
};
