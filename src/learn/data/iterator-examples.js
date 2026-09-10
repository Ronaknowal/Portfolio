import { iteratorCoreExamples } from "./iterator-core-examples.js";

export const iterationExamples = {...iteratorCoreExamples};

iterationExamples.frame = {
  code: `from inspect import getgeneratorstate

def countdown(start):
    remaining = start
    try:
        while remaining > 0:
            yield remaining
            remaining -= 1
    finally:
        print("cleanup")

stream = countdown(2)
print(getgeneratorstate(stream))
print(next(stream), getgeneratorstate(stream))
print(next(stream), getgeneratorstate(stream))
print(next(stream, "END"), getgeneratorstate(stream))
print(next(stream, "END"))
unstarted = countdown(2)
unstarted.close()
print("unstarted:", getgeneratorstate(unstarted))`,
  output:`GEN_CREATED
2 GEN_SUSPENDED
1 GEN_SUSPENDED
cleanup
END GEN_CLOSED
END
unstarted: GEN_CLOSED`,
};

iterationExamples.pull = {
  code: `from itertools import islice

def source(lines):
    for number, line in enumerate(lines, start=1):
        print("read line", number)
        yield line

def readings(lines):
    for line in lines:
        text = line.strip()
        if text:
            yield float(text)

stream = readings(source(["18", "", "24", "30"]))
print("created")
print(list(islice(stream, 2)))
# The original stream is still available: a new consumer can resume it.
print("remaining:", list(stream))`,
  output:`created
read line 1
read line 2
read line 3
[18.0, 24.0]
read line 4
remaining: [30.0]`,
};

iterationExamples.tee = {
  code:`from itertools import tee

def source():
    for value in [18, 21, 24]:
        print("source produced", value)
        yield value

a, b = tee(source())
print("a", next(a), next(a))
print("b", next(b))
print("b", next(b))
print("a", next(a))
print("b", next(b))`,
  output:`source produced 18
source produced 21
a 18 21
b 18
b 21
source produced 24
a 24
b 24`,
};

iterationExamples.alarm = {
  code:`def first_crossing(values, threshold):
    """Return (index, value) of the first reading strictly above threshold.

    Input: one-pass iterable of finite numbers or None. None ends the feed.
    Return None if no reading crosses before end. Consume no later reading.
    """
    for index, value in enumerate(values):
        if value is None:
            return None
        if value > threshold:
            return index, value
    return None

stream = iter([0, 18, 24, 30])
print(first_crossing(stream, 20))
print("remaining:", list(stream))
print(first_crossing(iter([0, None, 24]), 20))
print(first_crossing(iter([]), 20))
print(first_crossing(iter([20, 21]), 20))`,
  output:`(2, 24)
remaining: [30]
None
None
(1, 21)`,
};
