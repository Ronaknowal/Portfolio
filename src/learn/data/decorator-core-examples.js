// Independently runnable decoratorCore examples; verified code/output pairs.
export const decoratorCoreExamples = {
decoratorBasic: {
    code: `from functools import wraps

def logged(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        print("enter", function.__name__)
        try:
            return function(*args, **kwargs)
        finally:
            print("leave", function.__name__)
    return wrapper

@logged
def mean(values):
    """Return the arithmetic mean of a nonempty sequence."""
    if not values:
        raise ValueError("no readings")
    return sum(values) / len(values)

print(mean([18, 24]))
print(mean.__name__)
try:
    mean([])
except ValueError as error:
    print("caught:", error)`,
    output: `enter mean
leave mean
21.0
mean
enter mean
leave mean
caught: no readings`,
  },
decoratorFactory: {
    code: `from functools import wraps

def tagged(label):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            print(label, "before")
            result = function(*args, **kwargs)
            print(label, "after")
            return result
        return wrapper
    return decorate

@tagged("outer")
@tagged("inner")
def report():
    print("body")
    return 21

print("result", report())`,
    output: `outer before
inner before
body
inner after
outer after
result 21`,
  },
decoratorTimer: {
    code: `from functools import wraps
from time import perf_counter

def timed(clock=perf_counter):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            started = clock()
            try:
                return function(*args, **kwargs)
            finally:
                print(f"elapsed: {clock() - started:.3f}s")
        return wrapper
    return decorate

ticks = iter([10.0, 10.25])  # fake clock: deterministic teaching output

@timed(clock=lambda: next(ticks))
def total(values):
    return sum(values)

print(total([18, 24]))`,
    output: `elapsed: 0.250s
42`,
  },
decoratorCache: {
    code: `from functools import lru_cache

@lru_cache(maxsize=4)
def fahrenheit(celsius):
    print("compute", celsius)
    return celsius * 9 / 5 + 32

print(fahrenheit(20))
print(fahrenheit(20))
print(fahrenheit.cache_info())
fahrenheit.cache_clear()
print(fahrenheit.cache_info().currsize)`,
    output: `compute 20
68.0
68.0
CacheInfo(hits=1, misses=1, maxsize=4, currsize=1)
0`,
  },
contextClass: {
    code: `from io import StringIO

class TextResource:
    def __enter__(self):
        print("enter")
        self.file = StringIO("18\\n24\\n")
        return self.file

    def __exit__(self, error_type, error, traceback):
        self.file.close()
        print("exit:", error_type.__name__ if error_type else "no error")
        return False

try:
    with TextResource() as file:
        print(file.readline().strip())
        raise ValueError("bad reading")
except ValueError as error:
    print("caught:", error)
print("closed:", file.closed)`,
    output: `enter
18
exit: ValueError
caught: bad reading
closed: True`,
  },
contextGenerator: {
    code: `from contextlib import contextmanager
from io import StringIO

@contextmanager
def text_resource(text):
    file = StringIO(text)
    print("acquire")
    try:
        yield file
    finally:
        file.close()
        print("release")

with text_resource("18\\n24\\n") as file:
    print([float(line) for line in file])
print(file.closed)`,
    output: `acquire
[18.0, 24.0]
release
True`,
  },
contextStack: {
    code: `from contextlib import ExitStack, contextmanager

@contextmanager
def resource(name, fail=False):
    print("acquire", name)
    if fail:
        raise ValueError("could not acquire " + name)
    try:
        yield name
    finally:
        print("release", name)

try:
    with ExitStack() as stack:
        stack.enter_context(resource("A"))
        stack.enter_context(resource("B"))
        stack.enter_context(resource("C", fail=True))
except ValueError as error:
    print("caught:", error)`,
    output: `acquire A
acquire B
acquire C
release B
release A
caught: could not acquire C`,
  },
decoratorPractice: {
    code: `from functools import wraps
from contextlib import redirect_stdout
from io import StringIO

def counted(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return function(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@counted
def report(value):
    print("reading", value)
    return value * 2

captured = StringIO()
with redirect_stdout(captured):
    result = report(value=18)
print(result, report.calls, report.__name__)
print(captured.getvalue().strip())`,
    output: `36 1 report
reading 18`,
  }
};
