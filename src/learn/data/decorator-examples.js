import { decoratorCoreExamples } from "./decorator-core-examples.js";

export const decoratorExamples = {...decoratorCoreExamples};

decoratorExamples.order = {
  code:`from functools import wraps

def double(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs) * 2
    return wrapper

def cap(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return min(function(*args, **kwargs), 10)
    return wrapper

def reading(value):
    return value

cap_outside = cap(double(reading))
double_outside = double(cap(reading))
for value in [3, 8, 12]:
    print(value, cap_outside(value), double_outside(value))`,
  output:`3 6 6
8 10 16
12 10 20`,
};

decoratorExamples.registry = {
  code:`handlers = {}

def register(name):
    def decorate(function):
        if name in handlers:
            raise ValueError("duplicate handler: " + name)
        handlers[name] = function
        return function  # Registration needs no per-call wrapper.
    return decorate

@register("celsius")
def convert_celsius(value):
    return value * 9 / 5 + 32

print(sorted(handlers))
print(handlers["celsius"] is convert_celsius)
print(handlers["celsius"](20))`,
  output:`['celsius']
True
68.0`,
};

decoratorExamples.restore = {
  code:`from contextlib import contextmanager

@contextmanager
def temporary_value(settings, key, value):
    """Temporarily replace one mapping entry, restoring its previous presence.

    Single-threaded use only. This is a shallow binding change, not a copy
    of mutable values or a transaction over every entry in settings.
    """
    missing = object()
    original = settings.get(key, missing)
    settings[key] = value
    try:
        yield settings
    finally:
        if original is missing:
            settings.pop(key, None)
        else:
            settings[key] = original

settings = {"units": "C", "limit": None}
try:
    with temporary_value(settings, "units", "F"):
        print("outer", settings["units"])
        with temporary_value(settings, "units", "K"):
            print("inner", settings["units"])
        print("restored outer", settings["units"])
        raise ValueError("stop experiment")
except ValueError as error:
    print("caught", error)
print("final", settings)
with temporary_value(settings, "debug", True):
    print("debug", settings["debug"])
print("debug present:", "debug" in settings)`,
  output:`outer F
inner K
restored outer F
caught stop experiment
final {'units': 'C', 'limit': None}
debug True
debug present: False`,
};
