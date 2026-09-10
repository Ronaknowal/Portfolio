// Independently runnable objectOrientedCore examples; verified code/output pairs.
export const objectOrientedCoreExamples = {
oopInstances: {
    code: `class ReadingLog:
    def __init__(self, name):
        self.name = name
        self.values = []

    def add(self, value):
        self.values.append(value)

    def mean(self):
        if not self.values:
            return None
        return sum(self.values) / len(self.values)

morning = ReadingLog("morning")
evening = ReadingLog("evening")
morning.add(18)
ReadingLog.add(morning, 24)
print(morning.values, evening.values)
print(morning.mean(), evening.mean())
alias = morning
alias.add(30)
print(alias is morning, morning.mean())`,
    output: `[18, 24] []
21.0 None
True 24.0`,
  },
oopShared: {
    code: `class WrongLog:
    values = []  # one list on the class, NOT a new list per instance

a, b = WrongLog(), WrongLog()
a.values.append(18)
print(b.values)
print(a.values is b.values)
b.values = []  # assignment now creates b's instance attribute
print(a.values, b.values)`,
    output: `[18]
True
[18] []`,
  },
oopValidated: {
    code: `from math import isfinite

class ReadingLog:
    def __init__(self, name):
        self.name = name
        self._values = []

    def add(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("reading must be a number")
        try:
            number = float(value)
        except OverflowError:
            raise ValueError("reading outside float range") from None
        if not isfinite(number):
            raise ValueError("reading must be finite")
        self._values.append(number)

    @property
    def values(self):
        return tuple(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"ReadingLog(name={self.name!r}, count={len(self)})"

log = ReadingLog("morning")
log.add(18)
snapshot = log.values
log.add(24)
print(log)
print(len(log), snapshot, log.values)
try:
    log.add(float("nan"))
except ValueError as error:
    print(error)
print(len(log))`,
    output: `ReadingLog(name='morning', count=2)
2 (18.0,) (18.0, 24.0)
reading must be finite
2`,
  },
oopMethods: {
    code: `class Reading:
    def __init__(self, celsius):
        self.celsius = celsius

    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32

    @classmethod
    def from_fahrenheit(cls, value):
        return cls((value - 32) * 5 / 9)

    @staticmethod
    def unit_label():
        return "Celsius"

reading = Reading.from_fahrenheit(68)
print(reading.celsius, reading.fahrenheit())
print(Reading.unit_label())`,
    output: `20.0 68.0
Celsius`,
  },
oopComposition: {
    code: `class CelsiusFormatter:
    def format(self, value):
        return f"{value:.1f} C"

class FahrenheitFormatter:
    def format(self, value):
        return f"{value * 9 / 5 + 32:.1f} F"

class Report:
    def __init__(self, formatter):
        self.formatter = formatter

    def render(self, values):
        return ", ".join(self.formatter.format(v) for v in values)

print(Report(CelsiusFormatter()).render([18, 24]))
print(Report(FahrenheitFormatter()).render([18, 24]))`,
    output: `18.0 C, 24.0 C
64.4 F, 75.2 F`,
  },
oopInheritance: {
    code: `class Sensor:
    def __init__(self, name):
        self.name = name

    def describe(self):
        return f"Sensor {self.name}"

class CalibratedSensor(Sensor):
    def __init__(self, name, offset):
        super().__init__(name)
        self.offset = offset

    def read(self, raw):
        return raw + self.offset

    def describe(self):
        return f"{super().describe()} (offset {self.offset:+.1f})"

sensor = CalibratedSensor("room", -0.5)
print(sensor.describe())
print(sensor.read(24))
print(isinstance(sensor, Sensor))`,
    output: `Sensor room (offset -0.5)
23.5
True`,
  },
oopDataclass: {
    code: `from dataclasses import dataclass, field

@dataclass
class RunConfig:
    name: str
    tags: list[str] = field(default_factory=list)

a = RunConfig("baseline")
b = RunConfig("baseline")
print(a == b, a is b)
a.tags.append("test")
print(a)
print(b.tags)`,
    output: `True False
RunConfig(name='baseline', tags=['test'])
[]`,
  },
oopPractice: {
    code: `class DatasetSplit:
    def __init__(self, name, examples):
        self.name = name
        self._examples = list(examples)

    def __len__(self):
        return len(self._examples)

    def describe(self):
        return f"{self.name}: {len(self)} examples"

source = ["a", "b"]
split = DatasetSplit("train", source)
source.append("c")
assert len(split) == 2
assert DatasetSplit("empty", []).describe() == "empty: 0 examples"
print(split.describe())
print("checks passed")`,
    output: `train: 2 examples
checks passed`,
  }
};
