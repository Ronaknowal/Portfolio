const example = (filename, code, output) => ({ filename, code: code.trim(), output: output.trim() });

export const simpleLogClass = `class ReadingLog:
    def __init__(self, name):
        self.name = name
        self.values = []

    def add(self, value):
        self.values.append(value)

    def mean(self):
        if not self.values:
            return None
        return sum(self.values) / len(self.values)`;

export const validatedLogClass = `import math

class ReadingLog:
    def __init__(self, name):
        self.name = name
        self._values = []

    def add(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected int or float, not bool")
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError("reading does not fit float storage") from None
        if not math.isfinite(converted):
            raise ValueError("reading must be finite")
        self._values.append(converted)

    @property
    def values(self):
        return tuple(self._values)

    def mean(self):
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"ReadingLog(name={self.name!r}, count={len(self)})"`;

export const compositionClasses = `class CelsiusFormatter:
    def format(self, celsius):
        return f"{celsius:.1f} C"

class FahrenheitFormatter:
    def format(self, celsius):
        return f"{celsius * 9 / 5 + 32:.1f} F"

class Report:
    def __init__(self, formatter):
        self.formatter = formatter

    def render(self, celsius):
        text = self.formatter.format(celsius)
        return f"Reading: {text}"`;

export const oopExamples = {
  functions: example("log_functions.py", `def new_log(name):
    return {"name": name, "values": []}

def add(log, value):
    log["values"].append(value)

morning = new_log("morning")
evening = new_log("evening")
add(morning, 18)
add(morning, 24)
print(morning)
print(evening)`, `{'name': 'morning', 'values': [18, 24]}
{'name': 'evening', 'values': []}`),

  instances: example("log_instances.py", `${simpleLogClass}

morning = ReadingLog("morning")
evening = ReadingLog("evening")
morning.add(18)
morning.add(24)
print(morning.values, morning.mean())
print(evening.values, evening.mean())
alias = morning
alias.add(30)
print(morning.values, morning.mean())
print(alias is morning, evening is morning)`, `[18, 24] 21.0
[] None
[18, 24, 30] 24.0
True False`),

  bound: example("bound_method.py", `${simpleLogClass}

morning = ReadingLog("morning")
evening = ReadingLog("evening")
add_to_morning = morning.add
morning = evening
add_to_morning(18)
print(add_to_morning.__self__.values)
print(morning.values)
print(add_to_morning.__func__ is ReadingLog.add)`, `[18]
[]
True`),

  shared: example("shared_state.py", `class SharedLog:
    values = []

a = SharedLog()
b = SharedLog()
a.values.append(18)
print(a.values, b.values, SharedLog.values)
b.values = [99]
a.values.append(24)
print(a.values, b.values, SharedLog.values)
print("values" in a.__dict__, "values" in b.__dict__)`, `[18] [18] [18]
[18, 24] [99] [18, 24]
False True`),

  validated: example("validated_log.py", `${validatedLogClass}

log = ReadingLog("morning")
print(len(log), log.mean())
log.add(18)
snapshot = log.values
log.add(24)
for value in (True, "24", float("nan"), float("inf"), 10 ** 400):
    try:
        log.add(value)
    except (TypeError, ValueError) as error:
        print(type(error).__name__ + ":", error)
print("snapshot:", snapshot)
print("current:", log.values)
print("count:", len(log), "mean:", log.mean())
print(repr(log))`, `0 None
TypeError: expected int or float, not bool
TypeError: expected int or float, not bool
ValueError: reading must be finite
ValueError: reading must be finite
ValueError: reading does not fit float storage
snapshot: (18.0,)
current: (18.0, 24.0)
count: 2 mean: 21.0
ReadingLog(name='morning', count=2)`),

  composition: example("report_composition.py", `${compositionClasses}

for formatter in (CelsiusFormatter(), FahrenheitFormatter()):
    report = Report(formatter)
    print(report.render(20))
    print(report.render(0))

class RecordingFormatter:
    def __init__(self):
        self.received = []

    def format(self, celsius):
        self.received.append(celsius)
        return "checked"

fake = RecordingFormatter()
assert Report(fake).render(20) == "Reading: checked"
assert fake.received == [20]
print("delegation test passed")`, `Reading: 20.0 C
Reading: 0.0 C
Reading: 68.0 F
Reading: 32.0 F
delegation test passed`),

  methods: example("method_kinds.py", `class Reading:
    def __init__(self, celsius):
        self.celsius = celsius

    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32

    @classmethod
    def from_fahrenheit(cls, value):
        return cls((value - 32) * 5 / 9)

    @staticmethod
    def unit():
        return "degrees Celsius"

reading = Reading.from_fahrenheit(68)
print(reading.celsius)
print(reading.fahrenheit())
print(Reading.unit())`, `20.0
68.0
degrees Celsius`),

  inheritance: example("sensor_inheritance.py", `class Sensor:
    def __init__(self, name):
        self.name = name

    def correct(self, raw):
        return raw

    def describe(self):
        return f"Sensor {self.name}"

class CalibratedSensor(Sensor):
    def __init__(self, name, offset):
        super().__init__(name)
        self.offset = offset

    def correct(self, raw):
        return raw + self.offset

    def describe(self):
        return f"{super().describe()} (offset {self.offset})"

sensor = CalibratedSensor("room", -0.5)
print(sensor.describe())
print(sensor.correct(24))
print(isinstance(sensor, Sensor))
print([kind.__name__ for kind in CalibratedSensor.__mro__])`, `Sensor room (offset -0.5)
23.5
True
['CalibratedSensor', 'Sensor', 'object']`),

  dataclass: example("reading_record.py", `from dataclasses import dataclass, field

@dataclass
class Reading:
    celsius: float
    tags: list[str] = field(default_factory=list)

a = Reading(20.0)
b = Reading(20.0)
print(a == b, a is b)
a.tags.append("reviewed")
print(a)
print(b)
print(a == b)
try:
    hash(a)
except TypeError:
    print("mutable value records are not hashable by default")`, `True False
Reading(celsius=20.0, tags=['reviewed'])
Reading(celsius=20.0, tags=[])
False
mutable value records are not hashable by default`),

  frozen: example("frozen_record.py", `from dataclasses import dataclass, field, FrozenInstanceError

@dataclass(frozen=True)
class FrozenReading:
    celsius: float
    tags: list[str] = field(default_factory=list)

reading = FrozenReading(20.0)
reading.tags.append("reviewed")
print(reading.tags)
try:
    reading.celsius = 21.0
except FrozenInstanceError:
    print("field reassignment is blocked")
print(FrozenReading("warm").celsius)`, `['reviewed']
field reassignment is blocked
warm`),

  mission: example("dataset_split.py", `class DatasetSplit:
    def __init__(self, name, sample_ids):
        self.name = name
        self._sample_ids = []
        for sample_id in sample_ids:
            self.add(sample_id)

    @property
    def sample_ids(self):
        return tuple(self._sample_ids)

    def add(self, sample_id):
        if not isinstance(sample_id, str):
            raise TypeError("sample ID must be text")
        if not sample_id or sample_id.isspace():
            raise ValueError("sample ID must not be blank")
        if sample_id in self._sample_ids:
            raise ValueError("duplicate sample ID")
        self._sample_ids.append(sample_id)

    def __len__(self):
        return len(self._sample_ids)

class CountFormatter:
    def format(self, split):
        return f"{split.name}: {len(split)} samples"

class NamesFormatter:
    def format(self, split):
        names = ", ".join(split.sample_ids) or "(empty)"
        return f"{split.name}: {names}"

class SplitReport:
    def __init__(self, formatter):
        self.formatter = formatter

    def render(self, split):
        return self.formatter.format(split)

source = ["s1", "s2"]
train = DatasetSplit("train", source)
validation = DatasetSplit("validation", [])
source.append("s3")
train.add("s4")
assert train.sample_ids == ("s1", "s2", "s4")
assert validation.sample_ids == ()
before = train.sample_ids
try:
    train.add("s1")
except ValueError as error:
    print(error)
assert train.sample_ids == before
for formatter in (CountFormatter(), NamesFormatter()):
    report = SplitReport(formatter)
    print(report.render(train))
    print(report.render(validation))
print("checks passed")`, `duplicate sample ID
train: 3 samples
validation: 0 samples
train: s1, s2, s4
validation: (empty)
checks passed`),
};
