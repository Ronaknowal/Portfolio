import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { oopExamples, simpleLogClass, validatedLogClass, compositionClasses } from "../src/learn/data/oop-foundations-examples.js";
import { initialObjectState, methodCallFrames, initialLookupState, lookupValues, lookupAction, readingCandidates, validateReading, compositionFrames } from "../src/learn/data/oop-foundations-model.js";

const outputDirectory = "scratch/oop-foundations";
fs.mkdirSync(outputDirectory, { recursive: true });
const python = code => {
  const result = spawnSync(process.env.PYTHON || "python", ["-c", code], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
  assert.equal(result.status, 0, result.stderr || String(result.error));
  return result.stdout.trim().replace(/\r\n/g, "\n");
};

const examples = [];
for (const [id, example] of Object.entries(oopExamples)) {
  fs.writeFileSync(`${outputDirectory}/${example.filename}`, example.code + "\n");
  assert.equal(python(example.code), example.output, `Displayed output mismatch: ${id}`);
  examples.push(id);
}

const calls = ["morning", "evening", "alias"].flatMap(receiver => [18, 24, 30].map(value => ({ receiver, value })));
const bindingCases = calls.flatMap(a => calls.map(b => [a, b]));
const bindingExpected = JSON.parse(python(`${simpleLogClass}
import json
cases = json.loads(${JSON.stringify(JSON.stringify(bindingCases))})
results = []
for calls in cases:
    morning = ReadingLog("morning")
    evening = ReadingLog("evening")
    names = {"morning": morning, "evening": evening, "alias": morning}
    for call in calls:
        names[call["receiver"]].add(call["value"])
    results.append({"A": morning.values, "B": evening.values})
print(json.dumps(results))`));
bindingCases.forEach((calls, index) => {
  let state = initialObjectState();
  for (const call of calls) {
    const saved = JSON.stringify(state);
    const frames = methodCallFrames(state, call.receiver, call.value);
    assert.equal(JSON.stringify(state), saved, "Trace must not mutate initial state");
    for (const frame of frames.slice(0, 4)) assert.deepEqual(frame.state, state, "No mutation before append");
    state = frames[4].state;
  }
  assert.deepEqual(state, bindingExpected[index]);
});

const actions = ["A", "B"].flatMap(receiver => ["append", "assign"].map(action => ({ receiver, action })));
const lookupCases = ["class", "instance"].flatMap(mode => actions.flatMap(a => actions.flatMap(b => actions.map(c => ({ mode, actions: [a, b, c] })))));
const lookupExpected = JSON.parse(python(`import json
cases = json.loads(${JSON.stringify(JSON.stringify(lookupCases))})
results = []
for case in cases:
    if case["mode"] == "class":
        class Log:
            values = []
    else:
        class Log:
            def __init__(self):
                self.values = []
    objects = {"A": Log(), "B": Log()}
    for action in case["actions"]:
        obj = objects[action["receiver"]]
        if action["action"] == "append":
            obj.values.append(18)
        else:
            obj.values = [99]
    results.append({"shared": getattr(Log, "values", []),
                    "ownA": vars(objects["A"]).get("values"),
                    "ownB": vars(objects["B"]).get("values"),
                    "readA": objects["A"].values,
                    "readB": objects["B"].values})
print(json.dumps(results))`));
lookupCases.forEach((test, index) => {
  let state = initialLookupState(test.mode);
  for (const action of test.actions) {
    const saved = JSON.stringify(state);
    const next = lookupAction(state, action.receiver, action.action);
    assert.equal(JSON.stringify(state), saved, "Lookup operation must preserve history snapshots");
    state = next;
  }
  assert.deepEqual({ shared: state.shared, ownA: state.ownA, ownB: state.ownB, readA: lookupValues(state, "A").values, readB: lookupValues(state, "B").values }, lookupExpected[index]);
});

const validationExpected = JSON.parse(python(`${validatedLogClass}
import json
results = []
for value in [${readingCandidates.map(candidate => candidate.code).join(", ")}]:
    log = ReadingLog("checked")
    log.add(18)
    error = None
    try:
        log.add(value)
    except (TypeError, ValueError) as caught:
        error = type(caught).__name__ + ": " + str(caught)
    results.append({"error": error, "after": list(log.values)})
print(json.dumps(results))`));
readingCandidates.forEach((candidate, index) => {
  const result = validateReading(candidate.id);
  assert.deepEqual({ error: result.error, after: result.after }, validationExpected[index]);
  assert.equal(result.gates.filter(gate => gate.status === "blocked").length, result.error ? 1 : 0);
  const failedIndex = result.gates.findIndex(gate => gate.status === "blocked");
  if (failedIndex !== -1) assert.ok(result.gates.slice(failedIndex + 1).every(gate => gate.status === "waiting"));
});

const formatCases = ["celsius", "fahrenheit"].flatMap(formatter => [0, 20, 30].map(value => ({ formatter, value })));
const formatExpected = JSON.parse(python(`${compositionClasses}
import json
results = []
for formatter in (CelsiusFormatter(), FahrenheitFormatter()):
    for value in (0, 20, 30):
        results.append({"formatted": formatter.format(value), "report": Report(formatter).render(value)})
print(json.dumps(results))`));
formatCases.forEach((test, index) => {
  const frames = compositionFrames(test.value, test.formatter);
  assert.deepEqual({ formatted: frames[3].output, report: frames[4].output }, formatExpected[index]);
  assert.ok(frames.slice(0, 3).every(frame => frame.output === null));
});

// Behaviours beyond the displayed worked cases: errors, read-only snapshots,
// input aliasing, blank IDs and cross-split scope; not merely model fixtures.
python(`class Repaired:
    def __init__(self, values=None):
        self.values = [] if values is None else list(values)
a, b = Repaired(), Repaired()
a.values.append(1)
assert b.values == []
source = [[1]]
c = Repaired(source)
source.append([2])
assert c.values == [[1]]
source[0].append(9)
assert c.values == [[1, 9]]
${validatedLogClass}
a, b = ReadingLog("a"), ReadingLog("b")
assert a.mean() is None and len(a) == 0
a.add(-5)
old = a.values
a.add(0)
assert old == (-5.0,) and a.values == (-5.0, 0.0) and b.values == ()
for invalid in (False, None, [], float("-inf")):
    before = a.values
    try:
        a.add(invalid)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("invalid input accepted")
    assert a.values == before
try:
    a.values = [999]
except AttributeError:
    pass
else:
    raise AssertionError("read-only property allowed assignment")
try:
    old.append(1)
except AttributeError:
    pass
else:
    raise AssertionError("snapshot was mutable")`);

python(`${oopExamples.mission.code}
for invalid, error_type in [(7, TypeError), ("", ValueError), ("   ", ValueError)]:
    before = train.sample_ids
    try:
        train.add(invalid)
    except error_type:
        pass
    else:
        raise AssertionError("invalid ID accepted")
    assert before == train.sample_ids
try:
    DatasetSplit("bad", ["x", "x"])
except ValueError:
    pass
else:
    raise AssertionError("constructor accepted duplicate")
snapshot = train.sample_ids
train.add("s5")
assert snapshot == ("s1", "s2", "s4")
overlap = DatasetSplit("other", ["s1"])
assert overlap.sample_ids[0] in train.sample_ids
class LinesFormatter:
    def format(self, split):
        return "\\n".join(split.sample_ids) or "(empty)"
assert SplitReport(LinesFormatter()).render(validation) == "(empty)"
assert SplitReport(LinesFormatter()).render(DatasetSplit("tiny", ["a", "b"])) == "a\\nb"`);

const report = { python: python("import sys; print(sys.version)"), examples: examples.length, exampleIds: examples, bindingCases: bindingCases.length, lookupCases: lookupCases.length, validationCases: readingCandidates.length, compositionCases: formatCases.length, additionalBehaviourChecks: "passed", status: "passed" };
fs.writeFileSync(`${outputDirectory}/runtime-verification.json`, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
