import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import { spawnSync } from "node:child_process";
import { pandasExamples } from "../src/learn/data/pandas-examples.js";
import { joinOrders, lookupRows, modelJoin } from "../src/learn/data/pandas-join-model.js";

const python = process.env.LESSON_PYTHON || "python";
const failures = [];
for (const [id, example] of Object.entries(pandasExamples)) {
  const result = spawnSync(python, ["-B", "-c", example.code], {
    cwd: os.tmpdir(), encoding: "utf8", timeout: 20000,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: "" },
  });
  const output = result.stdout?.replace(/\r\n/g, "\n").trimEnd();
  if (result.status !== 0 || output !== example.output.trimEnd()) {
    failures.push(id);
    console.log(JSON.stringify({ id, status: result.status, output, stderr: result.stderr, error: result.error?.message }));
  } else {
    console.log(id + ": exact output verified");
  }
}
assert.deepEqual(failures, [], "Example output or execution failures");
const source = fs.readFileSync("src/learn/data/topics/pandas-data-wrangling-joins-grouping.jsx", "utf8");
const { collectLessonExamples } = await import('./lib/lesson-examples.mjs');
const refs = (await collectLessonExamples('src/learn/data/topics/pandas-data-wrangling-joins-grouping.jsx')).filter(reference=>reference.collection==='pandasExamples').map(reference=>reference.key);
assert.deepEqual([...new Set(refs)].sort(), Object.keys(pandasExamples).sort(), "Every verified example must be displayed");
console.log(Object.keys(pandasExamples).length + " examples verified and linked.");

// Compare every explorer state against real Pandas, ignoring illustrative row order.
const cases = [];
for (const how of ["inner", "left", "outer"]) for (const duplicate of [false, true]) for (const validate of [false, true]) {
  cases.push({ how, duplicate, validate, left: joinOrders, right: lookupRows(duplicate) });
}
const reference = spawnSync(python, ["-B", "-c", `import json, sys
import pandas as pd
results = []
for case in json.load(sys.stdin):
    try:
        joined = pd.DataFrame(case["left"]).merge(pd.DataFrame(case["right"]), on="customer",
            how=case["how"], validate="many_to_one" if case["validate"] else None, indicator=True)
        results.append({"error": False, "rows": json.loads(joined.rename(columns={"_merge": "match"}).to_json(orient="records"))})
    except pd.errors.MergeError:
        results.append({"error": True, "rows": []})
print(json.dumps(results))`], { input: JSON.stringify(cases), encoding: "utf8", timeout: 20000 });
assert.equal(reference.status, 0, reference.stderr || reference.error?.message);
const canonical = rows => rows.map(({ order, customer, region, match }) => JSON.stringify([order, customer, region, match])).sort();
JSON.parse(reference.stdout).forEach((expected, i) => {
  const { how, duplicate, validate } = cases[i];
  const actual = modelJoin(how, duplicate, validate);
  assert.equal(Boolean(actual.error), expected.error, JSON.stringify(cases[i]));
  assert.deepEqual(canonical(actual.rows), canonical(expected.rows), JSON.stringify(cases[i]));
});
console.log("All 12 join explorer states agree with Pandas.");

// Execute the two project exercises, without the original fixture-specific assertions.
const prefix = pandasExamples.project.code.split("print(audit)")[0];
const exercises = spawnSync(python, ["-B", "-c", `import json, sys
source = json.load(sys.stdin)
namespace = {}
exec(source.replace("105,C9,40,paid", "105,C2,40,paid"), namespace)
assert namespace["audit"] == {"raw": 6, "exact_duplicates": 1, "invalid_amount": 1, "not_paid": 1, "unmatched": 0, "accepted": 3}
assert namespace["summary"]["orders"].tolist() == [1, 2]
assert namespace["summary"]["revenue_cents"].tolist() == [10, 60]
try:
    exec(source.replace("105,C9,40,paid", "105,C9,40,paid\\n102,C2,25,paid"), {})
except ValueError as error:
    assert str(error) == "missing or conflicting order ID"
else:
    raise AssertionError("Conflicting order ID was not rejected")
print("Both project exercise solutions verified.")`], { input: JSON.stringify(prefix), encoding: "utf8", timeout: 20000 });
assert.equal(exercises.status, 0, exercises.stderr || exercises.error?.message);
console.log(exercises.stdout.trim());
