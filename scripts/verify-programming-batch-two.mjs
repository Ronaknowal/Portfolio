// Legacy fixture regression checks plus live decorator/testing source coverage.
// Reimplemented NumPy uses numpy-foundations-verify.mjs as its authoritative
// current-lesson check; old numpy* fixtures are retained regression evidence.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import { decoratorCoreExamples as batchTwoExamplesPart1 } from "../src/learn/data/decorator-core-examples.js";
import { testingExamples as batchTwoExamplesPart2 } from "../src/learn/data/testing-examples.js";
import { numpyReferenceExamples as batchTwoExamplesPart3 } from "../src/learn/data/numpy-reference-examples.js";
const batchTwoExamples = { ...batchTwoExamplesPart1, ...batchTwoExamplesPart2, ...batchTwoExamplesPart3 };
import { decoratorTrace } from "../src/learn/data/decorator-introduction-trace.js";

const python = process.env.LESSON_PYTHON || "python";
const slugs = ["decorators-context-managers", "testing-debugging-dependency-management"];
const refs = new Set();
for (const slug of slugs) {
  const source = fs.readFileSync("src/learn/data/topics/" + slug + ".jsx", "utf8");
  assert.match(source, /hasIntegratedGuide: true/);
  for (const reference of (await collectLessonExamples('src/learn/data/topics/'+slug+'.jsx')).filter(reference=>Object.hasOwn(batchTwoExamples,reference.key))) {
    assert.ok(batchTwoExamples[reference.key]);
    refs.add(reference.key);
  }
}
assert.deepEqual([...refs].sort(), Object.keys(batchTwoExamples).filter(key => !key.startsWith('numpy')).sort());
assert.equal(decoratorTrace.steps.at(-1).output, batchTwoExamples.decoratorFactory.output);
let count = 0;
for (const [id, example] of Object.entries(batchTwoExamples)) {
  const root = path.resolve(os.tmpdir());
  const dir = fs.mkdtempSync(path.join(root, "python-batch-two-"));
  assert.equal(path.dirname(path.resolve(dir)), root);
  const run = args => spawnSync(python, ["-B", ...args], {
    cwd: dir, encoding: "utf8", timeout: 20000,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: "" },
  });
  try {
    for (const [filename, source] of Object.entries(example.files || {})) {
      assert.equal(path.basename(filename), filename);
      fs.writeFileSync(path.join(dir, filename), source);
    }
    const filename = example.filename || "lesson.py";
    assert.equal(path.basename(filename), filename);
    fs.writeFileSync(path.join(dir, filename), example.code);
    const result = run([filename]);
    assert.equal(result.status, 0, id + ": " + (result.stderr || result.error));
    assert.equal(result.stdout.replace(/\r\n/g, "\n").trimEnd(), example.output.trimEnd(), id + ": displayed output mismatch");
    if (id === "testingSuite") {
      const discovered = run(["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]);
      assert.equal(discovered.status, 0, discovered.stderr);
      assert.match(discovered.stderr, /Ran 4 tests/);
      // Deliberately break only this temporary copy: ensure the suite can actually reject a bug.
      fs.writeFileSync(path.join(dir, "metrics.py"), "def mean(values):\n    return values[0] / len(values)\n");
      const broken = run(["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]);
      assert.equal(broken.status, 1, "Known-bad implementation should fail");
      assert.match(broken.stderr, /FAILED/);
    }
    count++;
    console.log(id + ": executed output verified");
  } finally {
    // Dedicated mkdtemp child under the verified temporary root; never a user data directory.
    fs.rmSync(dir, { recursive: true, force: true });
  }
}
console.log(count + " examples passed; discovery and deliberate-bug detection passed.");
