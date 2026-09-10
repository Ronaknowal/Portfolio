// Legacy fixture regression checks plus live iterator source coverage.
// Reimplemented Python/OOP use python-foundations-verify.mjs and
// verify-oop-foundations.mjs as their authoritative current-lesson checks.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import { pythonCoreExamples as programmingExamplesPart1 } from "../src/learn/data/python-core-examples.js";
import { objectOrientedCoreExamples as programmingExamplesPart2 } from "../src/learn/data/oop-core-examples.js";
import { iteratorCoreExamples as programmingExamplesPart3 } from "../src/learn/data/iterator-core-examples.js";
const programmingExamples = { ...programmingExamplesPart1, ...programmingExamplesPart2, ...programmingExamplesPart3 };

const python = process.env.LESSON_PYTHON || "python";
const lessons = [
  "iterators-iterables-generators",
];
const referenced = new Set();
for (const slug of lessons) {
  const source = fs.readFileSync(`src/learn/data/topics/${slug}.jsx`, "utf8");
  assert.match(source, /hasIntegratedGuide: true/);
  for (const reference of (await collectLessonExamples(`src/learn/data/topics/${slug}.jsx`)).filter(reference=>Object.hasOwn(programmingExamples,reference.key))) {
    assert.ok(programmingExamples[reference.key], `Missing example: ${reference.key}`);
    referenced.add(reference.key);
  }
}
assert.deepEqual([...referenced].sort(), Object.keys(programmingExamples).filter(key => key.startsWith('iter')).sort(), "Every iterator fixture must appear in its current lesson");

const extraChecks = {
  basicsFunctions: "assert mean([0]) == 0\nassert mean([-2, 2]) == 0\n",
  basicsDefaults: "assert add_reading(1) is not add_reading(1)\n",
  basicsProject: "assert summarize(['0','24']) == {'count': 2, 'mean': 12.0}\ntry:\n    summarize([' ', ''])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('empty input must fail')\n",
  basicsPractice: "assert best_model({'a': -3, 'b': -1}) == 'b'\n",
  oopInstances: "assert morning.values is not evening.values\nassert evening.mean() is None\n",
  oopValidated: "for bad in [True, '18', None, float('inf')]:\n    try:\n        log.add(bad)\n    except (TypeError, ValueError):\n        pass\n    else:\n        raise AssertionError('invalid value accepted')\nassert len(log) == 2\n",
  oopDataclass: "assert a.tags is not b.tags\n",
  iterBatches: "assert list(batches((x for x in range(4)), 2)) == [(0, 1), (2, 3)]\nfor bad in [False, -1, 1.5]:\n    try:\n        list(batches([], bad))\n    except ValueError:\n        pass\n    else:\n        raise AssertionError('invalid batch size accepted')\n",
  iterCustom: "assert iter(cursor) is cursor\nassert list(CountdownSource(0)) == []\nassert iter(source) is not iter(source)\n",
  iterPractice: "assert list(take_until_missing([None, 1])) == []\nassert list(running_mean([-2, 2])) == [-2.0, 0.0]\n",
};
let count = 0;
for (const [id, example] of Object.entries(programmingExamples)) {
  const root = path.resolve(os.tmpdir());
  const dir = fs.mkdtempSync(path.join(root, "python-teaching-check-"));
  // Only remove the dedicated test directory we just created directly under the temp root.
  assert.equal(path.dirname(path.resolve(dir)), root);
  try {
    for (const [filename, code] of Object.entries(example.files || {})) {
      assert.equal(path.basename(filename), filename);
      fs.writeFileSync(path.join(dir, filename), code);
    }
    const filename = example.filename || "lesson.py";
    assert.equal(path.basename(filename), filename);
    fs.writeFileSync(path.join(dir, filename), example.code + "\n" + (extraChecks[id] || ""));
    const result = spawnSync(python, ["-B", filename], {
      cwd: dir, encoding: "utf8", timeout: 15000,
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: "" },
    });
    assert.equal(result.status, 0, `${id}: ${result.stderr || result.error}`);
    assert.equal(result.stdout.replace(/\r\n/g, "\n").trimEnd(), example.output.trimEnd(), `${id}: output mismatch`);
    if (id === "basicsProject") {
      const imported = spawnSync(python, ["-B", "-c", "import report"], { cwd: dir, encoding: "utf8", timeout: 15000 });
      assert.equal(imported.status, 0, imported.stderr);
      assert.equal(imported.stdout, "", "Importing report must not execute main()");
    }
    count++;
    console.log(`${id}: output and applicable edge checks passed`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}
console.log(`${count} independently runnable examples verified across 3 lessons.`);
