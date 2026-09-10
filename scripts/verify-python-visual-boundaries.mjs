import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { testingExamples } from "../src/learn/data/testing-examples.js";

// The new side-by-side I/O figure refers to these exact runnable examples.
// Exercise actual Mock and actual temporary files, not a JavaScript copy.
for (const name of ['testingIsolation', 'testingIntegration']) {
  const example = testingExamples[name];
  const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-c', example.code], {
    encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  assert.equal(result.status, 0, result.stderr || String(result.error));
  assert.equal(result.stdout.replaceAll('\r\n', '\n').trimEnd(), example.output);
  console.log(`${name}: displayed output and actual boundary behavior passed`);
}
