import assert from "node:assert/strict";
import fs from "node:fs";
import { linuxExamples } from "../src/learn/data/linux-command-examples.js";
import { investigation } from "../src/learn/data/linux-practice-examples.js";
import { resolveLinuxPath } from "../src/learn/data/linux-path-model.js";
import { linuxStreamModel } from "../src/learn/data/linux-stream-model.js";
import { pathPermissionModel } from "../src/learn/data/linux-permission-model.js";
import { transitionProcess } from "../src/learn/data/linux-process-model.js";
import { permissionModel } from "../src/learn/data/system-lesson-models.js";

const pathCases = [
  ["../../reports", "cd", true, "/project/reports"],
  ["/project/reports", "cd", true, "/project/reports"],
  ["reports", "cd", false, "ENOENT"],
  ["./run 1.csv", "locate", true, "/project/data/raw/run 1.csv"],
  ["./run 1.csv", "cd", false, "ENOTDIR"],
  ["./run 1.csv/..", "locate", false, "ENOTDIR"],
  ["missing/..", "cd", false, "ENOENT"],
  ["/../../", "cd", true, "/"],
  ["/project//data/./raw/", "cd", true, "/project/data/raw"],
  ["run 1.csv/", "locate", false, "ENOTDIR"],
  ["", "cd", false, "ENOENT"],
];
for (const [path, operation, ok, expected] of pathCases) {
  const result = resolveLinuxPath({ path, operation });
  assert.equal(result.ok, ok, path);
  assert.equal(ok ? result.result : result.error, expected, path);
  assert.equal(result.cwd, ok && operation === "cd" ? expected : "/project/data/raw");
  for (const step of result.steps.slice(0, -1)) assert.equal(step.cwd, "/project/data/raw");
}
for (let mask = 0; mask < 8; mask++) {
  const directoryRead = !!(mask & 4), directorySearch = !!(mask & 2), fileRead = !!(mask & 1);
  const model = pathPermissionModel({ directoryRead, directorySearch, fileRead });
  assert.equal(model.canReadFile, (mask & 3) === 3);
  assert.equal(model.canListNames, directoryRead);
  assert.equal(model.fileReadChecked, directorySearch);
  assert.equal(model.gates.file, !directorySearch ? "not-reached" : fileRead ? "passed" : "blocked");
}
for (const [index, subject] of ["owner", "group", "other"].entries()) {
  const bits = [0, 4, 7][index];
  assert.deepEqual(permissionModel("047", subject, false).map(row => row[1]), [!!(bits & 4), !!(bits & 2), !!(bits & 1)]);
}
let state = "ready";
for (const [action, expected] of [["wait", "ready"], ["start", "running"], ["stop", "stopped"], ["terminate", "stopped"], ["resume", "running"], ["terminate", "exited"], ["wait", "collected"], ["start", "collected"]]) {
  state = transitionProcess(state, action);
  assert.equal(state, expected);
}

// Independent expected routing, not computed from the model's destinations.
const metric = "metric=18\n", warning = "warning: tiny sample\n";
const routing = {
  terminal: { stdout: metric, stderr: warning, files: {} },
  "stdout-file": { stdout: "", stderr: warning, files: { "result.txt": metric } },
  split: { stdout: "", stderr: "", files: { "result.txt": metric, "errors.txt": warning } },
  pipe: { stdout: "1\n", stderr: warning, files: {} },
  merge: { stdout: "", stderr: "", files: { "combined.txt": metric + warning } },
  // Descriptor 2 copies the outer stdout destination before stdout moves.
  reverse: { stdout: warning, stderr: "", files: { "combined.txt": metric } },
};
const streams = [];
for (const [route, expected] of Object.entries(routing)) for (const status of [0, 7]) {
  const model = linuxStreamModel(route, status);
  assert.deepEqual(model.files, expected.files);
  assert.equal(model.terminal.filter(item => item.stream !== "stderr").map(item => item.content).join(""), route === "reverse" ? "" : expected.stdout);
  assert.equal(model.terminal.filter(item => item.stream === "stderr").map(item => item.content).join(""), route === "reverse" ? warning : expected.stderr);
  assert.equal(model.shellStatus, route === "pipe" ? 0 : status);
  streams.push({ id: route + "-" + status, code: model.command, ...expected, status: route === "pipe" ? 0 : status });
}
const examples = Object.entries(linuxExamples).map(([id, example]) => ({ id, ...example }));
examples.push({ id: "investigation", ...investigation });
examples.push({ id: "investigation-transfer", code: investigation.code.replace("grep -n -F 'WARN'", "printf 'WARN reviewed\\n' >> ../logs/run.log\ngrep -n -F 'WARN'"), output: investigation.output.replace("4:WARN retry\nwarning lines: 2", "4:WARN retry\n5:WARN reviewed\nwarning lines: 3") });
fs.mkdirSync("scratch/linux-lesson-review", { recursive: true });
fs.writeFileSync("scratch/linux-lesson-review/native-cases.json", JSON.stringify({ examples, streams, pathCases }, null, 2));
console.log("PASS: 11 path cases, 8 permission gate states, class-selection checks, process transitions and 12 stream/status combinations.");
console.log("Exported 10 runnable examples/practice cases and 12 actual Bash routing cases for native-Linux verification.");
