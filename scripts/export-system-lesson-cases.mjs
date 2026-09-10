import fs from "node:fs";
import assert from "node:assert/strict";
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import { gitExamples } from "../src/learn/data/git-examples.js";
import { linuxExamples } from "../src/learn/data/linux-command-examples.js";
import { gitTrace, permissionModel } from "../src/learn/data/system-lesson-models.js";

const groups = [
  ["git-github-collaborative-version-control", gitExamples],
  ["linux-basics-filesystems-processes", linuxExamples],
];
const cases = [];
for (const [slug, examples] of groups) {
  const source = fs.readFileSync("src/learn/data/topics/" + slug + ".jsx", "utf8");
  const refs = (await collectLessonExamples('src/learn/data/topics/'+slug+'.jsx')).filter(reference=>Object.hasOwn(examples,reference.key)).map(reference=>reference.key);
  assert.deepEqual(refs.sort(), Object.keys(examples).sort(), slug + " displayed examples");
  for (const [id, example] of Object.entries(examples)) cases.push({ id: slug + "/" + id, ...example });
}
const initial = gitExamples.first.code.split("git status --short")[0] + 'git add report.txt\ngit commit -qm "Initial"\n';
cases.push({ id: "git-explorer-states", code: initial + gitTrace.map(step => step.command + "\ngit show HEAD:report.txt\ngit show :report.txt\ncat report.txt").join("\n"), output: gitTrace.flatMap(step => step.versions.map(version => "version " + version)).join("\n") });
cases.push({ id: "practice/restage-version-three", code: gitExamples.staging.code.replace("git status --short", "git add report.txt\ngit status --short"), output: "M  report.txt\nversion 1\nversion 3\nversion 3\nversion 3" });
cases.push({ id: "practice/remote-divergence", code: gitExamples.remote.code.replace("git fetch -q origin", "printf 'local note\\n' > local.txt\ngit add local.txt\ngit commit -qm 'Local note'\ngit fetch -q origin").replace("git merge --ff-only -q origin/main", "if git merge --ff-only -q origin/main 2> \"$lab/refusal.log\"; then\n exit 1\nelse\n printf 'fast-forward refused\\n'\nfi"), output: "version 1\nversion 2\n1\t1\nfast-forward refused\nversion 1" });
cases.push({ id: "practice/third-warning", code: linuxExamples.search.code.replace("grep -n -F 'WARN'", "printf 'WARN reviewed\\n' >> logs/run.log\ngrep -n -F 'WARN'"), output: "2:WARN slow\n4:WARN retry\n5:WARN reviewed\n3\ngrep status: 1\nlogs/run.log" });
// Exhaust the displayed bit combinations independently of the UI rendering.
for (const mode of ["600", "640", "700", "750", "755"]) for (const [i, subject] of ["owner", "group", "other"].entries()) {
  const bits = Number(mode[i]).toString(2).padStart(3, "0").split("").map(bit => bit === "1");
  assert.deepEqual(permissionModel(mode, subject, false).map(row => row[1]), bits);
  assert.deepEqual(permissionModel(mode, subject, true).map(row => row[1]), [bits[0], bits[2], bits[1] && bits[2]]);
}
fs.mkdirSync("scratch", { recursive: true });
fs.writeFileSync("scratch/system-lesson-cases.json", JSON.stringify(cases));
console.log("17 displayed shell examples, staging trace and 3 practice variants exported; 30 permission-model combinations checked.");
