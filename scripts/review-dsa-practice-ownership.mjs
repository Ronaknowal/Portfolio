import fs from "node:fs";
import crypto from "node:crypto";
import { tracks } from "./lib/authoring-curriculum.mjs";

const module = tracks.find((entry) => entry.id === "data-structures-algorithms");
const report = { checkedAt: new Date().toISOString(), scope: "Read-only source and practice coverage review; no fresh platform submission or complete re-verification of every statement.", topics: [], patternLocations: [] };
const patterns = /monoton(?:ic|e)[ -]stack|next[ -]greater|fast[ /-]+slow|\btortoise\b|cycle.entry|strongly connected|articulation|low.link|Kosaraju|tree DP|digit DP|interval DP|bitwise|\bXOR\b|two.s complement/i;
for (const id of module.topicIds) {
  const path = `src/learn/data/practice/${id}.js`;
  const bodyPath = `src/learn/data/topics/${id}.jsx`;
  const data = (await import(`../${path}`)).default;
  const body = fs.readFileSync(bodyPath, "utf8");
  const issues = [];
  if (data.topicId !== id) issues.push("Topic identity differs");
  if (!body.includes(`practice/${id}.js`) || !body.includes("<DsaPractice")) issues.push("Missing topic-owned practice import/render");
  const entries = [];
  for (const group of data.groups) for (const problem of group.problems) {
    for (const field of ["number", "title", "slug", "difficulty", "focus", "hint", "transfer"]) if (!problem[field]) issues.push(`${problem.number}: missing ${field}`);
    if (!["Easy", "Medium", "Hard"].includes(problem.difficulty)) issues.push(`${problem.number}: unrecognized difficulty`);
    entries.push({ number: problem.number, title: problem.title, difficulty: problem.difficulty, optional: !!group.optional, prerequisite: problem.prerequisite ?? null, url: `https://leetcode.com/problems/${problem.slug}/` });
  }
  const hash = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  report.topics.push({ id, verifiedOn: data.verifiedOn, entries, issues, files: [{ path, sha256: hash(path) }, { path: bodyPath, sha256: hash(bodyPath) }] });
  body.split(/\r?\n/).forEach((line, index) => { if (patterns.test(line)) report.patternLocations.push({ topicId: id, line: index + 1, text: line }); });
}
report.topicCount = report.topics.length;
report.entryCount = report.topics.reduce((count, topic) => count + topic.entries.length, 0);
report.uniqueLeetCodeCount = new Set(report.topics.flatMap((topic) => topic.entries.map((entry) => entry.number))).size;
report.optionalEntryCount = report.topics.flatMap((topic) => topic.entries).filter((entry) => entry.optional).length;
report.issues = report.topics.flatMap((topic) => topic.issues.map((issue) => `${topic.id}: ${issue}`));
fs.mkdirSync("scratch/dsa-practice-ownership", { recursive: true });
fs.writeFileSync("scratch/dsa-practice-ownership/source-review.json", JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ topicCount: report.topicCount, entryCount: report.entryCount, uniqueLeetCodeCount: report.uniqueLeetCodeCount, optionalEntryCount: report.optionalEntryCount, issues: report.issues, patternCandidateCount: report.patternLocations.length }, null, 2));
