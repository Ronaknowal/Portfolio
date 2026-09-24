import assert from "node:assert/strict";
import fs from "node:fs";
import { trackDefinitions } from "../src/learn/data/track-definitions.js";
import { topicCatalogue, orderWithPrerequisites } from "../src/learn/data/curriculum/topic-catalogue.js";
import { learningPaths, trackGroups, getPathTopicIds, getPathNavigationGroups } from "./lib/authoring-curriculum.mjs";
import { getDomainGuidance } from "../src/learn/data/curriculum/domain-guidance.js";
import { slugify } from "../src/learn/data/topic-id.js";
import { tracks } from "./lib/authoring-curriculum.mjs";

const errors = [];
const check = (condition, message) => { if (!condition) errors.push(message); };
const titles = new Map();
const trackIds = trackDefinitions.map((track) => track.id);
check(new Set(trackIds).size === trackIds.length, "Duplicate module ID");
for (const track of trackDefinitions) {
  check(Boolean(getDomainGuidance(track.id)), `Missing guidance: ${track.id}`);
  const ownIds = new Set();
  for (const section of track.sections) for (const value of section.topics) {
    const topic = typeof value === "string" ? { title: value, level: "foundation" } : value;
    const id = slugify(topic.title);
    check(!ownIds.has(id), `Duplicate topic in ${track.id}: ${topic.title}`);
    ownIds.add(id);
    check(!titles.has(id) || titles.get(id) === topic.title, `Different titles collide at ${id}`);
    titles.set(id, topic.title);
    check(["foundation", "intermediate", "advanced", "frontier"].includes(topic.level), `Invalid level: ${topic.title}`);
  }
}
for (const topic of Object.values(topicCatalogue)) {
  for (const id of topic.prerequisiteIds) check(Boolean(topicCatalogue[id]), `Missing prerequisite for ${topic.title}: ${id}`);
  check(new Set(topic.prerequisiteIds).size === topic.prerequisiteIds.length, `Repeated prerequisite: ${topic.title}`);
  const b = topic.blueprint;
  if (!b) continue;
  for (const key of ["summary", "reviewFocus"]) check(typeof b[key] === "string" && b[key].length > 12, `Missing ${key}: ${topic.title}`);
  for (const [key, min] of [["outcomes", 2], ["sequence", 4], ["misconceptions", 1], ["sources", 1]]) check(Array.isArray(b[key]) && b[key].length >= min && b[key].every((s) => typeof s === "string" && s.length > 0), `Incomplete ${key}: ${topic.title}`);
  check(Array.isArray(b.prerequisites), `No explicit prerequisite review: ${topic.title}`);
  for (const title of b.prerequisites || []) check(topicCatalogue[slugify(title)]?.title === title, `Prerequisite is not an exact title: ${topic.title} → ${title}`);
  check(["core", "specialist", "frontier"].includes(b.depth), `Invalid depth: ${topic.title}`);
  for (const key of ["type", "question", "interaction"]) check(Boolean(b.visual?.[key]), `Missing visual ${key}: ${topic.title}`);
  for (const key of ["task", "success"]) check(Boolean(b.practice?.[key]), `Missing practice ${key}: ${topic.title}`);
  for (const url of b.sources || []) check(/^https?:\/\/[^\s]+$/.test(url), `Invalid research URL: ${topic.title}: ${url}`);
}
const grouped = trackGroups.flatMap((group) => group.trackIds);
check(grouped.length === new Set(grouped).size && grouped.length === trackIds.length && grouped.every((id) => trackIds.includes(id)), "Module shelves do not cover the catalogue exactly once");
check(new Set(learningPaths.map((route) => route.id)).size === learningPaths.length, "Duplicate guided path ID");

const baselinePath = new URL("../docs/curriculum/pre-expansion-topic-ids.json", import.meta.url);
const baseline = JSON.parse(fs.readFileSync(baselinePath, "utf8"));
for (const id of baseline) check(Boolean(topicCatalogue[id]), `Existing topic removed or renamed: ${id}`);
const baselineSet = new Set(baseline);
for (const topic of Object.values(topicCatalogue)) {
  if (!baselineSet.has(topic.id) || ["hardware-systems", "computational-neuroscience"].includes(topic.trackId)) {
    check(Boolean(topic.blueprint), `Required individual authoring brief missing: ${topic.title}`);
  }
}
if (errors.length) throw new Error(errors.join("\n"));

const fullOrder = orderWithPrerequisites(Object.keys(topicCatalogue));
assert.equal(fullOrder.length, Object.keys(topicCatalogue).length);
function verifyModuleNavigation(ids, moduleOrder) {
  const groups = getPathNavigationGroups(ids, { moduleOrder });
  assert.equal(new Set(groups.map(group => group.id)).size, groups.length, "A module was fragmented into repeated groups");
  assert.ok(groups.length <= trackIds.length);
  assert.deepEqual([...new Set(groups.flatMap(group => group.topicIds))].sort(), [...ids].sort(), "Sidebar lost or added a route topic");
  for (const group of groups) {
    assert.ok(trackIds.includes(group.id), "Sidebar group is not a real module");
    assert.equal(new Set(group.topicIds).size, group.topicIds.length, "Repeated topic within a module");
    assert.ok(group.topicIds.length <= group.totalTopicCount);
    group.topicIds.forEach(id => assert.ok(topicCatalogue[id].trackIds.includes(group.id), "Wrong module membership"));
    const selected = new Set(group.topicIds);
    assert.deepEqual(group.topicIds, tracks.find(track => track.id === group.id).topicIds.filter(id => selected.has(id)), "Sidebar differs from the module syllabus order");
  }
  assert.deepEqual(ids, [...new Set(groups.flatMap(group => group.topicIds))], "Route differs from sidebar module order");
  return groups;
}
for (const route of learningPaths) {
  route.trackIds.forEach((id) => assert.ok(trackIds.includes(id), `Unknown path module ${id}`));
  const ids = getPathTopicIds(route);
  const positions = new Map(ids.map((id, index) => [id, index]));
  assert.equal(positions.size, ids.length);
  for (const id of ids) for (const before of topicCatalogue[id].prerequisiteIds) assert.ok(positions.has(before), `Missing supporting prerequisite in ${route.id}: ${before} → ${id}`);
  const navigation = verifyModuleNavigation(ids, route.trackIds);
  if (route.id === "full-curriculum") {
    assert.equal(ids.length, fullOrder.length);
    assert.equal(navigation.length, trackIds.length, "Full curriculum must have exactly one group per module");
    for (const group of navigation) assert.equal(group.topicIds.length, group.totalTopicCount, "A shared topic disappeared from a full module");
  }
}
// Independent small graph fixtures check insertion, sharing, missing nodes and cycles.
for (const track of trackDefinitions) {
  const ids = getPathTopicIds([track.id]);
  const positions = new Map(ids.map((id, index) => [id, index]));
  for (const id of ids) for (const before of topicCatalogue[id].prerequisiteIds) assert.ok(positions.has(before), `Missing module prerequisite: ${track.id}`);
  assert.deepEqual(ids.slice(0, tracks.find(t => t.id === track.id).topicIds.length), tracks.find(t => t.id === track.id).topicIds, "Module route must start with its own syllabus");
  verifyModuleNavigation(ids, [track.id]);
}
// These meaningful pedagogical regressions must not be hidden by valid graphs.
const gpuRoute = getPathTopicIds(learningPaths.find(p => p.id === "gpu-engineer"));
assert.ok(topicCatalogue[slugify("Category Theory & Emerging Use in ML")]);
assert.ok(!gpuRoute.includes(slugify("Category Theory & Emerging Use in ML")), "Unrelated advanced maths re-entered the GPU core route");
const llmRoute = getPathTopicIds(learningPaths.find(p => p.id === "llm-engineer"));
assert.ok(!llmRoute.includes(slugify("GPU Engineering Capstone: Verified Kernel Library")), "Unrelated kernel specialist capstone re-entered the LLM route");
const fixture = { a: { prerequisiteIds: ["b", "c"] }, b: { prerequisiteIds: ["d"] }, c: { prerequisiteIds: ["d"] }, d: { prerequisiteIds: [] } };
assert.deepEqual(orderWithPrerequisites(["a", "c"], fixture), ["d", "b", "c", "a"]);
assert.throws(() => orderWithPrerequisites(["absent"], fixture), /Unknown/);
assert.throws(() => orderWithPrerequisites(["a"], { a: { prerequisiteIds: ["b"] }, b: { prerequisiteIds: ["a"] } }), /cycle/);
console.log(`PASS: ${trackIds.length} modules, ${fullOrder.length} stable topics, ${Object.values(topicCatalogue).filter((t) => t.blueprint).length} individual briefs, ${learningPaths.length} paths. Prerequisite graphs are acyclic and included as supporting topics; reading order follows module contents. Shared memberships and all ${baseline.length} pre-expansion topic IDs preserved. Unknown older dependencies remain unreviewed.`);
