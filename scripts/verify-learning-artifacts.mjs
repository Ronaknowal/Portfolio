import assert from "node:assert/strict";
import fs from "node:fs";
import { execFileSync } from "node:child_process";
import { topicCatalogue } from "../src/learn/data/curriculum/topic-catalogue.js";
import { topicMap, allTopicsOrdered, allTopicIds, categories } from "../src/learn/data/catalogue.js";
import { tracks } from "../src/learn/data/tracks.js";
import * as browserNavigation from "../src/learn/data/curriculum.js";
import * as authoringNavigation from "./lib/authoring-curriculum.mjs";

execFileSync(process.execPath, ["scripts/generate-learning-artifacts.mjs", "--check"], { stdio: "inherit" });
const manifest = JSON.parse(fs.readFileSync("src/learn/data/lesson-manifest.json", "utf8"));
assert.deepEqual(allTopicIds, Object.keys(topicCatalogue), "Compact catalogue preserves every ID and its order");
assert.deepEqual(tracks, authoringNavigation.tracks, "Module contents and memberships are unchanged");
assert.deepEqual(categories, tracks.map(track => ({ id: track.id, label: track.title })));
assert.deepEqual(allTopicsOrdered.filter(topic => topic.status === "published").map(topic => topic.id).sort(), Object.keys(manifest).sort());
for (const topic of allTopicsOrdered) {
  const authored = topicCatalogue[topic.id];
  assert.deepEqual(topic.prerequisiteIds, authored.prerequisiteIds, topic.id);
  assert.deepEqual(topic.trackIds, authored.trackIds, topic.id);
  assert.equal(topic.depth, authored.blueprint?.depth ?? null, topic.id);
  assert.equal(topic.hasOutline, Boolean(authored.blueprint && !manifest[topic.id]), topic.id);
  assert.ok(!("content" in topic) && !("blueprint" in topic), "Navigation cannot contain lesson bodies or full authoring plans");
  if (topic.hasOutline) {
    const outline = JSON.parse(fs.readFileSync(`src/learn/data/generated/outlines/${topic.id}.json`, "utf8"));
    const blueprint = authored.blueprint;
    assert.deepEqual(outline, { summary: blueprint.summary, outcomes: blueprint.outcomes, sequence: blueprint.sequence,
      visual: { question: blueprint.visual.question, interaction: blueprint.visual.interaction },
      practice: { task: blueprint.practice.task, success: blueprint.practice.success }, sources: blueprint.sources });
  }
}
for (const path of [...authoringNavigation.learningPaths, ...tracks.map(track => [track.id])]) {
  assert.deepEqual(browserNavigation.getLearningRoute(path), authoringNavigation.getLearningRoute(path), "Browser and authoring tools resolve exactly the same learning sequence");
}
assert.equal(Object.keys(topicMap).length, allTopicIds.length);
assert.ok(!fs.existsSync("src/learn/data/topics/index.js"), "Do not restore an eager lesson barrel");
const imports = fs.readFileSync("src/learn/data/generated/lesson-imports.js", "utf8");
assert.equal((imports.match(/\(\) => import\(/g) || []).length, Object.keys(manifest).length);
console.log(`PASS: ${allTopicIds.length} topic records, ${tracks.length} modules, ${Object.keys(manifest).length} publication mappings and every guided/module route match authoring sources; body and outline loading remain separate.`);
