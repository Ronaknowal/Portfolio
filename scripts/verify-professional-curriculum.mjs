import assert from "node:assert/strict";
import fs from "node:fs";
import { createHash } from "node:crypto";
import { trackDefinitions } from "../src/learn/data/track-definitions.js";
import { topicCatalogue } from "../src/learn/data/curriculum/topic-catalogue.js";
import { slugify } from "../src/learn/data/topic-id.js";
import { learningPaths, getLearningRoute } from "./lib/authoring-curriculum.mjs";
import { systemDesignCoverage } from "../src/learn/data/curriculum/system-design-coverage.js";
import { quantitativeTradingCoverage } from "../src/learn/data/curriculum/quantitative-trading-coverage.js";

const baseline = JSON.parse(fs.readFileSync("docs/curriculum/professional-modules-baseline.json", "utf8"));
const manifest = JSON.parse(fs.readFileSync("src/learn/data/lesson-manifest.json", "utf8"));
const originalIds = new Set(baseline.topicIds);
const selectedIds = ["quantitative-finance", "system-design"];
const selectedTracks = selectedIds.map(id => trackDefinitions.find(track => track.id === id));
for (const [moduleId, coverage] of [["system-design", systemDesignCoverage], ["quantitative-finance", quantitativeTradingCoverage]]) {
  const titles = new Set(selectedTracks.find(track => track.id === moduleId).sections.flatMap(section => section.topics.map(topic => topic.title)));
  assert.deepEqual(new Set(Object.keys(coverage)), titles, `Every professional topic needs explicit concept ownership: ${moduleId}`);
  for (const [title, subtopics] of Object.entries(coverage)) {
    assert.ok(titles.has(title), `Orphaned concept coverage: ${title}`);
    assert.ok(subtopics.length > 0 && subtopics.every(term => typeof term === "string" && term.trim()), `Invalid coverage: ${title}`);
    assert.equal(new Set(subtopics).size, subtopics.length, `Duplicate subtopics: ${title}`);
    assert.deepEqual(topicCatalogue[slugify(title)].subtopics, subtopics, `Lost coverage: ${title}`);
  }
}
for (const id of originalIds) assert.ok(topicCatalogue[id], `Original topic removed: ${id}`);
const originalFinance = baseline.modules.find(track => track.id === "quantitative-finance").sections.flatMap(section => section.titles);
const currentFinance = selectedTracks[0].sections.flatMap(section => section.topics.map(topic => topic.title));
for (const title of originalFinance) assert.ok(currentFinance.includes(title), `Retained finance topic missing: ${title}`);

for (const track of selectedTracks) {
  const topics = track.sections.flatMap(section => section.topics);
  const positions = new Map(topics.map((topic, index) => [slugify(topic.title), index]));
  assert.equal(positions.size, topics.length, `Duplicate topic in ${track.id}`);
  for (const section of track.sections) assert.ok(section.topics.length >= 5, `Fragmented section: ${section.name}`);
  for (const topic of topics) {
    const id = slugify(topic.title);
    for (const prerequisite of topicCatalogue[id].prerequisiteIds) {
      assert.ok(topicCatalogue[prerequisite], `Unresolved prerequisite: ${prerequisite}`);
      if (positions.has(prerequisite)) assert.ok(positions.get(prerequisite) < positions.get(id), `Forward local prerequisite: ${topic.title} -> ${prerequisite}`);
    }
    if (!originalIds.has(id)) {
      assert.ok(topic.blueprint?.sequence.length >= 4, `Incomplete new scope: ${id}`);
      assert.ok(topic.blueprint?.practice.success && topic.blueprint?.visual.question, `Missing teaching brief: ${id}`);
    }
  }
}

for (const pathId of ["quant-trading", "system-design-engineer", "full-curriculum"]) {
  const path = learningPaths.find(value => value.id === pathId);
  assert.ok(path, `Missing path: ${pathId}`);
  const route = getLearningRoute(path);
  for (const id of route.topicIds) for (const prerequisite of topicCatalogue[id].prerequisiteIds) {
    assert.ok(route.topicIds.includes(prerequisite), `Missing route prerequisite: ${id}`);
  }
  for (const moduleId of pathId === "full-curriculum" ? selectedIds : [pathId === "quant-trading" ? selectedIds[0] : selectedIds[1]]) {
    const expected = selectedTracks.find(track => track.id === moduleId).sections.flatMap(section => section.topics.map(topic => slugify(topic.title)));
    assert.deepEqual(route.navigationGroups.find(group => group.id === moduleId).topicIds, expected);
  }
}

// This optional flag validates this planning increment, not future lesson work.
// Later authorized publication or phase checkpoints legitimately change hashes.
if (process.argv.includes("--check-planning-boundary")) {
  for (const [filename, expected] of Object.entries(baseline.unchangedFiles)) {
    const actual = createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
    assert.equal(actual, expected, `Planning changed lesson delivery/publication: ${filename}`);
  }
  for (const previous of baseline.modules.filter(track => track.id !== "quantitative-finance")) {
    const current = trackDefinitions.find(track => track.id === previous.id);
    assert.deepEqual(current.sections.map(section => ({ name: section.name, titles: section.topics.map(topic => topic.title) })), previous.sections, `Unrelated module altered: ${previous.id}`);
  }
  for (const id of Object.keys(topicCatalogue).filter(id => !originalIds.has(id))) {
    assert.ok(!manifest[id], `Planning published a new lesson: ${id}`);
  }
}

if (process.argv.includes("--write-syllabi")) {
  for (const track of selectedTracks) {
    const lines = [
      `# ${track.title}: detailed syllabus`, "",
      "Generated from the live catalogue by `node scripts/verify-professional-curriculum.mjs --write-syllabi`. Do not edit this derived list independently.", "",
      "See [scope, role routes, research and authoring rules](PROFESSIONAL-TRADING-SYSTEM-DESIGN-PLAN.md). Listed order is module reading order; specialist branches are deliberate optional depth. Briefs are plans, not completed research/write or implementation checkpoints.", "",
    ];
    let position = 0;
    for (const section of track.sections) {
      lines.push(`## ${section.name}`, "");
      for (const topic of section.topics) {
        const id = slugify(topic.title);
        const record = topicCatalogue[id];
        const blueprint = topic.blueprint;
        lines.push(`### ${++position}. ${topic.title}`, "", `- Level: ${topic.level}; ${originalIds.has(id) ? "retained/shared topic" : "new planned topic"}; ${manifest[id] ? "existing published lesson (not re-reviewed by this expansion)" : "planned lesson"}.`, `- Stable ID: \`${id}\`.`);
        lines.push(`- Prerequisites: ${record.prerequisiteIds.map(value => topicCatalogue[value].title).join("; ") || "No topic-level prerequisites; begin here."}`);
        if (topic.subtopics?.length) lines.push(`- Named concept coverage (planned): ${topic.subtopics.join("; ")}.`);
        if (blueprint) {
          lines.push(`- Scope: ${blueprint.sequence.join("; ")}.`, `- Investigation: ${blueprint.visual.type} — ${blueprint.visual.question} ${blueprint.visual.interaction}.`, `- Practice: ${blueprint.practice.task}. Success: ${blueprint.practice.success}.`);
        } else {
          lines.push("- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.");
        }
        lines.push("");
      }
    }
    const filename = track.id === "system-design" ? "SYSTEM-DESIGN-SYLLABUS.md" : "QUANTITATIVE-TRADING-SYLLABUS.md";
    fs.writeFileSync(`docs/curriculum/${filename}`, lines.join("\n"));
  }
}

console.log(JSON.stringify({
  status: "passed",
  originalTopicsPreserved: originalIds.size,
  originalFinanceTopicsPreserved: originalFinance.length,
  totalTopics: Object.keys(topicCatalogue).length,
  newTopics: Object.keys(topicCatalogue).filter(id => !originalIds.has(id)).length,
  modules: selectedTracks.map(track => ({ id: track.id, topics: track.sections.reduce((sum, section) => sum + section.topics.length, 0), sections: track.sections.length })),
  planningBoundaryChecked: process.argv.includes("--check-planning-boundary"),
}, null, 2));
