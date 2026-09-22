import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { topicCatalogue } from "../src/learn/data/curriculum/topic-catalogue.js";
import { trackDefinitions } from "../src/learn/data/track-definitions.js";
import { learningPaths, getLearningRoute } from "./lib/authoring-curriculum.mjs";
import { getDomainGuidance } from "../src/learn/data/curriculum/domain-guidance.js";
import { slugify } from "../src/learn/data/topic-id.js";
import { readTopicAuthoringNotes } from "./topic-authoring-notes.mjs";
import { createHash } from "node:crypto";
import { deliveryLedgerPath, validateDeliveryLedger, getDeliveryState, assertDeliveryRequest, getImplementationReviewOverride } from './lib/lesson-delivery.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const deliveryLedger = JSON.parse(fs.readFileSync(path.join(root, deliveryLedgerPath), 'utf8'));
validateDeliveryLedger(deliveryLedger, new Set(Object.keys(topicCatalogue)));
const published = new Set(Object.keys(JSON.parse(fs.readFileSync(path.join(root, "src/learn/data/lesson-manifest.json"), "utf8"))));
// Recorded review evidence, not a status inferred merely from publication or a brief.
const pythonDataFoundationsReviewed = new Set([
  'python-basics-types-control-flow-functions-modules',
  'numpy-arrays-broadcasting-vectorization',
  'scientific-file-formats-schemas-reliable-data-i-o',
  'sql-relational-data-transactions-for-ml',
  'object-oriented-programming-in-python',
]);
const analysisWorkflowReviewed = new Set(['pandas-data-wrangling-joins-grouping','matplotlib-scientific-plotting','git-github-collaborative-version-control']);
const systemsStructuresReviewed = new Set(['os-processes-virtual-memory-isolation','arrays-strings-hash-maps','linked-lists-stacks-queues']);
const programmingReliabilityReviewed = new Set(['iterators-iterables-generators','decorators-context-managers','testing-debugging-dependency-management','reproducible-notebooks-experiment-structure','code-documentation-type-hints-api-design','bash-scripting-command-line-automation','threads-concurrency-locks-deadlocks']);
// Add an ID only after its final lesson/model/native/browser record is complete.
// A registered draft or a prepared practice set is not implementation review.
const dsaCoreStructuresReviewed = new Set(['trees-binary-search-trees','heaps-priority-queues-tries','graphs-representations-bfs-dfs']);
// Current rollout evidence is per topic and source-versioned. A publication or
// an old review record cannot make an edited lesson automatically reviewed.
const foundationsProgress = JSON.parse(fs.readFileSync(path.join(root, 'docs/teaching/dsa-math-foundations-progress.json'), 'utf8'));
const currentSourceReviews = new Map(foundationsProgress.topics.filter(topic => {
  if (topic.status !== 'implementation-reviewed' || !topic.reviewedFiles) return false;
  return Object.entries(topic.reviewedFiles).every(([filename, expectedHash]) => {
    const absolute = path.join(root, filename);
    return fs.existsSync(absolute) && createHash('sha256').update(fs.readFileSync(absolute)).digest('hex') === expectedHash;
  });
}).map(topic => [topic.id, topic.verificationRecord]));
// The active ML increment uses the same final, exact-source review contract.
// Author checks or independent review alone do not stand in for integration.
for (const progressPath of ['docs/teaching/classical-ml-supervised-progress.json', 'docs/teaching/classical-ml-unsupervised-progress.json']) {
  const classicalMlProgress = JSON.parse(fs.readFileSync(path.join(root, progressPath), 'utf8'));
  for (const topic of classicalMlProgress.topics) {
    if (topic.status !== 'implementation-reviewed' || !topic.reviewedFiles) continue;
    const current = Object.entries(topic.reviewedFiles).every(([filename, expectedHash]) => {
      const absolute = path.join(root, filename);
      return fs.existsSync(absolute) && createHash('sha256').update(fs.readFileSync(absolute)).digest('hex') === expectedHash;
    });
    if (current) currentSourceReviews.set(topic.id, topic.verificationRecord);
  }
}
// Targeted follow-ups in the previously reviewed first five DSA topics have
// their own exact-source integration evidence. An edited source invalidates it.
const extensionsPath = path.join(root, 'docs/teaching/evidence/dsa-math-foundations-complete-integration.json');
const extensionReviews = fs.existsSync(extensionsPath)
  ? JSON.parse(fs.readFileSync(extensionsPath, 'utf8')).extensionReviews : [];
for (const review of extensionReviews) {
  systemsStructuresReviewed.delete(review.topicId);
  dsaCoreStructuresReviewed.delete(review.topicId);
  if (Object.entries(review.reviewedFiles).every(([filename, expectedHash]) => {
    const absolute = path.join(root, filename);
    return fs.existsSync(absolute) && createHash('sha256').update(fs.readFileSync(absolute)).digest('hex') === expectedHash;
  })) currentSourceReviews.set(review.topicId, review.verificationRecord);
}
// Future phase-two completion uses the same source-identity rule as historical
// ledgers. A completed manuscript alone never becomes implementation review.
for (const [id, entry] of Object.entries(deliveryLedger.topics)) {
  if (!entry.legacyReview && getDeliveryState(root, entry).implementation === 'complete') {
    currentSourceReviews.set(id, entry.record);
  }
}
const topics = Object.values(topicCatalogue).map((topic) => ({
  ...topic,
  publicationStatus: published.has(topic.id) ? "published" : "planned",
  teachingReview: topic.id === "linux-basics-filesystems-processes" ? "user-approved-reference" : currentSourceReviews.has(topic.id) || pythonDataFoundationsReviewed.has(topic.id) || analysisWorkflowReviewed.has(topic.id) || systemsStructuresReviewed.has(topic.id) || programmingReliabilityReviewed.has(topic.id) || dsaCoreStructuresReviewed.has(topic.id) ? "implementation-reviewed-user-acceptance-pending" : "individual-review-required",
  teachingReviewRecord: topic.id === "linux-basics-filesystems-processes" ? "PROGRAMMING-REWRITE-LINUX.md" : currentSourceReviews.get(topic.id) || (dsaCoreStructuresReviewed.has(topic.id) ? "DSA-CORE-STRUCTURES-IMPLEMENTATION.md" : programmingReliabilityReviewed.has(topic.id) ? "PROGRAMMING-MODULE-COMPLETION.md" : pythonDataFoundationsReviewed.has(topic.id) ? "FIRST-FIVE-REIMPLEMENTATION.md" : analysisWorkflowReviewed.has(topic.id) ? "NEXT-THREE-REIMPLEMENTATION.md" : systemsStructuresReviewed.has(topic.id) ? "SYSTEMS-STRUCTURES-IMPLEMENTATION.md" : undefined),
  domainStrategy: getDomainGuidance(topic.trackId).strategy,
})).map(topic => {
  const entry = deliveryLedger.topics[topic.id];
  const delivery = getDeliveryState(root, entry, topic.teachingReview !== 'individual-review-required');
  return { ...topic, ...getImplementationReviewOverride(entry, delivery), delivery };
});

const selected = process.argv.indexOf("--topic");
const requestedWork = process.argv.indexOf('--work');
if (requestedWork !== -1 && selected === -1) throw new Error('--work requires --topic.');
if (selected !== -1) {
  const query = process.argv[selected + 1];
  const topic = topics.find((item) => item.id === query || item.title === query || item.id === slugify(query || ""));
  if (!topic) throw new Error(`Topic not found: ${query}`);
  if (requestedWork !== -1) assertDeliveryRequest(process.argv[requestedWork + 1], topic.delivery);
  // Named concepts are authoring obligations even when an older published body
  // or a compact starting brief does not yet contain their explanations.
  if (topic.subtopics?.length) topic.coverageInstruction = "Assess every named subtopic during design and writing. Teach its mechanism, assumptions, useful application and failure boundaries at appropriate depth, or explicitly record a justified ownership change. Search labels alone are not coverage evidence. Preserve the user's delivery-phase boundary.";
  const depthRecord = `docs/teaching/implementation-depth/${topic.id}.md`;
  const implementationDepth = {
    standard: 'LESSON-TEACHING-STANDARD.md#build-the-mechanism-then-control-the-library',
    review: 'docs/teaching/IMPLEMENTATION-DEPTH-REVIEW.md',
    preparedContentReview: topic.delivery.implementation === 'not-started' && fs.existsSync(path.join(root, `docs/teaching/drafts/${topic.id}/lesson.md`)) ? 'docs/teaching/implementation-depth/PREPARED-WRITING-REVISION.md' : null,
    topicRecord: fs.existsSync(path.join(root, depthRecord)) ? depthRecord : null,
    instruction: 'Plan and write complete explained scratch and idiomatic library/tool routes for every core computational outcome before completing content. Record exact reuse owners, abstraction boundaries, efficient/stable algorithms, complexity, matched comparisons and extension practice. Known missing core code or teaching cannot be left as a finish-agent TODO. Read the review and preparedContentReview where supplied; an old phase completion or missing topic record is not a depth certification. Defer full execution, independent implementation review and browser work during content-first delivery, with truthful execution status.',
  };
  console.log(JSON.stringify({ topic, requestedWork: requestedWork === -1 ? undefined : process.argv[requestedWork + 1], deliveryLedger: deliveryLedgerPath, domainGuidance: getDomainGuidance(topic.trackId), authoringNotes: readTopicAuthoringNotes(root, topic.id), implementationDepth, authoringContract: "Read LESSON-AUTHORING-HANDOFF.md, the teaching standard's delivery modes, docs/teaching/TOPIC-DESIGN-BRIEF.md and the returned authoringNotes and implementationDepth guidance. Follow the user's full/content-first/finish scope and topic.delivery; finishing requires a current complete content checkpoint. Revisit coverage/title and useful applications while writing. A brief, published old body or completed draft does not certify the new implementation." }, null, 2));
} else {
  const out = path.join(root, "docs/curriculum");
  fs.mkdirSync(out, { recursive: true });
  const modules = trackDefinitions.map((track) => {
    const rows = topics.filter((topic) => topic.trackIds.includes(track.id));
    return { id: track.id, title: track.title, topics: rows.length, published: rows.filter((t) => t.publicationStatus === "published").length, topicBriefs: rows.filter((t) => t.blueprint).length, prerequisitesRecorded: rows.filter((t) => t.prerequisiteStatus === "recorded").length, guidance: getDomainGuidance(track.id) };
  });
  const paths = learningPaths.map((route) => {
    const resolved = getLearningRoute(route);
    return { id: route.id, title: route.title, trackIds: route.trackIds, resolvedModuleIds: resolved.navigationGroups.map(group => group.id), moduleCount: resolved.moduleCount, focusTrackIds: route.focusTrackIds, coreTrackIds: route.coreTrackIds, backgroundMode: route.backgroundMode || "foundations", milestones: route.milestones || [], topicIds: resolved.topicIds };
  });
  const counts = { modules: modules.length, uniqueTopics: topics.length, published: topics.filter((t) => t.publicationStatus === "published").length, topicBriefs: topics.filter((t) => t.blueprint).length, prerequisiteReviewsRecorded: topics.filter((t) => t.prerequisiteStatus === "recorded").length, guidedPaths: paths.length, contentComplete: topics.filter(t => t.delivery.content === 'complete').length, implementationComplete: topics.filter(t => t.delivery.implementation === 'complete').length };
  const inventory = { scopeReviewedOn: "2026-09-09", generatedOn: new Date().toISOString().slice(0, 10), counts, meanings: { topicBrief: "Topic-specific scope, sequence, representation, practice and research starting points; full lesson not implied", individualDesignRequired: "Domain guidance available; individual outcomes, dependencies and lesson design still required", publication: "Content is registered, not necessarily verified or approved", prerequisiteOrder: "Recorded graphs are checked and supporting topics included; reading follows module syllabus order, with prerequisite review links. Unknown older dependencies are not certified." }, modules, paths, topics };
  fs.writeFileSync(path.join(out, "curriculum-inventory.json"), JSON.stringify(inventory, null, 2) + "\n");
  const lines = ["# Curriculum coverage inventory", "", "Generated from the live catalogue by `node scripts/build-curriculum-inventory.mjs`. Scope review: 9 September 2026. Regenerate after catalogue changes; this is a status report, not teaching policy.", "", `**${counts.uniqueTopics} unique topics · ${counts.modules} modules · ${counts.published} registered published lessons · ${counts.topicBriefs} topic-specific briefs · ${counts.guidedPaths} guided paths.**`, "", `**${topics.length - counts.topicBriefs} older topics still need individual design.** Every module has domain guidance; this does not make those older topics fully planned or fact-checked. ${counts.prerequisiteReviewsRecorded} topics have recorded prerequisite reviews; the remaining edges need individual review. Linux is the user-approved teaching reference.`, "", "See [the research and scope plan](../../LEARNING-CURRICULUM-PLAN.md), [authoring handoff](../../LESSON-AUTHORING-HANDOFF.md), and [full machine-readable inventory](curriculum-inventory.json).", "", "## Module coverage", "", "| Module | Topics | Published | Individual briefs | Prerequisite reviews recorded |", "| --- | ---: | ---: | ---: | ---: |"]; 
  for (const module of modules) lines.push(`| ${module.title} | ${module.topics} | ${module.published} | ${module.topicBriefs} | ${module.prerequisitesRecorded} |`);
  lines.push("", "Counts within modules may include shared topics. The headline counts each stable topic ID once.", "", "## Delivery phases", "", `**${counts.contentComplete} current content checkpoints complete · ${counts.implementationComplete} current implementations complete.**`, "", "[The delivery ledger](../teaching/lesson-delivery-progress.json) preserves historical completion and tracks the current revision's content and implementation separately. Source changes can make a recorded completion stale; these counts do not silently approve changed versions. Publication and user acceptance remain separate. Use the topic CLI's delivery field before continuing a phase.", "", "## Individual authoring inventory", "", "`brief` means a topic-specific starting plan. `design needed` means use the domain playbook, research the topic and complete its individual design before writing. Published content also needs an individual quality review unless the handoff records one.", "");
  for (const module of modules) {
    lines.push(`### ${module.title}`, "", `**Module anchor:** ${module.guidance.exampleAnchor}`, "", `**Teaching strategy:** ${module.guidance.flow.join(" → ")}.`, "", `**Practice:** ${module.guidance.practice}`, "", `**Verification:** ${module.guidance.verification}`, "", "| Topic | Level | Content | Design |", "| --- | --- | --- | --- |");
    const definition = trackDefinitions.find(track => track.id === module.id);
    const orderedIds = definition.sections.flatMap(section => section.topics.map(value => slugify(typeof value === 'string' ? value : value.title)));
    for (const id of orderedIds) { const topic = topics.find(t => t.id === id); lines.push(`| ${topic.title.replaceAll("|", "\\|")} | ${topic.level} | ${topic.publicationStatus} | ${topic.blueprint ? "brief" : "design needed"} |`); }
    lines.push("");
  }
  fs.writeFileSync(path.join(out, "CURRICULUM-INVENTORY.md"), lines.join("\n") + "\n");
  console.log(JSON.stringify(counts, null, 2));
}
