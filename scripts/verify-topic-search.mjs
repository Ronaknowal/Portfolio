import assert from "node:assert/strict";
import { allTopicsOrdered } from "../src/learn/data/generated/navigation.js";
import { topicCatalogue } from "../src/learn/data/curriculum/topic-catalogue.js";
import { createTopicSearchIndex, findTopicMatch, normalizeTopicQuery } from "../src/learn/data/topic-search.js";

const index = createTopicSearchIndex(allTopicsOrdered);
const examples = [
  ["HyperLogLog", "hyperloglog-hll-approximate-distinct-counting"],
  ["HLL++", "hyperloglog-hll-approximate-distinct-counting"],
  ["sNoWfLaKe", "distributed-ids-ordering-uniqueness-guarantees"],
  ["t digest", "streaming-quantiles-kll-t-digest-reservoir-sampling"],
  ["rendezvous", "sharding-consistent-hashing-hot-keys-rebalancing"],
  ["S2", "design-studio-maps-geospatial-search-ride-dispatch"],
  ["Newey West", "financial-inference-hac-errors-bootstrap-multiple-testing"],
  ["Ledoit Wolf", "covariance-estimation-shrinkage-multi-factor-risk-models"],
  ["SABR", "local-volatility-stochastic-volatility-jumps-rough-models"],
  ["Python", "python-basics-types-control-flow-functions-modules"],
];
for (const [query, id] of examples) {
  assert.ok(index.has(id), `Unknown expected topic: ${id}`);
  assert.ok(findTopicMatch(index.get(id), normalizeTopicQuery(query)).matches, `Missing search result: ${query}`);
}
for (const topic of allTopicsOrdered) {
  assert.deepEqual(topic.subtopics, topicCatalogue[topic.id].subtopics, `Stale search scope: ${topic.id}`);
  assert.equal(findTopicMatch(index.get(topic.id), "").matches, true);
  assert.equal(findTopicMatch(index.get(topic.id), "zzzz-no-such-concept-zzzz").matches, false);
  assert.ok(!("blueprint" in topic), `Full authoring data leaked into navigation: ${topic.id}`);
}
const named = findTopicMatch(index.get("distributed-ids-ordering-uniqueness-guarantees"), "snowflake");
assert.equal(named.subtopic, "Snowflake IDs");
assert.notEqual(normalizeTopicQuery("HLL++"), normalizeTopicQuery("HLL"));
console.log(`PASS: ${examples.length} title/subtopic searches, complete metadata parity, empty/no-result queries and compact navigation for ${allTopicsOrdered.length} topics.`);
