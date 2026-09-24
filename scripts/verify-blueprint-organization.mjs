import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { authoredBlueprints } from '../src/learn/data/curriculum/blueprints/index.js';
import { trackDefinitions } from '../src/learn/data/track-definitions.js';
import { slugify } from '../src/learn/data/topic-id.js';

// This is a one-time migration proof against the saved pre-migration state.
// Future intentional curriculum edits should use verify-curriculum instead.
const baselineFile = process.argv[2] || 'scratch/blueprint-organization/before.json';
const baseline = JSON.parse(fs.readFileSync(baselineFile, 'utf8'));
const directory = path.resolve('src/learn/data/curriculum/blueprints');
const titles = baseline.entries.map(entry => entry.title).sort();
assert.deepEqual(Object.keys(authoredBlueprints).sort(), titles, 'Every authored title is retained exactly once');
const ids = new Set();
for (const entry of baseline.entries) {
  assert.equal(entry.id, slugify(entry.title), 'Stable ID remains unchanged');
  assert.ok(!ids.has(entry.id), 'Unique stable topic owner');
  ids.add(entry.id);
  const module = await import(pathToFileURL(path.join(directory, entry.id + '.js')));
  assert.deepEqual(Object.keys(module), ['default'], entry.id + ' has one topic-owned default export');
  assert.deepEqual(module.default, entry.blueprint, entry.id + ' blueprint data is exactly preserved');
  assert.deepEqual(authoredBlueprints[entry.title], entry.blueprint, entry.id + ' aggregate has the same blueprint');
}
assert.deepEqual(fs.readdirSync(directory).filter(file => file.endsWith('.js')).sort(), [...ids].map(id => id + '.js').concat('index.js').sort(), 'Source directory has only owned plans and its index');
for (const filename of new Set(baseline.entries.map(entry => entry.source))) {
  assert.ok(!fs.existsSync(path.join('src/learn/data/curriculum', filename)), 'Legacy source bundle removed: ' + filename);
}
assert.deepEqual(trackDefinitions, baseline.trackDefinitions, 'Entire resolved catalogue, titles, prerequisites and all module/topic order remain byte-value equivalent');
const serialize = value => JSON.stringify(value);
const digest = value => createHash('sha256').update(serialize(value)).digest('hex');
const result = { passed: true, topicBlueprints: ids.size, removedSourceBundles: new Set(baseline.entries.map(entry => entry.source)).size,
  overlappingAuthoredTitles: baseline.overwritten.length, precedenceMismatches: baseline.precedenceMismatches.length,
  modules: trackDefinitions.length, topicMemberships: trackDefinitions.reduce((sum, track) => sum + track.sections.reduce((count, section) => count + section.topics.length, 0), 0),
  beforeCatalogueSha256: digest(baseline.trackDefinitions), afterCatalogueSha256: digest(trackDefinitions), checks: 'Exact per-topic default exports and aggregate equality; complete resolved catalogue equality; no legacy compatibility bundles.' };
fs.writeFileSync('scratch/blueprint-organization/conservation-results.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
