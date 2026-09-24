import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parse } from '@babel/parser';
import { channelContext, headBudget, scalingBudget, signedScoreMap, solveBudgetWidth, eligibleArchitectures } from '../src/learn/data/landmark-architecture-models.js';
const record = 'docs/teaching/evidence/landmark-architecture-author.json';
writeFileSync(record, JSON.stringify({ passed: false, status: 'running' }));
const groups = [], check = (name, action) => { action(); groups.push(name); };
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const id = 'landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet';
const packet = JSON.parse(readFileSync(`docs/teaching/drafts/${id}/calculated-inputs.json`));
const data = JSON.parse(readFileSync('src/learn/data/landmark-architecture-measurements.json'));
const observations = JSON.parse(readFileSync(`public/learn-assets/${id}/recorded-feature-maps.json`));
check('Both full source modules parse; all13sections and18practice disclosures preserved; draft links do not leak', () => {
  const body = readFileSync(`src/learn/data/topics/${id}.jsx`, 'utf8');
  parse(body, { sourceType: 'module', plugins: ['jsx'] }); parse(readFileSync('src/learn/components/lesson-labs/LandmarkArchitectureLabs.jsx', 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  assert.equal((body.match(/<H2>/g) || []).length, 13); assert.equal((body.match(/<details>/g) || []).length, 18); assert.doesNotMatch(body, /\/lesson\.md|Figure placement:/);
});
check('All compact measurement and deferred map values equal their actual native inputs', () => {
  assert.deepEqual(data.exact, packet.exact);
  packet.fits.forEach((fit, i) => { const { observations, head_weight, head_bias, ...rest } = fit; assert.deepEqual(data.fits[i], rest); });
  assert.deepEqual(observations, packet.fits.filter(row => row.seed === 1).map(({ kind, observations, head_weight, head_bias }) => ({ kind, observations, head_weight, head_bias })));
});
check('Allthree head modes agree with explicit native arithmetic and degenerate one-position contracts', () => {
  const base = headBudget({ channels: 512, height: 7, width: 7, dense: 4096, classes: 1000 });
  assert.deepEqual(base.pieces, packet.exact.vgg16.head_layers); assert.equal(base.gapParameters, 513000); assert.equal(base.parameters, 123642856);
  const changed = headBudget({ channels: 128, height: 7, width: 7, dense: 4096, classes: 7, mode: 'flat' }); assert.equal(changed.parameters, 43911); assert.equal(changed.gapParameters, 903);
  for (const channels of [1, 128, 2048]) for (const classes of [1, 7, 1000]) { const row = headBudget({ channels, height: 1, width: 1, classes, dense: 1, mode: 'flat' }); assert.equal(row.parameters, row.gapParameters); assert.equal(row.macs, row.gapMacs); }
});
check('Context gates match allnative fixtures, explicit cell edits, permutations and equal shifts', () => {
  for (const [a, b, key] of [[2, 1, 'means_2_1'], [2, 3, 'means_2_3'], [0, 0, 'means_0_0']]) channelContext([[a, a, a, a], [b, b, b, b]]).gates.forEach((value, i) => close(value, packet.exact.se_toy[key][i]));
  const maps = [[0, 4, 2, 2], [1, 0, 2, 1]], current = channelContext(maps); assert.deepEqual(current.gates, channelContext(maps.map(map => [...map].reverse())).gates); assert.deepEqual(current.gates, channelContext(maps.map(map => map.map(value => value + 1))).gates);
  assert.notEqual(current.gates[0], channelContext([maps[0], [4, 0, 2, 1]]).gates[0]);
});
check('Compound scaling matches every native coefficient step and reports infeasible width without clamping', () => {
  for (const row of packet.exact.compound_steps) { const current = scalingBudget(row.depth, row.width, row.resolution); close(current.parameters, row.dense_parameter_factor); close(current.macs, row.dense_mac_factor); }
  close(scalingBudget(1, 1, Math.SQRT2).parameters, 1); close(scalingBudget(1, 1, Math.SQRT2).macs, 2); assert.equal(solveBudgetWidth(2, 1.5, 1.25).feasible, false); const feasible = solveBudgetWidth(3, 1.5, 1.25); assert.equal(feasible.feasible, true); close(scalingBudget(1.5, feasible.width, 1.25).macs, 3);
});
check('Eligibility endpoints and seed invariance follow both cost constraints', () => {
  for (const seed of [1, 2, 3]) { const rows = data.fits.filter(row => row.seed === seed); assert.deepEqual(eligibleArchitectures(rows, 5000, 60000), ['parallel', 'inverted_gated']); assert.deepEqual(eligibleArchitectures(rows, 2502, 41920), ['parallel']); assert.deepEqual(eligibleArchitectures(rows, 2501, 41920), []); assert.deepEqual(eligibleArchitectures(rows, 2502, 41919), []); assert.equal(eligibleArchitectures(rows, 4650, 76192).length, 4); }
});
check('Constructed signed maps match allnative cases; all80saved class scores reconstruct independently', () => {
  const fixture = packet.exact.cam, maps = fixture.features.map(map => map.flat());
  for (const [values, weights, key] of [[maps, fixture.weights, 'base'], [[maps[0], [0, 1, 6, 1]], fixture.weights, 'edited_negative_channel'], [maps.map(map => [...map].reverse()), fixture.weights, 'joint_spatial_permutation'], [maps, [0, 0], 'zero_weights']]) { const current = signedScoreMap(values, weights, fixture.bias); assert.deepEqual(current.totalMap, fixture[key].map.flat()); close(current.viaMap, fixture[key].score); close(current.viaFeatures, current.viaMap); }
  const changed = signedScoreMap(maps, [-1, 2], -.5); close(changed.viaMap, 0); close(signedScoreMap([[5, 2, 0, 3], maps[1]], [-1, 2], -.5).viaMap, -1);
  for (const fit of observations) for (const sample of fit.observations) for (let label = 0; label < 10; label++) { const current = signedScoreMap(sample.feature_maps.map(map => map.flat()), fit.head_weight[label], fit.head_bias[label]); close(current.viaFeatures, current.viaMap, 1e-12); close(current.viaMap, sample.logits[label], 8e-6); }
});
const files = [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/LandmarkArchitectureLabs.jsx', 'src/learn/components/lesson-labs/landmark-architectures.css', 'src/learn/data/landmark-architecture-models.js', 'src/learn/data/landmark-architecture-measurements.json', 'scripts/generate-landmark-architecture-lesson.mjs'];
writeFileSync(record, JSON.stringify({ passed: true, groups, sources: Object.fromEntries(files.map(file => [file, createHash('sha256').update(readFileSync(file)).digest('hex')])), limits: 'Author model/content source checks. Native execution, independent review and browser integration have separate records.' }, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, substantiveGroups: groups.length, groups }));
