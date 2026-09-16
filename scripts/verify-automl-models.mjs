// Bounded independent checks of the AutoML & NAS browser models against the
// content packet's recorded calculations, the manuscript's worked values and
// analytic identities, plus structural checks of the generated data module, the
// displayed programs and the geometry the figures draw.
//
// "Independent" here means: an identity is checked by a second route, not by
// calling the same helper on both sides. Expected improvement is checked
// against a fine numerical integration of its own definition; the architecture
// gradient against a central difference; the halving schedule against an
// explicit re-implementation written here; every recorded fold accuracy against
// its own numerator and denominator; and every pooled accuracy against the
// stored list of mistaken rows.
//
// Run: node scripts/verify-automl-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  acquisitionDomain, acquisitionTable, activationKernel, activeRecipe, addOption,
  bilevelDerivatives, candidateSeries, checkRange, checkWhole, commitOperation, controlSteps, countConfigurations,
  decadeWeights, declaredCurves, declaredMixtureInputs, declaredOperations, declaredPareto,
  declaredPortfolio, declaredResource, declaredSpace, describeOption, enumerateConfigurations,
  erf, evidenceBoundary, evidenceFlow, expectedImprovement, familyOrder, halvingSchedule,
  hammingDistance, hyperbandBrackets, improvementDensity, limits, mixtureCurve, mixtureState,
  networkBlocks, normalCdf, normalPdf, normalizeSpace, paretoAnalysis, portfolioAnalysis,
  provenanceBoxWidth, provenanceLanes, removeOption, replayComparison, replayPrefix, samplingMeasure, sharedErrors,
  singleLayerCount, softmaxWeights,
} from '../src/learn/data/automl-models.js';
import {
  candidates, estimatorFits, foldRows, inspection, majorityBaseline, provenance, replayOrder,
  reserveScored, roles, selectedIndex, baselineIndex, sourceRows, versions,
} from '../src/learn/data/automl-data.js';
import { automlExamples } from '../src/learn/data/automl-examples.js';

const packetDirectory = 'docs/teaching/drafts/automl-neural-architecture-search-nas';
const sourcePathForMath = 'src/learn/data/topics/automl-neural-architecture-search-nas.jsx';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const constructed = recorded.constructed;
const extra = constructed.additional_checks;

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label} should have been refused`);
  record('refused input');
};
/** A value the prose tells a learner to type must land on the control's step. */
const onStep = (value, step) => Math.abs(value / step - Math.round(value / step)) < 1e-9;

// The assertion helpers must themselves be able to fail, or every group below
// would pass vacuously. Check that before using them on anything real.
assert.throws(() => close(1, 2, 'meta'), /meta/);
assert.throws(() => vector([1], [1, 2], 'meta'), /meta/);
assert.throws(() => refuses(() => 1, 'meta'));
record('the assertion helpers can fail');

/* ============================================ §1 · the contract and the loop */

const boundary = evidenceBoundary();
assert.deepEqual(boundary.loop, ['proposer', 'fit', 'score', 'record'], 'the four selection stages form the loop');
assert.equal(boundary.feedbackEdge.from, 'record', 'the only feedback arrow leaves the search record');
assert.equal(boundary.feedbackEdge.to, 'proposer', 'and reaches the proposer');
assert(boundary.inspectionFeedsNothing, 'nothing leaves the inspection stage');
assert(boundary.reservedIsIsolated, 'and the reserved partition is connected to nothing at all');
assert(boundary.validationRowsNeverFit, 'validation rows are not a source for the fitting stage');
assert.equal(boundary.scalerRefitPerFold, 'fit scaler + model on fitting rows', 'the nested step is named');
assert.deepEqual(boundary.selectionStages, ['proposer', 'fit', 'score', 'record'], 'exactly four stages are inside selection');
assert.equal(evidenceFlow.edges.filter(edge => edge.to === 'proposer').length, 1,
  'exactly one arrow can change the next proposal');
assert.equal(evidenceFlow.edges.filter(edge => edge.from === 'inspect' || edge.to === 'reserved').length, 0,
  'and none of them starts at inspection or ends at the reserve');
assert(evidenceFlow.stages.length >= 7 && evidenceFlow.edges.length >= 6, 'the flow is not empty');
record('the evidence boundary');

// A deliberately broken flow must be rejected, or the structural check is inert.
assert.throws(() => evidenceBoundary({
  ...evidenceFlow, edges: [...evidenceFlow.edges, { from: 'inspect', to: 'ghost' }],
}), RangeError, 'an arrow to a stage that does not exist is refused');
record('the boundary check can fail');

/* ============================================== §2 · the conditional grammar */

const space = countConfigurations(declaredSpace);
assert.equal(space.total, 11, 'the declared space holds eleven configurations');
assert.equal(space.total, constructed.conditional_counts.total, 'which is what the packet recorded');
assert.deepEqual(space.branches.map(branch => branch.count), [4, 2, 2, 3], 'the four branch counts');
assert.deepEqual(space.branches.map(branch => branch.count),
  familyOrder.map(family => constructed.conditional_counts[family]), 'and they match the packet branch by branch');
assert.equal(space.expression, '(2 × 2) + 2 + (1 × 2) + (1 × 1 × 3) = 11', 'the printed count expression');
// The enumeration is the independent route: count the rows, do not trust the product.
const enumerated = enumerateConfigurations(declaredSpace);
assert.equal(enumerated.length, space.total, 'enumerating the space gives the same total');
assert.equal(new Set(enumerated.map(row => JSON.stringify(row))).size, enumerated.length,
  'and every enumerated configuration is distinct');
familyOrder.forEach((family, index) => {
  assert.equal(enumerated.filter(row => row.family === family).length, space.branches[index].count,
    `${family} contributes its own branch count to the enumeration`);
});
// The misconception this exists to remove: multiplying everything together.
const productOfEverything = space.branches.reduce(
  (product, branch) => product * branch.dimensions.reduce((inner, entry) => inner * entry.size, 1), 1);
assert.equal(productOfEverything, 4 * 2 * 2 * 3, 'multiplying every branch together gives 48');
assert(productOfEverything !== space.total, 'which is not the number of valid configurations');
record('the declared grammar');

// The two edits the manuscript states, and the increment rule behind them.
const addedDepth = addOption(declaredSpace, 'tree', 'depth', 8);
assert.equal(addedDepth.after, 12, 'a fourth tree depth gives twelve configurations');
assert.equal(addedDepth.increment, 1, 'an increment of one');
assert.equal(addedDepth.pairedWith, 1, 'because a depth pairs with nothing else in its branch');
const addedPenalty = addOption(declaredSpace, 'logistic', 'penalty', 10);
assert.equal(addedPenalty.after, 13, 'a third logistic C gives thirteen configurations');
assert.equal(addedPenalty.increment, 2, 'an increment of two');
assert.equal(addedPenalty.pairedWith, 2, 'because it pairs with both preprocessing choices');
assert.equal(enumerateConfigurations(addedPenalty.space).filter(row => row.family === 'logistic').length, 6,
  'and the enumeration agrees');
const addedWidths = addOption(declaredSpace, 'mlp', 'widths', [16, 8]);
assert.equal(addedWidths.increment, 1, 'a fourth width pattern adds one configuration');
const removedScaling = removeOption(declaredSpace, 'logistic', 'scaling', 0);
assert.equal(removedScaling.after, 9, 'dropping raw preprocessing leaves nine configurations');
assert.equal(removedScaling.increment, -2, 'a decrement of two');
record('grammar edits');

// The practice-1 space: 6 + 4 + 2 + 4 = 16, and 64 four-fold fits.
const practiceSpace = {
  logistic: { scaling: ['raw', 'standard'], penalty: [0.1, 1, 10] },
  tree: { depth: [2, 4, 6, 8] },
  neighbors: { scaling: ['standard'], count: [3] },
  mlp: { scaling: ['standard'], activation: ['tanh'], widths: [[6], [12], [6, 6], [6, 12], [12, 6], [12, 12]] },
};
const practiceCounted = countConfigurations(practiceSpace);
assert.equal(practiceCounted.branches[0].count + practiceCounted.branches[1].count
  + practiceCounted.branches[3].count, 16, 'practice 1 has sixteen configurations in its three declared branches');
assert.equal(extra.practice_conditional_count, 16, 'which is what the packet recorded');
assert.equal(extra.practice_conditional_count * 4, extra.practice_cv_fits, 'and 64 four-fold fits');
assert.equal(extra.practice_cv_fits, 64, 'stated as 64');
assert.equal(enumerateConfigurations(practiceSpace).filter(row => row.family !== 'neighbors').length, 16,
  'the enumeration of those three branches also gives sixteen');
record('practice 1');

// The active/inactive distinction, which is investigation 1's null.
const baseSelection = {
  family: 'logistic',
  choices: { logistic: { scaling: 0, penalty: 1 }, tree: { depth: 0 }, neighbors: { scaling: 0, count: 0 }, mlp: { scaling: 0, activation: 0, widths: 0 } },
};
const retunedInactive = { ...baseSelection, choices: { ...baseSelection.choices, tree: { depth: 1 } } };
assert.equal(activeRecipe(declaredSpace, baseSelection).serialized,
  activeRecipe(declaredSpace, retunedInactive).serialized,
  'retuning an inactive tree depth leaves the active recipe untouched');
assert.equal(countConfigurations(declaredSpace).total, 11, 'and it cannot change the space, which is not even consulted');
assert(activeRecipe(declaredSpace, baseSelection).serialized.startsWith('logistic{'),
  'the active recipe names only the selected family');
assert(!activeRecipe(declaredSpace, baseSelection).serialized.includes('depth'),
  'and carries no second-layer or tree parameter at all');
assert.deepEqual(activeRecipe(declaredSpace, { family: 'mlp', choices: baseSelection.choices }).settings.map(setting => setting.dimension),
  ['scaling', 'activation', 'widths'], 'a one-layer network recipe has no second-layer width field');
assert.equal(activeRecipe(declaredSpace, { family: 'mlp', choices: baseSelection.choices }).settings.filter(setting => setting.fixed).length, 2,
  'and two of its three dimensions are fixed by the grammar');
record('active against inactive settings');

// Duplicates are refused rather than counted as new algorithms.
refuses(() => addOption(declaredSpace, 'tree', 'depth', 5), 'a repeated tree depth');
refuses(() => addOption(declaredSpace, 'mlp', 'widths', [8, 8]), 'a repeated width tuple');
refuses(() => addOption(declaredSpace, 'logistic', 'scaling', 'whitened'), 'an unknown preprocessing name');
refuses(() => addOption(declaredSpace, 'tree', 'depth', 2.5), 'a fractional tree depth');
refuses(() => addOption(declaredSpace, 'tree', 'depth', 40), 'a tree depth beyond the control range');
refuses(() => addOption(declaredSpace, 'logistic', 'penalty', 0), 'a non-positive C');
refuses(() => addOption(declaredSpace, 'mlp', 'widths', [8, 8, 8]), 'a three-layer width pattern');
refuses(() => addOption(declaredSpace, 'mlp', 'widths', [64]), 'a hidden width beyond the range');
refuses(() => removeOption(declaredSpace, 'neighbors', 'scaling', 0), 'emptying a branch dimension');
refuses(() => normalizeSpace({ ...declaredSpace, tree: { depth: [] } }), 'a branch with no values at all');
refuses(() => normalizeSpace({ logistic: declaredSpace.logistic }), 'a grammar missing three branches');
record('grammar guards');

// The sampling measures, exactly as the manuscript quotes them.
const familyUniform = samplingMeasure(declaredSpace, 'family-uniform');
const configurationUniform = samplingMeasure(declaredSpace, 'configuration-uniform');
close(familyUniform.total, 1, 'family-uniform probabilities sum to one', 1e-12);
close(configurationUniform.total, 1, 'configuration-uniform probabilities sum to one', 1e-12);
close(familyUniform.bands[0].probability, 1 / 4, 'logistic gets a quarter of family-uniform trials', 1e-12);
close(configurationUniform.bands[0].probability, 4 / 11, 'and four elevenths of configuration-uniform trials', 1e-12);
close(configurationUniform.bands[1].probability, 2 / 11, 'a tree gets two elevenths', 1e-12);
assert.equal(familyUniform.bands[0].fraction, '1/4', 'the printed family fraction');
assert.equal(configurationUniform.bands[0].fraction, '4/11', 'the printed configuration fraction');
assert(familyUniform.bands[0].probability !== configurationUniform.bands[0].probability,
  'so the two rules genuinely disagree');
refuses(() => samplingMeasure(declaredSpace, 'uniform'), 'an unnamed sampling rule');
record('sampling measures');

// Log-uniform decades: equal probability per decade, against a wildly unequal
// probability per decade when the value itself is sampled uniformly.
const decades = decadeWeights(-4, 2);
assert.equal(decades.length, 6, 'six decades from 1e-4 to 1e2');
close(decades.reduce((total, decade) => total + decade.logUniform, 0), 1, 'log-uniform weights sum to one', 1e-12);
close(decades.reduce((total, decade) => total + decade.valueUniform, 0), 1, 'value-uniform weights sum to one', 1e-12);
decades.forEach(decade => close(decade.logUniform, 1 / 6, 'each decade has equal log-uniform weight', 1e-12));
close(decades.at(-1).valueUniform, 0.9, 'the largest decade takes ninety percent of the uniform weight', 1e-5);
assert(decades.at(-1).valueUniform > 0.89 && decades[0].valueUniform < 1e-5,
  'so sampling the value uniformly nearly ignores the small decades');
refuses(() => decadeWeights(2, -4), 'a decreasing exponent range');
record('log-uniform sampling');

/* ================================================= §2 · expected improvement */

// The normal helpers, against known exact values and against each other.
close(erf(0), 0, 'erf(0)', 1e-15);
close(erf(1), 0.8427007929497149, 'erf(1)', 1e-13);
close(erf(-1), -0.8427007929497149, 'erf is odd', 1e-13);
close(erf(4), 0.9999999845827421, 'erf(4), through the continued fraction branch', 1e-12);
close(normalCdf(0), 0.5, 'Phi(0)', 1e-15);
close(normalCdf(1.959963984540054), 0.975, 'Phi at the 97.5th percentile', 1e-12);
close(normalCdf(-2) + normalCdf(2), 1, 'Phi is symmetric', 1e-13);
close(normalPdf(0), 1 / Math.sqrt(2 * Math.PI), 'phi(0)', 1e-15);
// Independent route: integrate the density and compare with the CDF.
for (const z of [-2, -0.5, 0.5, 1.5]) {
  const samples = 20001;
  const low = -12;
  const step = (z - low) / (samples - 1);
  let area = 0;
  for (let index = 1; index < samples; index += 1) {
    area += ((normalPdf(low + index * step) + normalPdf(low + (index - 1) * step)) / 2) * step;
  }
  close(area, normalCdf(z), `the integrated density equals Phi at ${z}`, 1e-7);
  record('Phi against its own integral');
}
record('the normal helpers');

// The three constructed acquisition values, against the packet.
const acquisitionCandidates = [
  { id: 'A', mean: 0.35, deviation: 0.02 },
  { id: 'B', mean: 0.40, deviation: 0.20 },
  { id: 'C', mean: 0.50, deviation: 0 },
];
const acquisition = acquisitionTable(0.4, acquisitionCandidates);
close(acquisition.rows[0].value, constructed.ei.A, 'EI for A', 1e-12);
close(acquisition.rows[1].value, constructed.ei.B, 'EI for B', 1e-12);
close(acquisition.rows[2].value, constructed.ei.C, 'EI for C', 1e-15);
close(acquisition.rows[0].value, 0.05004008274358261, "A's EI as the manuscript prints it", 1e-12);
close(acquisition.rows[1].value, 0.07978845608028655, "B's EI as the manuscript prints it", 1e-12);
assert.equal(acquisition.rows[2].value, 0, "C's EI is exactly zero, not a tiny number");
assert.equal(acquisition.winner, 'B', 'B wins the acquisition comparison');
assert.equal(acquisition.lowestMean, 'A', 'even though A has the lowest predicted mean');
assert(acquisition.rows[1].mean > acquisition.rows[0].mean, 'and a worse predicted mean');
record('the constructed acquisition table');

// Independent route: integrate (b − f) against the Gaussian density directly.
for (const candidate of [...acquisitionCandidates, { id: 'D', mean: -0.2, deviation: 0.5 }, { id: 'E', mean: 1.2, deviation: 0.3 }]) {
  if (candidate.deviation === 0) {
    close(expectedImprovement(0.4, candidate.mean, 0), Math.max(0.4 - candidate.mean, 0),
      'a point mass improves by its deterministic gap', 1e-15);
    record('EI against direct integration');
    continue;
  }
  const low = candidate.mean - 12 * candidate.deviation;
  const samples = 40001;
  const step = (0.4 - low) / (samples - 1);
  let area = 0;
  for (let index = 1; index < samples; index += 1) {
    const at = value => (0.4 - value) * normalPdf((value - candidate.mean) / candidate.deviation) / candidate.deviation;
    area += ((at(low + index * step) + at(low + (index - 1) * step)) / 2) * step;
  }
  close(area, expectedImprovement(0.4, candidate.mean, candidate.deviation),
    `EI for ${candidate.id} against its own integral`, 1e-6);
  record('EI against direct integration');
}

// The contrast, the null and the tie the investigation offers.
const certainB = acquisitionTable(0.4, [acquisitionCandidates[0], { id: 'B', mean: 0.4, deviation: 0 }, acquisitionCandidates[2]]);
assert.equal(certainB.rows[1].value, 0, "collapsing B's uncertainty leaves it no expected improvement");
assert.equal(certainB.winner, 'A', 'so A becomes the preferred candidate');
const offset = acquisitionTable(0.7, acquisitionCandidates.map(candidate => ({ ...candidate, mean: candidate.mean + 0.3 })));
acquisition.rows.forEach((row, index) => close(offset.rows[index].value, row.value,
  `a common offset leaves ${row.id}'s EI unchanged`, 1e-12));
assert.equal(offset.winner, acquisition.winner, 'and leaves the ordering unchanged');
const worseB = acquisitionTable(0.4, [acquisitionCandidates[0], { id: 'B', mean: 0.55, deviation: 0.2 }, acquisitionCandidates[2]]);
assert(worseB.rows[1].value < acquisition.rows[1].value, "raising B's mean lowers its expected improvement");
const tied = acquisitionTable(0.4, [{ id: 'A', mean: 0.3, deviation: 0.1 }, { id: 'B', mean: 0.3, deviation: 0.1 }]);
assert.equal(tied.winner, null, 'an exact tie is reported as a tie');
assert.deepEqual(tied.tied, ['A', 'B'], 'naming both candidates');
// Probability of improvement is a different number from expected improvement.
close(acquisition.rows[1].probabilityOfImprovement, 0.5, "B's probability of improvement is a half", 1e-12);
assert(acquisition.rows[1].probabilityOfImprovement !== acquisition.rows[1].value,
  'which is not its expected improvement');
record('acquisition contrasts and nulls');

// The drawn shading really is the expected improvement, not the probability.
const shadingDomain = acquisitionDomain(0.4, acquisitionCandidates);
assert(shadingDomain[0] < 0.4 && shadingDomain[1] > 0.4, 'the drawn axis contains the incumbent');
assert(shadingDomain[0] <= 0.4 - 5 * 0.2 && shadingDomain[1] >= 0.4 + 5 * 0.2,
  'and five deviations of the widest candidate');
for (const candidate of acquisitionCandidates) {
  const density = improvementDensity(0.4, candidate.mean, candidate.deviation, shadingDomain);
  assert(density.points.length >= limits.densitySamples && density.points.length <= limits.densitySamples + 244,
    'the drawn curve retains the common mesh and bounded local refinement');
  // Recomputed from the point's own loss and density, not restated from the
  // branch that produced it. A filter-then-every would also pass on an empty
  // filter, so both partitions are required to be non-empty.
  const below = density.points.filter(point => point.loss < 0.4);
  const above = density.points.filter(point => point.loss >= 0.4);
  assert(below.length > 10 && above.length > 10, 'the drawn grid straddles the incumbent');
  below.forEach(point => close(point.weighted, (0.4 - point.loss) * point.density,
    'the weighted density is the improvement times the density', 1e-12));
  above.forEach(point => assert.equal(point.weighted, 0, 'and is exactly zero at or above the incumbent'));
  if (candidate.deviation > 0) {
    close(density.area, density.exact, 'and the drawn area agrees with it', 3e-4);
    assert(!density.truncated, 'the adaptive domain is not clipped for the declared candidates');
  } else {
    assert(density.pointMass !== null, 'a zero-deviation candidate is drawn as a point mass');
    close(density.pointMass.improvement, 0, 'whose improvement here is zero', 1e-15);
  }
  record('drawn improvement shading');
}
// A deliberately narrow domain must report itself clipped.
assert(improvementDensity(0.4, 0.4, 0.2, [0.35, 0.45]).truncated, 'a clipped view says so');
record('clipping is reported');

// Practice 7.
close(expectedImprovement(0.3, 0.25, 0), extra.practice_ei[0], 'practice 7: U', 1e-15);
close(expectedImprovement(0.3, 0.3, 0.1), extra.practice_ei[1], 'practice 7: V', 1e-12);
close(expectedImprovement(0.3, 0.3, 0.1), 0.1 / Math.sqrt(2 * Math.PI), 'which is sigma over root two pi', 1e-12);
assert(extra.practice_ei[0] > extra.practice_ei[1], 'so U wins that comparison');
refuses(() => expectedImprovement(0.4, 0.3, -0.1), 'a negative predicted deviation');
refuses(() => acquisitionTable(5, acquisitionCandidates), 'an incumbent outside the control range');
refuses(() => acquisitionTable(0.4, [acquisitionCandidates[0]]), 'a single-candidate comparison');
refuses(() => acquisitionTable(0.4, [{ id: 'A', mean: 0.3, deviation: 2 }, acquisitionCandidates[1]]), 'a deviation beyond the range');
record('practice 7 and acquisition guards');

/* ================================================== §3 · successive halving */

/** An independent re-implementation, written here, of the rung rule. */
function independentHalving(rows, resource, keep) {
  let alive = rows.map((_, index) => index);
  const survivors = [];
  for (let rung = 0; rung < resource.length; rung += 1) {
    const target = rung === resource.length - 1 ? 1 : Math.floor(alive.length / keep);
    const ordered = alive.slice().sort((a, b) => (rows[a][rung] - rows[b][rung]) || (a - b));
    alive = ordered.slice(0, target);
    survivors.push(alive.slice());
    if (rung === resource.length - 1) break;
  }
  return survivors;
}

const halving = halvingSchedule();
const losses = declaredCurves.map(curve => curve.losses);
assert.equal(declaredCurves.length, 9, 'nine declared candidates');
assert.deepEqual(declaredCurves.map(curve => curve.losses), constructed.halving.curves,
  'the declared curves are the packet curves');
assert.deepEqual(declaredResource, constructed.halving.resource, 'on the packet resource ladder');
assert.deepEqual(halving.rungs[0].survivors, ['A', 'B', 'C'], 'A, B and C survive the first rung');
assert.deepEqual(halving.rungs[1].survivors, ['C'], 'C alone survives the second');
assert.equal(halving.selectedId, 'C', 'so C is selected');
close(halving.selectedFinalLoss, 0.07, 'finishing at 0.07', 1e-15);
assert.equal(halving.counterfactualId, 'D', 'D is the full-resource best');
close(halving.counterfactualFinalLoss, 0.02, 'at 0.02', 1e-15);
assert(halving.missedBetter, 'and the schedule missed it');
assert(!halving.rungs[0].survivors.includes('D'), 'because D was cut at the first rung');
// The independent implementation must agree, index for index.
const independent = independentHalving(losses, declaredResource, 3);
halving.rungs.slice(1).forEach((rung, index) => {
  assert.deepEqual(rung.started.slice().sort((a, b) => a - b), independent[index].slice().sort((a, b) => a - b),
    `the independent schedule agrees at rung ${index + 2}`);
});
assert.deepEqual(independent.at(-1), [2], 'and selects C by its own route');
// Work, three ways, against the packet.
assert.equal(halving.work.restart, constructed.halving.restart_cost, 'restart work');
assert.equal(halving.work.resume, constructed.halving.resume_cost, 'continuation work');
assert.equal(halving.work.allFull, constructed.halving.all_full_cost, 'all-full work');
assert.equal(halving.work.restart, 27, 'restart work is 27 units');
assert.equal(halving.work.resume, 21, 'continuation work is 21');
assert.equal(halving.work.allFull, 81, 'all-full work is 81');
assert.equal(halving.work.restartExpression, '9(1) + 3(3) + 1(9)', 'the printed restart expression');
assert.equal(halving.work.resumeExpression, '9(1) + 3(3 − 1) + 1(9 − 3)', 'the printed continuation expression');
assert(halving.work.resume < halving.work.restart && halving.work.restart < halving.work.allFull,
  'the three accountings are ordered as the prose says');
// Elimination reasons quote the actual compared loss and cutoff.
halving.rungs.forEach(rung => rung.eliminated.forEach(entry => {
  assert(entry.loss >= entry.cutoff, `${entry.id} was cut at or above the cutoff`);
  assert(entry.reason.includes(String(entry.loss)), 'and the reason quotes its own loss');
}));
assert(halving.rungs[0].eliminated.length === 6 && halving.rungs[1].eliminated.length === 2,
  'six then two candidates are eliminated');
// Purchased traces stop where a candidate was cut; unpurchased values are separate.
halving.traces.forEach(trace => {
  const rungsSurvived = halving.rungs.filter(rung => rung.started.includes(declaredCurves.findIndex(curve => curve.id === trace.id))).length;
  assert.equal(trace.purchased.length, rungsSurvived, `${trace.id}'s purchased trace stops where it was cut`);
  assert.equal(trace.purchased.length + trace.unpurchased.length, declaredResource.length + 1,
    `${trace.id}'s two traces share exactly one joining point`);
});
assert.equal(halving.traces.find(trace => trace.id === 'D').purchased.length, 1,
  'D bought exactly one measurement');
assert.equal(halving.traces.find(trace => trace.id === 'C').unpurchased.length, 1,
  'and the survivor has nothing unpurchased');
record('the declared halving schedule');

// The contrast: one edited first-rung value rescues the slow starter.
const rescued = declaredCurves.map(curve => ({ id: curve.id, losses: curve.id === 'D' ? [0.09, 0.07, 0.02] : [...curve.losses] }));
const rescuedSchedule = halvingSchedule(rescued);
assert.deepEqual(rescuedSchedule.rungs[0].survivors.slice().sort(), ['A', 'B', 'D'], 'D now survives the first rung');
assert.equal(rescuedSchedule.selectedId, 'D', 'and wins');
close(rescuedSchedule.selectedFinalLoss, 0.02, 'finishing at 0.02', 1e-15);
assert(!rescuedSchedule.missedBetter, 'so nothing better was missed');
assert.equal(rescuedSchedule.work.restart, halving.work.restart, 'the work is unchanged by the edit');
// The null: a uniform offset changes every label and no decision.
const shifted = declaredCurves.map(curve => ({ id: curve.id, losses: curve.losses.map(loss => loss + 0.05) }));
const shiftedSchedule = halvingSchedule(shifted);
assert.deepEqual(shiftedSchedule.rungs.map(rung => rung.survivors), halving.rungs.map(rung => rung.survivors),
  'a uniform offset leaves every survivor set');
assert.equal(shiftedSchedule.selectedId, halving.selectedId, 'and the selected candidate');
assert.deepEqual(shiftedSchedule.work, halving.work, 'and the whole work accounting');
close(shiftedSchedule.selectedFinalLoss, halving.selectedFinalLoss + 0.05, 'while the printed loss does move', 1e-12);
// A tie at a rung takes the earlier letter, and a non-monotone curve is accepted.
const tiedCurves = declaredCurves.map(curve => ({ id: curve.id, losses: curve.id === 'B' ? [0.10, 0.09, 0.08] : [...curve.losses] }));
assert.deepEqual(halvingSchedule(tiedCurves).rungs[0].survivors, ['A', 'B', 'C'],
  'a first-rung tie keeps the earlier letter first');
const nonMonotone = declaredCurves.map(curve => ({ id: curve.id, losses: curve.id === 'E' ? [0.20, 0.15, 0.30] : [...curve.losses] }));
assert.equal(halvingSchedule(nonMonotone).selectedId, 'C', 'a curve that gets worse is accepted, not rejected');
refuses(() => halvingSchedule(declaredCurves.slice(0, 2)), 'fewer candidates than the keep fraction');
refuses(() => halvingSchedule([{ id: 'A', losses: [0.1, 0.2] }, { id: 'B', losses: [0.1, 0.2] }, { id: 'C', losses: [0.1, 0.2] }]), 'a curve with the wrong number of rungs');
refuses(() => halvingSchedule(declaredCurves.map((curve, index) => ({ id: curve.id, losses: index === 0 ? [1.5, 0.1, 0.1] : [...curve.losses] }))), 'a loss above one');
record('halving contrasts, nulls and guards');

// Practice 3: four candidates, keep half.
const practiceCurves = [
  { id: 'A', losses: [0.10, 0.09, 0.08] }, { id: 'B', losses: [0.11, 0.08, 0.07] },
  { id: 'C', losses: [0.12, 0.07, 0.02] }, { id: 'D', losses: [0.20, 0.18, 0.15] },
];
const practiceHalving = halvingSchedule(practiceCurves, [1, 2, 4], 2);
assert.deepEqual(practiceHalving.rungs[0].survivors, ['A', 'B'], 'practice 3 keeps A and B');
assert.equal(practiceHalving.selectedId, 'B', 'then B');
close(practiceHalving.selectedFinalLoss, 0.07, 'finishing at 0.07', 1e-15);
assert.equal(practiceHalving.counterfactualId, 'C', 'while C would have reached the best final value');
assert.equal(practiceHalving.work.restart, extra.practice_halving_restart, 'practice 3 restart work');
assert.equal(practiceHalving.work.resume, extra.practice_halving_continue, 'practice 3 continuation work');
assert.equal(practiceHalving.work.restart, 12, 'stated as 12');
assert.equal(practiceHalving.work.resume, 8, 'and 8');
record('practice 3');

// Hyperband brackets, with the paper's rounding.
const brackets = hyperbandBrackets(9, 3);
assert.equal(brackets.brackets.length, 3, 'three brackets for R = 9, eta = 3');
assert.deepEqual(brackets.brackets.map(bracket => bracket.startCount), [9, 5, 3], 'their starting counts');
assert.deepEqual(brackets.brackets.map(bracket => bracket.startResource), [1, 3, 9], 'their starting resources');
assert.deepEqual(brackets.brackets[0].stages.map(stage => stage.candidates), [9, 3, 1], 'the first bracket ladder');
assert.deepEqual(brackets.brackets[1].stages.map(stage => stage.candidates), [5, 1], 'the second');
assert.deepEqual(brackets.brackets[2].stages.map(stage => stage.candidates), [3], 'the third');
assert.deepEqual(brackets.brackets.map(bracket => bracket.restartWork), [27, 24, 27], 'their restart work');
assert.equal(brackets.totalRestartWork, 78, 'totalling 78 units');
assert(brackets.totalRestartWork !== halving.work.allFull,
  'which is not the 81 units of fitting one fixed set of nine at full resource');
refuses(() => hyperbandBrackets(9, 1), 'an elimination factor of one');
record('hyperband brackets');

/* ===================================================== §4 · parameter counts */

assert.equal(networkBlocks(4, [8]).total, 49, 'width eight gives 49 parameters');
assert.equal(networkBlocks(4, [16]).total, 97, 'width sixteen gives 97');
assert.equal(networkBlocks(4, [8, 8]).total, 121, 'widths eight and eight give 121');
assert.deepEqual(networkBlocks(4, [8, 8]).blocks.map(block => block.total), [40, 72, 9],
  'whose blocks are 40, 72 and 9');
assert.equal(networkBlocks(4, [8, 8]).expression, '(4 + 1) × 8 + (8 + 1) × 8 + (8 + 1) × 1 = 121',
  'the printed block expression');
assert.equal(networkBlocks(4, [8, 8]).shapePath, '4 → 8 → 8 → 1', 'and the printed shape path');
// The 6h + 1 shortcut, checked against the block sum at every permitted width.
for (let width = limits.width.minimum; width <= limits.width.maximum; width += 1) {
  assert.equal(singleLayerCount(width), 6 * width + 1, `6h + 1 holds at width ${width}`);
  record('the single-layer shortcut');
}
// Bias counts and activation placement.
networkBlocks(4, [8, 8]).blocks.forEach(block => {
  assert.equal(block.biases, block.to, 'each block has one bias per output unit');
  assert.equal(block.total, block.weights + block.biases, 'and its total counts them');
});
assert.equal(networkBlocks(4, [8, 8]).blocks.at(-1).activation, 'sigmoid', 'the output block ends in a sigmoid');
assert.equal(networkBlocks(4, [8, 8]).blocks.at(-1).to, 1, 'over a single output unit, not two');
assert(networkBlocks(4, [8, 8]).blocks.slice(0, -1).every(block => block.activation === 'tanh'),
  'and every hidden block in tanh');
// Shapes chain: each block's input width is the previous block's output width.
networkBlocks(4, [8, 8]).blocks.slice(1).forEach((block, index) => {
  assert.equal(block.from, networkBlocks(4, [8, 8]).blocks[index].to, 'consecutive blocks are shape compatible');
});
// Practice 4.
assert.deepEqual(networkBlocks(5, [6, 3]).blocks.map(block => block.total), extra.practice_neural_parameter_blocks,
  'practice 4 blocks');
assert.deepEqual(networkBlocks(5, [6, 3]).blocks.map(block => block.total), [36, 21, 4], 'stated as 36, 21 and 4');
assert.equal(networkBlocks(5, [6, 3]).total, 61, 'totalling 61');
assert.equal(networkBlocks(5, [12]).total, 85, 'against 85 for one width-12 layer');
assert(networkBlocks(5, [12]).total > networkBlocks(5, [6, 3]).total,
  'so the one-layer alternative is the larger model here');
refuses(() => networkBlocks(4, []), 'a network with no hidden layer');
refuses(() => networkBlocks(4, [0]), 'a hidden width of zero');
refuses(() => networkBlocks(4, [8.5]), 'a fractional hidden width');
record('parameter counts');

/* ==================================== §5 · the recorded study and its replay */

// Provenance the page prints.
assert.equal(provenance.rows, 1372, 'the source has 1,372 rows');
assert.equal(provenance.uniqueGroups, 1348, 'and 1,348 unique feature vectors');
assert.equal(provenance.repeatedGroups, 11, 'eleven of which repeat');
assert.equal(provenance.repeatedExtraRows, 24, 'accounting for 24 further rows');
assert.equal(provenance.bytes, 46400, 'the served file size');
assert.equal(provenance.sha256.length, 64, 'and a full SHA-256');
assert(fs.existsSync(`public${provenance.file}`), 'the served dataset is actually present');
assert.equal(crypto.createHash('sha256').update(fs.readFileSync(`public${provenance.file}`)).digest('hex'),
  provenance.sha256, 'and its bytes hash to the printed digest');
assert(fs.existsSync(`public${provenance.attribution}`), 'the attribution file is present');
assert.equal(roles.reduce((total, role) => total + role.rows, 0), provenance.rows,
  'the three roles cover every source row');
assert.equal(roles.reduce((total, role) => total + role.groups, 0), provenance.uniqueGroups,
  'and every feature group');
assert.deepEqual(roles.map(role => role.rows), [919, 205, 248], 'the three role row counts');
assert.deepEqual(roles.map(role => role.positives), [407, 91, 112], 'and their class-1 counts');
assert.equal(foldRows.reduce((total, rows) => total + rows, 0), roles[0].rows,
  'the three folds partition the development pool');
assert.deepEqual(foldRows, [314, 300, 305], 'with 314, 300 and 305 validation rows');
assert.equal(reserveScored, false, 'and the reserve was never scored');
assert.equal(estimatorFits, candidates.length * foldRows.length + 2, '35 estimator fits');
assert.equal(estimatorFits, 35, 'stated as 35');
record('provenance and roles');

// Every fold accuracy, against its own numerator and denominator.
const series = candidateSeries();
assert.equal(series.rows.length, 11, 'eleven candidates are drawn');
assert(series.rows.length > 0, 'and the drawn set is not empty');
series.rows.forEach((row, index) => {
  const packetRow = recorded.candidates[index];
  assert.equal(row.id, packetRow.id, `candidate ${index} is the packet's candidate`);
  row.folds.forEach(fold => {
    close(fold.accuracy, fold.correct / fold.rows, `${row.id} fold ${fold.fold + 1} numerator over denominator`, 1e-15);
    close(fold.accuracy, packetRow.fold_accuracy[fold.fold], `${row.id} fold ${fold.fold + 1} against the packet`, 1e-15);
    assert.equal(fold.rows, foldRows[fold.fold], 'on the declared fold size');
  });
  close(row.mean, row.folds.reduce((total, fold) => total + fold.accuracy, 0) / 3,
    `${row.id}'s mean is the mean of its folds`, 1e-15);
  close(row.mean, packetRow.mean_fold_accuracy, `${row.id}'s mean against the packet`, 1e-15);
  // Pooled accuracy, recomputed from the stored list of mistaken rows.
  close(row.pooled, (row.pooledRows - row.errorRows.length) / row.pooledRows,
    `${row.id}'s pooled accuracy follows from its error list`, 1e-15);
  close(row.pooled, packetRow.pooled_oof_accuracy, `${row.id}'s pooled accuracy against the packet`, 1e-15);
  assert.equal(row.pooledCorrect, row.pooledRows - row.errorRows.length, 'and so does its numerator');
  assert.equal(packetRow.warnings.length, 0, `${row.id} fitted without a warning`);
  record('a recorded candidate');
});
// The manuscript's printed means, to six decimals.
const printedMeans = [0.977420, 0.983883, 0.964439, 0.975266, 0.905760, 0.962198,
  0.998907, 0.994629, 0.998907, 1.000000, 0.998907];
series.rows.forEach((row, index) => close(Number(row.mean.toFixed(6)), printedMeans[index],
  `${row.id}'s printed mean`, 1e-15));
assert(!series.meanAndPooledAgree, 'the mean of folds and the pooled accuracy are not the same rule');
const differing = series.rows.filter(row => Math.abs(row.mean - row.pooled) > 1e-12);
assert(differing.length > 0, 'and at least one candidate shows the difference');
close(series.rows[6].mean, 0.9989071038251366, 'the runner-up mean', 1e-15);
close(series.rows[6].pooled, 918 / 919, 'against its pooled 918 over 919', 1e-15);
assert(series.rows[6].mean !== series.rows[6].pooled, 'which differ');
record('the printed candidate table');

// Selection, the tie, and the shared mistake behind it.
assert.equal(selectedIndex, 9, 'registry index 9 is selected');
assert.equal(candidates[selectedIndex].id, 'mlp-tanh-16', 'that is the width-16 network');
assert.equal(candidates[selectedIndex].id, recorded.selected_id, 'as the packet recorded');
assert.equal(series.bestMean, 1, 'with mean fold accuracy exactly one');
assert.equal(candidates[selectedIndex].outOfFoldErrorRows.length, 0, 'and no out-of-fold mistake at all');
const runnersUp = series.rows.filter(row => Math.abs(row.mean - 0.9989071038251366) < 1e-12);
assert.deepEqual(runnersUp.map(row => row.id).sort(),
  ['mlp-tanh-8', 'mlp-tanh-8x8', 'neighbors-standard-k3'], 'three candidates tie for second');
runnersUp.forEach(row => assert.deepEqual(row.errorRows, [349], `${row.id} misses only source row 349`));
const shared = sharedErrors();
assert(shared.length > 0, 'the shared-error analysis is not empty');
assert.equal(shared[0].row, 349, 'the most widely shared mistake is row 349');
assert(shared[0].ids.length >= 3, 'made by at least three candidates');
assert.equal(sourceRows['349'].line, 350, 'whose file line is 350');
assert.equal(sourceRows['349'].label, 0, 'with true class 0');
assert.equal(sourceRows['349'].role, 'development', 'and it is a development row');
assert.equal(candidates[10].parameterCount, 121, 'the deeper network holds 121 parameters');
assert(candidates[10].meanFoldAccuracy < candidates[9].meanFoldAccuracy,
  'and still scores below the wider one');
assert.deepEqual([candidates[8], candidates[9], candidates[10]].map(entry => entry.parameterCount), [49, 97, 121],
  'the three network parameter counts');
assert(candidates.filter(entry => entry.parameterCount === null).length === 8,
  'and the other eight candidates have no parameter count');
// The tied set and the set that misses the hardest row are different sets, and
// the page states a count in words beside the list it introduces. A figure once
// printed "Three other candidates" above a list of seven, because it took the
// most-shared mistake instead of the runner-up score.
close(series.runnerUpMean, 0.9989071038251366, 'the runner-up mean', 1e-15);
assert.equal(series.tiedRunnersUp.length, 3, 'exactly three candidates tie for second');
assert.deepEqual(series.tiedRunnersUp.map(row => row.id).slice().sort(),
  ['mlp-tanh-8', 'mlp-tanh-8x8', 'neighbors-standard-k3'], 'and they are the three the manuscript names');
assert.deepEqual(series.sharedByTied, [349], 'the one mistake all three share');
assert(series.tiedRunnersUp.every(row => row.errorRows.length === 1), 'each of them makes exactly one');
assert(series.tieExplainedByOneMistake, 'so the tie is fully explained by that single shared mistake');
assert.equal(shared[0].ids.length, 7, 'while seven candidates in all miss that row');
assert(shared[0].ids.length > series.tiedRunnersUp.length,
  'so the two sets are genuinely different and must not be quoted for one another');
assert(series.tiedRunnersUp.every(row => shared[0].ids.includes(row.id)), 'the tied three are among the seven');
assert(shared[0].ids.some(id => !series.tiedRunnersUp.map(row => row.id).includes(id)),
  'and the seven include candidates that are not tied');
record('the tied set against the shared-mistake set');

record('selection and the shared mistake');

// The declared final comparison.
assert.equal(inspection.length, 2, 'exactly two models have a recorded inspection outcome');
const selectedInspection = inspection.find(entry => entry.role === 'selected');
const baselineInspection = inspection.find(entry => entry.role === 'declared_baseline');
assert.equal(baselineIndex, 3, 'the predeclared baseline is registry entry 3');
assert.equal(baselineInspection.id, candidates[baselineIndex].id, 'and that is what was refitted');
assert.equal(baselineInspection.id, 'logistic-standard-c1', 'the standardized logistic at C = 1');
assert.deepEqual(selectedInspection.confusion, [[114, 0], [0, 91]], 'the selected confusion matrix');
assert.deepEqual(baselineInspection.confusion, [[111, 3], [0, 91]], 'and the baseline matrix');
[selectedInspection, baselineInspection].forEach(entry => {
  const total = entry.confusion.flat().reduce((sum, value) => sum + value, 0);
  assert.equal(total, roles[1].rows, `${entry.role} matrix covers all 205 inspection rows`);
  assert.equal(entry.confusion[0][0] + entry.confusion[1][1], entry.correct,
    `${entry.role} diagonal equals its correct count`);
  close(entry.accuracy, entry.correct / entry.rows, `${entry.role} accuracy`, 1e-15);
  assert.equal(entry.errorRows.length, entry.rows - entry.correct, `${entry.role} error list length`);
  assert.equal(entry.confusion[0][0] + entry.confusion[0][1], roles[1].rows - roles[1].positives,
    `${entry.role} first row totals the class-0 rows`);
  assert.equal(entry.confusion[1][0] + entry.confusion[1][1], roles[1].positives,
    `${entry.role} second row totals the class-1 rows`);
  record('an inspection matrix');
});
assert.equal(selectedInspection.correct, 205, 'the selected network classifies all 205 correctly');
assert.equal(baselineInspection.correct, 202, 'the baseline classifies 202');
assert.deepEqual(baselineInspection.errorRows, [107, 195, 345], 'its three mistakes');
assert.deepEqual(baselineInspection.errorRows.map(row => sourceRows[String(row)].line), [108, 196, 346],
  'at file lines 108, 196 and 346');
baselineInspection.errorRows.forEach(row => {
  assert.equal(sourceRows[String(row)].label, 0, `row ${row} has true class 0`);
  assert.equal(sourceRows[String(row)].role, 'inspection', 'and belongs to the inspection partition');
  assert.equal(sourceRows[String(row)].features.length, 4, 'with four supplied descriptors');
});
assert.equal(majorityBaseline.predictedClass, 0, 'the development majority class is 0');
assert.equal(majorityBaseline.correct, 114, 'which gets 114 of 205 right');
close(majorityBaseline.accuracy, 114 / 205, 'its accuracy', 1e-15);
close(majorityBaseline.accuracy, recorded.always_majority_inspection_accuracy, 'against the packet', 1e-15);
assert(majorityBaseline.correct < baselineInspection.correct,
  'and it is worse than the fitted baseline, as the prose implies');
// No row is both a reserved row and an inspectable mistake.
Object.values(sourceRows).forEach(row => assert(row.role !== 'reserved', 'no reserved row is inspectable'));
assert(Object.keys(sourceRows).length > 0, 'and the inspectable set is not empty');
record('the declared final comparison');

// The replay, prefix by prefix.
assert.deepEqual(replayOrder, [7, 10, 6, 9, 3, 2, 1, 4, 8, 0, 5], 'the seed-75 reveal order');
assert.deepEqual(replayOrder, recorded.search_order, 'as the packet recorded');
assert.equal(new Set(replayOrder).size, candidates.length, 'which is a permutation of the registry');
const prefixes = [1, 2, 3, 4, 5, 11].map(budget => replayPrefix({ budget }));
assert.equal(prefixes[1].recommendedId, 'mlp-tanh-8x8', 'prefix 2 recommends the two-layer network');
assert.equal(prefixes[2].recommendedId, 'neighbors-standard-k3', 'prefix 3 recommends three-neighbor classification');
assert.equal(prefixes[3].recommendedId, 'mlp-tanh-16', 'prefix 4 recommends the width-16 network');
assert.equal(prefixes[1].recommended, 10, 'at registry index 10');
assert.equal(prefixes[2].recommended, 6, 'then 6');
assert.equal(prefixes[3].recommended, 9, 'then 9');
close(prefixes[1].best, prefixes[2].best, 'the best score is flat across that change', 1e-15);
assert(prefixes[3].best > prefixes[2].best, 'and rises at prefix 4');
assert.equal(prefixes[5].recommendedId, 'mlp-tanh-16', 'the full replay recommends what the study selected');
assert.equal(prefixes[5].recommended, selectedIndex, 'at the selected registry index');
// The tie really is decided by registry order, not arrival order.
assert(prefixes[2].steps.at(-1).tieDecided, 'prefix 3 is marked as a tie decision');
assert(!prefixes[2].steps.at(-1).scoreImproved, 'with no score improvement');
assert(prefixes[2].steps.at(-1).recommendationChanged, 'but a changed recommendation');
assert(prefixes[3].steps.at(-1).scoreImproved && prefixes[3].steps.at(-1).recommendationChanged,
  'and prefix 4 changes both');
assert.equal(prefixes[2].recommended, Math.min(6, 10), 'the tie takes the lower registry index');
// Reordering only the unrevealed tail changes nothing about the paid evidence.
const tailReordered = replayPrefix({
  order: [...replayOrder.slice(0, 4), ...replayOrder.slice(4).reverse()], budget: 4,
});
assert.equal(tailReordered.best, prefixes[3].best, 'a tail reorder leaves the best score');
assert.equal(tailReordered.recommended, prefixes[3].recommended, 'and the recommendation');
assert.equal(tailReordered.fits, prefixes[3].fits, 'and the fits paid for');
// Arrival order genuinely does not decide the tie: reveal k = 3 first instead.
const swapped = replayPrefix({ order: [6, 10, 7, 9, 3, 2, 1, 4, 8, 0, 5], budget: 2 });
assert.equal(swapped.recommended, 6, 'revealing the tied pair in the other order still recommends index 6');
// Fits, and the best-so-far series that the step chart draws.
prefixes.forEach(prefix => {
  assert.equal(prefix.fits, prefix.revealed.length * 3, 'three estimator fits per revealed candidate');
  assert(prefix.steps.every((step, index) => index === 0 || step.best >= prefix.steps[index - 1].best),
    'the best-so-far series never decreases');
  assert.equal(prefix.inspectionAvailable, false, 'and no prefix carries an inspection outcome');
  record('a replay prefix');
});
recorded.search_best_so_far.forEach((value, index) => {
  close(prefixes[5].steps[index].best, value, `the best-so-far series at step ${index + 1}`, 1e-15);
  record('a best-so-far value');
});
assert.equal(replayComparison(prefixes[1], prefixes[2]).outcome, 'recommendation',
  'budget 2 to 3 changes only the recommendation');
assert.equal(replayComparison(prefixes[2], prefixes[3]).outcome, 'both', 'budget 3 to 4 changes both');
assert.equal(replayComparison(prefixes[3], tailReordered).outcome, 'neither', 'the tail reorder changes neither');
assert.equal(replayComparison(prefixes[1], prefixes[2]).extraFits, 3, 'one more candidate costs three fits');
// Disabling a candidate really removes it from the evidence.
const withoutNeighbors = replayPrefix({ enabled: candidates.map((_, index) => index).filter(index => index !== 6), budget: 3 });
assert(!withoutNeighbors.revealed.includes(6), 'a disabled candidate is not revealed');
assert.notEqual(withoutNeighbors.recommendedId, 'neighbors-standard-k3', 'and cannot be recommended');
refuses(() => replayPrefix({ enabled: [], budget: 1 }), 'an empty candidate set');
refuses(() => replayPrefix({ budget: 0 }), 'a budget of zero');
refuses(() => replayPrefix({ budget: 12 }), 'a budget beyond the enabled candidates');
refuses(() => replayPrefix({ order: [0, 0, 1], enabled: [0, 1], budget: 1 }), 'a repeated candidate in the order');
record('the search replay');

/* ============================================= §6 · portfolio and deployment */

const portfolio = portfolioAnalysis();
assert.deepEqual(portfolio.oldTaskLosses, constructed.portfolio.old_task_losses,
  'the portfolio matrix the page prints is the packet matrix');
assert.deepEqual(portfolio.oldTaskLosses, declaredPortfolio.oldTaskLosses, 'and the declared one');
assert.equal(portfolio.oldTaskLosses.length, portfolio.ids.length, 'with one row per configuration');
assert(portfolio.oldTaskLosses.every(row => row.length === 2), 'and two old tasks in every row');
assert.deepEqual(declaredPortfolio.newTaskLosses, constructed.portfolio.new_task_losses, 'and so are the new-task losses');
vector(portfolio.means, [0.3, 0.3, 0.25], 'the three old-task means', 1e-15);
assert.equal(portfolio.singleId, 'C', 'C is the best single default');
close(portfolio.singleMean, extra.old_task_best_single_loss, 'at 0.25', 1e-15);
assert.deepEqual(portfolio.bestPair.members, ['A', 'B'], 'but A with B is the best pair');
close(portfolio.bestPair.mean, extra.old_task_portfolio_best_loss, 'at 0.10', 1e-15);
assert(portfolio.bestPair.mean < portfolio.singleMean, 'so the pair beats the best single default');
assert(!portfolio.bestPair.members.includes(portfolio.singleId), 'and it does not contain that default');
assert.equal(portfolio.pairs.length, 3, 'all three pairs were considered');
close(portfolio.newTaskPortfolioBest, extra.new_task_portfolio_best_loss, 'the portfolio reaches 0.40 on the new task', 1e-15);
close(portfolio.newTaskOverallBest, extra.new_task_alternative_loss, 'while 0.20 was available', 1e-15);
assert.equal(portfolio.newTaskOverallBestId, 'C', 'from the configuration the portfolio excluded');
assert(portfolio.portfolioMissesNewBest, 'so the transfer fails on the new task');
record('the portfolio counterexample');

const pareto = paretoAnalysis(declaredPareto, 5);
assert.deepEqual(declaredPareto.map(point => point.latency), constructed.pareto.latency_ms, 'the declared latencies');
assert.deepEqual(declaredPareto.map(point => point.accuracy), constructed.pareto.accuracy, 'and accuracies');
assert.deepEqual(pareto.frontierIds, constructed.pareto.nondominated, 'the frontier matches the packet');
assert.deepEqual(pareto.frontierIds, ['A', 'B', 'C'], 'stated as A, B and C');
assert.equal(pareto.rows.find(row => row.id === 'D').dominatedBy, 'B', 'B dominates D');
assert.equal(pareto.rows.find(row => row.id === 'E').dominatedBy, 'A', 'A dominates E');
assert.equal(pareto.selectedId, 'B', 'a 5 ms cap selects B');
assert.equal(paretoAnalysis(declaredPareto, 3).selectedId, 'A', 'a 3 ms cap selects A');
assert.equal(paretoAnalysis(declaredPareto, 1).selectedId, null, 'a 1 ms cap selects nothing');
assert(paretoAnalysis(declaredPareto, 1).infeasible, 'and says so rather than offering the nearest miss');
assert.deepEqual(paretoAnalysis(declaredPareto, 1).feasibleIds, [], 'with an empty feasible set');
// The null: lowering a dominated point changes nothing that matters.
const loweredE = declaredPareto.map(point => (point.id === 'E' ? { ...point, accuracy: 0.88 } : point));
assert.deepEqual(paretoAnalysis(loweredE, 5).frontierIds, pareto.frontierIds, 'lowering dominated E leaves the frontier');
assert.equal(paretoAnalysis(loweredE, 5).selectedId, 'B', 'and the choice');
// The second contrast: raising D changes the frontier and the choice.
const raisedD = declaredPareto.map(point => (point.id === 'D' ? { ...point, accuracy: 0.97 } : point));
assert.deepEqual(paretoAnalysis(raisedD, 5).frontierIds, ['A', 'B', 'D'], 'raising D puts it on the frontier');
assert.equal(paretoAnalysis(raisedD, 5).rows.find(row => row.id === 'C').dominatedBy, 'D', 'and dominates C');
assert.equal(paretoAnalysis(raisedD, 5).selectedId, 'D', 'so a 5 ms cap now selects D');
// Identical pairs do not dominate each other.
const twins = [{ id: 'A', latency: 2, accuracy: 0.9 }, { id: 'E', latency: 2, accuracy: 0.9 }, { id: 'B', latency: 4, accuracy: 0.94 }];
assert.deepEqual(paretoAnalysis(twins, 5).frontierIds, ['A', 'E', 'B'], 'identical pairs are both nondominated');
assert.equal(paretoAnalysis(twins, 3).selectedId, 'A', 'and a tie takes the earlier name');
// Equal accuracy with different latency, and equal latency with different accuracy.
const sameAccuracy = [{ id: 'A', latency: 2, accuracy: 0.9 }, { id: 'B', latency: 4, accuracy: 0.9 }];
assert.deepEqual(paretoAnalysis(sameAccuracy, 5).frontierIds, ['A'], 'the faster of two equally accurate models dominates');
const sameLatency = [{ id: 'A', latency: 2, accuracy: 0.9 }, { id: 'B', latency: 2, accuracy: 0.95 }];
assert.deepEqual(paretoAnalysis(sameLatency, 5).frontierIds, ['B'], 'and the more accurate of two equally fast ones does');
// Practice 6.
const practicePareto = paretoAnalysis([
  { id: 'P', latency: 3, accuracy: 0.92 }, { id: 'Q', latency: 6, accuracy: 0.96 }, { id: 'R', latency: 5, accuracy: 0.91 },
], 5);
assert.deepEqual(practicePareto.frontierIds, ['P', 'Q'], 'practice 6 frontier is P and Q');
assert.equal(practicePareto.rows.find(row => row.id === 'R').dominatedBy, 'P', 'P dominates R');
assert.equal(practicePareto.selectedId, 'P', 'and a 5 ms cap leaves P');
refuses(() => paretoAnalysis([declaredPareto[0]], 5), 'a single-candidate frontier');
refuses(() => paretoAnalysis(declaredPareto, 100), 'a cap beyond the control range');
refuses(() => paretoAnalysis(declaredPareto.map(point => ({ ...point, id: 'A' })), 5), 'repeated candidate names');
refuses(() => paretoAnalysis(declaredPareto.map(point => ({ ...point, accuracy: 1.5 })), 5), 'an accuracy above one');
// The drawn cap line is a rule a learner reasons from, so it must be exactly
// equivalent to the applied rule over the entire enterable grid, not at samples.
// Both latency and the cap are entered to two decimals, so a point sitting
// exactly on the line is representable and must read as feasible - the prose
// says "at or under". Sweep every value the control accepts.
let gridChecked = 0;
for (let hundredths = 10; hundredths <= 3000; hundredths += 1) {
  const value = Number((hundredths / 100).toFixed(2));
  const onTheLine = paretoAnalysis(
    [{ id: 'X', latency: value, accuracy: 0.9 }, { id: 'Y', latency: 30, accuracy: 0.1 }], value);
  assert.equal(onTheLine.rows[0].feasible, true, `a candidate exactly at a ${value} ms cap is feasible`);
  assert.equal(onTheLine.selectedId, 'X', 'and is the one shipped');
  gridChecked += 1;
  if (hundredths > 10 && value - 0.01 >= limits.latency.minimum) {
    const justOver = paretoAnalysis(
      [{ id: 'X', latency: value, accuracy: 0.9 }, { id: 'Y', latency: 30, accuracy: 0.1 }],
      Number((value - 0.01).toFixed(2)));
    assert.equal(justOver.rows[0].feasible, false, `and one hundredth over a ${value} ms cap is not`);
    assert.equal(justOver.infeasible, true, 'leaving nothing feasible');
    gridChecked += 1;
  }
}
assert(gridChecked > 5000, `only ${gridChecked} cap positions were swept`);
record('the drawn cap equals the applied rule across the enterable grid');

record('the frontier and the cap');

/* ============================================ §7 · the operation mixture */

const mixture = mixtureState(declaredMixtureInputs);
vector(mixture.probabilities, constructed.mixture.probabilities, 'the mixture probabilities', 1e-14);
vector(mixture.probabilities, [0.5, 0.25, 0.25], 'stated as a half and two quarters', 1e-14);
vector(mixture.outputs, constructed.mixture.operation_outputs, 'the operation outputs', 1e-15);
vector(mixture.outputs, [0, 2, -2], 'stated as 0, 2 and −2', 1e-15);
close(mixture.mixed, constructed.mixture.output, 'the mixed output', 1e-14);
close(mixture.mixed, 0, 'which is zero', 1e-15);
close(mixture.loss, constructed.mixture.half_squared_loss, 'the half-squared loss', 1e-14);
close(mixture.loss, 0.5, 'stated as 0.5', 1e-15);
vector(mixture.gradient, constructed.mixture.gradient, 'the architecture gradient', 1e-12);
vector(mixture.gradient, [0, -0.5, 0.5], 'stated as 0, −0.5 and 0.5', 1e-12);
vector(mixture.updatedLogits, constructed.mixture.updated_logits, 'the updated logits', 1e-12);
close(mixture.updatedOutput, constructed.mixture.updated_output, 'the output after one step', 1e-12);
close(mixture.updatedOutput, 0.1993359892499117, 'stated as 0.1993359892499117', 1e-12);
assert(mixture.movesTowardTarget, 'and the step moves the mixture toward the target');
assert.equal(mixture.descendId, 'identity', 'the identity logit is the one a descending step raises');
close(softmaxWeights([Math.log(2), 0, 0]).reduce((total, value) => total + value, 0), 1,
  'softmax weights sum to one', 1e-15);
record('the declared mixture');

// Independent route: a central difference of the loss in each logit.
for (const setup of [
  declaredMixtureInputs,
  { ...declaredMixtureInputs, x: -2 },
  { ...declaredMixtureInputs, target: -1.5 },
  { ...declaredMixtureInputs, logits: [0.3, -1.2, 2] },
  { active: ['identity', 'negation'], logits: [0, 0, 0], x: 2, target: 0, step: 0.4 },
]) {
  const state = mixtureState(setup);
  state.operations.forEach((operation, index) => {
    const position = declaredOperations.findIndex(entry => entry.id === operation.id);
    const step = 1e-6;
    const shift = delta => {
      const logits = [...setup.logits];
      logits[position] += delta;
      return mixtureState({ ...setup, logits }).loss;
    };
    close((shift(step) - shift(-step)) / (2 * step), operation.gradient,
      `a central difference reproduces the gradient at ${operation.id}`, 2e-6);
  });
  record('the gradient against a central difference');
}

// The null: a common offset on every active logit.
const offsetMixture = mixtureState({ ...declaredMixtureInputs, logits: declaredMixtureInputs.logits.map(value => value + 5) });
vector(offsetMixture.probabilities, mixture.probabilities, 'a common logit offset leaves the weights', 1e-14);
vector(offsetMixture.probabilities, constructed.mixture.translation_null_probabilities, 'as the packet recorded', 1e-14);
close(offsetMixture.mixed, mixture.mixed, 'and the output', 1e-14);
close(offsetMixture.loss, mixture.loss, 'and the loss', 1e-14);
vector(offsetMixture.gradient, mixture.gradient, 'and the gradient', 1e-12);
// The contrast: flipping the input reverses which operation is favourable.
const flipped = mixtureState({ ...declaredMixtureInputs, x: -2 });
vector(flipped.gradient, [0, 0.5, -0.5], 'flipping the input reverses the gradient', 1e-12);
assert.equal(flipped.descendId, 'negation', 'so negation becomes the favourable branch');
assert.notEqual(flipped.descendId, mixture.descendId, 'which is not what it was before');
// Degenerate: every operation returns the same value.
const degenerate = mixtureState({ ...declaredMixtureInputs, x: 0 });
assert(degenerate.degenerate, 'at x = 0 every operation returns zero');
vector(degenerate.gradient, [0, 0, 0], 'so the gradient is exactly zero', 1e-15);
assert.notEqual(degenerate.residual, 0, 'even though the residual is not');
// Deactivating an operation removes it from the denominator entirely.
const twoActive = mixtureState({ ...declaredMixtureInputs, active: ['identity', 'negation'] });
assert.equal(twoActive.operations.length, 2, 'two active operations');
vector(twoActive.probabilities, [0.5, 0.5], 'share the weight equally at equal logits', 1e-15);
close(twoActive.probabilities.reduce((total, value) => total + value, 0), 1,
  'and their weights still sum to one, so the inactive operation left the denominator', 1e-15);
record('mixture nulls, contrasts and the degenerate case');

// Commitment: the mixture can be strictly better than either discrete choice.
const commitSetup = { active: ['identity', 'negation'], logits: [0, 0, 0], x: 2, target: 0, step: 0.4 };
const commitState = mixtureState(commitSetup);
close(commitState.mixed, constructed.discretization.target, 'the commitment mixture outputs zero', 1e-15);
close(commitState.loss, constructed.discretization.mixed_loss, 'with zero loss', 1e-15);
vector(commitState.outputs, constructed.discretization.outputs, 'over outputs 2 and −2', 1e-15);
vector(commitState.probabilities, constructed.discretization.probabilities, 'at equal weights', 1e-15);
['identity', 'negation'].forEach(id => {
  const committed = commitOperation(commitState, id);
  close(committed.loss, constructed.discretization.selected_loss, `committing to ${id} costs two`, 1e-15);
  assert(committed.worseThanMixture, 'which is worse than the mixture');
  close(committed.difference, 2, 'by exactly two', 1e-15);
  record('a discretization gap');
});
// The largest weight is not automatically the best committed operation.
const lopsided = mixtureState({ active: ['identity', 'negation'], logits: [0, 2, -2], x: 1, target: -1, step: 0.4 });
const lopsidedCommit = commitOperation(lopsided, 'negation');
assert.equal(lopsidedCommit.argmaxId, 'identity', 'the largest softmax weight belongs to identity');
assert.equal(lopsidedCommit.bestDiscreteId, 'negation', 'while the lowest committed loss belongs to negation');
assert.notEqual(lopsidedCommit.argmaxId, lopsidedCommit.bestDiscreteId, 'so an argmax rule would choose wrongly here');
refuses(() => commitOperation(mixture, 'convolution'), 'committing to an operation that is not active');
refuses(() => mixtureState({ ...declaredMixtureInputs, active: ['identity'] }), 'a mixture of one operation');
refuses(() => mixtureState({ ...declaredMixtureInputs, x: 50 }), 'an input beyond the control range');
refuses(() => mixtureState({ ...declaredMixtureInputs, logits: [20, 0, 0] }), 'a logit beyond the control range');
refuses(() => softmaxWeights([0]), 'a softmax over one logit');
// The drawn function curve is the same model, sampled.
const curve = mixtureCurve({ logits: declaredMixtureInputs.logits, target: 1 });
assert(curve.points.length > 10, 'the drawn curve has points');
curve.points.forEach(point => close(point.mixed,
  mixtureState({ ...declaredMixtureInputs, x: point.x, step: 0 }).mixed,
  'every drawn point is the model evaluated there', 1e-12));
// No logit is favoured when the architecture gradient vanishes, and it vanishes
// for two different reasons. Answering "none" there is correct, so the model
// must not return an arbitrary argmin over a vector of zeros.
assert.equal(commitState.zeroGradient, true, 'the commitment fixture has an exactly zero gradient');
assert.equal(commitState.descendId, null, 'so no operation is favoured');
assert(!commitState.degenerate, 'even though its operations return different values');
assert.match(commitState.zeroGradientReason, /residual is zero/, 'because the mixture already sits on the target');
vector(commitState.gradient, [0, 0], 'and every component is zero', 1e-15);
assert.equal(degenerate.zeroGradient, true, 'identical outputs also give a zero gradient');
assert.equal(degenerate.descendId, null, 'with no operation favoured');
assert.match(degenerate.zeroGradientReason, /same value at this input/, 'for the other reason');
assert.equal(mixture.zeroGradient, false, 'the declared mixture does favour an operation');
assert.equal(mixture.descendId, 'identity', 'namely identity');
// A step from a zero gradient moves nothing, so "stay" is the honest verdict.
close(mixtureState({ ...commitSetup, step: 0.4 }).updatedOutput, commitState.mixed,
  'a step from a zero gradient leaves the output where it was', 1e-15);
record('the zero-gradient verdict');

record('commitment and the drawn function');

// Practice 8.
const practiceOperations = [
  { id: 'up', label: 'first', formula: 'o(x) = 3x', evaluate: x => 3 * x },
  { id: 'down', label: 'second', formula: 'o(x) = −x', evaluate: x => -x },
];
const practiceMixture = mixtureState({ operations: practiceOperations, active: ['up', 'down'], logits: [0, 0], x: 1, target: 0, step: 0 });
close(practiceMixture.mixed, extra.practice_mixture_output, 'practice 8 mixed output', 1e-15);
close(practiceMixture.mixed, 1, 'stated as 1', 1e-15);
close(practiceMixture.loss, 0.5, 'practice 8 loss', 1e-15);
vector(practiceMixture.gradient, extra.practice_mixture_gradient, 'practice 8 gradients', 1e-14);
vector(practiceMixture.gradient, [1, -1], 'stated as 1 and −1', 1e-14);
const practiceOffset = mixtureState({ operations: practiceOperations, active: ['up', 'down'], logits: [7, 7], x: 1, target: 0, step: 0 });
close(practiceOffset.mixed, practiceMixture.mixed, 'adding 7 to both logits leaves the output', 1e-14);
vector(practiceOffset.gradient, practiceMixture.gradient, 'and the gradients', 1e-14);
record('practice 8');

/* =================================================== §7 · bilevel derivatives */

const scalar = bilevelDerivatives({ w: 0, alpha: 0.2, xi: 0.1 });
close(scalar.stepped, constructed.bilevel_scalar.w_prime, "the one-step weight w'", 1e-15);
close(scalar.stepped, 0.02, 'stated as 0.02', 1e-15);
close(scalar.lanes[0].outer, constructed.bilevel_scalar.first_order_gradient, 'the direct derivative', 1e-15);
close(scalar.lanes[1].outer, constructed.bilevel_scalar.one_step_gradient, 'the one-step derivative', 1e-14);
close(scalar.lanes[2].outer, constructed.bilevel_scalar.exact_outer_gradient, 'the exact-inner derivative', 1e-15);
assert.equal(scalar.lanes[0].outer, 0, 'stated as exactly 0');
close(scalar.lanes[1].outer, -0.098, 'stated as −0.098', 1e-14);
close(scalar.lanes[2].outer, -0.8, 'stated as −0.8', 1e-15);
assert(scalar.allDistinct, 'and the three are genuinely different numbers');
assert.deepEqual(scalar.lanes.map(lane => lane.weightSymbol), ['w', "w'", 'w*'], 'three distinct weight symbols');
assert.deepEqual(scalar.lanes.map(lane => lane.dependency), [0, 0.1, 1], 'three distinct dependency derivatives');
// Independent route: differentiate each outer objective by central difference.
const difference = (objective, at) => (objective(at + 1e-6) - objective(at - 1e-6)) / 2e-6;
close(difference(() => 0.5 * (0 - 1) ** 2, 0.2), scalar.lanes[0].outer,
  'the direct objective does not depend on alpha at all', 1e-9);
close(difference(alpha => 0.5 * ((0 - 0.1 * (0 - alpha)) - 1) ** 2, 0.2), scalar.lanes[1].outer,
  'the one-step objective differentiated numerically', 1e-7);
close(difference(alpha => 0.5 * (alpha - 1) ** 2, 0.2), scalar.lanes[2].outer,
  'the exact-inner objective differentiated numerically', 1e-7);
// The stationary counterexample: equal values, different derivatives.
const stationary = bilevelDerivatives({ w: 0.2, alpha: 0.2, xi: 0.1 });
assert(stationary.stationary, 'at w = alpha the training gradient is zero');
close(stationary.stepped, stationary.w, "so w' has the same value as w", 1e-15);
close(stationary.lanes[1].outer, extra.stationary_one_step_gradient, 'yet the one-step derivative is not zero', 1e-14);
close(stationary.lanes[1].outer, -0.08, 'stated as −0.08', 1e-14);
assert.notEqual(stationary.lanes[1].outer, 0, 'which is the point of the counterexample');
assert.equal(stationary.lanes[1].dependency, 0.1, 'because the dependency derivative is still xi');
// Practice 9.
const practiceScalar = bilevelDerivatives({ w: 0, alpha: 0.5, xi: 0.2, valTarget: 2 });
close(practiceScalar.lanes[0].outer, extra.practice_bilevel.direct, 'practice 9 direct derivative', 1e-15);
close(practiceScalar.lanes[1].outer, extra.practice_bilevel.one_step, 'practice 9 one-step derivative', 1e-14);
close(practiceScalar.lanes[2].outer, extra.practice_bilevel.exact, 'practice 9 exact derivative', 1e-15);
close(practiceScalar.stepped, 0.1, "practice 9's one-step weight", 1e-14);
close(practiceScalar.lanes[1].outer, -0.38, 'stated as −0.38', 1e-14);
close(practiceScalar.lanes[2].outer, -1.5, 'and −1.5', 1e-15);
// Setting xi to zero really does give the first-order lane.
close(bilevelDerivatives({ w: 0, alpha: 0.2, xi: 0 }).lanes[1].outer, 0,
  'setting xi to zero collapses the one-step lane onto the direct one', 1e-15);
refuses(() => bilevelDerivatives({ w: 0, alpha: 0.2, xi: 2 }), 'an inner step size above one');
refuses(() => bilevelDerivatives({ w: 20, alpha: 0.2, xi: 0.1 }), 'a weight beyond the range');
record('the three architecture derivatives');

/* ================================================ §7 · weight provenance */

assert.equal(provenanceLanes.length, 3, 'three weight-provenance lanes');
assert.deepEqual(provenanceLanes.map(lane => lane.id), ['independent', 'shared', 'proxy'], 'named distinctly');
assert.equal(new Set(provenanceLanes.map(lane => lane.stateReused)).size, 3, 'each reusing different state');
assert(provenanceLanes[2].measures.includes('not an accuracy'), 'and the proxy lane says it is not an accuracy');
assert(provenanceLanes.every(lane => lane.steps.length >= 3), 'every lane has at least three steps');
assert(provenanceLanes[2].steps.some(step => step.includes('actual input batch')),
  'the proxy lane uses an actual input batch, not Gaussian noise');
// A drawn box must fit the label it is given. Every one of these boxes once
// truncated to an ellipsis, which turned the diagram into cut-off words.
const GLYPH = 6.7; // 11px monospace inside a 340-unit viewBox
provenanceLanes.forEach(lane => {
  assert.equal(lane.stepLabels.length, lane.steps.length, `${lane.id} has one short label per step`);
  const boxWidth = provenanceBoxWidth(lane.stepLabels.length);
  lane.stepLabels.forEach(label => {
    assert.ok(label.length * GLYPH <= boxWidth - 6,
      `${lane.id}: "${label}" needs ${(label.length * GLYPH).toFixed(1)} of ${boxWidth.toFixed(1)} units`);
  });
  // The short label must still be recognisable as its full step.
  lane.stepLabels.forEach((label, index) => {
    assert.ok(!label.endsWith('…'), `${lane.id} step ${index + 1} is not an ellipsis`);
  });
  record('a provenance lane label fits its box');
});

record('the provenance lanes');

assert.equal(hammingDistance('110', '101'), 2, 'the two codes differ in two positions');
assert.equal(hammingDistance('110', '110'), 0, 'identical codes differ nowhere');
const kernel = activationKernel(['110', '101']);
assert.deepEqual(kernel.matrix, extra.naswot_kernel, 'the activation kernel matches the packet');
assert.deepEqual(kernel.matrix, [[3, 1], [1, 3]], 'stated as [[3,1],[1,3]]');
assert.equal(kernel.determinant, extra.naswot_determinant, 'its determinant');
assert.equal(kernel.determinant, 8, 'stated as 8');
close(kernel.logDeterminant, extra.naswot_log_determinant, 'and its log determinant', 1e-15);
close(kernel.logDeterminant, Math.log(8), 'which is ln 8', 1e-15);
assert(!kernel.singular, 'it is not singular');
assert.equal(kernel.matrix[0][0], kernel.width, 'the diagonal is the number of recorded units');
assert.equal(kernel.matrix[0][1], kernel.width - kernel.distance, 'and the off-diagonal follows the Hamming distance');
const singular = activationKernel(['110', '110']);
assert.deepEqual(singular.matrix, [[3, 3], [3, 3]], 'identical codes give a constant matrix');
assert.equal(singular.determinant, extra.identical_code_determinant, 'with determinant zero');
assert.equal(singular.determinant, 0, 'stated as 0');
assert(singular.singular, 'reported as singular');
assert.equal(singular.logDeterminant, -Infinity, 'whose log determinant has no finite real value');
assert(typeof singular.logDeterminantLabel === 'string', 'and that state is labelled rather than stabilized');
assert.equal(kernel.logDeterminantLabel, null, 'while the ordinary case carries no such label');
refuses(() => activationKernel(['110']), 'an inset with one code');
refuses(() => activationKernel(['110', '10']), 'codes of different lengths');
refuses(() => activationKernel(['110', '1a1']), 'a code that is not a bit string');
record('the activation-code kernel');

/* ------------------------------ fields the lesson body reads off these models */

// A missing field reads as undefined and crashes the page at render time rather
// than failing a numeric check, so the shapes are asserted explicitly.
const requiredFields = {
  'countConfigurations': [space, ['branches', 'total', 'expression']],
  'halvingSchedule': [halving, ['rungs', 'work', 'selectedId', 'selectedFinalLoss', 'counterfactualId', 'counterfactualFinalLoss', 'traces', 'missedBetter']],
  'hyperbandBrackets': [brackets, ['brackets', 'totalRestartWork', 'maximumResource', 'eta']],
  'bilevelDerivatives': [scalar, ['lanes', 'stepped', 'alpha', 'xi', 'valTarget', 'trainingGradient', 'stationary']],
  'mixtureState': [mixture, ['operations', 'probabilities', 'outputs', 'mixed', 'residual', 'loss', 'gradient', 'updatedLogits', 'updatedOutput', 'updatedLoss', 'step', 'descendId', 'degenerate', 'movesTowardTarget']],
  'activationKernel': [kernel, ['codes', 'width', 'matrix', 'determinant', 'distance', 'singular', 'logDeterminant', 'logDeterminantLabel']],
  'portfolioAnalysis': [portfolio, ['ids', 'oldTaskLosses', 'means', 'singleId', 'singleMean', 'pairs', 'bestPair', 'newTaskLosses', 'newTaskPortfolioBest', 'newTaskOverallBest', 'newTaskOverallBestId', 'portfolioMissesNewBest']],
  'paretoAnalysis': [pareto, ['cap', 'rows', 'frontierIds', 'feasibleIds', 'selectedId', 'infeasible', 'reason', 'selectionRule']],
  'replayPrefix': [prefixes[3], ['order', 'revealed', 'steps', 'best', 'recommended', 'recommendedId', 'fits', 'tieRule', 'inspectionAvailable']],
  'candidateSeries': [series, ['rows', 'domain', 'bestMean', 'meanAndPooledAgree']],
  'samplingMeasure': [familyUniform, ['rule', 'bands', 'total']],
  'networkBlocks': [networkBlocks(4, [8, 8]), ['inputs', 'widths', 'outputs', 'sizes', 'blocks', 'total', 'expression', 'shapePath']],
  'evidenceBoundary': [boundary, ['loop', 'feedbackEdge', 'inspectionFeedsNothing', 'reservedIsIsolated', 'validationRowsNeverFit', 'scalerRefitPerFold', 'selectionStages']],
  'acquisitionTable': [acquisition, ['best', 'rows', 'highest', 'winner', 'tied', 'lowestMean']],
};
Object.entries(requiredFields).forEach(([name, [value, fields]]) => {
  fields.forEach(field => assert(value[field] !== undefined, `${name} returns a defined ${field}`));
  record('a model result shape');
});
[...series.rows, ...halving.traces, ...pareto.rows, ...prefixes[3].steps, ...mixture.operations,
  ...scalar.lanes, ...portfolio.pairs, ...brackets.brackets, ...familyUniform.bands].forEach(entry => {
  assert(entry && typeof entry === 'object', 'every listed row is an object the page can read');
});
record('model result shapes');

/* ============================================ the displayed programs and data */

const programs = Object.entries(automlExamples);
assert(programs.length >= 4, 'the examples module is not empty');
assert.equal(automlExamples.banknoteSearch.executed, true, 'the study program was executed');
assert(automlExamples.banknoteSearch.expected.includes('Role rows: 919 205 248'),
  'and its output prints the three role sizes');
assert(automlExamples.banknoteSearch.expected.includes('selected mlp-tanh-16 1.0'),
  'and the selected model with its inspection accuracy');
assert(automlExamples.banknoteSearch.expected.includes('[[114   0]'), 'and the selected confusion matrix');
assert(automlExamples.banknoteSearch.expected.includes('[[111   3]'), 'and the baseline matrix');
candidates.forEach(candidate => {
  assert(automlExamples.banknoteSearch.expected.includes(`${candidate.id} folds `),
    `the printed output names ${candidate.id}`);
  record('a printed candidate line');
});
assert(automlExamples.banknoteSearch.code.includes('threadpool_limits(limits=1)'),
  'the displayed study pins its thread count');
assert(!automlExamples.banknoteSearch.code.includes('[reserve]'), 'and never indexes the reserved rows');
['banknoteFlaml', 'banknoteKerasSearch'].forEach(key => {
  assert.equal(automlExamples[key].executed, false, `${key} is honestly marked unexecuted`);
  assert.equal(automlExamples[key].expected, undefined, 'and carries no recorded output');
  assert(automlExamples[key].note.length > 0, 'but says why');
  assert(automlExamples[key].code.includes('folds[0]'), 'and uses only the first development fold');
  record('an optional program');
});
assert(automlExamples.searchSpaceGrammar.code.includes('Choose one family'), 'the grammar block is the manuscript block');
assert.equal(automlExamples.searchSpaceGrammar.language, 'text', 'rendered as text, not as a program');
// Every family of the declared grammar appears in the displayed grammar block.
familyOrder.forEach(family => {
  const label = { logistic: 'Logistic regression', tree: 'Decision tree', neighbors: 'Nearest neighbors', mlp: 'Small neural network' }[family];
  assert(automlExamples.searchSpaceGrammar.code.includes(label), `the grammar block shows ${label}`);
  record('a grammar branch on the page');
});
assert.equal(versions.scikitLearn, recorded.versions.sklearn, 'the recorded scikit-learn version');
assert.equal(versions.numpy, recorded.versions.numpy, 'and NumPy version');
assert.equal(versions.python, recorded.versions.python, 'and Python version');
record('the displayed programs');

/* ------------------------------------------------------ control ranges */

assert(limits.width.maximum === 32 && limits.depth.maximum === 12 && limits.neighbors.maximum === 25,
  'the specified grammar ranges');
assert(limits.penalty.maximum === 100 && limits.penalty.minimum > 0, 'C stays positive and bounded');
assert(limits.surrogateDeviation.minimum === 0, 'a zero surrogate deviation is representable, not excluded');
assert(limits.logit.minimum === -8 && limits.logit.maximum === 8, 'the specified logit range');
assert(limits.input.minimum === -5 && limits.target.maximum === 5, 'the specified input and target ranges');
assert(limits.latency.minimum === 0.1 && limits.latency.maximum === 30, 'the specified latency range');
assert(limits.points.minimum === 2 && limits.points.maximum === 8, 'the specified candidate-count range');
assert(limits.maximumLayers === 2, 'the grammar permits at most two hidden layers');
// Every value a suggested setup or the prose asks a learner to type must land on
// its control's step, or the lab cannot reproduce the number the prose states.
// The step operand comes from `controlSteps`, which the components render, so
// this compares the prose against the page rather than against a second
// transcription of it in this file.
assert.equal(Object.keys(controlSteps).length, 13, 'every editable control declares its step');
Object.entries(controlSteps).forEach(([name, step]) => {
  assert(step > 0 && Number.isFinite(step), `${name} declares a positive step`);
});
[[8, 'depthWhole'], [10, 'penalty'], [0.09, 'loss'], [0.05, 'loss'], [0.4, 'incumbent'], [0.35, 'surrogateMean'],
  [0.2, 'surrogateMean'], [0.02, 'surrogateDeviation'], [3, 'latency'], [5, 'latency'], [0.88, 'accuracy'],
  [0.97, 'accuracy'], [2, 'input'], [-2, 'input'], [0.4, 'stepSize']].forEach(([value, control]) => {
  const step = control === 'depthWhole' ? 1 : controlSteps[control];
  assert(step !== undefined, `${control} has a declared step`);
  assert(onStep(value, step), `${value} is typeable at ${control}'s step of ${step}`);
  record('a typeable control value');
});
// Practice 6 tells the learner to enter its three pairs in the deployment
// investigation, so every one of them must be both typeable and in range there.
[['P', 3, 0.92], ['Q', 6, 0.96], ['R', 5, 0.91]].forEach(([id, latency, accuracy]) => {
  assert(onStep(latency, controlSteps.latency), `${id}'s latency is typeable at the latency step`);
  assert(onStep(accuracy, controlSteps.accuracy), `${id}'s accuracy is typeable at the accuracy step`);
  checkRange(latency, limits.latency, `${id}'s latency`);
  checkRange(accuracy, limits.accuracy, `${id}'s accuracy`);
  record('a practice 6 value the prose asks for');
});
checkRange(5, limits.cap, "practice 6's cap");
// Practice 1 tells the learner to build its space in the grammar investigation.
countConfigurations(practiceSpace);
assert(practiceSpace.mlp.widths.length <= limits.maximumBranchOptions,
  "practice 1's six width patterns fit the branch limit");
[0.1, 1, 10].forEach(value => checkRange(value, limits.penalty, "a practice 1 C"));
[2, 4, 6, 8].forEach(value => checkWhole(value, limits.depth, 'a practice 1 depth'));
[6, 12].forEach(value => checkWhole(value, limits.width, 'a practice 1 width'));
// Practice 5 tells the learner to reveal the first three candidates.
assert.doesNotThrow(() => replayPrefix({ budget: 3 }), "practice 5's budget is enterable");
record('practice pointers into the investigations');
[0.1, 1, 10].forEach(value => checkRange(value, limits.penalty, 'C'));
record('control ranges');

/* ============ every graded comparison, at its degenerate inputs ============ */

// Two blocking findings in a row were the same shape: a graded outcome computed
// by enumerating the reasons a thing might happen, then falling through to a
// wrong answer for a reason nobody listed. Both were fixed as instances. This
// block asserts the *property* instead, swept over the degenerate inputs —
// zero step, vanishing gradient, exact ties, identical rows — where such
// fall-throughs live.

// I4's step: the verdict must follow whether the displayed output actually
// moved, whatever the reason. A zero step and a zero gradient are different
// reasons for the same observable, and both must read 'stay'.
let stepCases = 0;
let stayCases = 0;
for (const step of [0, 0.0000001, 0.1, 0.4, 1]) {
  for (const x of [-2, 0, 1, 2]) {
    for (const target of [-1, 0, 1, 2]) {
      const state = mixtureState({ ...declaredMixtureInputs, x, target, step });
      const movedOnScreen = Math.abs(state.updatedOutput - state.mixed) > limits.tolerance;
      assert.equal(state.stepMoved, movedOnScreen,
        `stepMoved must equal whether the printed output changed (x=${x}, t=${target}, s=${step})`);
      assert.equal(state.stepOutcome === 'stay', !movedOnScreen,
        `a 'stay' verdict must mean the printed output did not move (x=${x}, t=${target}, s=${step})`);
      if (!movedOnScreen) {
        stayCases += 1;
        assert(typeof state.stepUnchangedReason === 'string' && state.stepUnchangedReason.length > 0,
          'and it must say why nothing moved');
        assert.equal(state.stepUnchangedReason.includes('step length is zero'), step === 0 && !state.zeroGradient,
          'naming the zero step only when the step is what was zero');
      } else {
        assert.equal(state.stepUnchangedReason, null, 'a step that moved has no unchanged-reason');
        assert(state.descendId !== null, 'and a step that moved has an operation to name');
      }
      stepCases += 1;
    }
  }
}
assert(stepCases === 80 && stayCases > 0, `${stepCases} step cases with ${stayCases} degenerate ones`);
// Specifically: a zero step with a live gradient reads 'stay', which is the case
// that was graded wrong.
const zeroStep = mixtureState({ ...declaredMixtureInputs, step: 0 });
assert.equal(zeroStep.stepOutcome, 'stay', 'a zero-length step moves nothing');
assert(!zeroStep.zeroGradient, 'even though its gradient is not zero');
assert.match(zeroStep.stepUnchangedReason, /step length is zero/, 'and it says which of the two it was');
assert.equal(zeroStep.updatedOutput, zeroStep.mixed, 'the printed output really is unchanged');
record('the step verdict follows the printed movement');

// I6's selection: whenever two feasible candidates share the greatest accuracy,
// the page must say the tie exists and must not claim uniqueness.
let tieCases = 0;
for (const cap of [2, 3, 4, 5, 8, 30]) {
  for (const accuracy of [0.88, 0.9, 0.93, 0.94, 0.96]) {
    const points = [
      { id: 'A', latency: 2, accuracy: 0.9 }, { id: 'B', latency: 4, accuracy: 0.94 },
      { id: 'C', latency: 8, accuracy: 0.96 }, { id: 'D', latency: 5, accuracy },
      { id: 'E', latency: 2, accuracy: 0.9 },
    ];
    const analysis = paretoAnalysis(points, cap);
    if (analysis.infeasible) continue;
    const best = Math.max(...analysis.rows.filter(row => row.feasible).map(row => row.accuracy));
    const sharing = analysis.rows.filter(row => row.feasible && Math.abs(row.accuracy - best) <= limits.tolerance);
    assert.equal(analysis.selectionTied, sharing.length > 1,
      `selectionTied must equal whether more than one feasible candidate holds the greatest accuracy (cap ${cap})`);
    assert.deepEqual(analysis.tiedOnAccuracyIds.slice().sort(), sharing.map(row => row.id).sort(),
      'and must name exactly those candidates');
    if (analysis.selectionTied) {
      tieCases += 1;
      assert(!analysis.reason.includes('has the greatest accuracy'),
        `the readout must not claim uniqueness when ${analysis.tiedOnAccuracyIds.join(' and ')} tie (cap ${cap})`);
      assert(analysis.reason.includes('tie on accuracy'), 'it must say a tie happened');
      assert(analysis.tieBrokenBy && analysis.reason.includes(analysis.tieBrokenBy),
        'and name the rule that settled it');
      assert(analysis.tiedOnAccuracyIds.includes(analysis.selectedId), 'the shipped candidate is one of the tied');
    } else {
      assert(analysis.reason.includes('has the greatest accuracy'), 'a unique winner may say so');
    }
  }
}
assert(tieCases > 0, 'the tie sweep actually reached a tie');
// The rule that settles it must be stated, not merely stored.
const tieAnalysis = paretoAnalysis([
  { id: 'A', latency: 2, accuracy: 0.9 }, { id: 'E', latency: 2, accuracy: 0.9 }, { id: 'B', latency: 4, accuracy: 0.94 },
], 3);
assert.equal(tieAnalysis.selectedId, 'A', 'identical pairs are settled by the earlier name');
assert.equal(tieAnalysis.tieBrokenBy, 'the earlier name', 'and the rule says which');
assert(tieAnalysis.selectionRule.includes('lower latency') && tieAnalysis.selectionRule.includes('earlier name'),
  'the declared rule names both tiebreaks');
record('the frontier readout cannot overstate uniqueness');

// The sibling investigations already behave; assert that rather than assume it.
assert.equal(acquisitionTable(0.4, [{ id: 'A', mean: 0.3, deviation: 0.1 }, { id: 'B', mean: 0.3, deviation: 0.1 }]).winner,
  null, 'an exact acquisition tie has no winner');
assert.deepEqual(halvingSchedule(declaredCurves.map((curve, index) => ({
  id: curve.id, losses: index === 1 ? [0.10, 0.09, 0.08] : [...curve.losses],
}))).rungs[0].survivors, ['A', 'B', 'C'], 'a halving tie takes the earlier letter');
assert.equal(replayPrefix({ budget: 3 }).recommended, 6, 'a replay tie takes the lower registry index');
record('sibling tie rules');

// S8: the guard admits three or more curves, so the body must handle them.
for (let count = 3; count <= declaredCurves.length; count += 1) {
  const subset = declaredCurves.slice(0, count);
  const schedule = halvingSchedule(subset);
  assert(subset.some(curve => curve.id === schedule.selectedId),
    `a ${count}-candidate ladder selects one of its own candidates`);
  assert.equal(schedule.rungs.length, declaredResource.length, 'over every rung');
  schedule.rungs.forEach(rung => assert(rung.started.length > 0, 'and no rung starts empty'));
  record('a short ladder the guard admits');
}

/* ------------------------------------------ the page's own math payloads */

// A backslash lost in transit — a shell heredoc eating one, or a generator
// emitting a literal backslash-n — ships a LaTeX command as bare prose, which
// KaTeX renders as italic letters rather than failing. Every command in every
// math payload on the page must therefore still carry its backslash.
const lessonSource = fs.readFileSync(sourcePathForMath, 'utf8');
const mathCommands = 'begin|end|frac|sum|nabla|alpha|lambda|sigma|tanh|hat|widehat|cdots|ldots'
  + '|times|approx|operatorname|mathbb|mathcal|mathrm|tfrac|partial|overline|sqrt|phi|Phi|xi'
  + '|epsilon|qquad|quad|log|det|arg|ge|le|sim|min|max|left|right|big|text';
// Both quotings, because the three template literals are the only payloads on
// the page built by interpolation — exactly the ones that can lose a backslash —
// and a single-quote-only pattern skipped all three. The floor is not a count
// guess: it is every `<Math`/`<MathBlock` opening in the file, so a payload the
// pattern cannot read fails here instead of being silently omitted.
const payloads = [
  ...[...lessonSource.matchAll(/<Math(?:Block)?>\{'((?:[^'\\]|\\.)*)'\}/g)].map(match => match[1]),
  ...[...lessonSource.matchAll(/<Math(?:Block)?>\{`((?:[^`\\]|\\.)*)`\}/g)].map(match => match[1]),
];
const mathOpenings = (lessonSource.match(/<Math(?:Block)?>/g) || []).length;
assert.equal(payloads.length, mathOpenings,
  `${payloads.length} payloads captured from ${mathOpenings} <Math> openings; the scan is not reading them all`);
assert(payloads.length > 40, `only ${payloads.length} math payloads were found; the scan is inert`);
assert(payloads.some(payload => payload.includes('bmatrix')),
  'the interpolated matrix payloads are among those scanned');
let intactCommands = 0;
payloads.forEach(payload => {
  for (const token of payload.matchAll(new RegExp(`(?<![A-Za-z])(${mathCommands})(?![A-Za-z])`, 'g'))) {
    const previous = payload[token.index - 1];
    // `gathered` and `bmatrix` also occur bare, as the argument of an
    // environment; everything else must be preceded by its backslash.
    if (previous === '{') continue;
    assert.equal(previous, '\\',
      `a LaTeX command lost its backslash: ${payload.slice(Math.max(0, token.index - 30), token.index + 25)}`);
    intactCommands += 1;
  }
  assert(!/(^|[^\\])\\n/.test(payload), `a bare backslash-n reached a math payload: ${payload.slice(0, 60)}`);
  record('a math payload');
});
assert(intactCommands > 100, `only ${intactCommands} LaTeX commands were inspected; the scan is inert`);
// `\widehat` emits a wide accent SVG that has defeated three rewrites of a line
// at 320 px in this project; the narrow accent is used instead throughout.
assert(!payloads.some(payload => payload.includes('\\widehat')), 'no payload uses the wide accent');
assert(payloads.some(payload => payload.includes('\\hat')), 'and the narrow one is used where a hat is meant');
record('math payload escapes');

/* ------------------------------------ sequential audit: editable edge cases */
// Independent 70-digit quadrature of integral_0^infinity u*phi(a+u) du,
// evaluated with mpmath, rather than the production CDF/continued fraction.
for (const [a, expected] of [
  [2, 0.0084907026168296375499989260776467],
  [4, 0.00000714525843240566675899146155208068],
  [8, 7.55026241194649891373884774402839e-17],
  [12, 1.46052011698455478015266857992299e-34],
]) {
  const actual = expectedImprovement(0, a, 1);
  assert(actual > 0 && Math.abs(actual / expected - 1) < 2e-13,
    `negative tail z=-${a}: ${actual} versus independently integrated ${expected}`);
  record('sequential EI relative tail pins');
}
for (let z = -38; z <= 8; z += 0.125) {
  assert(expectedImprovement(0, -z, 1) >= 0, `EI cannot be negative at z=${z}`);
}
record('sequential EI nonnegativity sweep (369 values)');
const narrowDomain = acquisitionDomain(0.4, [
  { mean: 0.35, deviation: 0.001 }, { mean: 0.4, deviation: 1 }, { mean: 0.5, deviation: 0 },
]);
const narrowDensity = improvementDensity(0.4, 0.35, 0.001, narrowDomain);
const meanPoint = narrowDensity.points.find(point => point.loss === 0.35);
assert(meanPoint, 'the narrow density includes its exact mean');
close(meanPoint.density, 398.9422804014327, 'narrow peak from 1/(sigma*sqrt(2*pi))', 1e-14);
assert(Math.abs(narrowDensity.area / 0.05 - 1) < 0.002,
  `narrow drawn area resolves .05 rather than the old .116: ${narrowDensity.area}`);
record('sequential narrow density mesh and area');
const cutoffTie = structuredClone(declaredCurves);
cutoffTie[3].losses[0] = 0.13;
assert.match(halvingSchedule(cutoffTie).rungs[0].eliminated.find(row => row.id === 'D').reason, /ties the cutoff/);
const finalTie = structuredClone(declaredCurves);
finalTie[0].losses[2] = finalTie[3].losses[2] = 0.07;
assert.equal(halvingSchedule(finalTie).missedBetter, false, 'another identifier at equal final loss is not better');
record('sequential halving cutoff and final ties');
const reduced = replayComparison(replayPrefix({ budget: 4 }), replayPrefix({ budget: 1 }));
assert(reduced.after.best < reduced.before.best && reduced.scoreChanged && !reduced.scoreImproved);
assert.equal(reduced.outcome, 'both', 'reducing budget changes both best score and recommendation');
record('sequential replay score decrease');
const tinyGradient = mixtureState({ logits: [8, -8, -8], x: 0.001, target: 0.001, step: 1 });
const tinyWeight = Math.exp(-16) / (1 + 2 * Math.exp(-16));
assert(Math.abs(tinyGradient.gradient[1] / (-tinyWeight * 1e-6) - 1) < 1e-13);
assert.equal(tinyGradient.zeroGradient, false, 'small nonzero derivative is not an exact zero');
assert.equal(tinyGradient.descendId, 'identity');
assert(tinyGradient.operations.some(row => row.probability !== row.updatedProbability));
assert(tinyGradient.updatedLogits[2] < -8, 'computed next logit may cross the input control bound without clamping');
assert.throws(() => mixtureState({ logits: [8.01, -8, -8], x: 0.001, target: 0.001, step: 1 }), RangeError,
  'entered logits still obey the declared control range');
assert.match(tinyGradient.stepUnchangedReason, /within 10⁻¹²/);
assert(!/step length is zero|gradient is exactly zero/.test(tinyGradient.stepUnchangedReason));
const twoIncreases = mixtureState({ logits: [0, 0, 1], x: 2, target: 1, step: 0.4 });
assert(twoIncreases.gradient.filter(value => value < 0).length === 2);
assert.equal(twoIncreases.descendId, 'identity', 'identity has the greatest of two positive logit updates');
record('sequential nonzero tiny gradient and multiple increases');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/automl-models.js',
  'src/learn/data/automl-data.js',
  'src/learn/data/automl-examples.js',
  'src/learn/components/lesson-labs/AutomlShared.jsx',
  'src/learn/components/lesson-labs/AutomlLabs.jsx',
  'src/learn/components/lesson-labs/AutomlFigures.jsx',
  'src/learn/components/lesson-labs/automl-labs.css',
  'src/learn/data/topics/automl-neural-architecture-search-nas.jsx',
  'src/learn/data/curriculum/blueprints/automl-neural-architecture-search-nas.js',
  'public/learn-assets/automl-nas/banknote-data.csv',
  'public/learn-assets/automl-nas/ATTRIBUTION.txt',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-automl-models.mjs'),
  packetInputsHash: hash(`${packetDirectory}/calculated-inputs.json`),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((total, value) => total + value, 0),
  scope: 'Browser AutoML and NAS models against the content phase\'s calculated-inputs.json (the conditional counts, '
    + 'both expected-improvement tables, the mixture and its updated logits, the discretization gap, the scalar '
    + 'bilevel triple and its stationary counterexample, all nine halving curves with their survivors and three work '
    + 'accountings, the portfolio matrix, the Pareto points, the activation kernel and every recorded practice '
    + 'value), the recorded banknote study (all 33 fold accuracies recomputed from their own numerators and '
    + 'denominators, all eleven pooled accuracies recomputed from their stored error lists, both inspection '
    + 'confusion matrices reconciled against the role class counts, the majority baseline and every replay prefix), '
    + 'and independent identities: expected improvement against a 40,001-point integration of its own definition, '
    + 'the normal CDF against an integration of its density, the halving schedule against a second implementation '
    + 'written inside the verifier, every architecture gradient against a central difference, all three bilevel '
    + 'derivatives against central differences of their own outer objectives, the configuration count against a full '
    + 'enumeration of the space, and the 6h + 1 shortcut against the block sum at all 32 permitted widths. Also '
    + 'checks that every entry point refuses out-of-range, duplicated, empty and malformed input, that the values '
    + 'the prose asks a learner to type land on their controls\' steps, and that the served dataset\'s bytes hash to '
    + 'the digest the page prints.',
  limitations: [
    'The banknote outcomes are precomputed native fits; the browser reproduces their recorded numbers, not the fitting.',
    'Displayed program execution is checked separately by scripts/verify-automl-examples.py.',
    'The data module is regenerated from the served dataset separately by scripts/verify-automl-data.py, which also re-derives the packet trust root consumed here.',
    'Rendering, visual layout, interaction and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/automl-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped AutoML model checks across ${Object.keys(counts).length} groups.`);
