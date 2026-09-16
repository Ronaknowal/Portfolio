// Bounded independent checks of the Bayesian-networks browser models against
// the content packet's recorded calculations, the manuscript's stated values,
// analytic identities and a second algorithm for every graphical claim.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// The d-separation claims are the reason this file exists. A graph question has
// an exact answer, so a lab that grades correct reasoning wrong is the easiest
// defect to ship on this page. Rather than re-running the module's own path
// enumeration, the verdicts below are recomputed by the ancestral-moralisation
// algorithm — build the ancestral subgraph of the three sets, connect parents
// that share a child, drop the arrow directions, delete the observation set,
// and ask whether any undirected path survives. That reaches the same theorem
// by a different route, and it is applied to every enterable combination of
// endpoints and observation set on every graph the page can show, not to
// samples of them.
//
// Run: node scripts/verify-bayesnet-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  alarmNetwork, ancestorsOf, backdoorCriterion, changeDirection, checkGraph, checkProbability,
  checkState, childrenOf, classPosterior, compareUncertainty, counterfactualOfUnit, counterfactualPair,
  dSeparation, descendantsOf, distanceToShape, edgeRoute, eliminationRun, entropyNats, enumerateWorlds,
  controlSteps, factorCells,
  fixtures, freeParameters, frontdoorConditions, frontdoorModel, graphNodes, graphRoutes, inducedWidth,
  jointProbability, laneGeometry, layeredLayout, limits, markovBlanket, multiplyFactors, networkFactors,
  nodeKeepOut,
  parentsOf, pathPolyline, purchaseComparison, queryFamilies, queryPosterior, serviceModel, simplePaths,
  sumOut, unchangedTolerance, withAlarmRow, withCallerRow, withRootChance, worldFactors,
} from '../src/learn/data/bayesnet-models.js';
import {
  conditionalInformation, finalModel, protocol, provenance, scores, trainingModels, validationSpecimens,
} from '../src/learn/data/bayesnet-data.js';

const packetDirectory = 'docs/teaching/drafts/bayesian-networks-causal-graphical-models';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/bayesnet-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const bayesnetExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => {
  assert(typeof actual === 'number' && Number.isFinite(actual), `${label}: ${actual} is not a finite number`);
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
};
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
/** A loop must have something to look at, or it asserts nothing. */
const nonEmpty = (collection, expected, label) => {
  assert(collection.length > 0, `${label}: the subject set is empty, so nothing was checked`);
  if (expected !== undefined) {
    assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
  }
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};

/* =========================== an independent d-separation, by a second theorem */

/**
 * Ancestral moralisation. This shares no code with the module's path
 * enumeration: it never lists a path, and it never classifies a collider.
 */
function separatedByMoralisation(edges, start, end, observed) {
  const keep = new Set([start, end, ...observed]);
  [...keep].forEach(node => ancestorsOf(edges, node).forEach(name => keep.add(name)));
  const induced = edges.filter(([from, to]) => keep.has(from) && keep.has(to));
  const undirected = new Map();
  [...keep].forEach(node => undirected.set(node, new Set()));
  const join = (a, b) => {
    if (a === b) return;
    undirected.get(a).add(b);
    undirected.get(b).add(a);
  };
  induced.forEach(([from, to]) => join(from, to));
  [...keep].forEach(node => {
    const parents = induced.filter(([, to]) => to === node).map(([from]) => from);
    parents.forEach(left => parents.forEach(right => join(left, right)));
  });
  const blocked = new Set(observed);
  const seen = new Set([start]);
  const stack = [start];
  while (stack.length) {
    const current = stack.pop();
    for (const next of undirected.get(current) ?? []) {
      if (blocked.has(next) || seen.has(next)) continue;
      if (next === end) return false;
      seen.add(next);
      stack.push(next);
    }
  }
  return !seen.has(end);
}

/** Every subset of a list, smallest first. */
function subsets(items) {
  return Array.from({ length: 2 ** items.length }, (_, code) =>
    items.filter((_value, index) => (code >> index) & 1));
}

/* ================================================= §1 · the network itself */

assert.deepEqual(alarmNetwork.nodes, ['B', 'E', 'A', 'J', 'M'], 'the network names its five variables in order');
assert.deepEqual(parentsOf(fixtures.alarmEdges, 'A'), ['B', 'E'], 'the alarm listens to both roots');
assert.deepEqual(childrenOf(fixtures.alarmEdges, 'A'), ['J', 'M'], 'and both callers listen to the alarm');
assert.deepEqual(descendantsOf(fixtures.alarmEdges, 'B'), ['A', 'J', 'M'], 'burglary reaches the alarm and both callers');
assert.deepEqual(ancestorsOf(fixtures.alarmEdges, 'J'), ['A', 'B', 'E'], 'a call has three ancestors');
assert.deepEqual(descendantsOf(fixtures.alarmEdges, 'J'), [], 'and no descendants of its own');
record('graph structure');

Object.entries(alarmNetwork.chance).forEach(([node, rows]) => {
  const expected = 2 ** alarmNetwork.parents[node].length;
  assert.equal(Object.keys(rows).length, expected,
    `${node} has one table row per parent setting`);
  Object.values(rows).forEach(value => {
    assert(value >= 0 && value <= 1, `${node} holds a probability`);
  });
  record('CPT shape');
});

const alarmParameters = freeParameters(fixtures.alarmParameters);
close(alarmParameters.stored, 20, 'the network stores twenty CPT entries', 1e-12);
close(alarmParameters.free, 10, 'and has ten free numbers', 1e-12);
close(alarmParameters.jointEntries, 32, 'the unrestricted joint has 32 entries', 1e-12);
close(alarmParameters.jointFree, 31, 'and 31 free numbers', 1e-12);
const practiceParameters = freeParameters(fixtures.practiceParameters);
close(practiceParameters.free, 14, 'practice 3 counts fourteen free parameters', 1e-12);
close(practiceParameters.jointFree, 23, 'against 23 in the unrestricted joint', 1e-12);
close(freeParameters(fixtures.naiveBayesParameters).free, 14, 'naive Bayes has fourteen free parameters', 1e-12);
close(freeParameters(fixtures.treeAugmentedParameters).free, 23, 'the tree-augmented model has 23', 1e-12);
// Each row of the count is q_i(r_i − 1) and nothing else.
nonEmpty(practiceParameters.rows, 4, 'practice 3 parameter rows');
practiceParameters.rows.forEach(row => {
  close(row.free, row.configurations * (row.states - 1), `${row.name} contributes q(r − 1)`, 1e-12);
  close(row.stored, row.configurations * row.states, `${row.name} stores q·r entries`, 1e-12);
  record('parameter row');
});
record('parameter counts');

/* ================================== §1–2 · worlds, by an independent sum */

/** Enumerate the joint with an explicit bitmask walk, not the module's helper. */
function independentWorlds(network) {
  const order = network.nodes;
  const out = [];
  for (let code = 0; code < 1 << order.length; code += 1) {
    const assignment = {};
    order.forEach((node, index) => { assignment[node] = (code >> index) & 1; });
    let probability = 1;
    order.forEach(node => {
      const key = network.parents[node].map(parent => assignment[parent]).join(',');
      const chance = network.chance[node][key];
      probability *= assignment[node] ? chance : 1 - chance;
    });
    out.push({ assignment, probability });
  }
  return out;
}

const worlds = enumerateWorlds(alarmNetwork);
nonEmpty(worlds, 32, 'enumerated worlds');
const independent = independentWorlds(alarmNetwork);
close(worlds.reduce((sum, world) => sum + world.probability, 0), 1,
  'the 32 world probabilities sum to one', 1e-12);
close(independent.reduce((sum, world) => sum + world.probability, 0), 1,
  'and so do the independently enumerated ones', 1e-12);
worlds.forEach(world => {
  const twin = independent.find(other => alarmNetwork.nodes
    .every(node => other.assignment[node] === world.assignment[node]));
  assert(twin, 'every world has an independently enumerated twin');
  close(world.probability, twin.probability, 'the two enumerations agree world by world', 1e-15);
  record('world probability');
});

const assembled = jointProbability(alarmNetwork, fixtures.assembledWorld);
close(assembled, 0.0005910156, 'the worked world is .0005910156', 1e-12);
const factors = worldFactors(alarmNetwork, fixtures.assembledWorld);
nonEmpty(factors, 5, 'assembled-world factors');
close(factors.reduce((product, factor) => product * factor.value, 1), assembled,
  'the five drawn strip values multiply to exactly the printed product', 1e-15);
vector(factors.map(factor => factor.value), [0.001, 0.998, 0.94, 0.9, 0.7],
  'each strip cell shows the entry its own parents select', 1e-15);
// The strip must name the parent setting it read, or the figure is decoration.
assert.equal(factors[2].parentKey, '1,0', 'the alarm cell reads the B=1, E=0 row');
assert.deepEqual(factors[2].parentStates, [1, 0], 'and labels those parent states');
record('world assembly');

/* ================================== §2 · posteriors against the packet */

const packetQueries = recorded.alarmQueries;
nonEmpty(packetQueries, 7, 'recorded alarm queries');
nonEmpty(fixtures.publishedEvidence, 7, 'published evidence sets');
fixtures.publishedEvidence.forEach((entry, index) => {
  const packet = packetQueries[index];
  assert.deepEqual(packet.evidence, entry.evidence, `evidence set ${index} matches the packet's`);
  const result = queryPosterior(alarmNetwork, entry.evidence);
  close(result.evidenceProbability, packet.evidenceProbability,
    `${entry.label}: evidence probability`, 1e-12);
  close(result.posterior, packet.burglary, `${entry.label}: burglary posterior`, 1e-12);
  // A second route: sum the independently enumerated worlds directly.
  const mass = [0, 0];
  independent.forEach(world => {
    if (Object.entries(entry.evidence).every(([node, state]) => world.assignment[node] === state)) {
      mass[world.assignment.B] += world.probability;
    }
  });
  close(mass[0] + mass[1], result.evidenceProbability,
    `${entry.label}: the independent sum reaches the same evidence mass`, 1e-14);
  close(mass[1] / (mass[0] + mass[1]), result.posterior,
    `${entry.label}: and the same posterior`, 1e-12);
  assert.equal(result.compatibleWorlds,
    independent.filter(world => Object.entries(entry.evidence)
      .every(([node, state]) => world.assignment[node] === state)).length,
    `${entry.label}: the compatible-world count is the count, not an estimate`);
  record('published posterior');
});

const bothCalls = queryPosterior(alarmNetwork, { J: 1, M: 1 });
close(bothCalls.posterior, 0.28417183536439294, 'the headline posterior', 1e-15);
close(bothCalls.evidenceProbability, 0.002084100239, 'and the evidence probability', 1e-11);
close(queryPosterior(alarmNetwork, {}).posterior, 0.001, 'no evidence returns the prior', 1e-15);
const alarmKnown = queryPosterior(alarmNetwork, { A: 1 });
const alarmAndJohn = queryPosterior(alarmNetwork, { A: 1, J: 1 });
// John adds nothing once the alarm state is known. The two sums run over
// different sets of compatible worlds, so the doubles differ in the last bit
// even though the quantity is identical; what has to hold is that the lesson's
// own grading rule calls it unchanged, and by a wide margin.
close(alarmAndJohn.posterior, alarmKnown.posterior,
  'John adds nothing once the alarm state is known', 1e-15);
assert.equal(changeDirection(alarmAndJohn.posterior, alarmKnown.posterior), 'unchanged',
  'and the rule the investigation grades with calls that unchanged');
const screeningGap = Math.abs(alarmAndJohn.posterior - alarmKnown.posterior);
assert(screeningGap < unchangedTolerance / 1000,
  `the screening-off residue is ${screeningGap}, a thousandfold inside the unchanged tolerance`);
assert(queryPosterior(alarmNetwork, { A: 1, E: 1 }).posterior < alarmKnown.posterior / 100,
  'an earthquake explains the alarm away by two orders of magnitude');
record('headline posteriors');

/* ============== §2 · the degenerate inputs the investigation can reach */

const impossible = queryPosterior(
  withCallerRow(withCallerRow(alarmNetwork, 'J', 0, 0), 'J', 1, 0), { J: 1 });
assert.equal(impossible.posterior, null, 'impossible evidence has no posterior');
assert.equal(impossible.evidenceProbability, 0, 'its evidence probability is exactly zero');
assert(typeof impossible.undefinedBecause === 'string' && impossible.undefinedBecause.length > 20,
  'and it says why, rather than printing a number');
close(recorded.changedAndNullChecks.impossibleJohn.evidenceProbability, 0,
  'the packet recorded the same impossible evidence', 1e-15);
assert.equal(recorded.changedAndNullChecks.impossibleJohn.burglary, null,
  'and recorded its posterior as absent');

const uninformative = withCallerRow(
  withCallerRow(alarmNetwork, 'J', 0, fixtures.uninformativeCallRow), 'J', 1, fixtures.uninformativeCallRow);
close(queryPosterior(uninformative, { J: 1 }).posterior, 0.001,
  'equal caller rows return the prior exactly', 1e-15);
close(queryPosterior(uninformative, { J: 1 }).posterior,
  recorded.changedAndNullChecks.uninformativeJohn.burglary,
  'matching the packet', 1e-15);

const raisedFalseCall = withCallerRow(alarmNetwork, 'J', 0, fixtures.falseCallJohn);
close(queryPosterior(raisedFalseCall, { J: 1, M: 1 }).posterior,
  recorded.changedAndNullChecks.falseJohn.burglary,
  'raising the false-call chance moves the two-call posterior to the recorded value', 1e-12);
// The null the prose promises: a row the evidence never reads cannot matter.
assert.equal(queryPosterior(raisedFalseCall, { A: 1, J: 1 }).posterior, alarmAndJohn.posterior,
  'changing the A=0 caller row cannot move a posterior that fixes A=1 — bit-identical');
close(recorded.changedAndNullChecks.unusedRow.burglary, alarmAndJohn.posterior,
  'and the packet recorded that null too', 1e-15);
record('degenerate evidence');

// A probability edit that IS read must move the answer by far more than the
// tolerance the "unchanged" verdict uses, or that verdict means nothing.
const nullMargin = Math.abs(queryPosterior(raisedFalseCall, { J: 1, M: 1 }).posterior - bothCalls.posterior);
assert(nullMargin > 1e-6,
  `the smallest genuine change a preset produces is ${nullMargin}, which must dwarf the unchanged tolerance`);
record('null margin');

refuses(() => queryPosterior(alarmNetwork, { Q: 1 }), 'evidence on a variable that does not exist');
refuses(() => queryPosterior(alarmNetwork, { J: 2 }), 'a state that is not 0 or 1');
refuses(() => queryPosterior(alarmNetwork, { B: 1 }), 'evidence on the query variable itself');
refuses(() => checkProbability(1.5, 'a probability'), 'a probability above one');
refuses(() => checkProbability(-0.001, 'a probability'), 'a negative probability');
refuses(() => checkProbability(Number.NaN, 'a probability'), 'a probability that is not a number');
refuses(() => checkState(0.5, 'a state'), 'a fractional state');
refuses(() => withRootChance(alarmNetwork, 'A', 0.5), 'a root edit on a node that has parents');
refuses(() => withAlarmRow(alarmNetwork, '2,2', 0.5), 'an alarm row that does not exist');

/* ============================================ §3 · factors and elimination */

const elimination = eliminationRun(alarmNetwork, fixtures.eliminationEvidence, fixtures.eliminationOrder);
close(elimination.posterior, bothCalls.posterior,
  'elimination and enumeration agree to the last bit the double carries', 1e-14);
vector(elimination.mass, bothCalls.mass, 'and on the two unnormalised masses', 1e-14);
close(elimination.mass[1], 0.00059224259, 'h(1) is the value the prose prints', 1e-10);
close(elimination.mass[0], 0.001491857649, 'and h(0) likewise', 1e-10);

const earthquakeStep = elimination.steps.find(step => step.variable === 'E');
assert(earthquakeStep, 'the run eliminates the earthquake');
assert.deepEqual(earthquakeStep.resultScope, ['B', 'A'], 'leaving a factor over B and A');
nonEmpty(earthquakeStep.cells, 4, 'the g(B, A) cells');
vector(earthquakeStep.cells.map(cell => cell.value),
  [0.998422, 0.001578, 0.05998, 0.94002], 'the four drawn g(B, A) cells', 1e-9);
// The packet recorded the same trace by a different arrangement.
nonEmpty(recorded.elimination, 4, 'recorded elimination trace');
recorded.elimination.forEach(row => {
  const cell = earthquakeStep.cells.find(entry => entry.states.B === row.b && entry.states.A === row.a);
  assert(cell, `the trace has a cell for B=${row.b}, A=${row.a}`);
  close(cell.value, row.afterSumE, `g(B=${row.b}, A=${row.a}) matches the packet`, 1e-12);
  record('elimination cell');
});
// Every step shrank its scope, or it was not an elimination.
nonEmpty(elimination.steps, 2, 'elimination steps');
elimination.steps.forEach(step => {
  assert(step.resultScope.length < step.productScope.length,
    `eliminating ${step.variable} must remove a variable from the scope`);
  assert(!step.resultScope.includes(step.variable), `${step.variable} is gone from the result`);
  record('elimination step');
});

// Factor algebra: multiplication commutes, and summing two variables out in
// either order gives the same table.
const [firstFactor, secondFactor] = networkFactors(alarmNetwork).map(entry => entry.factor);
const leftFirst = multiplyFactors(firstFactor, secondFactor);
const rightFirst = multiplyFactors(secondFactor, firstFactor);
nonEmpty(factorCells(leftFirst), 4, 'product cells');
factorCells(leftFirst).forEach(cell => {
  const twin = factorCells(rightFirst).find(other =>
    Object.entries(cell.states).every(([name, state]) => other.states[name] === state));
  close(twin.value, cell.value, 'factor multiplication commutes', 1e-15);
  record('factor commutes');
});
const alarmFactor = networkFactors(alarmNetwork).find(entry => entry.node === 'A').factor;
const sumBE = sumOut(sumOut(alarmFactor, 'B'), 'E');
const sumEB = sumOut(sumOut(alarmFactor, 'E'), 'B');
vector(sumBE.values, sumEB.values, 'summing two variables out commutes', 1e-15);
refuses(() => sumOut(alarmFactor, 'Q'), 'summing out a variable the factor does not hold');
record('factor algebra');

const chainNetwork = {
  nodes: ['A', 'B', 'C', 'D'],
  parents: { A: [], B: ['A'], C: ['B'], D: ['C'] },
  chance: { A: { '': 0.5 }, B: { 0: 0.5, 1: 0.5 }, C: { 0: 0.5, 1: 0.5 }, D: { 0: 0.5, 1: 0.5 } },
};
const endpointsFirst = inducedWidth(chainNetwork, fixtures.chainEndpointOrder);
const middleFirst = inducedWidth(chainNetwork, fixtures.chainMiddleOrder);
close(endpointsFirst.width, 1, 'eliminating endpoints first reaches width one', 1e-12);
close(middleFirst.width, 2, 'eliminating the middle first reaches width two', 1e-12);
close(endpointsFirst.largestBinaryFactorCells, 4, 'four binary cells for the endpoint order', 1e-12);
close(middleFirst.largestBinaryFactorCells, 8, 'eight for the middle order', 1e-12);
assert.deepEqual(endpointsFirst.fillEdges, [], 'the endpoint order creates no fill edge');
assert.deepEqual(middleFirst.fillEdges, [['A', 'C'], ['A', 'D']],
  'the middle order creates two fill edges, A–C and then A–D, and the figure draws both');
record('elimination cost');

/* ================= §4 · d-separation over the whole enterable grid */

const drawableGraphs = [
  ['the alarm network', fixtures.alarmEdges],
  ['the alarm network with a second route', fixtures.secondRouteEdges],
  ['a collider with a chain below it', fixtures.colliderChainEdges],
  ['the adjustment graph', fixtures.educationEdges],
  ['the adjustment graph with a direct effect', fixtures.educationWithDirectEffect],
  ['the frontdoor graph', fixtures.frontdoorEdges],
  ['the service graph', fixtures.serviceEdges],
  ...fixtures.equivalenceClass.map(member => [`the ${member.name} orientation`, member.edges]),
  ['the collider orientation', fixtures.collider.edges],
];
let separationCases = 0;
let separatedCases = 0;
let connectedCases = 0;
nonEmpty(drawableGraphs, 11, 'graphs the page can draw');
drawableGraphs.forEach(([name, edges]) => {
  checkGraph(edges);
  const nodes = graphNodes(edges);
  let perGraph = 0;
  nodes.forEach(start => nodes.forEach(end => {
    if (start === end) return;
    const others = nodes.filter(node => node !== start && node !== end);
    subsets(others).forEach(observed => {
      const verdict = dSeparation(edges, start, end, observed);
      const second = separatedByMoralisation(edges, start, end, observed);
      assert.equal(verdict.separated, second,
        `${name}: ${start} vs ${end} given {${observed.join(',')}} — path enumeration says `
        + `${verdict.separated}, ancestral moralisation says ${second}`);
      verdict.paths.forEach(entry => {
        assert.equal(entry.blocked, entry.blockerNode !== null,
          `${name}: a blocked path names its blocker and an active one does not`);
        if (entry.blocked) {
          const blocker = entry.interior.find(node => node.node === entry.blockerNode);
          assert(blocker && blocker.blocks, `${name}: the named blocker is the node that blocks`);
        }
      });
      separationCases += 1;
      perGraph += 1;
      if (verdict.separated) separatedCases += 1; else connectedCases += 1;
    });
  }));
  assert(perGraph > 0, `${name}: no separation case was generated, so nothing was checked`);
  // Pinned once per graph rather than once per case: it restates the module's
  // own definition, so evaluating it 125,000 times inflated a count without
  // adding a check. It still catches `separated` and `verdict` drifting apart.
  const pinned = dSeparation(edges, graphNodes(edges)[0], graphNodes(edges)[1], []);
  assert.equal(pinned.separated, pinned.active.length === 0,
    `${name}: the verdict is exactly "no active path remains"`);
  assert.equal(pinned.verdict, pinned.separated ? 'guaranteed-independent' : 'dependence-possible',
    `${name}: the graded label follows the same boolean`);
  record('graph swept');
});
assert(separationCases > 900,
  `the preset separation sweep must be exhaustive, not sampled; it covered ${separationCases} cases`);
record('separation sweep on presets');

/**
 * The presets are not the enterable grid. Investigation 2 lets a learner build
 * any acyclic graph within its limits, so the two algorithms are compared on
 * generated structures as well: every DAG on four nodes, and a deterministic
 * stride through the five- and six-node spaces. A learner's own graph is the
 * one most likely to expose a disagreement, and it is the one a preset sweep
 * never reaches.
 */
function generatedGraphs(nodeCount, stride) {
  const names = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'].slice(0, nodeCount);
  const pairs = [];
  for (let i = 0; i < nodeCount; i += 1) {
    for (let j = i + 1; j < nodeCount; j += 1) pairs.push([names[i], names[j]]);
  }
  const out = [];
  for (let code = 0; code < 2 ** pairs.length; code += stride) {
    const edges = pairs.filter((_pair, index) => (code >> index) & 1);
    if (edges.length === 0 || edges.length > limits.maximumEdges) continue;
    out.push(edges);
  }
  return out;
}

let generatedCases = 0;
let generatedGraphCount = 0;
let generatedSeparated = 0;
let generatedConnected = 0;
[[4, 1], [5, 7], [6, 61]].forEach(([nodeCount, stride]) => {
  const family = generatedGraphs(nodeCount, stride);
  nonEmpty(family, undefined, `${nodeCount}-node generated graphs`);
  family.forEach(edges => {
    checkGraph(edges);
    const nodes = graphNodes(edges);
    if (nodes.length < 2) return;
    generatedGraphCount += 1;
    nodes.forEach(start => nodes.forEach(end => {
      if (start >= end) return;
      const others = nodes.filter(node => node !== start && node !== end);
      subsets(others).forEach(observed => {
        const verdict = dSeparation(edges, start, end, observed);
        assert.equal(verdict.separated, separatedByMoralisation(edges, start, end, observed),
          `generated graph ${edges.map(pair => pair.join('>')).join(',')}: `
          + `${start} vs ${end} given {${observed.join(',')}}`);
        // The verdict a learner is graded against is the verdict the list shows.
        assert.equal(verdict.separated, verdict.active.length === 0,
          'the graded verdict is exactly "no active path remains" on generated graphs too');
        if (verdict.separated) generatedSeparated += 1; else generatedConnected += 1;
        generatedCases += 1;
      });
    }));
  });
  record('generated graph family swept');
});
assert(generatedGraphCount > 300,
  `the generated sweep covered ${generatedGraphCount} graphs the presets never reach`);
assert(generatedCases > 20000,
  `the generated sweep covered ${generatedCases} separation cases`);
// The preset sweep's both-verdicts requirement said nothing about these, so a
// change that made all 124,000 generated cases come out the same way passed.
assert(generatedSeparated > 1000 && generatedConnected > 1000,
  `the generated sweep must exercise both verdicts; it saw ${generatedSeparated} separated `
  + `and ${generatedConnected} connected`);
// A sweep that only ever produced one answer would prove nothing.
assert(separatedCases > 50 && connectedCases > 50,
  `the sweep must exercise both verdicts; it saw ${separatedCases} separated and ${connectedCases} connected`);
record('separation sweep');

/* Every suggested setup in investigation 2 must be able to produce BOTH
 * verdicts over the observation sets a learner can reach from it. A preset on
 * which every set gives the same answer is a fixture that cannot show the
 * contrast it is placed there to show -- and the practice-5 graph on T and Y
 * was exactly that, because T-to-Y is a single edge and a path with no
 * interior node can never be blocked. */
nonEmpty(fixtures.pathPresets, 4, 'investigation 2 presets');
fixtures.pathPresets.forEach(preset => {
  checkGraph(preset.edges);
  const others = graphNodes(preset.edges).filter(node => node !== preset.start && node !== preset.end);
  const verdicts = new Set(subsets(others)
    .map(observed => dSeparation(preset.edges, preset.start, preset.end, observed).separated));
  assert.equal(verdicts.size, 2,
    `the preset "${preset.label}" gives only "${[...verdicts][0] ? 'separated' : 'dependence possible'}" `
    + `for all ${2 ** others.length} observation sets a learner can reach, so it cannot show a contrast`);
  record('preset shows a contrast');
});

// The specific claims the prose makes about the alarm collider.
assert.deepEqual(dSeparation(fixtures.alarmEdges, 'B', 'E', []).active, [],
  'B and E are separated with nothing observed');
assert.equal(dSeparation(fixtures.alarmEdges, 'B', 'E', ['A']).active.length, 1,
  'observing the alarm opens the collider');
assert.equal(dSeparation(fixtures.alarmEdges, 'B', 'E', ['J']).active.length, 1,
  'and so does observing only a descendant of it');
assert.equal(dSeparation(fixtures.alarmEdges, 'B', 'E', ['M']).active.length, 1,
  'either descendant will do');
const descendantReason = dSeparation(fixtures.alarmEdges, 'B', 'E', ['J']).paths[0].reason;
assert(descendantReason.includes('descendant'),
  'the reason shown to a learner names the descendant rule, not just the verdict');
// And the packet agrees about which paths are active.
Object.entries(recorded.paths).forEach(([key, active]) => {
  const observed = JSON.parse(key.replace(/'/g, '"'));
  const verdict = dSeparation(fixtures.alarmEdges, 'B', 'E', observed);
  assert.deepEqual(verdict.active.map(entry => entry.path), active,
    `the active-path list for observed ${key} matches the packet`);
  record('packet path list');
});
const twoRoute = dSeparation(fixtures.secondRouteEdges, 'B', 'E', ['K', 'J']);
assert.deepEqual(twoRoute.active.map(entry => entry.path), recorded.changedAndNullChecks.twoPathsKJ,
  'observing K and J leaves exactly the collider route active, as recorded');
assert.deepEqual(
  dSeparation(fixtures.secondRouteEdges, 'B', 'E', ['K']).active.map(entry => entry.path),
  recorded.changedAndNullChecks.twoPathsK,
  'observing K alone blocks both routes, as recorded');
assert.equal(dSeparation(fixtures.secondRouteEdges, 'B', 'E', []).active.length, 1,
  'with nothing observed the second route is active while the collider route is not');
record('collider claims');

// Practice 2's graph, stated exactly as the solution states it.
assert(dSeparation(fixtures.colliderChainEdges, 'R', 'T', []).separated,
  'practice 2: R and T are separated with no observations');
assert(!dSeparation(fixtures.colliderChainEdges, 'R', 'T', ['W']).separated,
  'practice 2: observing W two steps below the collider opens the path');
assert(!dSeparation(fixtures.colliderChainEdges, 'R', 'T', ['S', 'W']).separated,
  'practice 2: observing the collider itself as well keeps it open');
record('practice 2');

assert.deepEqual(markovBlanket(fixtures.alarmEdges, 'B').blanket, ['A', 'E'],
  'the burglary blanket is the alarm and the earthquake');
assert.deepEqual(markovBlanket(fixtures.alarmEdges, 'A').blanket, ['B', 'E', 'J', 'M'],
  'and the alarm blanket is everything else');
// A blanket really does separate: nothing outside it stays connected.
graphNodes(fixtures.alarmEdges).forEach(node => {
  const { blanket } = markovBlanket(fixtures.alarmEdges, node);
  const outside = graphNodes(fixtures.alarmEdges).filter(name => name !== node && !blanket.includes(name));
  outside.forEach(other => {
    assert(dSeparation(fixtures.alarmEdges, node, other, blanket).separated,
      `${other} is separated from ${node} by its blanket`);
    record('blanket separates');
  });
});
record('Markov blanket');

refuses(() => checkGraph([['A', 'A']]), 'a self-edge');
refuses(() => checkGraph([['A', 'B'], ['A', 'B']]), 'a repeated edge');
refuses(() => checkGraph([['A', 'B'], ['B', 'A']]), 'a two-node cycle');
refuses(() => checkGraph([['A', 'B'], ['B', 'C'], ['C', 'A']]), 'a three-node cycle');
refuses(() => checkGraph(Array.from({ length: 13 }, (_, index) => ['A', `N${index}`])), 'too many edges');
refuses(() => dSeparation(fixtures.alarmEdges, 'B', 'E', ['B']), 'observing a query endpoint');
refuses(() => dSeparation(fixtures.alarmEdges, 'B', 'Q', []), 'an endpoint that is not in the graph');
refuses(() => simplePaths(fixtures.alarmEdges, 'B', 'B'), 'a query whose endpoints coincide');

/* =============== §6 · the backdoor criterion, and the destination note */

/** The criterion by a second route: no descendant of the treatment in the set,
 * and separation in the graph with the treatment's outgoing arrows removed. */
function backdoorByMoralisation(edges, treatment, outcome, adjustment) {
  const descendants = descendantsOf(edges, treatment);
  if (adjustment.some(node => descendants.includes(node))) return false;
  const cut = edges.filter(([from]) => from !== treatment);
  if (!graphNodes(cut).includes(treatment) || !graphNodes(cut).includes(outcome)) return true;
  return separatedByMoralisation(cut, treatment, outcome, adjustment);
}

let backdoorCases = 0;
let validSets = 0;
let invalidSets = 0;
[
  ['the note’s graph', fixtures.educationEdges, 'T', 'Y'],
  ['with E to Y added', fixtures.educationWithDirectEffect, 'T', 'Y'],
  ['the service graph', fixtures.serviceEdges, 'X', 'Y'],
  ['the frontdoor graph', fixtures.frontdoorEdges, 'X', 'Y'],
  ['the frontdoor graph with a direct effect', fixtures.frontdoorWithDirectEffect, 'X', 'Y'],
  ['the alarm network, treating the alarm', fixtures.alarmEdges, 'A', 'J'],
].forEach(([name, edges, treatment, outcome]) => {
  const others = graphNodes(edges).filter(node => node !== treatment && node !== outcome);
  subsets(others).forEach(set => {
    const result = backdoorCriterion(edges, treatment, outcome, set);
    const second = backdoorByMoralisation(edges, treatment, outcome, set);
    assert.equal(result.valid, second,
      `${name}: {${set.join(',')}} — path enumeration says ${result.valid}, moralisation says ${second}`);
    assert(typeof result.reason === 'string' && result.reason.length > 10,
      `${name}: every verdict states its reason`);
    backdoorCases += 1;
    if (result.valid) validSets += 1; else invalidSets += 1;
  });
  record('backdoor graph swept');
});
assert(backdoorCases >= 24, `the backdoor sweep covered ${backdoorCases} candidate sets`);
assert(validSets > 0 && invalidSets > 0, 'the backdoor sweep exercises both verdicts');
record('backdoor sweep');

// The destination note's exact finding, checked rather than accepted.
const soleBackdoor = backdoorCriterion(fixtures.educationEdges, 'T', 'Y', []);
assert.equal(soleBackdoor.backdoorPaths.length, 1,
  'the note’s graph has exactly one backdoor path from T to Y');
assert.deepEqual(soleBackdoor.backdoorPaths[0].path, ['T', 'E', 'S', 'Y'],
  'and it is T–E–S–Y');
assert.equal(simplePaths(fixtures.educationEdges, 'T', 'Y').length, 2,
  'there are two T-to-Y paths in total: the direct edge and that backdoor route');
assert(!soleBackdoor.valid, 'the empty set does not satisfy the criterion there');
[['E'], ['S'], ['E', 'S']].forEach(set => {
  const result = backdoorCriterion(fixtures.educationEdges, 'T', 'Y', set);
  assert(result.valid, `{${set.join(',')}} is a valid backdoor set in the note’s graph`);
  assert.equal(result.descendantViolations.length, 0, `and contains no descendant of T`);
  record('note adjustment set');
});
// Conditioning on E opens nothing, because the graph's only collider is the
// outcome itself and an endpoint is never an interior node of its own path.
const colliderNodes = graphNodes(fixtures.educationEdges)
  .filter(node => parentsOf(fixtures.educationEdges, node).length > 1);
assert.deepEqual(colliderNodes, ['Y'],
  'the only node with two parents in that graph is Y, the outcome');
assert.equal(dSeparation(fixtures.educationEdges, 'T', 'Y', ['E']).active.length, 1,
  'conditioning on E leaves only the direct causal edge active, and creates no new route');
// The changed graph of practice 5.
assert.equal(simplePaths(fixtures.educationWithDirectEffect, 'T', 'Y').length, 3,
  'adding E to Y gives three T-to-Y paths');
assert(backdoorCriterion(fixtures.educationWithDirectEffect, 'T', 'Y', ['E']).valid,
  'practice 5: {E} stays valid');
assert(backdoorCriterion(fixtures.educationWithDirectEffect, 'T', 'Y', ['E', 'S']).valid,
  'practice 5: {E, S} stays valid');
const sOnly = backdoorCriterion(fixtures.educationWithDirectEffect, 'T', 'Y', ['S']);
assert(!sOnly.valid, 'practice 5: {S} stops being valid');
assert.deepEqual(sOnly.openPaths.map(entry => entry.path), [['T', 'E', 'Y']],
  'and the route it leaves open is exactly T–E–Y');
assert.deepEqual(recorded.changedAndNullChecks.changedAdjustmentS, [['T', 'E', 'Y']],
  'which is what the packet recorded');
assert.deepEqual(recorded.changedAndNullChecks.changedAdjustmentE, [],
  'while {E} leaves nothing open, as recorded');
// The packet computed the same thing on the backdoor subgraph.
Object.entries(recorded.educationBackdoor).forEach(([key, active]) => {
  const observed = JSON.parse(key.replace(/'/g, '"'));
  const result = backdoorCriterion(fixtures.educationEdges, 'T', 'Y', observed);
  assert.deepEqual(result.openPaths.map(entry => entry.path), active,
    `the open backdoor paths for observed ${key} match the packet`);
  record('packet backdoor list');
});
record('destination note finding');

/* ============================================ §6 · observation and action */

const service = serviceModel(fixtures.service);
close(service.lanes[1].observedShares[1], 0.75, 'three quarters of serviced units are at high load', 1e-15);
close(service.lanes[1].observedRisk, recorded.service.observational[1], 'serviced observed risk', 1e-15);
close(service.lanes[0].observedRisk, recorded.service.observational[0], 'unserviced observed risk', 1e-15);
close(service.lanes[1].interventionRisk, recorded.service.interventional[1], 'interventional risk with', 1e-15);
close(service.lanes[0].interventionRisk, recorded.service.interventional[0], 'interventional risk without', 1e-15);
close(service.associationDifference, 0.1225, 'the observed difference', 1e-15);
close(service.causalDifference, 0.07, 'the causal difference', 1e-15);
close(service.bias, 0.0525, 'and the gap between them', 1e-15);
assert(service.positivity, 'the declared settings have both treatment levels at both loads');
// An independent route: build the eight-cell joint and read both answers off it.
const joint = {};
[0, 1].forEach(z => [0, 1].forEach(x => [0, 1].forEach(y => {
  const assign = x ? fixtures.service.assignment[z] : 1 - fixtures.service.assignment[z];
  const fail = fixtures.service.outcome[x][z];
  joint[`${z}${x}${y}`] = 0.5 * assign * (y ? fail : 1 - fail);
})));
const observedRisk = x => {
  const mass = [0, 1].reduce((sum, z) => sum + joint[`${z}${x}0`] + joint[`${z}${x}1`], 0);
  return [0, 1].reduce((sum, z) => sum + joint[`${z}${x}1`], 0) / mass;
};
close(observedRisk(1) - observedRisk(0), service.associationDifference,
  'the independently built joint gives the same observed difference', 1e-14);
const truncated = x => [0, 1].reduce((sum, z) => sum + 0.5 * fixtures.service.outcome[x][z], 0);
close(truncated(1) - truncated(0), service.causalDifference,
  'and the truncated factorisation gives the same causal difference', 1e-15);
record('service model');

const randomised = serviceModel(fixtures.serviceRandomised);
close(randomised.associationDifference, randomised.causalDifference,
  'random assignment makes observation and intervention agree', 1e-15);
close(randomised.causalDifference, service.causalDifference,
  'while leaving the causal difference exactly where it was', 1e-15);
const changedResponse = serviceModel(fixtures.serviceChangedResponse);
close(changedResponse.causalDifference, 0.05, 'practice 7: the new causal difference', 1e-15);
close(changedResponse.associationDifference, 0.1125, 'practice 7: the new observed difference', 1e-15);
close(changedResponse.lanes[1].observedRisk, 0.1525, 'practice 7: the serviced observed risk', 1e-15);
close(changedResponse.lanes[1].interventionRisk, 0.105, 'practice 7: the treated intervention risk', 1e-15);
const noOverlap = serviceModel(fixtures.serviceNoOverlap);
assert(!noOverlap.positivity, 'assignment [0, 1] is reported as having no overlap');
close(noOverlap.causalDifference, 0.07, 'the fully specified model still returns .07 there', 1e-15);
vector(noOverlap.lanes.map(lane => lane.observedRisk), recorded.noSupport.observational,
  'and the observed risks match the packet', 1e-15);
// The truly degenerate corner: nobody receives the procedure at all.
const empty = serviceModel({ assignment: [0, 0], outcome: fixtures.service.outcome });
assert.equal(empty.lanes[1].observedRisk, null, 'with nobody treated, the treated risk has no value');
assert.equal(empty.associationDifference, null, 'so the observed difference has none either');
close(empty.causalDifference, 0.07, 'while the causal difference is unaffected', 1e-15);
assert(typeof empty.lanes[1].undefinedBecause === 'string',
  'and the empty lane names its empty denominator');
record('service degenerate corners');

// The drawn lane widths are the weights the arithmetic used, not a redrawing.
const lanes = laneGeometry(service);
nonEmpty(lanes, 4, 'drawn population lanes');
lanes.forEach(lane => {
  const shares = lane.segments.map(segment => segment.share);
  close(shares[0] + shares[1], 1, `${lane.kind} lane ${lane.lane}: the drawn shares sum to one`, 1e-12);
  lane.segments.forEach(segment => {
    assert(segment.width >= 0, 'no segment is drawn with negative width');
    close(segment.width, 300 * segment.share, 'segment width is proportional to its share', 1e-9);
  });
  const reconstructed = shares.reduce((sum, share, index) =>
    sum + share * fixtures.service.outcome[lane.lane][index], 0);
  close(reconstructed, lane.risk, `${lane.kind} lane ${lane.lane}: the drawn split reproduces the printed risk`, 1e-14);
  record('lane geometry');
});
assert.equal(lanes[0].segments[0].x, 0, 'the first segment starts at the left edge');
close(lanes[0].segments[1].x, lanes[0].segments[0].width, 'and the second starts where the first ends', 1e-12);
record('lane layout');

refuses(() => serviceModel({ assignment: [0.5, 1.5], outcome: fixtures.service.outcome }),
  'an assignment probability above one');
refuses(() => serviceModel({ assignment: [0.5, 0.5], outcome: [[0.1, 0.1], [0.1, 2]] }),
  'a failure probability above one');
refuses(() => serviceModel({ ...fixtures.service, loadPrior: [0.4, 0.4] }),
  'a load prior that does not sum to one');

/* ================================================== §7 · the frontdoor */

const frontdoor = frontdoorModel();
nonEmpty(frontdoor.queries, 2, 'frontdoor queries');
vector(frontdoor.marginX, [0.5, 0.5], 'hiding U leaves an even treatment split', 1e-15);
vector(frontdoor.inner, [0.225, 0.7], 'the two inner mediator responses', 1e-15);
frontdoor.queries.forEach((query, index) => {
  const packet = recorded.frontdoor.queries[index];
  close(query.frontdoor, packet.frontdoor, `do(X=${index}) by the frontdoor formula`, 1e-14);
  close(query.truncated, packet.direct, `do(X=${index}) by the truncated factorisation`, 1e-14);
  close(query.observational, packet.observational, `and the plain observational probability`, 1e-14);
  close(query.frontdoor, query.truncated,
    `do(X=${index}): the two routes agree, which is what identification means here`, 1e-13);
  assert(Math.abs(query.frontdoor - query.observational) > 0.1,
    `do(X=${index}): and both differ substantially from conditioning`);
  record('frontdoor query');
});
close(frontdoor.queries[0].frontdoor, 0.2725, 'do(X=0) is .2725', 1e-14);
close(frontdoor.queries[1].frontdoor, 0.6525, 'do(X=1) is .6525', 1e-14);
assert(frontdoor.agreesWithTruncatedFactorisation, 'the model reports that agreement itself');
vector(frontdoor.observedRiskGivenMediatorAndTreatment[0], [0.12, 0.33],
  'the drawn M=0 row of the inner tray', 1e-13);
vector(frontdoor.observedRiskGivenMediatorAndTreatment[1], [0.58, 0.82],
  'the drawn M=1 row of the inner tray', 1e-13);
// The joint the observed summaries come from really is a distribution.
close(Object.values(frontdoor.joint).reduce((sum, value) => sum + value, 0), 1,
  'the 16 latent worlds sum to one', 1e-14);
record('frontdoor model');

assert(frontdoorOkChecks(), 'the three frontdoor conditions hold on the stated graph');
function frontdoorOkChecks() {
  const ok = frontdoorConditions(fixtures.frontdoorEdges, 'X', 'M', 'Y');
  assert(ok.intercepts, 'the mediator intercepts every directed path');
  assert.deepEqual(ok.directedPaths, [['X', 'M', 'Y']], 'and there is exactly one such path');
  assert.equal(ok.treatmentToMediatorOpen.length, 0, 'no backdoor path from X to M is open');
  assert.equal(ok.mediatorToOutcomeOpen.length, 0, 'conditioning on X blocks the M-to-Y backdoor');
  return ok.satisfied;
}
const latentMediator = frontdoorConditions(fixtures.frontdoorWithLatentMediator, 'X', 'M', 'Y');
assert(!latentMediator.satisfied, 'practice 9: adding U to M breaks the criterion');
assert(latentMediator.intercepts, 'and it is not the interception condition that broke');
assert.equal(latentMediator.treatmentToMediatorOpen.length, 1,
  'it is the X-to-M backdoor, which now has exactly one open path');
const directEffect = frontdoorConditions(fixtures.frontdoorWithDirectEffect, 'X', 'M', 'Y');
assert(!directEffect.satisfied, 'practice 9: adding X to Y also breaks it');
assert(!directEffect.intercepts, 'and this time it is the interception condition');
assert.equal(directEffect.directedPaths.length, 2, 'because there are now two directed paths');
assert.equal(directEffect.directedPaths.filter(path => path.includes('M')).length, 1,
  'and the mediator lies on only one of them');
record('frontdoor conditions');

/* ============================================= §7 · counterfactual pairs */

const pair = counterfactualPair();
nonEmpty(pair.models, 2, 'structural models');
pair.models.forEach(model => {
  nonEmpty(model.rows, 2, `${model.name} rows`);
  close(model.averageUnderZero, 0.5, `${model.name}: the do(X=0) average is one half`, 1e-15);
  close(model.averageUnderOne, 0.5, `${model.name}: the do(X=1) average is one half`, 1e-15);
  record('counterfactual model');
});
assert(pair.agreeOnAverages, 'the two models agree on every population average');
const modelA = pair.models[0];
const modelB = pair.models[1];
assert(modelA.rows.every(row => row.outcomeUnderZero === row.outcomeUnderOne),
  'in model A no unit’s outcome moves when the treatment is changed');
assert(modelB.rows.every(row => row.outcomeUnderZero !== row.outcomeUnderOne),
  'in model B every unit’s does');
const observedUnit = counterfactualOfUnit(0, 0, 1);
assert.equal(observedUnit[0].inferredExternalState, 0, 'both models infer U = 0 from X=0, Y=0');
assert.equal(observedUnit[1].inferredExternalState, 0, 'both of them');
assert.equal(observedUnit[0].prediction, 0, 'model A still predicts Y = 0');
assert.equal(observedUnit[1].prediction, 1, 'model B predicts Y = 1');
const practiceUnit = counterfactualOfUnit(1, 0, 0);
assert.equal(practiceUnit[0].inferredExternalState, 0, 'practice 10: model A infers U = 0');
assert.equal(practiceUnit[1].inferredExternalState, 1, 'practice 10: model B infers U = 1');
assert.equal(practiceUnit[0].prediction, 0, 'practice 10: model A predicts Y = 0');
assert.equal(practiceUnit[1].prediction, 1, 'practice 10: model B predicts Y = 1');
refuses(() => counterfactualOfUnit(2, 0, 1), 'an observed treatment that is not a state');
record('counterfactuals');

/* ================================================ §8 · the query families */

const mapQuery = queryFamilies(fixtures.queryMasses);
assert.equal(mapQuery.mostProbableWorld.query, 1, 'the most probable world has Q = 1');
assert.equal(mapQuery.marginalMapQuery, 0, 'while the marginal-MAP answer is Q = 0');
close(mapQuery.marginalMapMass, 0.6, 'whose rows combine to .60', 1e-15);
close(mapQuery.rowMasses[1], 0.4, 'against .40', 1e-15);
assert(!mapQuery.agree, 'so the two queries disagree on this distribution');
const practiceMap = queryFamilies(fixtures.practiceQueryMasses);
assert.equal(practiceMap.mostProbableWorld.query, 1, 'practice 8: the most probable world has Q = 1');
assert.equal(practiceMap.marginalMapQuery, 1, 'practice 8: and so does the marginal-MAP answer');
close(practiceMap.marginalMapMass, 0.55, 'practice 8: at mass .55', 1e-15);
assert(practiceMap.agree, 'practice 8: they coincide, which is the point of the question');
refuses(() => queryFamilies({ '00': 0.5, '01': 0.1, 10: 0.1, 11: 0.1 }), 'masses that do not sum to one');
record('query families');

/* ============================== §5 · the fitted models and their queries */

assert.equal(protocol.selected, 'NB', 'naive Bayes won the declared criterion');
assert(scores.naiveBayesValidation.logLoss < scores.treeAugmentedValidation.logLoss,
  'because its validation log loss is lower');
assert.equal(scores.naiveBayesValidation.correct, scores.treeAugmentedValidation.correct,
  'while the two accuracies tie, so accuracy could not have decided it');
close(scores.naiveBayesValidation.logLoss, recorded.wine.validation.NB.logLoss,
  'the recorded NB validation log loss', 1e-12);
close(scores.treeAugmentedValidation.logLoss, recorded.wine.validation.TAN.logLoss,
  'the recorded TAN validation log loss', 1e-12);
close(scores.selectedTest.logLoss, recorded.wine.finalTest.logLoss, 'the reserved test log loss', 1e-12);
assert.equal(scores.selectedTest.correct, recorded.wine.finalTest.correct, 'and its correct count');
assert.equal(finalModel.kind, 'NB', 'the refit model is the selected recipe');
assert.deepEqual(trainingModels.treeAugmented.parents, [null, 0, 0, 0],
  'the learned tree gives every other measurement alcohol as its feature parent');
nonEmpty(conditionalInformation, 6, 'conditional-information pairs');
conditionalInformation.forEach(entry => {
  assert(entry.informationNats >= -1e-12, 'a conditional mutual information is never negative');
  record('information pair');
});
// Every CPT row is a distribution.
[trainingModels.naiveBayes, trainingModels.treeAugmented, finalModel].forEach(model => {
  close(model.prior.reduce((sum, value) => sum + value, 0), 1, `${model.kind} prior sums to one`, 1e-12);
  nonEmpty(model.tables, 4, `${model.kind} tables`);
  model.tables.forEach((column, index) => {
    column.forEach((klass, klassIndex) => {
      klass.forEach((row, parentState) => {
        close(row[0] + row[1], 1,
          `${model.kind} column ${index}, class ${klassIndex}, parent state ${parentState} sums to one`, 1e-12);
        assert(row[0] > 0 && row[1] > 0,
          'the Dirichlet pseudo-count leaves no zero entry that a query could divide by');
        record('CPT row');
      });
    });
  });
});

// Every validation specimen, under every published measurement subset.
const maskKeys = { none: [], 0: [0], 2: [2], '0,2': [0, 2], '0,1,2,3': [0, 1, 2, 3] };
nonEmpty(validationSpecimens, 36, 'validation specimens');
validationSpecimens.forEach(specimen => {
  const packet = recorded.wine.validationRows.find(row => row.id === specimen.id);
  assert(packet, `specimen ${specimen.id} appears in the packet`);
  const bits = specimen.features.map((value, index) =>
    (value > trainingModels.treeAugmented.medians[index] ? 1 : 0));
  Object.entries(maskKeys).forEach(([key, visible]) => {
    const result = classPosterior(trainingModels.treeAugmented, bits, visible);
    vector(result.posterior, packet.tanPosteriors[key],
      `specimen ${specimen.id}, mask ${key}`, 1e-12);
    close(result.posterior.reduce((sum, value) => sum + value, 0), 1,
      `specimen ${specimen.id}, mask ${key}: the answer is a distribution`, 1e-12);
    assert.equal(result.compatibleStates, 2 ** (4 - visible.length),
      `specimen ${specimen.id}, mask ${key}: the compatible-state count is 2^(hidden)`);
    record('specimen query');
  });
});
// With nothing visible the conditional tables sum away.
vector(classPosterior(trainingModels.treeAugmented, [0, 0, 0, 0], []).posterior,
  trainingModels.treeAugmented.prior, 'an empty mask returns the class prior exactly', 1e-12);
vector(classPosterior(trainingModels.treeAugmented, [1, 1, 1, 1], []).posterior,
  trainingModels.treeAugmented.prior, 'whatever the hidden states happen to be', 1e-12);
record('empty mask');

// The alcohol edits: one crosses a median, one does not, one is hidden.
const editBits = alcohol => [alcohol, 5.19, 0.63, 7.9]
  .map((value, index) => (value > trainingModels.treeAugmented.medians[index] ? 1 : 0));
const crossed = classPosterior(trainingModels.treeAugmented, editBits(13.17), [0, 2]);
const belowMedian = classPosterior(trainingModels.treeAugmented, editBits(12.9), [0, 2]);
const stillBelow = classPosterior(trainingModels.treeAugmented, editBits(12.8), [0, 2]);
vector(crossed.posterior, recorded.changedAndNullChecks['alcohol13.17'][0],
  'the measured alcohol value', 1e-12);
vector(belowMedian.posterior, recorded.changedAndNullChecks['alcohol12.9'][0],
  'an edit that crosses the median', 1e-12);
assert.deepEqual(belowMedian.posterior, stillBelow.posterior,
  'a further edit inside the same bin is an exact null, not a small change');
assert.notEqual(crossed.leading, belowMedian.leading,
  'crossing the median changes which cultivar leads');
const hiddenEdit = classPosterior(trainingModels.treeAugmented, editBits(12.8), [2]);
const hiddenOriginal = classPosterior(trainingModels.treeAugmented, editBits(13.17), [2]);
assert.deepEqual(hiddenEdit.posterior, hiddenOriginal.posterior,
  'editing a hidden measurement is an exact null, because the value is summed over');
vector(hiddenEdit.posterior, recorded.changedAndNullChecks.hiddenEdit[0],
  'matching the packet', 1e-12);
record('measurement edits');

refuses(() => classPosterior(trainingModels.treeAugmented, [0, 0, 0], [0]), 'a short state vector');
refuses(() => classPosterior(trainingModels.treeAugmented, [0, 0, 0, 2], [0]), 'a state that is not 0 or 1');
refuses(() => classPosterior(trainingModels.treeAugmented, [0, 0, 0, 0], [7]), 'a measurement index out of range');

// Figure 4 recomputes both models on all 36 rows; its column means must be the
// recorded scores, or the figure is telling a different story from the table.
const figureRows = validationSpecimens.map(specimen => {
  const bits = specimen.features.map((value, index) =>
    (value > trainingModels.treeAugmented.medians[index] ? 1 : 0));
  const naive = classPosterior(trainingModels.naiveBayes, bits, [0, 1, 2, 3]);
  const tree = classPosterior(trainingModels.treeAugmented, bits, [0, 1, 2, 3]);
  return {
    naiveLoss: -Math.log(naive.posterior[specimen.cultivar]),
    treeLoss: -Math.log(tree.posterior[specimen.cultivar]),
    naiveCorrect: naive.leading === specimen.cultivar,
    treeCorrect: tree.leading === specimen.cultivar,
  };
});
nonEmpty(figureRows, 36, 'figure 4 rows');
close(figureRows.reduce((sum, row) => sum + row.naiveLoss, 0) / figureRows.length,
  scores.naiveBayesValidation.logLoss, 'figure 4’s NB column mean is the recorded log loss', 1e-12);
close(figureRows.reduce((sum, row) => sum + row.treeLoss, 0) / figureRows.length,
  scores.treeAugmentedValidation.logLoss, 'figure 4’s TAN column mean is the recorded log loss', 1e-12);
assert.equal(figureRows.filter(row => row.naiveCorrect).length, scores.naiveBayesValidation.correct,
  'and its NB correct count is the recorded one');
assert.equal(figureRows.filter(row => row.treeCorrect).length, scores.treeAugmentedValidation.correct,
  'and its TAN correct count likewise');
record('figure 4 reassembly');

/* ============================= the rules the investigations grade with */

// changeDirection at every degenerate input the labs can reach.
const directionCases = [
  [null, null, 'undefined', 'two absent values still have no defined result'],
  [null, 0.5, 'undefined', 'a value that disappeared'],
  [0.5, null, 'defined', 'a value that appeared'],
  [0.5, 0.5, 'unchanged', 'an exact tie'],
  [0, 0, 'unchanged', 'two exact zeros'],
  [1, 1, 'unchanged', 'two exact ones'],
  [0.5 + 1e-16, 0.5, 'unchanged', 'a difference below the tolerance'],
  [0.5 + 1e-6, 0.5, 'higher', 'a difference above it'],
  [0.5 - 1e-6, 0.5, 'lower', 'a fall above it'],
  [1e-12, 0, 'unchanged', 'a value at the tolerance itself'],
  [1e-6, 0, 'higher', 'a small rise from exactly zero'],
  [0, 1e-6, 'lower', 'a small fall to exactly zero'],
];
nonEmpty(directionCases, 12, 'change-direction cases');
directionCases.forEach(([after, before, expected, label]) => {
  assert.equal(changeDirection(after, before), expected, `changeDirection on ${label}`);
  record('change direction');
});
// It is symmetric in the sense that reversing a real change reverses the answer.
assert.equal(changeDirection(0.4, 0.6), 'lower', 'a fall reads as a fall');
assert.equal(changeDirection(0.6, 0.4), 'higher', 'and its reverse reads as a rise');
refuses(() => changeDirection(Number.NaN, 0.5), 'a change direction from a value that is not a number');

assert.equal(compareUncertainty(1, 2), 'first', 'less uncertainty on the first candidate');
assert.equal(compareUncertainty(2, 1), 'second', 'less on the second');
assert.equal(compareUncertainty(1, 1), 'equal', 'an exact tie is called equal, not arbitrary');
assert.equal(compareUncertainty(1, 1 + 1e-15), 'equal', 'and so is a tie below the tolerance');
record('uncertainty comparison');

// A reachable five-factor world has positive evidence mass below 10^-12.
// Support is a zero/nonzero question, independent of the comparison tolerance.
const rareNetwork = withCallerRow(withCallerRow(
  withRootChance(withRootChance(alarmNetwork, 'B', 0), 'E', 0.9999),
  'J', 1, 0.001), 'M', 1, 0.001);
const rareEvidence = { E: 0, A: 1, J: 1, M: 1 };
const rareQuery = queryPosterior(rareNetwork, rareEvidence);
const rareElimination = eliminationRun(rareNetwork, rareEvidence, []);
const rareMass = 1e-4 * 1e-3 * 1e-3 * 1e-3;
for (const [name, result] of [['enumeration', rareQuery], ['elimination', rareElimination]]) {
  assert.ok(result.evidenceProbability > 0 && result.evidenceProbability < 1e-12,
    `${name}: small supported evidence must remain positive`);
  assert.ok(Math.abs(result.evidenceProbability / rareMass - 1) < 2e-12,
    `${name}: independent four-factor mass ${rareMass}`);
  assert.equal(result.posterior, 0, `${name}: certain absence is a defined zero posterior`);
  record('sequential rare supported evidence');
}

// purchaseComparison over every specimen, every mask and every candidate pair.
let purchaseCases = 0;
const purchaseOutcomes = { first: 0, second: 0, equal: 0 };
let entropyRose = 0;
validationSpecimens.forEach(specimen => {
  const bits = specimen.features.map((value, index) =>
    (value > trainingModels.treeAugmented.medians[index] ? 1 : 0));
  subsets([0, 1, 2, 3]).forEach(visible => {
    const hidden = [0, 1, 2, 3].filter(index => !visible.includes(index));
    if (hidden.length < 2) return;
    const before = classPosterior(trainingModels.treeAugmented, bits, visible);
    hidden.forEach(first => hidden.forEach(second => {
      if (first >= second) return;
      const comparison = purchaseComparison(trainingModels.treeAugmented, bits, visible, [first, second]);
      // The graded outcome must equal a direct comparison of the two entropies.
      const firstEntropy = entropyNats(
        classPosterior(trainingModels.treeAugmented, bits, [...visible, first].sort((a, b) => a - b)).posterior);
      const secondEntropy = entropyNats(
        classPosterior(trainingModels.treeAugmented, bits, [...visible, second].sort((a, b) => a - b)).posterior);
      const expected = Math.abs(firstEntropy - secondEntropy) <= unchangedTolerance
        ? 'equal' : firstEntropy < secondEntropy ? 'first' : 'second';
      assert.equal(comparison.outcome, expected,
        `specimen ${specimen.id}, visible {${visible.join(',')}}, candidates ${first} and ${second}`);
      close(comparison.first.entropyNats, firstEntropy, 'the reported first entropy', 1e-12);
      close(comparison.second.entropyNats, secondEntropy, 'the reported second entropy', 1e-12);
      if (firstEntropy > before.entropyNats + 1e-9) entropyRose += 1;
      purchaseOutcomes[comparison.outcome] += 1;
      purchaseCases += 1;
    }));
  });
});
assert(purchaseCases > 400, `the purchase sweep covered ${purchaseCases} cases, not a sample`);
assert(purchaseOutcomes.first > 0 && purchaseOutcomes.second > 0,
  'the sweep produces both answers, so the grader is not constant');
// The lab tells a learner that revealing a measurement can RAISE uncertainty.
assert(entropyRose > 0,
  'the claim that a single reveal can raise uncertainty must be exhibited somewhere in the reachable grid');
// Choosing the same candidate twice is a tie by construction, not by arithmetic.
assert.equal(purchaseComparison(trainingModels.treeAugmented, [1, 0, 0, 1], [], [0, 0]).outcome, 'equal',
  'the same candidate chosen twice is called equal');
record('purchase sweep');

/* ================== every value the prose asks a learner to type is typeable */

/** A number a control cannot accept is an instruction a learner cannot follow. */
const onStep = (value, decimals) => {
  const scale = 10 ** decimals;
  return Math.abs(value * scale - Math.round(value * scale)) < 1e-9;
};
/* Every entry names the control it would actually be typed into, and the
 * control's own declaration supplies the step and range. The previous version
 * hard-coded both in the verifier, so the check agreed with a copy of the
 * control rather than with the control -- and it swept four alarm-CPT rows
 * against an editor the page does not have. */
const typeable = [
  { value: alarmNetwork.chance.B[''], control: 'rootPrior', where: 'the burglary prior' },
  { value: alarmNetwork.chance.E[''], control: 'rootPrior', where: 'the earthquake prior' },
  { value: alarmNetwork.chance.J[0], control: 'callerRow', where: "John's false-call chance" },
  { value: alarmNetwork.chance.J[1], control: 'callerRow', where: 'John when the alarm sounds' },
  { value: alarmNetwork.chance.M[0], control: 'callerRow', where: "Mary's false-call chance" },
  { value: alarmNetwork.chance.M[1], control: 'callerRow', where: 'Mary when the alarm sounds' },
  { value: fixtures.falseCallJohn, control: 'callerRow', where: 'the raised false-call chance' },
  { value: fixtures.uninformativeCallRow, control: 'callerRow', where: 'the uninformative caller row' },
  ...fixtures.service.assignment.map(value => ({ value, control: 'serviceProbability', where: 'an assignment probability' })),
  ...fixtures.serviceRandomised.assignment.map(value => ({ value, control: 'serviceProbability', where: 'a randomised assignment' })),
  ...fixtures.serviceNoOverlap.assignment.map(value => ({ value, control: 'serviceProbability', where: 'a no-overlap assignment' })),
  ...fixtures.service.outcome.flat().map(value => ({ value, control: 'serviceProbability', where: 'a failure-table entry' })),
  ...fixtures.serviceChangedResponse.outcome.flat()
    .map(value => ({ value, control: 'serviceProbability', where: 'a changed failure-table entry' })),
  ...fixtures.alcoholEdits.map(value => ({ value, control: 'measurement', where: 'an alcohol edit' })),
];
nonEmpty(typeable, undefined, 'values the prose asks a learner to enter');
typeable.forEach(entry => {
  const control = controlSteps[entry.control];
  assert(control, `${entry.where}: names a control that is not declared`);
  assert(onStep(entry.value, control.decimals),
    `${entry.where}: ${entry.value} needs more than ${control.decimals} decimals, so ${entry.control} cannot accept it`);
  assert(entry.value >= control.minimum && entry.value <= control.maximum,
    `${entry.where}: ${entry.value} lies outside ${entry.control}'s range`);
  record('typeable value');
});
// Every declared step is consistent with the decimals it claims.
Object.entries(controlSteps).forEach(([name, control]) => {
  assert(onStep(Number(control.step), control.decimals),
    `${name}: its step ${control.step} is finer than the ${control.decimals} decimals it accepts`);
  record('declared control');
});
/* And the component uses the declaration rather than a copy of it. A number
 * field that spelled its own step out would pass every check above while
 * accepting something the sweep never tested. */
const labSource = fs.readFileSync('src/learn/components/lesson-labs/BayesNetLabs.jsx', 'utf8');
const numberFields = labSource.match(/<NumberField[\s\S]*?\/>/g) ?? [];
nonEmpty(numberFields, 13, 'number fields in the investigations');
numberFields.forEach(field => {
  assert(/\{\.\.\.controlSteps\./.test(field),
    `a NumberField configures itself instead of using controlSteps: ${field.slice(0, 80)}`);
  assert(!/decimals=\{/.test(field), 'a NumberField still hard-codes its decimals');
  record('field uses the declaration');
});
// The controls really do refuse a value with more decimals than they declare.
assert(!onStep(0.0005, 3), 'a four-decimal value is correctly rejected by a three-decimal control');
assert(onStep(0.0005, 4), 'and accepted by a four-decimal one');
record('control steps');

/* ================================================= geometry a figure draws */

/* The shapes the COMPONENT draws, written out here rather than obtained from
 * `nodeKeepOut`. The router takes its obstacles from that function, so a
 * verifier that also called it would lose a shape and gain an agreement: the
 * falsification harness removed the endpoint badge from `nodeKeepOut` and the
 * whole suite stayed green. These numbers come from BayesNetShared's own
 * drawing code -- badge at (x - r - 5, y - r - 5) sized 2r + 10, observed tag
 * at y + r + 13 -- and the equality with `nodeKeepOut` is then asserted
 * explicitly, so a change to either side is a failure rather than a match. */
const drawnShapes = (point, radius) => ([
  { kind: 'circle', x: point.x, y: point.y, radius },
  { kind: 'rect', x: point.x, y: point.y, halfWidth: radius + 5, halfHeight: radius + 5 },
  { kind: 'rect', x: point.x, y: point.y + radius + 10, halfWidth: 24, halfHeight: 7 },
]);
{
  const sample = { x: 100, y: 80 };
  const fromModule = nodeKeepOut(sample, 17, { endpoint: true, observed: true });
  assert.deepEqual(fromModule, drawnShapes(sample, 17),
    'the keep-out shapes the router is given are the shapes the component draws');
  assert.equal(nodeKeepOut(sample, 17).length, 1, 'a plain node keeps out only its circle');
  assert.equal(nodeKeepOut(sample, 17, { endpoint: true }).length, 2, 'a query endpoint adds its badge');
  record('keep-out shapes');
}

let geometryCases = 0;
drawableGraphs.forEach(([name, edges]) => {
  [190, 220, 250, 280, 300].forEach(width => {
    const layout = layeredLayout(edges, { width });
    const nodes = graphNodes(edges);
    assert.equal(Object.keys(layout.positions).length, nodes.length,
      `${name} at ${width}: every node is placed`);
    // No two circles overlap, and everything sits inside the drawn box.
    nodes.forEach(left => nodes.forEach(right => {
      if (left >= right) return;
      const a = layout.positions[left];
      const b = layout.positions[right];
      const gap = Math.hypot(a.x - b.x, a.y - b.y);
      assert(gap >= 2 * layout.radius,
        `${name} at ${width}: ${left} and ${right} are ${gap.toFixed(2)} apart with radius ${layout.radius.toFixed(2)}`);
      geometryCases += 1;
    }));
    nodes.forEach(node => {
      const point = layout.positions[node];
      assert(point.x - layout.radius >= 0 && point.x + layout.radius <= width,
        `${name} at ${width}: ${node} stays inside the horizontal bounds`);
      assert(point.y - layout.radius >= 0 && point.y + layout.radius <= layout.height,
        `${name} at ${width}: ${node} stays inside the vertical bounds`);
    });
    // Every drawn edge starts on one circle, ends on the other, and — the part
    // that a straight-line drawing gets wrong — misses every node it does not
    // join. A layered layout puts a whole chain in one column, so a skipping
    // edge drawn straight runs through the nodes between its endpoints and
    // through their labels: correct in the DOM, wrong on screen.
    // Every node is treated as a badged, observed node here: that is the
    // largest shape the lesson can draw for it, so a route that clears this
    // clears every state the lab can put the graph into.
    const routes = graphRoutes(edges, layout.positions, layout.radius, { endpoints: nodes, observed: nodes });
    nonEmpty(routes, edges.length, `${name} at ${width}: drawn edges`);
    routes.forEach(({ from, to, route }) => {
      const start = layout.positions[from];
      const end = layout.positions[to];
      close(Math.hypot(route.start.x - start.x, route.start.y - start.y), layout.radius,
        `${name} at ${width}: the ${from} to ${to} edge starts on the source circle`, 1e-6);
      close(Math.hypot(route.tip.x - end.x, route.tip.y - end.y), layout.radius,
        `${name} at ${width}: and its arrow tip touches the target circle`, 1e-6);
      assert.equal(route.arrow.length, 3, 'an arrowhead is a triangle');
      // The arrowhead really does point along the curve's tangent at the tip:
      // its two base corners straddle the shaft end perpendicular to `unit`.
      const across = { x: route.arrow[1][0] - route.arrow[2][0], y: route.arrow[1][1] - route.arrow[2][1] };
      close(Math.hypot(across.x, across.y), 5.5, `${name} at ${width}: the arrowhead base is one width across`, 1e-9);
      close(across.x * route.unit.x + across.y * route.unit.y, 0,
        `${name} at ${width}: and that base is perpendicular to the tangent at the tip`, 1e-9);
      const alongTip = { x: route.tip.x - route.shaft.to.x, y: route.tip.y - route.shaft.to.y };
      assert(alongTip.x * route.unit.x + alongTip.y * route.unit.y > 0,
        `${name} at ${width}: the ${from} to ${to} arrowhead points toward ${to}`);

      // The path string the browser renders must be the curve `trace` samples.
      // These were two different curves once, diverging by nine pixels, which
      // is how an edge crossed a badge behind a green run.
      const rebuilt = `M ${route.shaft.from.x.toFixed(2)} ${route.shaft.from.y.toFixed(2)} `
        + `Q ${route.shaft.control.x.toFixed(2)} ${route.shaft.control.y.toFixed(2)} `
        + `${route.shaft.to.x.toFixed(2)} ${route.shaft.to.y.toFixed(2)}`;
      assert.equal(route.path, rebuilt,
        `${name} at ${width}: the drawn path is the sub-curve trace samples`);
      const onCurve = (t) => {
        const inverse = 1 - t;
        return [
          inverse * inverse * route.shaft.from.x + 2 * inverse * t * route.shaft.control.x + t * t * route.shaft.to.x,
          inverse * inverse * route.shaft.from.y + 2 * inverse * t * route.shaft.control.y + t * t * route.shaft.to.y,
        ];
      };
      [0, 0.25, 0.5, 0.75, 1].forEach(t => {
        const [x, y] = onCurve(t);
        const nearest = route.trace.reduce((low, [tx, ty]) => {
          const distance = Math.hypot(tx - x, ty - y);
          return distance < low ? distance : low;
        }, Infinity);
        assert(nearest < 0.5,
          `${name} at ${width}: trace follows the drawn path at t=${t} (nearest sample ${nearest.toFixed(3)})`);
        geometryCases += 1;
      });

      // Clearance, measured on the drawn geometry against the drawn shapes, at
      // the router's own threshold rather than a looser one.
      assert(route.cleared,
        `${name} at ${width}: no route for ${from} to ${to} clears the shapes in between`);
      Object.entries(layout.positions).forEach(([node, point]) => {
        if (node === from || node === to) return;
        drawnShapes(point, layout.radius).forEach(shape => {
          const nearest = route.trace.reduce((low, [x, y]) => {
            const gap = distanceToShape(shape, x, y);
            return gap < low ? gap : low;
          }, Infinity);
          assert(nearest >= route.clearance,
            `${name} at ${width}: the ${from} to ${to} edge passes ${nearest.toFixed(2)} from ${node}'s `
            + `${shape.kind}, inside the ${route.clearance} it is routed to clear — it would be drawn `
            + 'through that node or its label');
          geometryCases += 1;
        });
      });
      // Nothing routed off the canvas to achieve that clearance.
      route.trace.forEach(([x, y]) => {
        assert(x >= 0 && x <= width && y >= 0 && y <= layout.height,
          `${name} at ${width}: the ${from} to ${to} edge leaves the drawn box at ${x.toFixed(1)}, ${y.toFixed(1)}`);
      });
      geometryCases += 1;
    });
    record('layout swept');
  });
});
assert(geometryCases > 200, `the geometry sweep covered ${geometryCases} checks`);
// The two intervention panels are drawn at declared positions rather than a
// computed layout, so that cutting one arrow changes one arrow and nothing
// else. Hand-placed coordinates get the same checks as computed ones.
const servicePanels = [
  ['observation', fixtures.serviceEdges],
  ['intervention', fixtures.serviceEdges.filter(([, to]) => to !== 'X')],
];
nonEmpty(servicePanels, 2, 'intervention panels');
assert.deepEqual(Object.keys(fixtures.servicePositions).sort(), ['X', 'Y', 'Z'],
  'the declared positions cover exactly the three service variables');
servicePanels.forEach(([panel, edges]) => {
  const { width, height, radius } = fixtures.serviceCanvas;
  const nodes = Object.keys(fixtures.servicePositions);
  nodes.forEach(left => nodes.forEach(right => {
    if (left >= right) return;
    const a = fixtures.servicePositions[left];
    const b = fixtures.servicePositions[right];
    assert(Math.hypot(a.x - b.x, a.y - b.y) >= 2 * radius,
      `${panel}: ${left} and ${right} do not overlap`);
  }));
  nodes.forEach(node => {
    const point = fixtures.servicePositions[node];
    assert(point.x - radius >= 0 && point.x + radius <= width
      && point.y - radius >= 0 && point.y + radius <= height,
      `${panel}: ${node} stays inside the declared canvas`);
  });
  graphRoutes(edges, fixtures.servicePositions, radius, { endpoints: ['X', 'Y'], observed: nodes })
    .forEach(({ from, to, route }) => {
      assert(route.cleared, `${panel}: the ${from} to ${to} edge finds a clearing route`);
      nodes.forEach(node => {
        if (node === from || node === to) return;
        drawnShapes(fixtures.servicePositions[node], radius).forEach(shape => {
          const nearest = route.trace.reduce((low, [x, y]) => {
            const gap = distanceToShape(shape, x, y);
            return gap < low ? gap : low;
          }, Infinity);
          assert(nearest >= route.clearance,
            `${panel}: the ${from} to ${to} edge clears ${node}'s ${shape.kind} by ${nearest.toFixed(2)}`);
          geometryCases += 1;
        });
      });
      geometryCases += 1;
    });
  record('intervention panel geometry');
});
// Both panels place every node identically; only the arrow set differs.
assert.equal(
  fixtures.serviceEdges.length - servicePanels[1][1].length, 1,
  'the intervention panel removes exactly one arrow');

// A highlighted path is drawn through the node centres it names.
const highlightLayout = layeredLayout(fixtures.educationEdges, { width: 280 });
const trail = pathPolyline(['T', 'E', 'S', 'Y'], highlightLayout.positions);
assert.equal(trail.length, 4, 'the drawn trail has one point per node on the path');
trail.forEach((point, index) => {
  const node = ['T', 'E', 'S', 'Y'][index];
  close(point[0], highlightLayout.positions[node].x, `the trail passes through ${node}`, 1e-12);
  close(point[1], highlightLayout.positions[node].y, `at its centre`, 1e-12);
  record('trail point');
});
refuses(() => pathPolyline(['T', 'Q'], highlightLayout.positions), 'a trail through a node with no position');
refuses(() => edgeRoute({ x: 0, y: 0 }, { x: 5, y: 0 }, 17), 'an edge shorter than the two circles');
// An edge walled in on both sides reports that it could not clear, rather than
// throwing: a crashed page is worse than a tight edge, and `cleared` is what
// every drawn graph is asserted on.
const walledIn = edgeRoute({ x: 0, y: 0 }, { x: 0, y: 120 }, 17,
  Array.from({ length: 40 }, (unused, index) => (
    { kind: 'circle', x: index * 10 - 195, y: 60, radius: 17 })));
assert.equal(walledIn.cleared, false, 'an edge with no clearing route says so instead of throwing');
assert(walledIn.path.startsWith('M '), 'and still returns something drawable');
record('unclearable edge');
record('geometry');

/* ============================== the published module and the programs */

assert.equal(provenance.bytes, 12100, 'the served dataset is the size the attribution claims');
assert.equal(provenance.sha256,
  crypto.createHash('sha256').update(fs.readFileSync('public/learn-assets/bayesian-networks/wine.csv')).digest('hex'),
  'and its bytes hash to the recorded digest');
assert.equal(provenance.specimens, 178, 'it holds 178 specimens');
assert(fs.existsSync('public/learn-assets/bayesian-networks/ATTRIBUTION.txt'),
  'the attribution file is served beside it');
assert.equal(
  crypto.createHash('sha256')
    .update(fs.readFileSync('public/learn-assets/bayesian-networks/network-experiments.py')).digest('hex'),
  crypto.createHash('sha256')
    .update(fs.readFileSync(`${packetDirectory}/network-experiments.py`)).digest('hex'),
  'the downloadable experiment is the packet program, byte for byte');
assert.equal(protocol.trainSize + protocol.validationSize + protocol.testSize, provenance.specimens,
  'the three roles cover every specimen');
const roleIds = [...protocol.trainIds, ...protocol.validationIds, ...protocol.testIds];
assert.equal(new Set(roleIds).size, provenance.specimens, 'and do not overlap');
record('data module shape');

const programKeys = ['enumerate', 'pgmpy', 'experiment'];
nonEmpty(programKeys, 3, 'displayed and downloadable programs');
programKeys.forEach(key => {
  const program = examples[key];
  assert(program, `the examples module holds the ${key} program`);
  assert(program.code.length > 200, `${key}: the recorded code is a real program`);
  assert(program.executed, `${key}: it was actually executed`);
  assert(program.environment && program.environment.python,
    `${key}: the environment it ran in is recorded, not typed into the prose`);
  if (key !== 'experiment') {
    assert(typeof program.expected === 'string' && program.expected.length > 0,
      `${key}: its recorded output is real text`);
  }
  record('program record');
});
assert(examples.enumerate.expected.trim().startsWith('0.284171'),
  'the enumeration program printed the posterior the lesson quotes');
assert(examples.pgmpy.expected.includes('0.2841'),
  'and the library route printed the same number');
assert(examples.pgmpy.code.includes('VariableElimination'),
  'the library route really calls the library');
assert(examples.experiment.downloadOnly,
  'the full experiment is offered as a download rather than a displayed block');
// The displayed programs are the packet's own bytes, not a transcription.
const packetPgmpy = fs.readFileSync(`${packetDirectory}/pgmpy-example.py`, 'utf8')
  .replace(/\r\n/g, '\n').replace(/\n+$/, '');
assert.equal(examples.pgmpy.code, packetPgmpy,
  'the displayed pgmpy program is the packet file verbatim');

// The page tells a reader that isolating pgmpy was necessary because resolving
// it moves NumPy and pandas. That is a factual claim about two environments,
// and it is checked here rather than asserted in prose.
assert.equal(examples.pgmpy.runtime, 'isolated', 'the library route ran in the isolated environment');
assert.equal(examples.enumerate.runtime, 'shared', 'the enumeration ran in the shared one');
assert(examples.pgmpy.environment.pgmpy, 'the isolated environment names its pgmpy version');
assert.notEqual(examples.pgmpy.environment.numpy, examples.enumerate.environment.numpy,
  'the isolated NumPy really does differ from the shared one, which is the reason given on the page');
assert.notEqual(examples.pgmpy.environment.pandas, examples.enumerate.environment.pandas,
  'and so does pandas');
record('program provenance');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/bayesnet-models.js',
  'src/learn/data/bayesnet-data.js',
  'src/learn/data/bayesnet-examples.js',
  'src/learn/components/lesson-labs/BayesNetShared.jsx',
  'src/learn/components/lesson-labs/BayesNetLabs.jsx',
  'src/learn/components/lesson-labs/BayesNetFigures.jsx',
  'src/learn/components/lesson-labs/bayesnet-labs.css',
  'src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx',
  'src/learn/data/curriculum/blueprints/bayesian-networks-causal-graphical-models.js',
  'public/learn-assets/bayesian-networks/wine.csv',
  'public/learn-assets/bayesian-networks/ATTRIBUTION.txt',
  'public/learn-assets/bayesian-networks/network-experiments.py',
  'scripts/verify-bayesnet-escapes.py',
];
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sumValue, value) => sumValue + value, 0);
/* A counter that is reported but never floored is decoration: deleting every
 * `record()` call still printed PASS with a smaller number. These floors sit
 * below the current values but far above zero, so a wholesale loss of coverage
 * fails the run instead of quietly shrinking the headline. */
assert(total >= 450, `only ${total} grouped checks ran; the suite has lost coverage`);
assert(Object.keys(counts).length >= 50, `only ${Object.keys(counts).length} groups ran`);
assert(separationCases >= 1000, `only ${separationCases} preset separation cases ran`);
assert(generatedCases >= 100000, `only ${generatedCases} generated separation cases ran`);
assert(geometryCases >= 2000, `only ${geometryCases} geometry checks ran`);
assert(purchaseCases >= 800, `only ${purchaseCases} measurement comparisons ran`);
assert(backdoorCases >= 24, `only ${backdoorCases} backdoor candidate sets ran`);
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  packetCalculatedInputsSha256: hash(`${packetDirectory}/calculated-inputs.json`),
  verifierHash: hash('scripts/verify-bayesnet-models.mjs'),
  counts,
  totalGroupedChecks: total,
  separationCasesSwept: separationCases,
  generatedGraphsSwept: generatedGraphCount,
  generatedSeparationCasesSwept: generatedCases,
  generatedVerdictsSeen: { separated: generatedSeparated, dependencePossible: generatedConnected },
  separationVerdictsSeen: { separated: separatedCases, dependencePossible: connectedCases },
  backdoorCandidateSetsSwept: backdoorCases,
  purchaseComparisonsSwept: purchaseCases,
  geometryChecks: geometryCases,
  smallestGenuinePosteriorChange: nullMargin,
  unchangedTolerance,
  scope: 'Browser Bayesian-network models against the content packet\'s calculated-inputs.json and every number and '
    + 'graphical claim the lesson states. The alarm network is enumerated twice by independent bitmask walks and its '
    + 'seven published posteriors, evidence probabilities and compatible-world counts checked against both; '
    + 'impossible evidence, an uninformative caller row and a table row the evidence never reads are exercised as '
    + 'the degenerate cases they are, and the margin between the smallest genuine change and the unchanged '
    + 'tolerance is measured rather than assumed. Variable elimination is checked against enumeration and against '
    + 'the packet trace, with factor multiplication and summation shown to commute and both chain orders’ '
    + 'induced widths and fill edges asserted. Every d-separation verdict is recomputed by ancestral moralisation '
    + 'rather than by the module’s own path enumeration, over every ordered endpoint pair and every subset of '
    + 'the remaining nodes on all eleven graphs the page can draw; the backdoor criterion is likewise recomputed by '
    + 'separation in the graph with the treatment’s outgoing arrows removed, over every candidate set on three '
    + 'graphs. The destination note’s finding is checked from the path enumeration itself: the note’s graph '
    + 'has exactly two T-to-Y paths, one backdoor route, and {E}, {S} and {E,S} are all valid while the empty set is '
    + 'not. The service model is checked at its declared, randomised, changed-response, no-overlap and nobody-treated '
    + 'settings against an independently built eight-cell joint, and its drawn lane widths are shown to reproduce the '
    + 'printed risks. The frontdoor model is checked by both identification routes and its three conditions checked '
    + 'against three graphs. Both grading rules the investigations use are exercised at their degenerate inputs, and '
    + 'the measurement comparison is swept over every specimen, mask and candidate pair with the claim that a reveal '
    + 'can raise uncertainty exhibited rather than asserted. Every drawn node position, edge endpoint and arrowhead '
    + 'is checked at five widths for overlap, bounds and direction.',
  limitations: [
    'The per-path blocked flags are checked against the definition directly; it is the overall separation verdict '
      + 'that is recomputed by a second algorithm.',
    'The Wine results are precomputed native fits; the browser reproduces their recorded outcomes, not the fitting. '
      + 'That regeneration is checked separately by scripts/verify-bayesnet-data.py.',
    'Displayed program output is executed separately by scripts/verify-bayesnet-examples.py.',
    'Rendering, interaction, visual layout and independent review are separate steps and are not claimed here.',
  ],
  passed: true,
};
/* `--no-evidence` lets an independent reviewer re-run this without writing to
 * the record they are reviewing. The Python verifiers are read-only by default
 * for their generated modules; this is the same courtesy for the evidence. */
if (!process.argv.includes('--no-evidence')) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync('docs/teaching/evidence/bayesnet-models.json', JSON.stringify(evidence, null, 2) + '\n');
}
console.log(`PASS: ${total} grouped Bayesian-network model checks across ${Object.keys(counts).length} groups, `
  + `including ${separationCases.toLocaleString('en-US')} preset d-separation cases and `
  + `${generatedCases.toLocaleString('en-US')} more on ${generatedGraphCount} generated graphs, all recomputed by a `
  + `second algorithm (${separatedCases} separated, ${connectedCases} not among the presets), `
  + `${backdoorCases} backdoor candidate sets, `
  + `${purchaseCases.toLocaleString('en-US')} measurement comparisons and ${geometryCases.toLocaleString('en-US')} `
  + `geometry checks.`);
