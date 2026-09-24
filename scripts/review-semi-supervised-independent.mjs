/** Complementary reviewer checks: analytical networks, invariants, and rendered changed cases. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { parseGraph, propagateGraph, prototypeTraining, parsePrototype, categoricalCoTraining } from '../src/learn/data/semi-supervised-models.js';

const checks = [];
let comparisons = 0;
function close(actual, expected) { comparisons += 1; assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) < 2e-10, `${actual} != ${expected}`); }
// The voltage drop along a series circuit is proportional to resistance 1/w.
for (const conductances of [[1, 1, 1], [1, 2, 4], [0.1, 3, 0.2], [5, 0.002, 1]]) {
  const graph = parseGraph('A,0\nB,?\nC,?\nD,1', conductances.map((weight, i) => `${'ABCD'[i]},${'ABCD'[i + 1]},${weight}`).join('\n'));
  const resistance = conductances.map(weight => 1 / weight);
  const total = resistance.reduce((a, b) => a + b);
  const solved = propagateGraph(graph);
  close(solved.scores[1], resistance[0] / total);
  close(solved.scores[2], (resistance[0] + resistance[1]) / total);
  for (const factor of [1e-300, 1e-14, 1e-4, 1]) {
    const changed = { ...graph, edges: graph.edges.map(([a, b, weight]) => [a, b, weight * factor]) };
    const result = propagateGraph(changed);
    result.scores.forEach((score, i) => close(score, solved.scores[i]));
  }
}
checks.push('Series-resistance reference and global conductance-scale invariance, including the previously rejected 1e-14 case');

// Three-node star: hard score = b/(a+b); symmetric soft normalization instead
// gives a class-1 readout sqrt(b)/(sqrt(a)+sqrt(b)). Solve the three equations
// by substitution, without either production solver or its iterative trace.
for (const a of [0.02, 0.3, 1, 5]) for (const b of [0.02, 0.3, 1, 5]) {
  const graph = parseGraph('A,0\nU,?\nB,1', `A,U,${a}\nU,B,${b}`);
  close(propagateGraph(graph).scores[1], b / (a + b));
  for (const alpha of [0.2, 0.8, 0.99]) {
    const result = propagateGraph(graph, 'soft', alpha);
    close(result.scores[1], Math.sqrt(b) / (Math.sqrt(a) + Math.sqrt(b)));
    close(result.equilibrium[1][0], alpha / (1 + alpha) * Math.sqrt(a / (a + b)));
    close(result.equilibrium[1][1], alpha / (1 + alpha) * Math.sqrt(b / (a + b)));
  }
}
checks.push('Independent star-network hard/soft formulas and raw-versus-normalized distinction');

const ambiguousRows = [{ views: ['a', 'left'], label: 0 }, { views: ['b', 'right'], label: 1 }, { views: ['a', 'shared'], label: null }, { views: ['b', 'shared'], label: null }];
for (const rows of [ambiguousRows, [...ambiguousRows].reverse(), ambiguousRows.map(row => ({ ...row, views: [...row.views].reverse() }))]) {
  const result = categoricalCoTraining(rows);
  for (const round of result.history) {
    const newRules = round.rulesAfter.reduce((sum, rules, view) => sum + Object.keys(rules).filter(key => !Object.hasOwn(round.rulesBefore[view], key)).length, 0);
    assert.equal(round.newRules, newRules, 'Count supported rules after refitting, not just offered categories');
    rows.forEach((row, i) => { if (row.label !== null) round.labelsAfter.forEach(labels => assert.equal(labels[i], row.label)); });
  }
  assert.equal(result.history[0].newRules, 0);
}
checks.push('Contradictory same-recipient-category offers, row/view permutations, and observed-label conservation');

const input = parsePrototype('-2,0\n2,1', '-1,0,1,3', 0.8, 1.25);
const original = prototypeTraining(input);
const shifted = prototypeTraining({ ...input, observed: input.observed.map(point => ({ ...point, x: point.x + 2 })), pool: input.pool.map(x => x + 2), query: input.query + 2 });
original.final.forEach((value, i) => close(shifted.final[i], value + 2));
assert.equal(shifted.queryChange, original.queryChange);
for (const emptyPool of [[], [0, 0]]) {
  const result = prototypeTraining({ ...input, pool: emptyPool });
  assert.equal(result.movement, 'unchanged');
  assert.equal(result.queryChange, 'no');
  assert.equal(result.history[0].accepted.length, 0);
}
checks.push('Prototype translation symmetry and empty/rejected-pool nulls with final refit');

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const screenshots = [];
const errors = [];
try {
  for (const width of [1366, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 980 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184'}/learn/path/full-curriculum/semi-supervised-learning-label-propagation-self-training-co-training?module=classical-ml`);
    await page.locator('.ssl-lesson').waitFor();
    const graph = page.locator('[data-ssl-lab="graph"]');
    assert.equal(await graph.locator('.ssl-result').count(), 0);
    await graph.getByLabel('Nodes · name, observed label (0, 1 or ?)', { exact: true }).fill('A,0\nB,?\nC,1');
    await graph.getByLabel('Edges · from, to, weight (0 removes; maximum 5)', { exact: true }).fill('A,B,1e-14\nB,C,1e-14');
    await graph.getByLabel('My final score prediction', { exact: true }).selectOption('tie');
    await graph.getByRole('button', { name: 'Commit prediction and calculate', exact: true }).click();
    await graph.getByRole('button', { name: 'Reveal equilibrium and feedback', exact: true }).click();
    assert.match(await graph.locator('.ssl-feedback').innerText(), /Prediction matched.*B is 0\.5/s);

    const cotrain = page.locator('[data-ssl-lab="co-training"]');
    await cotrain.getByLabel('Paired rows · view 1 | view 2 | observed label', { exact: true }).fill('a|left|0\nb|right|1\na|shared|?\nb|shared|?');
    await cotrain.getByLabel('Recipient view', { exact: true }).selectOption('1');
    await cotrain.getByLabel('My final rule prediction', { exact: true }).selectOption('unknown');
    await cotrain.getByRole('button', { name: 'Commit prediction and calculate', exact: true }).click();
    await cotrain.getByRole('button', { name: 'Next transfer stage', exact: true }).click();
    await cotrain.getByRole('button', { name: 'Next transfer stage', exact: true }).click();
    assert.match(await cotrain.innerText(), /Distinct newly reachable category rules this round: 0/);
    await cotrain.getByRole('button', { name: 'Reveal final rules and feedback', exact: true }).click();
    assert.match(await cotrain.locator('.ssl-feedback').innerText(), /Prediction matched/);
    await cotrain.getByLabel('Recipient view', { exact: true }).selectOption('0');
    assert.equal(await cotrain.locator('.ssl-result').count(), 0);

    const prototype = page.locator('[data-ssl-lab="self-training"]');
    await prototype.getByRole('button', { name: 'Move 3 to 9', exact: true }).click();
    await prototype.getByLabel('My boundary movement prediction', { exact: true }).selectOption('right');
    await prototype.getByLabel('Does the query class change?', { exact: true }).selectOption('yes');
    await prototype.getByRole('button', { name: 'Commit prediction and calculate', exact: true }).click();
    await prototype.getByRole('button', { name: 'Reveal final model and feedback', exact: true }).click();
    const offsets = await prototype.evaluate(element => {
      const svg = element.querySelector('svg');
      return [...element.querySelectorAll('[data-ssl-tick]')].map(tick => {
        const point = new DOMPoint(25 + (Number(tick.dataset.sslTick) + 10) / 20 * 550, 0).matrixTransform(svg.getScreenCTM());
        const rect = tick.getBoundingClientRect();
        return Math.abs(point.x - (rect.left + rect.width / 2));
      });
    });
    assert.equal(offsets.length, 5);
    assert.ok(offsets.every(offset => offset < 0.1), `Tick-to-coordinate offsets: ${offsets}`);
    await prototype.scrollIntoViewIfNeeded();
    fs.mkdirSync('scratch/semi-supervised-independent', { recursive: true });
    const path = `scratch/semi-supervised-independent/prototype-${width}.png`;
    await prototype.screenshot({ path, style: '.learn-nav { visibility: hidden !important; }' });
    screenshots.push({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex') });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    checks.push(`Rendered counterexamples, exact tick projection and stale-result invalidation at ${width}px`);
    await page.close();
  }
} finally { await browser.close(); }
assert.deepEqual(errors, []);
const sourceFiles = ['src/learn/data/semi-supervised-models.js', 'src/learn/components/lesson-labs/SemiSupervisedFigures.jsx', 'src/learn/components/lesson-labs/SemiSupervisedLabs.jsx', 'src/learn/components/lesson-labs/semi-supervised.css', 'src/learn/data/topics/semi-supervised-learning-label-propagation-self-training-co-training.jsx', 'scripts/review-semi-supervised-independent.mjs'];
fs.writeFileSync('docs/teaching/evidence/semi-supervised-independent.json', JSON.stringify({ status: 'passed', reviewedAt: new Date().toISOString(), comparisons, checks, screenshots, sourceHashes: Object.fromEntries(sourceFiles.map(path => [path, createHash('sha256').update(fs.readFileSync(path)).digest('hex')])) }, null, 2) + '\n');
console.log(`PASS: ${comparisons} analytical numerical comparisons and ${checks.length} complementary groups.`);
