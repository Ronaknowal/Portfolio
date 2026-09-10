import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import katex from 'katex';
import { betaCdf, betaDensity, betaQuantile, bridgeGraph, coverageIntervals } from '../src/learn/components/lesson-labs/math.js';

const python = process.env.LESSON_PYTHON || 'python';
const lessons = ['hypothesis-testing-confidence-intervals', 'bayesian-inference-conjugate-priors', 'spectral-graph-theory'];
for (const slug of lessons) {
  const source = fs.readFileSync(`src/learn/data/topics/${slug}.jsx`, 'utf8');
  const blocks = [...source.matchAll(/<CodeBlock language="(python|output)">\{`([\s\S]*?)`\}<\/CodeBlock>/g)];
  for (let i = 0; i < blocks.length; i += 2) {
    assert.equal(blocks[i][1], 'python');
    assert.equal(blocks[i + 1][1], 'output');
    const result = spawnSync(python, ['-c', blocks[i][2]], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim().replace(/\r\n/g, '\n'), blocks[i + 1][2].trim().replace(/\r\n/g, '\n'), `${slug}: displayed output differs from executed code`);
  }
  for (const match of source.matchAll(/<MathBlock>\{String.raw`([\s\S]*?)`\}<\/MathBlock>/g)) {
    katex.renderToString(match[1], { throwOnError: true, strict: 'error' });
  }
  console.log(`${slug}: Python output and KaTeX checked`);
}

const near = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
near(betaQuantile(.025, 10, 4), .461868460765959);
near(betaQuantile(.975, 10, 4), .9090796054279033);
near(1 - betaCdf(.7, 10, 4), .579394354239);
// Independent quadrature checks CDF and density normalisation, including control extremes.
for (const [a, b] of [[1, 1], [20, 20], [100, 40], [81, 1], [1, 21]]) {
  const integrate = end => {
    const n = 8000, h = end / n;
    let sum = betaDensity(0, a, b) + betaDensity(end, a, b);
    for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * betaDensity(i * h, a, b);
    return sum * h / 3;
  };
  near(integrate(1), 1, 1e-6);
  near(integrate(.7), betaCdf(.7, a, b), 1e-6);
}
// Compare the browser's Jacobi spectrum with an independent NumPy implementation.
for (const w of [0, .05, .2, 1, 2]) {
  const graph = bridgeGraph(w);
  const result = spawnSync(python, ['-c', 'import sys,json,numpy as np; a=np.array(json.loads(sys.stdin.read())); print(json.dumps(np.linalg.eigvalsh(a).tolist()))'], { encoding: 'utf8', input: JSON.stringify(graph.laplacian) });
  assert.equal(result.status, 0, result.stderr);
  JSON.parse(result.stdout).forEach((v, i) => near(graph.values[i], v));
  graph.vectors.forEach((u, k) => {
    near(u.reduce((s, v) => s + v * v, 0), 1);
    graph.laplacian.forEach((row, i) => near(row.reduce((s, v, j) => s + v * u[j], 0), graph.values[k] * u[i]));
  });
}
// Validate the simulated sampling distribution, not an exact coverage count per batch.
let count = 0;
for (let batch = 0; batch < 500; batch++) count += coverageIntervals(25, 1.959963984540, batch).filter(d => d.low <= 100 && d.high >= 100).length;
near(count / 20000, .95, .01);
const small = coverageIntervals(25, 1.959963984540), large = coverageIntervals(100, 1.959963984540);
near(small[0].high - small[0].low, 2 * (large[0].high - large[0].low));
console.log('Labs: coverage, Beta boundaries/integrals/quantiles, and graph eigenpairs checked.');
