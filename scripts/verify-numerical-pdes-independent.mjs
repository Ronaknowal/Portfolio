import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/numerical-pde-models.js';
import { numericalPdeExamples as examples } from '../src/learn/data/numerical-pde-examples.js';

const directory = 'scratch/numerical-pdes-independent-review';
const start = JSON.parse(fs.readFileSync(`${directory}/start.json`, 'utf8'));
function sourceHashes() {
  return start.sourceHashes.map(({ path }) => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
}
const data = { sourceHashes: sourceHashes(), examples, poisson: [], heat: [], transport: [], elements: [], triangles: [], coarse: [], neumann: [], interfaces: [] };
for (const intervals of [3, 5, 11, 23]) {
  for (const length of [.7, 1.3, 2.5]) {
    for (const profile of ['linear', 'quadratic', 'quartic']) {
      const input = { intervals, length, profile, scale: 1.7, left: -2.3, right: 4.1, method: 'jacobi', iterations: intervals * 3, tolerance: .03 };
      data.poisson.push({ input, result: model.poissonProblem(input) });
    }
  }
}
for (const intervals of [5, 9, 17, 29]) {
  for (const alpha of [.3, 1.7]) {
    for (const mode of [1, 2, intervals - 1]) {
      const input = { intervals, alpha, mode, finalTime: .017, timeSteps: 80 };
      data.heat.push({ input, result: model.diffusionEvolution(input) });
    }
  }
}
for (const cells of [9, 13, 27]) for (const velocity of [-1, 1]) for (const courant of [.2, .7, 1]) for (const profile of ['sine', 'pulse']) {
  const input = { cells, velocity, courant, profile, steps: 19 };
  data.transport.push({ input, result: model.advectionEvolution(input) });
}
for (const nodes of [[0,.1,.4,.8,1], [0,.13,.17,.71,1], [0,.001,.003,.42,.999,1]]) {
  for (const sourceKind of ['constant', 'point']) for (const point of [.001, .17, .37, .999]) for (const conductivity of [.3, 7]) {
    const input = { nodes, sourceKind, point, conductivity, source: 3, left: 1.2, right: -2.3 };
    data.elements.push({ input, result: model.finiteElement1D(input) });
  }
}
for (const triangle of [[[0,0],[1,0],[0,1]], [[1,-2],[3,1],[-1,2]], [[2,3],[5,2],[1,6]], [[-.5,1],[1.5,1],[.4,1.07]]]) {
  for (const order of [[0,1,2],[0,2,1],[2,0,1]]) {
    const vertices = order.map(index => triangle[index]);
    data.triangles.push({ vertices, result: model.triangleElement(vertices) });
  }
}
for (let mode = 1; mode <= 7; mode += 1) data.coarse.push({ mode, result: model.twoGridCorrection(mode) });
for (const cells of [3,7,19]) for (const source of [-3,0,2.5]) {
  const input = { cells, source, leftOutward: -.5, rightOutward: source + .5 };
  data.neumann.push({ input, result: model.neumannBalance(input) });
}
for (const interfacePosition of [.13,.37,.81]) for (const leftConductivity of [.3,7]) {
  const input = { interfacePosition, leftConductivity, rightConductivity: 2.3, leftTemperature: -2, rightTemperature: 4 };
  data.interfaces.push({ input, result: model.materialInterface(input) });
}
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify(data, null, 2));
const run = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-numerical-pdes-independent.py'], { encoding: 'utf8', maxBuffer: 1024 * 1024 });
process.stdout.write(run.stdout || '');
process.stderr.write(run.stderr || '');
if (run.status !== 0) process.exit(run.status || 1);
if (JSON.stringify(sourceHashes()) !== JSON.stringify(data.sourceHashes)) throw new Error('Production sources changed during independent checks.');
