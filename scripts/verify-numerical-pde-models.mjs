import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import katex from 'katex';
import * as model from '../src/learn/data/numerical-pde-models.js';
import { numericalPdeExamples } from '../src/learn/data/numerical-pde-examples.js';

const output = 'scratch/numerical-pde-native';
fs.mkdirSync(output, { recursive: true });
const fixtures = { poisson: [], diffusion: [], interfaces: [], neumann: [], advection: [], fem: [], rectangles: [], triangles: [], coarse: [], godunov: [], tridiagonal: [] };
for (const intervals of [2, 3, 5, 8, 17, 32]) for (const length of [0.5, 1, 1.5]) for (const profile of ['quartic', 'quadratic', 'linear']) for (const method of ['direct', 'jacobi']) {
  const input = { intervals, length, profile, method, iterations: 37, scale: 2, left: 2, right: -1 };
  fixtures.poisson.push({ input, result: model.poissonProblem(input) });
}
for (const intervals of [3, 4, 8, 16]) for (const timeSteps of [2, 8, 32]) for (const mode of [1, intervals - 1]) for (const finalTime of [0.01, 0.1]) {
  const input = { intervals, timeSteps, mode, finalTime };
  fixtures.diffusion.push({ input, result: model.diffusionEvolution(input) });
}
for (const leftConductivity of [0.1, 1, 3]) for (const rightConductivity of [0.5, 10, 20]) for (const interfacePosition of [0.1, 0.5, 0.9]) {
  const input = { leftConductivity, rightConductivity, interfacePosition, leftTemperature: 2, rightTemperature: -1 };
  fixtures.interfaces.push({ input, result: model.materialInterface(input) });
}
for (const cells of [2, 4, 13]) for (const source of [-2, 0, 3]) for (const leftOutward of [-1, 0, 1]) for (const offset of [0, 0.25]) {
  const input = { cells, source, leftOutward, rightOutward: source - leftOutward + offset };
  if (Math.abs(input.rightOutward) <= 4) fixtures.neumann.push({ input, result: model.neumannBalance(input) });
}
for (const cells of [8, 16, 21]) for (const courant of [0, 0.25, 1, 1.5]) for (const velocity of [-1, 1]) for (const profile of ['pulse', 'sine']) for (const scheme of ['upwind', 'centered']) {
  const input = { cells, courant, velocity, profile, scheme, steps: 9 };
  fixtures.advection.push({ input, result: model.advectionEvolution(input) });
}
for (const nodes of [[0, .25, .5, .75, 1], [0, .2, .6, 1], [0, .1, .3, .7, .9, 1]]) for (const sourceKind of ['constant', 'point']) for (const point of [.1, .3, 1 / 3, .8]) for (const conductivity of [.5, 1, 3]) {
  const input = { nodes, sourceKind, point, conductivity, source: 2, left: 1, right: -2 };
  fixtures.fem.push({ input, result: model.finiteElement1D(input) });
}
for (const xIntervals of [2, 5, 9]) for (const yIntervals of [2, 4, 7]) for (let selectedX = 1; selectedX < xIntervals; selectedX++) for (let selectedY = 1; selectedY < yIntervals; selectedY++) {
  const input = { xIntervals, yIntervals, selectedX, selectedY, xLength: 2, yLength: 1.5 };
  fixtures.rectangles.push({ input, result: model.rectangleStencil(input) });
}
for (const vertices of [[[0,0],[1,0],[0,1]], [[1,2],[3,2],[1,5]], [[0,0],[1,2],[3,1]], [[1,1],[-1,2],[0,-1]]]) {
  fixtures.triangles.push({ vertices, result: model.triangleElement(vertices) });
}
for (let mode = 1; mode <= 7; mode++) fixtures.coarse.push(model.twoGridCorrection(mode));
for (const left of [-3, -1, 0, 1, 3]) for (const right of [-3, -1, 0, 1, 3]) fixtures.godunov.push({ left, right, result: model.burgersGodunovFlux(left, right) });
for (let size = 1; size <= 20; size++) {
  const diagonal = Array.from({ length: size }, (_, index) => 3 + index / 10);
  const off = Array.from({ length: size - 1 }, (_, index) => index % 2 ? -.7 : .4);
  const rhs = Array.from({ length: size }, (_, index) => Math.sin(index + .3));
  fixtures.tridiagonal.push({ diagonal, off, rhs, values: model.solvePdeTridiagonal(diagonal, off, rhs) });
}
let invalid = 0;
for (const operation of [
  () => model.poissonProblem({ intervals: 0 }), () => model.poissonProblem({ profile: 'missing' }),
  () => model.diffusionEvolution({ mode: 8 }), () => model.materialInterface({ leftConductivity: 0 }),
  () => model.neumannBalance({ source: NaN }), () => model.advectionEvolution({ velocity: 0 }),
  () => model.finiteElement1D({ nodes: [0, .5, .5, 1] }), () => model.triangleElement([[0,0],[1,1],[2,2]]),
  () => model.solvePdeTridiagonal([0], [], [1]), () => model.solvePdeTridiagonal([2,,2], [-1,-1], [0,0,0]),
  () => model.piecewiseLinearValue([0,1], [0,1], 2),
  () => { const row = [0,,1]; Object.setPrototypeOf(row, Object.assign(Object.create(Array.prototype), { 1: .5 })); return model.finiteElement1D({ nodes: row }); },
]) { assert.throws(operation); invalid++; }
let equations = 0;
function walk(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'TaggedTemplateExpression' && node.tag.property?.name === 'raw') {
    katex.renderToString(node.quasi.quasis.map(part => part.value.raw).join(''), { displayMode: true, throwOnError: true });
    equations++;
  }
  for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(walk); else if (value && typeof value === 'object') walk(value);
}
const files = ['src/learn/data/topics/numerical-pdes-grids-finite-elements-stability.jsx', 'src/learn/data/numerical-pde-models.js', 'src/learn/data/numerical-pde-examples.js', 'src/learn/components/lesson-labs/NumericalPdeLabs.jsx', 'src/learn/components/lesson-labs/numerical-pde-labs.css', 'src/learn/data/curriculum/blueprints/numerical-pdes-grids-finite-elements-stability.js'];
walk(parse(fs.readFileSync(files[0], 'utf8'), { sourceType: 'module', plugins: ['jsx'] }));
parse(fs.readFileSync(files[3], 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
fs.writeFileSync(output + '/model-fixtures.json', JSON.stringify({ checkedAt: new Date().toISOString(), equations, invalid, fixtures, examples: numericalPdeExamples, sourceHashes: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) }));
console.log(JSON.stringify({ equations, invalid, exportedCases: Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length])) }));
