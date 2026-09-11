import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as model from '../src/learn/data/topology-tda-models.js';

const graphCases = [];
const pairs = [];
for (let first = 0; first < 5; first += 1) {
  for (let second = first + 1; second < 5; second += 1) pairs.push([first, second]);
}
for (let mask = 0; mask < 1024; mask += 1) {
  const facets = pairs.filter((_, index) => mask & (1 << index));
  for (let first = 0; first < 5; first += 1) {
    facets.push([first]);
    for (let second = first + 1; second < 5; second += 1) {
      for (let third = second + 1; third < 5; third += 1) {
        if ([[first, second], [first, third], [second, third]].every(pair =>
          mask & (1 << pairs.findIndex(candidate => candidate.join() === pair.join())))) {
          facets.push([first, second, third]);
        }
      }
    }
  }
  const complex = model.complexFromFacets(facets);
  const result = model.persistentHomology(complex);
  graphCases.push({ mask, betti: model.bettiAt(complex).betti, bars: result.intervals.filter(bar => bar.death > bar.birth) });
}
const pointCases = [];
for (const [name, fixture] of Object.entries(model.TOPOLOGY_POINT_FIXTURES)) {
  for (const scale of [0.3, 1, 2]) {
    const points = fixture.points.map(point => point.map(value => value * scale));
    const complex = model.ripsFiltration(points);
    const result = model.persistentHomology(complex);
    pointCases.push({ name, scale, points, ordered: result.ordered, intervals: result.intervals,
      snapshots: [...new Set(complex.map(simplex => simplex.birth))].map(threshold => ({ threshold, ...model.bettiAt(complex, threshold) })) });
  }
}
const traceResult = model.persistentHomology(model.ripsFiltration(model.TOPOLOGY_POINT_FIXTURES.square.points), { captureTrace: true });
let traceAssertions = 0;
for (const step of traceResult.trace) {
  let previousPivot = Infinity;
  for (const stage of step.stages) {
    const boundary = new Set();
    for (const column of stage.combination) {
      for (const row of traceResult.boundaries[column]) {
        if (!boundary.delete(row)) boundary.add(row);
      }
    }
    assert.deepEqual([...boundary].sort((a, b) => a - b), stage.boundary);
    const pivot = Math.max(-1, ...stage.boundary);
    assert.ok(pivot < previousPivot);
    previousPivot = pivot;
    traceAssertions += 2;
  }
}
const pixels = Array.from({ length: 512 }, (_, mask) => ({ mask,
  result: model.pixelComplex(Array.from({ length: 9 }, (_, pixel) => mask & (1 << pixel) ? 0 : 1), 0) }));
const normal = [];
for (const lower of [-38, -12, -8, -4, -2, -1, -0.001, 0, 0.001, 1, 2, 4, 8, 12, 38]) {
  for (const width of [1e-12, 1e-8, 1e-6, 0.01, 0.5, 2]) normal.push({ lower, upper: lower + width, value: model.normalInterval(lower, lower + width) });
}
for (const [lower, upper] of [[0, 1e-310], [0, 1e-20], [-1e-20, 1e-20], [1, 1 + 1e-12], [20, 20.001], [-35, -34.999]]) {
  normal.push({ lower, upper, value: model.normalInterval(lower, upper) });
}
const imageCases = [[], [[0, 3]], [[0, 3], [1, 4]], [[0.2, 0.20001]], [[-20, 20]]]
  .flatMap(diagram => [0.1, 0.5, 2].map(bandwidth => ({ diagram, bandwidth, result: model.persistenceImage(diagram, { bandwidth }) })));
for (const width of [1e-20, 1e-12, 0.01]) {
  const options = { bandwidth: 0.5, xEdges: [0, width], yEdges: [0, 1] };
  imageCases.push({ diagram: [[0, 1]], ...options, result: model.persistenceImage([[0, 1]], options) });
}
const diagrams = [[], [[1, 4]], [[1.2, 3.8]], [[0, 5], [1, 3]], [[0, 4], [0, 4]], [[0.1, 0.2], [2, 5], [3, 4]]];
const matching = diagrams.flatMap(first => diagrams.flatMap(second => [1, 2, Infinity].map(power => ({ first, second, power: String(power), result: model.optimalDiagramMatching(first, second, power) }))));
const mapper = [2, 3, 4, 5].flatMap(intervalCount => [0.1, 0.4, 0.65].flatMap(overlap => [0.1, 0.6, 1.1, 2.1]
  .map(clusterDistance => ({ intervalCount, overlap, clusterDistance, result: model.mapperGraph({ intervalCount, overlap, clusterDistance }) }))));
const invalid = [
  () => model.normalizeFiltration([{ vertices: [0, 1], birth: 1 }]),
  () => model.normalizeFiltration([{ vertices: [0], birth: 2 }, { vertices: [1], birth: 0 }, { vertices: [0, 1], birth: 1 }]),
  () => model.complexFromFacets([[0, 0]]),
  () => model.ripsFiltration(Array.from({ length: 15 }, () => [0, 0])),
  () => model.ripsFiltration([[NaN, 0]]),
  () => model.chainBoundary(model.complexFromFacets([[0, 1]]), ['0', '0-1']),
  () => model.pixelComplex([1], 0),
  () => model.evaluateDiagramMatching([[0, 2], [0, 2]], [[0, 2]], [0, 0]),
  () => model.optimalDiagramMatching([[0, Infinity]], []),
  () => model.persistenceImage([[1, 0]]),
  () => model.persistenceImage([], { xEdges: [0, 0] }),
  () => model.mapperGraph({ overlap: 1 }),
];
invalid.forEach(operation => assert.throws(operation, RangeError));
const output = { createdAt: new Date().toISOString(), graphCases, pointCases, traceResult, pixels, normal, imageCases, matching, mapper, traceAssertions, invalidCases: invalid.length };
fs.mkdirSync('scratch/topology-tda-review', { recursive: true });
fs.writeFileSync('scratch/topology-tda-review/model-fixtures.json', JSON.stringify(output));
console.log(JSON.stringify({ graphCases: graphCases.length, pointCases: pointCases.length, pixels: pixels.length, normal: normal.length, imageCases: imageCases.length, matching: matching.length, mapper: mapper.length, traceAssertions, invalidCases: invalid.length }));
