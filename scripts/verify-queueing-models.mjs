import fs from 'node:fs';
import * as m from '../src/learn/data/queueing-models.js';
import { queueingExamples } from '../src/learn/data/queueing-examples.js';
const out = 'scratch/queueing-verification';
fs.mkdirSync(out, {
  recursive: true
});
const data = {
  examples: queueingExamples,
  traces: [],
  mm1: [],
  mixtures: [],
  pools: [],
  buffers: [],
  invalid: []
};
for (const gap1 of [0, 1, 3]) for (const gap2 of [0, 1, 3]) {
  for (const s1 of [0, 1, 2]) for (const s2 of [0, 1, 3]) for (const s3 of [0, 1, 2]) {
    const jobs = [s1, s2, s3].map((service, i) => ({
      id: `j${i}`,
      arrival: [0, gap1, gap1 + gap2][i],
      service
    }));
    data.traces.push({
      jobs,
      trace: m.fcfsTrace(jobs),
      windows: [0.5, 1, 2.5, 5, 9].flatMap(h => ['system', 'queue'].map(b => m.occupancyWindow(jobs, h, b)))
    });
  }
}
data.traces.push({
  jobs: [],
  trace: [],
  windows: [m.occupancyWindow([], 2)]
});
for (const rate of [0.1, 1, 10, 100]) for (const load of [0.01, 0.2, 0.5, 0.8, 0.9, 0.99]) for (const p of [0.1, 0.5, 0.9, 0.99]) {
  data.mm1.push({
    lambda: rate * load,
    mu: rate,
    p,
    state: m.mm1State(rate * load, rate, p),
    survival: [0, 0.25 / rate, 1 / rate, 10 / rate].map(t => ({
      t,
      ...m.mm1Survival(rate * load, rate, t)
    }))
  });
}
for (const lambda of [0.5, 2, 5, 8, 9.5]) for (const kind of Object.keys(m.serviceMixtureNames)) data.mixtures.push(m.serviceVariabilityState(lambda, kind));
for (let c = 1; c <= 12; c++) for (const mu of [0.1, 1, 5, 100]) for (const load of [0.01, 0.2, 0.5, 0.8, 0.99]) data.pools.push(m.pooledQueueState(c * mu * load, mu, c));
for (const lambda of [0.5, 2, 5, 10, 12, 20]) for (const mu of [1, 5, 10, 20]) for (const capacity of [1, 2, 3, 6, 12, 20]) data.buffers.push(m.finiteBufferState(lambda, mu, capacity));
for (const [lambda, mu] of [[1e-6, 1e6], [1e6, 1e-6]]) for (const capacity of [1, 20]) data.buffers.push(m.finiteBufferState(lambda, mu, capacity));
const invalid = [['duplicate job ID', () => m.fcfsTrace([{
  id: 'a',
  arrival: 0,
  service: 1
}, {
  id: 'a',
  arrival: 1,
  service: 1
}])], ['unsorted jobs', () => m.fcfsTrace([{
  id: 'a',
  arrival: 1,
  service: 1
}, {
  id: 'b',
  arrival: 0,
  service: 1
}])], ['negative service', () => m.fcfsTrace([{
  id: 'a',
  arrival: 0,
  service: -1
}])], ['missing ID', () => m.fcfsTrace([{
  arrival: 0,
  service: 1
}])], ['NaN arrival', () => m.fcfsTrace([{
  id: 'a',
  arrival: NaN,
  service: 1
}])], ['unrepresentable positive service', () => m.fcfsTrace([{
  id: 'a',
  arrival: 1e6,
  service: 1e-20
}])], ['zero horizon', () => m.occupancyWindow([], 0)], ['wrong boundary', () => m.occupancyWindow([], 1, 'departures')], ['zero lambda', () => m.mm1State(0, 1)], ['infinite mu', () => m.mm1State(1, Infinity)], ['percentile one', () => m.mm1State(1, 2, 1)], ['negative percentile', () => m.mm1State(1, 2, -1)], ['noninteger visible count', () => m.mm1State(1, 2, 0.5, 2.5)], ['unstable survival', () => m.mm1Survival(2, 2, 1)], ['invalid mixture', () => m.serviceVariabilityState(1, 'fiction')], ['boolean server count', () => m.pooledQueueState(1, 2, true)], ['fractional capacity', () => m.finiteBufferState(1, 2, 2.5)], ['negative capacity', () => m.finiteBufferState(1, 2, -2)]];
for (const [name, fn] of invalid) {
  let caught = false;
  try {
    fn();
  } catch (error) {
    caught = error instanceof RangeError;
  }
  if (!caught) throw new Error(`Missing RangeError: ${name}`);
  data.invalid.push(name);
}
for (const arrival of [10, 11]) if (m.mm1State(arrival, 10).stable) throw new Error('Unstable M/M/1 accepted');
for (const arrival of [10, 11]) if (m.pooledQueueState(arrival, 5, 2).stable) throw new Error('Unstable pool accepted');
fs.writeFileSync(`${out}/cases.json`, JSON.stringify(data));
console.log(JSON.stringify({
  traces: data.traces.length,
  mm1: data.mm1.length,
  mixtures: data.mixtures.length,
  pools: data.pools.length,
  buffers: data.buffers.length,
  invalid: data.invalid.length
}));
