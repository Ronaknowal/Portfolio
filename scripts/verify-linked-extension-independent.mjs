import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { linkedCycleTrace, middleSplitState } from '../src/learn/data/linked-traversal-models.js';
import { nextGreaterTrace, histogramState } from '../src/learn/data/monotonic-stack-models.js';
import { linkedTraversalExamples } from '../src/learn/data/linked-traversal-examples.js';
import { monotonicStackExamples } from '../src/learn/data/monotonic-stack-examples.js';
const directory = 'scratch/linked-extension-independent';
fs.mkdirSync(directory, { recursive: true });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const authorPath = 'docs/teaching/evidence/linked-traversal-extension-author-review.json';
const authorBytes = fs.readFileSync(authorPath), author = JSON.parse(authorBytes);
const sources = author.productionSources.map(source => {
  const bytes = fs.readFileSync(source.path); assert.equal(hash(bytes), source.sha256);
  const archive = path.join(directory, 'original-author-sources', source.path);
  fs.mkdirSync(path.dirname(archive), { recursive: true });
  if (!fs.existsSync(archive)) fs.writeFileSync(archive, bytes); else assert.equal(hash(fs.readFileSync(archive)), source.sha256);
  return { path: source.path, sha256: source.sha256, bytes: bytes.length };
});
const authorArchive = `${directory}/original-author-packet.json`;
if (!fs.existsSync(authorArchive)) fs.writeFileSync(authorArchive, authorBytes); else assert.equal(hash(fs.readFileSync(authorArchive)), hash(authorBytes));
function firstRepeat(next, head) {
  const visited = new Map(), walk = [];
  let current = head;
  while (current !== null && !visited.has(current)) {
    visited.set(current, walk.length); walk.push(current); current = next[current];
  }
  return { walk, entry: current, prefix: current === null ? null : visited.get(current),
    cycleLength: current === null ? 0 : walk.length - visited.get(current) };
}
const hop = (next, head, count) => {
  let current = head;
  for (let step = 0; step < count && current !== null; step += 1) current = next[current];
  return current;
};
const graphs = [];
for (let size = 0; size <= 5; size += 1) {
  for (let code = 0; code < (size + 1) ** size; code += 1) {
    const next = Array.from({ length: size }, (_, index) => { const digit = Math.floor(code / (size + 1) ** index) % (size + 1); return digit === size ? null : digit; });
    for (const head of [null, ...next.map((_, index) => index)]) {
      const saved = [...next], expected = firstRepeat(next, head), trace = linkedCycleTrace(next, head);
      assert.deepEqual(next, saved);
      assert.equal(trace.entry, expected.entry);
      assert.equal(trace.prefixLength, expected.prefix);
      assert.equal(trace.cycleLength, expected.cycleLength);
      for (const frame of trace.frames.filter(frame => frame.phase === 'detect')) {
        assert(frame.rounds > 0);
        assert.equal(frame.slow, hop(next, head, frame.rounds));
        assert.equal(frame.fast, hop(next, head, 2 * frame.rounds));
        assert.equal(frame.via, hop(next, head, 2 * frame.rounds - 1));
      }
      if (expected.entry !== null) {
        const firstMeeting = Math.max(expected.cycleLength, Math.ceil(expected.prefix / expected.cycleLength) * expected.cycleLength);
        assert.equal(trace.frames.find(frame => frame.phase === 'meeting').rounds, firstMeeting);
        assert.equal(trace.meeting, hop(next, head, firstMeeting));
        for (const frame of trace.frames.filter(frame => ['reset','entry-walk','entry'].includes(frame.phase))) {
          assert.equal(frame.slow, hop(next, head, frame.entrySteps));
          assert.equal(frame.fast, hop(next, trace.meeting, frame.entrySteps));
        }
      }
      graphs.push({ next, head, entry: trace.entry, cycleLength: trace.cycleLength, prefixLength: trace.prefixLength });
    }
  }
}
const middleCases = [];
for (let length = 0; length <= 12; length += 1) for (const policy of ['first','second']) for (const cut of [false,true]) {
  const state = middleSplitState(length, policy, cut);
  assert.equal(state.middle, length ? Math.floor((length - (policy === 'first' ? 1 : 0)) / 2) : null);
  const walk = head => firstRepeat(state.next, head).walk;
  if (cut) {
    const left = walk(state.leftHead), right = walk(state.rightHead);
    assert.deepEqual([...left,...right], Array.from({length},(_,i)=>i));
    assert.equal(left.length, Math.ceil(length / 2)); assert.equal(right.length, Math.floor(length / 2));
    assert.equal(left.filter(index => right.includes(index)).length, 0);
  }
  middleCases.push(state);
}
const arrays = [];
function directDistances(values, inclusive) {
  return values.map((value, index) => {
    const relative = values.slice(index + 1).findIndex(candidate => inclusive ? candidate >= value : candidate > value);
    return relative < 0 ? 0 : relative + 1;
  });
}
// All length-seven ternary histograms, plus signed versions, extend the author's shorter exhaustive set.
for (let serial = 0; serial < 3 ** 7; serial += 1) {
  const heights = Array.from({ length: 7 }, (_, index) => Math.floor(serial / 3 ** index) % 3);
  const histogram = histogramState(heights);
  const left = heights.map((height, index) => {
    for (let candidate = index - 1; candidate >= 0; candidate -= 1) if (heights[candidate] < height) return candidate;
    return -1;
  });
  const right = heights.map((height, index) => {
    for (let candidate = index + 1; candidate < heights.length; candidate += 1) if (heights[candidate] < height) return candidate;
    return heights.length;
  });
  let optimum = 0;
  for (let start = 0; start < heights.length; start += 1) for (let end = start + 1; end <= heights.length; end += 1) optimum = Math.max(optimum, (end - start) * Math.min(...heights.slice(start, end)));
  assert.deepEqual(histogram.leftTrace.boundaries, left); assert.deepEqual(histogram.rightTrace.boundaries, right);
  assert.equal(histogram.area, optimum);
  if (histogram.best) {
    assert.equal((histogram.best.end - histogram.best.start) * Math.min(...heights.slice(histogram.best.start, histogram.best.end)), optimum);
  } else assert.equal(optimum, 0);
  const values = heights.map(value => value - 1);
  const traces = [false,true].map(inclusive => nextGreaterTrace(values, inclusive));
  traces.forEach(trace => {
    assert.deepEqual(trace.distances, directDistances(values, trace.inclusive));
    assert.equal(trace.pushes, values.length); assert(trace.pops <= values.length);
    for (const frame of trace.frames.filter(frame => frame.phase === 'push')) {
      const prefix = values.slice(0, frame.current + 1), expected = prefix.flatMap((value, index) => directDistances(prefix, trace.inclusive)[index] === 0 ? [index] : []);
      assert.deepEqual(frame.stack, expected);
    }
  });
  arrays.push({ heights, left, right, area: optimum, values, strict: traces[0].distances, inclusive: traces[1].distances });
}
const payload = { checkedAt: new Date().toISOString(), sources, authorPacketSha256: hash(authorBytes),
  graphs, middleCases, arrays, examples: { ...linkedTraversalExamples, ...monotonicStackExamples } };
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(payload));
execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-linked-extension-independent.py'], { stdio: 'inherit', timeout: 120000 });
console.log(JSON.stringify({ graphs: graphs.length, middleCases: middleCases.length, changedSevenElementArrays: arrays.length, sourceHashes: sources.length }));
