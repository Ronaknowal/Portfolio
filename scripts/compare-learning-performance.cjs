const fs = require('node:fs');
const assert = require('node:assert/strict');
const before = JSON.parse(fs.readFileSync('scratch/learning-performance/before.json', 'utf8'));
const after = JSON.parse(fs.readFileSync('scratch/learning-performance/after.json', 'utf8'));
for (const key of ['browser', 'viewport', 'repeats', 'cache', 'timing', 'bytes']) assert.deepEqual(after.methodology[key], before.methodology[key], `Measurement method changed: ${key}`);
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const comparisons = before.summary.map(route => {
  const first = before.samples.filter(s => s.name === route.name), second = after.samples.filter(s => s.name === route.name);
  assert.equal(first.length, second.length, `Different repeat counts for ${route.name}`);
  assert.ok(second.every(s => s.path === route.path), `Changed measured route ${route.name}`);
  const metric = key => ({ before: median(first.map(key)), after: median(second.map(key)) });
  const javascript = metric(s => s.js.encodedBodyBytes);
  return {
    route: route.name, path: route.path,
    jsEncodedBodyBytes: javascript,
    jsEncodedReductionPercent: 100 * (javascript.before - javascript.after) / javascript.before,
    jsDecodedBodyBytes: metric(s => s.js.decodedBodyBytes),
    jsTransferBytes: metric(s => s.js.transferBytes),
    jsRequests: metric(s => s.js.count),
    cssEncodedBodyBytes: metric(s => s.css.encodedBodyBytes),
    cssRequests: metric(s => s.css.count),
    readyMs: metric(s => s.readyMs),
    longTaskCount: metric(s => s.longTaskCount),
    longTaskTotalMs: metric(s => s.longTaskTotalMs),
  };
});
const report = {
  beforeMeasuredAt: before.measuredAt, afterMeasuredAt: after.measuredAt,
  methodology: after.methodology,
  interpretation: 'Byte totals and dependency boundaries are the primary evidence. Timings are medians of three local unthrottled observations; they are not field metrics or guarantees for users/devices/networks.',
  comparisons,
};
fs.writeFileSync('scratch/learning-performance/comparison.json', JSON.stringify(report, null, 2));
console.log('Route | Encoded JS before → after | Decoded JS before → after | JS reduction | Local median ready ms before → after');
for (const item of comparisons) console.log(`${item.route} | ${item.jsEncodedBodyBytes.before} → ${item.jsEncodedBodyBytes.after} | ${item.jsDecodedBodyBytes.before} → ${item.jsDecodedBodyBytes.after} | ${item.jsEncodedReductionPercent.toFixed(1)}% | ${Math.round(item.readyMs.before)} → ${Math.round(item.readyMs.after)}`);
