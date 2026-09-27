// Small exact operators and a frozen two-read classifier. No training runs in the reader.
export const longContextDefaults = () => [6, 1, 8, 2, 0].map((value, i) => ({ id: `R${i}`, key: i === 0 ? Math.log(9) : 0, value }));
export function stableSoftmax(scores) {
  const peak = Math.max(...scores);
  if (!Number.isFinite(peak)) throw new RangeError('At least one finite legal score is required.');
  const weights = scores.map(x => Math.exp(x - peak)), sum = weights.reduce((a, b) => a + b, 0);
  return weights.map(x => x / sum);
}
export function retainedRead(records, segmentLength, memory, query, beta = 0, excluded = []) {
  if (!records.length || !Number.isInteger(query) || query < 0 || query >= records.length || !Number.isInteger(segmentLength) || segmentLength < 1 || !Number.isInteger(memory) || memory < 0 || !Number.isFinite(beta) || records.some(r => !Number.isFinite(r.key) || !Number.isFinite(r.value))) throw new RangeError('Invalid record or cache configuration.');
  const start = Math.floor(query / segmentLength) * segmentLength, stop = Math.min(records.length, start + segmentLength), first = Math.max(0, start - memory);
  const rows = records.map((record, position) => ({ ...record, position, status: position < first ? 'evicted' : position >= stop ? 'later segment' : position > query ? 'future / masked' : excluded.includes(record.id) ? 'excluded by edit' : 'legal', score: record.key - beta * (query - position) }));
  const legal = rows.filter(r => r.status === 'legal');
  if (!legal.length) throw new RangeError('Keep at least one legal key.');
  const weights = stableSoftmax(legal.map(r => r.score));
  legal.forEach((r, i) => { r.weight = weights[i]; r.contribution = weights[i] * r.value; });
  return { rows, start, stop, first, output: legal.reduce((sum, row) => sum + row.contribution, 0), denominator: legal.reduce((sum, row) => sum + Math.exp(row.score), 0), legalIds: legal.map(r => r.id) };
}
export function recurrenceTrace(events, base = .8, initial = 0) {
  if (!(base > 0 && base < 1) || !Number.isFinite(initial) || !events.length || events.some(e => ![e.x, e.input, e.recurrence].every(Number.isFinite) || e.input < 0 || e.input > 1 || e.recurrence < 0 || e.recurrence > 1)) throw new RangeError('Invalid recurrence input.');
  let state = initial, contributions = [], initialContribution = initial;
  return events.map(e => {
    const logDecay = 8 * e.recurrence * Math.log(base), decay = Math.exp(logDecay);
    const retained = decay * state, injection = Math.sqrt(-Math.expm1(2 * logDecay)) * e.input * e.x;
    initialContribution *= decay;
    contributions = [...contributions.map(c => c * decay), injection];
    state = retained + injection;
    return { decay, retained, injection, state, contributions, initialContribution };
  });
}
export function latentRead(records, queries) {
  if (!records.length || !queries.length || records.some(r => !Number.isFinite(r.position) || !Number.isFinite(r.value)) || queries.some(q => !Number.isFinite(q))) throw new RangeError('Invalid latent read.');
  return queries.map(query => {
    const weights = stableSoftmax(records.map(r => query * r.position));
    return { query, weights, contributions: weights.map((w, i) => w * records[i].value), output: weights.reduce((sum, w, i) => sum + w * records[i].value, 0) };
  });
}
export const composeAffine = (later, earlier) => [later[0] * earlier[0], later[0] * earlier[1] + later[1]];

const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const linear = (row, parameters, name) => parameters[name + '.weight'].map((weights, i) => dot(row, weights) + parameters[name + '.bias'][i]);
const add = (a, b) => a.map((row, i) => row.map((value, j) => value + b[i][j]));
const normalize = (row, p, name) => {
  const mean = row.reduce((a, b) => a + b, 0) / row.length, variance = row.reduce((sum, x) => sum + (x - mean) ** 2, 0) / row.length;
  return row.map((x, i) => (x - mean) / Math.sqrt(variance + 1e-5) * p[name + '.weight'][i] + p[name + '.bias'][i]);
};
// Numerical Recipes erfc approximation: bounded absolute error ~1.2e-7.
// The browser uses the erf form of GELU, never the tanh approximation.
export function erf(value) {
  if (value === 0) return 0;
  const x = Math.abs(value), t = 1 / (1 + .5 * x);
  const erfc = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (.37409196 + t * (.09678418 + t * (-.18628806 + t * (.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 + t * .17087277)))))))));
  return Math.sign(value) * (1 - erfc);
}
function projectedAttention(queries, inputs, p, name, valid) {
  const q = queries.map(row => linear(row, p, name + '.query')), k = inputs.map(row => linear(row, p, name + '.key')), v = inputs.map(row => linear(row, p, name + '.value'));
  const weights = q.map(query => stableSoftmax(k.map((key, i) => valid && !valid[i] ? -Infinity : dot(query, key) / Math.sqrt(query.length))));
  const mixed = weights.map(row => v[0].map((_, coordinate) => row.reduce((sum, weight, i) => sum + weight * v[i][coordinate], 0)));
  return { output: mixed.map(row => linear(row, p, name + '.output')), weights };
}
export function trajectoryForward(parameters, records) {
  if (!records.length || !records.some(r => r.valid) || records.some(r => ![r.x, r.y, r.position].every(Number.isFinite))) throw new RangeError('Keep a finite valid trajectory point.');
  const valid = records.map(r => r.valid);
  const inputs = records.map(r => linear([2 * r.x - 1, 2 * r.y - 1, r.position], parameters, 'input_projection'));
  let latent = parameters.latent.map(row => [...row]), weights;
  for (let round = 0; round < 2; round++) {
    const read = projectedAttention(latent.map(row => normalize(row, parameters, 'cross_norm')), inputs, parameters, 'cross', valid);
    weights = read.weights;
    latent = add(latent, read.output);
    const normalized = latent.map(row => normalize(row, parameters, 'self_norm'));
    latent = add(latent, projectedAttention(normalized, normalized, parameters, 'self_attention').output);
    const feed = latent.map(row => linear(linear(normalize(row, parameters, 'final_norm'), parameters, 'feedforward.0').map(x => .5 * x * (1 + erf(x / Math.SQRT2))), parameters, 'feedforward.2'));
    latent = add(latent, feed);
  }
  const pooled = latent[0].map((_, j) => latent.reduce((sum, row) => sum + row[j], 0) / latent.length);
  const logits = linear(pooled, parameters, 'classifier'), probabilities = stableSoftmax(logits);
  return { logits, probabilities, weights, latent, predictedClass: probabilities.indexOf(Math.max(...probabilities)) + 1 };
}
export function meanTrajectory(parameters, records) {
  const valid = records.filter(r => r.valid);
  if (!valid.length) throw new RangeError('Keep a valid point.');
  const mean = ['x', 'y'].map(key => valid.reduce((sum, r) => sum + 2 * r[key] - 1, 0) / valid.length);
  const logits = parameters.coef.map((row, i) => dot(row, mean) + parameters.intercept[i]), probabilities = stableSoftmax(logits);
  return { probabilities, predictedClass: probabilities.indexOf(Math.max(...probabilities)) + 1 };
}
