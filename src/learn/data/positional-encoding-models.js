import { add, dense, dot, gelu, layerNorm, mean, softmax } from './sequence-tensor-operations.js';
export function frequencies(width, base = 10000) {
  if (!Number.isInteger(width) || width < 2 || width % 2 || !(base > 1)) throw new Error('Use an even width and a base greater than one.');
  return Array.from({
    length: width / 2
  }, (_, pair) => base ** (-2 * pair / width));
}
export function sinusoidal(position, width, base = 10000) {
  return frequencies(width, base).flatMap(frequency => [Math.sin(position * frequency), Math.cos(position * frequency)]);
}
export function rotate(values, position, base = 10000) {
  return frequencies(values.length, base).flatMap((frequency, pair) => {
    const angle = position * frequency,
      even = values[2 * pair],
      odd = values[2 * pair + 1];
    return [even * Math.cos(angle) - odd * Math.sin(angle), even * Math.sin(angle) + odd * Math.cos(angle)];
  });
}
export function alibiSlopes(heads) {
  if (!Number.isInteger(heads) || heads < 1 || heads > 128) throw new Error('Use 1–128 heads.');
  const powerSchedule = count => Array.from({
    length: count
  }, (_, i) => 2 ** (-8 * (i + 1) / count));
  const lower = 2 ** Math.floor(Math.log2(heads));
  return lower === heads ? powerSchedule(lower) : [...powerSchedule(lower), ...powerSchedule(2 * lower).filter((_, i) => i % 2 === 0).slice(0, heads - lower)];
}
export function learnedPosition(table, position) {
  if (!Number.isInteger(position) || position < 0 || position >= table.length) throw new Error('Position is outside the learned table.');
  return table[position];
}
export function alibiCompetition(scores, keys, query = 3, slope = .5) {
  const legal = keys.map(key => key <= query);
  const bias = keys.map(key => -slope * (query - key));
  const final = scores.map((score, i) => legal[i] ? score + bias[i] : -Infinity);
  return {
    legal,
    bias,
    final,
    weights: legal.some(Boolean) ? softmax(final) : null
  };
}
export function relativeBucket(offset) {
  const distance = Math.abs(offset);
  const index = distance < 8 ? distance : Math.min(15, 8 + Math.floor(Math.log(distance / 8) / Math.log(16) * 8));
  return index + (offset > 0 ? 16 : 0);
}
export function extensionFrequencies(width = 64, base = 10000, context = 4096, factor = 8) {
  if (width <= 2) throw new Error('Base scaling requires rotary width greater than two.');
  const original = frequencies(width, base);
  const baseScaled = frequencies(width, base * factor ** (width / (width - 2)));
  const pi = original.map(value => value / factor);
  const yarn = original.map(value => {
    const turns = context * value / (2 * Math.PI);
    const ramp = Math.min(1, Math.max(0, (turns - 1) / 31));
    return (1 - ramp) * value / factor + ramp * value;
  });
  return {
    original,
    pi,
    baseScaled,
    yarn,
    qkScale: 1 + .1 * Math.log(factor)
  };
}
export const positionCacheDefault = () => ({
  query: [2, -.5, .25, 1],
  keys: [[.5, 1, 1, 0], [1, -.5, .5, 1], [-.5, .75, 1, -1]],
  values: [[2, 0], [0, 3], [1, -1]],
  ids: [7, 8, 9],
  queryId: 9,
  rotaryId: 9,
  maskId: 9,
  base: 10000,
  cacheBase: 10000,
  mode: 'rope',
  slope: .5
});
export function cachePositionRead(settings, consistent = false) {
  const {
    query,
    keys,
    values,
    ids,
    mode,
    queryId,
    rotaryId,
    maskId,
    base,
    cacheBase,
    slope
  } = settings;
  const q = mode === 'rope' ? rotate(query, consistent ? queryId : rotaryId, base) : query;
  const k = mode === 'rope' ? keys.map((key, i) => rotate(key, ids[i], consistent ? base : cacheBase)) : keys;
  const content = k.map(key => dot(q, key) / Math.sqrt(query.length));
  const legal = ids.map(id => id <= (consistent ? queryId : maskId));
  const scores = content.map((score, i) => legal[i] ? score - (mode === 'alibi' ? slope * (queryId - ids[i]) : 0) : -Infinity);
  if (!legal.some(Boolean)) return {
    content,
    scores,
    legal,
    weights: null,
    output: null
  };
  const weights = softmax(scores);
  const output = values[0].map((_, coordinate) => dot(weights, values.map(value => value[coordinate])));
  return {
    content,
    scores,
    legal,
    weights,
    output
  };
}
export function positionMovementForward(model, points, positions, padding = []) {
  const {
    mode,
    state_dict: weights
  } = model;
  if (!points.length || points.length > 50 || points.some(row => row.length !== 2 || row.some(value => !Number.isFinite(value))) || positions.length !== points.length || positions.some(value => !Number.isFinite(value)) || points.every((_, i) => padding[i])) throw new Error('Use 1–50 finite point records with at least one valid point.');
  const linear = (row, prefix) => dense(row, weights[`${prefix}.weight`], weights[`${prefix}.bias`]);
  const norm = (row, prefix) => layerNorm(row, weights[`${prefix}.weight`], weights[`${prefix}.bias`]);
  const stem = points.map(([x, y]) => linear([2 * x - 1, 2 * y - 1], 'stem'));
  const positionVectors = positions.map(position => mode === 'sinusoidal' ? sinusoidal(position, 24) : mode === 'learned' ? learnedPosition(weights.position_table, position) : Array(24).fill(0));
  const state = stem.map((row, i) => add(row, positionVectors[i]));
  const packed = state.map(row => linear(norm(row, 'norm_attention'), 'query_key_value'));
  const heads = [0, 1].map(head => {
    const rawQuery = packed.map(row => row.slice(12 * head, 12 * head + 12));
    const rawKey = packed.map(row => row.slice(24 + 12 * head, 36 + 12 * head));
    const value = packed.map(row => row.slice(48 + 12 * head, 60 + 12 * head));
    const query = mode === 'rope' ? rawQuery.map((row, i) => rotate(row, positions[i])) : rawQuery;
    const key = mode === 'rope' ? rawKey.map((row, i) => rotate(row, positions[i])) : rawKey;
    const content = query.map(row => key.map(other => dot(row, other) / Math.sqrt(12)));
    const bias = positions.map(queryPosition => positions.map(keyPosition => mode === 'alibi' ? -alibiSlopes(2)[head] * Math.abs(queryPosition - keyPosition) : 0));
    const attention = content.map((row, i) => softmax(row.map((item, j) => padding[j] ? -Infinity : item + bias[i][j])));
    const output = attention.map(row => value[0].map((_, coordinate) => dot(row, value.map(item => item[coordinate]))));
    return {
      rawQuery,
      rawKey,
      query,
      key,
      value,
      content,
      bias,
      attention,
      output
    };
  });
  const context = state.map((row, i) => add(row, linear([...heads[0].output[i], ...heads[1].output[i]], 'attention_output')));
  const final = context.map(row => norm(add(row, linear(linear(norm(row, 'norm_feedforward'), 'feedforward.0').map(gelu), 'feedforward.2')), 'final_norm'));
  const valid = final.filter((_, i) => !padding[i]);
  const pooled = final[0].map((_, coordinate) => mean(valid.map(row => row[coordinate])));
  const logits = linear(pooled, 'classifier');
  return {
    logits,
    probabilities: softmax(logits),
    heads,
    pooled,
    positionVectors,
    stem,
    combined: state
  };
}
