// Pure, bounded operators. State S is key-by-value; delta state M is value-by-key.
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const softmax = values => {
  const maximum = Math.max(...values);
  const weights = values.map(value => Math.exp(value - maximum));
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map(value => value / total);
};
export const summaryDefaults = () => ({
  queries: [[1, 1], [2, 1], [1, 2], [3, 1]],
  keys: [[1, 0], [0, 1], [1, 1], [2, 1]], values: [3, -2, 7, 1], chunk: 3,
});
export function kernelSummary({ queries, keys, values, chunk = 1 }) {
  let state = [0, 0], normalizer = [0, 0];
  const rows = [];
  for (let start = 0; start < values.length; start += chunk) {
    const incoming = [...state], incomingNormalizer = [...normalizer];
    for (let t = start; t < Math.min(start + chunk, values.length); t++) {
      const scores = keys.map((key, j) => j <= t ? dot(queries[t], key) : 0);
      const denominator = scores.reduce((a, b) => a + b, 0);
      const numerator = dot(scores, values);
      const local = keys.slice(start, t + 1).map(key => dot(queries[t], key));
      const carriedNumerator = dot(queries[t], incoming);
      const localNumerator = dot(local, values.slice(start, t + 1));
      const chunkDenominator = dot(queries[t], incomingNormalizer) + local.reduce((a, b) => a + b, 0);
      state = state.map((entry, i) => entry + keys[t][i] * values[t]);
      normalizer = normalizer.map((entry, i) => entry + keys[t][i]);
      rows.push({ scores, weights: scores.map(score => denominator > 0 ? score / denominator : null), numerator, denominator,
        output: denominator > 0 ? numerator / denominator : null,
        chunkOutput: chunkDenominator > 0 ? (carriedNumerator + localNumerator) / chunkDenominator : null,
        carriedNumerator, localNumerator, state: [...state], normalizer: [...normalizer] });
    }
  }
  return rows;
}
export function weightedMemory(keys, values, retention = .5, bonus = Math.log(2), offset = 0) {
  let a = 0, b = 0, p = -Infinity;
  return keys.map((key, t) => {
    const k = key + offset, readScale = Math.max(p, bonus + k);
    const old = Math.exp(p - readScale), current = Math.exp(bonus + k - readScale);
    const output = (old * a + current * values[t]) / (old * b + current);
    const logits = keys.slice(0, t + 1).map((entry, i) => entry + offset + (i === t ? bonus : (t - 1 - i) * Math.log(retention)));
    const weights = softmax(logits);
    const scale = Math.max(p + Math.log(retention), k);
    a = Math.exp(p + Math.log(retention) - scale) * a + Math.exp(k - scale) * values[t];
    b = Math.exp(p + Math.log(retention) - scale) * b + Math.exp(k - scale);
    p = scale;
    return { output, weights, direct: dot(weights, values.slice(0, t + 1)), a, b, p, old, current };
  });
}
export function deltaMemory(keys, values, rate, initial = [0, 0]) {
  let additive = [...initial], delta = [...initial];
  return keys.map((key, t) => {
    const before = [...delta], read = dot(delta, key), residual = values[t] - read;
    additive = additive.map((entry, i) => entry + values[t] * key[i]);
    delta = delta.map((entry, i) => entry + rate * residual * key[i]);
    return { before, read, residual, additive: [...additive], delta: [...delta], target: values[t], key };
  });
}
export function gooseUpdate(removal = [1, 0]) {
  const initial = [[2, 7], [-1, 3]], retention = [.8, .9], rate = [.6, .2], value = [5, 2], replacement = [1, 0];
  const decayed = initial.map(row => row.map((v, j) => v * retention[j]));
  const erased = initial.map(row => removal.map((k, j) => dot(row, removal) * rate[j] * k));
  const written = value.map(v => replacement.map(k => v * k));
  const next = decayed.map((row, i) => row.map((v, j) => v - erased[i][j] + written[i][j]));
  return { initial, decayed, erased, written, next };
}
const sigmoid = x => 1 / (1 + Math.exp(-x));
const linear = (x, weights, bias) => weights.map((row, i) => dot(row, x) + (bias?.[i] || 0));
const normalize = (x, weight, bias) => {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance = x.reduce((sum, v) => sum + (v - mean) ** 2, 0) / x.length;
  return x.map((v, i) => (v - mean) / Math.sqrt(variance + 1e-5) * weight[i] + bias[i]);
};
const mixed = (x, previous, coefficients) => x.map((v, i) => sigmoid(coefficients[i]) * v + (1 - sigmoid(coefficients[i])) * previous[i]);

export function trajectoryMemoryForward(parameters, points, kind, { cut = null, reset = false } = {}) {
  const width = parameters['input.weight'].length;
  let states = [null, null];
  const features = [];
  for (let t = 0; t < points.length; t++) {
    if (reset && t === cut) states = [null, null];
    else if (t === cut) states = structuredClone(states);
    let x = linear(points[t].map(v => 2 * v - 1), parameters['input.weight'], parameters['input.bias']);
    for (let block = 0; block < 2; block++) {
      const prefix = `blocks.${block}.`, p = name => parameters[prefix + name];
      let saved = states[block];
      if (!saved) saved = {
        time: Array(width).fill(0), channel: Array(width).fill(0), a: Array(width).fill(0), b: Array(width).fill(0), scale: Array(width).fill(-Infinity),
        memory: Array.from({ length: width }, () => Array(width).fill(0)), normalizer: Array(width).fill(0),
      };
      const normalized = normalize(x, p('time_norm.weight'), p('time_norm.bias'));
      const q = linear(mixed(normalized, saved.time, p('mixer.mix')[0]), p('mixer.query.weight'));
      const k = linear(mixed(normalized, saved.time, p('mixer.mix')[1]), p('mixer.key.weight'));
      const v = linear(mixed(normalized, saved.time, p('mixer.mix')[2]), p('mixer.value.weight'));
      let read;
      if (kind === 'rwkv4') {
        read = q.map((query, i) => {
          const bonusKey = p('mixer.bonus')[i] + k[i], scale = Math.max(saved.scale[i], bonusKey);
          const old = Math.exp(saved.scale[i] - scale), fresh = Math.exp(bonusKey - scale);
          const result = sigmoid(query) * (old * saved.a[i] + fresh * v[i]) / (old * saved.b[i] + fresh);
          const decay = -Math.exp(p('mixer.raw_decay')[i]);
          const nextScale = Math.max(saved.scale[i] + decay, k[i]);
          saved.a[i] = Math.exp(saved.scale[i] + decay - nextScale) * saved.a[i] + Math.exp(k[i] - nextScale) * v[i];
          saved.b[i] = Math.exp(saved.scale[i] + decay - nextScale) * saved.b[i] + Math.exp(k[i] - nextScale);
          saved.scale[i] = nextScale;
          return result;
        });
      } else {
        const phi = vector => vector.map(entry => entry >= 0 ? entry + 1 : Math.exp(entry));
        const query = phi(q), key = phi(k);
        for (let i = 0; i < width; i++) {
          saved.normalizer[i] += key[i];
          for (let j = 0; j < width; j++) saved.memory[i][j] += key[i] * v[j];
        }
        const denominator = dot(query, saved.normalizer);
        read = v.map((_, j) => query.reduce((sum, entry, i) => sum + entry * saved.memory[i][j], 0) / denominator);
      }
      saved.time = normalized;
      const projected = linear(read, p('mixer.output.weight'));
      x = x.map((entry, i) => entry + projected[i]);
      const channel = normalize(x, p('channel_norm.weight'), p('channel_norm.bias'));
      const key = linear(mixed(channel, saved.channel, p('channel_mix')[0]), p('channel_key.weight')).map(entry => Math.max(0, entry) ** 2);
      const gate = linear(mixed(channel, saved.channel, p('channel_mix')[1]), p('channel_gate.weight')).map(sigmoid);
      const update = linear(key, p('channel_value.weight'));
      x = x.map((entry, i) => entry + gate[i] * update[i]);
      saved.channel = channel;
      states[block] = saved;
    }
    features.push(x);
  }
  const mean = features[0].map((_, i) => features.reduce((sum, row) => sum + row[i], 0) / features.length);
  const logits = linear(mean, parameters['classifier.weight'], parameters['classifier.bias']);
  const probabilities = softmax(logits);
  return { logits, probabilities, predicted: probabilities.indexOf(Math.max(...probabilities)) + 1, features };
}
