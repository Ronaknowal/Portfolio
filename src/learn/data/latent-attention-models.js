// Bounded MLA references and selected one-block forecast inference.
import { dot, add, softmax, linear, layerNorm, gelu, rotate } from './grouped-query-models.js';
export const matvec = (matrix, vector) => matrix.map(row => dot(row, vector));
export const transpose = matrix => matrix[0].map((_, j) => matrix.map(row => row[j]));
export const multiply = (left, right) => left.map(row => transpose(right).map(column => dot(row, column)));
export const difference = (left, right) => Math.max(0, ...left.flat(Infinity).map((value, i) => Math.abs(value - right.flat(Infinity)[i])));
export function rotatePair(vector, angle) {
  return [vector[0] * Math.cos(angle) - vector[1] * Math.sin(angle), vector[0] * Math.sin(angle) + vector[1] * Math.cos(angle)];
}
export function latentDefaults() {
  return {
    queries: [[.5, 1], [-1, 2]],
    latent: [[2, -1], [.5, 1], [-.5, 2]],
    keyUp: [[[1, .5], [0, 2]], [[1, -1], [.5, 1]]],
    valueUp: [[[1, 0], [.5, 1]], [[1, 1], [-1, 2]]],
    queryRotary: [[1, .5], [-.5, 1]],
    keyRotary: [[1, 0], [.5, 1], [1, -1]],
    positions: [0, 1, 3],
    queryPosition: 3,
    frequency: Math.PI / 3
  };
}
export function latentRead(state, {
  expanded = false,
  nonlinear = false,
  changedBasis = false
} = {}) {
  const scale = changedBasis ? [2, .5] : [1, 1];
  const latents = state.latent.map(row => row.map((value, i) => value * scale[i]));
  const keyMaps = state.keyUp.map(matrix => matrix.map(row => row.map((value, i) => value / scale[i])));
  const valueMaps = state.valueUp.map(matrix => matrix.map(row => row.map((value, i) => value / scale[i])));
  return state.queries.map((query, h) => {
    const effective = matvec(transpose(keyMaps[h]), query);
    const keys = latents.map(c => matvec(keyMaps[h], c));
    const values = latents.map(c => matvec(valueMaps[h], c).map(value => nonlinear ? Math.max(0, value) : value));
    const content = expanded ? keys.map(k => dot(query, k)) : latents.map(c => dot(effective, c));
    const qr = rotatePair(state.queryRotary[h], state.queryPosition * state.frequency);
    const rotary = state.keyRotary.map((key, i) => dot(qr, rotatePair(key, state.positions[i] * state.frequency)));
    const logits = content.map((value, i) => (value + rotary[i]) / 2);
    const weights = softmax(logits, state.positions.map(position => position <= state.queryPosition));
    const mixture = latents[0].map((_, d) => latents.reduce((sum, c, i) => sum + weights[i] * c[d], 0));
    const output = expanded ? values[0].map((_, d) => values.reduce((sum, v, i) => sum + weights[i] * v[d], 0)) : matvec(valueMaps[h], mixture).map(value => nonlinear ? Math.max(0, value) : value);
    return {
      content,
      rotary,
      logits,
      weights,
      mixture,
      output,
      keys,
      values,
      effective
    };
  });
}
export function rotationOrder(matrix, vector, angle) {
  const rotation = [[Math.cos(angle), -Math.sin(angle)], [Math.sin(angle), Math.cos(angle)]];
  const ru = multiply(rotation, matrix),
    ur = multiply(matrix, rotation);
  return {
    projected: matvec(matrix, vector),
    rotated: matvec(rotation, vector),
    projectThenRotate: matvec(ru, vector),
    rotateThenProject: matvec(ur, vector),
    commutator: ru.map((row, i) => row.map((value, j) => value - ur[i][j]))
  };
}
export function rankOne(matrix, input) {
  const gram = multiply(transpose(matrix), matrix),
    a = gram[0][0],
    b = gram[0][1],
    d = gram[1][1];
  const discriminant = Math.hypot(a - d, 2 * b),
    eigenvalue = (a + d + discriminant) / 2;
  const tied = discriminant <= 1e-12;
  let direction = Math.abs(b) > 1e-12 ? [b, eigenvalue - a] : a >= d ? [1, 0] : [0, 1];
  const norm = Math.hypot(...direction);
  direction = direction.map(value => value / norm * (direction[0] < 0 ? -1 : 1));
  const projector = direction.map(value => direction.map(other => value * other));
  const reduced = multiply(matrix, projector);
  const matrixError = matrix.reduce((sum, row, i) => sum + row.reduce((s, value, j) => s + (value - reduced[i][j]) ** 2, 0), 0);
  return {
    direction,
    reduced,
    full: matvec(matrix, input),
    output: matvec(reduced, input),
    matrixError,
    singular: [Math.sqrt(Math.max(0, eigenvalue)), Math.sqrt(Math.max(0, (a + d - discriminant) / 2))],
    tied
  };
}
export function latentBudget({
  batch,
  layers,
  length,
  heads,
  content,
  value,
  latent,
  rotary,
  bytes,
  queries
}) {
  const record = latent + rotary;
  const multiplier = batch * layers * length * bytes;
  const [B, N, L, H, K, V, C, R, S, T] = [batch, layers, length, heads, content, value, latent, rotary, bytes, queries].map(BigInt);
  const M = B * N * L * S;
  const exact = { record: C + R, compact: M * (C + R), mha: M * H * (K + V), mqa: M * (K + V), expanded: M * H * (K + R + V), sharedRotary: M * (H * (K + V) + R), expandedOps: 2n * B * H * T * L * (K + R + V), absorbedOps: 2n * B * H * T * L * (2n * C + R), reconstruction: 2n * B * L * H * C * (K + V) };
  return {
    exact,
    record,
    compact: multiplier * record,
    mha: multiplier * heads * (content + value),
    mqa: multiplier * (content + value),
    expanded: multiplier * heads * (content + rotary + value),
    sharedRotary: multiplier * (heads * (content + value) + rotary),
    expandedOps: 2 * batch * heads * queries * length * (content + rotary + value),
    absorbedOps: 2 * batch * heads * queries * length * (2 * latent + rotary),
    reconstruction: 2 * batch * length * heads * latent * (content + value)
  };
}
function rmsNorm(vector, weights, name) {
  const scale = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0) / vector.length + 1e-6);
  return vector.map((value, i) => value / scale * weights[name + '.scale'][i]);
}
export function latentForecast(weights, points, {
  basis = null,
  expanded = false,
  wrongScale = false,
  shift = 0,
  cache = null
} = {}) {
  const start = cache?.positions.length ?? 0;
  const states = points.map(point => linear(point.map(v => 2 * v - 1), weights, 'stem'));
  const normalized = states.map(state => layerNorm(state, weights, 'norm_attention'));
  const fullLatents = normalized.map(row => rmsNorm(linear(row, weights, 'kv_down'), weights, 'kv_norm'));
  const newLatents = basis ? fullLatents.map(row => matvec(transpose(basis), row)) : fullLatents;
  const queryLatents = normalized.map(row => rmsNorm(linear(row, weights, 'query_down'), weights, 'query_norm'));
  const queryContent = queryLatents.map(row => linear(row, weights, 'query_content'));
  const queryRotary = queryLatents.map(row => linear(row, weights, 'query_rotary'));
  const newRotary = normalized.map((row, t) => rotate(linear(row, weights, 'rotary_key'), start + t + shift));
  const latents = [...(cache?.latents ?? []), ...newLatents];
  const rotaryKeys = [...(cache?.rotaryKeys ?? []), ...newRotary];
  const positions = [...(cache?.positions ?? []), ...points.map((_, t) => start + t + shift)];
  const keyMaps = Array.from({
    length: 4
  }, (_, h) => weights['key_up.weight'].slice(4 * h, 4 * h + 4)).map(matrix => basis ? multiply(matrix, basis) : matrix);
  const valueMaps = Array.from({
    length: 4
  }, (_, h) => weights['value_up.weight'].slice(4 * h, 4 * h + 4)).map(matrix => basis ? multiply(matrix, basis) : matrix);
  const traces = [];
  const predictions = states.map((state, t) => {
    const headTraces = [];
    for (let h = 0; h < 4; h++) {
      const q = queryContent[t].slice(4 * h, 4 * h + 4),
        qr = rotate(queryRotary[t].slice(2 * h, 2 * h + 2), start + t + shift);
      const effective = matvec(transpose(keyMaps[h]), q);
      const contentScores = latents.map(c => expanded ? dot(q, matvec(keyMaps[h], c)) : dot(effective, c));
      const rotaryScores = rotaryKeys.map(k => dot(qr, k));
      const divisor = Math.sqrt((wrongScale ? latents[0].length : 4) + 2);
      const logits = contentScores.map((score, i) => (score + rotaryScores[i]) / divisor);
      const probabilities = softmax(logits, positions.map(position => position <= start + t + shift));
      const mixture = latents[0].map((_, d) => latents.reduce((sum, c, i) => sum + probabilities[i] * c[d], 0));
      const output = expanded ? valueMaps[h].map(row => latents.reduce((sum, c, i) => sum + probabilities[i] * dot(row, c), 0)) : matvec(valueMaps[h], mixture);
      headTraces.push({
        effective,
        contentScores,
        rotaryScores,
        logits,
        weights: probabilities,
        mixture,
        output
      });
    }
    traces.push(headTraces);
    const residual = add(state, linear(headTraces.flatMap(h => h.output), weights, 'output'));
    const hidden = linear(layerNorm(residual, weights, 'norm_feedforward'), weights, 'feedforward.0').map(gelu);
    const final = add(residual, linear(hidden, weights, 'feedforward.2'));
    return linear(layerNorm(final, weights, 'norm_final'), weights, 'forecast').map(v => (v + 1) / 2);
  });
  return {
    predictions,
    traces,
    fullLatents,
    cache: {
      latents,
      rotaryKeys,
      positions
    }
  };
}
