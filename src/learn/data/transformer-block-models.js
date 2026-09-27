import { add, attention, dense, dot, gelu, layerNorm, mean, scale, softmax } from './sequence-tensor-operations.js';
export const blockDefault = () => ({
  inputs: [[1, 2, 5, 8], [3, 0, 2, 1]],
  up: [[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]],
  down: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, -.5]],
  branch: 1,
  pre: true,
  epsilon: 1e-5
});
const rowMultiply = (row, matrix) => matrix[0].map((_, j) => dot(row, matrix.map(values => values[j])));
export function gatedFeedforwardTrace(input, up, gate, down) {
  const value = rowMultiply(input, up);
  const gateLogits = rowMultiply(input, gate);
  const gateResponse = gateLogits.map(x => x / (1 + Math.exp(-x)));
  const hidden = value.map((x, i) => x * gateResponse[i]);
  return { value, gateLogits, gateResponse, hidden,
    contributions: hidden.map((x, i) => scale(down[i], x)), output: rowMultiply(hidden, down) };
}
export function feedforwardTrace(input, up, down, activation = 'relu') {
  const responses = rowMultiply(input, up);
  const hidden = responses.map(value => activation === 'gelu' ? gelu(value) : activation === 'silu' ? value / (1 + Math.exp(-value)) : Math.max(0, value));
  const contributions = hidden.map((value, i) => scale(down[i], value));
  return {
    responses,
    hidden,
    contributions,
    output: rowMultiply(hidden, down)
  };
}
export function traceBlock(settings) {
  const {
    inputs,
    up,
    down,
    branch,
    pre,
    epsilon
  } = settings;
  const normalize = row => layerNorm(row, [], [], epsilon);
  const attentionInput = pre ? inputs.map(normalize) : inputs;
  const mixed = attention(attentionInput, attentionInput, attentionInput.map(row => scale(row, .25)));
  const update = mixed.output.map(row => scale(row, branch));
  const residual = inputs.map((row, i) => add(row, update[i]));
  const context = pre ? residual : residual.map(normalize);
  const ffnInput = pre ? context.map(normalize) : context;
  const feedforward = ffnInput.map(row => feedforwardTrace(row, up, down));
  const ffnUpdate = feedforward.map(result => scale(result.output, branch));
  const sum = context.map((row, i) => add(row, ffnUpdate[i]));
  return {
    input: inputs,
    attentionInput,
    weights: mixed.weights,
    update,
    residual,
    context,
    ffnInput,
    hidden: feedforward.map(value => value.hidden),
    ffnUpdate,
    output: pre ? sum : sum.map(normalize)
  };
}
export function normalizationProbe(input, probe, epsilon = 1e-5, step = 1e-5) {
  const center = mean(input),
    deviations = input.map(value => value - center);
  const denominator = Math.sqrt(mean(deviations.map(value => value * value)) + epsilon);
  const meanProbe = mean(probe),
    projection = dot(deviations, probe) / (input.length * denominator ** 2);
  const gradient = probe.map((value, i) => (value - meanProbe - deviations[i] * projection) / denominator);
  const finite = input.map((_, i) => {
    const plus = input.map((value, j) => value + (i === j ? step : 0));
    const minus = input.map((value, j) => value - (i === j ? step : 0));
    return (dot(layerNorm(plus, [], [], epsilon), probe) - dot(layerNorm(minus, [], [], epsilon), probe)) / (2 * step);
  });
  return {
    gradient,
    finite,
    scalar: dot(layerNorm(input, [], [], epsilon), probe)
  };
}
export function movementBlockForward(model, points, times, padding = []) {
  if (!points.length || points.length > 60 || points.some(row => row.length !== 2 || row.some(value => !Number.isFinite(value))) || times.length !== points.length || times.some(value => !Number.isFinite(value)) || points.every((_, i) => padding[i])) throw new Error('Use 1–60 finite point/time records with at least one valid record.');
  const weights = model.state_dict;
  const linear = (row, prefix) => dense(row, weights[`${prefix}.weight`], weights[`${prefix}.bias`]);
  const norm = (row, prefix) => layerNorm(row, weights[`${prefix}.weight`], weights[`${prefix}.bias`]);
  let state = points.map(([x, y], i) => linear([2 * x - 1, 2 * y - 1, times[i]], 'stem'));
  const traces = [];
  for (let layer = 0; layer < 2; layer++) {
    const prefix = `blocks.${layer}`;
    const input = state;
    const attentionInput = model.preNorm ? input.map(row => norm(row, `${prefix}.norm_attention`)) : input;
    const packed = attentionInput.map(row => dense(row, weights[`${prefix}.attention.in_proj_weight`], weights[`${prefix}.attention.in_proj_bias`]));
    const heads = [0, 1].map(head => attention(packed.map(row => row.slice(12 * head, 12 * head + 12)), packed.map(row => row.slice(24 + 12 * head, 36 + 12 * head)), packed.map(row => row.slice(48 + 12 * head, 60 + 12 * head)), (_, key) => !padding[key]));
    const update = input.map((_, i) => linear([...heads[0].output[i], ...heads[1].output[i]], `${prefix}.attention.out_proj`));
    const residual = input.map((row, i) => add(row, update[i]));
    const context = model.preNorm ? residual : residual.map(row => norm(row, `${prefix}.norm_attention`));
    const ffnInput = model.preNorm ? context.map(row => norm(row, `${prefix}.norm_feedforward`)) : context;
    const hidden = ffnInput.map(row => linear(row, `${prefix}.feedforward.0`).map(gelu));
    const ffnUpdate = hidden.map(row => linear(row, `${prefix}.feedforward.2`));
    const sum = context.map((row, i) => add(row, ffnUpdate[i]));
    state = model.preNorm ? sum : sum.map(row => norm(row, `${prefix}.norm_feedforward`));
    traces.push({
      input,
      attentionInput,
      update,
      residual,
      context,
      ffnInput,
      hidden,
      ffnUpdate,
      output: state,
      weights: heads.map(head => head.weights)
    });
  }
  const valid = state.map((row, i) => padding[i] ? null : norm(row, 'final_norm')).filter(Boolean);
  const pooled = valid[0].map((_, i) => mean(valid.map(row => row[i])));
  const logits = linear(pooled, 'classifier');
  return {
    logits,
    probabilities: softmax(logits),
    traces
  };
}
export function blockCosts(length, width = 512, hidden = 2048, heads = 8) {
  return {
    parameters: 4 * width ** 2 + 2 * width * hidden + hidden + 9 * width,
    maps: 4 * length * width ** 2 + 2 * length * width * hidden,
    pairs: 2 * length ** 2 * width,
    attentionEntries: heads * length ** 2,
    cacheEntries: 2 * length * width,
    hiddenEntries: length * hidden
  };
}
