// Mechanism and frozen-model computations deliberately independent of React.
const sum = values => values.reduce((a, b) => a + b, 0);
export const dot = (a, b) => sum(a.map((value, i) => value * b[i]));
export function softmax(scores) {
  const maximum = Math.max(...scores);
  const mass = scores.map(value => Math.exp(value - maximum));
  return mass.map(value => value / sum(mass));
}
export function patchProjection(pixels, weights, biases) {
  const contributions = weights.map(row => row.map((weight, i) => pixels[i] * weight));
  return { contributions, output: contributions.map((row, i) => sum(row) + biases[i]) };
}
export function imagePatches(image, side = 2) {
  const patches = [];
  for (let row = 0; row < image.length; row += side) for (let column = 0; column < image[0].length; column += side) {
    const pixels = [], coordinates = [];
    for (let r = 0; r < side; r += 1) for (let c = 0; c < side; c += 1) { pixels.push(image[row + r][column + c]); coordinates.push([row + r, column + c]); }
    patches.push({ pixels, coordinates, row: row / side, column: column / side });
  }
  return patches;
}
export function visionMACs(height, width, patch = 16, channels = 3, dimension = 768, blocks = 12, classes = 1000, prefixes = 1) {
  const patches = height / patch * width / patch, sequence = patches + prefixes;
  const projection = 4 * sequence * dimension ** 2, ffn = 8 * sequence * dimension ** 2, pairs = 2 * sequence ** 2 * dimension;
  return { patches, sequence, projection, ffn, pairs, total: patches * patch ** 2 * channels * dimension + blocks * (projection + ffn + pairs) + dimension * classes, pairFraction: pairs / (projection + ffn + pairs) };
}
export function windowRead(values, height, width, window, shift, { wrap = false, bias = null } = {}) {
  // The browser's fixed 6x6 grid is divisible by both supported window widths.
  // The full native module separately handles padding and learned QKV projections.
  const coordinates = Array.from({ length: height * width }, (_, i) => [Math.floor(i / width), i % width]);
  const rolled = coordinates.map(([r, c]) => [(r - shift + height) % height, (c - shift + width) % width]);
  const allowed = coordinates.map(([row, column], receiver) => coordinates.map(([r, c], donor) => {
    if (wrap) return Math.floor(rolled[receiver][0] / window) === Math.floor(rolled[donor][0] / window) && Math.floor(rolled[receiver][1] / window) === Math.floor(rolled[donor][1] / window);
    return Math.floor((row - shift) / window) === Math.floor((r - shift) / window) && Math.floor((column - shift) / window) === Math.floor((c - shift) / window);
  }));
  const weights = allowed.map((row, receiver) => {
    const scores = row.map((legal, donor) => {
      if (!legal) return -Infinity;
      const dr = rolled[receiver][0] % window - rolled[donor][0] % window;
      const dc = rolled[receiver][1] % window - rolled[donor][1] % window;
      return bias?.[dr + window - 1]?.[dc + window - 1] ?? 0;
    });
    return softmax(scores);
  });
  return { coordinates, allowed, weights, output: weights.map(row => dot(row, values)) };
}
export function composeInfluence(second, first) {
  return second.map(row => first[0].map((_, donor) => sum(row.map((weight, intermediate) => weight * first[intermediate][donor]))));
}
export function dinoDirection(teacher, center, student, teacherTemperature, studentTemperature, offset = 0) {
  if (Math.min(teacherTemperature, studentTemperature) <= 0) throw new Error('Temperatures must be positive.');
  const centered = teacher.map((v, i) => v + offset - center[i]);
  const target = softmax(centered.map(v => v / teacherTemperature)), probability = softmax(student.map(v => v / studentTemperature));
  const scaled = student.map(v => v / studentTemperature), maximum = Math.max(...scaled);
  const logPartition = maximum + Math.log(sum(scaled.map(v => Math.exp(v - maximum))));
  return { centered, target, probability, gradient: probability.map((p, i) => (p - target[i]) / studentTemperature), loss: -sum(target.map((p, i) => p * (scaled[i] - logPartition))) };
}
export function gramGeometry(angles, referenceAngles = [0, 90, 180, 270]) {
  const vectors = angles.map(angle => [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)]);
  const baseline = referenceAngles.map(angle => [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)]);
  const gram = vectors.map(a => vectors.map(b => dot(a, b))), reference = baseline.map(a => baseline.map(b => dot(a, b)));
  const difference = gram.map((row, i) => row.map((v, j) => v - reference[i][j]));
  return { vectors, gram, reference, difference, loss: sum(difference.flat().map(v => v * v)) };
}
function erf(value) {
  const x = Math.abs(value);
  if (x > 8) return Math.sign(value);
  let term = x, total = x;
  for (let n = 1; n < 400; n += 1) { term *= 2 * x * x / (2 * n + 1); total += term; if (term <= total * 2e-16) break; }
  return Math.sign(value) * Math.min(1, 2 / Math.sqrt(Math.PI) * Math.exp(-x * x) * total);
}
const add = (a, b) => a.map((v, i) => v + b[i]);
const linear = (input, state, prefix) => state[`${prefix}.weight`].map((row, i) => dot(row, input) + (state[`${prefix}.bias`]?.[i] ?? 0));
function norm(input, state, prefix) {
  const mean = sum(input) / input.length, variance = sum(input.map(v => (v - mean) ** 2)) / input.length;
  return input.map((v, i) => (v - mean) / Math.sqrt(variance + 1e-5) * state[`${prefix}.weight`][i] + state[`${prefix}.bias`][i]);
}
export function frozenVision(state, image, { order = Array.from({ length: 16 }, (_, i) => i), movePositions = false, pca = null } = {}) {
  const raw = imagePatches(image).map(patch => patch.pixels);
  const projected = raw.map(patch => state['patch.weight'].map((weight, i) => dot(weight.flat(Infinity), patch) + state['patch.bias'][i]));
  let tokens = [add(state.cls[0][0], state.cls_position[0][0]), ...order.map((source, slot) => add(projected[source], state.patch_position[0][movePositions ? source : slot]))];
  const attention = [];
  for (let block = 0; block < 2; block += 1) {
    const prefix = `blocks.${block}`;
    const qkv = tokens.map(row => linear(norm(row, state, `${prefix}.norm1`), state, `${prefix}.qkv`));
    const allWeights = [], heads = [];
    for (let head = 0; head < 4; head += 1) {
      const q = qkv.map(row => row.slice(head * 8, head * 8 + 8));
      const k = qkv.map(row => row.slice(32 + head * 8, 40 + head * 8));
      const v = qkv.map(row => row.slice(64 + head * 8, 72 + head * 8));
      const weights = q.map(query => softmax(k.map(key => dot(query, key) / Math.sqrt(8))));
      allWeights.push(weights);
      heads.push(weights.map(row => Array.from({ length: 8 }, (_, axis) => sum(v.map((value, donor) => value[axis] * row[donor])))));
    }
    attention.push(allWeights);
    tokens = tokens.map((row, receiver) => {
      const mixed = heads.flatMap(head => head[receiver]);
      const residual = add(row, linear(mixed, state, `${prefix}.output`));
      const hidden = linear(norm(residual, state, `${prefix}.norm2`), state, `${prefix}.ffn.0`).map(v => .5 * v * (1 + erf(v / Math.sqrt(2))));
      return add(residual, linear(hidden, state, `${prefix}.ffn.2`));
    });
  }
  const features = tokens.map(row => norm(row, state, 'norm'));
  const logits = linear(features[0], state, 'head'), probabilities = softmax(logits);
  const patchPCA = pca ? features.slice(1).map(row => [0, 1, 2].map(axis => sum(row.map((v, j) => (v - pca.training_mean[j]) * pca.basis[j][axis])))) : null;
  return { raw, projected, features, attention, logits, probabilities, predicted: probabilities.indexOf(Math.max(...probabilities)), patchPCA };
}
export function featureCosines(query, candidates) {
  const norm = row => Math.sqrt(dot(row, row)), qNorm = norm(query);
  return candidates.map(row => qNorm > 0 && norm(row) > 0 ? dot(query, row) / (qNorm * norm(row)) : null);
}
