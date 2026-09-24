// Bounded teaching mechanisms. The digit investigation uses fixed measured weights.
export const channelFixture = { inputs: [[1, 2, 3], [2, 0, 1]], filters: [[1, -1], [.5, 1]], mixing: [2, -1] };
export const initialSignal = [1, 2, 3, 4, 5, 6, 7, 8, 9];

export function channelForward(inputs, filters, mixing) {
  const filtered = inputs.map((row, channel) => [0, 1].map(position => filters[channel].reduce((sum, weight, tap) => sum + weight * row[position + tap], 0)));
  const contributions = filtered.map((row, channel) => row.map(value => value * mixing[channel]));
  return { filtered, contributions, output: [0, 1].map(position => contributions.reduce((sum, row) => sum + row[position], 0)) };
}

export function channelGradient(inputs, filters, mixing, target, rate) {
  const before = channelForward(inputs, filters, mixing);
  const derivative = before.output[0] - target;
  const mixingGradient = before.filtered.map(row => derivative * row[0]);
  const filterGradient = filters.map((row, channel) => row.map((_, tap) => derivative * mixing[channel] * inputs[channel][tap]));
  const nextMixing = mixing.map((value, index) => value - rate * mixingGradient[index]);
  const nextFilters = filters.map((row, channel) => row.map((value, tap) => value - rate * filterGradient[channel][tap]));
  const after = channelForward(inputs, nextFilters, nextMixing);
  return { mixingGradient, filterGradient, nextMixing, nextFilters, before: before.output[0], after: after.output[0], loss: derivative ** 2 / 2, nextLoss: (after.output[0] - target) ** 2 / 2 };
}

export function rankProbe(filters, mixing, dependent = false) {
  const kernel = mixing.map(row => [0, 1].map(tap => filters.reduce((sum, filter, component) => sum + row[component] * filter[tap], 0)));
  const probes = [[2, 0], dependent ? [2, 0] : [0, 3]];
  const outputs = probes.map(probe => kernel.map(row => row.reduce((sum, weight, tap) => sum + weight * probe[tap], 0)));
  const residuals = outputs.map((row, index) => row.map((value, coordinate) => value - probes[index][coordinate]));
  return { kernel, probes, outputs, residuals, error: Math.max(...residuals.flat().map(Math.abs)) };
}

export function sampleStencil(signal, center, dilation, weights = [1, 1, 1]) {
  const terms = weights.map((weight, tap) => {
    const index = center + (tap - 1) * dilation;
    const padded = index < 0 || index >= signal.length;
    return { index, padded, value: padded ? 0 : signal[index], weight, product: weight * (padded ? 0 : signal[index]) };
  });
  return { terms, sum: terms.reduce((sum, term) => sum + term.product, 0) };
}

export function samplingSupport(rates) {
  if (!rates.length || rates.length > 4 || rates.some(rate => !Number.isInteger(rate) || rate < 1 || rate > 9)) throw new Error('Use one to four integer rates from 1 to 9.');
  let sites = [0];
  const layers = [sites];
  for (const rate of rates) {
    sites = [...new Set(sites.flatMap(site => [-rate, 0, rate].map(offset => site + offset)))].sort((a, b) => a - b);
    layers.push(sites);
  }
  const bound = rates.reduce((sum, rate) => sum + rate, 0);
  const holes = Array.from({ length: 2 * bound + 1 }, (_, index) => index - bound).filter(site => !sites.includes(site));
  return { sites, layers, bound, width: 2 * bound + 1, holes };
}

export function finiteTaps(row, column, dilation) {
  return [-1, 0, 1].flatMap(dr => [-1, 0, 1].map(dc => [row + dr * dilation, column + dc * dilation])).filter(([r, c]) => r >= 0 && r < 8 && c >= 0 && c < 8);
}

export function parallelContext(signal, mixing, rates = [1, 2, 4]) {
  const branches = [signal[4], ...rates.map(rate => sampleStencil(signal, 4, rate).sum), signal.reduce((sum, value) => sum + value, 0) / signal.length];
  return { branches, contributions: branches.map((value, index) => value * mixing[index]), output: branches.reduce((sum, value, index) => sum + value * mixing[index], 0) };
}

function convolve(image, weights, bias, dilation, groups = 1) {
  const channels = image.length, height = image[0].length, width = image[0][0].length;
  const outputs = weights.length, taps = weights[0][0].length, padding = Math.floor(taps / 2) * dilation;
  return weights.map((kernel, output) => Array.from({ length: height }, (_, row) => Array.from({ length: width }, (_, column) => {
    let sum = bias?.[output] || 0;
    const channelStart = Math.floor(output / (outputs / groups)) * (channels / groups);
    for (let localChannel = 0; localChannel < kernel.length; localChannel++) {
      for (let kr = 0; kr < taps; kr++) for (let kc = 0; kc < taps; kc++) {
        const r = row - padding + kr * dilation, c = column - padding + kc * dilation;
        if (r >= 0 && r < height && c >= 0 && c < width) sum += image[channelStart + localChannel][r][c] * kernel[localChannel][kr][kc];
      }
    }
    return sum;
  })));
}
const relu = maps => maps.map(map => map.map(row => row.map(value => Math.max(0, value))));

export function savedDigitLogits(run, pixels, multiplier = null) {
  const state = run.dense_state;
  const stem = relu(convolve([pixels], state['stem.weight'], state['stem.bias'], 1));
  let spatial;
  if (multiplier === null) spatial = convolve(stem, state['spatial.weight'], state['spatial.bias'], run.dilation);
  else {
    const factor = run.factorizations.find(item => item.multiplier === multiplier).factorized_spatial_state;
    const filtered = convolve(stem, factor['0.weight'], null, run.dilation, 8);
    spatial = convolve(filtered, factor['1.weight'], factor['1.bias'], 1);
  }
  const features = relu(spatial);
  const pooled = features.flatMap(map => Array.from({ length: 4 }, (_, row) => Array.from({ length: 4 }, (_, column) => (map[2 * row][2 * column] + map[2 * row + 1][2 * column] + map[2 * row][2 * column + 1] + map[2 * row + 1][2 * column + 1]) / 4)).flat());
  return state['head.weight'].map((weights, output) => weights.reduce((sum, weight, index) => sum + weight * pooled[index], state['head.bias'][output]));
}

export function reconstructedKernel(run, multiplier, input, output) {
  const factor = run.factorizations.find(item => item.multiplier === multiplier).factorized_spatial_state;
  return Array.from({ length: 3 }, (_, row) => Array.from({ length: 3 }, (_, column) => Array.from({ length: multiplier }, (_, component) => {
    const channel = input * multiplier + component;
    return factor['1.weight'][output][channel][0][0] * factor['0.weight'][channel][0][row][column];
  }).reduce((sum, value) => sum + value, 0)));
}
