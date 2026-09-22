export const architectureKinds = [['plain', 'Plain'], ['residual', 'Residual'], ['parallel', 'Parallel branches'], ['inverted_gated', 'Inverted with gate']];
export function headBudget({ channels, height, width, dense, classes, mode = 'dense' }) {
  const flattened = channels * height * width;
  const pieces = mode === 'dense' ? [(flattened + 1) * dense, (dense + 1) * dense, (dense + 1) * classes] : [(mode === 'flat' ? flattened + 1 : channels + 1) * classes];
  const macs = mode === 'dense' ? flattened * dense + dense * dense + dense * classes : (mode === 'flat' ? flattened : channels) * classes;
  const parameters = pieces.reduce((a, b) => a + b, 0);
  return { pieces, parameters, macs, fourArraysBytes: parameters * 16, gapParameters: (channels + 1) * classes, gapMacs: channels * classes };
}
export function channelContext(maps) {
  const means = maps.map(map => map.reduce((sum, value) => sum + value, 0) / map.length);
  const hidden = Math.max(means[0] - means[1], 0);
  const gates = [1 / (1 + Math.exp(-hidden)), 1 / (1 + Math.exp(hidden))];
  return { means, hidden, gates, output: maps.map((map, i) => map.map(value => value * gates[i])) };
}
export function scalingBudget(depth, width, resolution) {
  return { parameters: depth * width ** 2, macs: depth * width ** 2 * resolution ** 2 };
}
export function solveBudgetWidth(budget, depth, resolution) {
  const width = Math.sqrt(budget / (depth * resolution ** 2));
  return { width, feasible: width >= 1 && width <= 3 };
}
export function signedScoreMap(maps, weights, bias) {
  const contributions = maps.map((map, channel) => map.map(value => value * weights[channel]));
  const totalMap = maps[0].map((_, cell) => contributions.reduce((sum, map) => sum + map[cell], 0));
  const means = maps.map(map => map.reduce((sum, value) => sum + value, 0) / map.length);
  const viaFeatures = means.reduce((sum, value, channel) => sum + value * weights[channel], bias);
  const viaMap = totalMap.reduce((sum, value) => sum + value, 0) / totalMap.length + bias;
  return { contributions, totalMap, means, viaFeatures, viaMap };
}
export function eligibleArchitectures(fits, parameters, macs) {
  return fits.filter(row => row.cost.parameters <= parameters && row.cost.macs <= macs).map(row => row.kind);
}
