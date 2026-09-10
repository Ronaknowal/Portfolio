// Small, exact teaching examples; these are not measured performance data.
export function pairedSaving(shift = 0) {
  const differences = [2, 4, -1, 3, 2].map(value => value + shift);
  const mean = differences.reduce((sum, value) => sum + value, 0) / differences.length;
  const variance = differences.reduce((sum, value) => sum + (value - mean) ** 2, 0) / 4;
  const se = Math.sqrt(variance / 5);
  const margin = 2.7764451051977987 * se; // Two-sided 95% t interval, df = 4.
  return {differences, mean, se, low: mean - margin, high: mean + margin};
}

export function laplacianRowExample(split = false) {
  const center = 1;
  const neighbors = [
    {name: 'A', value: 1, weight: 1},
    {name: 'B', value: 1, weight: 1},
    {name: 'D', value: split ? -1 : 1, weight: 0.2},
  ].map(node => ({...node, contribution: node.weight * (center - node.value)}));
  return {center, neighbors, total: neighbors.reduce((sum, node) => sum + node.contribution, 0)};
}
