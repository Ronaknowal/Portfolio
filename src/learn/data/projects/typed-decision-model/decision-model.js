export function exploreDecision({ logits, temperature, wrongCost, reviewCost }) {
  const scaled = logits.map(value => value / temperature);
  const maximum = Math.max(...scaled);
  const masses = scaled.map(value => Math.exp(value - maximum));
  const total = masses.reduce((sum, value) => sum + value, 0);
  const probabilities = masses.map(value => value / total);
  const bestIndex = probabilities.indexOf(Math.max(...probabilities));
  const entropy = -probabilities.reduce((sum, probability) => sum + probability * Math.log(probability), 0);
  const expectedActCost = wrongCost * (1 - probabilities[bestIndex]);
  const tolerance = 1e-12 * Math.max(1, Math.abs(expectedActCost), Math.abs(reviewCost));
  return {
    probabilities,
    bestIndex,
    concentration: 1 - entropy / Math.log(probabilities.length),
    expectedActCost,
    action: expectedActCost < reviewCost - tolerance ? 'act' : 'review',
  };
}
