export function stableProbabilities(scores) {
  const maximum = Math.max(...scores);
  const masses = scores.map(score => Math.exp(score - maximum));
  const total = masses.reduce((sum, mass) => sum + mass, 0);
  return masses.map(mass => mass / total);
}

export function encodeRequestTrace({ question, state, options }, vocabulary) {
  const words = text => text.toLowerCase().match(/[a-z]+/g) || [];
  if (!words(state).length) return { error: 'Enter at least one a–z word. This tokenizer cannot represent a digits-only or non-Latin request.' };
  const cells = [];
  const markers = [];
  function add(text, section, special = false) {
    for (const word of special ? [text] : words(text)) {
      cells.push({ word, section, position: cells.length, id: vocabulary[word] ?? 1, special });
    }
  }
  add('<cls>', 'question', true);
  add(question, 'question');
  add('<sep>', 'question', true);
  for (const option of options) {
    markers.push(cells.length);
    add('<option>', option.id, true);
    add(option.description, option.id);
    add('<sep>', option.id, true);
  }
  add(state, 'state');
  add('<sep>', 'state', true);
  if (cells.length > 128) return { error: `This request needs ${cells.length} tokens. The reference limit is 128; no text has been silently truncated.` };
  return { cells, markers, target: options.findIndex(option => option.id === 'billing'), unknownCount: cells.filter(cell => cell.id === 1).length };
}

export const attentionKeys = [[1, 0], [0, 1], [-1, 0], [4, 0]];
export const attentionValues = [[1, 0], [0, 2], [1, 1], [10, 10]];

export function attentionMixture(queryFirst, maskPadding) {
  const scores = attentionKeys.map(key => queryFirst * key[0] / Math.sqrt(2));
  const probabilities = stableProbabilities(scores.map((score, index) => maskPadding && index === 3 ? -Infinity : score));
  const contributions = attentionValues.map((value, index) => value.map(component => component * probabilities[index]));
  const output = [0, 1].map(column => contributions.reduce((sum, row) => sum + row[column], 0));
  return { scores, probabilities, contributions, output };
}

export const candidateFeatures = [[1, 0], [0, 1], [-1, 1]];

export function scoringHeadStep(weights, target, learningRate) {
  const scores = candidateFeatures.map(features => features.reduce((sum, value, index) => sum + value * weights[index], 0));
  const probabilities = stableProbabilities(scores);
  const scoreGradients = probabilities.map((probability, index) => probability - Number(index === target));
  const gradient = weights.map((_, column) => candidateFeatures.reduce((sum, features, index) => sum + scoreGradients[index] * features[column], 0));
  const nextWeights = weights.map((weight, index) => weight - learningRate * gradient[index]);
  const nextScores = candidateFeatures.map(features => features.reduce((sum, value, index) => sum + value * nextWeights[index], 0));
  const nextProbabilities = stableProbabilities(nextScores);
  return { scores, probabilities, scoreGradients, gradient, nextWeights, nextProbabilities, loss: -Math.log(probabilities[target]), nextLoss: -Math.log(nextProbabilities[target]) };
}
