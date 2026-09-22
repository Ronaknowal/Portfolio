// Pure bounded protocol mechanisms. Token support and gate order match the saved CPU model.
export const sequenceTokens = ['<pad>', '<bos>', '<eos>', '<past>', '<participle>', '<third_person>', ...'abcdefghijklmnopqrstuvwxyz'];
export const allowedSequenceOutputs = [2, ...Array.from({ length: 26 }, (_, index) => index + 6)];
export const sequenceIndex = Object.fromEntries(sequenceTokens.map((token, index) => [token, index]));
export function shiftedTracks(target, padding = 0, leak = false) {
  const outputs = [...target, '<eos>'];
  const targets = [...outputs, ...Array(padding).fill('<pad>')];
  // Shift the padded batch target, matching BOS + target[:, :-1]. The first
  // ignored target therefore receives EOS rather than an invented PAD input.
  return { inputs: leak ? [...targets] : ['<bos>', ...targets.slice(0, -1)], targets, valid: outputs.length };
}

export function scalarBridge(inputs = [-.3, .6], weight = .7, rate = .1, detached = false) {
  let state = 0, sensitivity = 0;
  const encoder = [];
  for (const value of inputs) {
    state = Math.tanh(weight * value + .4 * state + .1);
    sensitivity = (1 - state ** 2) * (value + .4 * sensitivity);
    encoder.push(state);
  }
  const context = state;
  const decoder = [.1, .4].map(embedding => { state = Math.tanh(.6 * embedding + .5 * state + .05); return state; });
  const probabilityA = decoder.map(value => 1 / (1 + Math.exp(-2 * value)));
  const losses = [-Math.log(probabilityA[0]), -Math.log1p(-probabilityA[1])];
  const contextGradient = ((probabilityA[0] - 1) + probabilityA[1] * .5 * (1 - decoder[1] ** 2)) * .5 * (1 - decoder[0] ** 2);
  const gradient = detached ? 0 : contextGradient * sensitivity;
  const nextWeight = weight - rate * gradient;
  // The two-token mean cancels the factor two from [s,-s] softmax derivatives.
  let next = 0;
  for (const value of inputs) next = Math.tanh(nextWeight * value + .4 * next + .1);
  const nextProbabilities = [.1, .4].map(embedding => { next = Math.tanh(.6 * embedding + .5 * next + .05); return 1 / (1 + Math.exp(-2 * next)); });
  return { encoder, context, decoder, probabilityA, losses, loss: (losses[0] + losses[1]) / 2, contextGradient, sensitivity, gradient, nextWeight, nextLoss: (-Math.log(nextProbabilities[0]) - Math.log1p(-nextProbabilities[1])) / 2 };
}

export function probabilityTree(firstA = .55, afterAEnd = .6, afterBEnd = .85, width = 2) {
  let candidates = [{ path: '', probability: 1 }];
  const history = [];
  for (let step = 0; step < 3; step++) {
    const expanded = candidates.flatMap(candidate => {
      const { path, probability } = candidate;
      if (path.endsWith('!')) return [candidate];
      if (!path) return [{ path: 'A', probability: probability * firstA }, { path: 'B', probability: probability * (1 - firstA) }];
      if (path === 'A' || path === 'B') {
        const end = path === 'A' ? afterAEnd : afterBEnd;
        return [{ path: path + '!', probability: probability * end }, { path: path + 'C', probability: probability * (1 - end) }];
      }
      return [{ path: path + '!', probability }];
    }).sort((a, b) => b.probability - a.probability || a.path.localeCompare(b.path));
    candidates = expanded.slice(0, width);
    history.push({ candidates, pruned: expanded.slice(width) });
  }
  return { history, winner: candidates[0], leaves: [{ path: 'A!', probability: firstA * afterAEnd }, { path: 'AC!', probability: firstA * (1 - afterAEnd) }, { path: 'B!', probability: (1 - firstA) * afterBEnd }, { path: 'BC!', probability: (1 - firstA) * (1 - afterBEnd) }] };
}
export const lengthScore = (logProbability, length, alpha) => logProbability / (((5 + length) / 6) ** alpha);
const dot = (weights, values) => weights.reduce((sum, weight, index) => sum + weight * values[index], 0);

export function sequenceGru(vector, hidden, weights, name) {
  const incoming = weights[`${name}.weight_ih_l0`].map((row, index) => dot(row, vector) + weights[`${name}.bias_ih_l0`][index]);
  const recurrent = weights[`${name}.weight_hh_l0`].map((row, index) => dot(row, hidden) + weights[`${name}.bias_hh_l0`][index]);
  const width = hidden.length;
  return hidden.map((value, index) => {
    const reset = 1 / (1 + Math.exp(-(incoming[index] + recurrent[index])));
    const retain = 1 / (1 + Math.exp(-(incoming[width + index] + recurrent[width + index])));
    const candidate = Math.tanh(incoming[2 * width + index] + reset * recurrent[2 * width + index]);
    return retain * value + (1 - retain) * candidate;
  });
}

export function encodeSequence(weights, lemma, feature) {
  if (!/^[a-z]{1,12}$/.test(lemma) || !['past', 'participle', 'third_person'].includes(feature)) throw new Error('Use 1–12 lowercase a–z letters and a supported grammatical request.');
  const ids = [sequenceIndex[`<${feature}>`], ...[...lemma].map(letter => sequenceIndex[letter]), 2];
  let state = Array(64).fill(0);
  const states = ids.map(id => { state = sequenceGru(weights['embedding.weight'][id], state, weights, 'encoder'); return state; });
  return { ids, states, context: state };
}

export function sequenceStep(weights, state, previous) {
  const nextState = sequenceGru(weights['embedding.weight'][previous], state, weights, 'decoder');
  const logits = weights['readout.weight'].map((row, index) => allowedSequenceOutputs.includes(index) ? dot(row, nextState) + weights['readout.bias'][index] : -Infinity);
  const maximum = Math.max(...logits), exponentials = logits.map(value => Math.exp(value - maximum));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  return { state: nextState, probabilities: exponentials.map(value => value / denominator), logProbabilities: logits.map(value => value - maximum - Math.log(denominator)) };
}

export function generatedTrace(weights, lemma, feature, { prefix = '', context = null, cap = 16 } = {}) {
  if (!/^[a-z]{0,6}$/.test(prefix) || !Number.isInteger(cap) || cap < 1 || cap > 16) throw new Error('Use up to six prefix letters and an integer limit from 1 to 16.');
  const encoded = encodeSequence(weights, lemma, feature);
  let state = context || encoded.context, previous = 1;
  const steps = [], tokens = [];
  for (let index = 0; index < cap; index++) {
    const current = sequenceStep(weights, state, previous);
    const selected = index < prefix.length ? sequenceIndex[prefix[index]] : current.probabilities.indexOf(Math.max(...current.probabilities));
    steps.push({ ...current, previous, selected, forced: index < prefix.length });
    tokens.push(selected); state = current.state; previous = selected;
    if (selected === 2) break;
  }
  return { ...encoded, usedContext: context || encoded.context, steps, tokens, word: tokens.filter(token => token !== 2).map(token => sequenceTokens[token]).join(''), ended: tokens.at(-1) === 2, logProbability: steps.reduce((sum, step) => sum + step.logProbabilities[step.selected], 0) };
}

export function sequenceBeam(weights, lemma, feature, width = 3, cap = 16) {
  const encoded = encodeSequence(weights, lemma, feature);
  let candidates = [{ tokens: [], state: encoded.context, logProbability: 0 }];
  const history = [];
  const order = (left, right) => {
    if (left.logProbability !== right.logProbability) return right.logProbability - left.logProbability;
    for (let index = 0; index < Math.min(left.tokens.length, right.tokens.length); index++) if (left.tokens[index] !== right.tokens[index]) return left.tokens[index] - right.tokens[index];
    return left.tokens.length - right.tokens.length;
  };
  for (let step = 0; step < cap; step++) {
    const expanded = candidates.flatMap(candidate => {
      if (candidate.tokens.at(-1) === 2) return [candidate];
      const next = sequenceStep(weights, candidate.state, candidate.tokens.at(-1) ?? 1);
      return allowedSequenceOutputs.map(token => ({ tokens: [...candidate.tokens, token], state: next.state, logProbability: candidate.logProbability + next.logProbabilities[token] }));
    });
    candidates = expanded.sort(order).slice(0, width);
    history.push(candidates);
    if (candidates.every(candidate => candidate.tokens.at(-1) === 2)) break;
  }
  return { candidates, history };
}
