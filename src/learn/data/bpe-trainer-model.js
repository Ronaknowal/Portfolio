// Bounded character-level teaching BPE; not a production tokenizer.
export const DEFAULT_BPE_CORPUS = 'low low low low low lower lower newest newest newest newest newest newest widest widest widest';
export const BPE_MERGE_LIMIT = 64;

export function prepareBpeCorpus(corpus) {
  if ([...corpus].length > 1200) return { error: 'Use at most 1,200 characters in this small trainer.' };
  if (corpus.includes('</w>')) return { error: 'The literal </w> marker is reserved for word ends in this teaching model.' };
  const words = corpus.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return { error: 'Enter a few words to inspect their merges.' };
  if (new Set(words).size > 40) return { error: 'Use at most 40 distinct words in this small trainer.' };
  if (words.some(word => [...word].length > 40)) return { error: 'Keep each word to 40 characters or fewer.' };
  const state = Object.create(null);
  for (const word of words) {
    if (state[word]) state[word].count += 1;
    else state[word] = { symbols: [...word, '</w>'], count: 1 };
  }
  return { state, alphabet: [...new Set(Object.values(state).flatMap(word => word.symbols))] };
}

export function nextBpePair(state) {
  const counts = new Map();
  for (const { symbols, count } of Object.values(state)) {
    for (let index = 0; index < symbols.length - 1; index += 1) {
      const pair = JSON.stringify([symbols[index], symbols[index + 1]]);
      counts.set(pair, (counts.get(pair) || 0) + count);
    }
  }
  let best = null;
  for (const [key, count] of counts) {
    if (!best || count > best.count) best = { pair: JSON.parse(key), count };
  }
  return best;
}

export function mergeBpePair(state, [left, right]) {
  const merged = left + right, next = Object.create(null);
  for (const [word, { symbols, count }] of Object.entries(state)) {
    const result = [];
    for (let index = 0; index < symbols.length; index += 1) {
      if (symbols[index] === left && symbols[index + 1] === right) {
        result.push(merged);
        index += 1;
      } else result.push(symbols[index]);
    }
    next[word] = { symbols: result, count };
  }
  return { next, merged };
}

export function learnedBpeVocabulary(alphabet, merges) {
  return [...new Set([...alphabet, ...merges.map(merge => merge.merged)])].sort();
}
