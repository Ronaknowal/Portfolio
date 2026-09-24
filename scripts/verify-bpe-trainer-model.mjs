import assert from 'node:assert/strict';
import { DEFAULT_BPE_CORPUS, prepareBpeCorpus, nextBpePair, mergeBpePair, learnedBpeVocabulary } from '../src/learn/data/bpe-trainer-model.js';

const original = prepareBpeCorpus(DEFAULT_BPE_CORPUS);
assert.deepEqual(Object.values(original.state).map(word => word.count), [5, 2, 6, 3]);
assert.deepEqual(nextBpePair(original.state), { pair: ['e', 's'], count: 9 });
const alphabet = [...original.alphabet];
let state = original.state, merges = [];
for (let index = 0; index < 8; index += 1) {
  const chosen = nextBpePair(state), before = JSON.stringify(state);
  const result = mergeBpePair(state, chosen.pair);
  assert.equal(JSON.stringify(state), before, 'Each merge leaves its input state intact');
  state = result.next;
  merges.push({ ...chosen, merged: result.merged });
  const vocabulary = learnedBpeVocabulary(alphabet, merges);
  assert(alphabet.every(symbol => vocabulary.includes(symbol)), 'Initial symbols remain available after merging');
  assert(merges.every(merge => vocabulary.includes(merge.merged)));
  for (const [word, value] of Object.entries(state)) {
    assert.equal(value.symbols.join(''), `${word}</w>`, 'Segmentation conserves the exact word');
    assert.equal(value.count, original.state[word].count);
  }
}
assert.equal(learnedBpeVocabulary(alphabet, merges).length, alphabet.length + 8);
const repeated = prepareBpeCorpus('aaaa aaaa').state;
assert.deepEqual(nextBpePair(repeated), { pair: ['a', 'a'], count: 6 });
assert.deepEqual(mergeBpePair(repeated, ['a', 'a']).next.aaaa.symbols, ['aa', 'aa', '</w>']);
const reservedNames = prepareBpeCorpus('__proto__ constructor toString __proto__');
assert.equal(reservedNames.state.__proto__.count, 2);
assert.equal(reservedNames.state.constructor.count, 1);
assert.equal(reservedNames.state.toString.count, 1);
assert.deepEqual(prepareBpeCorpus('😀😀').state['😀😀'].symbols, ['😀', '😀', '</w>']);
for (const invalid of ['', ' \n\t ', 'a'.repeat(41), 'a '.repeat(601), Array.from({ length: 41 }, (_, index) => `w${index}`).join(' '), '</w>']) {
  assert(prepareBpeCorpus(invalid).error, 'Invalid/oversized input must be explained, not silently changed');
}
let terminal = prepareBpeCorpus('x').state;
terminal = mergeBpePair(terminal, nextBpePair(terminal).pair).next;
assert.equal(nextBpePair(terminal), null);
console.log('PASS BPE trainer: original weighted merges, cumulative vocabulary, exact reconstruction, overlap rule, Unicode, reserved object keys, bounds, empty and terminal states.');
