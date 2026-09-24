import fs from 'node:fs';
import assert from 'node:assert/strict';
import practice from '../src/learn/data/practice/arrays-strings-hash-maps.js';
import brief from '../src/learn/data/curriculum/blueprints/arrays-strings-hash-maps.js';

assert(!practice.groups.some(group => group.id === 'bitwise-foundations'));
const bitwiseGroup = {
  id: 'bitwise-foundations', title: 'Bitwise foundation · membership, parity and width',
  introduction: 'Use sections 9–12 first. These four public official statements were inspected on 11 September 2026; their integer and multiplicity promises are part of the problem, not optional implementation details.',
  problems: [
    { number: 136, title: 'Single Number', slug: 'single-number', difficulty: 'Easy',
      focus: 'Derive paired-value cancellation from per-bit parity, then explain how the exact input promise turns a fold into the required answer.',
      prerequisite: 'The XOR invariant and invalid-promise counterexamples in section 11; the judge also permits negative values.',
      hint: 'What does combining a value with itself contribute at every bit? Separate the identity computed by the loop from the multiplicity promise needed to interpret it.',
      transfer: 'Test a zero singleton and a negative singleton. Replace a pair with three equal occurrences and explain why a plausible XOR result no longer proves uniqueness; compare with the earlier first-unique-event task.' },
    { number: 191, title: 'Number of 1 Bits', slug: 'number-of-1-bits', difficulty: 'Easy',
      focus: 'Count occupied positions in a positive bounded integer. Explain the bit removed by each loop iteration instead of memorizing an expression.',
      prerequisite: 'The subtraction/AND proof in section 12; the inspected statement uses a positive value through 2^31 − 1.',
      hint: 'Compare a positive number with the number one smaller at and below its lowest 1. Which positions survive AND?',
      transfer: 'Extend to zero, then explicitly choose whether a negative input means its magnitude or a fixed-width word. State how repeated calls or very large Python integers change the engineering question without making an unsupported timing claim.' },
    { number: 231, title: 'Power of Two', slug: 'power-of-two', difficulty: 'Easy',
      focus: 'Turn the shape of a binary representation into a necessary-and-sufficient test, including the no-loop follow-up.',
      hint: 'How many occupied positions does a positive power of two have? Which input satisfies the clearing equality despite not being a power?',
      transfer: 'Explain both directions of the proof and test zero, negative values and one. Changing the base to three does not preserve the one-set-bit argument; identify what must be rederived.' },
    { number: 461, title: 'Hamming Distance', slug: 'hamming-distance', difficulty: 'Easy',
      focus: 'Compose a per-position difference indicator with population count for nonnegative bounded values. Distinguish bit differences from numerical distance.',
      hint: 'Which truth-table operation gives 1 exactly where two input bits differ? What quantity should then be counted?',
      transfer: 'Compare 7 and 8, whose numerical difference is one but whose bit patterns differ at four positions. For signed input, require a declared width before counting representation differences.' },
  ],
};
const twoSingletonGroup = {
  id: 'bitwise-partition', title: 'Optional bitwise transfer · separate two survivors', optional: true,
  introduction: 'Return after the two-singleton proof and complete program in section 13. This official statement was inspected on 11 September 2026; it asks for two values in any order, with a linear-time and constant-extra-space target.',
  problems: [{ number: 260, title: 'Single Number III', slug: 'single-number-iii', difficulty: 'Medium',
    focus: 'Find one bit that distinguishes the two surviving values, then prove equal pairs stay in the same partition.',
    prerequisite: 'XOR cancellation, lowest-set-bit isolation, and the two-pass reusable-input contract from section 13.',
    hint: 'The total XOR does not directly give either singleton. What does a 1 in that total reveal about those two values at the corresponding position?',
    transfer: 'Use a zero singleton, negative inputs and both output orders. Explain why a nonzero total alone is not a full promise check and why a one-shot iterator cannot be silently traversed twice.' }],
};
practice.groups.splice(2, 0, bitwiseGroup, twoSingletonGroup);
practice.verifiedOn = '10 September 2026 for the original ten problems, and 11 September 2026 for the five bitwise additions';
practice.introduction += ' The bitwise stages belong to the deeper representation branch: read its proofs and contract counterexamples before attempting them.';
practice.readiness.push('Choose membership, exact counts or parity from the contract; justify width/sign assumptions, input promises and working-word versus bit costs.');
practice.localBridge += ' Also solve the changed bitmap report and signed-word task locally; judge success on a promised input is not a validation proof for arbitrary data.';
fs.writeFileSync('src/learn/data/practice/arrays-strings-hash-maps.js', 'export default ' + JSON.stringify(practice, null, 2) + ';\n');

brief.summary += ' A deeper bitwise branch derives finite bitsets, signed-word interpretation, XOR promises, sparse population count and two-singleton partitioning.';
brief.outcomes.push('Represent a finite set with bit positions and prove intersection/union/difference/XOR/complement membership',
  'Decode signed and unsigned words; distinguish explicit width from Python integer shifts and magnitude bit_count',
  'Prove one- and two-singleton XOR algorithms under their exact promises and diagnose invalid inputs',
  'Derive lowest-bit clearing, positive power tests and Hamming distance; account for arbitrary-integer bit costs');
brief.sequence.push('After the original independent task, offer a deeper representation branch without changing module order',
  'Derive binary place values and per-bit truth rules; compare bitmap postings with ordinary posting sets',
  'Derive two’s-complement signed weights, sign/zero fill and explicit truncation versus Python integers',
  'Prove the prefix-XOR invariant and parity cancellation; inspect negative/zero and invalid-promise examples',
  'Prove borrow-based lowest-bit clearing and population-count termination; apply XOR to Hamming distance',
  'Optionally isolate a distinguishing bit and recover two singletons in two passes',
  'Solve changed finite-set, word, promise and bitmap-report tasks before guided bitwise practice');
brief.visuals.push(
  { type: 'Aligned finite-set membership columns', question: 'Which members satisfy this set query?', interaction: 'Toggle two sets, change the declared universe or query and inspect each resulting membership bit.' },
  { type: 'Weighted word and source-position shift map', question: 'How can the same bits mean a negative value, and where do shifted bits go?', interaction: 'Toggle a four/eight-bit word and compare unsigned/signed sums and zero/sign-filled shift results.' },
  { type: 'Parity stream with separate frequency audit', question: 'What does XOR preserve, and which promise turns it into a singleton?', interaction: 'Step bounded events, apply custom data, compare one/two/invalid-promise scenarios and inspect extra diagnostic storage.' },
  { type: 'Borrow suffix and disappearing-bit trace', question: 'Why does one AND remove exactly one set position?', interaction: 'Build a byte and step x, x−1 and their AND while preserving the removed-plus-remaining count invariant.' },
  { type: 'Two-bucket singleton partition figure', question: 'Why do pairs stay together while the two survivors separate?', interaction: 'A static computed split for the worked example makes the bucket and cancellation proof visible.' },
);
brief.practice.task += ' The deeper branch has five additional complete native programs and four visible changed-contract tasks, plus four core and one optional verified bitwise problem.';
brief.practice.success += ' Bitmap queries match sets, word shifts match the declared interpretation, XOR answers are used only under valid promises, and count/partition witnesses agree with independent oracles.';
brief.misconceptions.push('Arithmetic addition is not idempotent membership insertion', 'Finite complement needs a universe', 'Signed two’s complement is not sign plus magnitude', 'XOR retains parity, not exact frequencies or first-occurrence order', 'A bounded number of Python integers is not a constant bit budget', 'Zero XOR/nonzero XOR alone does not validate a singleton promise');
brief.sources.push('https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types', 'https://cses.fi/book/book.pdf');
brief.reviewFocus += ' The bitwise extension preserves all original teaching and validates actual exported models/native programs against sets, binary-string arithmetic and frequencies; full desktop/narrow/keyboard reading plus independent review are required. Native integer rules follow Python, not the Handbook’s nonportable C++ width/overflow assumptions.';
brief.extensionDesignRecord = 'docs/teaching/BITWISE-FOUNDATIONS-EXTENSION-DESIGN.md';
fs.writeFileSync('src/learn/data/curriculum/blueprints/arrays-strings-hash-maps.js', '// Topic-owned plan; see original and extension design records.\nexport default ' + JSON.stringify(brief, null, 2) + ';\n');
console.log('Extended Arrays plan and practice: original ten problems conserved, five bitwise problems added.');
