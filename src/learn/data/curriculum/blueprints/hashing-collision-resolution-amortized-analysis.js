export default {
  summary: 'Build collision-resolving maps with explicit deletion and rebuild contracts, then separate randomized lookup guarantees from sequence-wide resize accounting.',
  outcomes: ['Preserve complete key/value identity despite collisions', 'Prove bounded probe lookup and correct tombstone reuse', 'Rehash entries across capacity changes and track live versus used occupancy', 'Distinguish worst, expected, amortized and expected-amortized claims', 'Derive collision-indicator and geometric-growth bounds under explicit models', 'Prevent shrink/grow thrashing through separated thresholds', 'Compose a dense set and analyze consecutive runs and pair-frequency counting'],
  prerequisites: ['Arrays, Strings & Hash Maps', 'Complexity Analysis & Recursion', 'Algorithm Correctness, Loop Invariants & Termination'],
  sequence: ['Start from mapping identity and a list baseline', 'Compare chains and bounded probe sequences', 'Preserve lookup after deletion and replacement', 'Rebuild the abstract mapping under a new capacity', 'Enumerate a finite hash family and justify expectation', 'Prove aggregate and potential resize bounds', 'Separate grow/shrink thresholds on mixed updates', 'Compose reverse indices and charge work to unique objects', 'Choose practical hash contracts and alternatives', 'Implement, prove, counterexample and transfer independently'],
  visual: {
    type: 'Key-to-home inline figures, EMPTY/DELETED probe trace, exact hash-family outcome lattice, paired operation-cost timelines and dense reverse-index mutation',
    question: 'What information permits this lookup to stop, and over which choices or operations is this cost bound taken?',
    interaction: 'Edit tiny operation streams, inject a deletion fault, step probes and rebuilds, vary a hash choice while holding keys fixed, compare resize policies and inspect swap-delete relationships.'
  },
  practice: {
    task: 'Implement absent/value-safe maps, repair deletion and resize failures, prove expected and amortized bounds, compose two representations and count multiplicities.',
    success: 'Matches independent dict/sort/exhaustive oracles, explains exact model counts and invariant/termination clauses, and adapts to changed key and ordering contracts.'
  },
  misconceptions: ['Equal hashes imply equal keys', 'A deleted probe slot can always become EMPTY', 'The first tombstone is safe to use before checking for an existing equal key', 'A low live load factor rules out long unsuccessful probes', 'Rehashing means copying the old physical positions', 'Expected and amortized mean the same kind of average', 'A pairwise collision bound automatically proves linear-probing performance', 'Shrinking as soon as a doubled table is half full preserves constant amortized updates'],
  sources: ['https://opendatastructures.org/ods-python/5_2_LinearHashTable_Linear_.html', 'https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ce9e94705b914598ce78a00a70a1f734_MIT6_006S20_lec4.pdf', 'https://docs.python.org/3/reference/datamodel.html#object.__hash__'],
  depth: 'core',
  reviewFocus: 'Distinct slot sentinels, bounded complete probe cycles, duplicate replacement, migration cost under collisions, exact threshold inequalities, declared allocation/hash costs, expectation over fixed inputs and independent native/browser evidence.',
  designRecord: 'docs/teaching/HASHING-AMORTIZED-LESSON-DESIGN.md'
};
