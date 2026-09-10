export default {
  summary: 'Reuse exact prefix evidence to find overlapping patterns, preserve matches across streamed chunks, and distinguish rolling fingerprints from verified string identity.',
  outcomes: ['Specify zero-based code-point offsets, overlaps and empty-pattern behavior', 'Construct a prefix table and justify each fallback without rereading consumed text', 'Implement all-match KMP and account for total comparisons', 'Carry exact matching state across decoded chunk boundaries', 'Derive borders, periods and prepend-palindrome transformations', 'Roll and query polynomial fingerprints while verifying collisions and stating honest costs', 'Distinguish Unicode coordinate systems and normalization contracts', 'Use restricted-alphabet exact encoding and Z-box reuse in appropriate applications'],
  prerequisites: ['Arrays, Strings & Hash Maps', 'Complexity Analysis & Recursion', 'Hashing, Collision Resolution & Amortized Analysis'],
  sequence: ['Define exact occurrences and inspect a correct baseline', 'See a border as reusable prefix and suffix evidence', 'Build the prefix function through nested candidates', 'Search without consuming a symbol on a fallback', 'Carry prefix state across stream chunks', 'Use borders to expose periods and palindromic prefixes', 'Derive rolling updates and separate candidates from matches', 'Cancel prefix fingerprints and state collision assumptions', 'Choose Unicode units and restricted-alphabet representations', 'Contrast Z-box reuse with prefix-ending evidence', 'Implement and explain independent changes of contract'],
  visual: {
    type: 'Aligned overlap ribbons, paired prefix/suffix builder, KMP alignment trace, chunk delivery strip, rolling arithmetic and collision ledger, Unicode coordinate rows and Z-box reuse',
    question: 'Which evidence survives this mismatch, boundary or arithmetic compression?',
    interaction: 'Predict fallbacks, step and reset exact comparisons, feed chunks with a deliberately faulty reset mode, and vary tiny fingerprints while inspecting actual identity.'
  },
  practice: {
    task: 'Build exact detectors and prefix tables; prove overlap/stream preservation; expose collision and verification-cost mistakes; adapt to periods, Unicode and restricted alphabets.',
    success: 'Matches independent slicing and enumeration oracles, declares coordinates and empty cases, proves evidence sufficiency and cost assumptions, and transfers beyond the demonstrated input.'
  },
  misconceptions: ['A substring may skip symbols', 'A prefix table stores text indices', 'Every mismatch consumes the current text symbol', 'Resetting after a match preserves overlaps', 'A network chunk boundary is a semantic separator', 'The shortest period must divide the whole length', 'A matching rolling hash proves equality', 'Verifying all true hits is always linear', 'Code-point, UTF-16 and byte offsets are interchangeable', 'A fixed toy hash has a demonstrated random collision bound'],
  sources: ['https://users.encs.concordia.ca/~chvatal/notes/kmp.html', 'https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/160b3b5f9da2e03815ca1e6ee0dba62a_MIT6_006F11_lec09.pdf', 'https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/zalg.pdf', 'https://docs.python.org/3/howto/unicode.html'],
  depth: 'specialist',
  reviewFocus: 'Border length versus position conventions, fallback consumption, all-overlap reporting, chunk partition invariance, direct polynomial verification, true-hit output costs, Unicode coordinate changes and independent native/browser evidence.',
  designRecord: 'docs/teaching/STRING-MATCHING-LESSON-DESIGN.md'
};
