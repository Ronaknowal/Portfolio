export default {
  topicId: 'string-matching-prefix-functions-rolling-hashes',
  verifiedOn: '10 September 2026',
  introduction: 'Choose the state from the matching contract. These six problems vary first occurrence, borders, periods, repeated streams, exact alphabet encoding and palindromic prefixes. They supplement the local proof and collision exercises.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · exact occurrences and reusable borders',
    introduction: 'Attempt after the KMP and period sections. The platform difficulty is not a prerequisite list: Longest Happy Prefix is direct prefix-table transfer even though its label is Hard.',
    problems: [{
      number: 28,
      title: 'Find the Index of the First Occurrence in a String',
      slug: 'find-the-index-of-the-first-occurrence-in-a-string',
      difficulty: 'Easy',
      focus: 'Adapt all-occurrence search to a first-start or −1 result. The official input is nonempty lowercase English text and pattern; do not infer its empty policy from this lesson.',
      hint: 'What can you return the moment a full prefix has matched? Which information still matters after a mismatch?',
      transfer: 'Return every overlap, then the final occurrence. Test pattern longer than text and repeated near-matches. Extend the API to an empty pattern explicitly rather than relying on the platform constraints.'
    }, {
      number: 1392,
      title: 'Longest Happy Prefix',
      slug: 'longest-happy-prefix',
      difficulty: 'Hard',
      focus: 'Return the longest nonempty proper prefix that is also a suffix, or an empty result. Overlap is permitted; the entire string is excluded.',
      hint: 'Which final table entry already describes the required length?',
      transfer: 'Return all border lengths in decreasing order. Explain why stepping through the border of the previous border visits every candidate without scanning every length.'
    }, {
      number: 459,
      title: 'Repeated Substring Pattern',
      slug: 'repeated-substring-pattern',
      difficulty: 'Easy',
      focus: 'Decide whether a nonempty lowercase string is made from two or more whole copies of a shorter word. A partial final copy does not qualify.',
      hint: 'A border determines a candidate shift. What extra arithmetic condition makes that shift tile the entire string?',
      transfer: 'Return the shortest period even when it does not divide the length. Use ababa to separate those contracts, then return the primitive repeating block and copy count when tiling is possible.'
    }]
  }, {
    id: 'core',
    title: 'Core · boundaries and exact compact representations',
    introduction: 'Use the streaming and fixed-alphabet sections. Explain why the finite search bound or encoding is valid before coding.',
    problems: [{
      number: 686,
      title: 'Repeated String Match',
      slug: 'repeated-string-match',
      difficulty: 'Medium',
      focus: 'Find the minimum number of repetitions of one nonempty word that contains another word contiguously, or prove impossibility.',
      hint: 'A possible match can be moved to a start in the first copy without changing its symbols. How many copies can its end reach?',
      transfer: 'Prove that checking ceil(len(b)/len(a)) copies and one additional copy suffices. Feed virtual characters or whole copies without allocating an unbounded repeated text. Explain why merely checking the ceiling number can fail.'
    }, {
      number: 187,
      title: 'Repeated DNA Sequences',
      slug: 'repeated-dna-sequences',
      difficulty: 'Medium',
      focus: 'Report distinct length-10 windows that occur repeatedly in an A/C/G/T string. Overlapping occurrences count; each repeated word is returned once.',
      hint: 'How much information is in ten base-4 digits? When is a rolling integer an exact encoding rather than a lossy fingerprint?',
      transfer: 'Generalize the width and define behavior for an unknown nucleotide. Separate the fixed-width machine-word cost from arbitrary-width integer arithmetic, and compare results with direct substring counting.'
    }]
  }, {
    id: 'extension',
    title: 'Extension · reduce a new question to prefix structure',
    optional: true,
    introduction: 'Attempt after the palindromic-prefix derivation. Reversal here is over the official lowercase alphabet; a user-visible Unicode palindrome needs a separate unit contract.',
    problems: [{
      number: 214,
      title: 'Shortest Palindrome',
      slug: 'shortest-palindrome',
      difficulty: 'Hard',
      prerequisite: 'Longest proper borders, a unique sentinel and the proof that the retained prefix must be palindromic.',
      focus: 'Prepend the fewest symbols to make a palindrome; appending or inserting elsewhere is not permitted. The official statement permits an empty string.',
      hint: 'If a suffix is going to be mirrored in front, what property must the part left in the middle satisfy? Compare the original prefix with a suffix of the reversed string.',
      transfer: 'Use input containing your first-choice separator. Replace a literal separator with a distinct object in a sequence. Derive the analogous append-only transformation; explain why longest palindromic substring solves a different problem.'
    }]
  }],
  readiness: ['Can trace an unfamiliar repetitive pattern and explain each rejected alignment, not only recite the KMP loop.', 'Can preserve all starts under different chunk partitions and distinguish normalized indices from raw input indices.', 'Can exhibit unequal strings with equal hashes and state both false-candidate and true-hit verification costs.', 'Can reconstruct a solution later without the hint and select among KMP, Z, direct comparison, a dictionary trie and a verified fingerprint from changed constraints.'],
  localBridge: 'Retain the independent streaming, hash-collision and coordinate exercises even if every submission passes. These problems do not establish a worst-case proof, Unicode policy or production throughput; later specializations extend the toolkit to multiple patterns, approximate matching and text indexes.'
};
