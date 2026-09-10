export default {
  topicId: 'reductions-p-np-computational-intractability',
  verifiedOn: '2026-09-10',
  introduction: 'These directly inspected official statements let you practise how numerical bounds, structural restrictions and small search dimensions affect algorithm choice. They supplement the local reduction proofs; solving a platform instance is not a proof of a complexity classification. No judge submission or editorial access is claimed.',
  groups: [
    {
      id: 'encoded-size-and-structure', title: 'Revisit a familiar algorithm under a sharper contract',
      introduction: 'Use earlier DP skills to explain why these particular inputs are manageable. Platform difficulty is metadata, not a theorem about asymptotic complexity.',
      problems: [
        { number: 416, title: 'Partition Equal Subset Sum', slug: 'partition-equal-subset-sum', difficulty: 'Medium', focus: 'Relate an equal partition to an exact numerical target, and distinguish item count from target magnitude.', prerequisite: 'Positive values, at most 200 items and values at most 100; review once-only subset DP.', hint: 'If the sum is even, which target would one part have to reach?', transfer: 'Return original indices as a certificate. Replace small values by arbitrary binary integers: explain why a target-sized table loses its polynomial input-length guarantee, while the same recurrence stays correct.' },
        { number: 198, title: 'House Robber', slug: 'house-robber', difficulty: 'Medium', focus: 'Recognize that consecutive-position conflicts form a path and allow a sufficient prefix state.', prerequisite: 'Nonnegative rewards; only neighboring positions conflict. The local path-independent-set example permits empty input and negative weights as additional cases.', hint: 'Separate the best prefix solution that includes its last position from one that excludes it.', transfer: 'Add an arbitrary conflict between distant positions. Draw a counterexample to the unchanged recurrence, then explain why the hard general graph problem does not make its path restriction hard.' },
      ],
    },
    {
      id: 'bounded-search', title: 'Choose an exact search dimension deliberately',
      introduction: 'A small finite bound can make a correct exponential approach useful without changing the complexity of an unrestricted family.',
      problems: [
        { number: 698, title: 'Partition to K Equal Sum Subsets', slug: 'partition-to-k-equal-sum-subsets', difficulty: 'Medium', focus: 'Return or check a partition witness while reasoning about a search space indexed by items or bucket assignments.', prerequisite: 'At most 16 positive items; k nonempty parts, each with equal sum. Review backtracking ownership or DP subset state before attempting.', hint: 'What cheap necessary condition follows from the total? What state preserves which occurrences remain available?', transfer: 'Keep duplicate values as separate indices and validate every index is used exactly once. Explain why a timeout is not evidence of impossibility, and why adding one more bucket changes the contract from the two-part case.' },
      ],
    },
  ],
  readiness: [
    'State the decision version, input encoding, certificate size, verifier bound and the exact quantifiers for yes and no instances.',
    'Draw A ≤p B and justify yes preservation, no preservation, polynomial conversion and polynomial output length before transferring a solver or hardness.',
    'Recover an assignment from a clause-occurrence clique and carry the same witness through complement and vertex-cover identities.',
    'Distinguish membership in NP, NP hardness, NP completeness, unresolved P versus NP and practical performance on a restricted input family.',
    'Choose and justify a pseudopolynomial, parameterized, restricted exact or approximation contract; return witnesses and meaningful bounds.',
  ],
  localBridge: 'The local tasks assess reduction direction, faulty gadgets, encoding growth, existential certificates, approximation bounds and restricted-state failures. These theoretical obligations are not adequately tested by an accepted platform submission. Close the hints and explain a changed case independently before continuing to randomized algorithms.',
};
