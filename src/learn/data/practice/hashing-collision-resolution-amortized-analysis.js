export default {
  topicId: 'hashing-collision-resolution-amortized-analysis',
  verifiedOn: '10 September 2026',
  introduction: 'Practice three different kinds of reasoning: implementing a map contract, composing representations, and charging many apparent searches to a bounded set of objects. The official statements below are public; their platform labels do not distinguish expected from amortized costs for you.',
  groups: [
    {
      id: 'foundation',
      title: 'Foundation · implement the collision contract',
      introduction: 'The bounded key universe in these two official tasks permits direct addressing. First explain that baseline and its space cost; then deliberately require sparse or unbounded integer keys and implement a collision-resolving structure without a built-in hash table.',
      problems: [
        {
          number: 705, title: 'Design HashSet', slug: 'design-hashset', difficulty: 'Easy',
          focus: 'Make repeated add idempotent, absent remove harmless, and contains correct after collisions and deletion. The official task forbids built-in hash-table libraries.',
          hint: 'A set needs one identity per key. If you use probing, what distinguishes a slot that certifies absence from one that merely became reusable?',
          transfer: 'Use keys that all share a home, delete the first, and query the last. Then fill every slot, delete all entries, and insert again. Explain why every scan still terminates.',
        },
        {
          number: 706, title: 'Design HashMap', slug: 'design-hashmap', difficulty: 'Easy',
          focus: 'Add values and replacement without increasing the distinct-key count. Its −1 missing result is unambiguous because official stored values are nonnegative.',
          hint: 'The first tombstone may be a suitable destination, but have you already ruled out an equal key farther along the probe sequence?',
          transfer: 'Allow −1 and None as stored values and redesign the lookup result. Add resizing; verify the entire abstract mapping across every rebuild, including replacement just at a growth threshold.',
        },
      ],
    },
    {
      id: 'core',
      title: 'Core · combine invariants and charge the work',
      introduction: 'Attempt these after sections 7–8. State what each stored map value means. A table lookup is one component of the argument, not a complete complexity proof.',
      problems: [
        {
          number: 380, title: 'Insert Delete GetRandom O(1)', slug: 'insert-delete-getrandom-o1', difficulty: 'Medium',
          focus: 'Maintain a bijection between a dense array and its reverse index. The statement requires each stored value to have equal sampling probability and guarantees a nonempty set when sampling.',
          hint: 'If iteration order is irrelevant, which single element can replace a removed array entry without shifting the rest?',
          transfer: 'Remove a middle value, the last value and an absent value. Explain the probability argument, the separate expected/amortized costs, and why allowing duplicate values would change both the invariant and the sampling contract.',
        },
        {
          number: 128, title: 'Longest Consecutive Sequence', slug: 'longest-consecutive-sequence', difficulty: 'Medium',
          focus: 'Find a longest integer run in an unsorted array, including duplicates and an empty input. Explain the requested linear time under the expected-cost hash assumption.',
          hint: 'Which one value should be allowed to start scanning each maximal run? What should the outer iteration contain when the input has duplicates?',
          transfer: 'Count membership checks independently of the hash implementation. Compare with a sorted-unique oracle, then return a run as well as its length and specify ties. Explain what changes with adversarial collisions.',
        },
        {
          number: 454, title: '4Sum II', slug: '4sum-ii', difficulty: 'Medium',
          focus: 'Count index quadruples from four arrays. Pair-sum frequencies retain multiplicity; a set of possible sums loses required information.',
          hint: 'Once one pair sums to s, which other pair sum can finish a zero total, and how many different index pairs can supply it?',
          transfer: 'Use four arrays containing only repeated zeroes. Generalize to unequal lengths and a target other than zero; choose a pairing deliberately and state the storage bound in distinct pair sums, not merely input elements.',
        },
      ],
    },
  ],
  readiness: [
    'Can exhibit a concrete deletion failure, preserve the probe invariant and handle a full table without an unbounded loop.',
    'Can separate worst-case collision work, expected lookup work and amortized rebuild work with explicit assumptions.',
    'Can rebuild from a blank editor later, compare against an independent baseline and explain why a changed identity or multiplicity contract requires different state.',
    'Can mix a hash solution with sorting, a balanced search structure or direct addressing and select based on order, key universe, latency and adversarial requirements.',
  ],
  localBridge: 'These questions exercise this lesson’s mechanisms. Keep the local resize-policy proof and exact hash-family experiment as separate practice: passing the platform tasks alone does not validate those guarantees. Later string-matching, range-query and graph lessons add different uses and cost models.',
};
