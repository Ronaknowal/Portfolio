export default {
  topicId: 'binary-search-sorting-two-pointer-patterns',
  verifiedOn: '2026-09-10',
  introduction: 'Use the problem contracts to choose what can be discarded or summarized. These stages cover boundary search, ordering, pointer movement and windows; after learning them, mix the problems without looking at the stage names. An accepted answer is a starting point for explaining and adapting the invariant.',
  groups: [
    {
      id: 'boundaries', title: 'Reconstruct boundary search',
      introduction: 'Distinguish a matching index from an insertion boundary and a duplicate range.',
      problems: [
        { number: 704, title: 'Binary Search', slug: 'binary-search', difficulty: 'Easy', focus: 'Return a matching index or the absence sentinel while preserving logarithmic search work.', prerequisite: 'The statement uses a nonempty array with unique values; the local boundary function also handles empty and duplicate inputs.', hint: 'If you find an insertion boundary, what final check distinguishes a match from an absent target?', transfer: 'Allow duplicates and require the first matching index. Explain which condition changes and which invariant remains.' },
        { number: 35, title: 'Search Insert Position', slug: 'search-insert-position', difficulty: 'Easy', focus: 'Return the gap where a target belongs, including after the final element.', hint: 'Could the correct return value equal the array length? Decide before choosing high.', transfer: 'Return a position after every equal value instead. Test all-equal input and a value below the minimum.' },
        { number: 34, title: 'Find First and Last Position of Element in Sorted Array', slug: 'find-first-and-last-position-of-element-in-sorted-array', difficulty: 'Medium', focus: 'Build an inclusive answer range from two half-open boundaries without scanning a duplicate run.', hint: 'What does the difference between the first ≥ boundary and first > boundary count?', transfer: 'Count values in [a,b) and explain why no target+1 trick is necessary for a general ordered key.' },
      ],
    },
    {
      id: 'ordering', title: 'Preserve information while creating order',
      introduction: 'Reconstruct a sort, then handle a different storage or endpoint contract.',
      problems: [
        { number: 912, title: 'Sort an Array', slug: 'sort-an-array', difficulty: 'Medium', focus: 'Produce a sorted permutation with justified O(n log n) time and an explicit space tradeoff.', prerequisite: 'Read the optional iterative heapsort implementation for the statement’s smallest-space objective. The lesson’s buffer-based mergesort uses linear auxiliary space; built-in sorting is disallowed here.', hint: 'Check both sorted order and preservation of all occurrences. Which taught algorithm avoids a merge buffer?', transfer: 'Now require stable ordering of records with equal keys. Explain why your chosen method or its space requirements may need to change.' },
        { number: 88, title: 'Merge Sorted Array', slug: 'merge-sorted-array', difficulty: 'Easy', focus: 'Merge into the first array’s spare capacity without overwriting unread source entries.', prerequisite: 'The first m entries are real input; reserved trailing zeroes are capacity, while zero can also be a legitimate input value.', hint: 'Which end has a known destination that is safe to overwrite?', transfer: 'Attach original record IDs and require stability across both inputs. Decide which source wins a tie when writing from the end.' },
        { number: 26, title: 'Remove Duplicates from Sorted Array', slug: 'remove-duplicates-from-sorted-array', difficulty: 'Easy', focus: 'Separate a retained logical prefix from the physical list and advance read/write positions safely.', hint: 'State what the first write entries mean before inspecting the next source item.', transfer: 'Allow at most two copies of each value. Derive the acceptance test from the retained prefix, rather than memorizing another loop.' },
        { number: 56, title: 'Merge Intervals', slug: 'merge-intervals', difficulty: 'Medium', focus: 'Use start order to maintain the full union and handle nested intervals.', prerequisite: 'The official contract merges touching endpoints. Treat these as closed intervals; do not import a different meeting-room endpoint policy.', hint: 'When can a new interval affect any component other than the last merged one?', transfer: 'Switch to half-open occupancy intervals. Separate overlap from the independent policy of coalescing adjacent coverage.' },
      ],
    },
    {
      id: 'opposing', title: 'Explain what each pointer move eliminates',
      introduction: 'Use a proof about the complete remaining candidate region, then adapt the output contract.',
      problems: [
        { number: 167, title: 'Two Sum II - Input Array Is Sorted', slug: 'two-sum-ii-input-array-is-sorted', difficulty: 'Medium', focus: 'Find a pair using opposing pointers and constant extra space.', prerequisite: 'Return 1-based indices, never reuse one position, and account for the statement’s exactly-one-solution promise.', hint: 'When the endpoint sum is too small, which endpoint cannot work with any remaining partner?', transfer: 'Remove the one-solution promise. First return any pair, then discuss how enumerating index pairs changes output size and duplicate handling.' },
        { number: 15, title: '3Sum', slug: '3sum', difficulty: 'Medium', focus: 'Reduce one dimension of the search while suppressing duplicate value triples.', hint: 'After fixing one sorted value, what two-value sum remains? Which repeats would recreate an already emitted value triple?', transfer: 'Ask for all index triples instead of unique value triples. Explain why skipping equal values would now lose valid answers.' },
      ],
    },
    {
      id: 'windows', title: 'Choose the right summary for the input',
      introduction: 'Fixed size, monotone validity, uniqueness and exact signed sums demand different state.',
      problems: [
        { number: 643, title: 'Maximum Average Subarray I', slug: 'maximum-average-subarray-i', difficulty: 'Easy', focus: 'Reuse overlap in a fixed-width window; negative values are allowed.', hint: 'With a fixed positive width, which sum gives the largest average? What leaves when the right boundary advances?', transfer: 'Return the earliest maximizing interval as well as its average. Test all-negative values and ties.' },
        { number: 209, title: 'Minimum Size Subarray Sum', slug: 'minimum-size-subarray-sum', difficulty: 'Medium', focus: 'Justify growth and repeated shrinking for a shortest qualifying positive-sum window.', prerequisite: 'The statement uses positive values and returns zero when no interval qualifies; the local implementation returns an optional half-open interval.', hint: 'Once a start position has produced a qualifying interval, can a later right endpoint improve the length for that same start?', transfer: 'Allow negative values and exhibit a failing trace. For the optional O(n log n) follow-up on positive values, explain why prefix totals can be searched.' },
        { number: 3, title: 'Longest Substring Without Repeating Characters', slug: 'longest-substring-without-repeating-characters', difficulty: 'Medium', focus: 'Maintain a valid contiguous window and prevent the left boundary moving backward.', prerequisite: 'Previously an Arrays preview; the uniqueness-window mechanism is taught here. The statement distinguishes substring from subsequence.', hint: 'A repeated character before the current window must not pull the left boundary backward.', transfer: 'Return the earliest longest interval and state your character unit. Discuss the extra segmentation needed for user-perceived Unicode characters.' },
        { number: 560, title: 'Subarray Sum Equals K', slug: 'subarray-sum-equals-k', difficulty: 'Medium', focus: 'Count earlier prefix totals, including repeated totals, for nonempty signed subarrays.', prerequisite: 'Previously an Arrays preview; use this lesson’s prefix-frequency derivation rather than applying the positive-window rule.', hint: 'For current prefix P, which earlier prefix would make their difference k? Why count before inserting the current prefix?', transfer: 'Use three zeroes with target zero to distinguish a frequency map from a set. Then require minimum length and reconsider what each stored prefix entry must retain.' },
      ],
    },
    {
      id: 'answer-space', title: 'Transfer binary search to a feasible answer',
      introduction: 'The searched values are candidate decisions, not entries in the input array.',
      problems: [
        { number: 875, title: 'Koko Eating Bananas', slug: 'koko-eating-bananas', difficulty: 'Medium', focus: 'Build a monotone feasibility predicate with per-pile rounded time and justified integer bounds.', prerequisite: 'The official constraints guarantee at least one hour per pile. Contrast this with the local impossible-budget state.', hint: 'Unused capacity in an hour does not carry to another pile. What changes when speed increases?', transfer: 'Permit sharing spare capacity across piles during one hour. Re-derive the time formula and check whether the previous answer remains valid.' },
      ],
    },
  ],
  readiness: ['Explain why low=mid can fail to make progress and why high=n is a legal boundary.', 'Prove stable merging and each pointer move using an invariant, not a lucky trace.', 'Choose a window or prefix method from the input contract and construct a counterexample to the wrong one.', 'State sorting/preprocessing, query, auxiliary and output costs separately; handle duplicate identities and endpoint policies.', 'Reattempt a hinted problem later with hints closed, then mix unlabelled tasks so choosing the pattern is part of the work.'],
  localBridge: 'Finish the local time-range report and its signed/duplicate/empty tests first. This set builds transferable reasoning, but broader preparation still includes backtracking, dynamic programming, greedy proofs and later graph/string/range topics; no finite checklist guarantees every unfamiliar problem.',
};
