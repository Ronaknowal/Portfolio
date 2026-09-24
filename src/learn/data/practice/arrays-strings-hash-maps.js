export default {
  "topicId": "arrays-strings-hash-maps",
  "verifiedOn": "10 September 2026 for the original ten problems, and 11 September 2026 for the five bitwise additions",
  "introduction": "Practice choosing what to remember: positions, distinct values, frequencies or a reusable key. The first two stages use this lesson; the optional stage introduces additional algorithmic patterns for a later return. The bitwise stages belong to the deeper representation branch: read its proofs and contract counterexamples before attempting them.",
  "groups": [
    {
      "id": "foundation",
      "title": "Foundation · membership, counting and order",
      "introduction": "Begin after the array, text and hash-map investigations. Explain what each stored entry means before choosing a container.",
      "problems": [
        {
          "number": 217,
          "title": "Contains Duplicate",
          "slug": "contains-duplicate",
          "difficulty": "Easy",
          "focus": "Separate “have I seen this?” from “how many times?” and compare a repeated scan with a remembered prefix.",
          "hint": "At the start of an iteration, which earlier values must your state describe? Check the new value before updating that state.",
          "transfer": "Change the task to report the first position that repeats, then to report every repeated value once. State which version needs counts and which only needs membership."
        },
        {
          "number": 242,
          "title": "Valid Anagram",
          "slug": "valid-anagram",
          "difficulty": "Easy",
          "focus": "Match multiplicities rather than just the set of characters. The main task uses lowercase English letters; the Unicode follow-up needs an explicit text policy.",
          "hint": "Two strings may contain the same distinct letters with different frequencies. What summary distinguishes those cases without caring about order?",
          "transfer": "Use unequal frequencies as a counterexample to set equality. For arbitrary text, decide whether you compare code points or normalized text before discussing visible characters."
        },
        {
          "number": 387,
          "title": "First Unique Character in a String",
          "slug": "first-unique-character-in-a-string",
          "difficulty": "Easy",
          "focus": "Transfer the lesson’s first-unique-event exercise to a string index and a different absence marker.",
          "hint": "A character seen once so far may repeat later. Separate determining final frequencies from choosing the earliest qualifying position.",
          "transfer": "Explain why returning a character instead of its index fails the contract. Test a valid answer at index zero and an input with no unique character."
        }
      ]
    },
    {
      "id": "core",
      "title": "Core · combine storage with a clear invariant",
      "introduction": "Use changed contracts to make container choice deliberate. These tasks combine the lesson’s operations; they do not require learning an unrelated algorithm first.",
      "problems": [
        {
          "number": 283,
          "title": "Move Zeroes",
          "slug": "move-zeroes",
          "difficulty": "Easy",
          "focus": "Preserve relative order while editing the existing array. Count writes as well as visits, and distinguish allocated storage from the logical result.",
          "hint": "Describe the already processed portion: where should its nonzero values be, and which position is available for the next one?",
          "transfer": "Trace all zeroes, no zeroes and alternating values. Adapt the operation to remove a chosen sentinel value while returning the new logical length."
        },
        {
          "number": 349,
          "title": "Intersection of Two Arrays",
          "slug": "intersection-of-two-arrays",
          "difficulty": "Easy",
          "focus": "Revisit the search-index intersection from the lesson with duplicate inputs and an output that represents presence.",
          "hint": "Distinguish whether a value belongs to both inputs from how many times it appears. The output contract decides which information to retain.",
          "transfer": "Change the contract to preserve multiplicity. Explain why the same set-only representation now loses necessary information."
        },
        {
          "number": 1,
          "title": "Two Sum",
          "slug": "two-sum",
          "difficulty": "Easy",
          "focus": "Connect values to their original positions, and ensure two distinct elements satisfy the request.",
          "hint": "For the current value, what earlier value would complete the target? Be precise about what is stored before and after checking the current element.",
          "transfer": "Try a pair of equal values at different indices. Then remove the exactly-one-answer assumption and define whether you return one pair, every pair or absence."
        },
        {
          "number": 49,
          "title": "Group Anagrams",
          "slug": "group-anagrams",
          "difficulty": "Medium",
          "focus": "Design an immutable grouping key whose equality matches the intended text equivalence. This extends dictionary counting into a reusable index.",
          "prerequisite": "Be comfortable with frequency counts and hashable tuples from Python; the restricted alphabet makes a count-based key possible without a sorting lesson.",
          "hint": "What summary is identical for rearrangements of the same letters, yet differs when a letter’s multiplicity changes? Can that summary be a dictionary key?",
          "transfer": "Test empty strings and repeated identical words. Explain why a set of letters or a mutable list is an unsuitable key for this contract."
        }
      ]
    },
    {
      "id": "bitwise-foundations",
      "title": "Bitwise foundation · membership, parity and width",
      "introduction": "Use sections 9–12 first. These four public official statements were inspected on 11 September 2026; their integer and multiplicity promises are part of the problem, not optional implementation details.",
      "problems": [
        {
          "number": 136,
          "title": "Single Number",
          "slug": "single-number",
          "difficulty": "Easy",
          "focus": "Derive paired-value cancellation from per-bit parity, then explain how the exact input promise turns a fold into the required answer.",
          "prerequisite": "The XOR invariant and invalid-promise counterexamples in section 11; the judge also permits negative values.",
          "hint": "What does combining a value with itself contribute at every bit? Separate the identity computed by the loop from the multiplicity promise needed to interpret it.",
          "transfer": "Test a zero singleton and a negative singleton. Replace a pair with three equal occurrences and explain why a plausible XOR result no longer proves uniqueness; compare with the earlier first-unique-event task."
        },
        {
          "number": 191,
          "title": "Number of 1 Bits",
          "slug": "number-of-1-bits",
          "difficulty": "Easy",
          "focus": "Count occupied positions in a positive bounded integer. Explain the bit removed by each loop iteration instead of memorizing an expression.",
          "prerequisite": "The subtraction/AND proof in section 12; the inspected statement uses a positive value through 2^31 − 1.",
          "hint": "Compare a positive number with the number one smaller at and below its lowest 1. Which positions survive AND?",
          "transfer": "Extend to zero, then explicitly choose whether a negative input means its magnitude or a fixed-width word. State how repeated calls or very large Python integers change the engineering question without making an unsupported timing claim."
        },
        {
          "number": 231,
          "title": "Power of Two",
          "slug": "power-of-two",
          "difficulty": "Easy",
          "focus": "Turn the shape of a binary representation into a necessary-and-sufficient test, including the no-loop follow-up.",
          "hint": "How many occupied positions does a positive power of two have? Which input satisfies the clearing equality despite not being a power?",
          "transfer": "Explain both directions of the proof and test zero, negative values and one. Changing the base to three does not preserve the one-set-bit argument; identify what must be rederived."
        },
        {
          "number": 461,
          "title": "Hamming Distance",
          "slug": "hamming-distance",
          "difficulty": "Easy",
          "focus": "Compose a per-position difference indicator with population count for nonnegative bounded values. Distinguish bit differences from numerical distance.",
          "hint": "Which truth-table operation gives 1 exactly where two input bits differ? What quantity should then be counted?",
          "transfer": "Compare 7 and 8, whose numerical difference is one but whose bit patterns differ at four positions. For signed input, require a declared width before counting representation differences."
        }
      ]
    },
    {
      "id": "bitwise-partition",
      "title": "Optional bitwise transfer · separate two survivors",
      "optional": true,
      "introduction": "Return after the two-singleton proof and complete program in section 13. This official statement was inspected on 11 September 2026; it asks for two values in any order, with a linear-time and constant-extra-space target.",
      "problems": [
        {
          "number": 260,
          "title": "Single Number III",
          "slug": "single-number-iii",
          "difficulty": "Medium",
          "focus": "Find one bit that distinguishes the two surviving values, then prove equal pairs stay in the same partition.",
          "prerequisite": "XOR cancellation, lowest-set-bit isolation, and the two-pass reusable-input contract from section 13.",
          "hint": "The total XOR does not directly give either singleton. What does a 1 in that total reveal about those two values at the corresponding position?",
          "transfer": "Use a zero singleton, negative inputs and both output orders. Explain why a nonzero total alone is not a full promise check and why a one-shot iterator cannot be silently traversed twice."
        }
      ]
    },
    {
      "id": "stretch",
      "title": "Optional stretch · return with prefix and window reasoning",
      "optional": true,
      "introduction": "These connect arrays and maps to later algorithmic patterns. Read the extra prerequisite first; finishing them is not required to move to Linked Lists, Stacks & Queues.",
      "problems": [
        {
          "number": 238,
          "title": "Product of Array Except Self",
          "slug": "product-of-array-except-self",
          "difficulty": "Medium",
          "focus": "Combine summaries of the part before and after an index while respecting the no-division constraint.",
          "prerequisite": "Prefix/suffix accumulation and loop-invariant reasoning; these are an extension beyond this storage lesson.",
          "hint": "Split the contribution to one output into values on its left and values on its right. What neutral value should describe an empty side?",
          "transfer": "Test one zero, two zeroes and negative values. Separate output storage from auxiliary working space when analyzing the follow-up."
        },
        {
          "number": 3,
          "title": "Longest Substring Without Repeating Characters",
          "slug": "longest-substring-without-repeating-characters",
          "difficulty": "Medium",
          "focus": "Keep a contiguous region valid while its endpoints move; repeated symbols must change the region, not merely a global count.",
          "prerequisite": "Two-pointer/sliding-window invariants. Learn that pattern before using this as a timed exercise.",
          "hint": "When the new character repeats inside the current region, what must move to restore validity? Can moving an endpoint backward ever help?",
          "transfer": "Use a repeat whose previous occurrence lies outside the current region. Explain why a subsequence answer or a fixed-size window solves a different problem."
        },
        {
          "number": 560,
          "title": "Subarray Sum Equals K",
          "slug": "subarray-sum-equals-k",
          "difficulty": "Medium",
          "focus": "Use a map to count compatible earlier states rather than only remember whether a state occurred.",
          "prerequisite": "Prefix sums and the difference-of-prefixes identity. Negative values rule out a simple monotone shrinking-window assumption.",
          "hint": "Write a subarray sum as the current prefix total minus an earlier prefix total. What earlier total would make that difference equal the target?",
          "transfer": "Try zero-valued elements and a zero target. Explain why a set loses multiplicity and why the empty prefix needs a role in the count."
        }
      ]
    }
  ],
  "readiness": [
    "Choose between an index, set, count map and immutable composite key from the requested output, without a pattern label.",
    "State expected hash-operation assumptions; include text/key processing and output costs instead of calling every dictionary program O(n).",
    "Explain a counterexample involving duplicates, index zero, absence or normalization and repair the contract or implementation.",
    "Choose membership, exact counts or parity from the contract; justify width/sign assumptions, input promises and working-word versus bit costs."
  ],
  "localBridge": "LeetCode’s constrained inputs do not assess the full lesson. Also revisit the dynamic-array movement trace, forced hash collisions and Unicode normalization experiment; those establish representation and correctness limits that these problem statements only partly exercise. Also solve the changed bitmap report and signed-word task locally; judge success on a promised input is not a validation proof for arbitrary data."
};
