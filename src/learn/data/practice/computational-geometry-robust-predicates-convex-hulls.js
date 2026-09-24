export default {
  topicId: 'computational-geometry-robust-predicates-convex-hulls',
  verifiedOn: '10 September 2026',
  introduction: 'Use these official tasks to reconstruct predicates, adapt boundaries and transfer geometric invariants. Platform tolerance on a final area is different from permission to guess orientation signs. The local intersection and numerical-failure exercises remain essential.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · choose the decision and its degeneracies',
    introduction: 'Start with an exact small calculation and name what zero means before implementing.',
    problems: [{
      number: 1037,
      title: 'Valid Boomerang',
      slug: 'valid-boomerang',
      difficulty: 'Easy',
      focus: 'Distinguish three noncollinear locations from repeated or collinear points.',
      hint: 'Which one signed quantity vanishes both for a collapsed triangle and for a straight one?',
      transfer: 'Reverse two points, translate all points, and include repeated coordinates. Explain why slopes are unnecessary and why area uses the magnitude rather than the sign.'
    }, {
      number: 1232,
      title: 'Check If It Is a Straight Line',
      slug: 'check-if-it-is-a-straight-line',
      difficulty: 'Easy',
      focus: 'Use one consistent baseline to test an entire point set. Official inputs contain no duplicate locations.',
      hint: 'What happens on a vertical baseline, and which two distinct points determine the line?',
      transfer: 'Allow duplicate records and an empty set: define your API, find a distinct baseline if one exists, and explain the all-identical case. Test large coordinates without narrowing intermediate products.'
    }, {
      number: 812,
      title: 'Largest Triangle Area',
      slug: 'largest-triangle-area',
      difficulty: 'Easy',
      focus: 'Combine exact doubled-area comparisons with exhaustive triples under the small official input bound.',
      hint: 'Can candidates be compared before introducing the final division by two?',
      transfer: 'Report an original triple as a witness and specify ties. Explain why a final numeric acceptance tolerance does not make an epsilon-based collinearity predicate valid. A faster hull-based search needs its own argument.'
    }, {
      number: 836,
      title: 'Rectangle Overlap',
      slug: 'rectangle-overlap',
      difficulty: 'Easy',
      focus: 'The official task requires positive intersection area, so touching edges or corners do not count.',
      hint: 'What overlap length is required independently on each axis?',
      transfer: 'Change the question to closed-set contact and explain which strict comparisons become inclusive. Then exhibit intersecting bounding boxes for disjoint diagonal segments to show why a box test is only a rejection filter there.'
    }]
  }, {
    id: 'core',
    title: 'Core · preserve the requested boundary',
    introduction: 'The platform calls the hull task Hard, but this lesson supplies its mechanism. Adapt the output contract deliberately.',
    problems: [{
      number: 587,
      title: 'Erect the Fence',
      slug: 'erect-the-fence',
      difficulty: 'Hard',
      focus: 'Return every input location on the perimeter, including collinear edge points. The statement permits any output order.',
      hint: 'Does your stack remove zero turns? What happens when the entire input is on a line?',
      transfer: 'Change the API to counterclockwise corners only, with no repeated start. Explain the policy change, deduplicate outputs, and verify a small result against independent supporting-line tests rather than only the same hull code.'
    }]
  }, {
    id: 'stretch',
    optional: true,
    title: 'Optional transfer · exact geometric keys',
    introduction: 'Use the optional normalized-direction section plus the earlier hashing and Euclid ideas; this is not a gate for continuing the module.',
    problems: [{
      number: 149,
      title: 'Max Points on a Line',
      slug: 'max-points-on-a-line',
      difficulty: 'Hard',
      prerequisite: 'Canonical integer direction pairs using gcd and a sign convention; hash-map grouping. These are developed in this lesson’s optional section.',
      focus: 'Group directions around an anchor while preserving vertical and opposite-direction equivalence. Official positions are unique.',
      hint: 'How can proportional integer displacement pairs share a key without computing a floating slope?',
      transfer: 'Add duplicate records and distinguish their multiplicity from a zero direction. Compare tiny outputs with all-pair line enumeration, then explain arithmetic and expected hash-operation costs.'
    }]
  }],
  readiness: ['Can derive and test orientation signs under reversal, translation and degeneracy.', 'Can explain why closed contact, proper crossing and positive-area overlap require different predicates.', 'Can justify stack removals and adapt all-collinear/boundary output without losing or duplicating locations.', 'Can distinguish exact intended inputs, stored binary values, construction error and measurement uncertainty.', 'Can reconstruct an approach with hints closed, then handle a changed contract or an unfamiliar mixture of earlier structures.'],
  localBridge: 'Keep the independent polygon-ray, rational intersection and floating-input counterexamples: the six external statements do not establish the whole geometric contract. These tasks build transfer, not a guarantee for every interview or geometric algorithm.'
};
