# Conditional Random Fields — concept intuition author review

26 September 2026. Entire production JSX, deeper branches, all five changed practice tasks, library bridge explanation and existing figure/model interfaces read. No topic-body generator found in the current scripts; production JSX remains the content checkpoint. The prior programs, treebank extraction, training outputs and numerical engine are unchanged.

## Whole-lesson concept map

| Location / conceptual transition | Assessment and action |
| --- | --- |
| Opening → dependent labels → conditional model | Person/organization example, HMM comparison and direct definition supply task-level intuition. Retained. |
| §1 BIO, feature scopes and available information | Word/factor/label illustration and gold-label warning already make these distinctions explicit. Retained. |
| §2 log scores → factors → whole-path normalization | Exact four-path ledger explains every product, partition and probability. Shared-offset/uniform special cases and zero-interaction factorization retained. |
| §3 enumeration → forward sums → Viterbi maxima/backpointers | Live trellis, exact incoming masses and contrasting prefix quantities already explain reuse and what is discarded. Retained. |
| §3 backward messages → node and edge marginals | Prefix/suffix illustration, 3·9/30 and edge .8 calculation are sufficient. Retained. |
| §3 best path versus marginal modes | Four-path counterexample plus distinct decision losses retained. |
| §4 observed minus expected counts → regularization → convexity | Count figure, signed update and covariance-Hessian explanation retain intermediate reasoning and scope. No redundant new lab. |
| §4 log-space implementation | Existing stable recurrence and index orientation explain the code. Retained complete executable. |
| §5 local normalization / label bias | Existing movable branch lab and cancellation example are clear. Retained with information-structure limitations. |
| §6 finite preferences versus forbidden transitions | Existing BIO figure and input-20 versus penalty-5 counterexample suffice. Retained all constraint/API distinctions. |
| §7 real text, data roles, feature fitting, gradients and metrics | Real aligned errors, full NumPy/SciPy implementation, sentence split and model-selection record retained. |
| §7 CRFsuite score reconstruction and objective conversion | Existing controlled inference bridge and penalty conversion retained. Corrected a stale reference annotation that incorrectly said the optional package was never executed; it now distinguishes the executed small bridge from the NumPy/SciPy treebank experiment. |
| §8 neural encoder, gradients and padding | Existing shape/flow illustration and sequence-mask explanations sufficient. Retained. |
| §8 second-order state and K³ work | Added shared-label pair-window diagram with legal extensions and a concrete inconsistent extension, explaining why K² pair states have K, not K², possible successors. |
| §9 non-chain structures and semi-Markov spans | Existing factor-scope and segment-length explanations connect their costs to the preceding graphical model and recurrence. Retained. |
| §9 latent/partial labels → compatible sums; unlabeled likelihood | Added live partial-label likelihood lab. Learner changes observed label information and the A→B factor, sees included paths and both likelihood totals immediately. Fully unlabeled ordinary likelihood remains one while model paths change. |
| §9 backward conditional sampling | Added exact two-branch calculation recovering all four original joint probabilities, plus an independent-marginal sampling counterexample. |
| §9 alternative objectives/families | Structured-margin versus likelihood and Bayesian weight integration remain explicitly scoped extensions. Retained. |
| §10 changed practice/next topic | All tasks and solutions preserved; removed the obsolete “predict before fitting” requirement and updated reporting wording to actual saved results. No prediction gate added. |

## Visual and interaction contracts

- `CrfPairMemoryFigure`: label pair A,B and next-label alternatives A/B; repeated B is visible by identity and color. HTML tokens/branches wrap. Dense state counting K²×K is illustrated without inventing runtime benchmarks.
- `CrfPartialLabelLab`: reuses `chainDistribution(initialFactors)` with only AB factor varied. Factor is a real range input from .125 to 16 in .125 steps; label-information select chooses AB, A?, or ??. Four labeled path strips distinguish compatibility through text and amber edges. Readout is exact compatible mass / partition and negative log. Reset restores factor 4/A?. No expensive fit or sampling; four paths recompute on every event. Controls have explicit labels and focus styling; browser interaction still needs root verification.
- `CrfBackwardSamplingFigure`: final probabilities [4/30,26/30]; first-label conditional probabilities [3/4,1/4] or [12/13,1/13]. Products exactly recover [3,24,1,2]/30. Independent marginal sampling would give AB=.78, not .8. Branches describe probabilities, not empirical sample frequencies.

## Research actually read

Read the [Sutton–McCallum tutorial](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf) sections describing segment-level feature scopes (printed page 307), sample-based conditional expectations and graphical inference (around printed pages 310 and 320–323). Used the feature-scope/inference relationship to assess the deeper branch. The new pair, partial-label and exact chain-sampling examples are original derivations from the lesson's fixed factors; no external video or new native package execution is claimed. Existing primary teaching references remain annotated in the reader.

## Actual author verification

Changed JSX parsed; all 128 factor slider positions checked against independent masses [3,6f,1,2] and all three observation sets; all four backward-sampling path probabilities recovered; pair-overlap transitions counted for multiple K. Original inference engine is included in source-bound hashes because the new lab imports it. Full/native historical training and browser checks were not repeated. Browser/mobile and independent review remain pending root.
