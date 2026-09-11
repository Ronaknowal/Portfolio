# Decision Theory, Risk & Cost-Sensitive Decisions — author verification

Stable ID: `decision-theory-risk-cost-sensitive-decisions`; Mathematics 53. Root approved the [full design](DECISION-THEORY-LESSON-DESIGN.md) and centrally registered the completed lesson. The exact original planned scope is retained in [the original-plan archive](evidence/decision-theory-original-plan.json). There was no legacy published body or original program. Title, the two prerequisites and module order remain unchanged; the next module topic is Real Analysis, Sequences & Modes of Convergence.

This record covers author implementation and checks. Independent mathematical/source review, shared production integration and user acceptance are separate. Final source identities and copied results are in [the author packet](evidence/decision-theory-author-review.json); it is the authority for timestamps and final source hashes. No full application build was run by this author.

## What was implemented

The complete mechanism-first course has 13 sections, eight investigations with different visual encodings, two inline figures, 11 standalone standard-library Python programs and 14 checkpoint/practice groups with hidden hints and explained answers. It teaches full loss tables and action regions; conditional/procedure/Bayes/empirical risk; decision-dependent point summaries; fallback and capacity; calibrated coarse scores, proper scoring and held-out threshold selection; information value and timing; causal-action qualifications; ambiguity and regret; a certified randomized finite minimax rule; declared utility and atom-aware CVaR; and a complete contingent inspection/allocation capstone. Optional sections include a finite admissibility proof, shared-parameter predictive loss, information simulation ordering, a finite LP formulation and the hinge derivation.

The implemented-disposition section in the design records why each representation was chosen. All plotted quantities come from the stated functions or finite model. There are no illustrative performance benchmarks, fabricated measurements or external cost valuations. The chosen losses and fault probabilities are labelled synthetic. A finite policy being optimal under them does not establish a real-world probability model, causal effect or stakeholder valuation.

## Mathematical and executable evidence

Commands:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/generate-decision-theory-examples.py
node scripts/format-decision-theory-source.cjs
node scripts/verify-decision-theory-models.mjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-decision-theory-native.py
```

The generator formats each complete program with Black and captures actual stdout. All 11 programs use Python 3's standard library only. The native review executes the exact strings delivered to the browser and compares their full stdout. It then calls the actual displayed helpers on 46 changed cases; it does not merely execute a separate reference implementation.

The complementary model oracles use exact Fraction arithmetic, every feasible small policy/subset, independent 64-world enumeration, the independent hinge representation and 60-digit mpmath utility calculations. Final cases include 48 binary loss tables/probabilities with exact action sets; 21 prior/signal rule comparisons with posterior and tie checks; 55 provisioning states; 24 capacity subset comparisons; 112 information models evaluated over every signal rule; 200 atom/tail combinations including zero mass, duplicates, negative losses and exact cumulative boundaries; 24 high-precision utility states; 126 contingent choices each independently summed over 64 worlds; and 7 frozen-threshold cases. The model script additionally rejects 22 malformed/domain/arithmetic inputs, checks exact co-optimal test/fallback/rule states, validates all six KaTeX formulas and parses owned JSX.

This evidence verifies specific contracts rather than using counts as a proxy for quality. In particular, the capstone's item 4/item 5 tie and the no-test tie at price 10.4 are mathematically preserved; impossible signals retain no posterior; a calibrated coarse score is distinguished from full-information optimality; capacity is enforced in each information branch; and a partial atom is used to fill exactly the requested upper tail. No author test is described as an independent human/agent review of the whole course.

## Browser and ordinary-reading evidence

Command:

```text
node scripts/review-decision-theory-lesson.cjs
node scripts/review-decision-theory-final-boundaries.cjs
```

The script uses actual Edge/Playwright against the registered route on the shared development server, with public site fonts available. It exercises the intended Space Grotesk and JetBrains Mono fonts at 1440, 390 and 320 pixels. At each width it checks 27 changed/edge states, 34 recorded keyboard/control operations, 13 actual section-anchor arrivals, all 11 exact displayed code/output blocks and their preceding visible questions, all 14 hint/solution groups, six displayed equations, SVG text bounds, document overflow and the 11 annotated source links. Optional derivations and their code are opened before checking the complete lesson. The native programs run outside the browser; the controls are explicit finite models, not a hidden Python interpreter.

The final screenshot set includes ordinary section starts, both inline figures, each distinct investigation, equation detail, changed practice, complete code/output and alternate resources. The final author packet enumerates the screenshots actually opened and their hashes. An automated screenshot being saved is not counted as an opened visual review. Standard code blocks can scroll horizontally on phones; the article and compact diagrams do not require horizontal page scrolling.

The full three-width run passed at 07:02:26 UTC on 11 September 2026. The final model/native checks passed at 07:02:58/07:03:04 UTC. Afterward, the only production change was the fallback strip's explicit sampling caption. The focused three-width check passed at 07:08:37 UTC against all seven final source hashes, verifying that caption, the CVaR endpoint qualification, the exact rule/capstone ties, atom-tail readout, keyboard disclosure and normal rendering of all eight investigations. It also closes the source identity after the own-array validation amendment. The packet preserves earlier model hashes and distinguishes that caption-only difference instead of attributing older tests to silently changed bytes. Twenty-four final captures were actually opened and read; all saved captures are not claimed as individually inspected.

Before freeze, actual phone screenshots led to enlarged graph labels and four wrapped equations. Root identified the initially mismatched heading IDs, which were corrected locally and then verified by real arrival. Exact-decimal arithmetic repaired lost ties and atom cutoffs before the final control review. A first browser check failed because its harness expected `pre` elements; adapting to the shared renderer's text nodes restored actual content equality checking. These failures and their resolution are recorded openly; no initial failed run is treated as a final pass.

Root's independent source read also prompted three narrow corrections: the general CVaR attained-minimum statement is restricted to 0 < alpha < 1, with the alpha=0 mean/infimum extension explained separately; array validation requires own entries rather than inherited prototype entries; and the fallback strip is labelled as 101 probability samples whose thin boundaries may fall between samples. The selected-point risks still use direct exact finite comparisons. A separate author arithmetic check rejects a positive-support utility result whose certainty equivalent underflows the display range. Both malformed cases have actual model regressions; ordinary native outputs are unchanged. These author dispositions do not replace root's separate final independent-review record.

## Resources, boundaries and handoff

The design's research ledger records the exact sections inspected. Primary/authoritative sources include Shalizi and Fithian on risk, Gneiting–Raftery on proper/quantile scores, the official scikit-learn threshold/calibration documentation, MIT decision-tree and utility teaching, and Rockafellar–Uryasev's general-distribution CVaR treatment. Source annotations distinguish written pages/transcripts actually read from video metadata inspected without full playback. The complete tree-construction transcript and selected utility slides were read; the video links supplement the self-contained lesson. No unsupported claim of watching an entire playlist is made.

The two scoped destination discoveries remain in the MDP and calibration topic notes, with their reason and ownership. Full games, infinite-state minimax/complete-class results, sequential control, preference elicitation, causal identification and population validation are deliberately outside this finite first course. Those are qualified extensions, not silently solved by the finite examples. Root owns final independent review and integrated publication evidence.
