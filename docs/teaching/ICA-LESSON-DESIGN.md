# Independent Component Analysis — content design and continuation

Updated 14 September 2026 (content 12 September; implementation and review disposition 14 September). Stable ID: **independent-component-analysis-ica**. Display title retained: **Independent Component Analysis (ICA)**. Classical ML → Unsupervised Learning → position 18, after t-SNE, UMAP & Manifold Learning and before Non-Negative Matrix Factorization (NMF). Preserve ID, module route and progress.

## Current source and phase

Both phases are complete. The table below is the live state; the two sections that follow it describe the content phase as it stood at its close on 12 September 2026, and the [phase-two section](#phase-two-implementation--14-september-2026) and [review disposition](#disposition-of-the-independent-review--14-september-2026) describe the implementation.

| State | Actual evidence or boundary |
| --- | --- |
| Content/research | **Complete**, 12 September 2026. Packet linked below, unchanged by phase two. |
| Visual/lab implementation | **Complete**, 14 September 2026. Six figures and two investigations rendered; see the phase-two section. |
| Computational checks | **Complete.** [Model checks](evidence/ica-models.json), [executed programs](evidence/ica-native.json), [recomputed recording data](evidence/ica-data.json). |
| Browser/accessibility/performance | **Run**, Chromium only, at 1366/768/390/320 px plus a 200 % zoom equivalent: [browser evidence](evidence/ica-browser.json). No screen-reader session and no second engine. |
| Independent phase-two review | **Complete**, 14 September 2026: [ICA-INDEPENDENT-REVIEW.md](ICA-INDEPENDENT-REVIEW.md). Every finding is dispositioned in the last section of this record. |
| Blueprint registration | **Registered** by the integration owner in `src/learn/data/curriculum/blueprints/index.js`. |
| Phase ledger | Owned by the integration owner; not closed by the implementer. |
| User acceptance | Not recorded. |
| Next action | Integration owner closes the phase ledger with the final source hashes and reconciles the handoff. |

### At content-phase close, 12 September 2026

Requested mode was **content first, research/write only**. Author: scoped ICA content agent; root owned shared handoff, delivery ledger and integration. Stages 1–2 were complete: actual learner prose, hand mechanisms, teaching code/results, practice/hints/solutions, annotated resources, real offline input, visual/investigation contracts and author content checks. Implementation was **not started** at that point, and no production lesson, registry, component, manifest, navigation, blueprint or shared ledger had been edited by the ICA content author. Everything in the two sections below is that content-phase record, retained unchanged.

The authoritative pending packet is:

- [Complete manuscript](drafts/independent-component-analysis-ica/lesson.md).
- [Visual and investigation specifications](drafts/independent-component-analysis-ica/visual-specifications.md).
- [Offline real data](drafts/independent-component-analysis-ica/r01-first20s.csv).
- [Data provenance, calibration and attribution](drafts/independent-component-analysis-ica/data-provenance.md).
- [Necessary author calculation/extraction input](drafts/independent-component-analysis-ica/author-calculations.py).
- [ICA topic-note disposition](topic-notes/independent-component-analysis-ica.md).

Markdown convention is inline dollar-delimited math and display double-dollar math. Convert these to the project's Math/MathBlock components in phase two. Inline-figure instructions are placement annotations with complete learner explanations surrounding them; they are not implemented figures or unfinished requests to author the lesson.

## Baseline and what is conserved

Read the entire original src/learn/data/topics/independent-component-analysis-ica.jsx before drafting, including all original examples, failure cases, sources and exercises. Baseline commit **8c5da59f18516be77c29d5aeeafca3decca4f738**; the original Git blob and current untouched worktree file have identical SHA-256:

**03f93fa4ec9e005b30b80b3412ed6f682e749655648426b3348161c8439038d1**.

Git at that commit supplies the immutable original. No redundant source archive was created. The existing JSX remains the published source until authorized phase two. It is not reviewed or corrected merely because this manuscript exists.

| Existing treatment | Decision and learner benefit |
| --- | --- |
| Cocktail-party model, mixing/unmixing, independence vs covariance | Retain the problem, replace lengthy historical lead-in with an actual two-sensor sample; state instantaneous/noiseless conditions and distinguish source vs time independence. |
| PCA comparison, whitening, Gaussian nonidentifiability | Retain/deepen with one exact source-square → observed cloud → whitened diamond → separation chain, explicit probabilities and orthogonal-search derivation. Repair “PCA and ICA coincide on Gaussian data”: the Gaussian subspace is nonidentifiable. |
| Kurtosis/negentropy, CLT intuition | Retain with exact fourth-moment expansion and limits. Replace unsourced Laplace curves with exact population quantities. Add non-Gaussian zero-kurtosis counterexample. |
| FastICA derivation and from-scratch code | Retain depth with a hand-averaged update, approximate-Newton derivation, correct within-iteration deflation, symmetric orthogonalization and sign-aware stop test. Repair code that projected previous components out only after convergence. |
| NumPy sine/square/Laplace example and near-perfect recorded outputs | Replace with a full four-state probability fixture, avoiding claims that two deterministic time functions are mutually independent random sources. Preserve complete algorithm, shape, correlation and reconstruction reasoning. Actual real recording is added separately. |
| scikit-learn API, transforms, parameter alternatives | Retain current parameter/attribute distinctions and complete offline library example; correct truncated-PCA-subspace interpretation. Fixed real comparison uses train-only decomposition, reference-assisted development selection and held-out diagnostic. |
| Picard “production” code | Original snippet was wholly commented out, required an unspecified MNE raw object, and contained unsupported universal speed/preference claims and an incorrect “ortho=False solves overcomplete ICA” claim. Preserve the valid optimizer alternative in section 8 and annotated primary reference; do not promote incomplete specialist sample code to a second core workflow. Current complete workflows cover one transparent implementation and one applied library fit. No Picard benchmark/API execution is claimed. |
| Sign/scale/permutation, mixing matrix, artifact exclusion | Retain/deepen in section 7 with exact contribution subtraction and scale-invariant energy expression. Correct the equivalence of inverse-row and mixing-column norms and guaranteed one-component blink interpretations. |
| Method comparison, complexity, samples, noisy/temporal extensions | Retain in purposeful deeper branches. Remove unsupported universal sample counts, SNR thresholds, timing ratios and guaranteed convergence. Explicitly distinguish overcomplete, convolutive and nonlinear models. |
| Historical source list | Preserve the useful estimation lineage without first-ever/every-library claims. Current canonical tutorial/book and accessible Stanford notes/transcript offer relevant routes. Picard is advanced reading, not a fabricated timing result. |
| Exercises | Retain conceptual diagnosis and algebra goals, repair false equal-kurtosis failure premise, add changed numeric recipes, exact 1.875 transfer, reference-selection leakage and wanted-signal removal practice. |

Title assessed at planning, after real-data results and at final read. ICA remains the correct concise umbrella; the real ECG example illustrates a diagnostic decision and does not justify adding a clinical promise to the title.

## Learning contract and scoped prerequisite review

Target reader: a learner comfortable with the preceding Classical ML representations, who can multiply a 2×2 matrix and read basic Python. Core finish line: explain why whitened coordinates can remain dependent, calculate a changed non-Gaussian projection statistic, follow a constrained fixed-point update, run an offline decomposition and distinguish fitting, selection and evaluation.

Inspected the current PCA lesson's introduction, projection, covariance and whitening-related source context: its implemented body supplies dot-product direction, scores, eigenvalue scaling and reconstruction; no need to duplicate its full derivation. Inspected Probability Distributions & Bayes' Theorem's event independence and moments passages; supply a local covariance/product-factorization bridge because a prerequisite title alone does not carry that entire argument. Explicit review links use the declared /learn/path/full-curriculum/<topicId> route and retain module query parameters. The earlier vector/matrix and NumPy lessons remain optional foundational review, not inserted topics or changes to module order.

Deeper branches teach component contribution/removal, continuous entropy/likelihood and computation/extensions. A reader can finish the first-pass route without clinical physiology, a full information-theory course, MNE installation or a nonlinear separation tutorial. The readiness text distinguishes first-pass outcomes from optional component-removal readiness.

| Hurdle/outcome | Local bridge and example | Support/assessment | Depth |
| --- | --- | --- | --- |
| A sensor is a mixture, not a source | Row/column meanings, one sample, exact inverse | M1 weighted contributions; practice1 new matrix | core |
| Covariance misses dependence | Event product probabilities, u and u² | D1 conditional support; whitened-diamond probability mismatch | core |
| Whitening is not separation | Eigenvectors(±diagonal), eigenvalues9,1, divide standard deviations | W1 four corresponding states; R1 rotation action | core |
| Non-Gaussianity supplies additional structure | Fourth-moment expansion, equal Laplace kurtoses, Gaussian rotation, zero-kurtosis counterexample | R1 binary/Laplace/Gaussian contrast and null; practice2 | core |
| FastICA update and constraints | Unit variance, gradient stationarity, Newton approximation, projection removal | F1 averaged vector trace; complete NumPy; practice3 | core |
| Assess a real result fairly | Sample/channel shapes, held-out time boundaries, explicit correlation diagnostic | P1 split dependencies; E1 raw/PCA/ICA marks and traces; practice4,6 | core |
| Ambiguities and exclusion | Mixing column vs inverse row, sum of contributions, scale cancellation | C1 editable contributions; practice1,5 | deeper |
| Objective/algorithm and extension distinctions | Entropy/volume bridges, MI sign, conditional equivalence, iid-likelihood condition | Section8 derivations and model-change table | deeper |

No lab or length quota was used. M1, D1, W1, F1 and P1 serve immediately visible causal relationships; E1 joins measured data to the application diagnostic. R1 and C1 have distinct investigations: changing the distribution-sensitive direction and changing a reconstructed contribution. Real-data interpretation is a worked program/static result, not a fake interactive fit. Clinical source labels are not a learning target.

## Canonical coverage check

Canonical reference: Hyvärinen, Karhunen & Oja (2001), author-hosted book. Read its contents for the basic ICA and extension/application sequence, and selected chapter7.4–7.5 and8.2 passages. The 2000 author tutorial supplies a shorter full model/estimator connection. This is a scoped comparison, not a claim to have read all503 book pages.

| Canonical material | Disposition |
| --- | --- |
| Ch6–7: whitening, generative model, ambiguities, Gaussian impossibility | Core, with exact examples and deeper contribution interpretation. |
| Ch8: kurtosis extrema, negentropy, fast fixed-point, several components, projection pursuit | Core derivation/calculation; projection pursuit connection expressed as direction search. Equal-kurtosis canonical example explicitly repairs original false premise. |
| Ch9–10: likelihood, Infomax, mutual information and relationships | Deeper section8, with volume correction and correct minimization sign; independent-observation likelihood condition stated separately from temporal marginal fitting. |
| Ch11–12: tensorial cumulant methods, JADE/FOBI, nonlinear decorrelation algorithms | Not implemented as additional algorithms: they require cumulant-tensor/joint-diagonalization foundations beyond the chosen FastICA outcomes. Book provides the accessible deeper route; no claim of comprehensive ICA algorithm coverage. |
| Ch13–14: filtering, PCA subspace, component counts, algorithm choice, robustness | Core rank/subspace and real-protocol decisions plus deeper computation. No universal threshold or speed claim; filter pipeline details routed to their actual planned owner. |
| Ch15–17: noisy, overcomplete and nonlinear ICA | Model-change table explains which condition breaks and the extra structure needed. Full algorithms require a separate specialist treatment; neither PCA nor an ordinary VAE automatically solves these problems. |
| Ch18–19: temporal structure, lagged covariance, convolutive mixtures | Brief equations and distinction here; detailed audio operation routed to planned source-separation owner. |
| Ch20–23: additional structure, image features, brain imaging, telecommunications | Component contribution and EEG artifact interpretation retained because they clarify reconstruction. Detailed image basis learning, CDMA and complex-valued ICA not added: new application mechanics would interrupt the scoped route. The canonical book remains the appropriate follow-on. |

## Scoped ownership discoveries and dispositions

Topic inventory --topic independent-component-analysis-ica --work content was run before writing. It reported the existing publication, no complete current content, no destination note and no unresolved inbox item relevant to ICA. The resolved bit-manipulation inbox history was read and not reopened.

The following were reconsidered during research and writing:

- **Whitening not separation, same-kurtosis contrast, one-to-one matching, source vs time independence:** best taught here; included in complete manuscript and exact calculations. Own [ICA note](topic-notes/independent-component-analysis-ica.md) preserves the phase-two repair requirement without falsely marking production implemented.
- **NMF nonnegativity vs independence and semantic claims:** local forward bridge stays short. Root read NMF source and owns the saved [non-negative-matrix-factorization-nmf note](topic-notes/non-negative-matrix-factorization-nmf.md). No NMF source rewrite.
- **Common filtering and acquisition/model invariance, task-preserving exclusion, leakage:** inspected the substantive planned neural preprocessing blueprint. Root saved [neural-preprocessing-artifact-rejection-and-leakage-safe-pipelines](topic-notes/neural-preprocessing-artifact-rejection-and-leakage-safe-pipelines.md), with exact shared-filter identity, conditions and planned acausal-leakage practice. This lesson only states the local model/application bridge.
- **Instantaneous demonstration vs delayed/reverberant audio:** inventory confirms [source-separation-audio-denoising-demucs-band-split-rnn](topic-notes/source-separation-audio-denoising-demucs-band-split-rnn.md) is planned, with no bespoke existing body. Root owns that note. Equation x_t=ΣA_l s_(t−l) and the surviving delayed term are included here; modern audio-model implementation and evaluation are deferred to that owner.

## Real-data decision and provenance

Data candidate accepted: PhysioNet Abdominal and Direct Fetal ECG Database v1.0.0, original r01.edf. It contains actual simultaneous mixtures plus an independent acquisition channel usable for an explicit diagnostic; its small deterministic extract is more application-relevant than claiming a generated sine/square exercise is real source separation.

Choose record r01 and its first20 seconds before calculating results, retain all five channels without further filtering. Four abdominal channels feed ICA; the direct channel stays outside fitting. Train0–12 s; development12–16 s selects one coordinate within each method; test16–20 s reports a fixed absolute waveform-correlation diagnostic. The result intentionally retains the PCA advantage; no seed, interval or component-count hunt was made to manufacture ICA superiority.

The [provenance file](drafts/independent-component-analysis-ica/data-provenance.md) owns source URL, original/publication hash comparison, exact raw header, ADC calibration, extraction steps, license, author attribution and scientific limits. Offline input is390,723 bytes, 20,000×5 integer values plus header. Original EDF3,061,792 bytes, source SHA2567549bbd378ea23851c20c0b7924f0a1f9fd909a3a3683c2334144a4c156dcb62. Extract SHA2567c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc.

Only one short within-recording demonstration is claimed. The data were already filtered by the provider; temporal dependence means row count is not independent sample count. The reference waveform is a diagnostic comparison, not direct evidence that a component represents one physiological cause. No clinical advice or clinical performance claim is made.

## Claim/source ledger and actual review

Sources accessed12 September2026. All links below are primary/canonical. Equation calculations and teaching fixtures are independently derived here; prose is original, not source excerpts.

| Claim or purpose | Source and locator | What was actually read / uncertainty |
| --- | --- | --- |
| ICA model/ambiguities, non-Gaussian search, FastICA formula/deflation | [Hyvärinen & Oja2000 author tutorial](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf), §§1–6, especially2.1–2.3 and6.1–6.2/equations41–44 | Relevant substantive model and algorithm passages read, including approximation and per-iteration orthogonalization. Not a universal speed guarantee. |
| Canonical coverage and same-kurtosis repair | [HKO2001 author manuscript](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf), contents chapters6–23;7.4–7.5;8.2.1/equations8.8–8.11 | Contents and the cited whitening/Gaussian/kurtosis passages read. Book explicitly illustrates equal source kurtoses with isolated axis maxima. Full text and all experiments were not reviewed. |
| Continuous likelihood, determinant and observation independence | [Stanford CS229 notes](https://cs229.stanford.edu/notes2021fall/cs229-notes11.pdf), pp1–6, §§1–3 | Model/ambiguities, Gaussian covariance argument, density transformation and likelihood/temporal-observation remark read. Correct local full-dataset determinant factor retained. |
| Alternate video learning route | [Stanford Lecture15](https://see.stanford.edu/Course/CS229/45), [substantive transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture15.html), [YouTube](https://www.youtube.com/watch?v=QGd06MTRMHs) | Official listing/bookmarks verified; ICA transcript from introduction through CDF/likelihood and artifact application read. Video/audio not watched. Transcript contains formula errors and broad simplifications; lesson uses current primary model conditions instead. Bookmarks ICA39:49 and algorithm47:41 are listing-derived, not watched timestamps. |
| FastICA current defaults, shapes, inverse/whitening semantics | [scikit-learn FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html), Parameters/Attributes; [fastica function](https://scikit-learn.org/stable/modules/generated/fastica-function.html), whitening and source shapes | Substantive parameter/attribute sections read; local version1.9.1 confirmed. Unit-variance default changed in1.3; explicit settings used. No claim this snapshot remains newest. |
| Real-data acquisition, reference role, licensing | [PhysioNet v1.0.0](https://physionet.org/content/adfecgdb/1.0.0/), Data Description/Access/Files; [published checksums](https://physionet.org/content/adfecgdb/1.0.0/SHA256SUMS.txt); [ODC-By1.0](https://opendatacommons.org/licenses/by/1-0/) | Dataset description, acquisition metadata, file header, author citation and license attribution conditions read. Original hash agrees with source list. No clinical result inferred. |
| Inspect/exclude/reconstruct artifact workflow and common-filter conditions | [MNE artifact ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html), filtering/fitting, auxiliary-channel identification and exclusion | Relevant tutorial passages read, not executed. Current page1.13.2; its MNE API differs from sklearn. Detailed pipeline practice routed to neural owner. |
| Picard as optimizer alternative | [Ablin/Cardoso/Gramfort paper](https://arxiv.org/abs/1706.08171), abstract/author method description | Abstract and preconditioning/objective description checked. Benchmarks and full algorithm not reproduced; only that limited alternative-reading claim retained. |
| Delayed mixture and model boundaries | HKO2001 contents ch15–19; tutorial model/preprocessing passages | Canonical scope checked and local two-tap counterexample derived. No modern source-separation benchmark or nonlinear-identifiability theorem claimed. |

Alternative resources have no format quota. The Stanford transcript/bookmarks provide a substantively reviewed route to a video without pretending it was watched. Technical accuracy rests on the cited primary equations and local derivations, not transcript metadata.

## Actual author checks

This is elementary content substantiation, not phase-two acceptance. Reused policies after initial reading; no historical verifier campaign or recursive scratch scan.

- Read AGENTS, current handoff, complete current teaching standard, domain playbook's ML obligations, topic design brief, full relevant learning code contract and note policy. Inspected entire original ICA lesson and scoped prerequisite/future-owner sources.
- Ran ICA --work content inventory and exact future-owner inventories. Confirmed old source bytes equal the recorded baseline commit. No original copy needed.
- Independently expanded the fourth power, calculated covariance/eigenvectors/whitener, probability mismatch, inverse matrix, Gaussian rotational argument, zero-kurtosis sixth moment, MI sign and scale-cancelled contribution expression.
- Downloaded the openly licensed source after the environment's network restriction required approved escalation. Matched the published checksum, inspected the complete channel header and decoded the first four five-second EDF records into the exact offline CSV.
- Ran author-calculations.py --extract, then its retained final form. Environment: Python3.12.14, NumPy2.3.5, SciPy1.18.1, scikit-learn1.9.1. No environment package installation or mutation performed.
- Covariance [[5,4],[4,5]], whitened covariance I. Raw cubic update[-1.376,-1.368], sign-aligned next vector[.7091653,.70504225].
- R1 contrasts at0°,30°,45°,90°: binary[-2,-1.25,-1,-2]; Laplace[3,1.875,1.5,3]; Gaussian[0,0,0,0]. Analytic identity supplies continuous-angle behavior; rendered curves are deferred.
- C1 checked every keep-set on changed source(2,-3), zero-source(2,0), all-zero input, both selected-source columns and compensated scales+2/−2. Source(2,-3) gives none(0,0), keep1(4,2), keep2(−3,−6), both(1,−4). Scale compensation preserves every retained product exactly.
- Executed the displayed four-state NumPy program as one small author probe extracted directly from the manuscript: expected I/[1,1]/True output obtained; full absolute source correlation matrix was I to9 decimals. It is a complete small-example output, not formal multi-fixture production verification.
- Fixed real recording probe: channel development signed correlations[.043935,.101807,.201203,.079348], choose3, test absolute.119806; PCA[−.034740,.127795,.010770,.184580], choose4, test.450178; ICA[.079635,.169804,−.043955,−.116516], choose2, test.343966. ICA14 iterations; fitted IC variances[1,1,1,1]; full-rank test reconstruction MSE approximately1.0363e−28. No performance timing claim.
- Root read the complete manuscript and requested math-delimiter normalization, exact preceding display title, explicit setup/run commands and first-pass/deeper readiness consistency. Author applied these corrections before checkpoint. Source/wording corrections did not change the numerical model.
- Final author reread moved R1 after its kurtosis definition and Gaussian examples so the requested prediction uses already taught concepts. Added the eigenvalue-order bridge between the hand whitener and NumPy's ascending eigenbasis. Inspected every inline mathematical expression for preserved function arguments and grouping after normalization; repaired specification tuples, arithmetic grouping and Laplace contour notation.
- Author independently reread complete manuscript/specs for continuity and performed the checklist below. No human learner walkthrough was available.

## Learning-experience review — author's heuristic assessment

| Checklist item | Finding and disposition |
| --- | --- |
| Route | Route appears immediately after introduction; first pass1–6 then selected practice/readiness. Deeper7–8 labeled at entry. Corrected readiness so exclusion is optional-branch readiness rather than a hidden first-pass requirement. |
| Cautions | Model conditions have one early home; real diagnostic scope has one local home; source-label interpretation lives with exclusion. No printed program warnings/disclaimers. Removed original repeated Gaussian, sample-threshold and speed cautions. Different reconstruction/algebra and reference-purpose explanations are retained because they answer different questions. |
| Real question | Opens with mixed sensors, returns with a fixed real recording and an honest outcome favoring PCA on its declared diagnostic. Real input is small and offline; no invented sine/square “real” data. |
| Investigations | R1 angle and C1 amplitudes/keep-set/scaling have unset committed predictions tied to snapshots, new editable inputs, grading formulas, reset/invalidation and contrasting/null calculations. Specs are complete; actual browser behavior remains unknown. |
| Figures | M1 contribution diagram, D1 support restriction, W1 corresponding distributions, F1 vector operations, P1 split-information flow, E1 actual correlations/traces each has a distinct job. Exact axes and mobile recomposition specified. Perceptibility at rendered size has not been observed and must be verified later. |
| Connections | States why final algebraic inverse and learned direction agree; links whitened diamond and source-frame rotation; distinguishes marginal non-Gaussianity/continuous negentropy; canonical headline Gaussian and equal-kurtosis cases explicit. |
| Code | Two complete workflows with real supplied inputs; core algorithm occupies the teaching program rather than guards. Deflation projection is in its correct place. Minimal setup and exact runs given; no fake production Picard example. |
| Practice | New sensor recipe, new projection weight, broken implementation, changed reference-selection context, wanted-signal subtraction and modest study design. Hints precede full solutions; exact1.875 transfer result reproducible. |
| Screenshots | Explicitly deferred. No “passed visual review” claim or default-state-only evidence created. Phase-two plan requires informative comparison, prediction feedback and full-width/narrow figures. |

Inline visual reading pass was performed against the placement specifications: every major structure is visible where introduced without requiring a later lab state. The first-pass clinical dataset requires only reading a voltage trace and a correlation, with the specialist interpretation boundary stated before the result. No extra clinical workflow was added merely because the dataset permits one.

## Phase-two handoff

The next author reads all five packet files, this record and current delivery checkpoint. Implement the current semantic topic source, topic-owned figures/labs/models/examples, offline download and attribution, and an authored blueprint matching actual prerequisites. Keep existing ID and module order. R1/C1 exact mechanics do not justify arbitrary Python/browser execution or loading a full ECG archive.

Required future checks are specified in detail in the visual packet: native displayed outputs/current library semantics, independent mathematical fixtures, meaningful distribution/model nulls, reference-split conservation, source matching, figure/trace geometry, prediction-state binding, component-removal reconstruction, keyboard and narrow layouts, import/render recovery and production integration. Source-bound independent correctness and learning-experience review remains necessary then. A content checkpoint is readiness to implement, not a promise the finished page has passed those checks.

Removed the author's two exact disposable files, scratch/ica-content/normalize_math.py and scratch/ica-content/r01.edf, after preserving the CSV, calibration, original hash and reproducible source locator. No recursive scratch cleanup was performed. The pending packet, shared Python environment and other authors' files remain. Root will freeze the exact final packet hashes in the central phase ledger and own handoff reconciliation.

## Phase two: implementation — 14 September 2026

The content packet above was consumed and the lesson is implemented. The published body is the rewritten `src/learn/data/topics/independent-component-analysis-ica.jsx`: eleven sections, six inline figures, two investigations (the second with a gated follow-up question), two executed Python programs and practices 1–6. This section records what was built, where it departs from the manuscript, the checks actually run, what the screenshots changed and what is still not claimed. It does not close the phase ledger; the integration owner does that.

### What was built

| File | Role |
| --- | --- |
| `src/learn/data/topics/independent-component-analysis-ica.jsx` | The lesson body. Default export shape preserved. |
| `src/learn/data/ica-models.js` | Pure model layer: mixing algebra, a 2x2 symmetric eigen-decomposition, the whitener, the four-state fixture, the FastICA step with three contrasts, deflation, projection kurtosis, the rotation model, the contribution/rescale models, the covariance counterexample and an absolute-correlation helper. Every function refuses bad input rather than substituting a default. |
| `src/learn/data/ica-data.js` | Generated real-recording module: the twelve signed development correlations, the three frozen choices, the held-out diagnostics, a 500-column peak-preserving envelope of the 16–18 s traces, a 20-row exact window, provenance and stated limits. |
| `src/learn/data/ica-examples.js` | The two displayed programs with their actual executed output. |
| `src/learn/components/lesson-labs/IcaShared.jsx` | Investigation state machine, prediction control, number/range/select fields, exact-value table and the equal-scale plot. |
| `src/learn/components/lesson-labs/IcaFigures.jsx` | M1, D1, W1, F1, P1 and E1. |
| `src/learn/components/lesson-labs/IcaLabs.jsx` | R1 rotation, C1 contributions and the C1b scale-ambiguity question. |
| `src/learn/components/lesson-labs/ica-labs.css` | Topic-owned styles, every class prefixed `ic-`, plus the `.ica-lesson` body rules. |
| `src/learn/data/curriculum/blueprints/independent-component-analysis-ica.js` | Blueprint. The implementer did not edit `blueprints/index.js`; the integration owner registered it there (import at L16, entry at L139) after the implementation was handed over. |
| `public/learn-assets/ica/` | `r01-first20s.csv` (hash-checked), `data-provenance.md`, both displayed programs and `runtime-versions.json`. |
| `scripts/verify-ica-models.mjs`, `verify-ica-examples.py`, `verify-ica-data.py`, `verify-ica-browser.cjs` | The four verifiers, with evidence under `docs/teaching/evidence/ica-*.json`. |

Figure and investigation geometry is computed from the shared fixture in the model layer, never typed as drawing constants: W1's diamond, F1's averages and unit circle, and R1's support and kurtosis curve are all derived through `whitenedFixture`, `fastIcaStep` and `rotationModel`, and each is checked by an independent oracle in `verify-ica-models.mjs`.

### Departures from the manuscript, and why

1. **Section 5's closing caveat was updated, not dropped.** The manuscript said the expected results were supported by a hand calculation and a small author probe and were "not a completed production/runtime verification campaign". That sentence described phase one and is now false: the displayed program is executed by `verify-ica-examples.py` in the recorded environment and the page prints that run's output. The published sentence says so. No other honesty caveat was changed.
2. **The environment statement for the real program moved into E1's provenance block**, where it is joined by an explicit "another library version can change the rounding or the component identity" caveat. The program block itself is the shared `RunnableExample`, which prints code and output without room for a prose preamble.
3. **Packet-relative links became served assets.** `r01-first20s.csv` and `data-provenance.md` are offered from `/learn-assets/ica/`. The manuscript's closing pointer to this design record, and its in-text pointers to the visual packet, are authoring references and are not published.
4. **R1's family control.** The family stays staged with the angle, as the specification requires, and a committed prediction is retired by any edit to either. Applying a draft whose family differs restarts the run at that family and the staged angle and says explicitly that the recorded prediction was **not** graded, because the comparison the prompt describes is between two angles of one population. That is the specification's "avoid a changed-family comparison the prompt does not describe", implemented as a visible outcome rather than a silent one.
5. **R1's marker policy.** The applied angle is always marked on the kurtosis curve — its value is already printed in the exact readouts — while the proposed angle is marked only after a prediction is committed, so nothing pre-reveals the answer.
6. **R1's support extent is family-specific**: ±2 for the binary support, ±3.4 for the Laplace and Gaussian contours at levels 1–3. The specification fixes the contour levels and the curve's y extents, not the support panel's axis range; a single ±3.4 box left the binary states in an unreadable knot (see screenshots).
7. **C1's scale mode is a separate sibling investigation**, C1b, revealed only after one exclusion trial is applied, and keyed on the applied amplitudes and keep-set so that changing them retires its prediction by remount. The specification asks for "a focused second question after completing an exclusion trial"; a second investigation with its own reset and undo is the clearest form of that.
8. **P1 is laid out in HTML, not one wide SVG.** See the screenshot section.
9. **Two amplitude bounds.** The learner may enter amplitudes in [-4, 4] (`ENTERED_AMPLITUDE_LIMIT`), and the model's own bound is ±16, because a compensated rescale by up to 4 legitimately carries an entered 4 to 16. Both bounds are exported and tested.

### Checks actually run

| Check | Command | Result |
| --- | --- | --- |
| Model layer | `node scripts/verify-ica-models.mjs` | **PASS** — 9 groups, 14,978 assertions. Independent oracles: eigenpairs confirmed by applying the matrix (including the diagonal and isotropic branches that a closed-form helper gets wrong), K Sigma K-transpose = I on every non-degenerate fixture, the fixed-point averages recomputed by an explicit loop, the projection kurtosis checked against cos⁴θ + sin⁴θ = 1 − 2cos²θ sin²θ at 361 angles for each family, the contribution fixtures and compensated-scale nulls taken from the specification, and the published selection re-derived from the twelve signed development correlations. [Evidence](evidence/ica-models.json). |
| Displayed programs | `scratch/lesson-tools/Scripts/python.exe scripts/verify-ica-examples.py --write`, then again without `--write` | **PASS** both times — 2 programs, 48 oracle assertions. The second run proves the recorded output matches a fresh execution. Oracles cover the four-state probabilities, the mixed covariance [[5,4],[4,5]], eigenvalues 9 and 1, the whitened diamond and its absent joint zero, the one-to-one correlation matching, the exact section-5 trace (−1.376, −1.368) to (0.7091653, 0.70504225), the three kurtosis tables, the zero-kurtosis counterexample's sixth moment, the calibration, the 12,000/4,000/4,000 split, `n_iter_ = 14`, `mixing_ @ components_ = I`, unit fitted variances, the round-trip residual and all three selections and diagnostics. [Evidence](evidence/ica-native.json). |
| Generated data | `scratch/lesson-tools/Scripts/python.exe scripts/verify-ica-data.py` | **PASS** — 31 recomputed checks. Recomputes everything from the served CSV, asserts the packet's recorded values before writing the module, and re-derives the envelope extremes from the raw samples. [Evidence](evidence/ica-data.json). |
| Production build | `npx vite build --outDir dist-ica` | **PASS** — lesson chunk 144,917 bytes, 50,930 gzip. |
| Browser | `DIST_DIR=dist-ica LEARNING_BASE_URL=http://127.0.0.1:4181 ... node scripts/verify-ica-browser.cjs` | **PASS** — 6 behaviour groups, 25 captures, 4 layout inspections, 0 geometry findings. [Evidence](evidence/ica-browser.json). |

The browser groups cover: the eleven sections, six figures, two programs with their exact code and output, twelve closed hint/solution disclosures, ten named manuscript claims, four served assets and the 18-of-39 module sequence; both investigations starting with no prediction and a disabled commit; R1's matched, missed, decimal, exact-equal-angle, Laplace-1.875, Gaussian-null, family-restart, invalid-angle, retirement and undo paths; C1's required amplitude change, matched and missed numeric predictions, every keep-set, the zero-source null and the out-of-range refusal; C1b's compensated null, negative-c sign flip, source-only change and c = 0 refusal; 1366/768/390/320 layouts with no page or display-math overflow, the 200% zoom equivalent and 200% root text; and single-lesson loading, completion persistence and recovery from a failed body import.

### What the screenshots changed

Every capture was opened and looked at. Assertions passed before these repairs; none of them would have been caught by an assertion.

- **P1 was rebuilt.** Its "Direct reference" lane label was clipped to "irect reference", and a 520-unit viewBox rendered at 318 px gave roughly 7 px type at 390 px. It is now an HTML lane grid: real text at reading size at every width, with the reference lane drawn as an explicitly empty "not used" cell under the fit stage, the 0–12 / 12–16 / 16–20 ruler kept, and an arrow and output under each stage.
- **F1's third label was cut off** at the SVG edge, reading "sign-equi". The circle's viewBox grew to 260, the labels were shortened and repositioned, and none is now clipped or overlapping.
- **R1's axis caption disappeared.** "angle theta, degrees" was drawn six units below the viewBox, so it was invisible at every width. The viewBox grew by 16 units.
- **R1's binary support was unreadable** inside a ±3.4 box; the extent is now family-specific.
- **E1's trace strips collapsed to about 27 px** at 320 px, because `height: auto` with `preserveAspectRatio="none"` scales the height with the width. They now have a fixed 68 px height, a frame and a zero line, so only the time axis compresses.
- **E1's headline number was off-screen at 390 px.** The long signed-development column moved last so the chosen coordinate and both diagnostics are visible without scrolling.
- **Table captions were cut off on a phone** because they sat inside the horizontal scroll region. `DataTable` now renders the caption above the region and links it with `aria-labelledby`.
- **The equal-scale plots clipped their right-most tick label** by two pixels at 320 px, and their bottom-left corner labels overlapped once narrow-width type was enlarged. Right padding went from 8 to 22 units and the x tick labels moved from +17 to +22 with a taller bottom pad; the layout inspector now reports zero findings at all four widths.
- **C1 drew a removed contribution at full strength**, so an excluded component looked retained. Removed bars are faded and the bar scale is stated.
- **The first element captures were overlaid by the site's sticky header**, hiding both investigation titles. Chrome hiding moved before every capture, including the zoom page.
- **The signed formatter printed "+0"** in D1's product column and in R1's equal-angle feedback.
- Narrow-width SVG type was raised to 13 units at 520 px and below, so a 300-unit panel rendered at 248 px does not fall to 9 px.
- Two page-level defects came out of the same pass: five display-math blocks overflowed at 320 px (the FastICA update is now split across three lines, plus a measured font step at 520 px and 380 px), and 200% root text at 780 px overflowed once the lane grid existed, because its text cells could not shrink below their min-content width.

### What is still not claimed

- No independent phase-two review has been performed. This record is the implementer's own account.
- User acceptance is separate and not recorded.
- Accessibility work is limited to roles, labels, focus styles, a polite status region and keyboard-operable controls, all exercised through Playwright's accessible-name locators. No screen-reader session was run.
- The numerical results belong to the recorded environment (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1). Another version may change rounding or component identity; the page says so.
- The real-data finding remains one within-recording comparison on one 20-second extract, with PCA ahead of ICA on the declared diagnostic. That outcome is published exactly as it came out.
- Browser coverage is Chromium (msedge channel) at 1366, 768, 390 and 320 px plus a 200% zoom equivalent. No other engine was exercised.
- The phase ledger is not closed here; the integration owner owns it. `blueprints/index.js` was not edited by the implementer either, and the integration owner has since registered the blueprint in it — that registration is now covered by the browser evidence, which records the file's hash and exercises a build containing it.
- Nothing is committed.

## Disposition of the independent review

The [independent review](ICA-INDEPENDENT-REVIEW.md) recomputed every stated number from first principles, with exact fractions and 50-digit arithmetic, importing nothing from `ica-models.js` or any verifier. **It found no numerical disagreement anywhere**: the four-state covariance and eigenpairs, the whitened diamond, the whole fixed-point trace, the three kurtosis families, the zero-kurtosis counterexample, every contribution fixture and all six practice answers agree to full stated precision. The real-recording pipeline reproduces bit-exactly from the served CSV, including all twelve signed development correlations and all 4,000 published envelope values at zero difference, and both displayed programs reproduce their output exactly. All fifteen external links resolve and support their claims, and the served CSV is byte-identical to the packet's.

It raised two blocking findings, eight to fix and nineteen observations. Every blocking and should-fix finding is resolved.

| Finding | What was wrong | Resolution |
|---|---|---|
| B1 | The rotation investigation told every learner "Your direction changed the fourth moment", including in the Gaussian null where the section argues the contrast carries no direction, and at a binary rotation where it is false | The sentence now branches on what happened: the same direction, a direction that moved while the moment did not, or a direction that moved the moment |
| B2 | The design record's header still said implementation not started and review deferred, above its own phase-two section, and claimed the blueprint index was untouched | The header is now a live state table, and the blueprint row says the integration owner registered it |
| S1 | The scale-ambiguity follow-up displayed its own answer on first render, before any prediction | Before the reveal it shows only the state about to be rescaled, with a line saying the result appears after a prediction is recorded and applied |
| S2 | That follow-up chose its explanation from the mode rather than the outcome, so it could claim a changed contribution while reporting unchanged sensors | The explanation branches on the computed outcome, with separate cases for an excluded component and a zero amplitude |
| S3 | F1's table clipped mid-numeral at 390 px, after which the visible values averaged 2.95 against a stated 3 | The exact-value tables stack into one labelled block per row below 560 px; the browser check now asserts both that they stack and that nothing overflows |
| S4 | The prose printed 0.7050423 where the figure printed 0.705042249 | The prose carries the correctly rounded 0.7050422; the packet was wrong in two mutually inconsistent ways and the computed figure was right |
| S5 | The served provenance said LF line endings where all 20,001 lines are CRLF, and two internal links resolved to the single-page fallback | Corrected in the served file |
| S6 | P1 never stacked on a phone, breaking words mid-syllable, and had no lock marker | The lanes wrap cleanly at 390 px and the reference lane carries an explicit "not used" marker |
| S7 | E1 hid the held-out column at 320 px, so the lesson's headline numbers needed a horizontal scroll | E1's table stacks; every representation now shows its development and held-out values in full at 320 px |
| S8 | The blueprint registration postdated the browser evidence | A sequencing artefact: the integration owner registered the blueprint after the implementation agent finished. The evidence was re-captured afterwards |

Acted on from the observations: the data verifier runs read-only unless given `--write`, so it can check without rewriting the module; and three further tables whose prose columns were cut at the right edge on a phone now stack with the rest.

After the fixes all four verifiers pass again: 9 model groups with 14,978 assertions, 2 programs with 48 oracles, 31 recomputed data checks, and 6 browser behaviour groups with 37 captures. The narrow captures were re-read to confirm the two layout findings are actually gone in the images rather than only in the assertions.
