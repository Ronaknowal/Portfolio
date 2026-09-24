> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Bayesian Networks & Causal Graphical Models — research/write handoff

Stable ID: bayesian-networks-causal-graphical-models. Classical ML position28 (batch10). Content-first,12September2026. Root owns this packet; ownership transferred before authoring. Next action is authorized phase-two implementation/review from the content checkpoint, not publication now.

## Inputs, scope and conservation

Read repository AGENTS/current teaching policies, exact --topic --work content preflight, own incoming note and returned shared-note disposition. Read the entire original1,050-line JSX in ordered chunks before drafting. Source: src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx, baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738, SHA256797bd522e4f6fce6dffe2310eeaef499c8a5f672d1b8c98640099d27dbac0054. Original remains untouched.

Title and stable identity retained. Scope is ordinary BN representation/inference/learning plus a locally self-contained causal extension. A general DAG cannot be treated as causal automatically. Predecessor HMM supplies repeated chain factors; successor CRF contrasts conditional label modeling. Feature Selection's author specifically requested a Markov blanket/conditional-information bridge: defined locally here; no duplicate SHAP/selection implementation. The mathematics causal lesson is a deeper route, not a hidden prerequisite. First route§1–6/practice1–6; causal calculus/counterfactuals and broader computation later.

| Original substantial coverage | Current home and correction |
| --- | --- |
| Motivation, alarm, expert/diagnostic applications | §1–2 complete finite model; §8 diagnostic/reliability/measurement decision; avoid unsourced named-tool adoption and policy claims |
| DAG/product/local Markov/parameter savings | §1 normalization argument, local property,20stored/10free vs32stored/31free |
| D-separation, collider, faithfulness, Markov blanket | §4 all-path test with observed descendants; active does not prove actual dependence/sign |
| Exact enumeration, VE, table trace, posterior chart | §2–3 actual32-world/program calculation and factor contraction; all wrong old VE cells/posteriors replaced |
| SCM, truncated factorization, backdoor, frontdoor,3do-rules | §6–7 neutral intervention example, explicit path/support conditions, nested frontdoor arithmetic and correctly defined Z(W) |
| Counterfactuals/abduction/action/prediction | §7 two complete SCMs with same observational/interventional distributions but different paired outcomes |
| NumPy/source algorithm demonstrations | complete offline network-experiments.py including d-separation reference, exact sums, causal/calculated fixtures and real learning |
| pgmpy/structure learning/DoWhy | self-contained pgmpy-example.py with current explicit state-order API, execution deferred; §8 structure search/PC/FCI/MEC explanation; the old commented DoWhy placeholder is replaced by complete local causal computation, not copied as runnable |
| Learning from data, Bayesian smoothing | §5 actual NB/TAN Wine comparison and missing queries; counts/Dirichlet predictive distinction; EM in§8 |
| BN/NB/MRF/SCM/neural/probabilistic programming comparison | §8 scoped computational/semantic differences, no blanket superiority claims |
| Treewidth/junction tree/approximation/resource scaling | §3 fill/storage,§8 running intersection/triangulation/sampling/VI/loopy BP/MAP distinctions |
| Failure modes and all6old exercise themes | warnings placed once at their decision;10changed practice with separate hints/solutions, including repaired education graph and causal-policy distinction |

No original important outcome silently removed. Unsupported exact DAG-recovery, magical hardware cutoffs, industry rankings and fake numerical outputs are replaced with mechanisms and bounded actual evidence, not preserved as facts. Scope expansion adds real fixed-data CPD learning, missing-measurement marginalization and identifiable-vs-fully-known-model contrasts because this is their best local home.

## Canonical-reference section audit

Read the full actual [Stanford CS228 notes contents](https://ermongroup.github.io/cs228-notes/) (including HTML text because web extraction initially hid its index) and inspect relevant chapter material. Map the course's complete section list to teaching ownership:

- Preliminaries: introduction, probability review, applications — §1/2 and§8; independent probability depth belongs to existing mathematics probability/Bayesian lessons.
- Representation: Bayesian networks —§1/4; Markov random fields —§8 contrast and immediately following CRF lesson for conditional-factor mechanism.
- Inference: variable elimination —§3; belief propagation/junction tree —§8; MAP —§8 concrete sum/max counterexample; sampling —§8 mechanisms/rare-evidence example then existing Monte Carlo Methods & MCMC; variational inference —§8 then existing Variational Inference.
- Learning: directed maximum likelihood —§5; undirected/CRF gradients — successor CRF; latent-variable/EM —§8 local connection and prior GMM; Bayesian learning/conjugacy —§5 independent Dirichlet rows then existing Bayesian Inference & Conjugate Priors; structure learning/Chow–Liu/AIC/BIC —§5 actual TAN/§8 search and prior Regularization§9criteria.
- Bringing together: VAE/reparameterization — later actual VAE & ELBO topic under generative-models; further reading structured SVM/Bayesian nonparametrics remains optional specialized extension, not a missing BN-core prerequisite.

The notes mark several sections under construction. Do not claim unavailable completeness or reproduce informal mistakes (including treating every graph-active path as numeric dependence or confusing marginal-MAP with a most-probable full world). This is a coverage audit of the canonical section list, not a claim to have read every course chapter.

## Research and inspection extent

All inspected12September2026. Learner references carry annotations; no video count quota was imposed. Slide format is a useful alternate visual route; no lecture/video was watched or described as watched.

| Primary resource | Actually consumed and use |
| --- | --- |
| Stanford directed-model notes | substantive d-separation/motif/descendant/I-map passages; independent definitions contrasted with faithfulness in§4 |
| Stanford VE notes | opening factor operations/elimination mechanism and chain example; use own alarm arithmetic and storage example |
| Stanford directed-learning chapter | complete returned130-line chapter including local multinomial likelihood; note its empirical-risk sign typo is not reproduced |
| Stanford junction-tree page | page available; conceptual chapter locator, not claim of full chapter read; own cluster/running-intersection explanation retained from stable theory/original scope |
| Stanford sampling chapter | substantive forward/rejection/importance/self-normalization sections through MCMC introduction; no whole-course or all-sampler review claimed |
| Friedman/Geiger/Goldszmidt1997 author-host PDF | actual§4pp140–144 algorithm, empirical conditional MI, spanning tree theorem and smoothing motivation; not full33-page benchmark reproduction |
| Pearl r416-reprint | structural/abduction setup and printed2517–2519 graph/identification/3rules/backdoor; distinction Z(W),support and available-data identification; later mediation passages consulted only as context |
| Mohan/Pearl UAI2012 tutorial | verified author PDF, frontdoor definition/diagram locator and displayed conditions; not claim all117slides reviewed |
| CMU10708lecture18scribe | opened, did not locate needed frontdoor text, so not relied on for its criterion or included as a learner reference |
| Current pgmpy DiscreteBayesianNetwork/Parameter Learning/VariableElimination/inference example | constructors, CPT/state order, query signature, estimator categories and do method semantics inspected; former exact_infer/ve URL404, replace with working generated API; no asserted release version or package execution |
| UCI Wine DOI and prior extract provenance | current DOI resolves proper Wine page with178rows/13features, CC BY4; preserve exact existing offline CSV, no archive byte-equivalence claim |

## Author calculations and reproducibility

Executed network-experiments.py with shared read-only Python3.12.14/NumPy2.3.5/sklearn1.9.1. Three actual learned fits: NB/TAN on106train, selected NB refit on142dev. Fixed36validation/test; seeds61/62; actual Wine178 retained and four fields declared before scores. Selection NB validation logloss.162922448947 vsTAN.185232885991, both34/36; final NB33/36,.270943946448, prior14/36,1.089616221869. Retain this unfavorable TAN result. No alternate test score or post-result recipe change.

Exact alarm32-world queries agree with hand VE, including g(B1,A1)=.94002 and B|JM,E0=.3441995978. B|A,E1=.0032684236 replaces erroneous.116. John redundant after A fixed. Source service effect.07 vs association.1225 yields risk-difference bias.0525. Frontdoor direct and observed-formula computations agree. Education original both singleton sets pass; adding E→Y invalidates S-only, retained as changed exercise.

Author edited/null fixtures actually executed, then saved as the named changed_and_null_checks function and checked against its entire recorded output without rerunning unchanged fits. JSON holds all results. I1Johnfalsecall.2 gives.1654409766 withbothcalls; unused A0row is null whenA1fixed; equal rows uninformative; impossible evidence explicitlyNone. I2observed descendant/second path and changed adjustment correct. I3specimen156alcohol13.17→12.9 flips leading class2→1 with flavanoids visible;12.8samebin identical; hidden edit identical. I4responseedit gives causal.05/association.1125. Exact short displayed stdlib program was executed from manuscript and printed.28417183536439294.

Optional pgmpy-example.py is complete and syntax-checkable but unexecuted; its answer is mathematically expected, not native output. Main program's initial full run preceded adding the identical separately executed changed-check function; those added calculations were replayed exactly. Full assembled CLI/native-library, production model, independent correctness/teaching, browser/accessibility and integration campaigns belong to phase two.

## Author learning-experience checklist

Full manuscript reread in three ordered chunks, after writing all sections; full specifications/provenance/design read as part of closure. Content corrections during reread: define local Markov property and moralization at first use; preserve conditioning support, noncausal predictive graph, individual coupling, actual unfavorable NB/TAN outcome; attach actual module-aware continuation links.

1. **First-pass path:** §1–6/practice1–6 has probability refreshers and complete small examples; formal do-calculus is later. Wider source scope is preserved in named deeper sections.
2. **Caution load:** cautions attached to decisions (graph meaning, support, estimation and resource choice), not repeated after every table; ordinary probability explanations lead.
3. **Real question/data:** missing measurements and comparative probabilistic prediction on attributed178-specimen Wine; artificial alarm/service/frontdoor identified as constructed.
4. **Investigation quality:** I1CPT/evidence, I2graph/observation edits, I3measurement masks/values, I4assignment/response mechanisms. Each has a committed unset input-bound prediction, independent comparison and meaningful entity edit.
5. **Fixture suitability:** all four have changed/null/invalid contrasts with actual outputs where numeric; graph handles collider descendants and all paths. No required uniformly favorable result.
6. **Figure perceptibility:** graph axes/arrowheads, factor cells, two mixture lanes, paired units and measured probability comparison each have a distinct purpose; exact values/scale specified. Rendered visual inspection explicitly deferred.
7. **Connections/scope:** HMM→general DAG→CRF, earlier feature blanket query and learning criteria; canonical section audit routes genuine advanced branches. No topic rename needed.
8. **Mechanism code/practice:** compact displayed enumeration, complete downloadable real experiment/current-library alternative, ten changed problems with explained hints/solutions. Larger reproducibility code is downloadable rather than obscuring first-pass prose.
9. **Whole experience:** full prose and specs reread; actual browser screenshots/phone/keyboard/perceptibility untested. Formal independent authoring/implementation review remains later, distinct from this author pass.

## Retained files and continuation

lesson.md, visual-specifications.md, design.md, wine.csv, data-provenance.md, network-experiments.py, pgmpy-example.py, calculated-inputs.json. They are pending implementation inputs, not disposable scratch. No generated images/download caches or package installs. Topic incoming note is content-adapted but remains open for phase-two confirmation.

For an authorized finish: run exact topic --work finish preflight, consume all files, independently inspect complete prose/assumptions and calculations, implement semantic topic-owned model/figures/labs and real-data download, execute actual displayed programs, and perform required model/browser/accessibility/integration checks. Preserve stable identity/order and only then mark implementation complete. Do not publish or claim user acceptance from this content checkpoint.

## 16 September 2026 — phase two, part A: implementation and numerical verification

Authorized finish request for this packet alone. Phase A covers implementation, the three
non-browser verifiers and a falsification harness. Browser review is deliberately deferred: the
blueprint is not yet registered, and the increment owner registers it between phases so that browser
evidence is always captured against a registered page. **Nothing here claims a rendered page has been
looked at.** No file outside this repository's declared ownership for this topic was edited, and the
destination note was not edited.

### What replaced what

The published body at `src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx` was a
1,020-line legacy-template lesson with hard-coded prose values, no models module, no labs and no
figures. It matched git `HEAD` exactly (SHA-256 `797bd522e4f6fce6dffe2310eeaef499c8a5f672d1b8c98640099d27dbac0054`,
the same baseline this packet recorded) and was preserved to `scratch/bayesnet-baseline/` — both a
copy of the working file and a `git show HEAD:` extract — before being overwritten.

### Files created

| File | Role |
| --- | --- |
| `src/learn/data/bayesnet-models.js` | Pure models: the binary-network engine, a factor/elimination engine, d-separation and the backdoor criterion, the intervention and frontdoor models, the counterfactual pair, query families, fitted-classifier inference, both grading rules, and every coordinate a figure draws. |
| `src/learn/data/bayesnet-data.js` | Generated Wine results. Regenerated only by `verify-bayesnet-data.py --write`. |
| `src/learn/data/bayesnet-examples.js` | Generated program records. Regenerated only by `verify-bayesnet-examples.py --write`. |
| `src/learn/components/lesson-labs/BayesNetShared.jsx` | Investigation scaffolding, the DAG renderer, distributions and mixture lanes. |
| `src/learn/components/lesson-labs/BayesNetLabs.jsx` | Investigations 1–4. |
| `src/learn/components/lesson-labs/BayesNetFigures.jsx` | Figures 1–7. |
| `src/learn/components/lesson-labs/bayesnet-labs.css` | Lesson styling. |
| `src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx` | The reader body. |
| `src/learn/data/curriculum/blueprints/bayesian-networks-causal-graphical-models.js` | Authored plan. Not registered in `blueprints/index.js`; that file is the increment owner's. |
| `public/learn-assets/bayesian-networks/{wine.csv, ATTRIBUTION.txt, network-experiments.py}` | This lesson's own served copies. |
| `scripts/verify-bayesnet-{models.mjs, data.py, examples.py, browser.cjs}` | The four verifiers. |
| `docs/teaching/evidence/bayesnet-{models, data, native}.json` | Evidence from the three that have run. |

Everything in the packet is implemented: four investigations (I1–I4), seven inline figures (F1–F7),
ten practice items, both displayed programs and the downloadable experiment.

### The packet's numbers were checked before anything was built

An independent audit recomputed every number the manuscript states, using exact `Fraction`
arithmetic for the constructed models and a path enumerator written from Pearl's definitions rather
than adapted from `network-experiments.py`: **80 checks, 0 mismatches.** `verify-bayesnet-data.py`
then re-derived the entire `calculated-inputs.json` tree: **1,236 of 1,236 scalar leaves (100%)**,
with coverage measured as the set of leaf paths the comparison actually visited rather than counted
by hand, and the run failing unless that set is the complete set. **No disagreement with the packet
was found anywhere.** Nothing in the frozen packet was edited.

### The destination note — findings and recommended disposition

The note was verified rather than taken on trust, by enumerating every path in the stated graph and
then by a second algorithm (ancestral moralisation) that never lists a path at all.

**Finding 1 — the note's reasoning on `exercise4` is correct.** In the graph S→E, S→Y, E→T, T→Y
there are exactly **two** paths between T and Y: the direct edge T→Y, which is the causal path being
measured, and T←E←S→Y. Only the second is a backdoor path, because only its first arrow points into
T. Both of its interior nodes are non-colliders — at E the path reads ←E←, a chain; at S it reads
←S→, a fork — so observing either one blocks it. T's only descendant is Y, so neither E nor S
violates the descendant clause. **{E}, {S} and {E,S} are therefore all valid, and the empty set is
not.** Conditioning on E opens nothing: the only node in that graph with two parents is Y, and Y is
an endpoint of the query rather than an interior node of any T-to-Y path, so there is no collider for
E to be a descendant of. There is no graphical reason to prefer S. Asserted in
`verify-bayesnet-models.mjs`; the same conclusion is reached independently by the packet's own
program, recorded at `educationBackdoor` in `calculated-inputs.json`.

**Finding 2 — the published defect is real and is exactly where the note says.** At lines 991–997 of
the baseline file, prompt part (d) asks "Why is E alone NOT a valid backdoor set?", and the answer
then says E is "NOT a valid backdoor set", reverses to "so E would actually work as a backdoor
set!", reverses again to "in this specific DAG, {E} is valid", and closes by inventing a preference
for S because it "blocks the path more 'upstream'". It also contains a non-sequitur — "conditioning
on E can open a new path via S if S is a confounder" — which is false in this graph.

**Finding 3 — the §9.3 flag is correct, with one qualification the note does not make.** §9.3's
sentence "If there are unobserved confounders (common causes of X and Y that are not in the data),
no set of observed variables can block those paths" is true only when the unobserved variable is a
*direct* parent of both X and Y. Read as a claim about ancestral common causes it is false: if U
influences Y through an observed non-collider W, the backdoor path X←U→W→Y is blocked by observing
W. So the claim is too broad, as the note says. The qualification: the note's second concern, that
"lack of a valid backdoor set is not general nonidentification", is aimed at something §9.3 already
half-states — that section does go on to say the frontdoor criterion handles some cases of
unobserved confounding. The genuinely overbroad parts are the section heading "Unobserved confounders
invalidate identification" and the sentence quoted above, not the frontdoor sentence. Reporting this
distinction so the note's disposition is accurate rather than merely favourable.

**Does the new packet resolve both?** Yes, and independently of this implementation. Manuscript §6
states that either {E} or {S} blocks the sole backdoor path, that their union also works, and that
"the graph does not declare the more upstream variable universally better". Practice 5 supplies the
counterexample graph in which S stops being valid once E→Y is added, which is the right way to teach
the point. Manuscript §7 replaces the broad claim: "An unobserved common cause does not automatically
make all causal questions unidentifiable: an observed noncollider can block a longer backdoor route,
and a valid frontdoor construction can help when ordinary adjustment cannot." Both are now on the
published page: §6 carries a table checking every candidate set against both halves of the criterion,
and §7 carries a dedicated callout stating the corrected claim in both directions.

**What a reader should conclude.** Both {E} and {S} are valid adjustment sets for the effect of T on
Y in the stated graph, and the graph alone supplies no preference between them — measurement quality,
cost, support and efficiency are separate arguments that need their own evidence. And a sufficient
criterion failing is not a proof that an effect is unidentifiable.

**Recommended disposition — for the note's owner to apply; I did not edit the note.** Close both
findings as implemented. Suggested wording: *"16 September 2026 — implemented. Both findings are
resolved in the phase-two body. The exercise is replaced by §6's adjustment-set table, which checks
every candidate set against both halves of the criterion, and by practice 5, which adds E→Y so that
{S} becomes invalid while {E} does not. The overbroad unobserved-confounder claim is replaced by the
callout in §7. The note's reasoning was independently verified by exhaustive path enumeration and by
ancestral moralisation in `scripts/verify-bayesnet-models.mjs`. One qualification: §9.3's frontdoor
sentence was already correct; the overbroad parts were its heading and its 'no set of observed
variables can block those paths' sentence."* Status can then move from open to implemented.

### The optional `pgmpy` program — executed, not deferred

Nothing was installed into `scratch/lesson-tools`. An isolated environment was created at
`scratch/bayesnet-optional/` and `pgmpy` installed there; it resolved **pgmpy 1.1.2, NumPy 2.5.3,
pandas 3.0.5, networkx 3.6.1, SciPy 1.18.1, scikit-learn 1.9.1 on Python 3.12.14**. Note that NumPy
and pandas moved from the shared runtime's 2.3.5 and 3.0.1, which is exactly why the isolation was
required. The program ran and printed `[0.71582816 0.28417184]`; at full precision its burglary state
is `0.284171835364393` against the enumeration's `0.28417183536439294`, a difference of
5.55 × 10⁻¹⁷. The AutoML unexecuted-by-design precedent was therefore **not** needed.

This is a **departure from the manuscript**, which states the program "has **not been executed** in
the shared environment". That sentence is still literally true, but presenting a real output as
merely "mathematically expected" would now be inaccurate, so the page says it was executed in an
isolated environment and names the versions. The manuscript itself was not edited.

### Departures from the packet, each with its reason

1. **`pgmpy` is executed.** As above.
2. **`network-experiments.py` is served as a download** from this lesson's own asset directory rather
   than only linked inside the draft, so the page's instruction to "save it beside the CSV" works for
   a reader. `verify-bayesnet-examples.py` copies it into a scratch directory with the served dataset,
   runs it there, and requires its fresh output to equal the frozen `calculated-inputs.json` **byte
   for byte**. Nothing inside the packet directory is written.
3. **Figure 3 draws two fill edges, not one.** The visual specification's F3 says "fill A–C shown".
   Eliminating in the order B, C, A, D creates A–C when B is removed and then **A–D** when C is
   removed. The manuscript is not wrong — it only describes the first step — but the specified picture
   was incomplete. The figure now draws whatever `inducedWidth(...).fillEdges` returns, and the
   verifier asserts both edges. Found by the verifier, not by reading.
4. **`\widehat I` became `\hat I`** in the conditional-information formula. A `\widehat` accent over a
   multi-symbol group emits a wide accent SVG that has previously forced display math past 320 px; a
   `\hat` over a single letter does not. The mathematical content is unchanged.
5. **Every display equation is wrapped**, using `\begin{gathered}` and explicit line breaks, rather
   than reproducing the manuscript's single-line forms. The escape auditor refuses an unwrapped block
   above a visible-character threshold. Whether this is *sufficient* at 320 px is a browser question
   and is not claimed here.
6. **Section headings are reworded** as navigation labels (for example "One world is a product of
   local choices" for the manuscript's "Build one possible world from local choices"). Content,
   ordering and the first-pass route are unchanged.
7. **Both grading rules were moved out of the component into `bayesnet-models.js`.** `changeDirection`
   and `purchaseComparison` began life inside `BayesNetLabs.jsx`, where a `.mjs` verifier cannot reach
   them. A grading rule that only exists in a React file is a rule nothing can assert, which is
   precisely the shape of this lesson's dominant defect class.
8. **`pathStatus` now explains an active path node by node.** It previously returned "every node along
   this path passes", which hides the collider-descendant rule — the rule a learner most often gets
   wrong, and the one the drawn path list exists to teach. Found by a verifier assertion failing.
9. **`verify-bayesnet-data.py` reports comparison failures before the byte-identity gate.** A changed
   protocol trips both, and with the gate first the run complained about formatting while the actual
   mismatches went unprinted. Found by the falsification harness, not by inspection.
10. **Figure 4 is computed in the browser** from the saved fitted tables rather than baked into the
    generated data module, following the specification's note that phase two may materialise that
    view. Its column means are asserted to reproduce the two recorded validation log losses exactly,
    which is also the lesson's "same quantity by two routes" connection for §5.

### One numerical nuance worth recording

The manuscript's claim that John's call adds nothing once the alarm state is known is exactly true as
mathematics, but the two sums run over different sets of compatible worlds, so the doubles differ in
the last bit: `0.37355122828183607` against `0.373551228281836`. The investigations therefore grade
"unchanged" with a relative tolerance of 1 × 10⁻¹², and the verifier **measures** rather than assumes
the margin: the screening-off residue is a thousandfold inside that tolerance, while the smallest
genuine change any preset can produce exceeds it by more than six orders of magnitude. The nulls that
involve an unread table row are bit-identical.

### Verification actually run

| Verifier | Result |
| --- | --- |
| `verify-bayesnet-models.mjs` | **PASS** — 471 grouped checks across 55 groups: 1,004 preset d-separation cases plus **124,232** on 742 generated graphs, all recomputed by ancestral moralisation; 26 backdoor candidate sets by a second route; 864 measurement comparisons; 530 geometry checks; 24 refused inputs. |
| `verify-bayesnet-data.py` | **PASS** — 1,270 checks, 25 independent property statements, **100.0% of the trust root's 1,236 scalar leaves re-derived**, module regeneration byte-identical. |
| `verify-bayesnet-examples.py` | **PASS** — 3 programs read verbatim from frozen sources and pinned by SHA-256, all 3 executed, 17 oracle assertions, and the full experiment reproduced the packet's `calculated-inputs.json` byte for byte. |
| Escape auditor | **PASS at the time** — 14 files: no raw control byte, no eaten escape, every required KaTeX sequence present, every display-math block wrapped; run over the verifiers as well as the lesson sources. **Correction, added after the independent review: this ran as a session-local tool and was NOT in the repository, so this row described a check nobody else could execute. It is now committed as `scripts/verify-bayesnet-escapes.py`; see the disposition section below.** |
| `verify-bayesnet-browser.cjs` | Written, syntax-checked, **not run**. Phase C. |
| Production build | `npx vite build --outDir dist-bn` succeeds; the lesson chunk is 169 kB raw, 52 kB gzipped, plus a 9.9 kB CSS chunk. Build directory deleted afterwards. |

The d-separation sweep is the centre of this. A graph question has an exact answer, so a lab that
grades correct reasoning wrong is the cheapest defect to ship here. Every verdict is recomputed by a
genuinely different theorem — build the ancestral subgraph, moralise, drop directions, delete the
observation set, ask whether a path survives — which shares no code with the module's path
enumeration. It is applied to every ordered endpoint pair and every subset of the remaining nodes on
all eleven graphs the page can draw, **and** on 742 generated DAGs, because investigation 2 lets a
learner build graphs no preset sweep reaches. The sweep is also required to produce both verdicts, so
a constant grader could not pass it.

### Falsification harness

Writing a guard is not evidence that it can fire. Twelve breakages were applied one at a time, the
named checker run, the file restored, and the restoration verified by SHA-256; all four checkers were
required to pass before the first breakage and again after the last.

**12 of 12 made their guard go red**, each in the intended checker: the grading rule losing its
fourth answer; a collider ignoring its descendants; the backdoor criterion dropping its descendant
clause; a drawn edge starting at a node centre instead of its rim; every layer drawn at the same
height; the tree-augmented model losing its feature parents; one published evidence set disappearing;
an active path ceasing to explain itself; the validation split moving by one seed; a program pin no
longer matching its source; a shell eating a backslash in a KaTeX string; and a raw control byte
landing inside a verifier. Two of them initially failed to fire correctly and that is how departures
8 and 9 above were found.

### Trust-root coverage, stated plainly

**1,236 of 1,236 scalar leaves (100%)**, measured as asserted leaf paths visited by the tree
comparison, not by a hand-maintained counter. The run fails unless the covered set equals the
complete set, so a block added to the packet lowers the count and stops the build.

**What is not covered, and why.**

- **Rendering, layout, keyboard and accessibility.** Phase C. A green verifier is not a rendered page,
  and none of the screenshots exist yet.
- **The split itself.** Reproducing the declared stratified split necessarily calls scikit-learn's
  `train_test_split` at the declared seeds; a different library version could move it. Everything
  computed *from* the split is independent.
- **Per-path blocked flags.** The overall separation verdict is recomputed by a second algorithm; the
  per-path flags a figure draws are checked against the definition directly, because the second
  algorithm does not produce paths. The two are tied together by asserting that the verdict is
  exactly "no active path remains".
- **Investigation 3's sweep covers the tree-augmented model only**, which is the only model that lab
  exposes.
- **The manuscript's prose**, as prose. Its numbers, graphs and programs are all checked; its
  explanations were read but are a stage-4 reviewer's subject.

### Status

| Axis | State |
| --- | --- |
| Implementation | complete for phase A |
| Computational verification | complete — three verifiers pass, falsification harness 12/12 |
| Browser and visual review | **not started** — phase C |
| Independent review | not started |
| Integration | not started — blueprint deliberately unregistered |
| User acceptance | separate |

**Next action:** the increment owner registers the blueprint in `blueprints/index.js` and the topic's
delivery-ledger entry, then resumes phase C: build, preview on 4191, run
`verify-bayesnet-browser.cjs`, open the screenshots and look at them, fix what is visible, and report.

## 16 September 2026 — phase two, part C: browser review

Run against a production build previewed at `127.0.0.1:4191`, with the blueprint registered by the
increment owner between phases. **17 cases passed and 35 screenshots were captured, and every one of
them was opened and looked at.** Seven defects were found, all but one of them only by looking.

### The empty ternary, and what replaced it

`{bayesnetExamples.pgmpy.executed ? '' : ''}` rendered nothing either way — an intended conditional
phrase that had been lost. Rather than restore a bare conditional, the generated examples module now
records the **environment each program actually ran in**, and the prose names it from that record:
"installed pgmpy 1.1.2 into an isolated environment", and then why the isolation mattered — NumPy
2.5.3 and pandas 3.0.5 against the 2.3.5 and 3.0.1 the page's other programs ran on. The model
verifier asserts that those two environments really differ, so the sentence is a checked claim rather
than a remembered one. The escape auditor gained a guard that refuses any JSX conditional whose
branches are identical; it was confirmed to fire on exactly the expression that was removed.

### Defects found, and how

| # | Found by | Defect | Fix |
| --- | --- | --- | --- |
| 1 | the first browser run | **The lesson did not render at all.** §2 called `queryPosterior` with evidence on its own query variable, which the model correctly refuses, so the body threw during render and the error boundary replaced it. Every non-browser verifier was green. | Read the numerator from the mass already computed. The verifier's own refusal assertion was right; the page was calling it wrongly. |
| 2 | the browser run | An edge in the frontdoor graph ran **straight through the two nodes between its endpoints** and through their labels. A layered layout puts a whole chain in one column, so a skipping edge is drawn over everything in between. | New `edgeRoute`/`graphRoutes` in the model layer bow an edge around every node it does not join, choosing the smallest clearing bow from a fixed ladder and trimming the ends to the circles by bisection. The verifier now asserts the clearance on every edge of every drawable graph at five widths. |
| 3 | the browser run | The page **scrolled sideways by 24 px at 768 px**: the 64-character SHA-256 the provenance prints is one unbreakable word. | A lesson-scoped rule lets inline code break anywhere. |
| 4 | the browser run | Two display equations **overflowed at 320 px** by 11 px each. | Both split across lines. The frontdoor formula now names its inner average `q(m)` and is written as the two steps it is computed in, which also matches figure 5's two trays. |
| 5 | **looking at figure 5** | The outer tray's second identification route was **outside its scroll box**. The prose says "the two right-hand columns agree to the last digit", and a reader could not see the second one. | The outer tray moved below the two-column row and now spans the full figure. A new browser guard fails any figure table that clips at desktop width. |
| 6 | **looking at figure 6** | Same class: the "Y if X = 1" column — the second potential outcome, which is the entire point of the figure — was clipped in both half-width panels. The new guard then caught it a second time at four columns. | Four tighter columns, with the equal weights stated in the caption instead of occupying a column. |
| 7 | **looking at investigation 3** | **The lab showed the answer before asking the question.** Each hidden measurement's edit box was prefilled with that measurement's value, so a learner could read all four numbers off the controls and never had to decide which was worth buying. | `NumberField` gained a `blind` mode that publishes edits without displaying the value it edits, so the "edit a hidden value and watch nothing move" null still works. A browser case now asserts that none of the four measured values is readable anywhere in the panel — text or control value — before it is revealed. |

Three smaller things, also from looking: figure 2's post-elimination factor values sat beside the
final masses without saying the burglary prior had not been multiplied in yet; the masses table
printed a share of "1" when the evidence probability was exactly 0; and a developer-ish "recorded
under key 138 characters long" had leaked into learner-facing text. All three are fixed.

Two repetition defects: every small graph printed a 40-word legend of its own, so figure 7 carried it
four times, and a graph whose labels equal its node names printed "X = X; Z = Z; Y = Y". The caption
generator now omits identity pairs, and a `legend={false}` option lets a group of small graphs state
the legend once.

Re-reading practice 1 against the lab found a **promise the page could not keep**: the solution told
the reader to reproduce the result in investigation 1 by editing Mary's rows, and investigation 1
exposed only John's. Mary's two rows are now editable and there is a "Practice 1" setup that applies
exactly that change. The model verifier now also checks that **every value the prose asks a learner
to type lands on its control's step and inside its range** — 26 values across the probability and
measurement editors — and that check is itself shown to reject a four-decimal value on a
three-decimal control.

### Departures added in this phase

11. **Edges are routed, not drawn straight.** Defect 2. The bow is computed in the model layer and
    asserted, so the drawing is a verified object rather than a component's guess.
12. **The two intervention panels share declared positions.** A layered layout re-ranked X once its
    only parent was cut, so the two drawings a reader is asked to compare had different shapes. Both
    now use one fixed set of coordinates, verified for overlap, bounds and edge clearance, and only
    the arrow set differs.
13. **The frontdoor formula is written in two steps** with a named inner average. Presentation only;
    the quantity is unchanged.
14. **The examples module records each program's runtime environment**, and the page names it.

### What the browser review actually checks

Structure, position and neighbours; that **nothing is revealed on first paint** and every Apply is
disabled until a prediction is recorded; that investigation 2 accepts the correct separation verdict
both when the collider is closed and when only its descendant is observed, retires the verdict on an
edit, and refuses a cycle-creating edge beside the control without losing the graph; that
investigation 1 reports impossible evidence as having no posterior; that investigation 3 leaks none
of the four measured values before they are bought; that investigation 4 grades a randomised
assignment; layout at 1366, 768, 390 and 320 px with no label collision, **no curve through a label**
(the shared inspector walks only straight `<line>` elements, so path trails and fill arcs are sampled
separately), **every SVG label at least 8 rendered pixels** measured in real pixels across 40-odd
labels rather than in user units, no display-math overflow, no horizontal page scroll and no figure
table clipping a column; keyboard focus; completion under the stable ID; and recovery from both an
import failure and a render failure.

The underlay exemption is earned rather than asserted: the highlighted path trail joins node centres,
so it must pass under the labels of the nodes on the path, and a separate check confirms it really
does precede every node group in document order — SVG paint order is document order, so an underlay
that moved later would be drawn *on top* of the labels while the sampler kept skipping it.

The browser verifier also gained a harness fix: waiting for `.bn-lesson` alone reported defect 1 as a
bare 30-second timeout. It now races the lesson against the error boundary and fails with the message
the page actually logged.

### Screenshots

35 captures, each with a unique path, a unique SHA-256 and a recorded byte count; the verifier
refuses a reused path, a duplicate digest and any capture under 2 KB, so a stale orphan from a
pre-fix run cannot be mistaken for a fresh one. Every run in this phase deleted the previous captures
first. The narrow-width set was changed after the first pass: capturing the first three figures in
DOM order photographed three small graphs and missed every table, so it now captures the assembly
strip, the two elimination chains, the frontdoor trays and the paired-outcome matrices, plus
investigations 2 and 4.

Confirmed by eye at 320 px: the elimination chains stack with both fill arcs still distinguishable;
the frontdoor tables blockify into labelled rows with the two identification routes still adjacent;
and the paired-outcome matrices stack per unit with the row-versus-column contrast intact.

### Figure 3's two fill edges

Confirmed in the rendered image, as asked. The left panel says "No fill edge is created at all" and
draws none. The right panel draws **both** dashed arcs at different heights, A–C over one span and
A–D over three, its caption reads "Fill edges: A–C, A–D", and the prose says "removing B joins A and
C, and then removing C joins A and D in turn". Both arcs are visible and separable at 1366 px and at
320 px.

### Verification state at the end of phase C

| Verifier | Result |
| --- | --- |
| `verify-bayesnet-models.mjs` | **PASS** — 504 grouped checks across 58 groups: 1,004 preset d-separation cases plus 124,232 on 742 generated graphs by a second algorithm; 26 backdoor sets; 864 measurement comparisons; 960 geometry checks; 26 typeable values; 26 refused inputs. |
| `verify-bayesnet-data.py` | **PASS** — 1,270 checks, 100.0% of 1,236 trust-root leaves, byte-identical regeneration. |
| `verify-bayesnet-examples.py` | **PASS** — 3 programs pinned and executed; the experiment reproduced `calculated-inputs.json` byte for byte. |
| `verify-bayesnet-browser.cjs` | **PASS** — 17 cases, 35 screenshots, Edge 153.0.4234.32. |
| Escape auditor | **PASS at the time** — 14 files, also refusing CRLF and dead conditionals. **Correction, added after the independent review: this was a session-local tool, not a repository script, so this row overstated what the tree contained; and its CRLF guard was based on a misreading of this repository's `core.autocrlf=true` convention. It is now committed as `scripts/verify-bayesnet-escapes.py` without the CRLF guard; see the disposition section below.** |
| Falsification harness | **13 of 13** breakages made their guard go red; every file restored byte for byte; all four checkers pass before and after. |

One housekeeping repair: four files had been silently converted to CRLF by a scripted edit on
Windows, against this curriculum's LF convention. They are normalised, and the escape auditor now
refuses CRLF so it cannot recur.

### Status

| Axis | State |
| --- | --- |
| Implementation | complete |
| Computational verification | complete — three verifiers pass, falsification 13/13 |
| Browser and visual review | complete — 17 cases, 35 screenshots, all opened and inspected |
| Independent review | not started |
| Integration | blueprint registered; ledger entry is the increment owner's |
| User acceptance | separate |

**Remaining limits.** No beginner walkthrough was run, so the learning-experience assessment is the
author's own. Accessibility was checked only as far as focus visibility, label association, text
equivalents and stacking; no screen-reader pass was made. The narrow-width inspection covers 768, 390
and 320 px and the figures named above, not every figure at every width.

**Next action:** independent review.

## 16 September 2026 — disposition of the independent review

Reviewed at `docs/teaching/BAYESIAN-NETWORKS-INDEPENDENT-REVIEW.md`. The blocking finding and all five
should-fix findings are resolved; the eleven observations are dispositioned below, nine acted on and
two declined with reasons. Every guard repaired here was then broken deliberately and watched go red.

### Findings

| Finding | What was wrong | Resolution |
| --- | --- | --- |
| **B1** (blocking) — three of four investigations print the graded quantity before the prediction is recorded, and investigation 2 does it on first paint | Investigation 2 rendered `× B–A–E **blocked**` above a box reading "Before revealing the verdict"; investigations 1 and 4 printed both the draft and the applied value of the graded quantity in adjacent text. The guard asserted only that no verdict *banner* existed, which is why it passed. | Fixed as a **property, not three edits**. Every readout computed from the draft now lives inside a `bn-graded` element rendered only once `state.result` exists; what stays visible is the *baseline*, which a learner needs in order to answer "higher, lower or unchanged" at all. Investigation 2 shows the path anatomy — which paths exist, which interior nodes are colliders, which are observed, which have observed descendants — and withholds the per-path verdict, the trail's blocked styling and the drawing's own text alternative; the trail gets a third, neutral state so it does not wear either answer's colour. The browser review now asserts **zero `bn-graded`, zero verdicts and zero path verdicts in every investigation before commitment, and at least one after**, so a readout added without the gate fails rather than silently restoring the leak. Two specific values are additionally pinned **numerically** against every numeric token in the section, because a readout added without the marker would pass the class check while printing the answer in plain text. |
| **S1** — the escape auditor is reported as a passing verifier but is not in the repository | True, and the more serious half of it is that the record misled the reader. The auditor ran from the session scratchpad in phases A and C; four of its claims could not be re-run by anyone, and one of its guards was counted in a falsification total nobody could reproduce. | **Committed** as `scripts/verify-bayesnet-escapes.py`, a fifth verifier writing `docs/teaching/evidence/bayesnet-sources.json`. Its CRLF guard is **deliberately dropped**: the reviewer showed that `core.autocrlf=true` here and every long-tracked sibling is equally CRLF in the working tree, so the guard tested nothing and would have failed on correct files. That was my misreading, and the phase-C normalisation it prompted was unnecessary (harmless, and left as it is). The two guards the record credits — dead conditionals and eaten escapes — are now both falsification-tested in a harness anyone can run. |
| **S2** — the investigation-3 leak guard cannot fire in its text half | It searched for the value **surrounded by spaces**, and the card renders as adjacent inline elements with no separating text, so `" 13.17 "` could never match. Half-guarded, not unguarded. | The panel's text is now scanned for **numeric tokens compared as numbers**, not formatted substrings. Falsification-tested: printing a hidden measurement in its card makes the guard fire. |
| **S3** — the edge-clearance guard samples a curve the browser does not draw, below the router's own threshold, against a shape smaller than the one drawn | Three compounding gaps, and **one live defect**: the frontdoor `U→Y` edge cleared every circle and crossed `X`'s endpoint badge, visibly. | All three closed. `trace` now samples the curve the browser actually paints — the drawn shaft plus the arrowhead triangle — and the path string is generated from the same sub-curve object, so the two cannot diverge. Sub-arc control points are computed properly rather than reusing the untrimmed curve's. Clearance is measured against the **drawn shapes** — circle, endpoint badge, observed tag — at the router's own `clearance`, which is now one number used by both sides rather than two that could drift. An unclearable edge returns `cleared: false` instead of throwing, and `cleared` is asserted on everything the lesson draws. Measured independently in the rendered page afterwards: **57 edge/node pairs, zero violations**, the tightest being exactly the frontdoor edge, now clearing `X`'s badge by 3.9 units. |
| **S4** — investigation 2's practice-5 preset has an invariant verdict and answers a different question | On `T` and `Y` the direct edge can never be blocked, so all four observation sets return "dependence possible". Worse, the §6 sentence sending a learner there is about the **backdoor criterion**, and investigation 2 grades **d-separation**, which counts the causal path the criterion deliberately excludes — so a learner following the cross-reference and predicting "guaranteed independent" was marked wrong. | The preset now offers `S` and `T` on the same graph, where observing `E` separates them and the empty set does not. The presets moved into the model layer as data so that the verifier can **require every preset to produce both verdicts** over the observation sets a learner can reach from it. The §6 cross-reference now says plainly that investigation 2 grades a different question and will not confirm the adjustment table. |
| **S5** — "about" followed by nine to eleven significant figures | `num()` rounds to nine *decimals*, which on a percentage is nine to eleven significant figures: "about 28.417183536%". | A `percent()` helper at the precision the surrounding sentence claims: 28.42% and 0.2084%. |

### Observations

**Acted on.** **O1** — the assertions that could not fail are gone: the storage literal containing no
module symbol, the margin restated in two bit-identical forms, the per-case restatements of the
module's own object literal (pinned once per graph instead, where they still catch `separated` and
`verdict` drifting apart), the arrowhead `spread > 4` on a default parameter (replaced by real
perpendicularity and width checks against the curve's tangent), the `x == x` packet comparison
(replaced by a fingerprint of the whole packet directory taken *before* anything runs), and the pgmpy
oracle entailed by the two above it. **O2** — every reported counter now has a floor, in all three
verifiers; deleting the `record()` calls used to leave a PASS. **O3** — the generated sweep now
requires both verdicts of its own, rather than inheriting a requirement that constrained only the
1,004 preset cases. **O4** — the two skip branches referencing classes that exist nowhere (`bn-halo`,
`bn-grid`) are removed along with the unused CSS rule, so the documented exemption matches the one the
guard actually grants. **O5** — the keyboard check now asserts the outline style it was already
writing into the record. **O6** — the control declarations moved into the model as `controlSteps`;
the components spread them and the verifier reads the same object, and a source check requires every
`NumberField` to use the declaration rather than a copy. The four alarm-CPT rows swept against an
editor that does not exist are gone. **O7** — the second candidate select excludes the first, so two
identical options cannot be offered. **O9** — `--no-evidence` lets a reviewer re-run the models
verifier without writing to the record they are reviewing. **O10** — the optional numeric tolerance
is now relative, so typing `0` against the `.001` prior is no longer reported as "within .005" in the
one lab whose point is that zero and undefined differ. **O11** — every SVG now carries `<title>` and
`<desc>` referenced by id, as the visual specification asks, rather than `role="img"` with an
`aria-label`.

**Declined.** **O8**, the exact floating-point tie between two conditional-mutual-information weights,
is recorded rather than exercised. The reviewer established that it is benign — Kruskal's declared
`(−weight, a, b)` ordering resolves it deterministically, the losing edge would close a cycle
regardless, and byte-identical regeneration already pins the outcome. Constructing a tie that *would*
change the tree means inventing a fixture that is not this dataset, which would test a hypothetical
ordering rather than the protocol the lesson actually ran. Recorded here so a future author meets it.
The **examples verifier's packet-fingerprint guard** is repaired but **not falsification-tested**:
making it fire would require writing inside the frozen packet directory, which is the one thing this
phase must not do. Its structure — snapshot before, compare after — is sound, and that is stated
rather than claimed as tested.

### Falsification

**21 of 21 breakages made their guard go red**, each applied alone, each file restored and re-hashed,
with all five checkers passing before the first and after the last. Two of the twenty-one were
initially silent and both were informative rather than cosmetic:

- Removing the endpoint badge from `nodeKeepOut` changed nothing, because the verifier took its
  expected shapes from the same function the router used. The verifier now constructs the drawn
  shapes independently and asserts their equality with `nodeKeepOut` explicitly, so a change to
  either side is a failure rather than a match. **This was the guard protecting the live S3 defect.**
- Reverting the sub-curve control point also changed nothing — and on inspection that is correct:
  under the repaired design the drawn curve and the measured curve are one object, so a different
  control point produces a different but still-verified curve. The mutation that reproduces the
  *original* bug is a `trace` that samples something other than what is drawn, and that is what the
  harness now applies. Recorded because the distinction matters: the repair removed a whole class of
  divergence rather than fixing one coefficient.

### Verification at the end of this pass

| Verifier | Result |
| --- | --- |
| `verify-bayesnet-models.mjs` | **PASS** — 522 grouped checks across 63 groups: 1,004 preset and 124,232 generated d-separation cases by a second algorithm, 26 backdoor sets, 864 measurement comparisons, **2,775 geometry checks** against the drawn shapes, 26 typeable values, 26 refused inputs. |
| `verify-bayesnet-data.py` | **PASS** — 1,273 checks, 100.0% of 1,236 trust-root leaves, byte-identical regeneration. |
| `verify-bayesnet-examples.py` | **PASS** — 3 programs pinned and executed, 18 oracles, `calculated-inputs.json` reproduced byte for byte. |
| `verify-bayesnet-escapes.py` | **PASS** — 62 source-hygiene checks over 15 files. **New; this is S1's resolution.** |
| `verify-bayesnet-browser.cjs` | **PASS** — 18 cases, 37 screenshots, Edge 153. |
| Falsification harness | **21 of 21**, across all five checkers. |

Every screenshot was opened again. The three states the review named are confirmed by eye:
investigation 2 at first paint now shows the path anatomy and no verdict, with a neutral trail;
investigation 4 at first paint withholds both lane risks and the comparison table while showing the
baseline it asks a learner to predict against; and the frontdoor `U→Y` edge passes clearly outside
`X`'s badge.

**Limits unchanged from phase C**, and still accurate: no beginner walkthrough, so the
learning-experience assessment is the author's own; accessibility covers focus visibility, label
association, text equivalents and stacking, with no screen-reader pass; narrow-width inspection covers
768, 390 and 320 px and the named figures rather than every figure at every width. To these the review
added five exclusions the record did not name — all five are now closed rather than merely disclosed.
