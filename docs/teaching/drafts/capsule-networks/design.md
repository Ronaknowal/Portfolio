# Capsule Networks: design, research and content checkpoint

Stable ID **capsule-networks**. Deep Learning Fundamentals & Architectures, module position **13**, current 30-topic batch position **4**. Owner /root/deep_foundations_content. Content-only revision 1 prepared for root reconciliation; implementation not started.

## Scope and source conservation

Actual topic inventory preflight ran with --topic capsule-networks --work content. Read the returned topic, phase state and author notes; no pending destination note exists for this topic. The unrelated resolved bit-manipulation note is not an authoring obligation. Read current repository AGENTS/handoff, current architecture scope and retained current standard/design/domain/code-ownership-coordination-retention instructions. No full catalogue audit was performed.

Read the complete original src/learn/data/topics/capsule-networks.jsx in contiguous ranges0–120,120–310,310–525,525–740,740–end, including its programs, math, visual code, exercises, claims and references. Original baseline commit **8c5da59f18516be77c29d5aeeafca3decca4f738**; source SHA-256 **f980679c6504889399b8316e83880fe58c58038af7857dfcbc6a8587bf556a58** still matches. Runtime source is untouched.

The stable title “Capsule Networks” remains appropriate; the manuscript adds a descriptive teaching subtitle without changing topic identity. Scope covers vector grouping/voting/routing/objective, an actual controlled trained model, reconstruction/geometry, derivative/EM/cost branches and carefully distinguished alternative routing. No unsupported current production/SOTA verdict replaces the previous one.

Actual sequence is ConvNeXt → Capsule Networks → RNNs/LSTMs/GRUs → Seq2Seq → Bahdanau/Luong Attention. Use real full-curriculum routes with module=deep-learning-fundamentals, and the existing GMM route with module=classical-ml for optional mixture depth. The route IDs were checked against the manifest. The current lesson teaches local prerequisites rather than assuming earlier names guarantee coverage.

| Original coverage or defect | Decision / manuscript home |
| --- | --- |
| Part–whole motivation, scalar vs vector, pose and activation | Preserve through a concrete wheel/whole question and channel-group correspondence; remove claims that CNNs cannot encode geometry or that learned axes are automatically physical |
| Pooling destroys viewpoint information / capsules replace CNNs | Replace with precise local information-loss and architecture-dependent interpretation; capsule frontend remains a CNN |
| Squash and derivative | Preserve exact function at zero, radial/tangent sensitivity and independently checked Jacobian; correct inflection to1/sqrt3 and remove spurious zero singularity |
| Vote matrices and tensor shape | Preserve full shape chain, child/type/location indexing and matrix dimensions; fixed-state author code checks correspondence independently |
| Dynamic routing algorithm and iterations | Preserve exact parent-axis softmax, weighted sum, dot agreement and reset lifetime; replace convergence/certainty promises with contrasting and symmetric null cases |
| Claims that routing is non-differentiable or original TensorFlow must detach | Correct: finite unrolled computation differentiable; explicit full vs stop-gradient comparison shows same forward loss/different derivative; no unsupported implementation-origin claim |
| Margin loss and reconstruction | Preserve full objective/masking/reconstruction; fix SSE-vs-MSE coefficient, selected16-vs-masked160 parameter confusion, true-label conditioning and invalid “all large scores evade margin” claim |
| Full CapsNet topology, count and runtime claims | Preserve exact original topology/counts and formula-based complexity; remove fabricated timing/stdout and placeholder training loop |
| MNIST/affNIST/MultiMNIST | Distinguish ordinary .25% error from expanded-canvas99.23% transfer setup; overlap application retains source-specimen split and instance-capacity caveats; no reproduced-paper score claim |
| “smallNORB is rendered 3D” | Correct to photographs of physical toys; instance/view split matters |
| EM routing | Preserve pose/activation, diagonal mean/variance, responsibility, coding-cost activation, spread loss, floor/mass/log-domain safeguards and transforming-space distinction |
| EM uses full covariance/O(d²) by definition | Correct diagonal coordinate statistics; include local/shared transformations and structured matrix-map cost |
| Variational routing/STAR/current rankings | Verified distinct mechanisms/years; remove false no-ImageNet/no-production/ROI/capsule-obsolete categorical assertions. STAR-Caps has ImageNet experiments |
| Shift/adversarial/equivariance claims | Actual shift failure and explicit symmetry equations; no attack certificate or automatic geometric guarantee |
| Plain generic labs and arbitrary plots | Replace with vote arrows, contribution edges, grid correspondence, real pixel/latent edits, exact coordinate-frame square and EM distributions; no invented learning/benchmark curves |
| Original practice/next steps | Eight changed tasks with closed non-answer hints/explained solutions; concrete unsolved output investigations and correct next route |

The new material can be owned here without a topic rename or scope expansion: routing-gradient distinction, decoder information availability, exact symmetry conditions and controlled uniform comparison directly explain capsule behavior. Full video segmentation, probabilistic derivation, group-representation theory and later attention variants stay local bridges/references rather than sprawling new courses. No new destination note was necessary in this packet; root owns shared note handling.

## Learning design and first-pass boundaries

First pass begins immediately after the introduction: sections1–6 and practice1–5/8. It moves from grouped properties to transformed votes, a complete three-step arithmetic trace, model/target/objective, paired real data and a meaningful transformed-input failure. Sections7–8 and practice6–7 are optional depth. The core readiness checklist does not require completing the EM/representation-theory branch.

| Learning hurdle | Chosen representation / activity | Evidence of understanding |
| --- | --- | --- |
| A “capsule” sounds biological or mysterious | One cell → channel groups → arrows, with explicit indices | Identify coordinate/type/location without inventing pose semantics |
| A part's raw vector is confused with a vote | Child-specific transform and two parent planes | Change one vote while keeping another fixed and inspect its effect |
| Softmax axis and weighted average are confused | Child outgoing edges plus parent contribution arrows | Repeated-evidence exercise and row sums |
| Routing loop confused with training or temporal memory | Separate persistent W from reset b/c and the forward trace | Explain no carried logits between images; bridge to next RNN |
| More concentrated assignment assumed better | Actual paired fits and fixed-weight interventions | Separate correct counts, identity changes, margin and reconstruction |
| Reconstruction uses hidden label information | Visible mask and separately named metrics | Repair true-label-conditioned “inference” report |
| Latent direction assumed a physical pose | Actual coordinate edit versus explicit geometric matrices | Explain local decoder changes and test a commuting diagram |
| EM implementation described vaguely | Full declared diagonal equations and numeric3-step example | Changed activation/mean exercise and inactive-child null |
| Engineering intuition is confused with measured cost | Exact tensors/parameter accounting and formula-based work | State which quantities scale and which latency remains unmeasured |

Visual specifications contain six purpose-specific groups, with explanations and optional activities as appropriate; this is not a lab quota. Compute the comparison from the complete current inputs. Student controls do not simply animate a known answer.

Eight practice tasks cover independent numerical transfer, an axis misconception, objective units, protocol repair, a new left-shift investigation, exact symmetry counterexample, changed EM weights and a new latent edit. Each has a closed Hint and Solution. The live vote demonstration shows the result for the selected vote immediately; separate written-practice hints and solutions may remain closed. Procedural investigation solutions specify what to record and how to interpret it; they do not fabricate an unexecuted left-shift count.

## Primary/canonical source audit and actual review extent

Research checked13 September2026. Web extraction was used for PDFs and live records; reading extent below is intentional. Author programs provide independent calculations and actual small empirical findings. No claim that every linked video was watched or every reference codebase executed.

### Canonical core: Sabour, Frosst and Hinton, Dynamic Routing Between Capsules

[arXiv1710.09829 PDF](https://arxiv.org/pdf/1710.09829), v2,7November2017,11pages. Read actual main sections1–8 and routing/equations/architecture/tables/figure explanations; inspected bibliography for referenced directions, not every cited paper. Actual section agenda and decisions:

| Actual section | Coverage decision |
| --- | --- |
| 1 Introduction | Keep presence/properties, part–whole motivation and convolutional frontend; qualify rhetorical comparisons and automatic viewpoint claims |
| 2 Computing the vector inputs and outputs of a capsule | Core exact squash/vote/softmax/dot-product procedure and zero behavior |
| 3 Margin loss for digit existence | Full local objective and changed arithmetic practice |
| 4 CapsNet architecture;4.1 Reconstruction as a regularization method | Exact original shape/decoder count; scaled complete CPU route and explicit SSE coefficient/label mask |
| 5 MNIST;5.1 What the individual dimensions of a DigitCaps capsule learn;5.2 Robustness to affine transformations | Distinguish model/protocol-specific scores; local coordinate perturbation experiment without automatic semantic naming |
| 6 MultiMNIST;6.1 Dataset;6.2 Results | Overlap mechanism, component targets and original-specimen split; avoid treating generated pair count as independent sample count or claiming matched model capacity |
| 7 Other datasets | Original smallNORB physical-data correction; historical context rather than reproducing stale global rankings |
| 8 Discussion and previous work | Preserve instance-capacity/part–whole questions; replace speculative inevitability with explicit testable symmetry/robustness statements |

Consequential source checks: ordinaryMNIST3routing+reconstruction .25±.005% error;99.23% belongs to40×40 expandedMNIST transfer-trained model, compared with99.22 baseline and79/66 affNIST transfer. MultiMNIST source uses different-class composites, large counts of pairings and a larger CNN comparison; paper text5.0 vs table5.2 disagreement is not carried into the learner as an unqualified score. Baseline size selection looked at a10k test subset, so paper comparison is not treated as a pristine independent final protocol. No artificial exact speed claims retained.

### Matrix Capsules with EM Routing

[Author PDF](https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf), ICLR2018,15pages. OpenReview PDF was blocked by the browser; author-hosted PDF supplied the text. Read main sections1–8, Algorithm1, key architecture/equations/tables and AppendixA.1–A.3 text; AppendixB supplementary figure captions, not all image pixels. Actual agenda:

| Actual section | Decision |
| --- | --- |
| 1 Introduction;2 Capsules | Optional pose/activation/matrix relation and model-specific claims |
| 3 EM routing | Complete local diagonal statistics/activation/responsibility illustration with declared constants and guards |
| 4 Network architecture;4.1 Spread loss | Preserve local/sharing alternative and spread objective; no full native trained matrix model asserted |
| 5 Experiments;5.1 Generalization to novel viewpoints | Correct physical smallNORB/instance split and separate viewpoint protocol; do not portray fixed/selected familiar-view reporting as untouched broad test |
| 6 Adversarial robustness | Explain bounded experimental evidence, no certificate; paper's black-box comparison does not show the same advantage |
| 7 Related work;7.1 Previous work on capsules | Distinguish explicit equivariance and routing direction; do not repeat its “cosine” shorthand as the vector algorithm's literal dot product |
| 8 Conclusion | No undated state-of-art claim |
| A.1 Gaussian-mixture cost;A.2 Transforming Gaussians;A.3 Switchable transforming Gaussians | Optional local explanation: each parent sees different transformed votes; omitted determinants prevent ordinary original-space likelihood interpretation; activations are not sum-one mixture proportions |
| B Supplementary figures | Figure-caption checks contextualize views and vote distances; no copied figure or fabricated data |

Important: original EM coordinate variances are diagonal, not full covariance. A matrix4×4 relation has16 weights versus256 for a generic flattened16→16 map. Local capsule layers and shared class transforms change naive all-to-all cost. A density comparison in a transformed space requires interpretation; basic EM guarantees are not copied without their assumptions.

### Controlled alternative and related research

- [Paik, Kwak and Kim, ACML2019](https://proceedings.mlr.press/v101/paik19a/paik19a.pdf),14pages. Read abstract, routing background, all four §3 questions, relevant §4 protocol/minor-observation text and §§5–6 polarization/limitations discussion; inspect result tables1–4. This is an alternative empirical argument, not the full canonical curriculum. Uniform/random controls, restarting completely failed fits, ResNet-based backbones, matched dataset/configuration limits and mean/std across3 runs limit inference. Its random routing samples U(.8,1.2) each iteration, not a permanently fixed assignment. Its polarization argument permits symmetric/tied exceptions; our null examples retain those.
- [Capsule Routing via Variational Bayes](https://arxiv.org/pdf/1905.11455), v3,3December2019 preprint,9pages; [AAAI2020 publisher record](https://ojs.aaai.org/index.php/AAAI/article/view/5785) verifies conference year. Read abstract/introduction and §2.1/2.2 plus Algorithm1, conjugate-prior and posterior-factorization equations through the agreement discussion. Full later experiments/appendix derivations were not reproduced or needed for the short branch. Explain approximate parameter/assignment uncertainty and prior control; do not claim full VB implementation.
- [STAR-Caps](https://karim-ahmed.github.io/publications/starcaps.pdf), NeurIPS2019,10pages. Read abstract, introduction, background§2.1–2.3, method§3.1–3.3 through attention/decision-maker description and experiment dataset coverage. No code/native timing reproduction. Separate nonrecurrent attentive coefficients, binary routing decisions and surrogate training gradient; verify ImageNet experimentation rather than reusing the old false claim.
- [Group Equivariant CNNs](https://arxiv.org/pdf/1602.07576), Cohen/Welling2016. Read introduction and §2 equivariance equation/explanation for the exact group-action bridge; no complete implementation or paper-wide coverage claim. The explicit matrix/vector counterexamples here are independent author arithmetic.
- [UCI optical digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits): current data/license/schema record opened; local CSV transformation and exact duplicates/split independently checked.

### Alternate learning resources

[Sara Sabour's Introduction to Capsules slides](https://www.cs.toronto.edu/~saaraa/CapsuleSlides.pdf):69slides, complete extracted text/section progression read, especially slides10–40 coordinate frames/agreement and49–59 matrixEM. Screenshot attempts at18/28/51 failed in the web tool; no claim of a complete visual inspection. The text itself provides a useful complementary teaching route; some slides use cosine shorthand, explicitly qualified in the learner annotation.

[UCF CVPR2019 tutorial index](https://www.crcv.ucf.edu/cvpr2019-tutorial/) and [official CVPR tutorial schedule](https://cvpr2019.thecvf.com/program/tutorials): verified university-index search extraction identifies speakers, slide/video links and separate intro/survey/video/segmentation/point/subspace sessions. Direct index retrieval had502/cache errors; no guessed direct YouTube URL or timestamp. The learner links the university entry point with an honest historical/context annotation. Full videos were not watched. This follows usefulness, not a video-format quota. Phase two should recheck the index's availability before publication, retaining the accessible author slides as the alternate route.

## Programs, empirical design and bounded checks

See data-provenance.md for exact data/license/split, versions,6paired600-update fits and selected fixtures. Complete capsule-learning.py, capsule-mechanics.py and author-checks.py all executed. The local import helper allows checks without re-running training. All parameter/shape/routing/gradient/reconstruction numbers in the manuscript arise from these actual programs or explicitly cited historical papers. No placeholder training loop, unexplained dummy dependency, synthetic benchmark, invented curve or hand-labeled physical latent coordinate is present.

The full-gradient route and stopped-gradient optional route agree in forward output; numerical derivative comparison checks the actual full-function gradient and intentionally shows the stopped-gradient discrepancy. Independent selected-image inference uses separate spatial loops and explicit grouping/vote/decoder arithmetic, not the same Torch routine compared with itself. Maximum capsule/reconstruction discrepancies are6.39e−8/2.98e−7. Contrasts and nulls are in immutable JSON inputs for phase two.

The first draft of calculated-inputs.json included redundant complete reconstructions for every fixed-weight intervention. Before freeze, removed those duplicate per-image arrays while retaining all metrics, class lengths, predictions, selected models and meaningful selected reconstruction edits; compacted JSON with no numerical rounding. This reduced that author-evidence file from13,379,740 to2,828,067bytes. The program now emits the same retained schema. This serialization-only repair did not change fits or calculations, and no expensive repeat training was needed. Phase two must still extract the much smaller needed on-demand client subset.

No dependency installs, broad environment setup or implementation pipeline were performed. No native reference-library matrixCaps/STAR/VB execution. The shortened displayed route snippet is the exercised complete program's default arithmetic; setup/install commands document existing tested versions and were not run unnecessarily.

## Own reread and learning-experience checklist

Author reread the entire manuscript in two complete local text reads and all visual specifications after writing, reconciled actual outputs and practice solutions, and reread the consequential repaired sections. Repairs caught during own pass:
- Put both margin terms inside an explicit sum bracket and use inclusive threshold wording.
- Put the changed-vote answer behind a closed disclosure so the prediction remains meaningful. **Historical interaction record:** the earlier prediction/reveal behavior described here is superseded by the 21 September live-exploration contract; it is not a phase-two implementation requirement. Preserve the recorded mathematical checks and fixtures.
- Use “rotate the scene” for the active homogeneous transformation, avoiding a passive-coordinate ambiguity.
- Explain zero-effective-mass/nonidentified mean and the role of the variance floor.
- Preserve exact historical MNIST/affNIST distinctions and physical smallNORB correction.
- Clarify core readiness as normalizing over parents for each child.
- Remove redundant output arrays without changing any required learner evidence.

Checklist assessment:
- Beginner can explain the purpose before notation: yes, whole-from-compatible-parts question and local vocabulary.
- Single example follows every mechanism: yes,3children/2parents through sums/squash/softmax/updates, then one actual image through tensor/model/objective.
- Concreteness and connections: yes, local math refresh, actual ConvNeXt/GMM/RNN route links and explicit state-lifetime bridge.
- Visuals reduce a specific hurdle: yes, different representations and active real entity edits, not a generic lab template.
- Practice requires independent transfer: yes, changed child count, relative agreement, image size/loss scale, protocol repair, new shift, new symmetry map, new activation weights and coordinate edit.
- Hints/solutions concealed:8paired closed disclosures plus one closed changed-vote reveal,17closed details total; no open attribute or exposed plain answer block. **Historical interaction record:** the earlier prediction/reveal behavior described here is superseded by the 21 September live-exploration contract; it is not a phase-two implementation requirement. Preserve the recorded mathematical checks and fixtures.
- Research/correctness nuance: yes, source-specific claims separated from exact math and own fits; unfavorable data retained.
- Applied program complete: yes, local input, setup, full training/evaluation/export, actual outcomes, no unexplained model download.
- Phase boundaries honest: yes, no browser/production/phase-two review claims.

Scoped packet checks passed: Python AST parse for all4Python files; allJSON parsed; relative links resolve; actual local topic routes found in manifest; original source hash unchanged; manuscript math delimiters and tables inspected; all practice disclosures closed. Root will bind final file hashes and assess content reconciliation. Existing source-bound evidence remains reusable for unchanged files.

## Later authorized finish

Run the actual --work finish preflight after root creates the complete content checkpoint. Consume the full packet, implement the specified diagrams/investigations and precise model/asset contracts, render native formulas and disclosure blocks, verify calculation parity and real input interaction, check mobile/keyboard/accessibility/performance/failure recovery, perform required independent phase-two content/correctness/learning review and integrate only then. Do not regenerate this manuscript from an outline or infer publication from content completion.

Retain all12packet files, including offline data and necessary results/programs. Root owns shared ledger/handoff/inventory. No current need for a new topic note or runtime rename. Original source conservation and previous frozen packets remain intact.


## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Manipulate votes and vector geometry. Edit capsule votes, routing iterations, vector magnitude/direction and supported retained image/latent coordinates. Show coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs. Step routing to inspect its computation, with all current outputs visible. Distinguish agreement from activation magnitude, pose changes from class evidence and a model intervention from a new empirical result.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Follow routing all the way into a trainable program” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| Votes, squash, dynamic routing, margin and reconstruction training | `capsule-mechanics.py:routing, squash`; `capsule-learning.py:route, TinyCapsules, margin_loss` | Transparent PyTorch tensor composition is ordinary research route; author-checks supplies independent NumPy saved-state reconstruction | Routing temperature code change and numerical solution; full Matrix Capsules architecture not claimed |
| Diagonal EM and explicit frame geometry | `capsule-mechanics.py:diagonal_em`; local precise limited fixture | NumPy stable statistics, no invented standard full-EM-capsule API | Existing low-activation/coordinate-frame practice; research-scale pose systems are contextual limits |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.
