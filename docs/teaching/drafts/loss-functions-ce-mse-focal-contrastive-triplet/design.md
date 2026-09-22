# Loss Functions — content preparation design and evidence

Stable ID: loss-functions-ce-mse-focal-contrastive-triplet. Module: deep-learning-fundamentals, module position3 and authorized batch position24. Work: research/write only; implementation not started. Prepared2026-09-12. Display title proposal “Loss Functions: Predictions, Probabilities, and Learned Similarity” improves navigation across an already broader topic. Retain the stable ID and the canonical family names in metadata during authorized phase two.

## Scope, ownership, and local readiness

Ran node scripts/build-curriculum-inventory.mjs --topic loss-functions-ce-mse-focal-contrastive-triplet --work content. Published source exists; no pending own-topic notes. No registered prerequisite edges should be interpreted as evidence that a first-time reader knows probability, residuals or embeddings. The lesson refreshes these locally. It uses the previous Backprop lesson for the purpose of gradients, then supplies its own scalar chain example and output gradients. It ends at the actual next topic, batch-layer-group-rms-normalization.

This topic owns the regression→probability→metric-learning objective bridge, conventions/reductions, focal derivative and candidate masks. It gives bounded advanced entry points for Huber/quantile, likelihood, SupCon, ArcFace and the InfoNCE bound. It does not add a new curriculum topic or claim an audited destination for every large-vocabulary implementation. Detailed detection, retrieval systems, sampled-softmax kernels, ranking systems and distributed negative exchange remain applications/deeper reading, not unnamed prerequisites. Classification threshold/cost and validation discipline are refreshed from the classical sequence rather than assumed from a title.

## Original conservation

Baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738. Actual original src/learn/data/topics/loss-functions-ce-mse-focal-contrastive-triplet.jsx SHA256 e2807fe079a8cc4aaadb64808a6be2f804a5e357b448c96ca364a92d15366ad8. Entire original body read in explicit ranges; missing portions of the initially truncated805+ output were reread899–1054 and1054–end; early89–115 also recovered.

| Existing depth | Decision and reason |
|---|---|
| MSE/BCE/CE, likelihood, MAE/Huber | Retain and derive mean without Gaussian necessity; correct100×loss versus10×gradient; add quantile application and unit-change practice |
| Focal formula and imbalance plots | Retain, differentiate multiplier, explicit alpha convention; checked bias-gradient reversal; remove universal ratio/hyperparameter prescriptions |
| Pair contrastive and triplets | Retain both, conventions and squared units, active derivatives, no-candidate policy and collapse stationary fixture |
| NumPy/PyTorch examples | Replace inconsistent fragments with complete local executable functions and CPU experiment; eliminate unspecified pretrained sentence download and placeholder strings |
| InfoNCE and normalized geometry | Retain candidate CE, masks, false negatives, duplicate lower bound; correct logB “maximum,” universal normalization requirement and monotone batch-quality claims |
| SupCon/ArcFace | Retain a properly separated conceptual/formula bridge with original sources; no commercial-dominance claims |
| Triplet miners | Complete explicit nearest semi-hard/skip policy; fix original argmax fallback described as closest |
| Scaling | Retain dimensional memory/compute explanation; remove invented95%/3%runtime and incorrect model-vocabulary examples |
| Practice and failure modes | Preserve derivation/implementation/application/diagnosis roles using changed, nonmedical tasks; correct BCE prevalence optimum qualification and duplicate/shortcut conflation |
| Historical references | Retain relevant verified primary works; remove unsupported Gauss-Markov/priority and universal descendant/superhuman claims |

No published source or blueprint was modified; conservation decisions apply to the prepared replacement.

## Reference section-list audit

Canonical entry reference: Stanford CS231n “Linear Classification” notes, https://cs231n.github.io/linear-classify/ . Read its complete displayed section list: parameterized image→score mapping; interpreting a linear classifier; loss; multiclass SVM; practical considerations; softmax classifier; SVM versus softmax; interactive demo; summary; further reading. Read mapping, loss/practical portions and softmax/probabilistic/stability body, not every linked resource or the external interactive demo. The mapping is a local refresh; image-template geometry remains Perceptrons; SVM's full objective/dual belongs to classical SVM and is not repeated; CE/MLE/stability are core here; score-versus-probability distinction is retained. The canonical reference lacks focal and metric-learning depth, so the following primary section audits extend rather than shrink scope.

- Focal Loss for Dense Object Detection, arXiv1708.02002v2 HTML: reviewed roadmap and §3.1 balanced CE/§3.2 definition plus neighboring robust-estimation discussion. §4RetinaNet/§5detector experiments are context, not requirements for this first loss lesson. No original benchmark copied.
- FaceNet1503.03832v3: read method§3.1loss/§3.2selection and architecture-transition context; sections1intro,2relatedwork,3.3architectures,4datasets/evaluation,5results are not all read in full. Retain squared-distance/margin/mining; defer convolution models and originalscale.
- CPC1807.03748: read§2.3 objective/density-ratio/sampling/bound and adjacent§2.4. Its encoder/autoregression and domain experiments are scoped applications, not required local architecture. Appendix proof not read in full; no claim of reproducing it.
- SimCLR2002.05709v3: read§2.1 and Algorithm1, §2.2opening; retain2Bmask, both directions and explicit distinction from our one-way example. Augmentation/architecture/full benchmark sweeps not read in full.
- SupCon2004.11362: inspected§3.1 representation framework/§3.2loss setup; compare average-log versus log-sum locally, without importing performance claims.

## Additional source verification and actual extent

On2026-09-12 checked PyTorch2.14 CrossEntropyLoss actual API body including class-index versus probability-target mean denominators; TripletMarginLoss actual distance/epsilon/body; Charoenphakdee2011.09172v2 abstract for focal not-strictly-proper distinction. SciPy Huber API source page retrieved; local Huber convention independently derived. ArcFace1801.07698v3 method algorithm/formula excerpt inspected; currentv4abstract differs from originalCVPRpaper, so local formulation is pinned tov3.

Hadsell author .com PDF returned502. Author publication index confirmed metadata and .org PDF located, but both PDF hosts returned502 on direct retrieval. The lesson links the verified author index; neither PDF is claimed read. Pair formula independently calculated, with convention stated. Source citations are localized, not copied prose. Stanford official2017syllabus and School of Engineering YouTube Lecture3 metadata/description verified for a usable alternate video; video not watched and no timestamp invented. Current2026course video link is Canvas-gated, so do not send learners there as the direct alternate.

## Hurdles and instructional choices

Loss height versus tangent needs aligned plots, not a definition card. Regression uses an observation the learner actually edits, showing why the fit changes. Focal needs signed aggregate gradients and an independent null, not a curve alone. The real experiment separates model probabilities, decisions and ranking. Embeddings need manipulable coordinates and a distance-unit switch. Candidate learning needs a mask and probability competition, not a “contrastive” label over an unrelated generic slider. Seven specifications provide these distinct jobs, with result checks, meaningful edits, reset/invalidation/accessibility and bounded states.

## Author checks and limitations

Executed loss-experiments.py through nine matched CPU fits and exact mechanism outputs. Read produced metrics and per-run confusion matrices. Separate bounded author calculation recomputed seed1/BCE thresholds .01/.1/.5/.9/.99 from stored probabilities and labels; checked gamma0 versus stable BCE at extreme logits and obtained zero loss difference; checked focal pt.9 slope ratio and shared-bias sums. Collapsed squared-triplet positive loss/zero gradient is in the saved record. A probe emitted a benign requires-grad-to-scalar warning while printing a comparison; it did not change data or the teaching program.

Full manuscript and specification author reread: checked formulas, units, targets, first-pass placement, actual-vs-derived claims, no fake superiority, masks, practice solutions, source annotations, sequence and local bridge. Compared conservation decisions against the full original. The code is a complete teaching artifact; it is not a finished browser integration. Clean-environment download replay, fuller extreme/shape/API parity tests, actual widgets, independent phase-two correctness review, accessibility/responsiveness/rendering and publication remain deferred. Root binds final packet hashes and ledger status after its scoped content review.

Retention: keep manuscript, design, specs, CSV, provenance, program and calculated JSON as the pending handoff. No disposable image, downloaded pretrained weights, new environment or scratch folder created. No shared ledger, source or generated publication artifact edited.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Watch which errors receive influence. Move observations, change the loss and focal gamma, drag decision thresholds, edit pair/triplet coordinates and change InfoNCE temperature. Update loss, signed gradients, fitted location, confusion counts, eligible negatives and candidate probabilities together. Keep score-based metrics distinct from threshold decisions. Choose an objective or operating threshold from the error tradeoff rather than from a single loss number.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Prepared implementation completed — 21 September 2026

The authorized first-five Deep Learning finish request consumes this complete packet. All manuscript sections, mathematics, examples, independent practice and annotated resources are preserved in the manifest-owned production body. The scoped author-time renderer replaces only explicit representation placeholders and inserts each specified visual in its explanatory context; it does not parse Markdown in the browser.

Topic-specific figures and immediate-result laboratories now implement all seven visual jobs. The final independent review found and closed the spatial-observation, marker-category, reduction-table, threshold/specimen linkage, gradient-dependency and residual-path gaps relevant to these two lessons. Fractional counts now have local invalid-state feedback; shared tables keep whole numeric tokens in labelled local scrollers. Native range/number controls, finite state, exact nulls, reset, screenshots and mathematical SVG geometry were tested at desktop, 390 and 320 pixels. No learner-prediction feature exists.

All measured training curves/decisions use the retained real digit experiment, never simulated benchmarks. The complete CPU program was executed beside its original CSV with Python 3.12.14, PyTorch 2.14.0+cpu and NumPy 2.3.5. The Loss nine-run and Normalization twelve-run JSON results equal their packet records. Three displayed small snippets also ran. Final source programs remain downloadable; their formatted in-page views load and mount only when opened.

Independent review: [Loss/Normalization review](../../LOSS-NORMALIZATION-INDEPENDENT-REVIEW.md). Author model evidence: [114 assertions and native replay](../../evidence/loss-normalization-models.json); snippet execution: [receipt](../../evidence/loss-normalization-snippets.json). Final production: [16 browser groups](../../evidence/loss-normalization-browser-production.json) and [shared route/loading/download integration](../../evidence/deep-learning-core-production-integration.json). The [batch completion record](../../DEEP-LEARNING-CORE-IMPLEMENTATION.md) owns the final scope/counts and continuation. Both phases are complete; user acceptance is separate.
