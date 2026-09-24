# Typed decision model: independent depth review

Reviewed 22 September 2026. This review covers the added project explanation, contextual topic links, live mechanism labs and downloadable research/library tools. It is separate from the original runnable-core review and from the final production-browser review.

**Result:** no unresolved material content or numerical blocker was found in this scope. The expanded guide now explains the request-to-token-to-tensor-to-score-to-gradient-to-action chain locally, rather than asking prerequisite links to supply missing project steps. The project is an original, small, inspectable decision model. It remains neither a Jev reproduction nor evidence of useful pretrained Laya/ModernBERT performance.

## Teaching and source review

- Read `content.jsx`, `mechanism-depth.jsx`, `research-depth.jsx`, `mechanism-labs.jsx`, `mechanism-models.js` and `project-elements.jsx`, and compared their mechanisms and claims with all three downloadable Python programs.
- The eight stages retain one routing example and distinguish semantic IDs, current candidate indices, vocabulary IDs and sequence positions. The default 35-token request, marker positions 8/15/22, two distinct masks and gather shapes agree with the executable core.
- The architecture explanation correctly separates token communication through attention from token-wise feed-forward transformations, identifies pre-normalization/residual paths, explains axis permutations, and states the relevant quadratic score-matrix cost without pretending that it alone predicts runtime.
- Training connects stable cross-entropy to score derivatives, shared-parameter derivatives, autograd, gradient accumulation, clipping and AdamW. The independent-logit example and frozen-feature SGD lab are explicitly distinguished from a whole-transformer AdamW update. The gradient cap is not described as a parameter-update cap.
- Calibration and evaluation retain the negative results, denominators, constructed-data scope, calibration/development/test roles, and the distinction between expected and realized costs. Positive temperature preserves score order; it does not manufacture new evidence or guarantee out-of-sample improvement. Empty reliability bins and zero-coverage selective risk are represented as undefined rather than zero performance.
- The ordinary-library bridge is executable and explained: matched tokenizer/encoder, special-marker registration and embedding resizing, masks, marker gather, custom head, loss, parameter groups and artifact restoration. It explicitly explains that freezing the encoder also freezes the newly added marker embedding. The real pretrained checkpoint experiment remains unexecuted; offline architecture tests establish API mechanics only.
- The code viewers extract from the actual downloadable source. All 17 displayed source excerpts have unique existing start anchors, correctly ordered end anchors and nonempty bodies. The project contains 31 contextual topic links to 24 distinct catalogue entries. Twenty-one entries are published; contextual embeddings, model serving and monitoring are labeled planned in the UI. The separate typed-decision companion also remains explicitly planned. No topic completion was inferred from adding links.
- Four distinct interactive purposes are present: probability/cost exploration, encoding/marker tracing, a masked attention mixture, and a shared-scoring-head update. They use immediate changes with no learner prediction gate. Illustrative vectors/scores are labeled; the token fixture alone comes from the actual program. There is no claim that the browser runs the trained transformer.

No production author source was edited by the reviewer. The complementary checks below did not uncover a defect that required an author correction. The review did not expand into implementing the clearly marked RL, real-data, service or pretrained-quality experiments.

## Complementary executed checks

Retained executable: [verify-typed-decision-depth-review.py](evidence/verify-typed-decision-depth-review.py). Results and full source hashes: [typed-decision-depth-independent.json](evidence/typed-decision-depth-independent.json).

These checks use different constructions where practical, rather than calling the author's verifier again:

| Check | Executed result |
| --- | --- |
| Shared-head derivatives | Central finite differences for both shared weights, four weight vectors and three targets: 24 coordinates, maximum absolute error 1.8554e-10. Zero learning rate preserved both weights. |
| Attention arithmetic | All 41 slider positions from −2 to 2, with padding enabled/disabled: 82 states. Reconstructed by PyTorch matrix multiplication and softmax, matching JS probabilities and weighted outputs to 1e-14 tolerance. Masked padding always had exactly zero probability. The zero query has uniform weights over valid keys. |
| Browser/native representation | 15 requests covering candidate reorder/removal, case, punctuation, numerals mixed with words, unknown words and sequence overflow. IDs and markers agree with Python; over-budget input is rejected. The billing target follows the candidate ID. |
| Policy accounting | Seven review-cost settings on six hand-chosen distributions including exact confidence boundaries and probability 1. Independent vectorized cost selection reproduces acted counts, coverage, errors divided by acted cases, undefined risk when none act, and cost averaged across all cases. |
| Reliability accounting | Values on bin boundaries and at probability 1 are included exactly once; empty-bin means remain null; count-weighted ECE agrees with a separate sum. |
| Adapter identity and state | Marker registration is idempotent and its ID survives save/load. A frozen encoder with nonzero configured dropout remains deterministic. Mixed-batch padding and candidate-ID renaming preserve valid outputs. Exact local encoder/tokenizer/head restoration agrees before and after serialization. |
| Rejection behavior | Reserved marker injection, one candidate, duplicate IDs and excess length rejected; corrupt zero-temperature artifact rejected. |
| Real reported checkpoint | Read the existing 600-step artifact; checked its weight hash and independently reconstructed confusion via vectorized bincount, high-confidence-bin means and realized policy costs. No retraining or threshold selection was performed. |

The retained checkpoint recomputation confirms:

| Input probabilities | Acted | Errors among acted | Reviewed | Realized cost/request |
| --- | ---: | ---: | ---: | ---: |
| Raw | 15 | 4 | 3 | 2.3888888889 |
| Fitted temperature | 18 | 6 | 0 | 3.3333333333 |

Both confusion matrices are `[[6,0,0],[3,3,0],[3,0,3]]` in billing/access/delivery order. The raw 0.8–1.0 bin contains 16 cases with mean confidence 0.988501045426. These values agree with the displayed explanations and the retained author diagnostics. The assumed reviewer is perfect and has fixed cost; the guide correctly identifies that assumption.

The adapter checks ran on PyTorch 2.14.0+cpu and the isolated Transformers 4.57.6 environment. They used only a tiny random local BERT and local tokenizer. The author's separate [native record](typed-decision-model-native-depth.md) also records bounded random-ModernBERT architecture checks. Neither record establishes accuracy or calibration of a pretrained checkpoint.

The code/link verifier is retained as [verify-typed-decision-depth-links.mjs](evidence/verify-typed-decision-depth-links.mjs), with its [results](evidence/typed-decision-depth-links.json). It checks a minimum inspected node count so missing matches cannot vacuously pass. Both verifiers write a provisional nonpassing record before assertions, then mark completion only after success. Invalid input and corrupt-temperature cases actually exercise rejection; the review does not claim a broad source-mutation falsification campaign.

## Reproduction and source boundary

From the repository root, in an environment containing the declared dependencies:

```text
python -B docs/teaching/projects/evidence/verify-typed-decision-depth-review.py
node docs/teaching/projects/evidence/verify-typed-decision-depth-links.mjs
```

To reproduce the checkpoint-specific comparison, first create the documented 600-step seed-7 artifact with the recorded runtime, then append `--artifact PATH` to the Python review command. Its weight hash must match the retained diagnostics; this prevents an unrelated run from being treated as the reviewed run. The temporary review artifact can be removed after the result has been retained. The verifier itself uses automatically removed temporary directories for adapter serialization.

Exact reviewed hashes are stored in the two JSON evidence files. If a body changes after this review, distinguish the changed teaching/integration scope from unchanged mathematical evidence rather than relabeling old checks as execution of the new source. A substantive change to tokenizer/model/math/native files requires affected checks to run again.

The final integration edits were read separately: the training prose now explicitly distinguishes the native trace's 24-valid/11-pad second row from the diagram's 28-valid/7-pad row; the live table cells gained matching `data-label` values and phone stacking; the step counter now uses singular wording for one step. Their final hashes are under `final_integration_review.source_sha256` in the independent evidence. Mathematical and Python sources are unchanged, so their native results were retained without redundant reruns. The source/link verifier was rerun on the final text and still passes at 17 excerpts and 31 contextual links. CSS was inspected for label correspondence; final responsive paint remains the parent's browser check.

## Limits and integration handoff

This is a bounded content and computational review, not a guarantee about every possible model/input. The supported UI cannot create an all-masked attention row: valid requests always contain actual fields and candidate markers. Behavior of bare attention kernels on wholly invalid tensors is outside that contract. The native diagnostic confusion matrix is specific to the fixed ordered fixture labels; the prose explicitly warns that new variable-label datasets must aggregate semantic IDs rather than slot numbers.

Final paint, mobile layout, keyboard/pointer interactions, async source/fixture recovery and production loading are the integration owner's responsibility and must be linked from the final project record. No browser pass is claimed by this document. The separate depth companion remains a planned curriculum topic; existing lesson phase counts must not be changed for this project revision.
