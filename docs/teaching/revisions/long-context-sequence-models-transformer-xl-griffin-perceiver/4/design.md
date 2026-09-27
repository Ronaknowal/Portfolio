# Long-context sequence models — revision 4 teaching redesign

Date: 26 September 2026. Stable curriculum ID: `long-context-sequence-models-transformer-xl-griffin-perceiver`. This revision responds to the user's finding that the completed three-lesson sequence remained too abrupt and research-like for a first-time learner. The revision-3 draft, measured programs/data and `docs/teaching/LONG-CONTEXT-IMPLEMENTATION.md` are preserved. Root owns the delivery ledger, shared standards and final integration.

## Diagnosis and learner journey

The previous text supplied correct mechanisms and functional investigations but often introduced their representation before their purpose. It began with several kinds of vectors and capacity, put layered XL recurrence before a concrete lost record, introduced RG-LRU gates as equations before the surprising effect of silence, and opened Perceiver with N×T arrays before showing a distinction one average cannot retain. The program section described a complete implementation without sufficiently walking a novice through why its stages occur in that order. Correctness and component counts were insufficient evidence of a coherent first reading.

Revision 4 rebuilds the main route, not just its opening:

1. A visible entrance note falls outside the two-message window. Ask what information remains accessible, before naming model families.
2. Familiar temperature questions distinguish keeping each record from keeping a sum/count. Define representation, vector, cache, recurrent state and latent in that order.
3. Calculate a 2:1:3 mixture of values 2,4,8 with six shares. Only then introduce scores, projections, softmax, masks and matrix shapes.
4. Show a split sentence losing context; trace cache lengths 0/2/4 through actual records. Put layers, detach and relative-position math after this mechanism. The live control instructions name edits and the observation each edit explains.
5. Explain why an old .6 state decays to .48 even when new input is zero. Separate closing input from changing retention before deriving the gates and square-root injection. Explain the full block's temporal convolution and local read in relation to those memory paths.
6. Show two five-point paths with the same center. Introduce learned requests, data-dependent updated latents, input rereads and array sizes. Derive the 4:2:1 and 1:2:4 supports before their log keys. Connect tags to the concrete distinction between reordered storage and reordered time.
7. Begin Perceiver AR from the difference between a finished recording and an unknown next token. Trace a two-hop causal leak, not just two masks in isolation.
8. Follow five NumPy attention actions; then explain fit/validation/test roles, simple representation baselines, ordinal tags, logits, residual updates and how the loss teaches starting queries. Interpret individual measured outcomes without replacing them with a success narrative.
9. Connect explicit operations to matched maintained library operators. Retain deeper gradients, budgets, affine scan, evaluation and all changed practice problems. Add a concrete two-step affine instruction before its notation.

First-pass prose precedes notation; advanced detail remains in its relevant subsection or the explicit deeper section. Figures have a question to inspect before them and an interpretation after them. The live labs remain immediate and fully editable, without a prediction gate. No extra interaction was added merely to make a static explanation appear playable.

## Research actually consulted for this revision

These are primary creator or implementation-author resources, inspected on 26 September 2026. Historical architecture explainers are appropriate to these architecture versions. Their age is made clear by retaining paper-specific settings; no historical speedup is presented as a current benchmark. The pedagogical synthesis and new worked examples are original to this lesson.

| Resource and inspected portion | Teaching idea used | Boundary |
| --- | --- | --- |
| [Google Research: Transformer-XL](https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/), 29 Jan 2019, context fragmentation, segment recurrence and relative-position illustrations | Begin at a broken segment boundary; motivate position ambiguity only after carrying a record across it | Do not reuse benchmark claims or the article's wording |
| [Google Developers: RecurrentGemma architecture](https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/), 29 Aug 2024, recurrence/attention blocks, two branches and dimensions | Explain what the recurrent branch and local window separately preserve, then expand one block | RecurrentGemma released settings differ from original Griffin; no universal window claim |
| [Griffin paper HTML](https://arxiv.org/html/2402.19427v1), §§2.3–2.4, Griffin construction and Appendix A | Recheck input-dependent gates, exponent scale, coupled retention/injection and the hold limit; explain them from one silent update | Finite sigmoid outputs approach r=0 but do not equal it; variance argument remains conditional |
| [Hugging Face: Perceiver IO](https://huggingface.co/blog/perceiver), 15 Dec 2021, architecture introduction and input/latent/output role walkthrough | Make the identity of the query source determine the output rows; explain requests before shapes | Linked notebooks were not executed for this revision |
| [DeepMind: Building architectures that can handle the world's data](https://deepmind.google/blog/building-architectures-that-can-handle-the-worlds-data/), 3 Aug 2021, input arrays, latent workspace and output queries | Keep input size, working space and requested output locations conceptually separate | Linked audiovisual demonstrations were not watched; no watch claim |
| [DeepMind: Perceiver AR](https://deepmind.google/blog/perceiver-ar-general-purpose-long-context-autoregressive-generation/), 16 July 2022, autoregressive task and aligned latent explanation | Start causal restrictions from what an output is allowed to know, then trace the indirect leak | No contemporary performance comparison inferred |
| [Hugging Face Perceiver documentation](https://huggingface.co/docs/transformers/en/model_doc/perceiver), API page opened as a current orientation reference | No new API behavior adopted from this opening alone | This is not an executed library-version verification |

Prior canonical papers, UCI provenance, native runs and API receipts remain the basis for existing scientific and executable claims. Annotated references now include the added creator explanations and implementation tutorial. The prior ICML slides/video entry retains its honest playback boundary.

## Coverage conservation

| Existing depth | Revision 4 disposition |
| --- | --- |
| Attention score, softmax, mask, Q/K/V shape | All retained; weighted shares and term definitions now precede notation |
| XL layered memory, stopgrad, indirect receptive field, four-term relative score | Retained after cache trace; scalar recency bias still explicitly not the full XL score |
| RG-LRU equations, conditional variance reasoning, hybrid convolution/attention blocks, bounded-state limits | Retained and explained from state retention; no substitution of a Mamba recurrence |
| Perceiver repeated reads, position/Fourier features, IO output queries, AR two-mask causality | Retained; simpler concrete questions introduce them |
| Three complete source programs, imports, NumPy scratch and normal Torch paths, optional Griffin parity boundary | Same canonical bytes and lazy source readers; no retraining or library migration |
| UCI normalization, duplicates, data roles, fit settings, six measured rows and all four models | Unchanged; more explanatory context around procedure and interpretation |
| Sixteen existing figures and four live labs | Retained; four focused static explanatory figures added |
| Deeper derivatives, exact budgets/bytes, associative scan, evaluation protocol | Retained; budget and affine instruction examples improve entry into formal detail |
| Eight changed exercises with hints/solutions, plus library bridge exercise | Retained; only the bridge exercise's cramped numeric spacing repaired |

The generator reads the new `lesson.md`; its figure substitutions and complete-program additions remain deterministic. Existing small runtime operations, frozen model export, canonical Python, dataset and saved fits are unchanged. Current runtime files retain semantic names. No new model package, browser training loop or always-loaded model asset was added.

## New visual contracts and evidence

`visual-specifications.md` defines the four new diagrams and retention contract for the existing visual/lab system. The runtime owns new examples in `src/learn/data/long-context-intuition.js` and diagrams in `LongContextIntuitionFigures.jsx`; scoped CSS uses actual article-container width so a sidebar cannot squeeze paired panels into unreadability.

`node scripts/verify-long-context-teaching.mjs` writes a new source-bound `teaching-checks.json` in this revision. It checks new numerical fixtures, retains all former code/result blocks, displayed equations, measured rows and changed practice, parses the authored JS/JSX, and verifies the existing native/independent receipt hashes against unchanged mechanisms/assets. It does not overwrite revision-3 receipts, rerun fitting, claim browser review or substitute count checks for a learner review.

The complementary State Space author independently read the complete revised manuscript and new components. Three findings were closed: distinguish shared projection parameters from recomputed attention weights; put the valid Self-Attention route directly in the packet; repair a concatenated source-inspection date. Its source-bound receipt is `evidence/independent-review.json`, with no open findings within that review scope. In turn, this author reviewed the full State Space revision and its new fixtures; that receipt lives with the State Space revision.

This author's unavailable CUA surface is recorded in `evidence/browser-review.json`; it is an unsuccessful attempt, not the final rendered review. The integration owner subsequently used its working browser bridge to inspect the current production build at port 4194. All four new figures were inspected on desktop, paired paths were checked stacked at 760px, and the phone path/timeline renderings and all four figure bounds were checked at 320px. Increasing retained memory from 2 to 4 made R0/R1 available and changed the existing cache lab's output from 3.33333333 to 5; reset restored the original result. No revision-3 screenshot is relabeled. The [root browser receipt](../../../evidence/attention-memory-intuition/browser-review.json) closes the pending rendered checks. Both revision-4 phases are complete; see the [shared integration record](../../../ATTENTION-MEMORY-INTUITION-REVISION.md). User acceptance remains distinct from an internal review pass.
