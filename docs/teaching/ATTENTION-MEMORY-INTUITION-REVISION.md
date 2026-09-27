# Attention, long-context memory and state space: intuition revision

26 September 2026. Full revision 4 of the three recently implemented lessons, in response to the user's feedback about intuition, buildup, diagrams and readability. Both phases are complete. This work does not implement RWKV or authorize a curriculum-wide rewrite. User acceptance remains pending.

## What changed

| Lesson | New route and explanatory representations | Depth preserved |
| --- | --- | --- |
| [Attention](revisions/attention-mechanism-bahdanau-luong/4/design.md) | A familiar spelling decision motivates accessible reading notes. Weather retrieval separates matching information from returned information. Six new figures explain repeated reads, mixtures, softmax intermediates, additive comparison, the padding denominator and signed learning feedback. The first live lab ends at the returned context; training outputs appear after they are introduced. A small runnable NumPy core precedes the complete implementation. | All seven investigations, canonical training/inference programs, measured results, scorer distinctions, exact gradient derivation, local/copying/speech branches and eight changed exercises. |
| [Long context](revisions/long-context-sequence-models-transformer-xl-griffin-perceiver/4/design.md) | An unavailable earlier record motivates memory choices. Four new figures explain the segment boundary, weighted shares, retention versus injection and two different paths with the same mean. XL, Griffin and Perceiver each begin with the problem their mechanism addresses, before shapes and formulas. | The sixteen previous figures, four investigations, complete scratch/library programs, six measured result rows, positional and causal distinctions, scan/gradient/budget detail and full practice. |
| [State space](revisions/state-space-models-s4-mamba-mamba-2/4/design.md) | A retained/new contribution lab introduces state before notation. Impulse trails explain convolution; marked events separate selection from fixed delay; a two-by-two write/read introduces SSD. Fast/slow summaries and rotating pairs build toward the harder operators. Three complete derivations have explicit deeper-reading branches. | Existing operators, four larger investigations, complete scratch/library programs, measured classifier experiments, S4/Mamba/SSD distinctions, newer variant coverage and full exercises. |

These are changes throughout the lessons, not introductory paragraphs placed in front of unchanged difficult explanations. Prose, numeric examples, drawings and code follow the same named quantities. New figures use the established charcoal/amber theme. Constructed examples are identified; drawn widths and arithmetic follow the stated scales.

## Research and teaching choices

The per-topic design records name exactly what was read and the limits of each review. Resources include [3Blue1Brown's 2024 attention article](https://www.3blue1brown.com/lessons/attention/), [Distill's recurrent-attention explanation](https://distill.pub/2016/augmented-rnns/), [D2L's retrieval-to-pooling progression](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html), creator explanations of [Transformer-XL](https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/), [RecurrentGemma](https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/) and [Perceiver](https://huggingface.co/blog/perceiver), the published source of [Annotated S4](https://srush.github.io/annotated-s4/), and the authors' [SSD model](https://goombalab.github.io/blog/2024/mamba2-part1-model/) and [algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/) explanations.

The useful common approach is a concrete need, a visible operation, and only then terminology and formal detail. Recent creator explanations complement older clear references; source recency is not a quality guarantee. Examples and diagrams were authored for these lessons rather than copied. No full-video viewing, external tutorial execution or new benchmark is claimed.

The [teaching standard](../../LESSON-TEACHING-STANDARD.md) now explicitly requires this buildup and an actual reading-order review. The handoff links the new current packets so another agent can follow the same approach without imposing identical visuals or section templates on every subject.

## Checks and review

- Fresh topic checks validate the new arithmetic, the runnable attention core, masks, nine NumPy/Torch attention comparisons, retained lesson depth, parsed components and source identities.
- Independent reviewers read each revised lesson and its new diagrams. Closed findings include attention's query-dependent repeated-read wording, long-context parameter sharing versus recomputed attention weights, a stale manuscript route and a state-space matrix caption. Their attributed records remain inside each revision packet.
- The integration owner inspected the actual production pages at desktop, 760px and 320px widths. New figures, changed controls, resets, a keyboard interaction and a deeper-reading disclosure were checked. Two attention phone-readability findings were fixed and re-inspected. The [browser receipt](evidence/attention-memory-intuition/browser-review.json) records precise observations and retained screenshots.
- Production build, learning-artifact, curriculum, site-build and ledger checks passed. The [integration receipt](evidence/attention-memory-intuition/integration.json) checks current source-bound completion, exact preservation of 174 other ledger rows, all publication mappings and the module sequence. Each lesson remains outside the initial app/reader static import closure, and public downloads match the build.

The numerical engines, previous fitted data and complete original programs were not changed or retrained. Prior native evidence is reused only after exact source checks. Optional GPU/package execution limits remain as stated in the earlier records. Independent reviewers had no browser surface; the root browser inspection is separately attributed rather than presented as their work. No human beginner trial or universal mastery guarantee is claimed.

## Reproducible bounded verification

Run the selected topic's verifier after changes to its source. For the integrated revision:

    scratch/lesson-tools/Scripts/python.exe scripts/verify-recurrent-attention-intuition.py
    node scripts/verify-long-context-teaching.mjs
    node scripts/verify-state-space-teaching.mjs
    node scripts/verify-attention-memory-intuition.mjs

The last command needs a current production build. It writes only this revision's integration receipt, not historical evidence or completion rows. The previous revision's delivery verifier retains its original scope and should not be used to overwrite old results against revision 4.

Recorded totals remain 177 content-complete, 153 implementation-complete and 24 prepared implementations pending. The source-identity inventory still distinguishes the three current checkpoints from historical rows whose exact bytes were not recertified in this task. Continue only from a future scoped request.
