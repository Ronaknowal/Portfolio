# Interleaved / Cross-Attention: implemented teaching map

The entire current ten-section manuscript, eight changed practice problems, complete scratch/study program and native library cache bridge are retained. This is an author implementation record; independent review, real browser checks and shared integration remain separate.

## Concept-by-concept locations

| New hurdle | Visible reason and worked intermediate | Representation and actual code location |
| --- | --- | --- |
| Different questions need different reads of one image | Opening bicycle example; §1 keys distinguish relevance from returned values | `CrossFigure('read')`, `ReadContributions`, score/weight/product table |
| A memory of S entries returns T answers | §1 two queries, three keys, exact [1.5,1.5] and [1,2.5] | Connected reader→slots→sum fan with normalized path weights, signed coordinate contribution bars |
| Masks are information boundaries | §1 removed donor renormalizes first answer; forbidden [100,-100] edit changes only second | `CrossReadLab`: all Q/K/V coordinates, row-specific availability, empty-row undefined state, pinned outputs |
| Key/value pairing is part of the representation | §2 paired permutation preserves output; value-only edit changes correspondence | Same live editor's paired K/V+mask reversal and value-only contrast |
| Counts and widths are independent axes | §2 projection dimensions, 4-head T7/S11 changed example | `CrossShapeLab` / `ShapeTiles`: rectangular weight tile, live projection/concatenation dimensions |
| Input order does not specify operation or block depth | §3 projected prefix versus separate memory, independent block insertion | `CrossAccessLab`: moveable event ribbon, shared-stream causal matrix, rectangular direct-access matrix, separate vertical eight-block ruler |
| Recent direct access can coexist with older indirect influence | §3 earlier text carries context | Solid direct and dashed earlier-text paths from actual `accessGraph`; recent/all policy controls |
| A final causal mask cannot repair bidirectional upstream leakage | §3 whole-clip caveat and §7 dependency analysis | Explicit encoder-dependency matrix and future-source report, independent/causal/bidirectional choices; `CrossFigure('dependency')` |
| Attention context and supervised targets are different masks | §3 red/bicycle shifted input and output | `TeacherForcingFigure`: answer-start→red and red→bicycle paths, alternate three/birds sentence, position selector, independent question-loss switch |
| Learned slots compress before later text fusion | §4 4×16 read, anonymous slots, repeated reads | `CrossFigure('resampler')`: all-to-all feature/read paths into four anonymous latent outputs, then a later query; links actual long-context owner |
| Compression can erase distinctions no later query can recover | §4 [1,3] and [2,2] mean collision, maximum/spread differ | `CrossCompressionLab`: two parallel input→compressor→state flows, live equal-state/max/spread table and ordered two-slot repair |
| Zero residual change is not zero gate gradient | §4 scalar squared loss, alpha0/F2 gives derivative -4 and next alpha.4 | `CrossGateLab`: residual bypass, gated rail, derivative path, exact current/proposed step table; F0 and rate0 nulls |
| Frozen parameters can transmit input derivatives | §4 frozen backbone still trains preceding adapter | `CrossFigure('frozen')`: tracked frozen graph and explicitly cut no_grad graph; actual native autograd check |
| Named model recipes differ in connector and trainable stage | §5 dated primary-source table | Four `named` circuits with stage notes: Flamingo gated memory; BLIP-2 projected queries; original LLaVA linear prefix; Idefics2 pooled prefix |
| Shared Q-Former attention masks implement different objectives | §5 contrastive, matching and generation | `CrossFigure('blip')`: three complete row-receiver/column-donor matrices, query-only visual cross-attention stated separately |
| Ordinary trainable cross-attention differs from a pretrained assistant | §6 complete image/question tensor construction and PyTorch program | `CrossFigure('pipeline')`: measured counts8×8→16×4→16×24, query1×24, three heads, residual FFN and 12 outputs |
| Split the underlying image before deriving two tasks | §6 240/80/80 images then 480/160/160 question records; provenance and source IDs | Full executed `cross-attention-study.py`, retained split identities and exact reproduction evidence |
| Useful controls test whether the image matters | §6 question-only48/160, weak cyclic70/80 labels, stronger shuffled9/80 digit and40/80 parity matches | `CrossObservedWorkspace`: actual source/question/head maps, raw-pixel linkage, 12 probabilities, aggregate transformation controls and all nine result rows |
| More elaborate architecture need not win | §6 every run/seed, parameter count and task-specific outcomes | `CrossFigure('results')`: nine unconnected measured outcomes on common0–1 axes plus complete counts; measured fit/development curves only |
| Pair count is not latency or retained storage | §7 dense versus permitted causal formulas | `CrossFigure('pairs')`: exact illustrative two-visual/three-text masks (15 versus12 causal pairs); `CrossBudgetLab` evaluates large actual dimensions |
| Prefix and cross caches differ in declared layers, token counts and widths | §7 exact72MiB/2MiB/18MiB configurations | Parallel tensor byte bars, raw byte/MiB formula table, raw-feature width and recompute controls; equal-dimension null |
| A memory projection can be reused while each query remains new | §7 complete native MHA bridge | `CrossCacheLab` port: stored K/V, three query rows, source/model identity, fresh/cached/original answers, query-specific head weights |
| A null answer does not prove a cache identity is valid | §7 forbidden donor and source revision intervention | Query0/1 remain fixed after last donor edit; cache remains rejected until explicit rebuild; paired-order/mask reversal preserves fresh answer |
| Architecture choice follows failure cases | §8 image detail, video availability, repeated questions, frozen training, structured outputs | Full decision table and explained boundaries remain prose adjacent to the earlier working mechanisms |
| Transfer requires new inputs and constraints | §9 eight problems and §10 annotated sources | Closed hints/solutions retained; changed read, shapes, teacher forcing, gates, grouped split/control, cache, collision and unsupported-claim critique |

## Deliberate adaptations

The static first read stays inline; inexpensive editors are immediately available rather than hidden. Two matrix widths remain exactly two in the arbitrary read lab; the shape calculator independently varies projections and heads. Values directly support the explicitly specified ±100 contrast. A wholly masked row is shown as undefined with no fabricated products. Cache identity is explicit caller-managed source/parameter versioning, not tensor hashing.

Native trained inference was historically not retained, so the complete nine-fit program was executed once to reproduce every original result exactly. The inspector adds first-assessment-source-per-digit, both questions, all cross/gated seeds, chosen independently of correctness. All 120 outputs are recorded, not arbitrary neural inference; live arbitrary entities are supported in the exact mechanisms. Only the observed JSON and requested program texts load on expansion. A five-file learner download allowlist excludes author manuscripts, designs and review evidence.

Persistent byte categories use comparable parallel bars and an exact sum table rather than a stacked-only view. The small pair diagrams state their fixed dimensions; the adjacent calculator does not allocate millions of marks. Nominal 440px mechanism diagrams and 340px plots use local, focusable horizontal scrolling so 12px labels are not scaled down on phones. Colors convey labelled roles only. Teacher forcing keeps correct alignment visible while exposing its predictor context and independent loss mask.

## Author learning-experience check

The physical question/memory need precedes notation; every new operation has its own worked values and a representation at its explanation. Essential reasoning remains visible, while complete programs and optional actual-output browsing use disclosures. All controls recompute current results immediately and invalid boundaries remain explicit. Meaningful changes and nulls have practical interpretations rather than correctness badges. Saved curves preserve observed epochs only. All nine fits, unfavorable results and weak-control evidence are retained. No unsupported latency, GPU performance, pretrained execution or causal-attribution claim is made. No learner prediction entry or answer unlock is present.

## Author verification

Actual Python3.12.14 / NumPy2.3.5 / PyTorch2.14.0+cpu, one thread: all nine original fits reproduced byte-equivalent parsed results; complete native cache/library call and streamed query outputs differed by at most5.56e-17. Eighteen new NumPy reads, five autograd gate fixtures and frozen/no_grad derivative tests executed. JavaScript verification passed1,364 scalar comparisons (maximum2.775e-7 from float32 saved normalization), native cache projections/outputs, source identity, permutation/mask nulls, byte accounting and source/deployed equality. Eleven figure kinds, eight labs, the loaded observed workspace and whole ten-section body rendered successfully. Browser/network/keyboard/painted geometry and independent numerical/learning review are intentionally not certified by these author checks.
