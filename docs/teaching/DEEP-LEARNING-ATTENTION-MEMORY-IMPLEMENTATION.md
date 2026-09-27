# Attention and sequence memory: completed implementation

26 September 2026. User request: implement the next three topics, end to end.
Actual module order was preserved:

| Position | Topic | Current revision | Delivery result |
| --- | --- | --- | --- |
| 16 | Attention Mechanism (Bahdanau, Luong) | 3, content-first continued | Both phases complete |
| 17 | Long-Context Sequence Models (Transformer-XL, Griffin, Perceiver) | 3, content-first continued | Both phases complete |
| 18 | State Space Models: S4 and the Mamba Family | 3, content-first continued | Both phases complete |

These are implementation/review completions, not user approvals. No commit, push
or deployment was requested or performed. The next module topic is **RWKV & Linear
Attention Models**; its prepared packet is unchanged and not part of this request.

## Retained teaching and verification

- [Attention implementation](ATTENTION-IMPLEMENTATION.md) and its
  [separate independent review](ATTENTION-INDEPENDENT-REVIEW.md): complete prose,
  scratch/native code, eight figure families and seven immediate investigations.
  Six saved models reproduce their development outputs; fresh NumPy/native
  probability difference is at most 2.30e−7.
- [Long-context implementation](LONG-CONTEXT-IMPLEMENTATION.md): complete prepared
  prose, sixteen figure families, four investigations, scratch and native-library
  routes. The browser model matches 200 saved validation cases; the native route
  reproduces all 1,440 model/source cases. Independent fresh masked/edited cases
  agree within 4.04e−6.
- [State-space implementation](STATE-SPACE-IMPLEMENTATION.md): complete prepared
  prose, twenty-four visual mechanisms, four investigations and three complete
  program routes. Native/browser inference matches 108 cases within 7.14e−6 in
  logits. Independent direct/FFT and state-continuation checks complement this.

Both latter authors performed complementary independent reviews of each other's
topic. Their records distinguish authorship, independent numerical/source review
and rendered author review. No scoped finding remains open. Optional RecurrentGemma
package parity and the CUDA Mamba bridge remain explicitly unexecuted; neither is
represented as an executed production-model comparison.

All three were inspected on the production preview at desktop, 320px and 760px
widths. Real pointer/keyboard/numeric controls changed computed results, including
actual fitted-model inference. Diagram and lab repairs included preserved pointer
grab offsets, clear branch convergence, readable numeric matrices, window labels,
and container-based stacking when the sidebar reduces the article width. No
prediction-entry exercise gates were introduced. Screenshots were inspected during
the sessions; retained browser observations state their coverage and limitations.

Root also used the actual Previous/Next buttons to traverse Attention → Long Context
→ State Space and back. Module context remained `deep-learning-fundamentals`.
The preceding topic is Sequence-to-Sequence; State Space's next topic is RWKV.

## Source conservation and integration

[The baseline](evidence/attention-sequence-baseline.json) retains the three original
entries, all 177 row identities, previous publication mappings and the exact ten
line-ending restorations. All three prepared content checkpoints, delivery modes,
revision numbers and earlier revision histories are unchanged. Existing publication
mappings are preserved; Long Context adds one new published mapping, giving 232.
No pending packet was removed. No unrelated lesson body was rewritten.

The global curriculum check exposed two older metadata errors: Depthwise referenced
a non-canonical Landmark Architectures title, and ConvNeXt used obsolete scalar
visual/practice fields plus `references` instead of `sources`. The bounded repair
uses the exact existing title and current structured blueprint schema. Its learning
outcomes and already implemented mechanisms remain the same. Only each blueprint's
reviewed hash changed in those two historical rows; their content, revision, phase
status, dates and other reviewed files are untouched. The other 172 ledger rows
are byte-equivalent as serialized entries to the baseline.

`node scripts/verify-attention-memory-delivery.mjs` passes and writes
[the source-bound integration receipt](evidence/attention-memory-integration.json).
It checks exact current completion for all three topics, unchanged prepared/history
fields, bounded metadata changes, preserved publication mappings, module sequence,
nine current evidence receipts, lazy chunk boundaries, built/download identity,
local provenance links and absence of Python caches in the new public directories.

The production build passes. The three separate lesson chunks are approximately
32.4, 35.8 and 50.9 kB gzipped respectively, excluding shared dependencies and
on-demand parameters. They are absent from the static app/reader import closures.
Complete model parameters and long program text are fetched only when requested.
The existing shared-chunk size warning remains visible; these sizes are build
measurements, not browser-speed claims.

Also passed: `verify-learning-artifacts.mjs` (1,461 stable topics, 29 modules, 232
publication mappings, 486 separate outlines), `verify-curriculum.mjs` (639 briefs,
10 paths, preserved identities/order), `verify-lesson-delivery.mjs` (eight behavior
groups), and `verify-site-build.mjs` (section isolation and deployment artifacts).
The inventory was regenerated. The original development server stalled even on
the home route; it supplied no evidence. The verification preview used port 4196.
At delivery, the same built artifact was opened successfully on the user's usual
port 4194, with Attention's seven labs mounted; the temporary 4196 preview was stopped.

## Current versus historical status

Recorded totals are 177 content-complete, 153 implementation-complete and 24 pending
implementations. The strict source-identity inventory on this Windows checkout
reports three current content/implementation checkpoints. The other historical
rows were already outside this scoped source restoration; they are not silently
rehash-approved here. Before the three finish requests, their ten mismatched text
files were proven to equal the existing hashes after CRLF→LF normalization, then
restored to those exact bytes. This is not permission for a broad historical rewrite
or a new audit queue. Apply the same evidence-based distinction only to the next
selected topic when its preflight requires it.

The lesson records and ledger now contain the resumable state. Preserve their
pending source packets and evidence; continue RWKV only under the next scoped request.
