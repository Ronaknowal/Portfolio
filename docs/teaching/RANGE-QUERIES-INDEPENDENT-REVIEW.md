# Range Queries — independent finite review

Reviewed 10 September 2026 by an agent who did not author this lesson. Scope: the complete current lesson, its range/deque pure models, all thirteen exported Python examples, practice data, individual design, incoming-note resolutions and author verification record. The exact topic-plan command was run. The initial review was read-only; the coordinating agent subsequently authorized the one trace-message correction recorded below. No other lesson, model, lab, shared registry or curriculum source was edited by this review.

**Result:** no algorithmic correctness defect found within the stated contracts. One small learner-facing trace-message inconsistency was corrected and verified, below. Finite checks support the implementations; the independently read invariant arguments are still necessary for arbitrary inputs. This record does not replace the author's actual browser/visual review or the root production integration, and does not claim user acceptance or full video viewing.

## Actionable wording finding

At the reviewed snapshot, `src/learn/data/range-query-models.js:415` emits “Update its sum now and compose a tag for children” for every fully covered update, including a leaf. For example, `lazyRangeOperation(lazyState([2]), 'set', 0, 1, 5)` correctly leaves the multiplier/addition at their identity because a leaf has no descendants, but its apply-frame message says otherwise. The lesson explicitly teaches the correct rule at `segment-trees-fenwick-trees-range-queries.jsx:117`.

Suggested local correction: branch the message on `high - low > 1`; keep the internal-node explanation and say that a leaf updates its sum directly and needs no pending child tag. This is a trace explanation issue, not an incorrect stored sum, tag composition or query result. The coordinating agent received the precise line and owns the correction and its targeted recheck. No unrelated algorithm rerun is required solely for changing that sentence.

### Resolution and final fingerprint

The coordinating agent authorized this narrow source-freeze exception. The message now branches on interval length: internal nodes retain the child-tag explanation; leaves say “Update this leaf’s sum directly. It has no children and needs no pending tag.” Only this message selection changed; update/tag/trace-state arithmetic is unchanged.

`node scripts/verify-range-query-models.mjs` passed again after the edit: 320 arrays, 17,838 intervals, 8,000 lazy operations, 13,116 deque cases and 800 weighted cases. The targeted browser check `node scratch/review-range-leaf-message.cjs` passed **1440px and 390px at 10:09:14 UTC**. It confirmed the internal-node message, applied a singleton set to 5, used keyboard Enter to reach the real leaf apply frame, and verified the revised explanation, stored/logical value 5, absent leaf tag, no page errors and no page overflow. Both saved screenshots were actually opened: the complete message wraps readably, the leaf is visible and the focused stepping control is clear.

- [Targeted browser script](../../scratch/review-range-leaf-message.cjs)
- [Final browser result and fingerprint](../../scratch/range-queries-independent-review/leaf-message-results.json)
- [Desktop screenshot](../../scratch/range-queries-independent-review/leaf-message-1440.png)
- [390px screenshot](../../scratch/range-queries-independent-review/leaf-message-390.png)

Final `range-query-models.js` SHA-256: `5944d2dd0ca621c8cca9e78a5115317a2a4435047cb28bc54264a30d1053877e`. The earlier four-source snapshot below documents the pre-message-review source; only this model fingerprint is superseded. Source is frozen again after the authorized correction. The initial targeted browser attempt used `innerText` on SVG text and failed in the harness; changing that assertion to `textContent` produced the successful real-browser evidence above, without an application-source change.

## Independently assessed arguments

| Contract | Review conclusion |
| --- | --- |
| Ordered segment folds | The left accumulator appends; the right accumulator prepends. Their order matches concatenated original intervals. Padding/empty identity, pure immutable summaries and bounded-combine costs are explicit. Noncommutative matrix products additionally test ordering beyond the author's string examples. |
| Frequency rank selection | At each binary-lifting bit the accepted prefix is aligned, the next Fenwick cell owns the tested next block, and a skipped block contains fewer than the remaining rank. Nonnegative counts and a valid one-based occurrence rank establish the monotone search. Zero-count buckets, non-power-of-two lengths and very large exact integer counts work. Empty/zero-total inputs reject all ranks before computing a highest bit. |
| Difference weighting | Each difference at j appears in exactly t−j actual values of a prefix of length t, yielding tΣd−Σjd. The boundary at n is stored for updates but excluded from valid actual prefixes. |
| Lazy set/add | The new map acts after the old: `(m,a) after (p,b)` has coefficients `(mp,mb+a)`. Own tags are already represented in own sums; ancestors can still defer actions to raw descendant storage. Push changes storage ownership, not logical values. Padding remains outside legal update ranges. Independent sequences intentionally avoid intervening leaf queries so that pending-tag chains remain present. |
| Sparse overlap | Two largest-fitting blocks cover the nonempty requested interval. Minimum's associativity/idempotence make duplicate overlap harmless; addition counts it twice. The anchor's overlapping sum is 19 while the true interval sum is 15. Static preprocessing, nonempty queries and machine-word indexing assumptions are stated. |
| Fixed-width maximum | Expiry and newer-at-least-as-large domination have distinct valid proofs. A domination chain retains a no-older, no-smaller surviving representative. Removing equality intentionally selects the newest tied original index. Each original index is appended once and removed at most once; live deque and output storage are distinguished. |
| Signed shortest range | A later no-larger prefix dominates an older start for all future ends. A front that already yields a valid answer cannot improve that start's length at a later end. The increasing prefix order justifies stopping the threshold loop. Testing starts before appending the current end preserves the stated nonempty contract; this is not a count-all algorithm. Earliest-start length ties match the native tuple order. |
| Weighted strict-rank witness | The exclusive smaller-rank prefix excludes equal values, earlier input order is preserved, and max-updating a rank retains better earlier occurrences. Negative candidates can be omitted because the empty predecessor is allowed. Each occurrence keeps its own backward parent. The empty record wins zero ties; later endpoint ties recursively retain earlier predecessor endpoints. Independent exhaustive-subset selection checks the actual witness, not only its score. |
| Transfer practice | The four-field nonempty maximum-subarray merge, `None` identity, rooted-subtree DFS interval argument, changed-oldest-argmax rule, and count-all counterexample are consistent. The optional persistent/multidimensional orientations are not presented as implemented algorithms. |

The weighted-rank paragraph's “best nonnegative score ending at that value” is read together with the immediately following explicit empty-record convention; a rank with no positive occurrence holds the permitted empty alternative, not a claimed real endpoint. No incorrect witness is returned under that convention.

## Additional checks actually executed

Command from the repository root:

```text
node scratch/review-range-queries-independent.mjs
```

The driver imports the **current** exported examples and pure models, writes those examples into its own evidence directory, and executes the actual Python definitions. It does not import or call the author's verifier or use a second tree as an oracle. Python 3.12.14, standard library only; no packages installed.

| Native check | Observed count and independent oracle |
| --- | --- |
| Complete examples | 13 exact stdout comparisons against the exported expected output |
| Ordered folds before/after assignments | 7,084 intervals; 2×2 integer matrix summaries checked by applying the original operator word independently to both coordinate basis vectors |
| Frequency state/rank | 728 finite count states, 4,013 selected ranks, 2,912 invalid-rank cases; expanded occurrence lists, plus direct cumulative scans for counts up to 2¹⁰⁰ |
| Lazy update composition | 1,836 sequences, 33,660 final range answers; direct-array application, including every four-step full-cover word from six set/add maps, a later singleton overwrite, non-power-of-two lengths, empty ranges and large exact integers |
| Sparse minimum | 1,330 nonempty intervals; sorted direct slices |
| Moving maximum | 27,342 array/width cases; direct per-window maximum `(value,index)` with newest-tie ordering |
| Signed shortest range | 27,342 array/target cases; all contiguous nonempty intervals, minimum `(length,(left,right))` |
| Weighted witnesses | 1,100 fixtures against every legal subsequence, including duplicates, negative/zero rewards, score and the recursive endpoint tie contract |

| Pure-model check | Observed count and evidence |
| --- | --- |
| Segment cover/order | 495 intervals; actual selected node intervals concatenate to the requested original index sequence exactly once |
| Lazy state | 216 chains without intervening reads, 3,960 final ranges; direct arrays, unchanged input snapshots and push-frame logical-value preservation |
| Deques | 1,200 cases; direct windows/ranges, actual candidate order after append, exact output identities and append/pop bounds |
| Weighted model | 600 exhaustive-subset fixtures; optimum score and strictly increasing original witness identities |

Native completion: **2026-09-10 10:05:38 UTC**. Model completion and four source SHA-256 fingerprints: **10:05:37 UTC**.

- [Independent driver](../../scratch/review-range-queries-independent.mjs)
- [Independent native checks](../../scratch/review-range-queries-independent.py)
- [Native results](../../scratch/range-queries-independent-review/native-results.json)
- [Model results and reviewed source hashes](../../scratch/range-queries-independent-review/model-results.json)
- [Exact exported program snapshot](../../scratch/range-queries-independent-review/programs.json)

The author separately records desktop/390/320 browser interactions, keyboard/panning, actual opened screenshots, anchors and render checks in [RANGE-QUERIES-VERIFICATION.md](RANGE-QUERIES-VERIFICATION.md). This bounded independent task did not repeat that browser suite or inspect external videos. It tested finite exact integer contracts; it does not establish floating-point stability, platform judge performance, malicious arbitrary Python inputs, arbitrary lazy map families or universal interview coverage.
