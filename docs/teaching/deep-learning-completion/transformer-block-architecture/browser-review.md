# Initial rendered review — corrections pending

Root reviewed the production build on 26 September 2026 at the actual reader route, using the in-app browser. This is a partial observation record, not browser closure or implementation certification. The author is correcting the issues below. Recheck affected rendered source and record final hashes before closure.

## Observed working behavior

- Normalizer offset 0 makes both original/transformed routes equal. Offset 9 keeps LayerNorm unchanged while RMSNorm maximum change is 0.54643. A blank feature field retains the last valid output and supplies a local error. Reset clears the edit; ArrowRight on the offset slider changes the value and result immediately.
- In the complete-block lab, setting both branch multipliers to zero preserves `[1,2,5,8]` under pre-norm. Switching to post-norm gives `[-1.09544,-.73029,.36515,1.46059]` without changing parameters.
- Frozen pre-norm movement model defaults to class 7, class-4 probability .000211. Reversing coordinates gives class 5 and maximum logit change 4.88524. Reversing complete point/time records preserves output within 8.321e-7. Masked padding preserves it; unmasked padding changes logits by 1.92338. Switching to post-norm and setting selected x to 0 gives class 6 and probability .06155. All computations update through real controls.
- Contrast probe produces `[.3286,-.3895,.0122,.0487]`, derivative L2 .51208. Small-spread input increases sensitivity to L2 59.41012 with finite-difference discrepancy 8.677e-7. Reset restores the null probe.
- At length 4096 and width 512, the cost lab displays 12,884,901,888 map MACs and 17,179,869,184 pair MACs, with unchanged 3,152,384 parameters. Reset works.
- No browser console errors or KaTeX errors observed in this pass.

## Open findings sent to the author

1. The pre/post wiring figure is two numbered lists. Draw actual bypasses, operation nodes and sum junctions; preserve the text companion. Communication lanes and serial/parallel variants also need visible edges, not arrow text alone.
2. The inherited shared lab surface is green (`#0c1412`), as are control surfaces (`#101c18`). Root supplied a scoped neutral CSS variant for new lessons; apply and verify without altering previous lessons.
3. At a 390px viewport (375px client width), the movement grid expands to 360px inside a 309.33px container, giving a 393px page. The mobile `1fr` track inherits a table's automatic minimum. Use `minmax(0,1fr)` and zero child minimum width; retain local table scrolling. Inspect again at 320/390px.
4. The default movement trace exposes many 24-feature intermediate vectors at once. Keep immediate path/output visible, with the full intermediate trace in an optional disclosure for progressive depth.

Remaining coverage includes corrected geometry, feature-circuit matrix controls, deferred programs/practice, all newly drawn figures, intermediate/narrow widths and final source-bound closure. Unchanged observed numerical behavior can be reused after checking source identity.


# Transformer rendered follow-up

Continuation of `browser-review-initial.md`, preserving the actual earlier controls and outputs.

On the18:53 production build, changing normalizer offset5 changes mean4to9, leaves variance7.5 fixed, and changes mean-square23.5to88.5. The new denominator view shows LayerNorm2.73861 unchanged and RMSNorm4.84768to9.40744. Offset0 is null for both; scale0 produces zero outputs and finite .00316 denominators. Centered vectors and denominator effects are now visible rather than hidden in prose.

The new feed-forward write workspace starts with input`[1,2,-1,-2]`, responses`[2,4,3]`, writes`[2,4,1.5,-1.5]`. Editing only the second position's first input from1to2 changes only its write to`[1,0,1,-1]`. Changing the shared down-projection row1 column1 to2 changes first write to`[4,4,1.5,-1.5]`and second to`[2,0,1,-1]`. The distinction between position independence and shared weights is therefore observable. SwiGLU default up responses`[2,4,3]`, gates`[1,-2,-1]`, output`[1.46212,-.95362,-.40341,.40341]`; gate matrix row1 column1 set0 zeros the first gate and first write. Reset works.

New static representations show communication masks, cross-attention checkpoint shape, checkpoint ownership and parallelism. At320px the page client/scroll width is305; nominal diagrams preserve720communication,300bypass,440write,290mask,600checkpoint and340plot widths inside local scroll regions. There are no KaTeX error nodes. The six-layer fitted-feature dump is now behind a disclosure, leaving the first-pass route readable.

Desktop finding: three encoder/decoder/encoder-decoder columns were~218px wide and clipped the290px diagrams. Author changed this to auto-fit min320px tracks, yielding at most two on the reader desktop and one when narrower. Independent reviewer assessed this CSS-only change. Final rendered verification follows on the refreshed build.


## Final rendered closure

On production build19:16:57, desktop1280 mask figures occupy325.22px per column rather than the earlier clipped218px, with the third diagram wrapping onto the next row. Screenshot confirms encoder/decoder labels, Q/K/V origins, mask cells, padding and output paths are readable. At320px all three stack, preserve290px nominal SVG width in212px local scroll regions, and the page client/scroll width is305/305. ArrowRight scrolls the encoder diagram16.67px. The phone screenshot confirms the native text remains readable rather than scaling the whole figure down.

Edited normalizer offset5to9, emptied the number, then clicked Reset once: value5 restored immediately. The first practice Solution independently opens and calculates mean3/variance5, both normalized vectors and the changed RMS denominator41. The complete source download remains in the lesson at its semantic learn-code path; the independent review checks its actual content. No KaTeX errors or final console warnings/errors were observed. Completion control untouched. The initial and follow-up records below retain actual changed/null/model/program observations; no exhaustive input sweep or GPU timing is claimed.
