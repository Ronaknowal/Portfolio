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
