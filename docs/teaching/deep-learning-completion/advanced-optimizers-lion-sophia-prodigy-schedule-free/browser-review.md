# Advanced optimizer rendered review

Root inspected production build19:51:16 at1280×900. These are actual observations, not completion of the remaining checks.

## Mechanism investigations

Lion's default blend−.07 selects−1 despite positive gradients2. The larger-magnitude preset4 changes blend+.13 and sign+1, moving the weight−.4→−.4192. Appending a fourth entry and editing it to−4 yields blend−.5550571, new weight−.4365699 and memory−.2105628. Removing it works. Zero-history/no-decay preset gives one zero step with θ unchanged−.4 and memory0.

Sophia's default expected sampled estimate equals analytic curvature`.96375`; changing only real label1 to0 changes real-label squared gradient`.680625→.455625`, leaving both curvature quantities unchanged. Removing off-diagonal coupling makes all four probes`[4,1]`, with eigenvalues`[1,4]`. Zero-input/H preset makes all estimates, gradients and eigenvalues0.

Prodigy default12-step trace crosses target−2 and reaches−3.4163185 with loss1.002979. Multiplier1 first increases retained d after step2 and produces visible overshoots, includingθ−6.1100814/loss8.4463845 at step7. Exact-target preset keepsθ−2/loss0 and d1e−6 throughout. Reset restores the default. Desktop graph and input spacing inspected; mobile/other static figures remain pending.

Schedule-Free β0 changes step3 gradient point from−.39 to−.3 and final x to−.208/loss`.729632`, matching its separate endpoint table. Stationary null makes all x/y/z1, gradients0 and loss0.

## Actual saved classifier

Lazy state loads when its lab enters the viewport. Default AdamW source312 class0 probability`.999460263→.999456079` after the diagnostic update. Target-only change0→8 preserves every before probability while updating class0 to`.998092095` and class8 to`.000400615`. Keyboard Space selects pixel28; reflection0→16 changes normalized input to1 and class0 before/after`.995976425/.985679367`, class8`.000262585/.002586484`. Its pixel gradient for target8 is−.9997374, with weight`.2599812→.3135186`.

Cosine update401 has active rate0 and all10 exact deltas0 even for this altered input/target; weights unchanged and moments may update. Schedule-Free evaluation x has class0 approximately`.9896→.9893`; training y switches the displayed calculation to`.990766659→.964687769`, explicitly labeled as a different diagnostic mode.

Allocation default70B/8ranks shards only buffers: weights140GB, gradients280GB, buffers70GB per rank. Switching all listed arrays changes them to17.5/35/70GB respectively. Matrix4096² dense/factored counts remain67,108,864/32,768bytes.

## Initial rendered finding (resolved below)

The enlarged probability-change plot rounds all three y ticks to0 for target-only default AdamW deltas around1e−3. Exact tables remain correct, but the plot needs an explicit scaled unit or suitable local tick formatting. Author notified; frozen shared chart must not be changed broadly. Final review still needs this correction, desktop/phone representations, invalid/reset, saved-state/retry, code/practice access, console and exact final source/build binding.

## Follow-up on production20:17:33

The probability-difference axis now uses explicit10^-3 units for the reported target-only case, with ticks−1.5/−.2/1.1 and an exact unscaled table. Screenshot inspected; this finding is closed. Deeper Sophia-G disclosure shows the nonlinear logit-Jacobian path and missing-Hessian-term example clearly. All24 retained histories load on entering their region; last candidate Schedule-Free seed29 gives x CE.1922038 versus y.1866049, retaining its method-defined evaluation contract. Log10 toggle and candidate selection work. The complete runnable library disclosure contains5,319 displayed characters including its actual code. First independent solution explains bias-corrected AdamW and why coupled decay differs.

Blank Initial weight followed by one Reset restores−.4. At320px document widths305/305; all24 currently visible numerical SVGs retain nominal340–720px geometry with no text extending outside its SVG. Rotated-bowl screenshot retains equal axes, distinct update arrows and readable caption. Keyboard ArrowRight scrolls the239px Prodigy chart region over340px content. Desktop curve screenshot shows original fit/validation measurements with labels. Rapid seed29/Lion/Prodigy changes end at correct Prodigy state with d.09739142.

Actual Sophia-G29 snapshot failure was exercised by temporarily holding only the generated preview asset. The friendly error and retry button appeared; the exact SHA256 asset bytes were restored before retry, which loaded scale.01/rate.01 with the real classifier probabilities. No recovery artifact remains, console warning/error list empty.

A new worked-example finding remains: its initial pixel28 has zero intensity, hiding the direct score/gradient mechanism. Manually selecting pixel18(intensity16) exposes meaningful terms but shared32.16px/unit geometry makes≤.004 gradients subpixel, with some residual labels rounded0. Author is changing only this worked default, labeling separate column scales and preserving tiny nonzero labels. Fresh diagnostic remains pixel28 intentionally to show momentum can move zero-current-gradient weights. A green sixth history series is also being replaced with muted rose. These bounded changes require final rendered/source follow-up; other evidence remains reusable.

## Final bounded review on production20:29:10

The worked classifier now starts at source277 pixel18, raw16/normalized1. Class0 score contribution.6726 and gradient−.003998 are immediately visible; separate stated scales32.16 and15010pixels/unit preserve their distinct units. The largest gradient bar is60px and class5/8 gradients have20.65/17.71px bars. Class2 residual1.83e−4 remains visibly nonzero. Desktop screenshot shows separated rows, labels and zero rails. Selecting pixel28 gives all direct contributions/gradients exactly0 while retaining the other-pixel/bias scores. Returning to18 restores the mechanism. At320px the page remains305/305; the720px diagram retains legible geometry in its bounded scroll region. The final six curve strokes end in muted rose #d597ad, with no green series. Console warning/error list remains empty. All rendered findings are closed; no new fit or performance claim was introduced.
