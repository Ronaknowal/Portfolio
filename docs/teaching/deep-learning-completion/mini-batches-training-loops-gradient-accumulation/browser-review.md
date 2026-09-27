# Mini-batches — rendered review

Root review27 September2026, production preview4197. Main functional checks used22:11:18 build; final receipt follows bounded preset and narrow-math corrections. Desktop1280×960 and phone320×900 were inspected with actual controls.

## Four live investigations

State lab: default complete update gives weight0.3. Clearing each chunk leaves only the last derivative and weight0.133333333; stepping each chunk yields0.211111111 with two updates. Adding momentum changes the latter to0.361111111 and buffer−1.944444444 versus correct−3. One chunk makes the policies agree again. The cursor distinguishes forward, backward, step and clear; advancing from first backward leaves weight0 and gradient−1.666666667 while the example count advances independently. Reset restores all initial inputs.

Target mass: default weighted derivative−4.2 differs from equal group means−3.75 and exposes each row coefficient. Equal-mass/unequal-count case yields matching coefficients0.25,0.25,0.5,0. Removing all eligible mass gives undefined gradients and no plotted point/update. From the restored default, identical derivatives give matching−2 despite unequal coefficients/masses; this is explicitly not proof of objective equality. The identical-derivative preset initially failed to restore eligible mass after the zero-mass case; author correction must be rechecked before closure.

Normalization: default full mean5/variance10 differs from group means2/8 and variance1. Correctly weighted downstream gradients0.525657588 versus0.24999375 demonstrate the changed forward computation. The equal-output preset gives identical gradient0.4999925 yet running means0.1 versus0.19. Shared frozen statistics restore identical gradients0.525657588 and both running means0; no normalization similarly agrees at gradient29.75. These comparisons distinguish forward rules from reduction arithmetic and persistent state.

Actual Iris network: initial32-row group mean cross-entropy1.038175229618, selected derivative−0.252823367133 and all67-coordinate accumulated/full maximum gap5.551e−17. Editing original source21 petal length1.7→2.2cm changes loss1.0385005245 and selected derivative−0.251967651739; physical limit7 retains the same full-group derivative. Switching trained20 preserves the input edit. Changed targetclass2 and8-row prefix produce physical chunks7+1 with coefficients1/8, exact maximum gradient gap0, selected derivative−0.260310600156 and one update to selected parameter0.228193846757. The whole4→8→3 network, row residuals and per-chunk contributions are visible; these are hypothetical current updates, not new fitted accuracy claims.

Temporarily unavailable iris-initial.json produces a useful retry while source21 edit2.2cm persists. Recovery restores exact prior current loss/derivative. The dist-only asset was restored with SHA2566b1e6e36ddb8d404aa78a00a18732809aaa77d019a665258b8532393a938ec22.

## Reading and display

Opened complete train_iris.py4024characters and trace_update.py740characters (including headings). The latter is a complete top-level executable trace. Opened changed real-data practice's full1137-character success-criteria solution, preserving independently varied group-size requirements.

All22 rendered non-KaTeX SVGs passed painted text bounds. Actual desktop screenshot inspected the source flower→eight hidden coordinates→three class residuals and selected derivative formula with its scaled contribution plot. Phone slider ArrowRight changes petal2.2→2.3cm; diagram remains680px within219px focusable local region. Console warnings/errors were empty. Phone inspection found a4px page overflow from the full-precision inline normalized vector; the author was asked to move it into a suitable display with local overflow, preserving precision. Final receipt records its correction and the fresh-state preset recheck.

## Final corrections verified

Final build22:31:04: operation5 selects the second physical chunk and routes its derivative into the unchanged parameter store. The annotation is fully visible above that store; actual desktop screenshot inspected. Prior final-delta checks on22:26 confirmed zero-mass then identical-derivative restores fresh eligible weights1,1,3,0 (total5), matching derivatives−2, and the full-precision BatchNorm vector has local display overflow with phone document305/305. No precision was removed. All remaining previously recorded model/control checks reuse unchanged source.
