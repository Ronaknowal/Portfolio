# Mathematical axis-layout review

14 September 2026. This is a bounded layout repair for twelve mathematical lesson families in the user's illustrated-lesson review. It does not reopen their content or numerical models. The integration owner's whole-site geometry scan is triage; the screenshots linked by the evidence file are the separate rendered review.

## Scope and repairs

| Lesson family | Confirmed issue and repair |
| --- | --- |
| Gradient descent variants | Horizontal titles collided with the upper endpoint tick. Added a separate title row, preserving the plotted coordinate rectangle. Phone tick text also receives a scoped readability adjustment. |
| Learning-rate schedules | The calibration figure's common-scale title ran outside its viewBox and overlapped ticks. The target annotation and horizontal-axis title now reflow in ordinary HTML; phone calibration annotations have larger type. |
| Dynamical systems | Long vertical titles reached outside the SVG; negative x/y corner ticks collided. Axis descriptions now reflow outside the SVG and endpoint tick anchors face inward. The explicitly labelled, keyboard-accessible wide-plot region remains. |
| Itô calculus/SDEs | Compact chart titles collided with tick rows and lower-left values. External axis descriptions and inward endpoint anchors separate these meanings while preserving all model-coordinate transforms. |
| Algebra/functions | Lower-left x/y values overlapped in the function and quadratic plots. Endpoint anchors and a small dedicated tick-row offset separate them. |
| Random matrices | Axis titles and negative/decimal tick labels exceeded their SVG margins. Expanded the label gutters without changing data coordinates; separated corner ticks and horizontal titles. |
| Functional analysis/RKHS | Decimal and negative vertical values were clipped; corner x/y labels overlapped. Expanded the label margin and separated endpoint ticks. |
| Real analysis | Negative radian endpoints collided with the vertical minimum. Inward endpoint anchors separate them without changing the derivative or function geometry. |
| Random variables | A390px layout enlarged SVG labels enough to collide with the squared-loss corner ticks and conditional common-scale title. Moved the two axis titles into wrapping HTML and separated the squared-loss endpoint tick. |
| Single-variable calculus | The390px Taylor-plot corner labels collided. Endpoint tick anchors now face inward without changing the data transform. |
| Sampling/measurement | Phone text enlargement made the estimate title overlap the middle tick in exact sampling distributions. Moved that axis title into reflowing HTML. |
| Conditioning/stability | Phone text enlargement made exponent and amount labels collide with ticks. Only the affected cancellation and measurement drawings use new external axis-label rows. |

All numerical transformations, input domains, points, curves, bar values and model files are preserved. No shared plot component, catalogue, ledger or lesson manuscript is edited by this review.

## Evidence and boundaries

The bounded rendered review is complete for twelve lesson families. [Initial source-bound evidence](evidence/math-axis-layout-browser.json) contains 22 figure/viewport capture cases across the first ten families at 1366 and 320px. [The final targeted addendum](evidence/math-axis-layout-final-addendum.json) contains 18 cases for the later gradient, random-variable, single-variable-calculus and conditioning changes at 1366, 390 and 320px. The final closure records the current union of 24 component/stylesheet hashes. Each capture retains its actual production-build identity; earlier evidence is reused only for unchanged source or chart portions, rather than falsely claiming one global rerun.

Actual screenshots were viewed separately from the geometry assertions. The enlarged gradient endpoint ticks measure 12.39px at 320px, and calibration annotations measure 13.18px. Both are visibly separated from their titles. The sampling estimate title and conditioning exponent/amount labels now reflow outside their numerical plots. The dynamics wide-plot region responds to keyboard horizontal scrolling; its scroll hint remains visible at 320px. Decimal and negative tick labels are no longer cut off in the inspected RKHS and random-matrix figures.

The screenshot pass found an issue that text-to-text checks missed: a conditioning measurement line crossed the y=2 tick-label gutter. Moving the label anchors left of the entire drawn domain preserves the measurement lines while removing that foreground collision; all three final widths were actually inspected. The complete-width integration sweep also found gradient explanatory prose that inherited the enlarged tick font. That explanation now lives in each existing HTML figcaption. Both block panels and their captions were inspected at all three widths.

All 18 addendum screenshots were actually viewed. No unresolved overlap, clipped label or distorted coordinate scale remains in the representative affected states reviewed here. This is not an exhaustive interactive-state audit, renewed mathematical/program validation, a learner study or user acceptance. In particular, the dynamics plot intentionally retains its labelled horizontal-scroll region on a narrow screen.

All twelve changed JSX files parse with esbuild. Scoped diff whitespace checks pass. The integration owner performed the application builds and owns the whole-site geometry sweep and checkpoint reconciliation; this reviewer did not run a separate build or edit shared publication/ledger files.
