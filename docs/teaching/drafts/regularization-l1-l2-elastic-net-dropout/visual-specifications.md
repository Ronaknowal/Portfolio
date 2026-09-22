# Regularization: visual and investigation contracts

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Edit z, penalty and mixing fraction and follow the soft threshold and exact-zero interval live; edit rows and coordinate order and step actual residual updates; change physical airfoil measurements and inspect linked polynomial terms and fitted output; vary inputs, coefficients and keep probability and inspect all four dropout branches and expected loss immediately.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Content-only specifications, 12 September 2026. Read lesson.md and calculated-inputs.json together. Figures teach the mechanism at its home; four investigations serve different questions. No runtime, rendering, production integration or formal phase-two review is claimed.

## Shared implementation contract

An investigation starts with an unset prediction and meaningful editable inputs. Require a prediction before Apply; bind it to the complete applied input and the named output. Any relevant edit invalidates the commitment and conceals stale feedback, keeping a previous trial only as an explicit comparison. Reset restores baseline values and clears prediction. Controls may prefill input entities; they must not preselect the predicted outcome. Explanatory figures can show their already worked answer.

Use labeled signed contributions, curves, parameter-space geometry, row/residual traces and probability trees as appropriate. Do not render every investigation as the same text panel. No decorative randomness, confidence bands or invented empirical trajectories. State whether a value is an exact constructed calculation, an analytic formula or a saved observed-data result. Axes and tables must preserve objective normalization, parameter convention and physical units. Zero coefficients need explicit marks, not a color threshold. Display numerical precision honestly: exact soft-threshold zero differs from a tiny computed nonzero coefficient rounded for display.

All interaction must work by keyboard; draggable geometric points also have numeric controls. Positive/negative/zero and train/validation roles use text/shapes as well as color. On narrow screens, stack stages with a persistent inspected entity and keep a text equivalent for every graph. Avoid shrinking a twenty-column table to illegibility: use named term rows and an accessible full table. Apply performs bounded local calculations; no background hyperparameter search or full training on page load. Load only this topic's compact model/fixture data. The full author JSON is evidence, not a requirement to download every retained validation prediction into the browser.

## F1 — The objective is a sum (§1)

Analytic one-dimensional curves on w∈[−1,5]: data cost (w−3)²/2, ridge penalty w²/2 and their sum. Mark w=3: data 0, penalty 4.5, total 4.5; w=1.5: 1.125+1.125=2.25. Use separate curves/stacked value bars and labels for training cost and parameter preference. Show λ=1, ρ=0, scalar normalized curvature=1. Do not title it a measured generalization curve. Text equivalent gives the equations and both decompositions.

A small units inset transforms x meters→100x centimeters and w→w/100. The product is unchanged; L1 decreases by 100 and squared L2 by 10,000. This must show the correct direction of scale preference. It is not evidence that the rescaled input became less scientifically important.

## I1 — A coefficient's threshold (§2)

Inputs: z∈[−4,4], λ∈[0,4], ρ∈[0,1], plus numeric entry within sensible finite bounds. Baseline z=.4, λ=1, ρ=1. Prediction is negative / exactly zero / positive, unset initially; optionally let an advanced learner enter a numeric coefficient. Apply computes S(z,λρ)/(1+λ(1−ρ)), displays the actual objective minimum and its left/right slope conditions. Highlight threshold interval [−λρ,λρ] on the z input line and plot the resulting coefficient as a separate dependent output. Distinguish the coefficient-axis objective plot from the z-to-coefficient function plot.

Checked contrast: z=.4→1.4 with λ=1,ρ=1 changes zero→.4. Null: z=.4→−.6 remains exactly zero, with different data cost. λ=0 gives w=z for every ρ; changing only ρ there is a second null. Baseline scalar family comparison at z=3,λ=1 gives ridge1.5, lasso2, ENρ.5=5/3. Do not report these different objective minima as a criterion for choosing a family.

Feedback substitutes the applied values into the threshold and denominator. At a threshold endpoint, show zero and the valid subgradient interval. Mobile keeps a readable numeric decomposition even if two charts must stack. Phase two checks threshold equality, signs, λ=0, ratio endpoints, input invalidation and no prefilled prediction.

## F2 — Constraint geometry that includes nonzero coordinates (§2)

Use z=(3,.4), identity data curvature and exact minimizers for λ=.1 and1. At λ=.1 lasso w=(2.9,.3) lies on a diamond edge, not an axis; at λ=1 it is (2,0). Ridge minimizers are z/(1+λ). ENρ=.5 minimizers follow the coordinate formula. For each matching constrained figure set its budget to that solution's actual chosen penalty measure, and disclose that value; do not equate λ with a constraint radius. Draw contours of the declared quadratic data loss, the correct disk/diamond/mixed level set, origin and both coordinates with equal geometric scale.

The mixed penalty is ρ(|w1|+|w2|)+(1−ρ)(w1²+w2²)/2; sample its implicit level set accurately, preserving nonsmoothness on axes. A generic rounded rectangle is not this boundary. Include a text coordinate table. The claim is possibility/tendency of sparse contact over input regions, not that almost every contour touches a corner.

## I2 — The residual determines a coordinate update (§3)

Four rows and two editable numeric features/targets, source IDs0–3. Baseline X=[[1,1],[1,−1],[−1,1],[−1,−1]], y=[3.4,2.6,−2.6,−3.4], λ=1,ρ=1. Allow λ,ρ and the actual data to change; retain four rows for transparent bounded fitting. Offer the duplicate preset X=[[-1,-1],[1,1]],y=[−2,2] with its two-row identity explicitly separate. Missing/nonfinite inputs are invalid; a constant centered feature remains valid and selects the declared zero representative.

Before Apply choose a named output to predict: sign/zero of coefficient1 or2, or a numeric prediction for a selected row. Fit the centered common objective, recover the unpenalized intercept and reveal stepwise coordinate states. Use row contribution strips: original values→training means/centered values→partial residual→association c and curvature a→threshold/divisor→coefficient→fitted value. Do not animate a final coefficient into place without showing its residual association. Record a residual after every coordinate for the selected sweep, not only final prose. An objective-versus-sweep plot uses actual computed history; loss should not increase beyond rounding noise for exact updates. Distinguish objective from mean-square data error.

Algorithm reference is coordinate_fit in author-calculations.py. Preserve residual updates and optimality condition; default bounded convergence requires max KKT violation≤1e−8 with at most10,000 sweeps for the four/two-row fixture. If the cap is reached, label incomplete convergence and keep the current result inspectable rather than pretending it is an exact optimum. Bound edited magnitudes reasonably (e.g. each absolute value≤100) so work remains trivial. This solver supports arbitrary edited correlations, not only orthogonal presets.

Checked baseline: ridge(1.5,.2),lasso(2,0),EN(5/3,0),intercept0. Contrast row0 target3.4→7.4,λ1,ρ1 gives slopes(3,.4),intercept1. Null shift every original target by7: slopes(2,0),intercept7, same residuals/objective. Another null swaps row order with values/targets together; fitted coefficients and predictions per stable entity remain unchanged. The duplicate case needs18 sweeps for ridge and29 for EN in the author's1e−10 tolerance; UI tolerance differs, so do not claim those exact counts unless it uses the same setting.

For duplicate lasso, visiting column1 first gives(1,0); reversing coordinate order gives(0,1), with identical predictions/objective. Optional order control is useful here but must retain a complete deterministic ordering contract and record it with prediction. Phase two verifies nonzero and zero KKT conditions, target-shift invariance, constant columns, arbitrary edits, duplicate solution equivalence and honest convergence messages.

## F3 — Duplicate measurements do not identify an attribution (§4)

Constructed two-row duplicate example, common λ1. Parameter-space contours show data loss .5(w1+w2−2)² and constant-sum lines. Lasso minimizer segment joins(1,0) and(0,1); midpoint(.5,.5) is also optimal, objective1.5. Ridge solution(2/3,2/3),objective2/3. ENρ.5 solution(.6,.6),objective1.1. Label that these optima solve different objectives. Linked prediction strip at x=1 shows sums1,4/3,1.2; at x=−1 their signs reverse. Do not imply equal predictions across different penalty families, or universal equal coefficients for merely correlated features.

## F4 — Measured regularization paths (§5)

Use airfoil.candidate_results with exact family,strength,fold MSE and coefficients. Plot six λ values on a log axis; markers identify actually fitted values. Show per-fold lines faintly and mean lines prominently, with actual mean baseline45.0727679595 and OLS17.3502676828. A linear error axis labeled squared dB must make differences visible without implying significance. Provide zoom with clearly changed axis bounds if useful; no invented confidence band or smooth minimum between sampled points.

Coefficient panel uses a selected **single fold** across the six fits. All20 named terms have signed values; zero has an explicit mark. A selector can inspect one term or a small selected group, while full data remain accessible. Do not average fold coefficients as if they came from the same fitted scaler, and do not combine final all-development coefficients with fold-path values. Selecting λ=.1 lasso reveals9 active terms in each fold; λ10 and100 reveal zero slopes and the fold-specific training means. Selected final λ.001 lasso has20 active terms. Caption distinguishes fold prediction records, selection scores and final refit.

Source/development statement: observed Airfoil dataset;1200 development,303 reserved unpredicted,random-row protocol,threefoldseed202. This is distinct from root's later learning-curve protocol. No new airfoil/run inference, no held-out reserve score and no universal best family.

## I3 — A physical input becomes twenty contributions (§5)

Use the saved final ridge model at normalized λ=.001 in airfoil.selected, along with its scale_mean,scale_scale,coefficients,intercept and feature_names. Base source row1203: frequency1250Hz,angle17.4degrees,chord.0254m,speed31.7m/s,displacement.0176631m; observed target128.306dB. Recorded fitted prediction124.4653054006435dB. This is a development-row inference trace, not a held-out accuracy demonstration.

Editable entities are the five raw physical measurements. Recompute their original/square/pair-product features in the exact PolynomialFeatures order stored in feature_names, subtract saved means,divide saved scales,multiply saved coefficients and add the saved intercept125.01384916666669. Training parameters stay fixed. Display a measurement diagram/table→polynomial term network→signed contribution bars→total. A changed original value highlights every linked power/product, so changing frequency correctly affects six transformed terms, not only its linear term. All contributions are in target dB even when raw feature terms have compound units.

Prediction is increase/decrease/unchanged relative to the baseline or a numeric result; unset before Apply. Checked contrast frequency1250→1750Hz with all other values fixed changes output to123.62864627978571dB, a decrease≈.836659121dB. Null: edit only the reference target128.306→130; prediction stays124.4653054, while residual changes. Treat this optional target field as a comparison value, never a feature. Changing only the displayed unit conversion while converting the actual stored physical quantity consistently is another null if that optional control is implemented.

Accept finite positive frequency/chord/speed/thickness and a bounded angle; use observed column ranges as default control ranges, show units and reset. Edited combinations can be hypothetical even within marginal observed ranges: label the changed row a model scenario, not a guaranteed physically feasible new experiment or causal effect. Do not add a confidence interval to this deterministic trace. Phase two checks exact polynomial order, raw-to-product propagation, saved scaling, contribution sum, changed/null cases and local loading without fitting/downloads.

## I4 — Exact dropout mask tree (§6)

Inputs two x values,two coefficients,target and keep q∈[.1,1]. Baseline x(2,1),w(1,−1),y1,q.5. Require prediction of mean output and whether expected noisy half-loss exceeds the clean half-loss; both unset. Four branches correspond to masks00,01,10,11 with probabilities(1−q)²,(1−q)q,q(1−q),q². Display actual rescaled inputs,multiplication,output and weighted loss contribution. A probability bar tree and a discrete output distribution make this a different representation from a coefficient-fitting lab.

Baseline outputs0,−2,4,2; half-losses.5,4.5,4.5,.5; each probability.25; mean1,expectedloss2.5,cleanloss0. Compute expected output and loss by summing probabilities; also independently display the analytic extra term(1−q)/(2q)Σ(wjxj)². Do not average branch values uniformly when q≠.5. At q1 the zero-probability branches can remain as labeled impossible branches; do not divide by zero or allow q0.

Contrasts: q.5→1 gives the deterministic result and extra loss0. Practice fixture x(1,2),w(2,0),y1,q.75 has mean2,clean half-loss.5,expected half-loss7/6. Null: change y1→3 with baseline x,w,q; noisy-minus-clean loss remains2.5 although both losses change. Set w2=0 and change x2 or toggle its mask: no contribution/output changes. Feedback explains these input-specific quantities, not a generic 'dropout prevents overfitting' slogan.

A noninteractive nonlinear inset uses u0or2 equally likely,f(u)=max(0,u−1):average f=.5 versus f(average u)=0. This shows why the local mean-preservation fact does not imply exact nonlinear model averaging. More complex network masks/normalization belong to the linked deep lesson.

## F5 — Ridge information directions (§7)

Analytic singular values4,.5,parameter a=nλ. Fitted-data multipliersσ²/(σ²+a). At a1:.9411764706,.2;at a4:.8,.0588235294. Draw equal-scale coefficient directions paired with their different data sensitivity, then the multiplier table/curves. These are analytic chosen values,not fitted Airfoil singular values. Early-stopping inset has normalized Gram eigenvalues1,4,η.1,t1,filters.1,.4 and individually matching λ9,6; clearly distinguish Gram eigenvalue from singular value and nλ from normalized λ.

## F6 — Factor symmetry versus the whole objective (§8)

Plot a,b axes; zero-data-loss hyperbola ab1 and balanced point(1,1). At λ.25 show same-prediction alternative(2,.5),penalty1.0625;balancedzero-loss penalty.5;full optimuma=b=√.5,product.5,total.375. Include the equal negative factors. A scalar p chart uses .5(p−1)²+2λ|p|,withp*=max(1−2λ,0),and explains the inequalitya²+b²≥2|p|. Label λ0 case separately because all ab1 pairs minimize then. No unearned claim about better generalization from balanced deep weights.

## F7 — Smoothness and magnitude are different preferences (§8)

Three positions with observed y(0,2,0),identity-penalty output(0,1,0),difference-penalty output(.5,1,.5),λ1 under the **unaveraged** data objective. Explicitly announce this local convention because nλ factors elsewhere use mean loss. Show L=[[-1,1,0],[0,−1,1]],equation(I+LᵀL)w=y,and output edge differences(.5,−.5). Plot discrete positions joined only as an explanatory neighbor relation,not measured continuous interpolation. Shift example adds3 to every input,showing difference output(3.5,4,3.5) versusidentity(1.5,2.5,1.5). Include text matrix calculation.

## F8 — Different complexity accounts (§9)

Constructed fitted-model record n100,smallℓ−150,k3 andlargeℓ−146,k5. Stacked numeric components: AIC306vs302;BIC313.815510558vs315.025850930. Use natural logs and lower-is-better labels. The log-likelihood improvements and complexity charges must remain distinct. No probability-of-model label and no application to singular neural models from raw parameter counts.

MDL inset transmits actual bitstrings using the declared sixteen-bit protocol: flag0+sixteenliteralbits(length17),orflag1+fourpatternbits(length5,repeatfourtimes). Show 0101010101010101→1|0101→exactdecodedmessage. Changed message0101010001010101 requiresliteral mode. The first flag makes payload length unambiguous. Fixedlength16 and repeat rule are shared codebook assumptions. Pattern identification costsfourbits;never hardcodeaone-bit 'perfect' explanation. Keep bits on a different labeled scale from AIC/BIC. Refined NML remains a formula explanation,not an invented operational compressor.

## Phase-two closure

Implement stable descriptive topic modules with route-local imports. Execute the displayed complete programs in their named setup,check exact calculations and code/output agreement,verify convergence and arbitrary changed inputs,then render/inspect desktop/mobile/keyboard and text alternatives. Cross-check user predictions against actually applied state and meaningful visual contrasts. Verify saved data boundary and ensure no reserved score is introduced. Conduct formal independent review and integration separately. Preserve necessary offline inputs,results and calculations until their documented roles are replaced safely; no disposable scratch asset is required by this packet.
