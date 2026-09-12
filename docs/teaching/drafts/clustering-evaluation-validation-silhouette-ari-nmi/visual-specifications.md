# Clustering evaluation — visual and interaction specifications

Status: content specification, 12 September 2026. Nothing in this file claims rendered or implemented work. Use with [lesson.md](lesson.md), [author-calculations.json](author-calculations.json), [visual-input-calculations.json](visual-input-calculations.json) and [data-provenance.md](data-provenance.md). The design records coverage and research decisions. Implementations must calculate from active state, preserve exact observation identity and verify the declared behavior before publication.

## Reading structure and common contracts

The seven inline figures and five investigations have different jobs. A figure introduces the mechanism at its first explanatory hurdle; its later investigation lets the learner change the mechanism's inputs. Do not replace every figure with a uniform metric card or make every lesson illustration interactive. The pair board, actual silhouette distribution, information partition, specimen cohort filter and weighted probe ruler should look like the different objects they explain.

- Figures V1–V4 belong in sections 2–6, before their corresponding technical abstractions. V5 belongs after the real-data output; V6 is the denominator explanation; V7 is the report protocol. Each must remain useful in ordinary reading without opening a control.
- Each investigation starts with a meaningful active example but **no selected prediction**. Learners change a draft input, select a prediction and explicitly commit it. Only then does Apply/Check calculate the changed result and compare it with that recorded prediction. Do not preselect a correct answer, replace the record as controls change, or give feedback against an obsolete input. After committing, a further edit marks the prediction stale and requires another commitment.
- Two prediction modes are enough where useful: selected-point/pair outcome and a summary direction. Name the baseline in the question. Direction means lower/same/higher against that saved baseline, with a justified floating comparison tolerance; pair membership uses exact labels. Empty predictions cannot silently become “same.” Provide a clear no-score result for mathematically undefined states.
- Reset restores the lesson fixture, clears draft errors, the prediction and the before/after history. A baseline remains visible beside a changed state. Invalid text must not corrupt or silently replace the active state. Predictive feedback explains the changed neighbors/counts, not merely “incorrect.”
- Provide conventional labels, keyboard-operable inputs, explicit buttons and visible focus. No drag-only editing, required hover, timed animation, auto-advancing trace or network computation. Inputs and model state belong to each investigation, not to global reader progress. A polite live region announces only the completed update and its result, not every pointer move.
- Use actual generated axes, zero lines, units and labels. Shapes and written group labels supplement color. Stable observation IDs keep the same identity across every linked view. Group colors may follow canonical first-observed label order; color changes caused only by renaming must not imply changed membership.
- At 1440px a figure may use linked side-by-side views. At 390/320px stack the views and move legends into a short caption; retain readable labels rather than shrinking a desktop canvas. A local scroll region is acceptable for a dense contingency matrix or pair board, with an accessible name and all numeric data in a table. No page-wide overflow. Do not make a two-dimensional scatter imply the metric is also two-dimensional.
- Computed outputs are finite-model evidence. General properties come from the derivations in the manuscript. Optional explanatory animation may link known states; no random motion, force layout or aesthetic curve may stand in for an algorithm.

## V1 — observation identity survives a rename

Placement: section 2 after the two aligned label rows. Hurdle: label values are not group identity, and array position must represent the same specimen.

Draw A–H as eight vertically aligned columns. The top ribbon groups ABCD and EFGH with names 0/1. The next ribbon has names 7/3 but the same bracketed extents. Link one together-pair A–C and one apart-pair A–E through both ribbons, with written together/apart marks. A third, separated ribbon for `00110011` changes both relationships. Do not reorder the columns by group: the point is preserving identity across assignments.

Beside the ribbons, show “raw label equality: 0/8” for the renamed row and “partition agreement: unchanged.” A concise caption explains why these are compatible. This is categorical structure, not a quantitative x axis. Mobile: two pair-specific mini-ribbons can replace long crossing links, but preserve the full eight-ID table directly below. Alt text states which memberships changed. No controls required here; L2 supplies editing later.

## V2 — distance fan and genuine silhouette bars

Placement: section 3 immediately after C's calculation. Hurdle: b is a minimum of group averages, and the group mean is not an individual point's score.

Use A=0, B=1, C=2, D=7, E=8, F=9; L=ABC, R=DEF. Top number line has ticks 0–9 in arbitrary distance units; group braces are distinct from axes. From selected C, show the two own-group distances 2 and 1 in one lane and three foreign-group distances 5,6,7 in a second lane. A bracket summarizes each lane: a=1.5, b=6. Separate the lanes so lines do not falsely imply edge weights or a path.

Below, show six actual horizontal bars, sorted by s within each group, retaining ID labels. x axis is s from −1 to 1, zero at the center, with an overall mean marker .806548. Each observation has equal bar height; gaps between groups do not count as observations. A table exposes a,b,s and units. The exact rows are in the manuscript and `author-calculations.json`. The mean marker must use all six values, not the mean of group means. This static snapshot is sufficient to read the formula; do not require a hover to see C's inputs.

## L1 — point and membership laboratory

Placement: section 3's investigation, after the definition and conditions. Reuse V2's semantic entities, not a screenshot. Main active state: the six fixture IDs, finite coordinates and memberships. Selected ID defaults to C, while prediction defaults to unset.

Editable inputs: six named coordinate fields (or one labeled six-value text field plus per-row memberships); group choice L/R/S for each ID; selected ID. Coordinates are finite decimal values in [−20,20], at most six decimal places for readable traces; reject blanks, missing entries, infinities and extra coordinates. Coincident coordinates are allowed and deliberately tested. Membership labels need not all be used. One-group states are accepted data but the metric is displayed as undefined; do not turn undefined into zero. Coordinate edits and membership edits are separate actions so the learner can identify the intervention.

Prediction choices: selected s negative/zero/positive/undefined, or overall mean lower/same/higher/undefined than the saved active state. Record the relevant ID and active values with the prediction. Apply recomputes every a,b,s, not just the edited point. Tied competing means have a written tie annotation; choosing the first canonical tied label affects only highlighting. A singleton's a is displayed as “no other group member,” and s=0 is explicitly the convention, rather than a numerical a=0 claim.

Linked views: active number line, selected distance fans, group-by-group averages, sorted silhouette bars and a before/after table. Clicking an ID in any view selects the same observation everywhere. An optional distance table provides all 36 entries. Move the overall mean marker only after recomputing the whole partition. Dynamic coordinate extents include every active point with padding; silhouette extent remains [−1,1].

Required contrasts and nulls:

| Input | Expected observation | Author calculation status |
| --- | --- | --- |
| Original fixture | C=.75; mean .806547619… | Calculated and derived |
| Move C from L to R, keep positions | C=−.75; mean .4593944726…; other bars move | Calculated and derived |
| Move C coordinate 2→3, original groups | a(C)=2.5,b(C)=5,s(C)=.5 | Derived changed practice |
| Multiply every original coordinate by 2 | Every s is unchanged, all a/b double | Algebraic invariant; implementation must check |
| Make F the only S member | F=0 by convention; other groups' b may change | Author singleton state uses groups `000112`; reproduce exact specified state |
| All coordinates zero, original two groups | a=b=0, s=0 for each point | Calculated |
| All memberships L | No silhouette; preserve active points and name missing competing group | Future state check |

No randomly generated correct-looking bars. Future correctness checks compare complete changed vectors with a separate direct-distance implementation or sklearn, and verify input preservation only for the actually supported range. Future browser checks must use at least one learner-entered coordinate and an independently changed membership, plus reset and keyboard selection.

## L2 — contingency cells are pair-count shortcuts

Placement: section 5, after the concrete contingency table and before chance derivation. Hurdle: distinguish together/together, together/apart, apart/together, apart/apart, then see why cell combinations count the same pairs.

Use eight fixed IDs. Reference U initially `00001111`; candidate V initially the D/E swap `00010111`. Allow each candidate ID to choose a label 0–7 and allow reference editing in an optional “change reference” panel. Group names are tokens, not numbers with distances. Rename via a separate display-name field so the invariant is visible without a color shuffle. The initial prediction is unset; commit ARI lower/same/higher than the current partition before applying a membership change, or predict the exact together/apart status for a selected pair.

Representations:

1. Aligned eight-ID membership strips, preserving row identity.
2. Contingency matrix: each occupied cell contains its IDs, row/column margins and C(cell count,2). Do not substitute a heatmap whose color is the only count.
3. Upper-triangular board of the 28 unordered pairs. Each tile is labeled by two IDs and one of the four agreement categories; selecting a tile highlights its two IDs and relevant cell/margins.
4. Formula ledger: S=ΣC(nuv,2), A=ΣC(au,2), B=ΣC(bv,2), M=28; TP=S,FN=A−S,FP=B−S,TN=M−A−B+S; RI and ARI. Keep the expected-S marker AB/M distinguishable from an observed integer count: it can be fractional.

Predictions are checked against actual edited partitions. Required cases: rename unchanged; D/E swap yields RI4/7,ARI1/8; cross `00110011` yields RI3/7,ARI−1/6; refine `00112233` yields RI5/7,ARI4/11. Both one-group partitions and both all-singleton partitions have ARI1 by the stated convention; one versus the unchanged nontrivial reference has ARI0. The reference and candidate may use different numbers of groups. No matching optimization is needed for these scores. The matching branch in prose remains a separate concept.

Mobile: membership rows precede the matrix; give the 28-pair board its own compact scroll region if necessary. The numeric ledger stacks at equals signs. A text table enumerates pair categories for keyboard/nonvisual use. Future tests compare direct pair enumeration with contingency counts for several manually changed partitions; no exhaustive campaign is required merely to retest formatting.

## V4 — information retained versus extra distinctions

Placement: section 6 after the pure-refinement paragraph. Two area strips share eight equal specimen cells: U=`00001111`, V=`00112233`. Each U half splits into two V quarters. Brackets state H(U)=1 bit, H(V)=2 bits; a directional callout from a V quarter to its U half states “V determines U,” while the reverse leaves two possibilities. Beside it, two unit-width information bars distinguish the shared one bit and the additional one bit. Do not draw entropy as the number of groups without taking logarithms; the balanced example makes these exact lengths legitimate.

Caption gives H(U|V)=0,H(V|U)=1, arithmetic NMI2/3 and geometric NMI1/√2. A narrow table supplies the four occupied contingency cells. All quantities are computed from this finite table; no Venn area diagram should imply that every entropy relation admits an arbitrary Euclidean area interpretation. Static, no controls needed.

## L3 — a finite chance experiment, not a generic random slider

Placement: section 6 after the AMI equation. Hurdle: an empirically independent table can have NMI0 and negative AMI, while mean NMI is positive under random finite allocations.

Eight fixed IDs, binary U and binary V editors. Both label vectors can vary; permit 0–8 zeros for deliberate constant cases. The maximum number of distinct permutations of a binary V is C(8,4)=70, keeping the whole experiment small and exact. The default has balanced U and V with overlap2; initial prediction is unset. Let the learner predict the sign of observed AMI, or whether mean NMI across the fixed-margin null is zero/positive, before committing and calculating. Clearly state which of those two questions is being checked.

The active contingency table highlights overlap r. Below it, enumerate every distinct assignment of V's labels to the eight IDs while holding U and both margins fixed. Show a count histogram of r with y axis “number of assignments,” not probability density. A separate score plot has NMI on [0,1]; each attainable value carries its multiplicity. Show observed MI in bits, expected MI in bits, NMI and AMI with numerator/denominator decomposition. The dataset-specific null distribution is calculated whenever committed memberships change. Do not generate a random sample and label its mean the exact expectation.

Default acceptance: 70 assignments, overlap counts [1,16,36,16,1], mean NMI .11484428596…, mean ARI/AMI zero up to rounding. At overlap2 observed NMI0, AMI−.1297447264…. Renaming U/V gives identical values. Balanced agreement overlap0/4 yields NMI=AMI1. An unbalanced editor state must change the null weights appropriately; do not retain the 70-case default histogram. Both constant partitions use NMI=AMI1 convention and explicitly have no nondegenerate adjusted null; one constant versus one nonconstant gives0. Display “degenerate null” rather than implying mean adjusted0 in these cases.

Zero cells contribute0. Never evaluate log0. MI, its null mean and entropy use one base consistently; normalized outputs are base-invariant. Future implementation checks need an independent fixed-margin enumeration with at least one changed unbalanced pair, not only the balanced fixture. Keyboard can select an overlap/count bar and expose representative ID allocations, but all histogram counts also appear in a table. At 320px place observed table, histogram and adjustment ledger vertically.

## V3 — a shape mismatch on identical distances

Placement: section 4. Provenance: an original constructed 32-point fixture, not measured data and not DBSCAN output. Coordinates are r(cos θj,sin θj), r∈{1,2}, j=0…15, θj=2π(j+.25)/16. IDs r1-00…r1-15 then r2-00…r2-15. The .25 offset avoids points exactly on the left/right boundary.

Two equal-aspect coordinate plots show identical points: first colored by radius, second by sign of x. Both axes have identical bounds and arbitrary length units; circles are reference geometry only, not fitted decision boundaries. Below each, draw its actual per-point silhouette bars on the same [−1,1] scale. Mean ring score .07459431398…; left/right score .32061292874…. All coordinates/labels/per-point scores are saved in `visual-input-calculations.json`. Link the distributions to their respective names with more than color.

The caption states the two partitions were supplied to expose a criterion preference. No claim that any algorithm discovered either partition. This is a static contrast because changing ring parameters is not needed to establish the current hurdle; avoid adding a parameter dashboard solely to increase interaction count. On phones stack the two complete comparisons, retaining equal plot scales and a visible shared mean-score mini-table.

## V5 and L4 — real specimens, geometry and reference agreement

Placement: section 8 after Program4. Offline source is the licensed 150-row CSV; schema and exact hash are in provenance. Keep the real species names, units and stable row IDs. Do not call a species “cluster truth” or use it as a fit input. The visual question is the opening two-versus-three disagreement.

V5's reading snapshot: two rows, raw4 k2 versus raw4 k3, with separate columns for silhouette, ARI and group sizes. Use the exact Program4 values. An actual specimen scatter uses sepal/petal coordinates or the saved unwhitened PCA2 view; label its coordinate system and explicitly name the four-dimensional distance calculation. Selectable specimen details show all four cm measurements. A contingency table lists species rows and candidate columns. No shared y-axis or single “overall quality” bar combines silhouette and ARI.

L4 has two deliberately different modes:

- **Compare declared fits:** choose representation raw4/scaled4/PCA2/white2 and k2/3 from actual stored fit artifacts generated with n_init20,seed17 and the installed package version. These are inspected comparison states, not the only activity. Save labels and transformation provenance during implementation, then recompute displayed metrics from those artifacts and CSV. `author-calculations.json` currently records aggregate values; `visual-input-calculations.json` already contains scaled4/PCA2 k3 aligned labels and view parameters. Do not invent the missing per-fit labels from group sizes.
- **Rescore a frozen partition:** start with the supplied standardized k3 labels, then allow four independently editable positive feature weights in [.25,4], initially all1. Apply scales standardized columns by sqrt(weight) for Euclidean distances; it does **not** refit either scaler or clustering. Prediction is initially unset: commit mean silhouette lower/same/higher than the current baseline, or selected row's sign, before Apply. Controls name weights of squared differences, so “4” doubles that coordinate, not quadruples it. Include a common-weight action within the same allowed range for the invariant experiment.

On every applied rescore, update the full 150-point silhouette distribution, selected point's own/foreign averages, group means, negative count and a before/after mean. The fixed contingency table, ARI and AMI stay fixed because labels do. Mark those values “same memberships” instead of animating them as if recalculated independence evidence. A cosmetic two-dimensional point view must not move with hidden four-dimensional weights unless it is explicitly recomputed/labeled; the simplest choice is to keep the view fixed and update the distance table/bars.

Author fixtures: default frozen-label mean .459948239205…; petal-width weight .25 gives .454404648219…; weight4 gives .471241834929…; all weights4 leaves every s unchanged and mean .459948239205…. Baseline scaled4/PCA2 k3 label comparison has ARI1 and identical pair memberships, directly checked. The full-rank unwhitened PCA invariance is proved in the prose; an implementation may offer it as a separate null if it actually computes full-rank transformed coordinates.

Bounded computation: 150² distances with four features, triggered by Apply rather than every keystroke. Cache base squared coordinate differences if helpful; selection of a specimen reuses computed values. No browser k-means engine is needed for this lesson's main mechanism. Carry float precision through the computation and round only display labels. A screenshot of 150 tiny bars alone is insufficient: offer group distributions plus selected-ID values, and a text table or paged row list. At 320px axes/legend and selected-row detail remain readable, rather than shrinking a dense desktop dashboard.

Future checks: rederive the frozen-weight contrasts independently from raw CSV/labels; test a learner-entered intermediate weight and an invalid draft; verify fixed ARI/AMI and every point under common positive scaling; inspect one low silhouette specimen and its actual contributing means. Reuse the pinned fit calculation rather than adding a new seed or tuning to match desired output.

## V6 — the denominator changes when a case is rejected

Placement: section 9. Use original six line points with labels `[0,0,-1,1,1,1]`. Three aligned lanes:

1. All six IDs, −1 interpreted as an ordinary singleton group: silhouette .46984126984…; population6.
2. A clearly labeled assigned-row filter removes only C. Preserve C as an outlined rejected chip, not an invisible gap.
3. Retained A,B,D,E,F: silhouette .83831394096…; coverage5/6=83.33%, compared population5.

Draw actual retained per-point silhouette bars recomputed on the five-row set; do not simply erase C's old bar. The full-dataset mean before rejection is a separate baseline .80654761905…, not the same as the all-row −1 encoding. The caption distinguishes these three questions. Labels and filtering were supplied for this lesson; DBSCAN's density mechanism comes next. A final common-ID lane can schematically illustrate two methods' intersection but must not attach invented metrics to it.

This should be an inline flow, not another lab whose only action is revealing prose. A compact row/coverage table is an accessible equivalent. The key quantitative check is the changed b averages for points near C, not only the mean. No scoreboard declares the higher conditional score a winner.

## L5 — resampling moves the representatives, probes stay fixed

Placement: section 10 before Program5. Six locations `[0,1,4,5,8,9]` with IDs A–F. The observations used for training can have different multiplicities; the probe IDs and coordinates remain the original six. Teaching solver is exact one-dimensional contiguous-split minimization, explicitly distinct from iterative Lloyd fitting.

Two independent weight editors A/B, each six integers1…5, and k1/2/3. Strictly positive weights avoid absent-training-row bookkeeping in this particular investigation; real bootstrap zero frequencies are discussed in prose. Show frequencies as small stacked chips above the fixed locations, not as extra probe IDs. Learners can enter their own integer frequencies; preset left/right emphasis is only a starting contrast. Offer a separate shared-coordinate editor with six strictly increasing finite values in [−40,40] if it improves this lab; do not sort secretly and break IDs. This range includes the specified changed-input coordinates after tripling, including the endpoint30. It is optional because changed multiplicities already supply a genuinely variable input.

For each fit enumerate all C(5,k−1) contiguous cuts, compute each block's weighted mean and total squared cost, then select minimum cost; exact ties select lexicographically smaller ordered centers, matching Program5. Show the winning cut and a compact alternatives table; the two default optimal partitions change because the squared-error objective weights observations differently. Predict original probes by nearest center; exact equal-distance ties select the lower-index ordered center. Keep training-cut membership and probe assignment conceptually separate in the state model.

A shared number line carries two bands of ordered centers and midpoint decision boundaries. Below it, aligned six-ID probe membership strips connect to a disagreement/contingency view. Multiplicity changes move centers and boundaries, not the probes. Commit a prediction before Apply: a selected pair together/apart in fit B, or ARI lower/same/higher against the previous A/B comparison. Pair selection can use native selectors rather than dragging. Use exact pair feedback when the metric direction is undefined or degenerate.

Required states: left emphasis `[3,3,1,1,1,1]` gives centers[.5,6.5],labels001111; right emphasis `[1,1,1,1,3,3]` gives[2.5,8.5],labels000011. Their ARI is−1/14. Identical inputs produce identical partitions; for k1, any positive frequencies produce one-group partitions and ARI1 even while their center values differ. Uniform frequencies at changed locations `[0,1,2,8,9,10]`, k2, give centers[1,9] and cost4; tripling locations gives centers[3,27],cost36 and unchanged labels. Record selected minimum ties accurately; do not imply each trial has a unique optimum.

Mobile: stack each fit's frequency ruler and center line, then show the shared probe strip. Do not squeeze 12 frequency inputs into a single row; use named A–F rows with two labeled fit columns. Future correctness checks compare the tiny solver with an independent exhaustive partition search or direct weighted objective; future interaction checks use one novel weight edit, tie/null behavior, k1 and reset. No stochastic run loops, animation timers or general k-means engine are required.

## V7 — the final report has no backward arrow

Placement: section 12 after the fit/selection/report explanation. A flow diagram names immutable specimen IDs in three disjoint boxes: fit90, selection30, report30, generated by the seed23 permutation in Program6. It need not print all IDs; retain counts, roles and a downloadable/table list when implemented. Preprocessing and centers are learned from fit IDs; candidate evaluation on selection IDs chooses k among2/3/4; one frozen pipeline goes to report. Species labels bypass fit/selection and join only the report comparison.

Use arrows carrying objects: fit-derived mean/scale, training centers, chosen k, predicted report labels. A separate one-center baseline uses the same fit rows and standardized units. Show “distortion to training representatives” under both outcomes. Do not imply a report-set silhouette or an external ARI was used to select the model when Program6 does not do so. A lightweight task checklist belongs in prose; the figure's job is data ownership and information flow. At 320px make a vertical flow with side rails for the labels/baseline; no tiny three-column text boxes.

## Implementation acceptance, evidence and retention

Content-stage evidence exists for the six-point, eight-ID, finite-null, declared Iris and exact resample examples. `author-input-probes.py` is a reproducible small author calculation for ring coordinates, direct partition identity and frozen-weight contrasts. It is not a runtime model or independent verifier. Some figure values are proven algebraically; their actual renderer and controls have not been checked.

Phase two should first implement/execute the actual six displayed programs and reusable pure calculations for the five investigations, checking the promised finite contrasts with an independent method. Then inspect the actual route at1440/390/320, ordinary-reading placement of all seven inline figures, every prediction/reset lifecycle, a meaningful changed input per lab, undefined/invalid outcomes, focus order and hint-before-answer behavior. Open a small purposeful set of final screenshots, including the real specimen workspace and narrow pair/denominator/weighted-ruler views. Do not create a large gallery of every slider state or rerun unchanged numerical suites for spacing edits.

Preserve the authored numbers and source assumptions; if a necessary implementation refinement changes a model contract, update this packet and the affected explanation with a reason. Full program correctness, browser accessibility and publication integration remain deferred obligations, not completed author claims.
