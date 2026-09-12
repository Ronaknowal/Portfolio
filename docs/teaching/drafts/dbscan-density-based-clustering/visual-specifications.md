# DBSCAN visual and investigation specifications

Content revision 1, 12 September 2026. This is an implementation handoff, **not implemented UI or browser evidence**. It accompanies the complete [manuscript](lesson.md). Root owns the phase checkpoint. All geometry below comes from declared coordinates or the supplied measured dataset; there are no decorative invented output curves.

## Shared learning and rendering contract

Use the project's calm dark/amber reading style. Core status and membership are different variables: filled circles mean core, hollow circles mean border, crosses mean noise; component membership may add a labeled outline/color. A visible key and numeric/text readout must work without color. A shared border has one chosen final label but retains two eligible attachment edges. It is never visually promoted to a core bridge.

Define one coordinate-to-pixel transform per plot. Euclidean two-dimensional plots require equal physical scales on both axes. Stretching a panel must not turn the drawn ε-circle into an ellipse. Draw a transformed-space circle or an original-space weighted-metric ellipse, explicitly labeled, never a mixture. Axis units travel with a feature/unit change. Use at least 12 px rendered diagram labels, stable marker sizes and leader lines; do not shrink all SVG text into illegibility at 320 px. Keep small figures fully visible on phones. Stack paired scenarios, align their axes within each comparison, and wrap captions separately from the geometry. A larger table may scroll horizontally in its own named region, but not force page overflow.

All investigations have the same learning sequence, with topic-specific inputs/output:

1. Show input data and a question. Prediction fields initially have **no selected answer**. A learner may inspect coordinates but not the changed result.
2. The learner edits inputs and records a prediction plus a short reason. Committing freezes an input snapshot. Do not count merely looking at a preset as a committed prediction.
3. Apply that snapshot, calculate the result, and compare with the recorded prediction. Feedback cites the exact decisive neighbors, changed edges or denominator. No generic “correct/incorrect” without explanation.
4. Further edits mark the prior result as belonging to the earlier snapshot and require a new prediction before revealing the new answer. Reset clears prediction, committed snapshot and revealed result. Preset selection also clears these states.

A noninteractive text alternative always explains the fixture and worked answer in the surrounding lesson. Optional hint and complete answer are separate disclosures, keyboard-operable and closed initially. Controls need explicit label/input associations; output elements must not accidentally take the label association. Use button actions and bounded calculations, no timers, autoplay or continuous large recomputation. Plot panning/hover must not be the only means of learning a value. Every selected point has a named row in a text table.

Future phase-two checks must include initial/reset/changed-input states, actual label associations, keyboard operation and ordinary reading at 1440/390/320 px with loaded project fonts. Open selected final images, especially dense labels and mathematical annotations. None of these checks has been performed in this writing phase.

## Inline figures

### F1. One radius, two counts — manuscript §1

**Obstacle:** confusing a selected row with an external neighbor and excluding an exact boundary.

Two aligned horizontal trail strips use A–J at `[-1.75,-1.5,-1.25,-1,1,1.25,1.5,1.75,0,4]`, all y=0. Domain must include −2 and 4; the second interval starts −1. The upper strip selects D, draws the closed interval [−2,0] and lists A,B,C,D,I. The lower selects I, draws [−1,1] and lists D,I,E. Both use ε=1 meter, m=4 observations. Filled interval endpoints indicate ≤, while the point glyph still encodes type. A separate selected-ring outline avoids confusing selection with core status.

The selected row appears explicitly in each roster, with distances D→D=0 and I→I=0. Display totals 5 and 3 beside the threshold 4. Keep the exact point spacing: compressed visual crowding is solved with staggered labels/leader lines and the roster, not moving observations. On a narrow viewport stack the roster below each strip. Alt text contains the two intervals/counts and identifies I as border despite its own count.

**Provenance/verification handoff:** original exact fixture; `author-calculations.json.street_neighbor_counts_eps1` supplies counts. Future check the interval endpoints and visible self entries, not only a snapshot title.

### F2. A border cannot transmit — §2

**Obstacle:** treating every short edge as a transitive clustering connection.

Use the same scale and coordinates. A–D and E–H are two core components; show their within-component core edges (all pair distances ≤.75). Show D–I and I–E as dotted attachment edges labeled “may attach,” not solid core edges. I is hollow; J at x=4 is a cross. Final I membership can be illustrated in a second small assignment row, with a note that reversing component discovery can choose the other side.

Below, an explicitly incorrect diagram highlights the path D→I→E, with the forbidden outgoing arrow I→E struck through and the reason “I has only 3 neighbors; needs 4.” Avoid a red cross covering the actual row coordinates or obscuring correct edges. The explanatory core-graph proof is beside this figure, not hidden entirely behind interaction. Caption distinguishes invariant core components/type from the border assignment.

**Future check:** core-edge list excludes I; two attachment edges remain visible; no line from J; labels do not overlap at close A–D/E–H positions. Use two stacked views at 320 px if necessary.

### F3. Fixed count asks how far — §5

**Obstacle:** inconsistent nearest-neighbor indices and thinking an elbow proves the correct scale.

First show D's full sorted distance roster: D=0, C=.25, B=.5, A=.75, I=1, E=2, F=2.25, G=2.5, H=2.75, J=5. Highlight the **fourth including self**, A at .75. The manuscript's shorter prefix is not the complete neighbor search.

Second show the entire sorted c₄ sequence `.5,.5,.5,.5,.75,.75,.75,.75,1.25,2.75` at ranks 1–10, with y-axis distance in meters, ε=1 line and “8 rows are core.” Tied ranks are actual individual rows, not a smooth fitted elbow. J's 2.75 stays on the same truthful axis. Label the .5/.75/1.25 levels and provide the vector under the plot so small differences remain accessible. Clicking/selecting a rank may highlight its row, but the default figure does not require hover. Do not draw a “correct radius” badge.

**Source:** `street_core_distance_m4`; sorted sequence and count derived from it. Future verify query-includes-self convention and ties, and that a query-hidden API is not substituted in the program.

### F4. Incompatible radius intervals — §8

**Obstacle:** assuming parameter search can always recover a desired partition.

Three intended groups: `[0,.125,.25,.375]`, `[.75,.875,1,1.125]`, `[5,5.75,6.5,7.25]`, m=3. Use real unbroken x scale, with enough width for two stacked detail panels if needed; if a magnified inset is used, give its own axis and indicate its extent on the overview. Never shorten the long empty gap without an explicit break.

Below the geometry, use two interval bars on an ε axis: “first two remain separate” ends with an **open** endpoint at .375; “diffuse group can exist” starts **closed** at .75. Shade the intersection only if nonempty. Here it is empty. Show the equal-spacing variant `[5,5.125,5.25,5.375]` below: ε=.25 gives three core components. A numeric table contains each group's points, threshold and resulting types.

**Source:** `varied` and `varied_null`. Future verify the stated thresholds from actual core graphs, not a visual gap heuristic alone.

### F5. A high reachability can start a cluster — §9

**Obstacle:** coloring all bars above a horizontal cut as noise.

Use actual OPTICS ordering A,B,C,D,I,E,F,G,H,J. Finite ordered reachability values are `.75,.5,.5,1,1.25,.75,.5,.5` for B–H in that order; A/J are undefined. Show undefined as open restart glyphs in a separate lane, not finite bars or zero. At ε=1 mark A and E as “core start”; E's bar rises to 1.25 while its adjacent core-distance marker is .75. J has no finite core distance within max_eps=2, so it cannot start. The low stretches are not separate unlabeled abstract valleys: every bar keeps its original row ID.

Provide the full ordering/core/reachability table from the manuscript. A vertical guide from E connects high reachability, low core radius and new-cluster label. This is a static diagnostic, not an extra lab imposed by quota. Optional cut movement may be added only if it preserves the rule and text alternative; it is not required by this content scope.

**Source:** `author-calculations.json.optics`. Future distinguish mathematical undefined values from `inf` in code and `null` in JSON. The first core radius .75 is not lost when reachability is undefined.

### F6. Stability is lifetime area with a selection constraint — §10

**Obstacle:** choosing every locally stable branch or calling a membership score a posterior.

A deliberately abstract condensed tree: six-row parent born at λ=1, splits at λ=3 into two three-row children. Plot λ vertically and width proportional to retained count. Parent rectangle has area 6×2=12. In the first scenario children exit at 6: combined area 2×3×3=18, so select both children. In the second they exit at 4: combined area 6, so select parent. Label selections using brackets/outline, not fill alone. The λ axis is inverse distance, not time or a probability axis.

Show an explicit “parent plus children overlaps the same rows” bracket. Root exclusion/allow-single-cluster policy of a particular library is separate: this illustrated parent is an eligible nonroot candidate. No claim that these lifetimes came from the ten-row trail or Iris. A tie at exit λ=5 gives 12 versus 12; it can be noted as requiring a declared tie rule if interaction is later added. The required artifact is static, not an extra selector.

**Source:** arithmetic in supplementary JSON. Future check widths/areas/axis direction and distinction from a conventional distance dendrogram. On phones stack scenarios at readable scale.

### F7. Rings and a bisector answer different questions — optional §8 branch

**Obstacle:** interpreting arbitrary-shape capacity as universal superiority or showing fake method outputs.

Use the exact 48 generated coordinates and two fit results in `supplementary-author-calculations.json.shape`. Inner radius 1 has 12 equally spaced rows; outer radius 3 has 36. Both panels share domain [−3.4,3.4] on each axis and equal aspect. Left: ε=.6,m=3, every row core, two rings. Draw only the nearest clockwise/counterclockwise ring edges to explain connectedness; state these are a sufficient subset of the complete neighborhood edges. Right: actual fixed-seed K-Means centers and memberships. Draw the **actual** perpendicular bisector computed from those centers, clipped to the same domain.

Label radii and centers without labeling all 48 points. A selected-row table shows its ID, coordinates, ring reference and both labels. Caption gives the constructed-reference ARIs 1 and approximately −.016, not a universal quality verdict. This preserves the original curved-shape comparison with transparent inputs. No external image asset. Future compare geometric centers/bisector with numerical source; don't reuse a decorative straight boundary.

## Investigations

### L1. Edit one observation, then find the decisive neighbors — §§2–5

**Question:** Will your change keep two core components, merge them, or remove a component? What type will the selected row have? For a pure order change, will its membership change?

Default input is the ten-row street, ε=1,m=4. Coordinate editor selects a row by ID and edits x,y on a .125-meter lattice within x∈[−3,5], y∈[−2,2]. Radius .125–3 in .125 increments, m integer 1–10. Each editable coordinate is independent; learners are not limited to moving I or selecting solved presets. Use explicit apply after prediction. Numerical controls suffice; dragging may supplement them but cannot be required. A row-order selector uses original/reversed order while retaining original IDs; additionally allow choosing a starting row to put first, preserving remaining order.

View after reveal: actual equal-scale scatter, selected closed disk, exact sorted neighbor roster, all row counts/types, core components as canonical sorted ID sets, chosen border assignments and the visit order. If multiple core neighbors disagree on component, list all eligible components even though ordinary DBSCAN picks one. No probability language.

**Evidence-backed anchors:**

| Input change | Contrast or null and explanation |
| --- | --- |
| ε=.75 →1 →1.25, m=4 | I changes noise→border→core; two→two→one components. Main JSON. |
| Reverse order at ε=1 | Same core partition and types; I can change attachment. Main JSON. |
| I=(0,.125), ε=1 | No shared-border effect under reversal: distances to D/E exceed 1. Exact squared-distance argument in practice B. |
| m=1, ε=.125 | All ten rows core, ten components, no noise. Main JSON. |
| Four duplicate rows x=2, ε=.125,m=4, separate small input option | All four core, one component, count includes distinct IDs. Main JSON. |

The move-I case is a derived geometric null; it is not marked as a separately executed program. Future phase-two fixture checks must calculate it and all declared preset outputs, including user changes beyond the initial examples. Radius boundary and duplicate-zero-distance states matter more than an exhaustive arbitrary-float audit.

### L2. Change a unit or change a metric — §6

**Question:** Will the radius-neighbor graph remain the same after your proposed transformation? Commit a yes/no prediction plus a cited pair.

Default four corners `(0,0),(1,0),(0,2),(1,2)`; optional fifth row `(3,0)` makes a nonuniform-conversion counterexample visible. Permit learner-selected row coordinate edits on a .125 lattice within [−4,4]² and independent positive x/y multipliers `.125,.25,.5,1,2,4,10,100`, plus a radius field `.125–200`. Allow the paired action “convert both coordinates and radius by c”; this is a clearly defined operation, not a standardization slogan. m defaults to 2 and can change to 1–5, with m>row_count allowed and explained as all noise. The unit label must describe raw versus transformed coordinates.

Before/after panels plot actual transformed geometry with equal axes. A potentially very tall partial conversion should use a tall scrollable local plot or a normalized display with explicit numeric axis scale and equal units; do not independently fit x/y to a square and pretend visual circular neighborhoods. Prefer stack panels and show the chosen pair's numeric distance when the scale makes a short horizontal link hard to see. An optional original-coordinate ellipse can convey the metric without depicting false transformed geometry.

**Contrast:** y×.5, unchanged ε=1, four corners: two components become one (elementary counts, to be checked in implementation). **Invariant null:** both axes/radius ×100 on five rows: same neighborhood matrix; supplementary JSON. **Actual faulty conversion:** only y×100 and ε×100 on five rows changes x=3 from noise to lower component; supplementary JSON. **Accidental same-result null:** original four corners under that faulty conversion still form two horizontal pairs, also probed. The feedback must distinguish universal preservation proof from one dataset coincidentally not crossing a threshold.

### L3. Who disappears when the score improves? — §7

**Question:** For your chosen radius/count, predict coverage and number of groups before looking at species. Then state whether the result serves the declared whole-collection question.

Fixed input is supplied Iris CSV, all original 150 IDs, four measured columns in order. Default representation is all four standardized features, full finite descriptive fit. Inputs: radius .1–2 in .05 increments **plus an explicit decimal entry** bounded to this interval; m integer 1–20; raw-centimeter versus standardized representation. Learners must choose at least one setting outside the worked .3/.5/.8/1 grid. A snapshot retains exact representation, preprocessing means/scales, eps, m and selected projection axes. Changing projection axes does not refit the four-dimensional model.

Initial species overlay hidden. Prediction includes group count and coverage band with no preselection; before optional species reveal require a short grouping decision. After reveal show cluster count, core/border/noise counts, coverage with denominator 150, row-linked assigned/noise lists and a two-feature scatter titled **projection of a four-feature fit**. A named row report shows all four values, full-space neighbor distances and type. Pairwise plots are optional selectors, not four copies by default.

Metrics: assigned-only silhouette available only when 2≤number_of_surviving_clusters<assigned_rows; otherwise say why it is undefined. ARI all rows treats −1 as one label; assigned-only ARI explicitly names its subset. Comparing two snapshots shows each retained ID set and their intersection. Recompute conditional scores on the common IDs if mathematically defined, and label the changed population. A fixed color assigned to −1 is not a new biological cluster.

**Probed contrast:** standardized eps=.5, m=5 retains 116/150 (2 groups), assigned ARI≈.631; m=10 retains 61/150 (3 groups), assigned ARI=1 but all-row ARI≈.279. **Null:** revealing species or changing plot projection must leave the four-feature fit unchanged; follows because neither enters model input, future implementation must verify. **Limit states:** standardized eps=.3,m=5 retains 30/150; m=1 eliminates noise mathematically. Empty/one-group metric states must be handled explicitly in implementation; do not fabricate finite metric values now. No claim of predictive test performance.

### L4. Can any radius meet both requirements? — §8

**Question:** After your spacing edit, can one radius keep the first two intended groups separate and make the right group viable? Commit “yes/no/need to inspect” and, if yes, a candidate radius before revealing the interval calculation.

Left group fixed `[0,.125,.25,.375]`. Middle group is `offset+[0,.125,.25,.375]`, offset from .625 to 2 in .125 steps. Right group is `5+spacing*[0,1,2,3]`, spacing from .125 to 1 in .125 steps. These bounds keep the three labeled groups ordered and the distant group's gap outside the relevant candidate interval. m=3 fixed for this specific derivation. Radius editable .125–2 in .125 steps. Learners may enter an allowed offset/spacing pair never shown in the two solved examples.

The earliest useful radius is `max(.125, spacing)`; the two dense groups merge at `offset−.375` under these bounds. Thus a radius exists iff `max(.125,spacing)<offset−.375`; show the half-open interval when it exists. Still compute the actual graph at the selected radius so the analytic thresholds are connected to core/border observations. Do not reuse this formula if later implementation expands the allowed positions beyond its stated ordering assumptions.

Baseline offset=.75, spacing=.75: empty interval. Equal-density null offset=.75, spacing=.125: interval [.125,.375), so .25 works. Changed practice spacing=.25: [.25,.375), diffuse endpoints border at .25. Default/null values are probed; changed interval is exact hand reasoning. Text alternatives list all points and show which two thresholds conflict. This local proof is stronger than a sweep that merely fails to find an epsilon on a coarse grid.

## Completion boundary for the next stage

Writing-stage evidence: exact small fixture calculations, seven real-data settings, a second small shape/unit probe, and mathematical derivations above. Sources and versions are in the design/provenance records. No rendered figure, functioning control, real browser screenshot, full-program output execution or independent implementation review is claimed here.

Implement only the representations that earn their place in the reading flow; F1–F7 are different explanatory objects, not a card-count target. L1–L4 have four distinct reasoning tasks. If a future implementation combines adjacent representations, preserve their contracts, provenance and first-pass placement. Record any scope change before deleting useful explanatory support. The reader must be able to learn the mechanism from the prose and static figures even without operating every lab.
