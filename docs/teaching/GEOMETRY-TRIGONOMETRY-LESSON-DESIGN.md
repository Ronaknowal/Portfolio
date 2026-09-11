# Geometry, Trigonometry & Coordinate Reasoning — design

Status: implementation and author verification complete, 2026-09-11; independent review/integration remain parent-owned. Mathematics45, stable ID `geometry-trigonometry-coordinate-reasoning`. The prior entry was planned, with no legacy body/program/output. Implemented source `src/learn/data/topics/geometry-trigonometry-coordinate-reasoning.jsx`. Parent registered publication and the individual blueprint. Title and route order are retained; next Counting, Combinatorics & Mathematical Induction. Final source identity/evidence: [verification](GEOMETRY-TRIGONOMETRY-VERIFICATION.md).

## Contract and prerequisite decision

Require only **Algebra, Functions, Exponentials & Logarithms**. Its actual completed body teaches units, signs, fractions, square roots, equations, function domains, composition, inverse branches and small two-equation elimination. Refresh these specific meanings locally. Do not require Vectors: its existing brief requires Geometry, so reversing that dependency would create a cycle and hide the intended foundation. Coordinate pairs and rotation component formulas are derived here without matrix knowledge. Optional compact matrix notation can only follow the fully explained scalar formulas; the learner can finish without it. Sets/Logic is the adjacent earlier route topic but is not needed as a formal prerequisite.

Scope is Euclidean plane geometry and practical coordinate reasoning, not all synthetic geometry, curved surfaces or full robot kinematics. State perpendicular equally scaled physical axes for the squared-distance formula. Begin with a floor-plan point, then a right triangle, then a rotating radius, then a fixed point described by moving axes. Units and conventions stay visible. Python checks are optional complete standard-library scripts with setup before the first example.

Beginner finish line: locate a point, find a distance, interpret a right-triangle ratio and convert an angle correctly. Intermediate: recover a direction/branch, solve a general triangle, distinguish rotation from a frame change and reconstruct a world point. Deeper connections: derive angle-addition identities from rotation, compose translated maps, diagnose nonuniform screen units and compute a two-link endpoint with relative joint angles. Calculus, 3D rotations, quaternions, camera projection and robot Jacobians are future owners.

## Exact inventory and scoped ownership

Ran `node scripts/build-curriculum-inventory.mjs --topic "geometry-trigonometry-coordinate-reasoning"` before design on2026-09-11 local date. The returned brief specifies Algebra only and active/passive planar frames. No incoming destination note exists. Read the unresolved bit-mask note: unrelated DSA ownership, not inserted here. Verified no existing topic body at the proposed path.

| Candidate / hurdle | Current evidence / best owner | Treatment |
| --- | --- | --- |
| Distances, right angles, areas, triangle similarity | Essential prerequisite to declared outcomes | Teach locally, including area scaling and a Pythagoras dissection. |
| Vector projection, general basis, dot-product spaces | Actual Vectors lesson sections2–4 already owns formal operations and depends on Geometry | Provide geometric signed components here; link deeper formalism without importing its prerequisite. |
| Arc length, radians and angle conventions | Existing Geometry brief directly promises them | Dedicated arc/radius investigation, exact radians-per-turn argument and explicit finite-sector formula. |
| Inverse-trig branch and atan2 | Required for real coordinate recovery; Python math primary contract inspected | Teach arcsin/arccos branches, all sine solutions and quadrant-aware polar recovery; zero radius has no direction. |
| General triangle laws and SSA ambiguity | Trigonometry title would otherwise leave an important practical gap | Derive cosine law from coordinates and sine law from equal area; show two valid triangles from one inverse-sine datum. |
| Rotating objects versus axes and translated origins | Geometry brief promises this; full Robot Frames topic requires Geometry+Vectors | Fully derive 2D component maps and inverse; route 3D/convention extension to Robot Frames. |
| Screen coordinates and unequal scale | Concrete application of units/orientation; camera projection has a separate owner | Teach an invertible affine screen mapping; no optical calibration claim. |
| Two-link planar mechanism | Forward/Inverse Kinematics owns full subject | One complete forward endpoint example uses taught angle addition; save relative-angle/branch extension to that owner. |
| Spherical/ellipsoidal distance and curvature | Differential Geometry and geodesy would require more foundations | Explicitly delimit Euclidean scope; no unverified globe-distance approximation. |

## Learning route and representations

| Hurdle | Local mechanism and worked result | Representation and learner action | Independent evidence |
| --- | --- | --- | --- |
| Position is a description relative to axes | P=(1,2),Q=(4,6) have displacement(3,4) and length5 | Labeled plane and right-angle construction; original point versus difference | Changed point pairs, units and wrong pixel metric |
| Squared distance is a geometric theorem | Four right triangles inside an(a+b)-square leave a c-square; area gives c²=a²+b² | Exact static dissection with leg and area labels, not an unexplained formula | Polygon areas/orthogonality and changed3–4–5/5–12–13 fixtures |
| An angle is independent of circle size | θ=s/r; full turn2π; arc length scales with radius and sector area with radius squared | Radius/angle arc lab, same fixed spatial scale; distinguish radians, degrees and length | Exact fractions ofπ and changed radius/arc cases |
| Ratios depend on shape, not size | Similar3–4–5 and scaled triangle share3/5,4/5,3/4 | Two overlaid triangles on a common scale; change scale, choose a new shape | Similarity invariants, complementary-angle naming, negative-scale excluded |
| Signed circle coordinates extend triangles | (cosθ,sinθ), both signs by quadrant, sin²+cos²=1 | Unit circle with projected components linked to sine/cosine traces | Cardinal values, special-angle exact answers, negative/full turns and zero tangent denominator |
| An inverse branch is not the whole solution | sinθ=1/2 at30° or150° modulo360°; atan2 knows both coordinate signs | Editable Cartesian bearing lab, radius and angle; compare quotient-only ambiguity | Quadrants/axes/origin/branch endpoints and reconstructed coordinates |
| Non-right triangles and incomplete data | Coordinate expansion derives cosine law; area derives sine law | Inline triangle with side/opposite-angle labels; two SSA triangles displayed together | Changed SAS and two-branch SSA examples checked against side lengths |
| Point versus frame motion | Active Rθ(P−O)+O changes point; passive R−θ(P−O) changes coordinates | Mode-separated frame lab, fixed world grid, named origins and component projections | Exact90° cases, general-angle reconstruction, distances and transformed displacements |
| Multiple conventions can describe the same object | Screen y down, different pixels per metre; two-link second global angleθ+φ | Inline screen mapping and two-link joint diagram beside complete worked equations | Changed scales/offsets, lengths and relative-angle endpoint calculation |

## Specific visual and interaction contracts

**PythagorasDissectionFigure**: outer square side a+b, four congruent right triangles of area ab/2 and central square side c. Use a=3,b=4 initially. The central vertices are(a,0),(a+b,a),(b,a+b),(0,b); their adjacent vectors have zero dot product and squared length a²+b². This verifies geometry independently of visual appearance. Label areas in squared units. The caption explains the dissection argument; no animation required.

**AngleArcLab**: radius1..3 in half-unit steps, sweep15..330 degrees in15-degree steps. Positive sweep is counterclockwise from the positive horizontal ray; SVG y reversal is rendering only. Controls update immediately. The physical scale remains fixed within the figure, so doubling radius doubles the drawn arc and quadruples computed sector area. State θ in degrees and radians, s=rθ and area=r²θ/2. The sector formula is restricted to a single positive sweep below a full turn. Predict before changing radius; transfer asks what changes and what stays dimensionless.

**TriangleSimilarityLab**: fixed shape presets3–4–5,5–12–13 and equal-leg triangle; scale0.5..3 by0.25. Original and scaled triangles share a drawing scale, which may adjust together to fit. Length labels/readouts give actual units; no cross-screenshot pixel-distance claim. Same angle and ratios are invariant under positive scaling; area changes by the square of scale. Opposite/adjacent names refer to a visibly marked angle. No negative or collapsed scale.

**CircleComponentsLab**: signed angle−360..360 degrees by15; radians are displayed and used by native APIs. Endpoint represents angle modulo one turn, but the angle trace preserves the chosen signed input. Show cosine horizontal component, sine vertical component and linked component graphs over−2π..2π. Tangent is undefined at exact odd90° controls, never an enormous displayed finite answer. Cardinal controls use exact known zeros; other values are explicitly rounded. Initial30° geometry connects to the derived special triangle. No differential-equation or angular-velocity prerequisite.

**BearingLab**: integer x,y drafts in−6..6, Apply validates all before replacement; presets cover quadrants/axes/origin. The origin is a valid point with radius0 but no bearing, not an invented zero direction. The chosen bearing convention is(−π,π], canonicalizing the negative axis endpoint; code/API conventions and signed zero are explained. Show actual point, displacement, projected legs and the quadrant. Keep atan(y/x) comparison educational, explicitly missing/ambiguous where appropriate. Invalid drafts preserve active geometry.

**CoordinateFrameLab**: fixed point P=(4,2) by default, origin O=(1,−1), angle30°. Controls have finite discrete values. Passive mode changes the frame and local coordinates while world P stays fixed; active mode rotates P about the selected origin in the fixed world frame. Labels explicitly identify mode and which point is moving. The local-to-world reconstruction is always displayed and checked. Translating the origin affects point coordinates but not the conversion of a free displacement by rotation alone. Mode switches preserve draft controls but recompute the correct active meaning; reset restores the complete stated fixture. Grid/axis arrows, world and local markers remain readable on280px content without shrinking tiny labels.

**SSA and screen/two-link figures**: inline calculated geometry near the relevant equations; all lengths/angles supplied, axes conventions stated. SSA shows both triangles only when both satisfy positive remaining angles and side constraints. Screen figure distinguishes metres from pixels and axis orientation; no screenshot dimensions are mistaken for physical coordinates. Two-link diagram names the relative elbow angle and its absolute orientationθ+φ.

## Programs and independent practice

Complete native scripts will cover distance/area and units; angle/arc conversion; similar-triangle ratios; signed circle values and periodic samples; validated polar recovery; general-triangle construction and ambiguity; active/passive/frame reconstruction; screen conversion and two-link forward geometry. Count is set by useful outcomes, not a quota. All setup/imports/fixtures/functions/output are included; no network, plotting package or hidden helper needed.

Practice progresses from hand lengths and angle units to similar triangles, signed components, all inverse-trig candidates, general-triangle ambiguity, translated frame reconstruction, a screen-coordinate mistake and an independently changed endpoint/measurement scenario. Each has a hint and complete explained answer with domain/units/limiting cases. A free-response application gets an example acceptable answer and checks. Native oracles independently enumerate or reconstruct the requested changed cases; repeating the worked fixture is not enough.

## Research record and review bounds

2026-09-11 local date: opened OpenStax Algebra and Trigonometry2e7.1(angles/radians),7.2(right-triangle ratios),7.3(unit circle/Pythagorean identity/special triangles),8.3(inverse branches),10.1(sine law/SSA ambiguity),10.2(cosine law). Read relevant definitions and derivations; final claim locations will be retained in verification. Own examples/figures/programs are not copies.

Opened Modern Robotics3.2.1 Part2 official supplement and inspected its transcript distinction between orientation, changing coordinates and rotating an object. It is a deeper video/transcript alternative after the scalar2D explanation and later Vectors lesson; do not silently import its3D matrix vocabulary as a prerequisite. Obtain the direct official video link from that page. Whole-video playback is not claimed.

Opened3Blue1Brown’s official Trigonometry Fundamentals page. Retrieved title/creator/date and video container identity; its retrieved text is mostly supporter names, not a substantive transcript. Do not claim that the video’s full explanation was reviewed. Prefer the actually inspected OpenStax written route and Modern Robotics transcript for detailed annotations; an optional direct basics video may be linked with this honest bound.

Final resource decision: did not add the metadata-only 3Blue1Brown item to the learner references. The actually read OpenStax route plus Modern Robotics direct recording `https://www.youtube.com/watch?v=6KIPusOv5fA` and its substantive official transcript give clearer annotated choices. The direct recording ID was resolved from the official embed; playback/cache retrieval was not claimed. Modern Robotics is expressly a deeper 3D alternative after Vectors.

## Implemented design refinements

Five distinct investigations serve arcs, similarity, signed circular components, bearings and active/passive frames. Five inline figures serve distance, Pythagoras dissection, ambiguous ray-circle intersections, unequal screen scales and a two-link mechanism. Nine complete standard-library programs and twelve independently changed practice tasks follow the actual outcomes; those counts describe this lesson only.

Actual phone reading led to multiline distance/degree/ratio equations, inward axis/trace titles and adaptive rotated-point labels, including a coincident-point label. The SSA figure includes the relevant radius-7 arc, with both positive ray intersections. A root review corrected a temporal Vectors reference and replaced an inaccurate CSS-pixel claim with the actual fixed-scale-within-view contract. Model grids validate both the quotient and reconstructed value so subnormal inputs cannot underflow onto the zero grid point. These are implementation evidence refinements, not changes to the mathematical scope.

The two outgoing destination notes are open for their receiving authors, with this origin's verified evidence linked. Their exact stable IDs were checked against the live catalogue. There was no Geometry incoming note at design time; the unrelated unresolved bit-mask proposal remains outside this scope.

Opened Python3.12 math primary documentation: sin/cos take radians; asin/acos have specified principal ranges; atan2 keeps both coordinate signs; hypot/dist compute Euclidean lengths. Execute on local Python3.12.14 and compare independent references. A floating-point API’s origin convention is not a mathematically determined direction.

## Verification and handoff plan

Native checks: exact Fraction dissection/area/coordinate90° cases; high-precision or independently constructed special triangles; adaptive/quadrature or complex-exponential references for changed angles as appropriate; independent numerical linear solves for rotated frames and reconstruction; side-length/area oracles for SAS/SSA; explicit periodic/endpoint/origin/invalid-input cases. Avoid using the same component formula as both implementation and sole oracle. Finite sampling verifies implementations, not all-angle identities; derivations establish those identities with stated assumptions.

Actual-font Edge1440/390/320: all controls, presets/reset/mode switches, invalid retention, keyboard and local code scrolling, all code/questions/outputs, all practice solutions and anchors, ordinary diagrams without operating controls, equation widths, labels/signs/arc directions, no lesson errors or document overflow. Open final representative screenshots and record exact hashes. Parent independent review and integration remain separate from author verification.
