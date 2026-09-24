> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Visual and investigation contracts

Content-first handoff only. Implement these forms in phase two, preserving manuscript placement and all necessary explanation. The lesson owns the question; controls should manipulate graph structure, probability mass or observed measurements rather than produce generic text output. Four investigations teach distinct mechanisms; this count is not a template.

## Common interaction and evidence contract

Every investigation begins with an **unset** prediction bound to an exact input snapshot and query. Offer meaningful alternative predictions plus a short reason. A committed prediction permits Reveal; after any relevant input or query change clear both committed answer and revealed comparison. Separate illustrative starting evidence from the new investigation answer so nearby text does not pre-answer every task. Comparison may retain a clearly labeled prior snapshot; reset returns original values and unsets prediction. No forced “correct” response before continuing.

Exact probabilities label computed model results. Wine scores/inputs label real observations and measured author experiment; edited specimens label counterfactual inputs. Color always has shape/text equivalents. Render math and table labels at readable contrast, keyboard-operable controls, visible focus, explicit input labels, no hover-only values. At phone width stack graph and numeric panels; tables may have local horizontal scrolling with pinned row labels, never viewport overflow. SVGs need accessible titles/descriptions and adjacent table/text equivalence. No timed animation, hidden auto-run, expensive global dependency or whole-module payload. Bound all enumeration; memoize only useful stable results. Browser rendering and accessibility are **not yet assessed**.

## I1 — evidence and alternative explanations (§2)

**Question:** when does a clue change a burglary explanation, and when is it redundant?

**Form:** small five-node network beside probability-mass columns for B0/B1. State controls on E/A/J/M offer unknown/0/1, with B fixed as the query. Editable root probabilities and child chance-of1 CPT cells update their complementary state0 values; show parent configuration labels. Probability editor0–1 with numeric input and meaningful steps, not a slider unable to reach small priors. Full-world preview shows which factors a chosen assignment uses.

**Prediction tasks:** compare E unknown→1 with J=M=1; compare John's false-call .05→.20 under calls; contrast with A1 fixed. Predict direction or equality before revealing. Toggle to J-only and make both J rows .4 to test informativeness. Learner may alter an actual CPT entry rather than only step a scripted trace.

**Model:** enumerate32 worlds; filter evidence; sum into two B masses; report sum and normalized posterior only if denominator>0. Dataset from network-experiments.py ALARM. Reject nonfinite/out-of-range entries. Zero evidence probability is an explicit undefined result with repair guidance, never fabricated0 or uniform. Unknown is not state0.

**Checked starts:** Bprior .001; J1 .01628372995; J1M1 .28417183536; J1M1E0 .34419959776; A1 .37355122828; A1E1 .00326842359; A1J1 unchanged .37355122828. All nodes/tables visible on demand, one focused table open at a time.

**Contrasts/nulls:** equal J rows yield J-only prior; changing a row unused by fixed A1 cannot change that posterior; hidden E evidence setting changes the comparison; all J probabilities0 with J1 is impossible. Fixtures should compute input-specific feedback, not hard-code “earthquake always lowers” across arbitrary edited probabilities. Preserve the exact mass table as an accessible output.

## I2 — active-path inspector (§4)

**Question:** has the observation set blocked every path between two variables?

**Form:** editable small DAG with arrowheads, named query endpoints and observed-node toggles. A path list highlights collider positions and observed descendants; an explanation names the blocker or explains why every segment passes. It is a structural lesson, not an automatic causal-discovery result.

**Inputs:** preset alarm graph and changed R→S←T,S→V→W graph; user adds/removes named nodes/edges up to8nodes/12edges, edits endpoints and observation set. Validate no directed cycle/self-edge/duplicate, distinct existing endpoints and neither endpoint observed. Reject invalid edits beside the control without destroying prior valid state. Keep node positions deterministic, no physics loop.

**Algorithm:** enumerate simple undirected paths; for each internal node identify collider by the two incoming arrowheads on this path. Compute ancestors of observation set (including observed nodes). Noncollider observed blocks; collider outside this ancestor set blocks. Empty path set means separated. For sets of endpoints, extend only if needed; the initial UI uses a single pair. Enumerating within8nodes is bounded; no factorial unbounded arbitrary-graph editor.

**Committed alternatives:** “independence is guaranteed by this graph” versus “dependence remains possible.” Do not call d-connection proof of correlation. Topology/observation changes clear prediction.

**Exact contrast:** B→A←E separated with none, active with A or descendant J. Adding B→K→E creates a second active path without observations; observing K blocks that path but J observed keeps the collider route active. Removing J from observation set leaves both blocked when K observed. Correctly labeling the graph's conclusion does not depend on CPT values.

**Null:** changing a probability while topology remains fixed does not change d-separation; numerical independence can still change. Show the separate constant-alarm-rows counterexample with an explicit “extra independence” label. Dataset active_paths in author code is the bounded reference; phase two must separately verify checker and invalid-input behavior.

## I3 — missing-measurement inference (§5)

**Question:** how does a selected observation change a fixed probabilistic model's class belief?

**Form:** measurement cards with original raw values, fixed training-median bin and hidden/revealed status; cultivar distribution bars; graph highlighting only relevant CPT terms. Show the selected specimen's identity and observed provenance. Make “missing” visually different from “below median.”

**Model:** trainingModels.TAN from saved JSON, fit106; never final-refit NB. Support all36 validation specimens, all16 measurement masks, numerical edited values and all compatible binary states. Exact inference enumerates at most16 feature configurations for3classes; no browser fitting/download needed. Learned CPTs, medians and class labels are fixed. Actual cultivar stays hidden until Compare with recorded label; no test specimen is selectable.

**Prediction:** choose which candidate reveal will change leading class or which gives lower resulting entropy, and state reason; compute entropy in nats with0log0=0 only after reveal. Offer a neither/equal choice where appropriate. Optional labels explain that single-instance entropy reduction is not guaranteed for every observation. Changing specimen, mask or visible value invalidates committed answer.

**Checked fixture:** specimen156 [13.17,5.19,.63,7.9]. No evidence [.330275,.403670,.266055]; alcohol [.593532,.093197,.313270]; flavanoids [.032318,.466794,.500888]; both [.049898,.129279,.820823]; all [.022835,.013016,.964149]. Specimen80 differs meaningfully: alcohol alone [.086039,.691710,.222251], flavanoids [.625863,.341047,.033090], both [.191514,.766083,.042404].

**Learner edits:** cross alcohol threshold13.05 from13.17 to12.9, then compare below-threshold12.8 as a null. An edit within any unchanged bin is null. Hidden input edits are null until revealed. All-hidden returns model prior. Label edited measurements as hypothetical; restore-measured returns CSV values. Do not treat this exploratory trainer as validated test acquisition, missingness robustness or the selected final model.

## I4 — observation weights versus intervention weights (§6)

**Question:** does changing who receives a procedure change its effect or the composition of observed groups?

**Form:** two horizontal population mixtures, low/high-load strata visibly proportional to their weights. Graph above has intact assignment arrow for observation, cut incoming arrow for do. Beside each stratum show conditional outcome risk and its weighted contribution; final rows show two risks and their difference.

**Inputs:** P(Z1) fixed.5 for starting lesson; editable P(X1|Z0),P(X1|Z1) and four P(Y1|X,Z) entries in0–1. Starting assignment[.2,.6], outcomes indexed[x][z] [[.01,.10],[.05,.20]]. Support optional P(Z1) editor only if label/positivity logic handles zero strata consistently.

**Predictions:** change assignment to[.4,.4] and independently predict each difference; then edit outcome x1z0 .05→.01 and predict both changes. Bind prediction to selected comparison not the starting chart.

**Exact:** starting obs risks[.04,.1625], difference.1225; do risks[.055,.125], difference.07. Random assignment yields obs=do while unchanged causal mechanisms retain.07. Changed low-load service response gives obs.1525−.04=.1125,do.105−.055=.05. This task is not engineered to imply beneficial service.

**Support/nulls:** [0,1] creates no overlap; show observed risks[.01,.20] but mark adjustment-from-observed-data unavailable. The fully specified model's intervention still computes.07, separately labeled oracle-model calculation. If an observed treatment group has total mass0, its risk/difference is undefined. Assignment changes alone cannot alter oracle effect; outcome-table changes can. No causal claim about empirical deployment.

## Inline figures and placement

- **F1 world assembly (§1):** arrows B/E→A→J/M, probability strip .001×.998×.94×.9×.7=.0005910156; state0 complements visible. Distinguish this single joint cell from a posterior.
- **F2 factor workbench (§3):** input-factor scopes, sumE and sumA, exact four-cell table from elimination JSON, final unnormalized masses and normalization. Step trace optional; no unnecessary second lab. Axis unions/sums must be mathematical, not visual arrows implying causal effects.
- **F3 elimination geometry (§3):** undirected chain A–B–C–D. Compare endpoint elimination (largest initial/intermediate scope2,4binarycells) and B-first (scope3,8cells); fill A–C shown. This is constructed operation/storage accounting; no claimed runtime benchmark.
- **F4 measured model contrast (§5):** 36 validation row paired true-class probabilities and log-loss contributions for training NB/TAN. Compute from retained complete inputs/model; no test fitting, tune or ranking interaction. All rows available, original ordering with optional stable sort, exact values. The currently saved JSON gives all inputs/IDs/models for computing both; phase two can materialize this bounded view. Global scores .162922/.185233 displayed with same 34/36accuracy; show actual exceptions.
- **F5 frontdoor nested averages (§7):** graph with latentU; two inner mixtures giving .225/.700, then outer .9/.1 versus .1/.9 weights yielding .2725/.6525. Reveal original latent-model result separately, not as observed-data input. Caption all three path/support assumptions.
- **F6 same units across worlds (§7):** SCM A two rows [0,0],[1,1], SCM B [0,1],[1,0] for Y0/Y1. Unit identities persist; .5column averages align while row changes differ. Highlight abduction for observed X0Y0, then alternate X1Y0 as independent practice. Static paired matrix sufficient.
- **F7 equivalence gallery (§8):** three noncollider chain orientations with same skeleton and independence X⊥Y|Z; collider separate with unshielded-collider annotation. No spurious “exact DAG recovered” status from observational scores.

## Phase-two acceptance and reuse

Reuse unchanged author calculation outputs as content fixtures, not proof of production correctness. Execute exact displayed programs, including the pgmpy route with declared installed version; validate library CPT state order and agreement with explicit sums. Independently review probability arithmetic, collider descendants, causal assumptions/positivity and counterfactual pairing; test at least changed/null inputs above. Implement semantic topic-owned figures/models/labs and lazy registration. Inspect full lesson desktop/phone, actual numerical contrast visibility, keyboard/reader semantics and no overflow. Run author and independent learning-experience checklists separately, then record actual phase-two evidence before publication. No build, browser, native-library or independent review pass is claimed by this written specification.
