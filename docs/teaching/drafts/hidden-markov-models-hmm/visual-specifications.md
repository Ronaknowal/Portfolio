> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# HMM visual and investigation specifications

Content/specifications only. Read the complete manuscript and [calculated inputs](calculated-inputs.json); all toy calculations are from [hmm-experiments.py](hmm-experiments.py), all real sentences from [ewt-sequences.json](ewt-sequences.json). A graph, trellis, flow of fractional counts, duration bars and real labeled text serve different mechanisms. Do not flatten them into generic text boxes.

## Shared contract

Represent state by name, position and pattern as well as color. Supply exact numeric/text tables beside diagrams. Use keyboard-selectable cells and explicit form controls; dragging is optional. Labels remain readable when stacked on mobile, and wide trellises have a labeled scroll/table alternative. No answer relies on motion; honor reduced motion and avoid autoplay. Controls have visible focus/adequate touch targets. Announce compact result changes politely.

A graded prediction starts unset and is bound to the actual input/model/query/record revision. Submit before revealing; grade the computed quantity with its stated tolerance. Any relevant edit invalidates the prior response and reveal. Reset restores the named fixture and clears success. Construction tasks start unsolved and compute correctness from the resulting model/output, not selected preset names.

Probability editors enforce nonnegative normalized rows explicitly: edit all row entries then validate/apply, rather than silently changing an unrelated probability. Show structural zero edges distinctly. An impossible sequence has zero evidence and no defined conditional posterior; do not normalize it into a made-up distribution. Numeric smallness is handled with log-space or retained scaling factors.

Bound browser work to at most12 time steps/4 states/4 symbols for toy exploration. Real fits, EM trajectories and real-data predictions are selected from saved results; do not train a model or load unrelated topic datasets while scrolling. Exact toy recalculation is small. Native API examples are separately unexecuted and must not be portrayed as working browser backends.

## 1. From a state graph to an unrolled sequence

Home §1, inline before recurrences. Two-state graph has start probabilities, outgoing transition labels and separate emission rows. Unroll into four time columns with visible observation cards Walk–Shop–Walk–Clean underneath. Solid arrows connect successive hidden states; a different edge style connects each state to its observation.

Source mechanisms.model. Label generative direction versus inference question. Toggling the queried hidden time highlights its neighboring factors without claiming those are the only observed data that inform its posterior. Graph is a conditional-factorization diagram, not a causal claim about weather and choices.

Predict one joint path product before reveal; first provide Rainy→Rainy→Rainy→Rainy. Allow selection of all16 paths with factors from the actual model. The sum equals0.00933936 and the largest path is Sunny→Sunny→Sunny→Rainy with0.0031104. Those quantities remain distinct.

## 2. Sum, max and posterior trellis

Home §§3–5, progressively reused near each explanation. Use a consistent2×4 trellis, but change the aggregation operator and heading explicitly. Show incoming contribution arrows, selected predecessor only in max mode, and all summed edges in forward mode. Never draw a Viterbi predecessor from the larger prior cell without multiplying its transition.

Source mechanisms.main/observations/enumerated_paths/scaled. Forward ordinary rows:
[.06,.24];[.0552,.0486];[.005808,.027432];[.0075192,.00182016].
Maximum rows:
[.06,.24];[.0384,.0432];[.002688,.015552];[.0031104,.00093312].
Third-time Rainy must point to preceding Rainy (.02688 versus.01728 before emission); final backtracking produces [Sunny,Sunny,Sunny,Rainy]. Path posterior.3330420928 is its joint mass divided by total evidence.

Inset distinguishes filtered from smoothed rows. Rainy filtered [.2,.5317919,.1747292,.8051087]; smoothed [.1949427,.4338284,.2363160,.8051087]. Probability bars share0–1 scale and exact values; neither all curves nor all states should be called “confidence.”

### Investigation A — evidence arriving later

Initial fixture four observations above, time1 (the second observation) query. Prediction: if final Clean changes to Walk, will filtering and smoothing at time1 (the second observation) each increase/decrease/stay? Require separate answers; only then reveal the two timelines.

Contrast mechanisms.changed_future: past filtering is unchanged; time1 (the second observation) smoothing decreases .4338284→.3976241, and the full Viterbi path becomes allSunny. Keep “information available through time1 (the second observation)” shaded separately from “complete recording.”

Construction: initial time1 (the second observation) filtering=.5317919 and smoothing=.4338284; ask learner to edit only the final observation so time1 (the second observation) smoothed Rainy falls below.41 while time1 (the second observation) filtering remains fixed within1e−10. Walk is a verified solution. The original Clean fails. Grader actually runs inference and checks prefix unchanged; editing the model is disabled during this specific task. Reset unsolved.

Free exploration afterwards allows model and observation edits. Changed emission row [.2,.3,.5] alters evidence, posteriors and filtered history even though best-path joint mass remains.0031104; path posterior becomes.2571428571. This null in path joint score is a useful distinction from null in the whole model.

Optional missing-step contrast: replace middleShop with missing emission likelihood1 while retaining transition time. LastRainy≈.80278006 versus deleting that position≈.79532049. A missing card remains a time column, not a collapsed column.

## 3. Pointwise decisions can violate the graph

Home §5 after Viterbi. Three rows A/B/C and two time columns. Fixed prior[.4,.35,.25], transition[[0,.5,.5],[1,0,0],[1,0,0]], identical certain emission in each state. Only legal edges are shown; prohibited ones may be dotted on explicit inspection.

Source mechanisms.illegal_modes. Marginals first[.4,.35,.25], second[.6,.2,.2]. The pointwise modes A→A are not connected. Display path joint0, while Viterbi B→A has.35. Expected correct-state counts are1.0 for unrestricted pointwise modes and.95 for B→A. No conflict: different objectives/decision sets.

### Investigation B — repair a path, then change the evidence model

Prediction asks whether connecting the two highest marginal cells guarantees the highest-probability legal path. The graded part asks the actual joint probability of the selected A→A path, tolerance1e−10; explanation handles the general claim.

Construction starts at invalid A→A. Learner must choose a legal path with joint probability≥.3 by selecting both cells; B→A succeeds. A→B/A→C are.2, C→A.25. Grade current edge legality and product, not a named preset. Changed prior[.2,.55,.25] gives B→A.55 and makes the pointwise modes legal too. Require a new prediction when this changes. Total evidence remains1 because observations are uninformative; this is an explicit null despite changed path probabilities.

Do not add pseudocounts to forbidden edges to make every answer legal. Optional constrained Hamming decoding uses max-sum marginal rewards, not a renamed Viterbi product.

## 4. Fractional events become probability rows

Home §6. Two labeled recording strips [Walk,Shop] and [Walk,Clean]. Initial-state mass flows into a start counter, each edge into a transition grid and each observed symbol into its state-emission bin. The representation must show why counts are fractional but their totals are exact.

Source mechanisms.split_counts/joined_counts/split_update. Split totals2starts/2transitions/4emissions; joined1/3/4. Initial counts[.4814784662,1.5185215338]. Split transition grid[[.4083285842,.0731498820],[.9333224782,.5851990556]]. Updated A[[.8480723706,.1519276294],[.6146257774,.3853742226]], start[.2407392331,.7592607669]. One full EM update changes log likelihood−4.72804315→−3.52910814.

### Investigation C — which boundaries describe the recordings?

Prediction: show two recordings whose lengths sum4 and ask start/edge/emission totals before reveal. Then remove the boundary and require a new prediction. Recompute posterior counts for the changed sequence; do not merely add a fixed boundary edge to old counts.

Construction starts with four concatenated observations and no boundary. Task states that two independent two-step sessions were recorded. Insert the boundary after the second observation; numerical conditions are starts2/transitions2/emissions4 and the intended entity split. Other boundaries may have same totals but do not match the declared recordings, so both structure and counts matter.

Free edit observed categories, add a length-one session, or duplicate the entire dataset within max12observations. Duplication doubles all expected counts and log likelihood but preserves one-step parameters to1e−10. Length-one sessions add a start/emission and no edge; if there are no transitions at all, retain previous unvisited A rows. Null fixtures were executed.

## 5. EM history is a measured calculation, not a success animation

Home §6 guarantee discussion. Plot the actual recorded negative-valued log-likelihood on a labeled axis, increasing upward. Sources em.fits and em.uniform_start, data seed71,12×30 observations. Keep all41 checkpoints for seeds3/7/19 and4 for uniform.

Use an overview plus optional zoom into iterations1–40; initial extreme value must not flatten meaningful later differences. Initial/final pairs:
seed3−566.599897/−390.253507;
seed7−410.236336/−390.630410;
seed19−459.466552/−392.284393.
Uniform−395.500424 then−395.091859 and numerical plateau. Mark true generating-parameter score−393.562196 as a reference for this sample, not a performance ceiling.

Permit selecting a recorded start or inspecting a parameter row. Do not represent arbitrary new seeds as measured fits. Parameter-label permutation can reorder state labels while leaving evidence unchanged; explain index ambiguity separately from local-fit quality.

## 6. Numeric scale and duration are different visual problems

Home §7: parallel ordinary-product, log-sum and scaled-factor strips. Rare-event example400identical.01 factors has ordinary float0 but finite log−1842.068074. Display mathematical10^−800 separately from float64representation. Impossible-symbol example is genuinely zero/−infinity with no posterior. It is not the same outcome as underflow.

Home §9: discrete duration bars for d1–8 plus tail mass, with separate constant next-step exit probability and mean. Source mechanisms.duration. Defaults a.7 mean3.333; contrasts a0 mean1, a.95 mean20. Do not normalize the first8bars to sum1; tail a^8 is essential. Optional a1 card says absorbing/no finite mean instead of dividing by zero. Changed free a.8 gives mean5,P(D3).128,exit.2. A prediction about exit after1versus10 elapsed steps is graded using that same constant hazard.

## 7. Real text prediction: repairs and breaks

Home §8. Source real_tagging.configurations includes all40development sentences, parameters and per-token beliefs. Render actual token chips with stacked lexical/HMM labels and a concealed reference row. Original UPOS is available on reveal alongside its coarse mapping. Unknown symbols receive an explicit UNKNOWN label; never show their distinct spelling as a learned emission category.

A matched view holds smoothing fixed while switching decoders. Results .1lexical266/.1HMM268/1lexical266/1HMM270 out341; whole sentences8/9/8/9out40. Include majority216. Development selection and unscored reserve are visible once near the protocol.

Initial specimen dev index27, “Dear Nina ,”: lexical[NOUN,NOUN,OTHER], HMM[OTHER,NOUN,OTHER], truth[OTHER,NOUN,OTHER]. Contrast dev index3: “article” changes from correctNOUN to incorrectOTHER. Give source IDs from the actual records, not invented row IDs. Reveal10repairs/6breaks only after learner identifies at least one of each from the stored comparison.

Independent practice can request any different recorded sentence and ask which method has more correct tokens, binding answer to decoder/smoothing/specimen. It is an inspection task, not a browser retraining simulation. Optional free spelling edit requires inference with the fixed stored model and vocabulary; replacing one unknown spelling by another unknown spelling is an exact input-encoding null. Do not claim unseen edits have stored gold labels or grade them against fabricated truth.

## 8. Topology extensions

Home §10: small profile alignment diagram with separately labeled match/insert/delete states. Delete is silent and advances alignment position; insert emits without advancing the match column. This is an explicitly schematic illustration, not a fitted protein profile. Optional factorial diagram has two hidden chains feeding a shared observation node; indicate posterior coupling without implying factorized inference. These inline figures do not require another generic lab.

## Phase-two obligations

Execute the optional native program first; implement semantic topic-owned model/figure/lab files with bounded data loading. Verify mathematical parity, input-bound prediction and actual construction checks, incorrect/contrast/null/reset cases, all-zero evidence handling, sentence boundaries, source-ID alignment, numerical axis visibility, keyboard/mobile/reduced-motion and screen-reader tables. Formal independent correctness/learning review and browser/build checks remain pending.

