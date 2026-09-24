> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# CRF visual and investigation specifications

Status: content specifications, 12 September 2026. No browser visual or lab has been built. Consume [lesson.md](lesson.md), [checked-results.json](checked-results.json) and [data provenance](data-provenance.md). Numerical inputs are exact constructed factors or recorded real-data outputs, never illustrative performance measurements. Root owns the phase checkpoint.

## Shared interaction and rendering contract

The design uses label trellises, path-mass bars, count balances and span rules because those are the topic's actual objects. Reuse accessible controls but do not reduce these to one generic table with a slider.

Every investigation begins with prediction `null`, distinct from a default answer. A learner chooses and commits a prediction before Apply; the commitment stores the entire active input tuple and prediction question. Editing any factor, mode, length or target invalidates the prior result and marks its prediction stale; it never grades a prediction made for different inputs. Keep editable versus applied factors visually distinct. A deliberate “Explore without a prediction” action may reveal results but produces no correctness mark. Reset restores the documented initial factors and removes prediction/feedback. Feedback reports the computed comparison and its causal reason, then offers a changed-input transfer. No autoplay or random changes.

Use text labels A/B and token indices in addition to color. Keyboard-accessible numeric fields and buttons are required; drag controls are enhancements. Provide a table of exact factor, message, path, probability and prediction values. Each schematic has a descriptive caption and text equivalent. At 320 CSS px, keep a two-column label trellis vertically arranged by position rather than shrinking an entire long SVG. Each numeric field retains visible units or “dimensionless factor/log score/probability.” Fit factor labels at ordinary text size; scroll only an intentionally wide exact-value table, with a named region. Manual steps respect reduced motion. Announce a concise completed calculation through a polite status region; avoid speaking the entire matrix after every keypress.

Supported generic lab factors are positive finite numbers in `[0.125,16]`; log scores are computed after validation. Length is fixed at 2 for the hand investigation. Label bias uses positive branch factors `[0.001,10]`. There is no arbitrary-code execution or training in the browser. Real outputs are read-only recorded predictions; all displayed specimens come from the packet. Bound computation to four paths and two labels in the first investigation. A separate span figure uses five labels and at most five tokens. Browser loading, keyboard/layout inspection and formal model review remain phase two.

## F1 — observed input and output factorization (§1)

- Question: how can a factor inspect the whole sentence without coupling every output label to every other label?
- Visible input: `Maya | Chen | joined | Cedar | Labs`, five observed tokens. Output circles show `y1` through `y5`, with a legend of BIO names and one example strip `B-PER I-PER O B-ORG I-ORG`.
- Draw unary input-score rectangles beneath token columns and pair-factor squares between output circles. A thin bracket labeled “observed sentence x” points to the score construction; an output factor's edges connect only its label arguments. Edge placement represents factor scope, not causal direction or probability magnitude.
- Caption: the output chain is local; each score can use available observed context. Both noun names here are invented, not dataset annotations.
- Text equivalent: input tokens observed; output labels unknown; each unary score depends on one candidate label plus x; each pair score depends on two neighboring labels.
- Narrow layout: token and label strips can break into two rows, with a continuation marker and the broken pair factor repeated as a connector, or use a vertical chain. Never imply a missing edge at the wrap.
- Phase-two check: five labels, four pair factors, legal BIO strip and edge scope; diagram text introduces every label before use.

## F2 — four-path normalization ledger (§2)

- Exact `exp(emissions)=[[3,1],[1,2]]`; pair factors `[[1,4],[1,1]]`, rows previous and columns current.
- Four path rows AA, AB, BA, BB with factor products 3,24,1,2, probabilities .1,.8,1/30,1/15. Partition total30. Paired strips highlight the chosen unary cells and edge cell using shared label identity.
- Horizontal mass bars share x-range0–30, zero baseline and visible values. Do not set each bar's independent maximum. A grouping brace collects AA+AB=27 and BA+BB=3 for first-position marginals .9/.1.
- Caption: normalization compares whole paths once; marginalization sums selected paths. Table includes exact fractions, with displayed decimals rounded6 places.
- At phone width, put multiplication below each path strip; keep 24 visibly much longer than3 without scaling labels. At desktop use a maximum chart width around700px and normal text sizing rather than scaling glyphs with a huge viewBox.
- Checks: mass sum30; probability sum1; group sums27/3; bestpathAB; x-range contains baseline andlargestmass. Rendering perceptibility still needs actual screenshots.

## I1 — edit the chain, compare sums and winners (§3)

### Learner question and state

Question: when can a pair preference overturn a locally attractive label, and how is best-path probability different from total mass? Use F2's two positions and two labels. The model has four editable unary factors and four editable pair factors. Initial values are F2. The independent comparison always uses pair factors all1 with identical unary factors; this is a scientifically matched baseline, not separately editable text.

The first question records a choice `same winning path` or `different winning paths` for independent and chain decoding. The learner edits at least one factor, then explicitly commits the prediction for that input and presses Calculate. Outcome comparison uses ordered argmax tie-breaking A before B; expose “tied best paths” separately and compare the full maximizing sets for correctness, so an arbitrary tie-break never masquerades as a structural difference. The prediction label should ask whether the **sets of highest-scoring paths** agree when ties occur. A second optional recorded question asks which of AA/AB/BA/BB will win, allowing multiple selections for predicted ties.

The visible trellis has first-position A/B, second-position A/B and four connecting edges. Selecting a path highlights the two unary factors and pair factor; an accessible button list supports the same selection. A sum/max mode displays **both** values side by side at the second position: sum of incoming masses, largest incoming mass, and largest path's backpointer. Do not label the max as alpha. The path ledger and probability bars derive from one applied model state.

### Exact fixtures, checked in author Python

| Fixture | Unary factors | Pair factor AB (others1) | Independent winner / Z | Chain winner / Z | Teaching consequence |
| --- | --- | ---: | --- | --- | --- |
| Initial | `[[3,1],[1,2]]` |4|AB /12|AB /30|Same winner can have a different probability distribution |
| Contrasting edit |`[[3,1],[6,2]]`|4|AA /32|AB /50|Pair reward reverses the local choice |
| Strong unary null |`[[3,1],[12,2]]`|4|AA /56|AA /74|A preference need not change the winner |
| No interactions |Any valid edited unary|1|identical|identical|Independent factorization is exact |
| All equal tie |all unary1|1|allfour /4|allfour /4|Probability .25 each; argmax sets, not accidental tie order |

Contrast masses are `[18,24,6,2]`; chain node marginals are `[[.84,.16],[.48,.52]]`. The best path stays AB while its share falls from.8 to.48. Strong-unary masses `[36,24,12,2]` sum74. The null says **winner** stays same; it must not claim probabilities are unchanged unless all pair factors equal1 or a global common factor is applied.

### Feedback and transfer

Show the saved prediction, observed set comparison, path scores and the precise changed factor. For the contrast: “AA's mass is18; AB's is24. The AB pair factor outweighs A's stronger second-position input.” For the no-interaction null: “With pair factors1, every path mass is a product of unary factors; the normalizer factorizes.” State how total mass and max differ by actual incoming alternatives.

Transfer asks the learner to construct any bounded input where the independent and structured paths disagree **in the other direction**, or to make a tie deliberately. Do not prefill a solved answer. Accept by calculated argmax-set condition, not exact factor strings. A teacher-visible example is first unary `[1,3]`, second `[2,1]`, pair BB16 with others1: independent BA mass6, chain BB48. This constructed example must be checked in phase two before shipping if exposed as feedback; it is hand-derived here.

### State/verification boundary

Editing changes draft controls only until Apply; all computed displays show applied factors. Reject nonfinite/out-of-range input next to its field, retain last valid visualization and never silently clamp. Reset clears applied comparisons and predictions. Test normalization, enumeration against independent dynamic recurrence, global scale invariance, tie sets, input invalidation, keyboard edits and narrow/table equivalents. Keep any extreme-score handling outside displayed instructional code. The data/calculation script checked initial/contrast/strong-unary fixtures; rendering and interaction are deferred.

## F3 — forward/backward join and count balance (§§3–4)

F3a, messages: initial prefix masses `[3,1]`, suffix masses `[9,3]`, products `[27,3]`, commonZ30. Show a cut through the chain at position1. Left arrows collect prefixes; right arrows collect suffixes; the joining circle multiplies them. At position2 forward masses `[4,26]` and backward `[1,1]` yield `[4/30,26/30]`. Edge AB has mass24, probability.8. The caption explicitly links grouped path sums fromF2 to message multiplication here.

F3b, training: model expected AB count.8, goldAB count1, observed-minus-expected+.2. For goldAA, observed0 and gradient−.8. Use two aligned number lines0–1, a subtraction bracket and a signed update arrow. A selector of gold path is a figure inspection aid, not a separate scored investigation. It must say parameters are held fixed. If showing regularization, put a third shrinkage term `−lambda*w` beside rather than inside observed count.

Text equivalent contains all products, labels and count differences. On narrow screens place the two directions above/below the join. All data comes from `tiny` in checked-results. Verify node/edge marginals separately, and finite difference of logZ with respect to AB score (.7999999998 versus.8, tolerance1e−8 recorded). Do not graph an optimizer trajectory that has not been measured.

## I2 — where local evidence disappears (§5)

Initial first-branch weights `[.5,.5]`, each branch one continuation, later factors `[.01,1]`. Draw two routes that **do not merge until after normalization**. The local view places a separate normalizer inside each branch; the global view brings complete path masses to one normalizer.

Editable entities are both later compatibility factors, `[.001,10]`, and optionally the first-branch probability `p` in `[.05,.95]` with the second fixed to1−p. Default prediction is unset. Record the expected direction of the final probability of routeA when moving from local to global normalization: decrease, unchanged, increase. Key it to `(p,a,b)` and compare tolerance1e−10.

Calculations: local `[p,1−p]`; global `[p*a/(p*a+(1−p)*b), (1−p)*b/(...)]`. Display both private local denominators `a/a` and `b/b`, not merely a changing bar. Probability bars use0–1 and show .5baseline, exact numeric labels. At the default, `[.5,.5]` versus`[1/101,100/101]`; reversed factors `[1,.01]` reverses the global result. Equal factors `[1,1]` return `[.5,.5]` for both. Equal factors with `p=.3` return `[.3,.7]`, demonstrating a nontrivial null. Values are exact algebraic substitutions, additionally compare formula to enumerated path products in phase two.

Reset sets p=.5/a=.01/b=1 and clears the prediction. Apply is required; changing an input invalidates an earlier prediction and returns to editable state. Feedback explains that the compatibility disappears inside a one-exit state's denominator, while global normalization compares both completed routes. Transfer asks the learner to make global routeA probability exactly.5 with p=.25; any positive `a=3*b` inbounds qualifies. Example a3/b1 yields.5. Keep the qualifying relation hidden until answer reveal.

Boundary: constructed two-route model, not an empirical MEMM/CRF performance comparison. The initial branch model is held fixed; input has not been copied into the earlier local classifier. No claim that all directed models share this property. At mobile width place local/global panels vertically and keep route identity A/B in text. Test numerator/denominator agreement, reverse/default/null/changedprior, equality tolerance, invalid fields, prediction keys and reset.

## F4 — legal spans and preference overflow (§6)

Five BIO states plus start. Show permitted predecessors of selected I-PER and I-ORG, start prohibition, and valid strip `B-PER I-PER O` versus invalid `O I-PER O`. A second compact one-token score table `[O:0, B-PER:0, I-PER:10]` contrasts finite preference and an actual start mask. With the mask, show two legal maximizers and omit illegal I-PER from the normalizer. Do not substitute −10000 for −infinity without declaring a bound-dependent approximation.

Use dark invalid edges with a cross symbol, not red alone. Selection controls inspect rules rather than claim to learn parameters. The text list fully states BIO start/type conditions. The phase-two implementer must test every allowed/forbidden edge including cross-type I continuation, all-invalid input handling, and the exact legal partition. No `all_possible_transitions` flag may be used as an alternative legality engine.

## F5 — real sentences, errors and metric definitions (§7)

Exact dataset `ewt-sequences.json`; outputs `checked-results.json.real.chain.{dev,test}.outputs`. Default split dev; select an actual sentence ID. Display tokens/gold/prediction in an aligned strip with error markers and a selectable probability triple. Column order is NOUN/VERB/OTHER, indices 0/1/2. The scores for the chosen neighboring pair come from `real.chain.transitions`; the retained packet does not store token input-score matrices, so omit those until phase two records them from actual fitted parameters. Never reconstruct fictitious logits from marginals.

Dataset selection is inspection, not a new scored lab. A read-only comparison shows development correct counts 269 versus 281 of 341; exact sentences 11 versus 12 of 40. Final chosen-chain test counts are 293/370 and 7/40. Confusion matrix true rows are `[[49,1,26],[8,42,23],[17,2,202]]`; axis labels must declare true/predicted. Use dots on token accuracy 0–1 with counts and paired annotation, not a truncated axis that exaggerates gain. Use a separate exact-match metric row rather than one undifferentiated accuracy series.

The caption describes the small r2.16 short-sentence extract, coarse manual tag mapping and held-out sentence unit. A result panel provides original sentence ID/UPOS via expandable details, a license/provenance link and download. Do not load all 200 sentences into the global catalogue; when implemented only this topic consumes its data. On a phone use one token per row with expandable inspection; never wrap gold and prediction independently from tokens. Verify each displayed prediction equals its record, probabilities normalize, confusion counts and metric denominators match, and at least one actual error is visible in the default development selection. Actual perceptibility requires deferred browser work.

## F6 — neural data flow and generalized scopes (§§8–9)

Neural diagram: input IDs → encoder `(B,T,H)` → linear input scores `(B,T,K)`; a separate pair matrix `(K,K)` enters the chain. The training branch `logZ − gold score` returns gradient arrows into the encoder and pair matrix; the decoding branch max/backpointers leads to a label strip. A padding mask enters the normalizer, gold score and decoder; a recurrent encoder's padding behavior is a separate upstream contract. Scores are unnormalized, with previous rows/current columns for pair orientation.

The optional scope figure compares a chain, one added skip edge, and a segment factor over three tokens. Its circle/square legend matches F1. Label the closed cycle and the subset of labels in a segment; geometry represents factor involvement, not measured runtime. Caption: graph factorization determines which inference algorithm is valid. No animated model training or unmeasured speed curve is required.

## Implementation continuation

Use semantically topic-owned CRF figure/lab/model files under the code standard; choose actual destinations after inspecting consumers. Only this lesson imports its own computations and small real-data slice. Preserve the ID, current module order and publication until the authorized finish. Resolve representation improvements with recorded reasons. Phase two must execute complete displayed programs, check CPU results/library alternatives actually used, verify hard masks and independent enumeration/gradient risks, then perform formal independent correctness and learning-experience review, responsive/keyboard/browser closure and integration. A written specification is not evidence that those checks passed.
