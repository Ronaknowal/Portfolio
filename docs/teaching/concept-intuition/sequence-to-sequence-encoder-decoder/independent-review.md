# Encoder–decoder sequences — independent concept review

26 September 2026. Reviewer: classical_representation_intuition, separate from author. Complete manuscript including full model/search program, eight practice solutions, generated learner text and generator replacements, ScheduledPrefixFigure and CSS inspected. Root owns browser/integration review; no actual learner trial conducted.

## Entire lesson map

| Location | Source-level assessment |
| --- | --- |
| §§1–2 timelines and arrays | Different source/output lengths motivate two timelines. Request, embedding IDs, BOS/EOS/PAD, supported outputs and shifted targets each have a role; source packing and target masking are separated. |
| §§3–4 context, probability and training | Complete scalar encoder/decoder calculation precedes likelihood, token versus sequence weighting and joint gradient. Teacher forcing uses a previous reference token, not the current answer; detaching context changes only the derivative route. |
| §5 search | Exact complete tree establishes greedy versus beam. Candidate states, ties, caps, ended retention and normalized-score sign convention remain explicit; wider search is not promised better task quality. |
| §§6–8 experiment, full code and diagnosis | Actual grouped UniMorph split, strong suffix rules, neural generalization failure and limited beam gain remain. Complete model, batching, training and saved inference are retained; source edits and target edits follow different dependencies. |
| Scratch reuse and §9 | Explicit protocol reuses the native/manual GRU owner. Context variants and source-reversal path counts introduce attention naturally. New scheduled-prefix joint table makes the changed objective visible; expected-reward derivative distinguishes sequence objectives from argmax and smoothing. New stopping counterexample shows why changing length scores changes valid bounds. |
| §§10–11 | All eight changed exercises and solutions, batching-state task and annotated resources inspected. |

## Ten-item learning-experience checklist

1. **Route:** task → timelines → arrays → bridge → likelihood → generation precedes optional objective/search details.
2. **Cautions:** score versus correctness, empirical length effect versus universal limit, and model versus search are tied to actual examples.
3. **Data:** actual lexical source/split and unfavorable model result retained, without invented recovery curves.
4. **Investigations:** alignment, scalar bridge, probability tree and frozen source/prefix changes are distinct immediate interactions.
5. **Representations:** new two joint-mass tables expose lost prefix dependence directly and state the fully replaced limiting assumption. Responsive CSS read; browser evidence separate.
6. **Connections:** recurrent owner and next attention lesson keep the learning sequence; generator replaces historical prepared-owner status with current learner links.
7. **Code:** complete scratch protocol, ordinary GRU route and beam ownership remain. Unchanged program source is checked by the generator before disclosure replacement.
8. **Practice:** all eight tasks change shifts, masks, search, grouping, metrics, caps or experiment design; answers retain reasoning.
9. **Evidence:** measured fit records retained; no new native or browser claim made here.
10. **Transitions:** advanced contexts, source reversal, scheduled sampling, expected reward, stopping bounds and deployment contracts were reviewed as well as the opening.

Independent checks use unequal replacement-prefix probabilities, a different two-answer reward, four source alignments and a changed normalized-score stopping counterexample. No source-level blocker found. References assessed in context; no new web/video inspection claimed. Accepted at source level with root browser/integration separate.
