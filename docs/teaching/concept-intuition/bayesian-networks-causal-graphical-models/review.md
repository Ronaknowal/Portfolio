# Bayesian Networks — concept intuition author review

26 September 2026. Entire production lesson read, including all ten practice solutions, current numerical fixtures, all deeper branches, and the existing live illustrations' source interfaces. This is a direct JSX lesson; numerical Wine data and original runnable programs remain canonical and unchanged. Two deeper transitions needed more than the existing opening intuition.

## Entire-lesson concept map

| Location / transition | Evidence and decision |
| --- | --- |
| Introduction: probability model versus causal mechanism | Alarm question, required vocabulary, constructed-data disclosure and route already supply motivation and assumptions. Retained. |
| §1 DAG vocabulary → CPT rows → one joint world | Existing state-editable WorldAssemblyFigure shows the actual five factors and changes the selected entries. Retained. |
| §1 normalization, local Markov property, parameter counts | Parent-before-child probability allocation and reverse summation explain normalization; reporting randomness and explicit count illustrate independence and storage. Retained. |
| §2 marginalization versus conditioning versus a best explanation | Full compatible-world sum, seven evidence cases and live evidence lab provide direct support. Explaining away and screening off have different computed outcomes. Retained. |
| §2 tiny executable enumeration | Standard-library program and cost connection retained without changes to its output. |
| §3 factors → variable elimination → order/width | Computed factor workbench and moral-graph geometry already show exact reuse, fill and size; four-step dependency warning, 16-GiB calculation and exactness argument retained. |
| §4 chain/fork/collider, descendants, all paths, faithfulness | Existing graphical path lab and parameter counterexample separate graph guarantees from actual numerical associations. Retained. |
| §4 Markov blanket | Alarm's earthquake-parent example connects directly to the earlier collider; retained. |
| §5 counts, Dirichlet smoothing, posterior mean versus MAP | Concrete complete/empty CPT rows already explain the count mechanism and estimation distinction. Retained. |
| §5 TAN, conditional information, spanning tree, additional parameters | Local interpretation plus computed strongest pair and actual model layout retain the benefit/cost of extra dependencies. All split/data provenance and executed code retained. |
| §5 probability scoring, missing measurements, measurement value | Real confidence comparison and measurement lab make the likelihood difference and hidden-value null visible. Retained. |
| §6 observation → intervention → adjustment | Existing population-mixture lab gives exact different weights, mechanism removal and positivity failure. Backdoor candidate paths and changed graph are worked through; retained. |
| §7 frontdoor identification | Computed conditions, two-tray visual and two failure variants already explain the two averages. Retained. |
| §7 three do-calculus rules | Formal statements existed, but only rule 2 had a small worked graph. Added three original surgery diagrams and explanations: irrelevant observation, action/observation exchange, irrelevant action. The deleted-edge test is distinguished from an actual erased effect. Full conditioning and ancestor restrictions remain. |
| §7 identification versus estimation; counterfactual coupling | Existing two structural worlds and abduction/action/prediction figure are sufficient. Retained. |
| §8 structure search, BIC, PC/FCI, equivalence, EM | Existing parameter-count link, equivalence orientations and hidden-count bridge explain the scope and limits; retained. |
| §8 junction tree, separator, running intersection | Added complete two-cluster numerical message flow, showing why the message is a function of shared B rather than one scalar. Added connection to elimination and changed-evidence invalidation. |
| §8 approximate inference and limitations | Mechanism table plus concrete rejection rate and immobile Gibbs example are already locally explanatory. Retained; no new speed or convergence promise. |
| §8 marginal/MPE/marginal-MAP | Existing four-mass counterexample separates world maximum from sum-then-max. Retained. |
| §8 neighboring representations and pgmpy | Existing HMM/CRF/undirected/Gaussian/probabilistic-programming bridges and exact state-order contract retained, with recorded program versions. |
| §9–10 practice/readiness/next topic | All ten changed tasks, computed solutions, readiness mapping and sequence links preserved. |

## New visual contracts

- `DoCalculusSurgeryFigure`: three independent constructed graphs, not a single model reused silently. Rule 1 uses X→Y,X→Z with X conditioned/fixed; rule 2 tests X→Y by deleting that outgoing edge; rule 3 starts Z→X→Y and fixes X, cutting its incoming edge. Native SVG positions, trimmed lines, arrowheads, explicit cut marks and HTML explanations; no color-only distinction. States full causal/noise/support assumptions and the restrictions not illustrated by the miniature. Small panels stack automatically.
- `JunctionMessageFigure`: positive, unnormalized factors φ(A,B)=[[2,1],[1,3]] and ψ(B,C)=[[4,1],[1,2]]. Left message [3,4]; unnormalized C masses [16,11]; partition 27; P(C=1)=11/27. Actual table cells produce the message and result in the component. HTML tables retain row/column identities at narrow widths. No chart implies measurement or interactivity.

## Research actually read

Read [Stanford CS228's junction-tree chapter](https://ermongroup.github.io/cs228-notes/inference/jt/), including variable elimination as messages, separators, running intersection and cluster-message formula. Adopted the local-to-shared-variable explanation using a new numerical example. Did not repeat its informal universal constant-time query wording. Read [Pearl's causal-inference paper](https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf), printed pages 2517–2519, especially the three graph-modification tests and disjoint-set conditions. Both are already annotated sources in the lesson; no new video-viewing claim.

## Verification boundary

Scoped JSX parse; exact message flow compared with brute-force joint enumeration; concrete binary distributions supporting the three displayed rule uses; trimmed arrow geometry and all ten retained practice blocks checked. Hashes and actual author checks in `author-checks.json`. Existing native fits, UCI data, fixture formulas and runnable sources unchanged. Browser/mobile and independent conceptual review pending root.
