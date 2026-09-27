# Depthwise and dilated convolutions — independent concept review

26 September 2026. Reviewer: classical_representation_intuition, separate from author. Full manuscript, seven practice solutions, reference annotations, generated learner text/generator replacements and DilationIntervalFigure/CSS inspected. This is source review; root owns browser/integration evidence. No actual learner trial conducted.

## Entire lesson map

| Location | Source-level assessment |
| --- | --- |
| §1 spatial lanes, mixing and training | Complete forward example and simultaneous update explain both factors, shared uses and negative mixing. Shapes and cross-correlation convention are explicit. |
| §2 budgets and rank | MAC definition, one-output counterexample and latency boundary remain. Rank-one identity counterexample, multiplier, per-input rank and spatial-axis versus channel/spatial factorization are distinct. Intermediate nonlinearity invalidates the linear kernel identity. |
| §3 stencils and coverage | Ramp's equal sums are exposed as a fixture property. New placement enumeration explains floor and +1. Field/jump and exact reachable sites distinguish outline from coverage. New shifted-interval picture correctly uses integer adjacency, not continuous overlap; the construction is explicitly unbounded and stride-one. |
| §4 code, bias and normalization | Complete matching native pair and effective kernel remain. Bias composition and fixed-BN folding include the mean shift; singleton training BN is a separate case. |
| §5 mobile, segmentation and causal variants | Signed expansion/reconstruction explains the linear bottleneck without guaranteeing invertibility. Width, resolution and stride-sensitive cost remain. Hard-swish, output-stride units, parallel ASPP versus serial rates, global summary, V3+ skip evidence and causal availability all retain reasons. |
| §6 and scratch ownership | Actual six-fit compression experiment and failed cases remain. New input-dependent diagonal example shows why best weight approximation need not minimize task loss. Exact SVD-owner reuse, rank energy and heterogeneous grouped layout are explicit. |
| §§7–9 | Diagnosis distinguishes approximation, geometry, implementation and deployment measurement. All seven changed practices and complete solutions inspected. |

## Ten-item learning-experience checklist

1. **Route:** spatial/channel questions precede count/rank, then sampling and applications.
2. **Cautions:** claim-specific assumptions accompany the relevant derivation; no added generic warning block.
3. **Data:** actual compression outcomes, including severe loss, are preserved; constructed geometric and sign examples are labelled.
4. **Investigations:** six retained specialized controls expose local terms, rank probes, sample sites, coverage, context and frozen inference immediately.
5. **Representations:** new common-axis support picture displays gaps explicitly with labels and a text equivalent; CSS uses neutral background and explicit text fill.
6. **Connections:** preceding convolution/SVD owners and following ConvNeXt roles are clear; generator updates the historical prepared prerequisite wording for learners.
7. **Code:** complete native, direct-loop and factorization routes remain; no training campaign or optimizer was changed.
8. **Practice:** seven tasks change values, budgets, units, sampling and experiment design, with independent reasoning.
9. **Evidence:** source/CSS reviewed; no screenshot, actual learner or new native-run claim.
10. **Transitions:** optional interval proof, bias folding, mobile variants, parallel context and compression objective were reviewed, not just the introduction.

Complementary checks use a different covered interval and stencil shape, signed expansion over positive/negative inputs, and changed diagonal approximation probes. No source-level blocker found. References assessed in context; no new web/video reading claimed. Accepted at source level with browser checks separate.
