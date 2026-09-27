# Revision 4 — concrete intuition figure contracts

The previous sixteen inline figures, four manipulable labs and lazy complete-program readers remain implemented. This revision adds four static explanations at the point where their mechanism first becomes useful. Static diagrams do not imitate sliders or add prediction gates. All use black/charcoal/amber with text and shape distinctions, meaningful captions and adjacent interpretation.

## 1. Missing entrance note

- Location: opening familiar problem, before architectural terms.
- Question: was earlier information merely supplied once, or is it accessible to the current query?
- Encoding: five ordered messages, positions 0–4. Position 0 contains the east-entrance fact; positions 3–4 alone belong to the current two-message window. Dashed old-history edge versus solid amber current-window edge; literal status text accompanies every row.
- Claim: a direct read of the current window cannot consult position 0. No invented learned answer or guarantee that a memory mechanism recalls it.
- Small-screen layout: vertical timeline at every width; wrapped text in flow, no fixed-height boxes.

## 2. Weighted read as six shares

- Location: §2 before the score equation.
- Data: values [2,4,8], positive relative supports [2,1,3]. Supports total six; weights [2/6,1/6,3/6]; contributions [4/6,4/6,24/6]; output 32/6.
- Encoding: six equal-area cells labeled A,A,B,C,C,C, with both labels and border/fill differences. Contribution lines pair each source's value with its exact share; all arithmetic remains visible.
- Claim: shares represent normalized influence, not six observations or evidence that one source is factually reliable.
- Verification: fixture weights use the same stable-softmax helper, independently checked against rational arithmetic. Six cells remain six columns even at 320px; contribution lines wrap in ordinary HTML.

## 3. Silent step: input gate versus retention

- Location: §4 immediately after the surprising fact that a zero input can still decay old state, before RG-LRU notation.
- Data: initial state .6, current input 0, base .8 and exponent scale 8. Ordinary r=.125/i=1 gives retention .8 and state .48. Input-only closure r=.125/i=0 also gives .48. Mathematical hold limit r=0 gives retention 1 and state .6.
- Encoding: each row shows its controls, effective retention, zero injection and result; bars share a linear [0,.6] scale. Rows stack by actual container width. A caption identifies them as static bars.
- Claim: blocking zero new signal cannot repair decay on the old-state path. The exact hold row is a limiting mathematical fixture, not the output of finite sigmoid logits.
- Verification: all three use `recurrenceTrace` unchanged; analytical endpoint/null checks in the new scoped receipt.

## 4. Equal center, different path

- Location: §5 before latent dimensions.
- Data: five horizontal coordinates [0,.25,.5,.75,1]. Straight-path heights [.5,.5,.5,.5,.5]; changing-direction heights [.5,.75,0,.75,.5]. Both centers (.5,.5).
- Encoding: two square x/y plots on identical linear [0,1] domains and equal physical scales. Numbered points, open-circle start, square end, center ×, dotted middle guides. Explicit point-order legend. For the straight path, the mean coincides with point 3; coincidence is not a displaced annotation.
- Claim: a deterministic classifier receiving only the common mean cannot distinguish these constructed inputs. This does not claim all one-latent models compute means. No dataset label or measured status attached to these synthetic paths.
- Verification: calculate both means independently; preserve differing coordinates; render pair on wide articles and stack below 620px article width.

## Existing labs and visual evidence retained

Cache edits still expose legal records, normalization and immediate read; prose now names memory 2→4 and later-record null edits. Recurrence instructions reset before the input-only comparison so the controlled contrast is valid. Latent instructions use actual “Use one uniform query” and “Reset latents” controls; both fixtures remain visible. The real trajectory workbench still loads only when opened and uses all four original frozen models and 50 validation rows; edits change outputs without inventing labels for hypothetical paths.

Existing gradient, mask, shape, output-query, measured-error and budget representations keep their mathematical contracts. Adjacent prose now explains what to inspect, why the distinction matters and how it connects to the next operator. Final browser checks must inspect the four new figures at desktop, 320px and 760px viewport with the actual reader/sidebar geometry, then smoke-check a retained live control; prior numerical evidence remains reusable only while its exact engine/assets hashes match.
