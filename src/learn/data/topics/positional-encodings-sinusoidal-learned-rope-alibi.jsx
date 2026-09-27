// Generated from the complete manuscript by scripts/generate-positional-encoding-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { PositionJourneyFigure, AdditivePositionLab, SinusoidalPositionLab, RotaryPositionLab, AlibiPositionLab, RelativeBucketFigure, PositionMovementLab, PositionCacheLab, PositionFrequencyFigure, PositionExtensionLab, PositionApplicationsFigure, XposPositionFigure } from '../../components/lesson-labs/PositionalEncodingLabs.jsx';
export default {
  title: 'Positional Encodings: Sinusoidal, Learned, RoPE and ALiBi',
  readTime: '~90 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson neural-lesson-neutral positional-encoding-lesson"><LessonIntro prerequisites="Query/key/value attention and Transformer blocks. Rotations, relative distances and cache coordinates are developed locally." sections={[["1-what-information-is-missing","1. What information is missing?"],["2-add-a-position-vector-learned-and-sinusoidal-encodings","2. Add a position vector: learned and sinusoidal encodings"],["3-rope-let-relative-position-change-the-comparison","3. RoPE: let relative position change the comparison"],["4-alibi-express-a-preference-in-logit-space","4. ALiBi: express a preference in logit space"],["5-other-relative-encodings-explain-the-design-space","5. Other relative encodings explain the design space"],["6-a-real-investigation-can-the-model-see-movement-direction","6. A real investigation: can the model see movement direction?"],["7-implement-positions-and-verify-cached-attention","7. Implement positions and verify cached attention"],["8-deeper-route-extending-the-context-without-confusing-the-claims","8. Deeper route: extending the context without confusing the claims"],["9-choose-the-position-system-for-the-actual-task","9. Choose the position system for the actual task"],["10-practice-change-the-situation-then-explain-the-result","10. Practice: change the situation, then explain the result"],["what-comes-next","What comes next?"],["references-and-another-way-to-learn","References and another way to learn"]]}>An ordered array is useful only when the model can use its order. Follow where position enters the actual computation.</LessonIntro>
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit position IDs, frequencies, RoPE vectors, ALiBi slopes/scores, cache identities and supported trajectory coordinates. Synchronize phase geometry, relative-score changes, distance penalty, legal cache relations and final mixture. Show whole-record reorder and ID-only edit as different operations. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose and troubleshoot positional mechanisms by relative/absolute behavior and cache consistency; do not infer long-context quality from a toy phase plot."}</Prose>

<Prose>{"Imagine recording a hand moving around an arc. The same collection of points can describe clockwise or anticlockwise movement. To tell them apart, a model needs to know how the points are ordered—not just where the hand visited."}</Prose>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals"}>{"previous Transformer Block lesson"}</a>{" supplied that information by attaching a numerical time-slot tag to every point. This lesson studies more structured ways to supply position. Some add a position vector to the input. Some turn query and key vectors before comparing them. Some change the attention score according to distance. These choices affect what the model can represent, how cached generation works and what happens when a sequence becomes longer."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" follow §§1–6, run the small program in §7 and try exercises 1–4. You will be able to explain and implement the four main methods and diagnose a position-offset bug. The deeper route in §§8–9 develops context extension, geometric variants and engineering choices; exercises 5–8 use that material. Derivations sit beside the mechanism they explain, so you can return to them after trying the visual investigation."}</Prose>

<H2>{"1. What information is missing?"}</H2>

<H3>{"Rows store order; the computation must use it"}</H3>

<Prose>{"An array preserves its row order. That does not mean every function applied to the array uses that order. Adding all the rows gives the same sum after any rearrangement."}</Prose>

<Prose>{"Recall one head from "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention"}</a>{". A query describes what a row seeks, a key describes how a row can be matched, and a value is the information mixed into the result:"}</Prose>

<div className="neural-equation"><MathBlock>{"A=\\operatorname{softmax}_{\\text{keys}}(QK^\\top/\\sqrt{d_k}),\\qquad Y=AV."}</MathBlock></div>

<Prose>{"Here rows of Q correspond to queries, columns of the score matrix to keys, and each softmax row sums to 1. "}<InlineMath>{"d_k"}</InlineMath>{" is the width of "}<strong>{"one query/key head"}</strong>{", not the whole model. Dividing by its square root controls score scale."}</Prose>

<Prose>{"Suppose we reorder every input row using the same permutation matrix "}<InlineMath>{"P"}</InlineMath>{". Shared rowwise projections produce "}<InlineMath>{"PQ,PK,PV"}</InlineMath>{". The score matrix becomes "}<InlineMath>{"P(QK^\\top)P^\\top"}</InlineMath>{": its rows and columns are merely rearranged. Rowwise softmax respects that rearrangement, so"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{Attention}(PX)=P\\operatorname{Attention}(X)."}</MathBlock></div>

<Prose>{"This is "}<strong>{"permutation equivariance"}</strong>{". Output rows move with their inputs. It is different from invariance: an invariant output stays unchanged. Mean pooling the equivariant output gives an invariant whole-sequence representation. Shared feedforward networks, per-row LayerNorm and residual additions preserve the same symmetry when no other positional signal is present."}</Prose>

<Prose>{"For a set of measurements whose order is irrelevant, this is useful. For a movement whose direction matters, it is a limitation. An attention-based set classifier cannot distinguish a sequence from its reversal if the two differ only in row order."}</Prose>

<PositionJourneyFigure />

<H3>{"A causal mask already supplies some structure"}</H3>

<Prose>{"The proof assumed the permitted query–key pairs were also unchanged or consistently permuted. A fixed causal mask permits a query to read only itself and preceding slots. Arbitrarily shuffling content while leaving this triangle fixed changes who can read whom. The earlier equivariance proof no longer applies."}</Prose>

<Prose>{"For example, with equal scores and scalar values "}<code>{"[2,6,10]"}</code>{", causal attention produces prefix means "}<code>{"[2,4,6]"}</code>{". Reversing the inputs against the same mask gives "}<code>{"[10,8,6]"}</code>{", not a reversal of the old outputs. With a special beginning-of-sequence marker, uniform attention can even expose a quantity such as "}<InlineMath>{"1/(t+1)"}</InlineMath>{", because the marker is one item among a growing prefix. Additional layers can use such structure. This is why decoder-only Transformers without explicit position embeddings, often called "}<strong>{"NoPE"}</strong>{", are a meaningful research design. It does not imply that every such model automatically learns robust counting or long-context reasoning. The "}<a href={"https://arxiv.org/pdf/2305.19466"}>{"NoPE/length-generalization study"}</a>{" distinguishes the causal setting and evaluates actual tasks rather than equating computability with generalization."}</Prose>

<Prose>{"Position can therefore arrive through several routes: explicit coordinates, position vectors, score biases, causal/local connectivity, recurrent state or a combination. Our task is to identify which route the model actually has."}</Prose>

<H2>{"2. Add a position vector: learned and sinusoidal encodings"}</H2>

<H3>{"A label the model can learn"}</H3>

<Prose>{"Let a token or measurement have a content vector "}<InlineMath>{"x_t\\in\\mathbb R^d"}</InlineMath>{". The simplest construction is"}</Prose>

<div className="neural-equation"><MathBlock>{"h_t=x_t+p_t."}</MathBlock></div>

<Prose>{"Both vectors have the same width. Addition keeps the shape "}<InlineMath>{"L\\times d"}</InlineMath>{", so the existing Transformer block can consume it."}</Prose>

<Prose>{"A "}<strong>{"learned absolute embedding"}</strong>{" stores a table "}<InlineMath>{"P\\in\\mathbb R^{L_{\\max}\\times d}"}</InlineMath>{", and chooses "}<InlineMath>{"p_t=P[t]"}</InlineMath>{". Its rows begin as ordinary trainable parameters. During training, a prediction error updates the rows used by that example, together with the rest of the network. If several examples use slot 7, their gradients contribute to the same position row 7."}</Prose>

<Prose>{"Take a deliberately small table:"}</Prose>

<NeuralTable caption={"A label the model can learn"} headers={[<>{"Slot"}</>,<>{"Position vector"}</>]} rows={[[<>{"0"}</>,<>{"[0.2, 0.0]"}</>],[<>{"1"}</>,<>{"[0.0, 0.3]"}</>],[<>{"2"}</>,<>{"[−0.1, 0.1]"}</>]]} />

<Prose>{"If the content vector "}<code>{"[1,2]"}</code>{" occurs in slot 0, its combined vector is "}<code>{"[1.2,2]"}</code>{". The same content in slot 1 becomes "}<code>{"[1,2.3]"}</code>{". These numbers are a hand fixture, not learned weights from a language model. They make the operation visible: the table tells the model which slot a row occupies; training decides what to do with that signal."}</Prose>

<Prose>{"The table has "}<InlineMath>{"L_{\\max}d"}</InlineMath>{" parameters. A 512 × 768 table has 393,216. A table with 512 rows supports indices 0–511. Index 512 has no row. Enlarging the table solves the storage problem but leaves a learning problem: new rows need a considered initialization, interpolation or further training strategy. Full retraining from scratch is not mathematically required, and a larger table alone does not establish useful longer-context behavior."}</Prose>

<Prose>{"Do not silently replace an out-of-range index with "}<code>{"index % max_length"}</code>{". That makes different absolute positions share an embedding without having trained the model for this periodic rule."}</Prose>

<AdditivePositionLab />

<H3>{"Smooth clocks instead of a table"}</H3>

<Prose>{"We can compute the position vector from a fixed formula. The original Transformer used sine/cosine pairs at different frequencies:"}</Prose>

<div className="neural-equation"><MathBlock>{"p_{t,2r}=\\sin(t\\omega_r),\\quad\np_{t,2r+1}=\\cos(t\\omega_r),\\quad\n\\omega_r=b^{-2r/d},\\quad r=0,\\ldots,d/2-1."}</MathBlock></div>

<Prose>{"This definition assumes even "}<InlineMath>{"d"}</InlineMath>{", uses radians and commonly starts from "}<InlineMath>{"b=10000"}</InlineMath>{". Frequency means radians of phase change per position. The wavelength is "}<InlineMath>{"2\\pi/\\omega_r"}</InlineMath>{" positions for a complete turn."}</Prose>

<Prose>{"Think of several clock hands turning at different rates. One fast hand is ambiguous after it comes around again, but the other hands are at different phases. Together they supply a rich position signature. This analogy concerns multiple scales; it does not make a floating-point vector an infinitely precise position identifier."}</Prose>

<Prose>{"For "}<InlineMath>{"d=8,b=10000"}</InlineMath>{", the four frequencies are "}<code>{"[1,0.1,0.01,0.001]"}</code>{". Evaluating the formula gives:"}</Prose>

<NeuralTable caption={"Smooth clocks instead of a table"} headers={[<>{"Position"}</>,<>{"sin pair 0"}</>,<>{"cos pair 0"}</>,<>{"sin pair 1"}</>,<>{"cos pair 1"}</>,<>{"sin pair 2"}</>,<>{"cos pair 2"}</>,<>{"sin pair 3"}</>,<>{"cos pair 3"}</>]} rows={[[<>{"0"}</>,<>{"0"}</>,<>{"1"}</>,<>{"0"}</>,<>{"1"}</>,<>{"0"}</>,<>{"1"}</>,<>{"0"}</>,<>{"1"}</>],[<>{"1"}</>,<>{"0.841"}</>,<>{"0.540"}</>,<>{"0.100"}</>,<>{"0.995"}</>,<>{"0.010"}</>,<>{"1.000"}</>,<>{"0.001"}</>,<>{"1.000"}</>],[<>{"2"}</>,<>{"0.909"}</>,<>{"−0.416"}</>,<>{"0.199"}</>,<>{"0.980"}</>,<>{"0.020"}</>,<>{"1.000"}</>,<>{"0.002"}</>,<>{"1.000"}</>],[<>{"3"}</>,<>{"0.141"}</>,<>{"−0.990"}</>,<>{"0.296"}</>,<>{"0.955"}</>,<>{"0.030"}</>,<>{"1.000"}</>,<>{"0.003"}</>,<>{"1.000"}</>]]} />

<Prose>{"Slow channels appear constant at three decimal places even when their exact values differ. Conversely, the fastest hand turns nearly half a revolution from position 0 to 3. Adjacent positions need not look similar in every channel."}</Prose>

<SinusoidalPositionLab />

<H3>{"Why pairs make relative shifts expressible"}</H3>

<Prose>{"The addition identities for sine and cosine give"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{bmatrix}\\sin((t+\\delta)\\omega)\\\\\\cos((t+\\delta)\\omega)\\end{bmatrix}\n=\n\\begin{bmatrix}\\cos(\\delta\\omega)&\\sin(\\delta\\omega)\\\\-\\sin(\\delta\\omega)&\\cos(\\delta\\omega)\\end{bmatrix}\n\\begin{bmatrix}\\sin(t\\omega)\\\\\\cos(t\\omega)\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The matrix depends on the shift "}<InlineMath>{"\\delta"}</InlineMath>{", not on the starting slot "}<InlineMath>{"t"}</InlineMath>{". It is a rotation with a sign/order convention appropriate to the "}<code>{"[sin,cos]"}</code>{" column. This means a shared linear transformation can relate the code of a position to the code of a fixed offset. It does not mean that a learned attention head automatically selects “exactly three positions earlier.”"}</Prose>

<Prose>{"Another useful identity is"}</Prose>

<div className="neural-equation"><MathBlock>{"p_m^\\top p_n=\\sum_r\\cos((m-n)\\omega_r)."}</MathBlock></div>

<Prose>{"For the unprojected position vectors alone, the inner product depends on relative offset. But attention compares projected "}<strong>{"content-plus-position"}</strong>{" vectors. Let "}<InlineMath>{"M=W_Q^\\top W_K"}</InlineMath>{" under a column-vector convention. Then"}</Prose>

<div className="neural-equation"><MathBlock>{"(x_m+p_m)^\\top M(x_n+p_n)\n=x_m^\\top Mx_n+x_m^\\top Mp_n+p_m^\\top Mx_n+p_m^\\top Mp_n."}</MathBlock></div>

<Prose>{"There are content–content, content–position, position–content and position–position terms. An arbitrary learned "}<InlineMath>{"M"}</InlineMath>{" need not preserve the simple cosine identity. Those interactions are part of the representation, not inherently wasted capacity. They allow position to affect values and residual features as well as attention scores."}</Prose>

<Prose>{"The "}<a href={"https://arxiv.org/pdf/1706.03762"}>{"original Transformer §3.5"}</a>{" compares learned and fixed input encodings and motivates the shift identity. The "}<a href={"https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding"}>{"D2L explanation and runnable notebook"}</a>{" supplies another route through the multiscale picture. Neither the existence of a sine formula at position 100,000 nor a successful table lookup establishes that a trained model will use that position well."}</Prose>

<H2>{"3. RoPE: let relative position change the comparison"}</H2>

<H3>{"Rotate the features after the projections"}</H3>

<Prose>{""}<strong>{"Rotary position embedding"}</strong>{", RoPE, applies a deterministic rotation to each query and key. Standard RoPE leaves the values unrotated. Position enters the query–key comparison in each attention layer."}</Prose>

<Prose>{"Start with two-dimensional vectors. A counterclockwise rotation through angle "}<InlineMath>{"\\phi"}</InlineMath>{" is"}</Prose>

<div className="neural-equation"><MathBlock>{"R(\\phi)=\\begin{bmatrix}\\cos\\phi&-\\sin\\phi\\\\\\sin\\phi&\\cos\\phi\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The point "}<code>{"[1,0]"}</code>{" becomes "}<code>{"[0,1]"}</code>{" at "}<InlineMath>{"\\phi=\\pi/2"}</InlineMath>{". Its length stays 1. A general vector turns through the same angle without changing length."}</Prose>

<Prose>{"For a head of even width "}<InlineMath>{"d_k"}</InlineMath>{", divide the coordinates into pairs. Pair "}<InlineMath>{"r"}</InlineMath>{" gets frequency "}<InlineMath>{"\\theta_r=b^{-2r/d_k}"}</InlineMath>{". At query position "}<InlineMath>{"m"}</InlineMath>{", rotate each pair through "}<InlineMath>{"m\\theta_r"}</InlineMath>{"; at key position "}<InlineMath>{"n"}</InlineMath>{", rotate its corresponding pair through "}<InlineMath>{"n\\theta_r"}</InlineMath>{". Write the block-diagonal collection of rotations as "}<InlineMath>{"R_m"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"q'_m=R_mq_m,\\qquad k'_n=R_nk_n."}</MathBlock></div>

<Prose>{"The coordinate pairs are fixed by the implementation's basis. The projections that produce their content are learned. The rotation planes themselves are not separately learned in this standard construction."}</Prose>

<RotaryPositionLab />

<H3>{"The relative-offset identity, step by step"}</H3>

<Prose>{"Transpose reverses a rotation, and successive rotations add their angles. Therefore"}</Prose>

<div className="neural-equation"><MathBlock>{"(R_mq_m)^\\top(R_nk_n)\n=q_m^\\top R_m^\\top R_nk_n\n=q_m^\\top R_{n-m}k_n."}</MathBlock></div>

<Prose>{"Turn both vectors by 100 extra position units, using the same frequencies, and their dot product stays unchanged. Turn only the key, and the relative angle changes. That is the useful built-in structure."}</Prose>

<Prose>{"Notice what the identity holds fixed: the "}<strong>{"content vectors"}</strong>{" "}<InlineMath>{"q_m,k_n"}</InlineMath>{". Different words or hidden states still produce different vectors. RoPE has not replaced content similarity with a distance-only score. The vectors in a later layer may already reflect boundaries, masks and earlier context, so a statement about this local operation must not be mistaken for universal translation invariance of an entire language model."}</Prose>

<Prose>{"One pair contributes"}</Prose>

<div className="neural-equation"><MathBlock>{"(q_0k_0+q_1k_1)\\cos(\\Delta\\theta)\n +(q_1k_0-q_0k_1)\\sin(\\Delta\\theta),\\quad \\Delta=n-m."}</MathBlock></div>

<Prose>{"The sine term can distinguish the direction of the offset. With suitable content vectors, “three before” and “three after” can receive different scores. The denominator "}<InlineMath>{"\\sqrt{d_k}"}</InlineMath>{" and rowwise softmax follow this calculation as usual."}</Prose>

<H3>{"A complete four-coordinate example"}</H3>

<Prose>{"Take "}<InlineMath>{"q=[0.8,-0.5,0.3,1.2]"}</InlineMath>{" at position 3 and "}<InlineMath>{"k=[1,0.25,-0.5,0.75]"}</InlineMath>{" at position 7. For "}<InlineMath>{"d_k=4,b=10000"}</InlineMath>{", frequencies are 1 and 0.01."}</Prose>

<ol start={1}><li>{"Query pair 0 turns through 3 radians; query pair 1 through 0.03."}</li><li>{"Key pair 0 turns through 7 radians; key pair 1 through 0.07."}</li><li>{"The rotated vectors are"}</li></ol>

<Prose>{"   "}<InlineMath>{"q'\\approx[-0.721434,0.607892,0.263870,1.208459]"}</InlineMath>{" and    "}<InlineMath>{"k'\\approx[0.589656,0.845462,-0.551233,0.713192]"}</InlineMath>{"."}</Prose>

<ol start={4}><li>{"Their dot product is 0.804961. Dividing by "}<InlineMath>{"\\sqrt4=2"}</InlineMath>{" gives the attention logit 0.402481."}</li><li>{"Computing "}<InlineMath>{"q^\\top R_4k"}</InlineMath>{" gives the same 0.804961. Positions 103 and 107 also give that value."}</li></ol>

<Prose>{"The query's norm remains 1.555635. Moving the key to position 8 changes the dot product to 1.570549; bringing it to the same position as the query gives the raw dot product 1.425. A farther key can receive a "}<strong>{"larger"}</strong>{" score. Position modulates a content-dependent comparison; it is not a mandatory recency penalty."}</Prose>

<H3>{"A diagonal pattern requires a controlled example"}</H3>

<Prose>{"A matrix is "}<strong>{"Toeplitz"}</strong>{" when each diagonal is constant. If the same content query appears at every position and the same content key appears at every position, RoPE scores depend only on their offsets, producing such a matrix. With q above repeated four times for both Q and K, the diagonal entries are all 2.42 and first off-diagonal entries are 2.010793."}</Prose>

<Prose>{"Now double the content vector at position 1. Its self-score becomes 9.68; neighboring scores involving it double. Other scores remain unchanged. The matrix is no longer Toeplitz, although every comparison still satisfies the RoPE identity. This is a useful controlled contrast for understanding an attention heatmap."}</Prose>

<Prose>{"RoPE also does not make every score decrease monotonically with distance. Concentrate q and k on the first pair as "}<code>{"[1,0]"}</code>{": the score is simply "}<InlineMath>{"\\cos\\Delta"}</InlineMath>{", which falls toward −1 near distance 3 and rises to 0.96017 at distance 6. Frequency mixtures can create useful distance structure, but they do not remove this counterexample. The "}<a href={"https://arxiv.org/html/2104.09864v5#S3"}>{"RoFormer construction"}</a>{" and "}<a href={"https://arxiv.org/html/2306.15595v2#S2"}>{"Position Interpolation analysis"}</a>{" help distinguish the exact rotation identity from assumptions about longer-distance behavior."}</Prose>

<H3>{"Coordinate layout and partial rotation"}</H3>

<Prose>{"Our implementation pairs adjacent entries "}<code>{"(0,1),(2,3),…"}</code>{". Another valid convention pairs the first half with the second half: "}<code>{"(0,d_k/2),(1,d_k/2+1),…"}</code>{". For four coordinates, the permutation "}<code>{"[0,2,1,3]"}</code>{" converts the adjacent representation into the half-split representation. Permute the projection outputs, frequencies and inverse mapping consistently, and these are two coordinate descriptions of the same operation. Changing the rotation routine while loading unchanged checkpoint weights generally is not equivalent."}</Prose>

<Prose>{"Some architectures rotate only an even number "}<InlineMath>{"d_r\\leq d_k"}</InlineMath>{" of coordinates. Split q and k into rotary and unrotated portions:"}</Prose>

<div className="neural-equation"><MathBlock>{"(q_m^R)^\\top R_{n-m}k_n^R+(q_m^C)^\\top k_n^C."}</MathBlock></div>

<Prose>{"The content-only term and the relative positional term coexist. State whether frequencies use "}<InlineMath>{"d_r"}</InlineMath>{" or a checkpoint-specific rule; do not silently substitute the model width. We will use this split again in "}<a href={"/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals"}>{"Multi-Head Latent Attention"}</a>{"."}</Prose>

<Prose>{"Values are unrotated in the standard mechanism taught here, because the weights already determine how their content is mixed. Rotating values would define a different architecture and requires explaining how its output coordinates transform. It is not an impossible mathematical operation or proof that every such variant trains poorly."}</Prose>

<H2>{"4. ALiBi: express a preference in logit space"}</H2>

<H3>{"A score penalty has a precise probabilistic effect"}</H3>

<Prose>{"In causal attention, query position "}<InlineMath>{"i"}</InlineMath>{" may read key positions "}<InlineMath>{"j\\leq i"}</InlineMath>{". "}<strong>{"Attention with Linear Biases"}</strong>{", ALiBi, uses"}</Prose>

<div className="neural-equation"><MathBlock>{"s_{ij}=q_i^\\top k_j/\\sqrt{d_k}-a_h(i-j),\\qquad a_h>0."}</MathBlock></div>

<Prose>{"The positive slope "}<InlineMath>{"a_h"}</InlineMath>{" is fixed per head in the original scheme. Future positions remain masked. This finite penalty does not replace a causal or padding mask: it discourages a distant legal key; a mask prohibits a key."}</Prose>

<Prose>{"For two legal keys j and r, softmax gives the exact odds ratio"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{A_{ij}}{A_{ir}}\n=\\exp(c_{ij}-c_{ir})\\exp[-a_h((i-j)-(i-r))],"}</MathBlock></div>

<Prose>{"where "}<InlineMath>{"c"}</InlineMath>{" denotes scaled content scores. If content scores tie, an extra distance "}<InlineMath>{"D"}</InlineMath>{" multiplies the odds by "}<InlineMath>{"e^{-a_hD}"}</InlineMath>{". With "}<InlineMath>{"a_h=0.5,D=10"}</InlineMath>{", the factor is 0.006738, not one-half. The distance that halves these equal-content odds is "}<InlineMath>{"\\ln2/a_h\\approx1.3863"}</InlineMath>{". These are odds between keys; an individual normalized probability also depends on all other keys."}</Prose>

<Prose>{"The logit penalty grows linearly. The induced multiplicative factor in unnormalized attention grows or decays exponentially. That connection makes the name and effect easier to remember."}</Prose>

<H3>{"Content can overcome the preference"}</H3>

<Prose>{"Consider a query in slot 3 and keys in slots 0–3. Suppose their scaled content scores are "}<code>{"[2,0,0,0]"}</code>{". With slope 0.5:"}</Prose>

<NeuralTable caption={"Content can overcome the preference"} headers={[<>{"Key position"}</>,<>{"Distance"}</>,<>{"Content score"}</>,<>{"Bias"}</>,<>{"Final score"}</>,<>{"Final attention weight"}</>]} rows={[[<>{"0"}</>,<>{"3"}</>,<>{"2"}</>,<>{"−1.5"}</>,<>{"0.5"}</>,<>{"0.455054"}</>],[<>{"1"}</>,<>{"2"}</>,<>{"0"}</>,<>{"−1.0"}</>,<>{"−1.0"}</>,<>{"0.101536"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"0"}</>,<>{"−0.5"}</>,<>{"−0.5"}</>,<>{"0.167405"}</>],[<>{"3"}</>,<>{"0"}</>,<>{"0"}</>,<>{"0"}</>,<>{"0"}</>,<>{"0.276004"}</>]]} />

<Prose>{"Without the bias, key 0 receives 0.711235. The distance penalty weakens its advantage, but it still gets the largest weight. The model can learn content scores that counteract a fixed penalty; the penalty itself does not learn. There is no hard finite attention window in this formula."}</Prose>

<AlibiPositionLab />

<H3>{"Different slopes, different preferences"}</H3>

<Prose>{"For a power-of-two head count H, the original schedule gives"}</Prose>

<div className="neural-equation"><MathBlock>{"a_h=2^{-8h/H},\\quad h=1,\\ldots,H."}</MathBlock></div>

<Prose>{"With eight heads the slopes are 1/2, 1/4,…, 1/256. With two heads they are 1/16 and 1/256, not 1/2 and 1/256. With sixteen heads the first is "}<InlineMath>{"1/\\sqrt2"}</InlineMath>{". Steeper slopes prefer shorter distances more strongly when content scores are comparable. Head behavior still depends on its learned Q/K features."}</Prose>

<Prose>{"The "}<a href={"https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L693"}>{"author's implementation"}</a>{" extends the schedule to a non-power-of-two count by retaining a lower power-of-two schedule and inserting selected slopes from the doubled schedule. Its three-head result is "}<code>{"[1/16,1/256,1/4]"}</code>{". Preserve that convention when reproducing its checkpoints; a different geometric schedule is a design variation, not a reproduction."}</Prose>

<Prose>{"There is also a useful implementation identity. For one causal row,"}</Prose>

<div className="neural-equation"><MathBlock>{"-a_h(i-j)=a_hj-a_hi."}</MathBlock></div>

<Prose>{"The last term is constant across that row. Softmax ignores a common additive constant, so adding "}<InlineMath>{"a_hj"}</InlineMath>{" alone gives the same probabilities if the legal key set is the same. This permits compact bias construction. It does not mean you can omit the causal mask or forget which entries belong to a packed sequence."}</Prose>

<Prose>{"The "}<a href={"https://arxiv.org/html/2108.12409v2#S3"}>{"ALiBi paper"}</a>{" reports length-extrapolation results in specified language-model experiments. It does not prove that this bias solves every long-range task, that all distant facts remain retrievable or that runtime overhead is identical across kernels."}</Prose>

<H3>{"Bidirectional distance needs a direction decision"}</H3>

<Prose>{"For an encoder that can read both directions, one possible adaptation is "}<InlineMath>{"-a_h|i-j|"}</InlineMath>{". It favors proximity on either side. But absolute distance cannot tell left from right. Reverse a sequence of length "}<InlineMath>{"L"}</InlineMath>{": slots i, j become "}<InlineMath>{"L-1-i,L-1-j"}</InlineMath>{", and their absolute distance is unchanged."}</Prose>

<Prose>{"If all other operations are shared per row and the final readout is mean pooling, this symmetric adaptation produces the same prediction for a trajectory and its reversal. The real investigation below demonstrates it. Signed relative buckets, explicitly directed heads, absolute slots or another directional signal can break that symmetry. The original causal ALiBi model already has a directed mask, so this particular reversal argument does not apply to it."}</Prose>

<H2>{"5. Other relative encodings explain the design space"}</H2>

<H3>{"A learned relation vector: Shaw-style attention"}</H3>

<Prose>{"An offset can be represented by a trainable vector instead of a fixed rotation or scalar penalty. For query i, key j, set "}<InlineMath>{"r=\\operatorname{clip}(j-i,-k,k)"}</InlineMath>{". A Shaw-style head uses"}</Prose>

<div className="neural-equation"><MathBlock>{"s_{ij}=\\frac{q_i^\\top(k_j+a_r^K)}{\\sqrt{d_k}},\\qquad\ny_i=\\sum_j A_{ij}(v_j+a_r^V)."}</MathBlock></div>

<Prose>{"The score contribution "}<InlineMath>{"q_i^\\top a_r^K"}</InlineMath>{" depends on what the query seeks. With "}<InlineMath>{"q=[2,0]"}</InlineMath>{", relation vector "}<code>{"[0.5,1]"}</code>{" adds 1 to the unscaled dot product. Another query "}<code>{"[0,2]"}</code>{" gets 2 from that same relation. A learned scalar bias cannot express this particular query-dependent distinction on its own."}</Prose>

<Prose>{"Clipping at "}<InlineMath>{"k=2"}</InlineMath>{" assigns offsets −9 and −2 to the same relation category. This saves parameters but deliberately loses their exact distance at this component. There are "}<InlineMath>{"2k+1"}</InlineMath>{" relation vectors per table; the value relation can convey which relation supplied information, beyond just changing the weight. The "}<a href={"https://aclanthology.org/N18-2074.pdf"}>{"original relation-aware attention paper §§3.1–3.3"}</a>{" develops this construction and its efficient decomposition."}</Prose>

<H3>{"A learned scalar by distance bucket: T5-style bias"}</H3>

<Prose>{"A simpler mechanism learns one scalar per head and relative-distance bucket, then adds it to the content score. Nearby offsets receive finer categories; large distances share coarser categories. The model learns the preference for each category. Unlike ALiBi, it need not be monotone in distance."}</Prose>

<Prose>{"For the common bidirectional 32-bucket, maximum-distance 128 configuration, divide the buckets by direction. Within one direction, distances 0–7 have exact bins and larger distances enter logarithmically widening bins. Under the "}<a href={"https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L198"}>{"T5 implementation's key-minus-query convention"}</a>{":"}</Prose>

<NeuralTable caption={"A learned scalar by distance bucket: T5-style bias"} headers={[<>{"Offset "}<InlineMath>{"j-i"}</InlineMath>{""}</>,<>{"−129"}</>,<>{"−128"}</>,<>{"−16"}</>,<>{"−8"}</>,<>{"−7"}</>,<>{"−1"}</>,<>{"0"}</>,<>{"1"}</>,<>{"7"}</>,<>{"8"}</>,<>{"16"}</>,<>{"128"}</>,<>{"129"}</>]} rows={[[<>{"Bucket"}</>,<>{"15"}</>,<>{"15"}</>,<>{"10"}</>,<>{"8"}</>,<>{"7"}</>,<>{"1"}</>,<>{"0"}</>,<>{"17"}</>,<>{"23"}</>,<>{"24"}</>,<>{"26"}</>,<>{"31"}</>,<>{"31"}</>]]} />

<Prose>{"The bucket number is an index, not a magnitude. Bucket 31's learned value might be positive or negative. Positions 128 and 129 sharing a bin means this bias component cannot distinguish those two distances; content and other layers may still distinguish their tokens."}</Prose>

<Prose>{"For distance "}<InlineMath>{"D"}</InlineMath>{" in one direction, with "}<InlineMath>{"B"}</InlineMath>{" available buckets and exact range "}<InlineMath>{"E=B/2"}</InlineMath>{", the large-distance index is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\min\\left(B-1,\\;E+\\left\\lfloor\n\\frac{\\ln(D/E)}{\\ln(D_{\\max}/E)}(B-E)\n\\right\\rfloor\\right)."}</MathBlock></div>

<Prose>{"Use the exact index "}<InlineMath>{"D"}</InlineMath>{" for "}<InlineMath>{"D<E"}</InlineMath>{", and add the direction offset afterward. A causal variant allocates buckets differently because future keys are prohibited. It still needs the causal mask; mapping an illegal future offset to a bucket does not authorize attention to it."}</Prose>

<Prose>{"T5's original attention parameterization also omits the usual explicit "}<InlineMath>{"1/\\sqrt{d_k}"}</InlineMath>{" score scale. A lesson comparing "}<strong>{"bias mechanisms"}</strong>{" may use a common scaled-content convention, but a checkpoint reproduction must match its complete attention rule. “Relative bias” names the position component, not every surrounding implementation detail."}</Prose>

<Prose>{"These methods are alternatives with different expressive choices. There is no historical rule that every later method strictly replaces every earlier one."}</Prose>

<RelativeBucketFigure />

<H2>{"6. A real investigation: can the model see movement direction?"}</H2>

<H3>{"The data and the question"}</H3>

<Prose>{"Return to the real "}<strong>{"Libras Movement"}</strong>{" trajectories used in the preceding lessons. Each record contains 45 ordered x/y hand-centroid samples and one of 15 movement-type labels. Class 4 is an anticlockwise arc; class 5 is a clockwise arc. These are simplified movement measurements, not complete signed-language sentences or a deployed recognition system."}</Prose>

<Prose>{"The "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI source and original metadata"}</a>{" are available under CC BY 4.0, credited to Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro. The "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/movement_libras.data"}>{"offline input"}</a>{", "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/movement_libras.names"}>{"original metadata"}</a>{" and "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/data-provenance.md"}>{"provenance record"}</a>{" accompany this lesson. Coordinates lie in 0–1; our model uses the fixed transformation "}<InlineMath>{"2x-1"}</InlineMath>{". Position indices 0–44 describe sample order, not exact elapsed seconds."}</Prose>

<Prose>{"The practical question is: "}<strong>{"does changing direction become visible to the classifier, and how does that depend on the encoding?"}</strong>{" Accuracy on the whole dataset and sensitivity to this particular intervention are different measurements."}</Prose>

<Prose>{"The raw 360 rows contain 30 additional exact duplicates. We retain the first occurrence of each unique 90-coordinate trajectory after checking that duplicate labels agree. A fixed classwise split assigns 220 unique records to training, 50 to validation and 60 to test, with four test records per class. Whole trajectories stay together. The collection describes four performers and two sessions but does not provide reliable per-row identities, so this is a row-level study; it cannot establish performance on a new performer or session. The exact split and duplicate groups are saved with the program's results."}</Prose>

<H3>{"Five controlled models"}</H3>

<Prose>{"Each model has a 2-to-24 coordinate projection, one pre-normalized Transformer block with two heads of width 12 and a 48-wide GELU feedforward layer, final LayerNorm, mean pooling and a 24-to-15 classifier. Q/K/V and output maps have biases. LayerNorm uses epsilon "}<InlineMath>{"10^{-5}"}</InlineMath>{". There is no dropout."}</Prose>

<NeuralTable caption={"Five controlled models"} headers={[<>{"Model"}</>,<>{"Where position enters"}</>]} rows={[[<>{"No-position control"}</>,<>{"Nowhere; every row uses the same operations"}</>],[<>{"Sinusoidal"}</>,<>{"Add the width 24 vector to the coordinate embedding"}</>],[<>{"Learned"}</>,<>{"Add a learned 45 × 24 table row"}</>],[<>{"RoPE"}</>,<>{"Rotate the 12-dimensional Q/K vectors before comparing them"}</>],[<>{"Symmetric ALiBi"}</>,<>{"Add a two-head "}<InlineMath>{"-a_h|i-j|"}</InlineMath>{" proximity bias; no causal mask"}</>]]} />

<Prose>{"All shared tensors have exactly the same seed 101 initialization. The learned table uses its own generator 303, initialized with standard deviation 0.02, so creating it does not change the shared weights. The table adds 1,080 parameters: 6,447 versus 5,367 in each other model. Fixed sinusoidal and RoPE frequencies use base 10000. The ALiBi slopes are 1/16 and 1/256."}</Prose>

<Prose>{"Every model gets 180 full-batch Adam updates with learning rate 0.003 and no weight decay. Select its checkpoint by validation macro-F1, then lower validation cross entropy, then the earliest exact tie. Macro-F 1 averages the 15 classwise F1 scores equally. The test records are evaluated after this selection. This is one predeclared seed with one common training protocol, not a large tuning study or a language-model context-extension experiment."}</Prose>

<Prose>{"The actual CPU results were:"}</Prose>

<NeuralTable caption={"Five controlled models"} headers={[<>{"Encoding"}</>,<>{"Selected epoch"}</>,<>{"Train correct /220"}</>,<>{"Validation correct /50"}</>,<>{"Test correct /60"}</>,<>{"Test macro-F1"}</>,<>{"Test CE, nats/record"}</>]} rows={[[<>{"None"}</>,<>{"107"}</>,<>{"193"}</>,<>{"35"}</>,<>{"33"}</>,<>{"0.528"}</>,<>{"1.287"}</>],[<>{"Sinusoidal"}</>,<>{"154"}</>,<>{"217"}</>,<>{"38"}</>,<>{"42"}</>,<>{"0.689"}</>,<>{"1.039"}</>],[<>{"Learned"}</>,<>{"108"}</>,<>{"220"}</>,<>{"40"}</>,<>{"40"}</>,<>{"0.656"}</>,<>{"1.017"}</>],[<>{"RoPE"}</>,<>{"106"}</>,<>{"203"}</>,<>{"35"}</>,<>{"40"}</>,<>{"0.632"}</>,<>{"1.120"}</>],[<>{"Symmetric ALiBi"}</>,<>{"180"}</>,<>{"218"}</>,<>{"29"}</>,<>{"39"}</>,<>{"0.624"}</>,<>{"1.289"}</>]]} />

<Prose>{"Sinusoidal happens to have the highest test-correct count in this run. That does not establish a universal ranking. The models have different inductive biases, only one seed was run, the learned table changes parameter count, and the common hyperparameters need not suit all methods equally. Keep the table as an observation of this protocol. The sharper result comes from the following symmetry test."}</Prose>

<H3>{"Reverse the path and inspect what changes"}</H3>

<Prose>{"The displayed example is source row 77, chosen beforehand as the first held-out class 4 record in the fixed split. All five selected models classify its original version incorrectly. The example is kept because a real failure can still reveal the mechanism."}</Prose>

<Prose>{"There are two different edits:"}</Prose>

<ol start={1}><li>{""}<strong>{"Move records together."}</strong>{" Reverse the storage/display order of the point records and move each original position ID with its point. The sequence meaning has not changed. Every model's pooled logits stay the same within float32 rounding, with maximum changes below "}<InlineMath>{"1.6\\times10^{-6}"}</InlineMath>{"."}</li><li>{""}<strong>{"Reverse the movement."}</strong>{" Reverse the points but retain slot IDs 0–44. A point formerly attached to the beginning is now attached to the end. This changes the ordered trajectory."}</li></ol>

<Prose>{"For the second edit, the observed largest change among the 15 logits was:"}</Prose>

<NeuralTable caption={"Reverse the path and inspect what changes"} headers={[<>{"Model"}</>,<>{"Maximum absolute logit change"}</>,<>{"Original predicted class → reversed prediction"}</>]} rows={[[<>{"None"}</>,<>{"0.000001"}</>,<>{"5→5"}</>],[<>{"Sinusoidal"}</>,<>{"3.652706"}</>,<>{"9→9"}</>],[<>{"Learned"}</>,<>{"6.880572"}</>,<>{"7→5"}</>],[<>{"RoPE"}</>,<>{"6.110313"}</>,<>{"3→5"}</>],[<>{"Symmetric ALiBi"}</>,<>{"0.000002"}</>,<>{"9→9"}</>]]} />

<Prose>{"The unrounded ALiBi change is "}<InlineMath>{"1.55\\times10^{-6}"}</InlineMath>{", numerical noise around an exact real-arithmetic invariance. Its distance matrix is unchanged by reversal after the corresponding row/column rearrangement, so the same proof as in §1 propagates through the block and mean pool. No amount of ordinary parameter training can break this architectural symmetry while its assumptions remain in place."}</Prose>

<Prose>{"The sinusoidal classifier's predicted class stays 9, but its logits change substantially. Looking only at the largest-probability class would conceal its directional sensitivity. The RoPE model's class 5 probability changes from 0.084402 to 0.969648. That shows the intervention affected its decision; it does not supply a verified label for an artificially edited recording. We do not automatically grade any reversed or hand-edited sample as a newly labeled real example."}</Prose>

<H3>{"A visual workspace that exposes the mechanism"}</H3>

<PositionMovementLab />

<Prose>{"The coordinate edit is also real computation: reflect frame 23's x coordinate from "}<InlineMath>{"x"}</InlineMath>{" to "}<InlineMath>{"1-x"}</InlineMath>{". On this example it changes the maximum logit by 2.890378 in the no-position model and 3.264872 in symmetric ALiBi. Reversal invariance does not mean these models ignore the coordinates. In the ALiBi run this edit even changes the winning class 9→4; an artificially edited sample still has no newly established ground-truth label."}</Prose>

<Prose>{"As another control, append five padded points at coordinates "}<InlineMath>{"[0.75,0.75]"}</InlineMath>{", assign harmless position IDs 0, and exclude them from attention "}<strong>{"and pooling"}</strong>{". Original valid positions remain 0–44. The original logits are recovered within float32 rounding for all five models. Leaving the pads unmasked changes the computation; for the sinusoidal model the largest logit change is 8.816077. Masking keys alone is not sufficient if the final mean still includes padded query rows."}</Prose>

<Prose>{"No retraining occurs when the learner edits an input. The workspace evaluates the saved selected model. It must display that scope clearly: predictions are from a small movement classifier, not from a pretrained language model, and attention weights are one internal calculation rather than a causal explanation of the whole decision."}</Prose>

<H3>{"Reproduce and investigate further"}</H3>

<Prose>{"Download "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/author-calculations.py"}>{"the complete CPU program"}</a>{", the two original data files and "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/author-results.json"}>{"its recorded results"}</a>{" into one directory. With Python, NumPy, PyTorch and scikit-learn installed, run:"}</Prose>

<CodeBlock language={"bash"}>{"python author-calculations.py"}</CodeBlock>

<Prose>{"The program contains the full data-boundary checks, all five model definitions, training/checkpoint selection, metrics, actual interventions and weight export. It has no hidden notebook state, pretrained download or GPU dependency. The recorded run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, with one CPU thread and deterministic algorithms. Last-digit results can vary with a different numerical environment."}</Prose>

<Prose>{"Read "}<code>{"PositionClassifier.forward"}</code>{" in this order: construct the content rows; add an input position signal when selected; normalize; project Q/K/V; apply RoPE or ALiBi at its own location; mask; mix values; perform the residual/feedforward block; normalize and valid-pool; classify. This is the same Transformer mechanism from the previous lesson with a deliberately isolated positional choice."}</Prose>

<Prose>{"The retained "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/position-models.json"}>{"model/trace evidence"}</a>{" supports the interactive view, while the full training histories are optional reproduction material. A production page should load only the selected model's compact weights on demand. The source study remains offline; training all variants in the browser would add cost without improving this investigation."}</Prose>

<H2>{"7. Implement positions and verify cached attention"}</H2>

<H3>{"The cache needs a coordinate system"}</H3>

<Prose>{"A "}<strong>{"KV cache"}</strong>{" stores past keys and values so an autoregressive decoder does not project the same past tokens again at every new step. Standard RoPE implementations often cache the already-rotated keys "}<InlineMath>{"R_nk_n"}</InlineMath>{", with unrotated values. A new query at position "}<InlineMath>{"t"}</InlineMath>{" must use "}<InlineMath>{"R_tq_t"}</InlineMath>{"."}</Prose>

<Prose>{"Imagine a chunk whose real positions are 7, 8, 9. When processing its last token separately, the local query tensor has length 1. Calling "}<code>{"arange(1)"}</code>{" gives position 0, not 9. Its shape is valid, but its relative angles to the cached keys are wrong."}</Prose>

<Prose>{"Track three notions separately:"}</Prose>

<NeuralTable caption={"The cache needs a coordinate system"} headers={[<>{"Quantity"}</>,<>{"Meaning"}</>,<>{"Example"}</>]} rows={[[<>{"Logical position ID"}</>,<>{"The coordinate used by the encoding"}</>,<>{"9"}</>],[<>{"Cache slot"}</>,<>{"Where this key/value is stored"}</>,<>{"Slot 2 in this three-entry toy cache"}</>],[<>{"Valid/causal relation"}</>,<>{"Which stored entries this query may read"}</>,<>{"Positions 7, 8, 9 are legal"}</>]]} />

<Prose>{"In a contiguous cache these may have simple relationships. With left padding, packed documents, sliding windows, evictions or reused prefixes, they can differ. RoPE does not remove the need to maintain those relationships. Learned absolute embeddings also do not intrinsically require a separate position field in every cached vector; implementations track enough metadata for their chosen layout and next position."}</Prose>

<H3>{"A runnable, transparent reference"}</H3>

<Prose>{"This complete program uses NumPy and one head so the geometry and cache contract are visible. It implements sinusoidal features, strict learned lookup, adjacent-pair RoPE, the original ALiBi slope schedule, explicit logical-position causal masking, and a full-versus-cached comparison. It is a correctness reference, not a fused production kernel. The small projected Q/K/V arrays are declared hand fixtures, not claimed learned model activations."}</Prose>

<CodeBlock language={"python"}>{"import math\nimport numpy as np\n\n\ndef sinusoidal(positions, width, base=10000.0):\n    if width < 2 or width % 2:\n        raise ValueError(\"Use an even encoding width of at least two.\")\n    positions = np.asarray(positions, dtype=np.float64)\n    frequencies = base ** (-np.arange(0, width, 2) / width)\n    angles = positions[..., None] * frequencies\n    return np.stack((np.sin(angles), np.cos(angles)), -1).reshape(\n        *positions.shape, width)\n\n\ndef learned_positions(table, positions):\n    positions = np.asarray(positions)\n    if not np.issubdtype(positions.dtype, np.integer):\n        raise ValueError(\"Table positions must be integers.\")\n    if np.any(positions < 0) or np.any(positions >= len(table)):\n        raise ValueError(\"Position is outside the learned table.\")\n    return table[positions]\n\n\ndef rotate(values, positions, base=10000.0):\n    values = np.asarray(values, dtype=np.float64)\n    width = values.shape[-1]\n    if width < 2 or width % 2:\n        raise ValueError(\"Use an even rotary width of at least two.\")\n    angles = np.asarray(positions)[..., None] * base ** (\n        -np.arange(0, width, 2) / width)\n    even, odd = values[..., 0::2], values[..., 1::2]\n    return np.stack((even * np.cos(angles) - odd * np.sin(angles),\n                     even * np.sin(angles) + odd * np.cos(angles)),\n                    -1).reshape(values.shape)\n\n\ndef slopes(head_count):\n    if head_count < 1:\n        raise ValueError(\"A head count must be positive.\")\n    def powers(count):\n        start = 2 ** (-2 ** -(math.log2(count) - 3))\n        return [start ** (index + 1) for index in range(count)]\n    lower = 2 ** int(math.floor(math.log2(head_count)))\n    if lower == head_count:\n        return np.array(powers(lower))\n    return np.array(powers(lower) + powers(2 * lower)[::2][:head_count-lower])\n\n\ndef mix(query, keys, values, query_ids, key_ids, mode, slope=.5):\n    query_ids, key_ids = np.asarray(query_ids), np.asarray(key_ids)\n    scores = query @ keys.T / math.sqrt(query.shape[-1])\n    if mode == \"alibi\":\n        scores -= slope * (query_ids[:, None] - key_ids[None, :])\n    legal = key_ids[None, :] <= query_ids[:, None]\n    if np.any(~legal.any(axis=-1)):\n        raise ValueError(\"Every query needs at least one legal key.\")\n    scores = np.where(legal, scores, -np.inf)\n    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))\n    weights /= weights.sum(axis=-1, keepdims=True)\n    return weights @ values\n\n\nq = np.array([[1, 0, .5, -1], [.5, 1, -1, .25], [2, -.5, .25, 1]])\nk = np.array([[.5, 1, 1, 0], [1, -.5, .5, 1], [-.5, .75, 1, -1]])\nv = np.array([[2, 0], [0, 3], [1, -1]])\npositions = np.array([7, 8, 9])\n\nprint(\"position 1:\", np.round(sinusoidal([1], 8)[0], 3))\nprint(\"three-head slopes:\", slopes(3))\ntable = np.array([[.2, 0], [0, .3], [-.1, .1]])\nprint(\"learned slots 2,0:\", learned_positions(table, [2, 0]))\n\nfor mode in (\"rope\", \"alibi\"):\n    q_used = rotate(q, positions) if mode == \"rope\" else q\n    k_used = rotate(k, positions) if mode == \"rope\" else k\n    full = mix(q_used, k_used, v, positions, positions, mode)\n\n    # Prefill positions 7 and 8, then append a correctly positioned new key.\n    cached_keys, cached_values = k_used[:2].copy(), v[:2].copy()\n    cached_keys = np.concatenate((cached_keys, k_used[2:]), axis=0)\n    cached_values = np.concatenate((cached_values, v[2:]), axis=0)\n    last = mix(q_used[2:], cached_keys, cached_values,\n               positions[2:], positions, mode)\n    print(mode, \"last:\", np.round(last[0], 6),\n          \"matches full:\", np.allclose(last, full[2:], atol=1e-12))\n\n# Keep the legal cache entries fixed, but give the query the wrong RoPE angle.\nwrong = mix(rotate(q[2:], [0]), rotate(k, positions), v,\n            [9], positions, \"rope\")\nprint(\"wrong query angle:\", np.round(wrong[0], 6))"}</CodeBlock>

<Prose>{"Recorded outputs include:"}</Prose>

<CodeBlock language={"text"}>{"position 1: [0.841 0.54  0.1   0.995 0.01  1.    0.001 1.   ]\nthree-head slopes: [0.0625     0.00390625 0.25      ]\nlearned slots 2,0: [[-0.1  0.1]\n                  [ 0.2  0. ]]\nrope last: [1.035332 1.29716 ] matches full: True\nalibi last: [0.340434 2.281648] matches full: True\nwrong query angle: [0.65852  1.291697]"}</CodeBlock>

<Prose>{"For the RoPE last query, the actual weights over positions 7, 8, 9 are approximately "}<code>{"[0.487697,0.452366,0.059937]"}</code>{". With the wrong query angle they become "}<code>{"[0.185156,0.526635,0.288209]"}</code>{". There is no shape error to warn us. Comparing the actual output with a full causal reference catches the bug."}</Prose>

<PositionCacheLab />

<H3>{"Match position geometry to the ordinary attention API"}</H3>

<Prose>{"The "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/position_library_bridge.py"}>{"complete position/API bridge"}</a>{" composes standard adjacent-pair RoPE and ALiBi with PyTorch SDPA. The former transforms Q/K before matching; the latter supplies a floating additive score mask containing both distance penalties and negative infinity on illegal entries. A boolean mask would express legality alone. The program compares direct and API values and gradients for two offset queries reading four cached keys, with explicit positions 7–10 and zero dropout. Run "}<code>{"python position_library_bridge.py"}</code>{" with PyTorch."}</Prose>

<Prose>{"Learned absolute positions take the ordinary "}<code>{"nn.Embedding"}</code>{" route. The same program checks its output against direct table indexing and shows why position ID 7 appearing twice accumulates two gradient contributions into row 7, while unused rows remain unchanged. Fixed sinusoids and prescribed ALiBi slopes have no trainable table unless we deliberately introduce one. The scratch formulas and "}<code>{"PositionClassifier"}</code>{" remain the code owners; this API bridge opens no second positional model."}</Prose>

<Prose>{"This mapping is for the exact conventions taught here, not blanket parity with every checkpoint's RoPE. Half-split pairing, partial rotation, changed bases and context scaling are explicit different transformations in §§3/8. For a checkpoint, use its configuration and code; a shared name does not authorize substituting adjacent-pair rotation."}</Prose>

<Prose>{""}<strong>{"Change the constraint:"}</strong>{" add 100 to both query and key IDs, preserving their legal relation, then add 100 only to queries. "}<strong>{"Hint:"}</strong>{" standard RoPE and ALiBi depend on relative position within this fixed-input operator. "}<strong>{"Solution:"}</strong>{" the common shift leaves scores/outputs unchanged up to rounding; a query-only shift changes offsets and can change outputs. Compare the actual values, not merely whether the final argmax stays the same."}</Prose>

<H3>{"Padding, packed records and numerical precision"}</H3>

<Prose>{"For a left-padded batch, a common logical-position construction is "}<code>{"valid.cumsum(-1)-1"}</code>{", assigning the first valid token position 0. Replace padding indices with a safe value before table lookup, then exclude those entries with the mask. Whether this convention matches a specific checkpoint depends on its training and generation implementation. A uniform position shift cancels locally in standard RoPE when all relevant content and masks are held fixed; arbitrary per-token renumbering does not. Additive learned/sinusoidal vectors generally change under even a common shift."}</Prose>

<Prose>{"When independent documents are packed into one tensor, resetting their position IDs is not enough. A document/block mask must prevent an example from attending to another example. Otherwise equal position IDs do not stop information leakage. Likewise, a key-padding mask often suppresses padded keys but does not automatically erase padded query outputs; exclude those outputs from losses or pooling as appropriate."}</Prose>

<Prose>{"Compute phases at adequate precision. Adjacent large integers can become the same number if converted too early to a low-precision format. For example, BF16 cannot represent every integer beyond 256. Constructing all positions directly in BF16 can therefore give neighboring tokens identical phases before sine/cosine is evaluated. A common implementation calculates phases in float32 and casts the resulting sine/cosine values as needed; our small reference uses float64. Neither floating-point choice provides arbitrary precision at unlimited positions. Validate the actual target range, checkpoint convention and kernel."}</Prose>

<Prose>{"Standard PyTorch Transformer layers do not automatically choose a positional scheme. Use the model's actual embedding/attention implementation and versioned documentation rather than assuming "}<code>{"nn.TransformerEncoder"}</code>{" inserts RoPE or sinusoidal features for you. The real program above makes the injection site explicit; production implementations may fuse the same operation without materializing the intermediate arrays."}</Prose>

<H2>{"8. Deeper route: extending the context without confusing the claims"}</H2>

<H3>{"Three different length questions"}</H3>

<Prose>{"“This model supports a longer context” can mean several things:"}</Prose>

<ol start={1}><li>{""}<strong>{"The operation is defined."}</strong>{" The position lookup/rotation, mask and cache accept that length."}</li><li>{""}<strong>{"The model remains useful."}</strong>{" Loss and downstream quality stay acceptable on the new distribution."}</li><li>{""}<strong>{"The model uses the additional information."}</strong>{" It can retrieve or combine evidence far away, rather than ignoring most of the added context."}</li></ol>

<Prose>{"An ALiBi penalty is defined at arbitrarily large finite distances in real arithmetic, but grows without a finite bound as distance increases. A sine/rotation formula is also defined beyond training indices. Neither is a theorem about the model's behavior on a longer task. The softmax denominator sees more competitors, content distributions change, and long-range computations may require skills the training examples never demanded."}</Prose>

<Prose>{"Even good average language-model loss can hide a failure on an instruction that needs one distant fact. A long-context evaluation should vary both total length and evidence location, include distractors and multiple pieces of evidence, inspect short-context retention, and measure the task that matters. A single successful “needle in a haystack” example does not establish long-document reasoning. The "}<a href={"https://arxiv.org/pdf/2305.19466"}>{"NoPE study"}</a>{" is useful here because it evaluates several algorithmic tasks and separates them from perplexity claims."}</Prose>

<H3>{"Frequency, wavelength and the base"}</H3>

<Prose>{"For "}<InlineMath>{"d_k=64"}</InlineMath>{", base 10000 gives first frequency 1 and last frequency "}<InlineMath>{"10000^{-31/32}\\approx0.00013335"}</InlineMath>{". The first wavelength is "}<InlineMath>{"2\\pi\\approx6.2832"}</InlineMath>{" positions; the last is 47,117.243."}</Prose>

<Prose>{"Changing the base to 500000 leaves the first frequency exactly 1. For pair "}<InlineMath>{"r"}</InlineMath>{" it multiplies the wavelength by"}</Prose>

<div className="neural-equation"><MathBlock>{"\\left(\\frac{500000}{10000}\\right)^{2r/64}=50^{r/32}."}</MathBlock></div>

<Prose>{"The last wavelength becomes 2,084,764.773, about 44.25 times as long. It is not 50 times for every pair, and the first pair does not change at all. At position 128,000, a 47,117-position wavelength has completed about 2.72 turns, not 20. One channel wrapping is not the same as the entire multiscale code becoming identical, nor must every channel remain monotone over the whole context."}</Prose>

<PositionFrequencyFigure />

<Prose>{"The base and rotary dimension are checkpoint/architecture choices. There is no universal formula such as “base equals 60 times the target length.” A suitable choice depends on learned Q/K features, training lengths, positional scaling and task. The "}<a href={"https://arxiv.org/html/2407.21783v3#S3.S2"}>{"Llama 3 report"}</a>{" records a 500000 base; its "}<a href={"https://arxiv.org/html/2407.21783v3#S3.S4.SS2"}>{"long-context training section"}</a>{" also describes staged long-sequence training. The resulting capability was not created by editing a configuration field after an otherwise unchanged short-context training run."}</Prose>

<H3>{"Position Interpolation"}</H3>

<Prose>{"Let a model be trained with nominal context "}<InlineMath>{"L"}</InlineMath>{", and choose a target "}<InlineMath>{"L'=sL"}</InlineMath>{". "}<strong>{"Position Interpolation"}</strong>{", PI, feeds "}<InlineMath>{"t/s"}</InlineMath>{" into the same rotation formula:"}</Prose>

<div className="neural-equation"><MathBlock>{"R_t\\quad\\longrightarrow\\quad R_{t/s}."}</MathBlock></div>

<Prose>{"Equivalently, divide all angular frequencies by "}<InlineMath>{"s"}</InlineMath>{". Distances are compressed too: two new positions 8 slots apart have the old phase difference of 1 slot when "}<InlineMath>{"s=8"}</InlineMath>{". This moves a large range of offsets into a smaller phase range, but also changes local distinctions."}</Prose>

<Prose>{"For training length 64 and target 256, "}<InlineMath>{"s=4"}</InlineMath>{". Position 255 maps to 63.75, not 63. The interval "}<code>{"[0,256)"}</code>{" maps into "}<code>{"[0,64)"}</code>{", but the model was trained at integer positions 0–63; fractional positions and altered token spacing are a new input condition. Endpoint-preserving scaling "}<InlineMath>{"t(63/255)"}</InlineMath>{" is another possible rule, with slightly different spacing. Name which rule you use."}</Prose>

<Prose>{"The "}<a href={"https://arxiv.org/html/2306.15595v2#S2.SS3"}>{"PI paper §2.3"}</a>{" gives the rescaling and a bounded interpolation analysis for a fixed trigonometric score function. Its model experiments include further training. A bound on a fixed score function is not an end-to-end guarantee about hidden states, all softmax competitors or task correctness. Uniform rescaling can work well in a particular protocol, but it is not free of a short-distance tradeoff."}</Prose>

<H3>{"Base scaling, often called NTK-aware scaling"}</H3>

<Prose>{"One proposed alternative for rotary width "}<InlineMath>{"d"}</InlineMath>{" greater than 2 changes the base to"}</Prose>

<div className="neural-equation"><MathBlock>{"b'=b\\,s^{d/(d-2)}."}</MathBlock></div>

<Prose>{"Pairr then has"}</Prose>

<div className="neural-equation"><MathBlock>{"\\theta'_r=\\theta_r\\,s^{-2r/(d-2)}."}</MathBlock></div>

<Prose>{"At "}<InlineMath>{"r=0"}</InlineMath>{" there is no change. At the last pair "}<InlineMath>{"r=d/2-1"}</InlineMath>{", the frequency is divided by "}<InlineMath>{"s"}</InlineMath>{". Intermediate pairs receive intermediate scaling. This preserves the fastest phase changes while stretching the slowest wavelengths. The "}<InlineMath>{"d=2"}</InlineMath>{" formula is undefined and must not be applied blindly."}</Prose>

<Prose>{"The name refers to the reasoning that motivated the heuristic; it does not constitute a proof that any pretrained network behaves like its infinite-width neural tangent kernel or that this base change is optimal. The "}<a href={"https://arxiv.org/html/2309.00071v3#S3"}>{"YaRN paper's methodology and appendices"}</a>{" explain the relationship among PI, base changes and later interpolation schemes."}</Prose>

<H3>{"YaRN: select frequencies and adjust score sharpness"}</H3>

<Prose>{"Uniform interpolation changes every frequency; a base change changes most frequencies by different amounts. A "}<strong>{"by-parts"}</strong>{" construction instead asks how many turns each pair made within the original context:"}</Prose>

<div className="neural-equation"><MathBlock>{"r_i=\\frac{L\\theta_i}{2\\pi}=\\frac{L}{\\lambda_i}."}</MathBlock></div>

<Prose>{"Under the paper's linear ramp in this rotation count, let"}</Prose>

<div className="neural-equation"><MathBlock>{"\\gamma(r)=\\operatorname{clip}\\left(\\frac{r-\\alpha}{\\beta-\\alpha},0,1\\right),\n\\qquad\n\\widetilde\\theta_i=(1-\\gamma(r_i))\\frac{\\theta_i}{s}+\\gamma(r_i)\\theta_i."}</MathBlock></div>

<Prose>{"Pairs with few turns are fully interpolated; pairs with many turns remain unchanged; the middle blends the two. The paper uses "}<InlineMath>{"\\alpha=1,\\beta=32"}</InlineMath>{" in its Llama experiments. These are thresholds in "}<strong>{"rotations during the original context"}</strong>{", not raw feature indices or universal constants."}</Prose>

<Prose>{"YaRN combines this frequency treatment with an empirical attention-temperature adjustment. If softmax originally sees "}<InlineMath>{"q'^\\top k'/\\sqrt d"}</InlineMath>{", dividing this logit by temperature "}<InlineMath>{"T"}</InlineMath>{" makes it sharper when "}<InlineMath>{"T<1"}</InlineMath>{". Scaling "}<strong>{"both"}</strong>{" q and k by "}<InlineMath>{"c=\\sqrt{1/T}"}</InlineMath>{" has the same effect because the dot product scales by "}<InlineMath>{"c^2"}</InlineMath>{". The paper's suggested fit is"}</Prose>

<div className="neural-equation"><MathBlock>{"c=1+0.1\\ln s,\\qquad\\text{logit multiplier}=c^2."}</MathBlock></div>

<Prose>{"At "}<InlineMath>{"s=8"}</InlineMath>{", "}<InlineMath>{"c\\approx1.207944"}</InlineMath>{", so logits are multiplied by about 1.459129. Multiplying them by "}<InlineMath>{"\\sqrt{1+0.1\\ln s}"}</InlineMath>{" is a different rule. The temperature fit is an empirical recipe, not a conservation law that exactly compensates every longer softmax."}</Prose>

<Prose>{"The full mechanism therefore changes both relative phases and attention sharpness. In partial-rotation implementations, scaling only the rotary subset does not multiply the entire dot product uniformly. Check whether the attention scale is applied separately and how unrotated channels are treated."}</Prose>

<Prose>{"The displayed frequency comparison uses the "}<strong>{"paper's ramp in rotation count"}</strong>{". Practical checkpoint libraries can discretize the boundary indices and use a ramp over pair indices, giving a different intermediate curve. A label such as “YaRN” is not enough to reproduce every checkpoint: use its versioned implementation, parameter names, rotary width and scaling factors. "}<a href={"https://huggingface.co/docs/transformers/main/en/internal/rope_utils"}>{"Current Transformers RoPE documentation"}</a>{" distinguishes several schemes and per-layer configurations; its "}<code>{"main"}</code>{" documentation is mutable, so pin the version when reproducing a model."}</Prose>

<PositionExtensionLab />

<H3>{"Dynamic scaling and cached representations"}</H3>

<Prose>{"A dynamic rule can change frequencies as the current length grows. That raises a consistency problem: old keys might have been rotated with yesterday's frequency vector while the new query uses today's."}</Prose>

<Prose>{"For fixed content keys in one attention layer, you can store unrotated keys or rephase existing keys from the old rotation into the new one. In the hand fixture from §7, changing the base from 10000 to 100 and rotating only the new query gives last output "}<code>{"[0.973912,1.389528]"}</code>{"; recomputing all Q/K rotations consistently at the new base gives "}<code>{"[0.998788,1.344242]"}</code>{"."}</Prose>

<Prose>{"There is a further model-level distinction. In a multilayer decoder, cached hidden states and values may themselves have been computed under the old frequencies. Rephasing keys alone does not generally reproduce a fresh full-prefix pass through "}<strong>{"every layer"}</strong>{" at the new fixed frequency rule. Define the intended dynamic algorithm, maintain internally consistent cache metadata and compare against the appropriate reference. If exact equivalence to a fresh pass under a new global configuration is required, earlier states may need recomputation. A “cache fix” must state what equivalence it promises."}</Prose>

<H3>{"XPos: a relative amplitude factor as well as a rotation"}</H3>

<Prose>{"XPos extends the geometry using reciprocal query/key scaling. In a real-coordinate form, for each pair "}<InlineMath>{"i"}</InlineMath>{" choose "}<InlineMath>{"0<\\zeta_i<1"}</InlineMath>{" and a positive scale "}<InlineMath>{"S"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"q'_m=\\zeta_i^{m/S}R_mq_m,\\qquad\nk'_n=\\zeta_i^{-n/S}R_nk_n."}</MathBlock></div>

<Prose>{"The pair's score becomes"}</Prose>

<div className="neural-equation"><MathBlock>{"\\zeta_i^{(m-n)/S}\\,q_m^\\top R_{n-m}k_n."}</MathBlock></div>

<Prose>{"For a causal key "}<InlineMath>{"n\\le m"}</InlineMath>{", the amplitude factor attenuates older contributions. The query/key norms individually are no longer preserved. Scaling both vectors in the same direction would produce an unwanted absolute-position factor instead of this relative factor."}</Prose>

<Prose>{"The "}<a href={"https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py"}>{"TorchScale XPos implementation"}</a>{" uses a pair scale equivalent to "}<InlineMath>{"\\zeta_i=(2i/d+0.4)/1.4"}</InlineMath>{", default "}<InlineMath>{"S=512"}</InlineMath>{", a centered exponent and reciprocal scaling for keys. For the first pair, "}<InlineMath>{"\\zeta_0=2/7"}</InlineMath>{", so a 512-position causal separation multiplies that pair's score amplitude by 2/7. Higher-frequency pairs have smaller "}<InlineMath>{"\\zeta"}</InlineMath>{" under this rule. Omitting the division by "}<InlineMath>{"S"}</InlineMath>{" would instead apply "}<InlineMath>{"(2/7)^{512}"}</InlineMath>{", an entirely different and numerically extreme factor."}</Prose>

<Prose>{"The centering choice can improve numerical range while preserving a common relative factor when handled consistently. It also belongs in the cache convention. The "}<a href={"https://arxiv.org/pdf/2212.10554"}>{"XPos/LEX paper"}</a>{" separately studies its encoding and blockwise masking; partial rotary dimensions alone are not “XPos.” A fixed amplitude decay also does not establish monotonicity of every content-dependent signed score."}</Prose>

<XposPositionFigure />

<H2>{"9. Choose the position system for the actual task"}</H2>

<H3>{"What changes, and what stays expensive?"}</H3>

<NeuralTable caption={"What changes, and what stays expensive?"} headers={[<>{"Mechanism"}</>,<>{"Position parameters"}</>,<>{"Main injection site"}</>,<>{"A question to ask before using it"}</>]} rows={[[<>{"Learned absolute"}</>,<>{""}<InlineMath>{"L_{\\max}d"}</InlineMath>{""}</>,<>{"Input rows"}</>,<>{"Are supported indices and training coverage adequate?"}</>],[<>{"Sinusoidal absolute"}</>,<>{"None"}</>,<>{"Input rows"}</>,<>{"Does the model learn to use this multiscale signal on the relevant lengths?"}</>],[<>{"Shaw-style relative vectors"}</>,<>{"Depends on clipped offset range, width and sharing"}</>,<>{"Query-dependent scores and optionally values"}</>,<>{"Which offsets can share a category without losing needed distinctions?"}</>],[<>{"T5-style scalar buckets"}</>,<>{"Buckets×heads per shared table"}</>,<>{"Scores"}</>,<>{"How much distance/direction precision should the bins preserve?"}</>],[<>{"Standard RoPE"}</>,<>{"None for fixed frequencies"}</>,<>{"Q/K coordinates"}</>,<>{"Which basis, rotary width, frequencies and cache offsets does the model expect?"}</>],[<>{"ALiBi"}</>,<>{"None for fixed slopes"}</>,<>{"Scores"}</>,<>{"Is the chosen directional/proximity prior appropriate for the task and mask?"}</>]]} />

<Prose>{"For a new small model, a simple learned or sinusoidal baseline is useful if its position semantics match the problem. For a pretrained model, preserve its specified scheme first. A frequency change, table extension or bias replacement changes the function the checkpoint computes; it is not a cosmetic implementation substitution."}</Prose>

<Prose>{"RoPE adds work linear in the number of rotated coordinates. Full attention still computes pairwise scores, with "}<InlineMath>{"O(BHL^2d_k)"}</InlineMath>{" attention arithmetic and an "}<InlineMath>{"L^2"}</InlineMath>{" score/probability array if implemented eagerly. A tiled attention kernel can avoid storing the full matrix while calculating the same attention function, within floating-point differences. Positional encoding itself does not make full attention linear in sequence length."}</Prose>

<Prose>{"ALiBi can be generated from position vectors and head slopes. A naive implementation materializes an H × L × L bias; a compatible kernel can compute needed entries or use the causal row-constant identity. Thus “ALiBi requires an extra dense matrix” and “ALiBi is free everywhere” are both implementation-dependent claims. Report actual shapes and measured performance if making a speed comparison."}</Prose>

<Prose>{"With full rotary width and matching query/key widths, standard RoPE does not reduce KV-cache dimensions. For B batches, N layers, Hkv cached heads, length "}<InlineMath>{"L"}</InlineMath>{" and head widths "}<InlineMath>{"d_k,d_v"}</InlineMath>{", unquantized cache storage is"}</Prose>

<div className="neural-equation"><MathBlock>{"BNH_{kv}L(d_k+d_v)\\times\\text{bytes per stored element}."}</MathBlock></div>

<Prose>{"An example B=1, N=32, Hkv=8, L=4096, dk=dv=128 with two-byte elements needs 536,870,912 bytes, or 512 MiB, for those K/V tensors. Rotating keys changes their values, not this count. Quantization scales/metadata, padding allocation and other runtime state add their own storage. The "}<strong>{"next"}</strong>{" lesson, "}<a href={"/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals"}>{"Grouped-Query and Multi-Query Attention"}</a>{", changes "}<InlineMath>{"H_{kv}"}</InlineMath>{" and explains the actual sharing computation. RoPE can apply once to each stored key head and separately to each query head, provided corresponding frequency/basis conventions match. Reducing stored heads is a different mechanism from supplying position."}</Prose>

<H3>{"Useful applications beyond words in a sentence"}</H3>

<Prose>{""}<strong>{"Movement, irregular time and event streams."}</strong>{" An ordinal sample index measures order. An actual timestamp measures elapsed time. If one sensor records events at 0, 1 and 20 seconds, assigning positions 0, 1, 2 hides the nineteen-second gap. A continuous position formula can accept "}<code>{"[0,1,20]"}</code>{", but its frequency units are now radians per second and must suit that scale. A measurement system may need both event order and elapsed time; an additive learned lookup can instead encode a finite time-bin category. In our Libras example, only sample order was supplied reliably, so the lesson does not relabel slot differences as exact seconds."}</Prose>

<Prose>{""}<strong>{"Images and video."}</strong>{" Rasterizing a patch grid into one long list creates accidental one-dimensional neighbors: the last patch of one row and first patch of the next have consecutive flattened indices. A two-dimensional encoding can represent row and column separately. One construction allocates coordinate pairs to x and y, applying phases "}<InlineMath>{"x\\theta_i"}</InlineMath>{" in one subset and "}<InlineMath>{"y\\theta_j"}</InlineMath>{" in another. Their score contributions depend on "}<InlineMath>{"\\Delta x"}</InlineMath>{" and "}<InlineMath>{"\\Delta y"}</InlineMath>{", not just a flattened offset. A video can add temporal coordinates. A position design must decide how special tokens, different resolutions and multiple frames share these coordinates. The later "}<a href={"/learn/path/full-curriculum/vision-transformers-vit-deit-swin-dinov2?module=deep-learning-fundamentals"}>{"Vision Transformers lesson"}</a>{" develops patch grids, learned-grid interpolation and relative window geometry."}</Prose>

<Prose>{""}<strong>{"Coordinates versus identities."}</strong>{" In a set of physical objects, rearranging the storage order should not change a prediction if each object's actual coordinates move with it. In a sentence, swapping words while retaining their slots should change the represented sentence. This is the same distinction as our movement workspace. Choosing position IDs deliberately can preserve the symmetry you want and break the symmetry the task must distinguish."}</Prose>

<Prose>{"These examples explain why there is no universal “best position vector.” Start with the relation the learner or application needs—absolute slot, signed distance, elapsed time, two-dimensional displacement—and trace where that relation enters the function."}</Prose>

<PositionApplicationsFigure />

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"10. Practice: change the situation, then explain the result"}</H2>

<Prose>{"Try the questions before opening hints or solutions. Exercises 1–4 check the core route; 5–8 extend it. No pretrained download or long training run is needed."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Two rearrangements"}</H3>

<Prose>{"An encoder receives points A, B, C with position IDs 0, 1, 2 and uses learned absolute input embeddings, no dropout and mean pooling. Compare (a) storing the records in order C, A, B with their IDs 2, 0, 1, and (b) assigning points C, A, B to IDs 0, 1, 2. Which output is guaranteed to equal the original? Explain at the input to the first block."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write the three content-plus-position vectors before and after each edit. Check whether you merely permuted existing rows."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"In (a), the combined rows are "}<code>{"[xC+p2,xA+p0,xB+p1]"}</code>{", a permutation of "}<code>{"[xA+p0,xB+p1,xC+p2]"}</code>{". The equivariant encoder followed by mean pooling produces the same output. In (b), they are "}<code>{"[xC+p0,xA+p1,xB+p2]"}</code>{"; generally these are different vectors, so equality is not guaranteed. The model can still happen to predict the same class in (b), but the architectural guarantee concerns the full output function and applies to (a)."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. A different rotary pair"}</H3>

<Prose>{"Use one pair with frequency 1, query q="}<code>{"[1,0]"}</code>{" at position 2 and key k="}<code>{"[0,1]"}</code>{" at position 5. Compute the unscaled dot product after rotation. Then shift both positions by 20. Finally move only the key one further position. Does the score necessarily decrease?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use "}<InlineMath>{"q^\\top R_{n-m}k"}</InlineMath>{", and work out "}<InlineMath>{"R_\\phi[0,1]"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{""}<InlineMath>{"R_\\phi[0,1]=[-\\sin\\phi,\\cos\\phi]"}</InlineMath>{". The original score is "}<InlineMath>{"-\\sin3\\approx-0.141120"}</InlineMath>{". Shifting both positions to 22 and 25 leaves offset 3 and the score unchanged. Moving only the key to 6 gives offset 4 and score "}<InlineMath>{"-\\sin4\\approx0.756802"}</InlineMath>{", which is larger. RoPE encodes a relative phase; it does not require scores to fall with distance. If these are two-dimensional attention logits, divide the dot products by "}<InlineMath>{"\\sqrt2"}</InlineMath>{" before softmax."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. How much evidence overcomes the bias?"}</H3>

<Prose>{"A causal head uses ALiBi slope 0.25. Key A is 8 positions farther from the query than key B. How much larger must A's scaled content score be to tie B's final score? With equal content scores, what are their attention odds? Does that answer depend on how many other legal keys exist?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Subtract the final scores. Their difference controls the ratio of softmax probabilities."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"A needs an extra "}<InlineMath>{"0.25\\times8=2"}</InlineMath>{" content-logit units to tie. With equal content scores, "}<InlineMath>{"A_A/A_B=e^{-2}\\approx0.135335"}</InlineMath>{". Other keys change both normalized probabilities but cancel from their ratio. Neither probability itself is 0.135335 unless the remaining normalization happens to make that so. A finite penalty does not prohibit A."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Construct and repair a cache bug"}</H3>

<Prose>{"In the three-key reference example, shift the key positions from "}<code>{"[7,8,9]"}</code>{" to "}<code>{"[0,1,2]"}</code>{" and the last query's position from 9 to 2, retaining the same content vectors and legal order. Run it and compare the last RoPE output with the original. Now return to the original key positions and reset "}<strong>{"only"}</strong>{" the last query's rotation to 0 while retaining key rotations for "}<code>{"[7,8,9]"}</code>{" and the original allowed-key mask. Explain why the two edits have different results. How would you separately expose a wrongly offset causal mask?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"A common shift preserves all pairwise offsets. The query's rotary angle and its legal-key relation are separate inputs in this transparent program."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The common shift preserves offsets, so the last output remains approximately "}<code>{"[1.035332,1.297160]"}</code>{". Resetting only the query's angle changes those offsets and gives "}<code>{"[0.658520,1.291697]"}</code>{". The latter experiment keeps the legal entries fixed to isolate the rotation error. To expose a mask error, keep rotations correct but use a local query index 0 to construct legality against key IDs 7–9: this falsely masks every key, and the reference must reject that empty legal set. In a full decoder, use both correct logical IDs and the correct valid/document relation."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Design a directional encoder"}</H3>

<Prose>{"A model uses "}<InlineMath>{"-a|i-j|"}</InlineMath>{", shared rowwise blocks and mean pooling. You want it to distinguish a list from its reversal. Propose one change that can remove the symmetry and one change that cannot. Explain why increasing the number of identical-structure layers does not solve the issue by itself."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Check whether the entire pipeline commutes with the reversal permutation. Changing parameter values cannot break an equality that holds for all parameter values."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Adding distinct absolute position vectors, using signed relative categories with independently learned values, or introducing an appropriate directional mask can remove reversal symmetry. Increasing width or adding more shared blocks with the same symmetric-distance rule cannot: each block remains reversal-equivariant, and mean pooling remains invariant. Removing the symmetry creates the capacity to distinguish directions; it does not guarantee successful training or correct predictions on all movements. If order should be irrelevant in the application, breaking it may be undesirable."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. A stretching rule is not a performance curve"}</H3>

<Prose>{"A width 8 RoPE head uses base 10000 and extends nominal length 64 to 256 by PI. Find the four original frequencies, the four new frequencies, the mapped value of position 255 and the original wavelength of the slowest pair. What happens under "}<InlineMath>{"s=1"}</InlineMath>{"? What evidence would still be missing before claiming good 256-token task performance?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use "}<InlineMath>{"\\theta_r=10000^{-2r/8}"}</InlineMath>{", divide by 4 for PI, and calculate a full turn as "}<InlineMath>{"2\\pi/\\theta_r"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Original frequencies are "}<code>{"[1,0.1,0.01,0.001]"}</code>{"; PI gives "}<code>{"[0.25,0.025,0.0025,0.00025]"}</code>{". Position 255 maps to 63.75. The slowest original wavelength is "}<InlineMath>{"2000\\pi\\approx6283.1853"}</InlineMath>{" positions. With "}<InlineMath>{"s=1"}</InlineMath>{" the frequencies and positions are unchanged; a YaRN-style score multiplier also becomes 1. These calculations establish the geometric transformation. They provide no actual loss, retrieval or reasoning result from a trained model at length 256. That requires a declared evaluation protocol, representative held-out examples and controls for retained short-context behavior."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. A partial-rotation temperature trap"}</H3>

<Prose>{"An attention score has rotary contribution 2 and unrotated contribution 3, before the common head-width scaling. A developer multiplies only the rotary Q/K coordinates by "}<InlineMath>{"c=2"}</InlineMath>{" and claims to multiply the entire logit by 4. What actually happens? Give a way to achieve the claimed uniform factor."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Each rotary dot product receives two factors of "}<InlineMath>{"c"}</InlineMath>{", but the unrotated dot product receives neither."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The original unscaled score is 2+3=5. Scaling only rotary coordinates gives "}<InlineMath>{"4\\times2+3=11"}</InlineMath>{", not 20. To get a uniform factor 4, multiply the completed dot-product score by 4, or scale "}<strong>{"all"}</strong>{" query and key coordinates by 2. The usual head-width division can remain separate. This is why matching a context-extension checkpoint requires its attention-scale placement as well as its frequency vector."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. A time unit changes the meaning"}</H3>

<Prose>{"A sensor produces three observations at 0, 1 and 20 seconds. A sinusoidal time encoding uses frequency 0.1 radians/second. Another programmer passes timestamps in milliseconds without changing the frequency. Calculate the phase of the last observation under each program. Give the correct frequency in radians/millisecond and explain why index positions 0, 1, 2 solve a different problem."}</Prose>

<details><summary>Hint</summary>

<Prose>{"An angle is timestamp multiplied by frequency. Convert both quantities consistently."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The intended phase is "}<InlineMath>{"20\\times0.1=2"}</InlineMath>{" radians. Passing 20,000 milliseconds with unchanged 0.1 gives 2000 radians. The corresponding frequency is 0.0001 radians/millisecond, restoring phase 2. Index positions 0, 1, 2 retain event order but discard the unequal temporal gaps. Neither is universally wrong: choose the coordinate whose relation matters, then state its units."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--next" data-lesson-ending="next"><H2>{"What comes next?"}</H2>

<Prose>{"You can now locate position at the input, in Q/K geometry or in score biases; distinguish invariance from sensitivity; and verify that a cached computation uses the intended offsets. Continue in the planned sequence to "}<a href={"/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals"}>{"Grouped-Query Attention and Multi-Query Attention"}</a>{". That lesson asks how several query heads can share keys and values, and what this changes in cache storage and computation. Its position operations build directly on the conventions here."}</Prose></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References and another way to learn"}</H2>

<Prose>{"Use these selectively according to the part you want to understand better. Formula definitions, our independently calculated fixtures and the real local experiment serve different purposes."}</Prose>

<ul><li>{""}<strong>{"A visual refresher before the positional details:"}</strong>{" "}<a href={"https://www.3blue1brown.com/lessons/attention/"}>{"3Blue1Brown, Attention in transformers, step-by-step"}</a>{" includes its video and illustrated written adaptation. The inspected Q/K, dot-product, softmax and masking explanations help reconnect geometry to value mixing. Its diagrams use query columns rather than this lesson's query rows, which the notes explain. The adjective/noun behavior is explicitly hypothetical; this is a background attention resource, not a RoPE implementation tutorial. The written adaptation was inspected; the video was not independently watched for this packet."}</li><li>{""}<strong>{"A compact comparison with diagrams:"}</strong>{" "}<a href={"https://sebastianraschka.com/faq/docs/rope-vs-absolute-positional-embeddings.html"}>{"Sebastian Raschka, RoPE versus absolute positional embeddings"}</a>{". The full inspected article traces the input-table versus Q/K-rotation distinction, partial RoPE and cache offsets. Use it after §3 for another concise explanation; follow this lesson's examples for the detailed calculations."}</li><li>{""}<strong>{"A book/notebook route:"}</strong>{" "}<a href={"https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html"}>{"Dive into Deep Learning §11.6"}</a>{", especially 11.6.3–11.6.5. The inspected code, frequency visuals, absolute/relative subsections and exercises provide a second path through sinusoidal encodings. Its CNN/RNN comparison connects to earlier architecture lessons. A binary-counter analogy is about scales, not a claim that floating-point encodings always use fewer physical bits."}</li><li>{""}<strong>{"Learned position tables inside a complete GPT:"}</strong>{" "}<a href={"https://www.youtube.com/watch?v=kCc8FmEb1nY"}>{"Andrej Karpathy's Let's build GPT video"}</a>{", linked from the "}<a href={"https://karpathy.ai/zero-to-hero.html"}>{"creator's Zero to Hero course"}</a>{", is a longer code-first learning option. The inspected "}<a href={"https://github.com/karpathy/nanoGPT/blob/master/model.py"}>{"nanoGPT implementation"}</a>{" shows "}<code>{"wpe"}</code>{", strict context checks and addition to token embeddings inside a real decoder. It is companion implementation evidence from the same author, not a claim that the video transcript or its unavailable older repository was inspected. This option teaches the learned-input mechanism and surrounding model, not the RoPE/ALiBi extensions."}</li><li>{""}<strong>{"Original fixed encoding:"}</strong>{" "}<a href={"https://arxiv.org/pdf/1706.03762"}>{"Vaswani et al., Attention Is All You Need, §3.5"}</a>{". Read for the position formula, original comparison and the motivation behind its relative-shift identity."}</li><li>{""}<strong>{"Learned relations and buckets:"}</strong>{" "}<a href={"https://aclanthology.org/N18-2074.pdf"}>{"Shaw et al., §§3.1–3.3"}</a>{" defines relation vectors and their score/value roles; "}<a href={"https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py"}>{"T5's maintained attention source"}</a>{" makes signed bucket conventions and cache offsets concrete. Code on a moving branch must be version-pinned for reproduction."}</li><li>{""}<strong>{"Rotary geometry:"}</strong>{" "}<a href={"https://arxiv.org/html/2104.09864v5"}>{"Su et al., RoFormer §§2–3"}</a>{" derives the rotation construction. Read its claims about distance behavior together with the explicit counterexample here and the "}<a href={"https://arxiv.org/html/2306.15595v2"}>{"PI analysis §2"}</a>{". A geometric identity and a broad empirical tendency are different statements."}</li><li>{""}<strong>{"Linear bias:"}</strong>{" "}<a href={"https://arxiv.org/html/2108.12409v2"}>{"Press et al., ALiBi §§2–3"}</a>{" supplies the causal mechanism and comparison context; the "}<a href={"https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py"}>{"author's implementation"}</a>{" supplies the non-power-of-two schedule and row-constant optimization."}</li><li>{""}<strong>{"Extending a trained model:"}</strong>{" "}<a href={"https://arxiv.org/html/2306.15595v2"}>{"Position Interpolation"}</a>{", "}<a href={"https://arxiv.org/html/2309.00071v3"}>{"YaRN §§3–4 and appendices"}</a>{", and "}<a href={"https://arxiv.org/html/2407.21783v3#S3.S4.SS2"}>{"the Llama 3 training report §3.4.2"}</a>{". Use the inspected methods/training/evaluation sections to separate frequency changes from the complete adaptation recipe. This packet did not repeat their large training runs."}</li><li>{""}<strong>{"Further geometric and evaluation depth:"}</strong>{" "}<a href={"https://arxiv.org/pdf/2212.10554"}>{"XPos/LEX"}</a>{" and its "}<a href={"https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py"}>{"TorchScale code"}</a>{" explain reciprocal amplitude factors; "}<a href={"https://arxiv.org/pdf/2305.19466"}>{"Kazemnejad et al."}</a>{" tests length generalization in decoder-only models and motivates the causal NoPE distinction. The full appendix proof campaign is optional research depth, not required to start the next lesson."}</li><li>{""}<strong>{"Implementation reference:"}</strong>{" "}<a href={"https://huggingface.co/docs/transformers/main/en/internal/rope_utils"}>{"Transformers RoPE utilities"}</a>{" documents named variants and layer-specific configuration. Consult the version matching your model; changing a field is not proof of extended-context quality."}</li><li>{""}<strong>{"Real data and reproduction:"}</strong>{" "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI Libras Movement"}</a>{", "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/data-provenance.md"}>{"our provenance"}</a>{", "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/author-calculations.py"}>{"complete training program"}</a>{", "}<a href={"/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/mechanism-calculations.py"}>{"independent mechanism calculations"}</a>{". These support the actual observations and editable examples in this lesson."}</li></ul></section>
  </div>,
};
