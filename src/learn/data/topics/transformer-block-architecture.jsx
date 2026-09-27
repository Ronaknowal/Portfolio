// Generated from the complete prepared manuscript by scripts/generate-transformer-block-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { BlockCommunicationFigure, BlockWiringFigure, BlockNormalizationLab, BlockFeedforwardLab, BlockTraceLab, BlockMovementLab, BlockProbeLab, BlockEvidenceFigure, BlockTaskFigure, BlockVariantsFigure, BlockCostsLab, BlockSystemsFigure } from '../../components/lesson-labs/TransformerBlockLabs.jsx';
export default {
  title: 'Transformer Block Architecture',
  readTime: '~85 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson neural-lesson-neutral transformer-block-lesson"><LessonIntro prerequisites="Self-attention, residual additions and per-position feature normalization. Their essential operations are refreshed where used." sections={[["1-a-repeated-workspace-for-each-sequence-position","1. A repeated workspace for each sequence position"],["2-read-the-wiring-pre-norm-and-post-norm","2. Read the wiring: pre-norm and post-norm"],["3-what-the-normalizer-actually-normalizes","3. What the normalizer actually normalizes"],["4-the-ffn-detect-a-feature-pattern-and-write-an-update","4. The FFN: detect a feature pattern and write an update"],["5-follow-a-complete-block-and-implement-it","5. Follow a complete block and implement it"],["6-a-complete-model-on-real-ordered-movements","6. A complete model on real ordered movements"],["7-deeper-what-a-gradient-plot-can-and-cannot-establish","7. Deeper: what a gradient plot can and cannot establish"],["8-deeper-assemble-task-models-and-recognize-real-variants","8. Deeper: assemble task models and recognize real variants"],["9-deeper-count-the-work-then-measure-the-implementation","9. Deeper: count the work, then measure the implementation"],["10-practice-change-the-problem-before-checking-the-answer","10. Practice: change the problem before checking the answer"],["where-this-leads-and-another-way-to-learn","Where this leads, and another way to learn"]]}>Follow one representation through communication, feature processing and the saved residual path.</LessonIntro>
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit features, normalizer offset/scale, FFN matrices, branch multiplier, pre/post placement and supported retained trajectory inputs. Update normalization reference sets, feature writes, residual state, block output and derivative paths together. Show exact zero-branch and common-shift cases. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to reason about placement and branch scaling using their actual function and gradient effects rather than an architecture slogan."}</Prose>

<Prose>{"A hand follows a curved path. To recognize the movement, a model needs more than a list of isolated coordinates: it needs to relate different moments, combine the resulting evidence, and revise its description of the movement. The previous "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"self-attention lesson"}</a>{" supplied the communication mechanism. A "}<strong>{"Transformer block"}</strong>{" packages that communication with a small feature-processing network and carefully arranged update paths. Several blocks can then refine the same sequence of representations."}</Prose>

<Prose>{"This lesson follows one question throughout: "}<strong>{"what changes, and what remains available, as information passes through a block?"}</strong>{" Answering it lets you read an architecture diagram, implement the actual computation, diagnose a misleading gradient plot, and understand why two models both called Transformers need not have the same behavior."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" read §§1–6, try the block and movement investigations, and solve exercises 1, 2 and 4. You should be able to trace the tensor shapes and explain normalization placement before continuing. Sections 7–9 develop gradient geometry, architecture variants and systems costs; they are deeper branches you can return to. The remaining exercises follow those deeper sections. The next topic is positional encoding, not a requirement to master every large-model training method first."}</Prose>

<H2>{"1. A repeated workspace for each sequence position"}</H2>

<Prose>{"Represent one sequence as a table "}<code>{"X"}</code>{" with "}<code>{"L"}</code>{" rows and "}<code>{"d"}</code>{" columns. A row belongs to a position: a word piece, an image patch, or a sampled hand location. Its columns are learned features. With a batch of "}<code>{"B"}</code>{" sequences the shape is "}<code>{"(B,L,d)"}</code>{". The raw hand coordinates have only two columns; a learned input projection can turn them into a wider representation before any block runs."}</Prose>

<Prose>{"A standard block preserves this outer shape. Its intermediate feed-forward computation may use more columns, and its attention creates relationships between rows, but its output is again "}<code>{"(B,L,d)"}</code>{". That compatible interface is why blocks can be stacked."}</Prose>

<Prose>{"The important operations have different jobs:"}</Prose>

<NeuralTable caption={"1. A repeated workspace for each sequence position"} headers={[<>{"Operation"}</>,<>{"What it reads to update one position"}</>,<>{"What it does"}</>]} rows={[[<>{"Self-attention"}</>,<>{"Other permitted positions as well as this one"}</>,<>{"Builds context-dependent mixtures of projected values"}</>],[<>{"Positionwise feed-forward network, or FFN"}</>,<>{"The current representation at this position"}</>,<>{"Detects and combines learned feature patterns"}</>],[<>{"Residual addition"}</>,<>{"A saved representation and a same-shape update"}</>,<>{"Adds the update to the existing representation"}</>],[<>{"Feature normalization"}</>,<>{"The features within this position"}</>,<>{"Re-centers/rescales, or just rescales, the vector using a declared rule"}</>]]} />

<Prose>{"“Positionwise” does not mean “a separate network trained for every position.” The "}<strong>{"same"}</strong>{" FFN parameters are reused at every row. Its input can already contain information from distant positions because attention ran earlier. A local operation on a contextual representation can therefore respond to a global pattern."}</Prose>

<BlockCommunicationFigure />

<H3>{"Refresh the attention part"}</H3>

<Prose>{"In one head, project the input into queries "}<code>{"Q"}</code>{", keys "}<code>{"K"}</code>{" and values "}<code>{"V"}</code>{". A receiving query compares against permitted keys, normalizes those scores, and mixes their values:"}</Prose>

<div className="neural-equation"><MathBlock>{"A=\\operatorname{softmax}_{\\text{keys}}\\left(\\frac{QK^\\top}{\\sqrt{d_h}}+M\\right),\n\\qquad Y=AV."}</MathBlock></div>

<Prose>{"Here "}<code>{"d_h"}</code>{" is the query/key width, and "}<code>{"M"}</code>{" represents a mask: zero for a legal pair and negative infinity for a forbidden pair in the mathematical formula. Multiple heads perform separate mixtures; concatenation and an output projection return width "}<code>{"d"}</code>{". The previous lesson derives these operations and explains numerical masking. In this lesson write the whole same-shape attention transformation as "}<code>{"Attn(X)"}</code>{" so that we can concentrate on its surrounding architecture."}</Prose>

<H3>{"A block, a stack and a task model are different things"}</H3>

<Prose>{"A block transforms representations. A "}<strong>{"stack"}</strong>{" applies several blocks, usually with separately learned parameters. A "}<strong>{"task model"}</strong>{" also includes an input representation and an output rule. A language model projects each final row into vocabulary logits. A movement classifier can pool final rows and project the pooled vector into movement classes. An image model first constructs patch representations. The same block does not decide those task boundaries for you."}</Prose>

<Prose>{"The original 2017 model combined an encoder stack and a decoder stack. Its encoder block had self-attention followed by an FFN. Its decoder block inserted cross-attention between masked self-attention and the FFN; queries came from the decoder, while keys and values came from the encoder. The original FFN used "}<strong>{"ReLU"}</strong>{", and normalization followed each residual addition. These are historical design choices, not a definition requiring every Transformer to retain them. "}<a href={"https://arxiv.org/pdf/1706.03762"}>{"Vaswani et al., §3"}</a>{""}</Prose>

<H2>{"2. Read the wiring: pre-norm and post-norm"}</H2>

<Prose>{"A residual connection saves the incoming representation and adds a learned update. If a sublayer produces "}<code>{"U"}</code>{" from "}<code>{"X"}</code>{", the addition is "}<code>{"X+U"}</code>{", not concatenation. Both tensors must describe matching positions and features. The "}<a href={"/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals"}>{"residual connections lesson"}</a>{" explains why an identity path helps information and derivatives travel through a network."}</Prose>

<Prose>{"Normalization can occur "}<strong>{"inside the update branch"}</strong>{" or "}<strong>{"after the addition"}</strong>{". That choice changes the function even if every parameter tensor is identical."}</Prose>

<Prose>{"For a "}<strong>{"pre-norm block"}</strong>{", omitting dropout for the moment:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{aligned}\nH&=\\operatorname{Norm}_1(X),\\\\\nU&=\\operatorname{Attn}(H),\\\\\nZ&=X+U,\\\\\nG&=\\operatorname{Norm}_2(Z),\\\\\nV&=\\operatorname{FFN}(G),\\\\\nY&=Z+V.\n\\end{aligned}"}</MathBlock></div>

<Prose>{"The attention branch reads a normalized view of the input; the addition still receives the original "}<code>{"X"}</code>{". The FFN branch reads a normalized view of the already updated "}<code>{"Z"}</code>{"; the second addition retains "}<code>{"Z"}</code>{". Do not accidentally use "}<code>{"Norm₂(X)"}</code>{" in the FFN branch: it would miss the attention update from this block."}</Prose>

<Prose>{"For a "}<strong>{"post-norm block"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{aligned}\nU&=\\operatorname{Attn}(X),\\\\\nZ&=\\operatorname{Norm}_1(X+U),\\\\\nV&=\\operatorname{FFN}(Z),\\\\\nY&=\\operatorname{Norm}_2(Z+V).\n\\end{aligned}"}</MathBlock></div>

<Prose>{"Now every path to "}<code>{"Z"}</code>{", including the bypass, passes through the first normalization. The second normalization similarly acts on the whole updated state. “Pre” and “post” describe this placement relative to the sublayer/residual computation; they do not mean input-data preprocessing versus postprocessing."}</Prose>

<BlockWiringFigure />

<H3>{"The zero-update test"}</H3>

<Prose>{"Temporarily set both learned sublayer outputs to zero. Pre-norm returns "}<code>{"Y=X"}</code>{": both additions preserve the incoming stream. Post-norm returns "}<code>{"Norm₂(Norm₁(X))"}</code>{". It can still change the input even though attention and the FFN contribute nothing."}</Prose>

<Prose>{"For "}<code>{"X=[1,2,5,8]"}</code>{" with identity LayerNorm gains, zero offsets and epsilon "}<code>{"10⁻⁵"}</code>{", pre-norm leaves "}<code>{"[1,2,5,8]"}</code>{". Post-norm produces approximately "}<code>{"[-1.095440,-.730293,.365147,1.460586]"}</code>{" after its two normalizations. The latter is a useful representation, but it is not the identity function."}</Prose>

<Prose>{"This is a precise structural difference. It does not establish that one design always learns better. Training behavior also depends on initialization, depth, optimizer, task and the actual loss. We will test the distinction without converting it into a universal architecture ranking."}</Prose>

<H2>{"3. What the normalizer actually normalizes"}</H2>

<Prose>{"For a vector "}<code>{"x"}</code>{" of width "}<code>{"d"}</code>{", "}<strong>{"LayerNorm"}</strong>{" first calculates statistics across its features:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mu=\\frac1d\\sum_i x_i,\\qquad\nv=\\frac1d\\sum_i(x_i-\\mu)^2,\\qquad\n\\operatorname{LN}(x)_i=\\gamma_i\\frac{x_i-\\mu}{\\sqrt{v+\\epsilon}}+\\beta_i."}</MathBlock></div>

<Prose>{"The learned gain "}<code>{"γ"}</code>{" and offset "}<code>{"β"}</code>{" have one entry per feature. Epsilon is a small positive number that keeps the denominator usable near a constant vector. The variance here uses divisor "}<code>{"d"}</code>{", not the sample-estimation divisor "}<code>{"d−1"}</code>{". This operation is repeated independently at each "}<code>{"(batch,position)"}</code>{" pair. It does not collect statistics across the time axis, and it does not need running averages from earlier batches. "}<a href={"https://arxiv.org/pdf/1607.06450"}>{"Ba et al., §3"}</a>{""}</Prose>

<Prose>{"For our four-feature vector, "}<code>{"μ=4"}</code>{" and "}<code>{"v=7.5"}</code>{". Subtracting four yields "}<code>{"[-3,-2,1,4]"}</code>{"; dividing by "}<code>{"√(7.5+10⁻⁵)"}</code>{" gives approximately:"}</Prose>

<div className="neural-equation"><MathBlock>{"[-1.095444,-.730296,.365148,1.460593]."}</MathBlock></div>

<Prose>{"The centered, rescaled vector has root mean square almost one and Euclidean length almost "}<strong>{"two"}</strong>{", because "}<code>{"√d=2"}</code>{". Calling its length “one” confuses RMS with L2 norm. Its variance is "}<code>{"v/(v+ε)"}</code>{", slightly below one. Learned unequal gains and offsets can subsequently change its mean, variance and length again."}</Prose>

<Prose>{""}<strong>{"RMSNorm"}</strong>{" uses the same feature group but skips subtracting the mean:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{RMSNorm}(x)_i=\n\\gamma_i\\frac{x_i}{\\sqrt{\\frac1d\\sum_jx_j^2+\\epsilon}}."}</MathBlock></div>

<Prose>{"For "}<code>{"[1,2,5,8]"}</code>{", the denominator is approximately "}<code>{"√23.5"}</code>{", giving "}<code>{"[.206284,.412568,1.031421,1.650274]"}</code>{". Its mean is positive. The common form shown here has learned gain and no additive offset. Removing the centering calculation changes both invariances and computation; it is not a claim that the two functions are numerically interchangeable. "}<a href={"https://arxiv.org/pdf/1910.07467"}>{"Zhang & Sennrich, §4"}</a>{""}</Prose>

<H3>{"An offset and a scale are different interventions"}</H3>

<Prose>{"Add five to every feature, producing "}<code>{"[6,7,10,13]"}</code>{". LayerNorm removes the common offset and gives the same output, up to rounding. RMSNorm changes because it measures distance from zero rather than distance from this vector's mean. For a zero-mean input with matching gain, offset zero and matching epsilon, the two normalizers agree exactly."}</Prose>

<Prose>{"Multiplying every feature by a positive constant leaves either normalized direction unchanged when epsilon is zero and the denominator is nonzero. With positive epsilon, that scale invariance is approximate. Negative scaling flips the normalized direction before affine offsets. A constant vector becomes zero under centered LayerNorm before its affine offset; a nonzero constant vector does not become zero under RMSNorm."}</Prose>

<BlockNormalizationLab />

<Prose>{"For causal prediction, these axes matter. "}<code>{"LayerNorm(d)"}</code>{" on "}<code>{"(B,L,d)"}</code>{" only uses one position's features. Normalizing jointly over "}<code>{"(L,d)"}</code>{" can let future inputs change an earlier normalized state even when attention has a perfect causal mask. Similarly, padding must be excluded from any pooling over positions. An architecture name cannot repair an incorrect reduction axis."}</Prose>

<H2>{"4. The FFN: detect a feature pattern and write an update"}</H2>

<Prose>{"For row-vector notation, a two-layer FFN is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{FFN}(x)=\\phi(xW_{\\rm up}+b_{\\rm up})W_{\\rm down}+b_{\\rm down},\n\\quad\nW_{\\rm up}\\in\\mathbb R^{d\\times f},\\quad\nW_{\\rm down}\\in\\mathbb R^{f\\times d}."}</MathBlock></div>

<Prose>{"The hidden width "}<code>{"f"}</code>{" is the number of intermediate feature responses. A common design expands it beyond "}<code>{"d"}</code>{", but expansion is a modeling choice. The up projection asks learned questions about the representation; the activation changes each response; the down projection combines those responses into a "}<code>{"d"}</code>{"-feature update. Each row of "}<code>{"W_down"}</code>{" is an update direction multiplied by its corresponding hidden activation."}</Prose>

<Prose>{"Without an activation, the composition collapses to one affine map: "}<code>{"x(W_up W_down)+(b_up W_down+b_down)"}</code>{". The intervening nonlinearity is what makes this particular two-map FFN more expressive than a single affine transformation. "}<strong>{"Attention itself is already nonlinear through its input-dependent softmax"}</strong>{", and normalization is nonlinear too. An FFN is not the only source of nonlinearity in a Transformer."}</Prose>

<H3>{"A feature circuit you can calculate"}</H3>

<Prose>{"Consider representation "}<code>{"g=[1,2,-1,-2]"}</code>{" and no biases:"}</Prose>

<div className="neural-equation"><MathBlock>{"W_{\\rm up}=\\begin{bmatrix}1&0&1\\\\0&1&1\\\\-1&0&0\\\\0&-1&0\\end{bmatrix},\n\\qquad\nW_{\\rm down}=\\begin{bmatrix}1&0&0&0\\\\0&1&0&0\\\\0&0&.5&-.5\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The three up-projection responses are "}<code>{"[g₁−g₃,g₂−g₄,g₁+g₂]=[2,4,3]"}</code>{". ReLU preserves these positive responses, and the down projection writes "}<code>{"[2,4,1.5,−1.5]"}</code>{". The third response writes equal-and-opposite changes in two output coordinates. A hidden activation is not itself the final feature update."}</Prose>

<Prose>{"For a second row "}<code>{"[-1,0,1,0]"}</code>{", the up responses are "}<code>{"[-2,0,-1]"}</code>{"; ReLU makes all three zero. The same network can therefore write a substantial update for one position and no update for another. Editing the first row does not change the second row's FFN output "}<strong>{"when these are the direct FFN inputs"}</strong>{". If you edit an earlier input before attention, that edit may first change both rows' context."}</Prose>

<Prose>{"This pointwise distinction explains a useful connection to a "}<code>{"1×1"}</code>{" convolution over channels: both can apply the same feature transformation at every spatial position without mixing neighboring positions at that step. Stacking such transformations with communication operations gives a different model from either kind alone."}</Prose>

<BlockFeedforwardLab />

<H3>{"ReLU, GELU and a multiplicative gate"}</H3>

<Prose>{"ReLU is "}<code>{"max(0,z)"}</code>{": it sets negative responses to zero. GELU is "}<code>{"z Φ(z)"}</code>{", where "}<code>{"Φ"}</code>{" is the standard normal cumulative distribution function. SiLU, also called Swish with parameter one, is "}<code>{"z σ(z)"}</code>{", where "}<code>{"σ(z)=1/(1+e⁻ᶻ)"}</code>{". GELU and SiLU smoothly attenuate negative values; they do not generally turn half of a layer into exact zeros. Their standard scalar forms have no learned parameters. A library may use an approximation to GELU, so specify it when exact comparisons matter."}</Prose>

<Prose>{"A "}<strong>{"SwiGLU FFN"}</strong>{" uses two up projections and an elementwise product:"}</Prose>

<div className="neural-equation"><MathBlock>{"u=xW_{\\rm up},\\qquad\ng=\\operatorname{SiLU}(xW_{\\rm gate}),\\qquad\n\\operatorname{FFN}_{\\rm SwiGLU}(x)=(u\\odot g)W_{\\rm down}."}</MathBlock></div>

<Prose>{"The representation creates both the value-like response and the multiplier. If a hidden coordinate has "}<code>{"u=2"}</code>{" and gate preactivation "}<code>{"1"}</code>{", its product is "}<code>{"2×.731059≈1.462117"}</code>{". If the gate preactivation becomes "}<code>{"−1"}</code>{", the multiplier is "}<code>{"−.268941"}</code>{" and the product is "}<code>{"−.537883"}</code>{". Unlike a sigmoid gate, the SiLU multiplier is neither restricted to "}<code>{"[0,1]"}</code>{" nor always nonnegative. Calling it a gate is a description of multiplicative control, not a probability interpretation."}</Prose>

<Prose>{"The bias-free two-map FFN has "}<code>{"2df"}</code>{" weights. The bias-free SwiGLU FFN has "}<code>{"3df"}</code>{" because it adds an independent up projection. To match the dominant weight count of a conventional FFN with "}<code>{"f=4d"}</code>{", choose a gated width near "}<code>{"8d/3"}</code>{": "}<code>{"3d(8d/3)=8d²"}</code>{". Rounding widths for implementation changes the exact count. For "}<code>{"d=512"}</code>{", a biased GELU FFN with "}<code>{"f=2048"}</code>{" has 2,099,712 parameters; a bias-free SwiGLU FFN rounded to "}<code>{"f=1408"}</code>{" has 2,162,688. They are close, not identical."}</Prose>

<Prose>{"Shazeer's GLU-variant study compared these mechanisms under a declared T5 training setup with reduced gated widths. It is evidence for a useful option, not proof that a ratio or activation is optimal for every task. "}<a href={"https://arxiv.org/html/2002.05202v1"}>{"GLU Variants Improve Transformer, §§1–3"}</a>{""}</Prose>

<H3>{"A useful memory interpretation, with the algebra visible"}</H3>

<Prose>{"The sum "}<code>{"Σᵣ φ(x·wᵣ+bᵣ)vᵣ"}</code>{" resembles reading from learned keys and writing their associated values. Here "}<code>{"wᵣ"}</code>{" is a column of the up matrix, and "}<code>{"vᵣ"}</code>{" is a row of the down matrix. These “keys” and “values” are persistent parameters, rather than representations generated from the current neighboring tokens. The coefficients need not add to one; with GELU they can be negative. This is different from the probability-normalized attention mixture."}</Prose>

<Prose>{"Researchers have investigated such pattern-reading behavior in trained language-model FFNs. It offers a way to ask which inputs activate a direction and what that direction contributes. It does not require every neuron to mean one clean human concept, or imply that all factual knowledge resides only in FFNs. An interpretable hand-built circuit explains the operation; discovering a trained model's features requires evidence. "}<a href={"https://aclanthology.org/2021.emnlp-main.446.pdf"}>{"Geva et al., §§2–3"}</a>{""}</Prose>

<H2>{"5. Follow a complete block and implement it"}</H2>

<Prose>{"We will first use a small exact fixture so that every stage can be inspected. There are two positions, four features, one attention head, no biases and no dropout. "}<code>{"Q=K=H"}</code>{", "}<code>{"V=H/4"}</code>{", and the attention output projection is identity, where "}<code>{"H"}</code>{" is whichever input the wiring supplies to attention. Scaling the values by a quarter keeps the update easy to compare with the bypass. The FFN uses the two matrices above and ReLU. This is a declared mathematical example, not a trained language model."}</Prose>

<Prose>{"Start with"}</Prose>

<div className="neural-equation"><MathBlock>{"X=\\begin{bmatrix}1&2&5&8\\\\3&0&2&1\\end{bmatrix}."}</MathBlock></div>

<Prose>{"For pre-norm, the first row entering attention is approximately "}<code>{"[-1.095444,-.730296,.365148,1.460593]"}</code>{". The second row is approximately "}<code>{"[1.341635,-1.341635,.447212,-.447212]"}</code>{". Scores divide their dot products by "}<code>{"√4=2"}</code>{". After the two-key softmax, mix the rows of "}<code>{"H/4"}</code>{" and add those updates to the "}<strong>{"original"}</strong>{" "}<code>{"X"}</code>{". Normalize that new state for the FFN, compute the hidden responses, and add the resulting FFN update to the state."}</Prose>

<Prose>{"Follow the second position all the way through. Its attention weights on positions 1 and 2 are approximately "}<code>{"[.076571,.923429]"}</code>{". The resulting stages are:"}</Prose>

<NeuralTable caption={"5. Follow a complete block and implement it"} headers={[<>{"Stage"}</>,<>{"Second position's vector, approximately"}</>]} rows={[[<>{"Original carried input"}</>,<>{""}<code>{"[3,0,2,1]"}</code>{""}</>],[<>{"Attention update"}</>,<>{""}<code>{"[.288757,−.323706,.110232,−.075282]"}</code>{""}</>],[<>{"First addition, "}<code>{"Z"}</code>{""}</>,<>{""}<code>{"[3.288757,−.323706,2.110232,.924718]"}</code>{""}</>],[<>{"Normalized FFN input"}</>,<>{""}<code>{"[1.330590,−1.356588,.453929,−.427931]"}</code>{""}</>],[<>{"Three ReLU responses"}</>,<>{""}<code>{"[.876661,0,0]"}</code>{""}</>],[<>{"FFN update"}</>,<>{""}<code>{"[.876661,0,0,0]"}</code>{""}</>],[<>{"Final output"}</>,<>{""}<code>{"[4.165418,−.323706,2.110232,.924718]"}</code>{""}</>]]} />

<Prose>{"The first hidden response is positive because "}<code>{"1.330590−.453929≈.876661"}</code>{"; its write direction adds only to feature 1. At the first position all three FFN responses happen to be zero, so its final output equals its contextual state. Repeating the computation with the same parameters under post-norm gives second-position output approximately "}<code>{"[.695474,−1.720968,.632590,.392905]"}</code>{". The changed result comes from the wiring, including which vectors determine attention scores, not from learning new parameters between the two calculations."}</Prose>

<BlockTraceLab />

<Prose>{"The figure is useful because it displays actual vectors at named junctions. A heatmap of vaguely labeled “activation strength” would hide the difference between a normalized branch input, an update and the carried state. We can compute RMS or L2 summaries too, but their labels must say which tensor and which reduction they summarize."}</Prose>

<H3>{"A complete PyTorch block"}</H3>

<Prose>{"The following program uses standard multi-head attention from the previous lesson and implements both wirings directly. Install PyTorch in your own Python environment if needed with "}<code>{"python -m pip install torch"}</code>{". Save the program as "}<code>{"transformer_block.py"}</code>{" and run "}<code>{"python transformer_block.py"}</code>{". It uses CPU float64 for a small comparison. All dropout probabilities are zero so that randomness does not obscure the wiring."}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch import nn\n\ntorch.set_num_threads(1)\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, width, heads, hidden, pre_norm=True):\n        super().__init__()\n        self.pre_norm = pre_norm\n        self.attention = nn.MultiheadAttention(\n            width, heads, dropout=0, batch_first=True)\n        self.norm_attention = nn.LayerNorm(width)\n        self.norm_feedforward = nn.LayerNorm(width)\n        self.feedforward = nn.Sequential(\n            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))\n\n    def forward(self, inputs, blocked=None):\n        h = self.norm_attention(inputs) if self.pre_norm else inputs\n        update, _ = self.attention(\n            h, h, h, attn_mask=blocked, need_weights=False)\n        context = inputs + update\n        if not self.pre_norm:\n            context = self.norm_attention(context)\n        g = self.norm_feedforward(context) if self.pre_norm else context\n        output = context + self.feedforward(g)\n        return output if self.pre_norm else self.norm_feedforward(output)\n\ntorch.manual_seed(31)\nblock = TransformerBlock(8, 2, 16).double()\ninputs = torch.randn(2, 4, 8, dtype=torch.float64)\nblocked = torch.ones(4, 4, dtype=torch.bool).triu(1)\n\nfor pre_norm in (True, False):\n    block.pre_norm = pre_norm\n    reference = nn.TransformerEncoderLayer(\n        8, 2, dim_feedforward=16, dropout=0, activation=\"gelu\",\n        batch_first=True, norm_first=pre_norm).double()\n    reference.self_attn.load_state_dict(block.attention.state_dict())\n    reference.norm1.load_state_dict(block.norm_attention.state_dict())\n    reference.norm2.load_state_dict(block.norm_feedforward.state_dict())\n    reference.linear1.load_state_dict(block.feedforward[0].state_dict())\n    reference.linear2.load_state_dict(block.feedforward[2].state_dict())\n    actual = block(inputs, blocked)\n    expected = reference(inputs, src_mask=blocked)\n    print(pre_norm, tuple(actual.shape),\n          torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))\n    block.zero_grad(set_to_none=True)\n    reference.zero_grad(set_to_none=True)\n    left = inputs.clone().requires_grad_()\n    right = inputs.clone().requires_grad_()\n    probe = torch.linspace(-.9, 1.1, inputs.numel(), dtype=inputs.dtype).reshape_as(inputs)\n    (block(left, blocked)*probe).sum().backward()\n    (reference(right, src_mask=blocked)*probe).sum().backward()\n    torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-10)\n    pairs = [(block.attention, reference.self_attn),\n             (block.norm_attention, reference.norm1),\n             (block.norm_feedforward, reference.norm2),\n             (block.feedforward[0], reference.linear1),\n             (block.feedforward[2], reference.linear2)]\n    with torch.no_grad():\n        for ours, theirs in pairs:\n            for (name, parameter), (other_name, other) in zip(\n                    ours.named_parameters(), theirs.named_parameters(), strict=True):\n                assert name == other_name\n                torch.testing.assert_close(parameter.grad, other.grad, atol=1e-10, rtol=1e-10)\n                parameter.add_(parameter.grad, alpha=-.01)\n                other.add_(other.grad, alpha=-.01)\n    torch.testing.assert_close(block(inputs, blocked), reference(inputs, src_mask=blocked),\n                               atol=1e-10, rtol=1e-10)\n    print({\"pre_norm\": pre_norm, \"gradient_and_update_checks\": \"passed\"})"}</CodeBlock>

<Prose>{"The expected printed lines are "}<code>{"True (2, 4, 8) True"}</code>{" and "}<code>{"False (2, 4, 8) True"}</code>{". The same-parameter comparisons were executed with PyTorch 2.14.0 CPU; the packet records the actual maximum differences. We are checking the complete computation against another implementation, not expecting separately initialized blocks to agree."}</Prose>

<Prose>{"Notice three small decisions. We calculate "}<code>{"Norm(X)"}</code>{" once and reuse it as the input to the three attention projections. The FFN receives the updated contextual state. For this "}<code>{"MultiheadAttention"}</code>{" API, boolean "}<strong>{"True means blocked"}</strong>{"; do not transfer that convention to an API where True means allowed. Padding masks are additionally needed for variable-length batches, and padded query outputs must be excluded from the task's loss or pooling."}</Prose>

<Prose>{"The extended comparison follows the input derivative and each attention, normalizer and FFN parameter through the same nonuniform output probe, then makes one equal SGD update. It reuses attention from the "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"prepared Self-Attention §5 owner"}</a>{", and normalizer mechanisms from the "}<a href={"/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals"}>{"implemented Normalization lesson"}</a>{". The new operation owned here is the block's wiring, including the two placements and parameter/state correspondence."}</Prose>

<Prose>{""}<strong>{"Take control:"}</strong>{" change both normalizers' epsilon to .001 in both routes and repeat. "}<strong>{"Hint:"}</strong>{" epsilon is part of the function, even with copied affine parameters. "}<strong>{"Solution:"}</strong>{" a matched change preserves the checks; changing epsilon on only one side generally breaks both output and derivative equality. For nonzero dropout, instead compare under identical masks or study the stochastic distributions; do not demand pointwise agreement from unrelated draws."}</Prose>

<H3>{"Defaults and training behavior are part of the model"}</H3>

<Prose>{"In the installed reference API, "}<code>{"TransformerEncoderLayer"}</code>{" defaults to post-norm, ReLU, "}<code>{"dim_feedforward=2048"}</code>{", dropout "}<code>{".1"}</code>{" and sequence-first tensors. The FFN default does "}<strong>{"not"}</strong>{" automatically become four times your chosen model width. The explicit arguments above make the intended architecture reviewable. "}<code>{"TransformerEncoder"}</code>{" constructs separate copies of a supplied layer, initially with equal parameter values; those parameters are not shared storage. If independent initial values are intended, initialize the copies appropriately or construct separate blocks with a "}<code>{"ModuleList"}</code>{". "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html"}>{"PyTorch TransformerEncoderLayer"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html"}>{"TransformerEncoder"}</a>{""}</Prose>

<Prose>{"When dropout is part of the intended design, specify its locations. Attention-probability dropout, FFN hidden-activation dropout, and dropout on a branch before residual addition are different operations. In a common pre-norm pattern, for example, "}<code>{"Z=X+Drop(Attn(Norm(X)))"}</code>{"; the bypass itself is retained. The reference encoder layer has additional FFN/residual dropout sites beyond the probability dropout inside its attention module. Matching only the attention constructor's probability does not make two training implementations equivalent. "}<code>{"eval()"}</code>{" disables ordinary module dropout; "}<code>{"no_grad()"}</code>{" alone does not switch it off."}</Prose>

<Prose>{"A pre-norm stack commonly applies a final normalizer before the output head because its carried stream can change scale across blocks. That final normalizer is part of the architecture; it is not supplied automatically by a custom block. Keeping shapes identical while silently moving a normalizer, replacing an activation or deleting a bias changes the represented function. Loading a pretrained checkpoint requires matching its exact architecture and conventions before considering optimization."}</Prose>

<H2>{"6. A complete model on real ordered movements"}</H2>

<Prose>{"Can a stack turn sampled hand positions into useful movement classes? We use the same openly licensed "}<strong>{"Libras Movement"}</strong>{" source introduced in self-attention: 360 recorded trajectories, 45 normalized two-dimensional points each, and 15 movement types. The task is to classify those movement types. These centroid traces are not full signs, body-pose recordings or language translations. "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI Libras Movement"}</a>{""}</Prose>

<H3>{"Give the block an explicit time tag"}</H3>

<Prose>{"The previous pooled self-attention model had no way to distinguish permutations of the same point collection. A positionwise FFN and feature normalization do not by themselves remove that symmetry: both apply the same rule at every row. Here each sampled point receives a simple third coordinate,"}</Prose>

<div className="neural-equation"><MathBlock>{"s_t=2t/44-1,\\quad t=0,\\ldots,44,\n\\qquad h_t=[2x_t-1,\\,2y_t-1,\\,s_t]W_{\\rm stem}+b_{\\rm stem}."}</MathBlock></div>

<Prose>{"The first and last time tags are "}<code>{"−1"}</code>{" and "}<code>{"1"}</code>{". They describe position within the sampled trajectory, not seconds measured by a clock. A linear map sends these three numbers into 24 features. The model can now respond differently to the same location early and late in a movement. This is a deliberately simple position representation; the next lesson explores better choices for different sequence tasks."}</Prose>

<Prose>{"The complete model is:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\text{tagged points}\n\\rightarrow \\text{Linear}(3,24)\n\\rightarrow \\text{Block}_1\n\\rightarrow \\text{Block}_2\n\\rightarrow \\operatorname{LN}(24)\n\\rightarrow \\text{mean over valid positions}\n\\rightarrow \\text{Linear}(24,15)."}</MathBlock></div>

<Prose>{"Each block has two heads and a GELU FFN of width 48. The last linear layer returns logits; softmax is used to inspect class probabilities, while the training cross-entropy function accepts logits directly. Attention is bidirectional because the whole recorded movement is available for classification. That would be the wrong information boundary for predicting a future point before observing it."}</Prose>

<H3>{"The comparison is specified before looking at its results"}</H3>

<Prose>{"There are 330 unique coordinate sequences: 30 rows repeat other trajectories exactly, with consistent labels. Keep the first occurrence of each unique trajectory, then use the fixed class-stratified seed 73 split of 220 training,50 validation and 60 test records. All 45 points from one trajectory stay together; every test class has four examples. The exact row IDs and duplicate groups are retained in the "}<a href={"/learn-code/transformer-block-architecture/author-results.json"}>{"results"}</a>{". The source describes four performers and two collection sessions, but does not attach usable performer/session IDs to each row. This study therefore cannot establish generalization to an unseen signer or collection session."}</Prose>

<Prose>{"Compare pre-norm and post-norm placement with otherwise identical model definitions. For this controlled comparison, "}<strong>{"both"}</strong>{" models have the final LayerNorm before pooling, including the post-norm model. This keeps their 10,263 parameters and readout convention matched; it is not a reproduction of the original translation architecture. Each seed gives the same initial parameter tensors to the two wirings. Both use no dropout, no weight decay, Adam at "}<code>{".003"}</code>{", and 180 full-training-batch epochs. Three declared seeds are 101,102,103."}</Prose>

<Prose>{"For each fit, select the checkpoint with highest validation macro F1, then lower validation cross entropy, then the earliest exact tie. Macro F1 calculates a precision/recall balance per class and averages the 15 class scores. Evaluate the 60 test records after selection. Keeping the same data boundary as the previous lesson supports continuity, but the new task model changes several architectural ingredients; comparing its score with the earlier model does not isolate the benefit of any one added component."}</Prose>

<H3>{"What actually happened"}</H3>

<Prose>{"The complete six-fit CPU experiment produced:"}</Prose>

<NeuralTable caption={"What actually happened"} headers={[<>{"Placement"}</>,<>{"Seed"}</>,<>{"Selected epoch"}</>,<>{"Training correct /220"}</>,<>{"Validation correct /50"}</>,<>{"Test correct /60"}</>,<>{"Test macro F1"}</>]} rows={[[<>{"Pre-norm"}</>,<>{"101"}</>,<>{"83"}</>,<>{"216"}</>,<>{"43"}</>,<>{"49"}</>,<>{".8070"}</>],[<>{"Pre-norm"}</>,<>{"102"}</>,<>{"146"}</>,<>{"219"}</>,<>{"41"}</>,<>{"49"}</>,<>{".8050"}</>],[<>{"Pre-norm"}</>,<>{"103"}</>,<>{"112"}</>,<>{"218"}</>,<>{"43"}</>,<>{"51"}</>,<>{".8435"}</>],[<>{"Post-norm"}</>,<>{"101"}</>,<>{"125"}</>,<>{"218"}</>,<>{"45"}</>,<>{"52"}</>,<>{".8668"}</>],[<>{"Post-norm"}</>,<>{"102"}</>,<>{"170"}</>,<>{"220"}</>,<>{"39"}</>,<>{"51"}</>,<>{".8520"}</>],[<>{"Post-norm"}</>,<>{"103"}</>,<>{"129"}</>,<>{"220"}</>,<>{"42"}</>,<>{"47"}</>,<>{".7814"}</>]]} />

<Prose>{"Both placements learn this small task. Post-norm has more test-correct examples for seeds 101 and 102; pre-norm has more for 103. The outcome does not support declaring pre-norm a universal winner or post-norm unable to train. These six fits vary initialization within one fixed small data split; they are not independent samples of all possible datasets or a hardware benchmark. They also show a train-to-test gap despite near-perfect training accuracy."}</Prose>

<BlockEvidenceFigure />

<H3>{"Inspect a failure before designing a repair"}</H3>

<Prose>{"Choose source row 77, the first test example of class 4, before choosing any favorable prediction. Its source class is "}<strong>{"anticlockwise arc"}</strong>{". Under the seed 101 checkpoints, the pre-norm model predicts class 7 and assigns class 4 probability about "}<code>{".000211"}</code>{"; the post-norm model predicts class 3 and assigns class 4 probability about "}<code>{".200043"}</code>{". Both are wrong. The trace remains useful precisely because we can inspect an actual error instead of showing only impressive predictions."}</Prose>

<BlockMovementLab />

<Prose>{"Three contrasts make the mechanism concrete:"}</Prose>

<ol start={1}><li>{""}<strong>{"Reverse the coordinates while keeping time slots fixed."}</strong>{" The sequence now visits the observed locations in the opposite order. The largest logit change is about 4.885 for pre-norm and 5.746 for post-norm. Both predict class 5, clockwise arc. This is evidence that these models use the relationship between locations and time tags. It is not a new labeled test case proving accuracy on every reversed gesture."}</li><li>{""}<strong>{"Reverse coordinates and their time tags together."}</strong>{" This just permutes the complete tagged records. Each block is permutation-equivariant and the mean is invariant, so the pooled classifier should give the same result. The observed maximum logit changes are below "}<code>{"1.5×10⁻⁶"}</code>{", consistent with floating-point reduction order. Time information lives in the tags; arbitrary storage-row ordering is not an additional secret input."}</li><li>{""}<strong>{"Append five padding records."}</strong>{" Preserve the original 45 tags, assign zero tag to the five pads, block padded keys in "}<strong>{"both"}</strong>{" layers, and exclude padded rows from pooling. The output agrees within "}<code>{"10⁻⁶"}</code>{". If the pads are treated as real observations instead, the largest logit changes are about 1.923 and 2.424. A mask only in the first block would not be enough."}</li></ol>

<Prose>{"Reflecting only frame 23's x coordinate, "}<code>{"x→1−x"}</code>{", gives a further genuine editable-input contrast. The seed 101 pre-norm prediction becomes class 2 and the post-norm prediction becomes class 11. The point editor should compute arbitrary new changes from actual saved weights, not look up one of these presets. Keep the original class attached to the original observed record; an edited path has no newly measured ground-truth label."}</Prose>

<H3>{"Reproduce the study"}</H3>

<Prose>{"Download "}<a href={"/learn-code/transformer-block-architecture/author-calculations.py"}>{"the complete program"}</a>{", "}<a href={"/learn-code/transformer-block-architecture/movement_libras.data"}>{"the original data"}</a>{" and "}<a href={"/learn-code/transformer-block-architecture/movement_libras.names"}>{"source metadata"}</a>{" into one directory. Install "}<code>{"numpy torch scikit-learn"}</code>{" in your own environment, then run "}<code>{"python author-calculations.py"}</code>{". The program contains the full block, time-tagged classifier, deduplication/split, training loop, checkpoint selection, fixtures and interventions. It writes "}<a href={"/learn-code/transformer-block-architecture/author-results.json"}>{"all measured results"}</a>{" and "}<a href={"/learn-code/transformer-block-architecture/block-models.json"}>{"the two actual seed 101 models"}</a>{". The "}<a href={"/learn-code/transformer-block-architecture/data-provenance.md"}>{"provenance record"}</a>{" supplies attribution, exact hashes and the observed runtime versions. No network or GPU is needed after obtaining those files."}</Prose>

<Prose>{"This gives you a full system to inspect: real inputs, a defined representation, a differentiable model, a loss, a selection rule and a held-out result. It also leaves a useful research question: would a representation designed around movement direction or a different validation protocol help? That question calls for a new declared experiment, not a retroactive reinterpretation of this one."}</Prose>

<H2>{"7. Deeper: what a gradient plot can and cannot establish"}</H2>

<Prose>{"A gradient is sensitivity of a "}<strong>{"specified output quantity"}</strong>{" to a specified input or parameter. It is not an intrinsic score attached to a layer. Before plotting gradients, write down the loss, reduction, tensor being differentiated, initialization and measurement norm."}</Prose>

<H3>{"A convincing-looking diagnostic can be identically uninformative"}</H3>

<Prose>{"Take LayerNorm with all gains one and all offsets zero. Its output coordinates sum to zero:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\sum_i\\operatorname{LN}(x)_i\n=\\frac{\\sum_i(x_i-\\mu)}{\\sqrt{v+\\epsilon}}=0."}</MathBlock></div>

<Prose>{"The gradient of this sum with respect to "}<code>{"x"}</code>{" is therefore zero. If you place "}<code>{"loss=output.sum()"}</code>{" after the final default-affine LayerNorm of a post-norm stack, tiny upstream gradients may simply reveal that you asked the network to change a constant. "}<strong>{"One normalizer is already enough."}</strong>{" The learned final gains or offsets can still receive gradients; that does not make the upstream diagnostic informative."}</Prose>

<Prose>{"Use a nonconstant contrast, such as the first normalized coordinate minus the second, to see the distinction. This standalone program was executed:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch.nn import functional as F\n\nx = torch.tensor([1., 2., 5., 8.], dtype=torch.float64,\n                 requires_grad=True)\ny = F.layer_norm(x, (4,), eps=1e-5)\nsum_gradient = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]\ncontrast_gradient = torch.autograd.grad(y[0] - y[1], x)[0]\nprint(sum_gradient)\nprint(contrast_gradient)"}</CodeBlock>

<Prose>{"The first gradient is "}<code>{"[0,0,0,0]"}</code>{". The second is approximately "}<code>{"[.328633,−.389491,.012172,.048686]"}</code>{". The normalizer is differentiable and sensitive to the contrast even though it cannot change the centered sum. If gains are unequal or a nonlinear readout follows the normalizer, the sum need not remain constant; inspect the actual function rather than applying this particular null indiscriminately."}</Prose>

<H3>{"Derive the normalization Jacobian"}</H3>

<Prose>{"This extends the derivative rule in the normalization lesson. Let "}<code>{"c=x−μ1"}</code>{", "}<code>{"s=√(cᵀc/d+ε)"}</code>{", and initially take identity affine parameters. Differentiating the centering and scaling steps gives"}</Prose>

<div className="neural-equation"><MathBlock>{"J_{\\rm LN}(x)\n=\\frac1s\\left(I-\\frac{11^\\top}{d}-\\frac{cc^\\top}{d s^2}\\right)."}</MathBlock></div>

<Prose>{"The first subtraction removes the common-offset direction: "}<code>{"J1=0"}</code>{". For a vector perpendicular to both "}<code>{"1"}</code>{" and "}<code>{"c"}</code>{", the Jacobian multiplies it by "}<code>{"1/s"}</code>{". For the centered radial direction "}<code>{"c"}</code>{", it multiplies by "}<code>{"ε/s³"}</code>{". When epsilon is zero and "}<code>{"c≠0"}</code>{", scaling this vector does not change its normalized direction, so that radial derivative is also zero."}</Prose>

<Prose>{"The crucial point is that "}<code>{"1/s"}</code>{" can be greater than one when the input feature spread is small. A LayerNorm Jacobian is "}<strong>{"not necessarily a contraction"}</strong>{". With learned gains, left-multiply the expression by "}<code>{"diag(γ)"}</code>{", introducing further scale and direction dependence. These equations concern one position's feature vector, not a scalar “gradient retention percentage” shared by every block."}</Prose>

<Prose>{"For a pre-norm residual sublayer "}<code>{"y=x+F(Norm(x))"}</code>{", the local derivative is"}</Prose>

<div className="neural-equation"><MathBlock>{"J_{\\rm pre}=I+J_F(Norm(x))J_{\\rm Norm}(x)."}</MathBlock></div>

<Prose>{"For a post-norm sublayer "}<code>{"y=Norm(x+F(x))"}</code>{", it is"}</Prose>

<div className="neural-equation"><MathBlock>{"J_{\\rm post}=J_{\\rm Norm}(x+F(x))(I+J_F(x))."}</MathBlock></div>

<Prose>{"Pre-norm exposes an explicit identity contribution outside normalization. Other terms can reinforce or cancel it. Through a stack, these Jacobians multiply in order. An identity term in every factor is useful architecture, but does not prove that the product's every direction stays well-conditioned at arbitrary depth."}</Prose>

<H3>{"A declared initialization measurement"}</H3>

<Prose>{"The packet computes stacks of 1,4 and 12 blocks at width 8, two heads and FFN width 16. Both placements use the same initial tensors at each depth. A separately seeded random input has shape "}<code>{"(1,3,8)"}</code>{". After a shared affine-free final LayerNorm, the output is dotted with a fixed random probe of unit L2 length. Autograd measures the L2 norm of the gradient with respect to each "}<strong>{"carried state"}</strong>{". This is a vector-Jacobian product for that probe, not the full Jacobian's largest singular value and not an average of incompatible parameter tensors."}</Prose>

<NeuralTable caption={"A declared initialization measurement"} headers={[<>{"Blocks"}</>,<>{"Pre-norm input-gradient L2"}</>,<>{"Post-norm input-gradient L2"}</>,<>{"Pre-norm final carried-state RMS"}</>,<>{"Post-norm final carried-state RMS"}</>]} rows={[[<>{"1"}</>,<>{".868298"}</>,<>{".828193"}</>,<>{"1.023500"}</>,<>{".999995"}</>],[<>{"4"}</>,<>{".980193"}</>,<>{".851472"}</>,<>{"1.100291"}</>,<>{".999996"}</>],[<>{"12"}</>,<>{".927944"}</>,<>{".893780"}</>,<>{"1.653763"}</>,<>{".999995"}</>]]} />

<Prose>{"Neither placement loses this probe's sensitivity in the measured depths. Replacing the probe with an all-ones output sum makes both input gradients effectively zero after their shared final normalizer. That contrast explains why specifying the output quantity is essential."}</Prose>

<BlockProbeLab /><BlockEvidenceFigure gradients />

<Prose>{"Xiong et al. analyze initialization with a simplified mean-field setting, including single-head attention, zero-initialized query/key matrices giving uniform weights, Gaussian input assumptions, and a loss on a prediction readout. Their analysis and experiments explain why normalization placement can affect initial gradient scale and the usefulness of learning-rate warmup. In particular, the post-norm concern includes large gradients near the output, not just a slogan about vanishing lower-layer gradients. Their result does not make every pre-norm training setup safe without warmup. "}<a href={"https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf"}>{"Xiong et al., §§3–4"}</a>{""}</Prose>

<H3>{"The residual stream's magnitude needs its own explanation"}</H3>

<Prose>{"Across a population of examples, for one coordinate of a carried state "}<code>{"x"}</code>{" and update "}<code>{"u"}</code>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{Var}(x+u)\n=\\operatorname{Var}(x)+\\operatorname{Var}(u)+2\\operatorname{Cov}(x,u)."}</MathBlock></div>

<Prose>{"If independent zero-mean updates of fixed variance accumulate, variance grows linearly with the number of updates. If every update equals the same random vector "}<code>{"z"}</code>{", then after "}<code>{"k"}</code>{" additions the contribution is "}<code>{"kz"}</code>{", whose variance is "}<code>{"k² Var(z)"}</code>{". If an update cancels the state, its variance can shrink instead. Normalizing branch inputs does not erase these correlations or fix the carried stream's variance by itself."}</Prose>

<Prose>{"A final norm, scaled residual branches, depth-aware initialization and alternative norm placement are different tools for controlling a model. Their effects should be measured with a real objective and the intended training recipe. Do not confuse RMS across feature coordinates in one activation with variance across a population; the labels in the table above deliberately use the former."}</Prose>

<H2>{"8. Deeper: assemble task models and recognize real variants"}</H2>

<H3>{"Three familiar attention layouts"}</H3>

<NeuralTable caption={"Three familiar attention layouts"} headers={[<>{"Task layout"}</>,<>{"Self-attention boundary"}</>,<>{"Extra block component"}</>,<>{"Typical output rule"}</>]} rows={[[<>{"Encoder for a complete observation"}</>,<>{"All valid input positions"}</>,<>{"None required"}</>,<>{"Per-position output or pooling/classification"}</>],[<>{"Autoregressive decoder-only model"}</>,<>{"Present and past positions"}</>,<>{"None required"}</>,<>{"Next-token logits for each position"}</>],[<>{"Encoder–decoder model"}</>,<>{"Encoder sees valid source; decoder self-attention is causal"}</>,<>{"Cross-attention from decoder to encoder"}</>,<>{"Target-token logits conditioned on source"}</>]]} />

<Prose>{"For next-token training, a causal mask is only half the specification. Shift inputs and targets. With sequence "}<code>{"[BOS, a, b, c]"}</code>{", input positions "}<code>{"[BOS,a,b]"}</code>{" predict targets "}<code>{"[a,b,c]"}</code>{". The diagonal of causal attention is legal because the current "}<strong>{"input"}</strong>{" token is not the next target. If the current input already contains the answer being scored, masking future tokens does not repair the leakage."}</Prose>

<Prose>{"At generation time, produce a distribution, choose a next token, append it and repeat. A cache can retain each layer's keys and values from earlier steps; cached values must correspond to the same block inputs, positional convention and causal computation as a full-prefix run. Previous outputs stay valid because their legal information did not include the newly appended future token. Cache arithmetic and sharing get their own lessons after positional encoding."}</Prose>

<BlockTaskFigure />

<H3>{"A copy task teaches why the loss denominator matters"}</H3>

<Prose>{"Suppose an autoregressive exercise uses"}</Prose>

<Prose>{""}<code>{"[BOS, a₁,…,a₈, SEP, a₁,…,a₈, EOS]"}</code>{","}</Prose>

<Prose>{"where the eight source symbols are drawn independently and uniformly from ten choices for every new example. There are 19 tokens and 18 next-token prediction targets. The first eight source-symbol targets are fresh randomness given the prefix: no causal architecture can predict them better than the uniform distribution in expectation. Their best expected contribution is eight times "}<code>{"ln10"}</code>{". The repeated symbols and delimiters are predictable in principle. Thus a model that performs the repeat perfectly can still have mean next-token loss approaching"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac8{18}\\ln10\\approx1.023371\\ \\text{nats per target}."}</MathBlock></div>

<Prose>{"Report accuracy on the eight repeated-symbol targets separately from the total loss. "}<code>{"SEP"}</code>{", at input index 9 under zero-based indexing, predicts the first repeated symbol at target index 9. Changing the source length, symbol distribution or set of scored targets changes the loss floor. A finite memorized training set can also behave differently from fresh independent examples. This is a useful debugging task, not proof that only Transformers can copy or that any named small model has already achieved perfect copying."}</Prose>

<H3>{"Modern designs change more than one switch"}</H3>

<Prose>{"The precise computation is more informative than grouping every model into one recommended recipe:"}</Prose>

<NeuralTable caption={"Modern designs change more than one switch"} headers={[<>{"Design"}</>,<>{"Example update rule for one sublayer or block"}</>,<>{"What to notice"}</>]} rows={[[<>{"Sequential pre-norm"}</>,<>{""}<code>{"z=x+A(N₁(x)); y=z+F(N₂(z))"}</code>{""}</>,<>{"FFN reads the attention-updated state"}</>],[<>{"Parallel attention and FFN"}</>,<>{""}<code>{"y=x+A(N(x))+F(N(x))"}</code>{""}</>,<>{"Both branches read the same incoming state; the FFN does not consume this block's attention output"}</>],[<>{"Residual-post normalization"}</>,<>{""}<code>{"y=x+N(F(x))"}</code>{""}</>,<>{"Normalize the branch output before adding it to an unnormalized bypass"}</>],[<>{"Norms before and after a branch"}</>,<>{""}<code>{"y=x+N_out(F(N_in(x)))"}</code>{""}</>,<>{"Controls both branch input and branch update"}</>],[<>{"Scaled post-norm residual"}</>,<>{""}<code>{"y=N(αx+F(x))"}</code>{" with a matching initialization"}</>,<>{"Changes bypass scale and initialization together"}</>]]} />

<Prose>{"PaLM uses parallel attention/MLP branches. Swin V2 uses residual-post normalization; its name must not be confused with normalization "}<strong>{"after the whole addition"}</strong>{". Gemma 2 normalizes sublayer inputs and outputs with RMSNorm. DeepNorm uses a scaled residual together with architecture-dependent initialization. These are real alternatives, so "}<code>{"x+Norm(F(x))"}</code>{" is not automatically a bug. Their cited studies have their own tasks and ablations; changing a line in a pretrained model does not reproduce them. "}<a href={"https://arxiv.org/html/2204.02311v5"}>{"PaLM, §2"}</a>{", "}<a href={"https://arxiv.org/html/2111.09883v2"}>{"Swin V2, §3.2"}</a>{", "}<a href={"https://arxiv.org/html/2408.00118v3"}>{"Gemma 2, §2"}</a>{", "}<a href={"https://arxiv.org/pdf/2203.00555"}>{"DeepNet, §§2,4.1"}</a>{""}</Prose>

<Prose>{"A practical architecture description should therefore state the norm's formula and placement, FFN type and actual hidden width, serial versus parallel branches, attention head layout, positional convention, masks, biases, dropout sites and final readout. “Llama-like” or “modern Transformer” alone does not specify a compatible checkpoint. RMSNorm and SwiGLU are useful examples, but neither is universal across contemporary models."}</Prose>

<BlockVariantsFigure />

<H3>{"A compact RMSNorm/SwiGLU branch you can read"}</H3>

<Prose>{"This standalone program implements the two components explicitly and checks their shapes. It uses float32; a mixed-precision implementation should consider accumulation precision in its reductions. It is a component example, not a claim to reproduce a named model's complete attention or positional convention."}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\nclass RMSNorm(nn.Module):\n    def __init__(self, width, eps=1e-5):\n        super().__init__()\n        self.gain = nn.Parameter(torch.ones(width))\n        self.eps = eps\n\n    def forward(self, x):\n        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True)\n                                 + self.eps)\n        return self.gain * x * inverse_rms\n\nclass SwiGLU(nn.Module):\n    def __init__(self, width, hidden):\n        super().__init__()\n        self.up = nn.Linear(width, hidden, bias=False)\n        self.gate = nn.Linear(width, hidden, bias=False)\n        self.down = nn.Linear(hidden, width, bias=False)\n\n    def forward(self, x):\n        return self.down(self.up(x) * F.silu(self.gate(x)))\n\ntorch.manual_seed(41)\ninputs = torch.randn(2, 3, 12)\nnorm = RMSNorm(12)\nffn = SwiGLU(12, 32)\noutput = inputs + ffn(norm(inputs))\nprint(tuple(output.shape))\nprint(sum(p.numel() for p in ffn.parameters()))"}</CodeBlock>

<Prose>{"It prints "}<code>{"(2,3,12)"}</code>{" and "}<code>{"1152"}</code>{", since the three bias-free matrices contain "}<code>{"3×12×32"}</code>{" weights. Insert this component only where the intended architecture calls for it. The earlier API-comparison program uses a GELU FFN and LayerNorm, so it would no longer be comparing the same function after this replacement."}</Prose>

<H2>{"9. Deeper: count the work, then measure the implementation"}</H2>

<H3>{"Parameters belong to maps, not sequence positions"}</H3>

<Prose>{"For standard attention with total head width equal to "}<code>{"d"}</code>{", its query/key/value/output weights contain "}<code>{"4d²"}</code>{" scalars. With biases they add "}<code>{"4d"}</code>{". A two-map FFN adds "}<code>{"2df+f+d"}</code>{". Two ordinary affine LayerNorms add "}<code>{"4d"}</code>{". One block therefore has"}</Prose>

<div className="neural-equation"><MathBlock>{"P_{\\rm block}=4d^2+2df+f+9d."}</MathBlock></div>

<Prose>{"At "}<code>{"f=4d"}</code>{", this becomes "}<code>{"12d²+13d"}</code>{". For "}<code>{"d=256"}</code>{", it is 789,760 parameters. The exact source program checks its chosen model's count rather than assuming every block has this formula: grouped/latent heads, gated FFNs, different biases or extra normalizers change it. Block count multiplies the count when blocks have independent parameters. Token/position embeddings and the output head must be added separately; tying input/output embeddings changes the total."}</Prose>

<Prose>{"Increasing sequence length creates more activations and work, but does not create new parameters in these shared maps. This is why a parameter count alone does not tell you whether a long-context run will fit in memory."}</Prose>

<H3>{"Distinguish MACs, FLOPs and elapsed time"}</H3>

<Prose>{"A multiply–accumulate, or MAC, multiplies two numbers and accumulates the result. Under the convention of two FLOPs per MAC, the dominant dense work for one forward block on "}<code>{"(B,L,d)"}</code>{" is:"}</Prose>

<NeuralTable caption={"Distinguish MACs, FLOPs and elapsed time"} headers={[<>{"Part"}</>,<>{"MACs"}</>]} rows={[[<>{"Q/K/V and output projections"}</>,<>{""}<code>{"4BLd²"}</code>{""}</>],[<>{"Two FFN projections"}</>,<>{""}<code>{"2BLdf"}</code>{""}</>],[<>{"Score matrix and value mixture"}</>,<>{""}<code>{"2BL²d"}</code>{""}</>]]} />

<Prose>{"At "}<code>{"f=4d"}</code>{", the approximate forward FLOP count is "}<code>{"24BLd²+4BL²d"}</code>{". Divide by "}<code>{"BL"}</code>{" for a per-token count: "}<code>{"24d²+4Ld"}</code>{". These expressions omit softmax, normalization, activations, bias additions and masking overhead. A conventional backward pass through dense maps often adds roughly twice their forward work; that is an approximation for planning, not an exact universal three-times total."}</Prose>

<Prose>{"The quadratic term becomes important as "}<code>{"L"}</code>{" grows. For "}<code>{"d=512,f=2048,B=1"}</code>{", projections plus FFN take about 1.611 billion MACs at "}<code>{"L=512"}</code>{"; the attention matrix products add about .268 billion. At "}<code>{"L=4096"}</code>{", those terms become 12.885 billion and 17.180 billion. The source of the growth matters more than giving either number the unqualified label “Transformer cost.”"}</Prose>

<BlockCostsLab />

<H3>{"Activation memory and inference cache are different objects"}</H3>

<Prose>{"A materialized attention matrix has "}<code>{"BH L²"}</code>{" entries for "}<code>{"H"}</code>{" heads. A training implementation may retain intermediate activations for backward computation, including FFN hidden tensors of shape "}<code>{"(B,L,f)"}</code>{", subject to its particular schedule and recomputation choices. The autoregressive key/value cache instead stores prior keys and values at every layer. With ordinary equal-width K/V heads, its per-layer element count is "}<code>{"2BLd"}</code>{". Neither is the same as parameter storage or optimizer-state storage."}</Prose>

<Prose>{"Exact tiled attention can avoid materializing the whole score/probability matrix while computing the same mathematical softmax-attention operation. Different reduction orders need not give bit-identical floating-point outputs. It changes memory traffic and the execution schedule, not the fact that every permitted query/key interaction is part of dense attention. "}<a href={"https://arxiv.org/pdf/2205.14135"}>{"FlashAttention, §§2–3"}</a>{""}</Prose>

<Prose>{""}<strong>{"Activation checkpointing"}</strong>{" stores selected boundary activations and recomputes omitted intermediates during backward. It trades additional computation for less saved activation memory. Recomputed stochastic operations must preserve the intended randomness and side effects; otherwise the backward path may not correspond to the forward computation. It does not automatically halve every model's memory or double its elapsed time. Prefer measurements for the exact checkpoint policy and workload."}</Prose>

<Prose>{"Fusing a sequence of kernels can reduce launches and intermediate memory transfers. Lower-precision kernels change numerical and hardware conditions. A shorter formula is not a guaranteed speedup: a gated FFN's extra elementwise product, matrix shapes, alignment, fusion and memory traffic all matter. Measure training and inference separately, report batch/length/dtype/hardware, and keep mathematical counts separate from wall-clock results."}</Prose>

<H3>{"How a block can be distributed"}</H3>

<Prose>{"There are two useful levels of partitioning. "}<strong>{"Pipeline parallelism"}</strong>{" places groups of blocks on different devices and passes activations between them. "}<strong>{"Tensor parallelism"}</strong>{" splits a large map within a block. In our row-vector convention, split "}<code>{"W_up"}</code>{" by its output columns, so each device computes a subset of hidden features; split the corresponding rows of "}<code>{"W_down"}</code>{", so each device forms a partial output. Summing those partial outputs recovers the full down projection. That sum requires communication. A gated FFN must keep matching gate and up coordinates together."}</Prose>

<BlockSystemsFigure />

<Prose>{"Partitioning positions is another choice, with attention communication across position partitions. The later Ring Attention/Sequence Parallelism lesson develops that mechanism. None of these mathematical partitions specifies a universal device count or speed; workload, communication, scheduling and available memory determine the useful implementation."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"10. Practice: change the problem before checking the answer"}</H2>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Follow a new four-feature vector"}</H3>

<Prose>{"Use "}<code>{"x=[0,2,4,6]"}</code>{", identity affine parameters and epsilon "}<code>{"10⁻⁵"}</code>{". Calculate LayerNorm and RMSNorm approximately. Then add three to all entries. Which output remains unchanged? Explain why an output's RMS near one does not make its L2 norm one."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The centered vector is "}<code>{"[-3,-1,1,3]"}</code>{". Compute the mean squared deviation and the mean squared raw value separately."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The mean is 3 and variance is 5. LayerNorm is approximately "}<code>{"[-1.341639,-.447213,.447213,1.341639]"}</code>{". The raw mean square is 14, so RMSNorm is approximately "}<code>{"[0,.534522,1.069045,1.603567]"}</code>{". After the offset, LayerNorm is unchanged. RMSNorm uses "}<code>{"[3,5,7,9]"}</code>{" and mean square 41, giving approximately "}<code>{"[.468521,.780869,1.093216,1.405564]"}</code>{". For four coordinates, "}<code>{"L2=√4×RMS=2×RMS"}</code>{"; an RMS of one corresponds to L2 two. Affine parameters and epsilon affect the exact values as described in §3."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Build a new FFN response"}</H3>

<Prose>{"Use the matrices from §4, ReLU and no biases, but input "}<code>{"[2,-1,0,1]"}</code>{". Find the three preactivations, hidden responses and four-feature update. Change only the third row of "}<code>{"W_down"}</code>{" from "}<code>{"[0,0,.5,-.5]"}</code>{" to "}<code>{"[0,0,1,0]"}</code>{". Which output coordinates change? Would editing a different token's direct FFN input change this update?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The three questions remain "}<code>{"x₁−x₃"}</code>{", "}<code>{"x₂−x₄"}</code>{" and "}<code>{"x₁+x₂"}</code>{". The third hidden response multiplies the third write direction."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The preactivations are "}<code>{"[2,-2,1]"}</code>{"; ReLU gives "}<code>{"[2,0,1]"}</code>{". The original update is "}<code>{"[2,0,.5,-.5]"}</code>{". Changing the third write direction gives "}<code>{"[2,0,1,0]"}</code>{", so coordinates 3 and 4 each increase by .5. The first two coordinates do not change. A different direct FFN input row cannot affect this row because the function is shared but positionwise. An earlier attention edit is a different intervention because it can change this row's contextual input."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Diagnose a result that cannot answer the question"}</H3>

<Prose>{"A colleague compares two 20-block networks. The post-norm network ends in "}<code>{"LayerNorm(16)"}</code>{" with all gains one and all offsets zero, and they differentiate "}<code>{"output.sum()"}</code>{" with respect to the input. They obtain nearly zero and conclude the network cannot train. Identify the exact problem, propose a meaningful replacement diagnostic, and name one limitation of that replacement."}</Prose>

<details><summary>Hint</summary>

<Prose>{"What is the sum across the features of each final normalized row? Does the number of preceding blocks matter?"}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The final centered features sum to zero, so the chosen scalar is constant with respect to upstream inputs. The same result occurs with a single default-affine LayerNorm. A fixed nonconstant projection of the normalized output, or a classification cross-entropy through a declared readout and targets, can measure a useful sensitivity. Match the inputs, initial tensors, readout convention and reduction when comparing wirings. One projection measures one vector-Jacobian product; it is neither the whole singular-value spectrum nor proof of optimization/generalization after training. Final affine parameter gradients also need to be distinguished from upstream input gradients."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Preserve a tagged movement while changing its storage"}</H3>

<Prose>{"A classifier uses point/time records "}<code>{"(x,y,s)"}</code>{", shared pointwise stems, bidirectional attention, per-position normalization/FFNs and mean pooling. You sort all records by x coordinate, carrying each record's original time tag with it. Should the prediction change in exact arithmetic? What if you replace the carried time tags with newly increasing tags after sorting? What must a padding fix do across two blocks and pooling?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate a permutation of complete records from changing the relationship between location and time. Follow padding after the first block as well as before it."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Sorting complete tagged records only permutes the inputs. The shared stem, norms and FFNs are equivariant, and attention without an additional order-specific mask or bias is equivariant. Mean pooling removes the permutation, so logits are unchanged in exact arithmetic. Reassigning new time tags changes the data; there is no such guarantee. Padded keys must be blocked at both attention layers, and padded query rows must be excluded from mean pooling. Preserve the original valid records' time tags instead of recomputing their spacing over the padded length. A tiny floating-point change under a legal permutation is different from a large model change caused by incorrect tagging or masking."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Count a different block"}</H3>

<Prose>{"A biased standard-attention block has "}<code>{"d=96"}</code>{", FFN width "}<code>{"f=240"}</code>{", and two affine LayerNorms. Count its parameters. Then replace only the FFN with a bias-free SwiGLU FFN having the same dominant FFN weight count; find the gated width and the resulting complete block count."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Keep the attention weights/biases and both normalizers. Match "}<code>{"2df"}</code>{" to "}<code>{"3dg"}</code>{" before deciding what happens to the removed FFN biases."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Attention contains "}<code>{"4×96²+4×96=37,248"}</code>{" parameters. The original FFN has "}<code>{"2×96×240+240+96=46,416"}</code>{". Two LayerNorms have "}<code>{"4×96=384"}</code>{". Total 84,048. The gated width is "}<code>{"g=2×240/3=160"}</code>{"; its bias-free FFN has 46,080 weights. The new total is 83,712, which is 336 lower because the original FFN's240+96 biases are gone. Matching leading matrix counts is not the same as matching every scalar parameter."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. A different copying sequence"}</H3>

<Prose>{"An exercise uses "}<code>{"[BOS, six independent uniform symbols from eight choices, SEP, the same six symbols, EOS]"}</code>{". It trains on fresh samples and scores all next-token targets. Calculate the best possible expected average loss contribution from the unpredictable part. How does that contribution change if only repeated-symbol targets are scored?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"There are 15 tokens and 14 targets. Count the six first-occurrence predictions, then use the entropy of a uniform eight-way choice."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The unavoidable contribution is "}<code>{"(6/14)ln8≈.891189"}</code>{" nats per scored target. A model can in principle make the predictable targets' loss arbitrarily small, approaching that floor overall. If the loss scores only the six repeated-symbol targets, they are predictable from the observed source and this particular entropy floor becomes zero. That does not guarantee a finite trained model reaches it; it changes what the objective is measuring."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Can a normalized branch still accumulate a large stream?"}</H3>

<Prose>{"Across examples, let "}<code>{"z"}</code>{" have mean zero and variance one. Start from zero and add exactly the same update "}<code>{"z"}</code>{" eight times. Compare the final variance with adding eight independent copies of "}<code>{"z"}</code>{". Explain which assumption is required for saying accumulated variance grows linearly."}</Prose>

<details><summary>Hint</summary>

<Prose>{"One state is "}<code>{"8z"}</code>{". The other is a sum whose cross-covariances vanish."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The repeated update gives variance 64. The independent zero-mean sum gives variance 8. Independence is sufficient; zero cross-covariances with controlled per-update variance are enough for the variance addition itself. Normalizing each branch input does not establish those covariance assumptions for learned updates. The result concerns variance across examples, not the across-feature RMS of one vector."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Explain a parallel branch to someone reading code"}</H3>

<Prose>{"Compare "}<code>{"z=x+A(N(x)); y=z+F(N(z))"}</code>{" with "}<code>{"y=x+A(N(x))+F(N(x))"}</code>{". Set "}<code>{"A(v)=a"}</code>{", a nonzero constant update, and let "}<code>{"F"}</code>{" be nonlinear. Is the second an algebraic rearrangement of the first? Explain what information reaches the FFN and why a pretrained checkpoint cannot generally be switched without changing outputs."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Look at the FFN's argument, not only the three terms visible near the final addition."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The sequential version returns "}<code>{"x+a+F(N(x+a))"}</code>{"; the parallel version returns "}<code>{"x+a+F(N(x))"}</code>{". These are generally different because normalization and "}<code>{"F"}</code>{" can respond to the changed input. The sequential FFN consumes this block's attention update; the parallel FFN consumes the original normalized state. They agree under particular nulls, such as zero attention update, but not as a general identity. A switch changes the function even if parameter dimensions and names are compatible."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"Where this leads, and another way to learn"}</H2>

<Prose>{"You can now trace how a block communicates across positions, processes features locally, preserves or normalizes its carried state, and becomes a task model. The next lesson in the module is "}<a href={"/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals"}>{"Positional Encodings: Sinusoidal, Learned, RoPE and ALiBi"}</a>{". It asks how order and distance should enter the representation or attention calculation; our simple movement time tag is a starting point, not the end of that design problem."}</Prose>

<Prose>{"For a different presentation or deeper source reading:"}</Prose>

<ul><li>{""}<a href={"https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html"}>{"D2L 1.0.3, Transformer Architecture"}</a>{" connects a post-norm block to an encoder–decoder translation model. Its FFN, add/norm, encoder/decoder and training sections are useful next reading if you want the larger sequence-to-sequence assembly. The examples use the book's helper framework and a different training task; their scores are not comparisons with our movement study."}</li><li>{""}<a href={"https://www.3blue1brown.com/lessons/mlp/"}>{"3Blue1Brown, How might LLMs store facts?"}</a>{" offers an accompanying video and illustrated creator notes on FFN feature detection and update directions. Read the assumptions around its deliberately simplified feature circuit. The current notes also correct a near-orthogonality demonstration from the video; use the corrected notes, and do not treat the numerical capacity illustration as a measured fact about a particular language model. The matrix interpretation is the useful bridge here."}</li><li>{""}<a href={"https://arxiv.org/pdf/1706.03762"}>{"Attention Is All You Need, §3"}</a>{" defines the original post-norm encoder–decoder model and its ReLU FFN. Read it to distinguish historical choices from the family of later variants."}</li><li>{""}<a href={"https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf"}>{"On Layer Normalization in the Transformer Architecture"}</a>{" is the deeper source for placement, initialization gradients and warmup. Its stated theoretical assumptions are essential to interpreting the conclusions."}</li><li>{""}<a href={"https://arxiv.org/pdf/1607.06450"}>{"Layer Normalization"}</a>{" and "}<a href={"https://arxiv.org/pdf/1910.07467"}>{"RMSNorm"}</a>{" explain the distinct normalization mechanisms. The latter's efficiency measurements belong to its evaluated implementations, not a universal current speed ratio."}</li><li>{""}<a href={"https://arxiv.org/html/2002.05202v1"}>{"GLU Variants Improve Transformer"}</a>{" provides the gated FFN equations and parameter-matched experimental setup. "}<a href={"https://aclanthology.org/2021.emnlp-main.446.pdf"}>{"Transformer Feed-Forward Layers Are Key-Value Memories"}</a>{" develops the persistent-memory interpretation and investigates trained examples."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html"}>{"PyTorch's reference encoder layer"}</a>{" is the API matched in §5. The complete topic program and retained inputs are linked in §6 for an offline hands-on route."}</li></ul></section>
  </div>,
};
