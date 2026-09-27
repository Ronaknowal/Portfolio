// Generated from the complete prepared manuscript by scripts/generate-latent-attention-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements';
import { LatentStorageFigure, LatentScoreFigure, LatentPathsLab, LatentRotationLab, LatentBudgetLab, LatentRankLab, LatentSoftmaxFigure, LatentForecastLab, LatentProgram } from '../../components/lesson-labs/LatentAttentionLabs';
import '../../components/lesson-labs/neural-lesson-neutral.css';
const lesson = { title: 'Multi-Head Latent Attention (MLA)', readTime: '~65 min read + 90 min practice', content: () => <div className="mla-lesson neural-lesson neural-lesson-neutral">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit latent vectors/projections, rotation, retained rank, payload dimensions and supported frozen-model prefixes. Show expanded and absorbed paths, commutation residuals, singular-direction effects, bytes/arithmetic and resulting outputs together. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose compression by the function and input directions it preserves; distinguish a low parameter error from low task error."}</Prose>

<Prose>{"Suppose several people need different summaries of the same record. We could store every summary. Or we could store a compact description from which each person's summary can be computed. The second option is useful only if the description retains what those people actually need."}</Prose>

<Prose>{"Multi-head latent attention, or "}<strong>{"MLA"}</strong>{", applies this idea to a Transformer's memory. Each past position keeps a learned compact vector, plus the positional information required by its attention design. Different query heads read that shared representation through different learned maps. An algebraic rearrangement lets them do so without rebuilding every past head's keys and values for each new query."}</Prose>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals"}>{"previous GQA/MQA lesson"}</a>{" reduced the number of distinct stored heads. Here we change the coordinates of the stored information itself. We will carefully distinguish "}<strong>{"an exact rearrangement of one MLA model"}</strong>{" from "}<strong>{"a lossy change to what that model can represent"}</strong>{"."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" follow §§1–6 and exercises 1–5. You will understand the latent representation, compute both equivalent attention paths, preserve the positional and scaling contracts, and interpret a real cache-compression experiment. Section 7 develops rank, conversion, differentiation and deployment connections; exercises 6–9 extend that reasoning. The core drawings and investigations appear beside the mechanisms they explain."}</Prose>

<H2>{"1. What should a decoder remember?"}</H2>

<H3>{"Refresh the dependency before changing the representation"}</H3>

<Prose>{"In "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention"}</a>{", a query compares with keys. Softmax turns the legal scores into weights, and a weighted sum of values produces a head's output. A causal query at position t can read positions up to t."}</Prose>

<Prose>{"For a fixed causal model, a new future input does not change earlier hidden states. We can therefore retain their keys and values instead of projecting them again at every generation step. This is the KV cache. It is separate at each layer. New queries are transient readers; past keys and values are reusable information."}</Prose>

<Prose>{"Ordinary multi-head attention stores separate K/V representations for each head. GQA shares one K/V representation within each group; MQA shares one across all query heads. Their outputs still differ because the queries differ. MLA asks another question: "}<strong>{"can all those head-specific representations be generated from a smaller common vector?"}</strong>{""}</Prose>

<H3>{"A latent is a learned coordinate vector"}</H3>

<Prose>{"Let an attention input have D coordinates. A learned down-projection maps it to a smaller vector c with "}<InlineMath>{"d_c"}</InlineMath>{" coordinates. “Latent” means this is an internal representation. Its coordinates need not have names such as direction, subject or verb, and their values are not probabilities."}</Prose>

<Prose>{"Each head has its own key and value up-projection. These maps can turn the same c into different keys and values. Sharing c therefore does not mean that the reconstructed heads are identical. The distinction from GQA is concrete: GQA shares a head representation directly; MLA shares coordinates used to produce head representations."}</Prose>

<LatentStorageFigure />

<Prose>{"We are describing the dense, decoupled-rotary design introduced in DeepSeek-V2 and retained in the V3 architecture. Later variants can change the positional construction, sparsity or cache layout. The stable principle is to state exactly what is stored and how the model reads it; a model-family label alone does not determine a cache format."}</Prose>

<H3>{"Three claims that must stay separate"}</H3>

<Prose>{"First, restricting information to a latent representation is an architectural choice. It may affect learnability and quality. Second, once an MLA model is defined, its reconstructed and absorbed computations can be mathematically identical. Third, whether either computation is faster depends on the workload and implementation."}</Prose>

<Prose>{"A smaller cache does not prove equal quality or the same factor of speedup. Conversely, a lossy rank reduction does not invalidate the exact algebra of the original model. Much of MLA becomes easier once these three questions are separated."}</Prose>

<H2>{"2. Build the representations and keep their shapes visible"}</H2>

<H3>{"Name the dimensions"}</H3>

<Prose>{"We use column vectors in the equations and explicit row/batch axes in code."}</Prose>

<NeuralTable caption={"Name the dimensions"} headers={[<>{"Symbol"}</>,<>{"Meaning"}</>]} rows={[[<>{"D"}</>,<>{"Attention input and output row width"}</>],[<>{"H"}</>,<>{"Number of query/output heads"}</>],[<>{""}<InlineMath>{"d_c"}</InlineMath>{""}</>,<>{"Width of the cached content latent"}</>],[<>{""}<InlineMath>{"d_q"}</InlineMath>{""}</>,<>{"Width of the optional query latent"}</>],[<>{""}<InlineMath>{"d_k"}</InlineMath>{""}</>,<>{"One head's content-query/content-key width"}</>],[<>{""}<InlineMath>{"d_r"}</InlineMath>{""}</>,<>{"Rotary-query/shared-rotary-key width, an even number"}</>],[<>{""}<InlineMath>{"d_v"}</InlineMath>{""}</>,<>{"One head's value/output width"}</>],[<>{"L, T"}</>,<>{"Number of legal/stored memory positions and number of new query positions"}</>]]} />

<Prose>{"The important comparison is often "}<InlineMath>{"d_c"}</InlineMath>{" versus "}<strong>{"all heads' stored coordinates"}</strong>{", not versus one head's width. In a published V2-style configuration, "}<InlineMath>{"d_c=512"}</InlineMath>{" while "}<InlineMath>{"d_k=128"}</InlineMath>{". The latent is four times wider than one content head, yet much narrower than 128 heads together. Calling every operation on the latent “smaller” would already be misleading."}</Prose>

<H3>{"Content keys and values come from the same latent"}</H3>

<Prose>{"For attention input "}<InlineMath>{"h_s\\in\\mathbb R^D"}</InlineMath>{" at memory position s, begin with"}</Prose>

<div className="neural-equation"><MathBlock>{"c_s=W^{DKV}h_s,\\qquad\nk^C_{s,i}=U_{K,i}c_s,\\qquad\nv_{s,i}=U_{V,i}c_s."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"W^{DKV}"}</InlineMath>{" has shape "}<InlineMath>{"[d_c,D]"}</InlineMath>{", "}<InlineMath>{"U_{K,i}"}</InlineMath>{" has shape "}<InlineMath>{"[d_k,d_c]"}</InlineMath>{" and "}<InlineMath>{"U_{V,i}"}</InlineMath>{" has shape "}<InlineMath>{"[d_v,d_c]"}</InlineMath>{". The superscript C identifies the content-key branch."}</Prose>

<Prose>{"The actual implementation can normalize the down-projected vector before these up-projections. For example, a learned RMSNorm has the form"}</Prose>

<div className="neural-equation"><MathBlock>{"c=\\gamma\\odot\\frac{z}{\\sqrt{\\frac1{d_c}\\sum_a z_a^2+\\epsilon}},\n\\qquad z=W^{DKV}h."}</MathBlock></div>

<Prose>{"It rescales by a root-mean-square statistic without subtracting the mean. In that case, "}<strong>{"c in our subsequent equations is the normalized vector"}</strong>{", and that is the vector to cache. The original report's practical settings and the "}<a href={"https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py"}>{"official V3 inference implementation"}</a>{" use RMSNorm on the compressed latents. Replacing it with LayerNorm changes the model; omitting it is not automatically guaranteed to cause NaNs, but it changes a checkpoint's defined computation."}</Prose>

<Prose>{"This distinction also matters for algebra. We can move linear up-projections across sums of c. We cannot treat RMSNorm as a fixed matrix and move it across an arbitrary down-projection or weighted sum."}</Prose>

<H3>{"Queries have their own path"}</H3>

<Prose>{"A query can be produced directly from the current input, or through a separate latent:"}</Prose>

<div className="neural-equation"><MathBlock>{"c^Q_t=\\operatorname{RMSNorm}(W^{DQ}h_t),\\qquad\nq^C_{t,i}=U_{Q,i}c^Q_t."}</MathBlock></div>

<Prose>{"The query latent is not the KV cache. It is computed for current queries; its factorization changes parameters, intermediate activations and computation. Whether it reduces peak training memory depends on which expanded tensors are saved or recomputed. A factorized projection does not magically remove every larger activation from autograd."}</Prose>

<Prose>{"The "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json"}>{"published V2 configuration"}</a>{" uses "}<InlineMath>{"d_q=1536"}</InlineMath>{" and "}<InlineMath>{"d_c=512"}</InlineMath>{": the former is three times the latter. They are separate choices, not a universal ratio. Some implementations allow a direct query path when no query compression is desired."}</Prose>

<H2>{"3. Preserve position without rebuilding every key"}</H2>

<H3>{"Why a usual rotary key obstructs a fixed absorption"}</H3>

<Prose>{"Recall "}<a href={"/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals"}>{"Positional Encodings"}</a>{". RoPE rotates coordinate pairs according to their logical position. Write that rotation as "}<InlineMath>{"R_s"}</InlineMath>{". A normally rotated reconstructed key would be "}<InlineMath>{"R_sU_Kc_s"}</InlineMath>{", giving score contribution"}</Prose>

<div className="neural-equation"><MathBlock>{"(R_tq)^T(R_sU_Kc_s)\n=q^T R_t^T R_s U_Kc_s."}</MathBlock></div>

<Prose>{"To compare a single effective query with every c, we would need to remove the key-position dependence from the factor beside c. In general, "}<InlineMath>{"R_t^TR_s"}</InlineMath>{" depends on s. There is no one key-position-independent query transform that absorbs all these rotations through an arbitrary "}<InlineMath>{"U_K"}</InlineMath>{"."}</Prose>

<Prose>{"A small example makes the obstruction visible. Let"}</Prose>

<div className="neural-equation"><MathBlock>{"U_K=\\begin{bmatrix}2&0\\\\0&1\\end{bmatrix},\\qquad\nR=\\begin{bmatrix}0&-1\\\\1&0\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Stretching the x coordinate and then rotating does not equal rotating and then stretching it. Their difference is"}</Prose>

<div className="neural-equation"><MathBlock>{"RU_K-U_KR=\\begin{bmatrix}0&1\\\\1&0\\end{bmatrix}."}</MathBlock></div>

<Prose>{"With current query "}<InlineMath>{"[1,0]^T"}</InlineMath>{", a key with no rotation requires effective latent query "}<InlineMath>{"[2,0]^T"}</InlineMath>{"; a key with this quarter-turn rotation requires "}<InlineMath>{"[0,-1]^T"}</InlineMath>{". The transformation would depend on which past key we are reading."}</Prose>

<Prose>{"This is a statement about the general fixed-matrix rearrangement. Special structured maps or different positional architectures can have other identities. Applying RoPE to reconstructed keys is still a valid attention computation; it simply loses this particular easy absorption unless additional structure is supplied."}</Prose>

<LatentRotationLab />

<H3>{"A separate rotary branch"}</H3>

<Prose>{"The V2/V3-style solution keeps content keys unrotated and adds a small rotary branch:"}</Prose>

<div className="neural-equation"><MathBlock>{"k^R_s=R_s W^{KR}h_s,\\qquad\nq^R_{t,i}=R_t U_{QR,i}c^Q_t."}</MathBlock></div>

<Prose>{"There is "}<strong>{"one shared rotary key per memory position"}</strong>{", while query heads have distinct rotary queries. Cache "}<InlineMath>{"c_s"}</InlineMath>{" and "}<InlineMath>{"k^R_s"}</InlineMath>{". The score for head i is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\ell_{t,s,i}=\n\\frac{(q^C_{t,i})^T U_{K,i}c_s+(q^R_{t,i})^Tk^R_s}\n{\\sqrt{d_k+d_r}}."}</MathBlock></div>

<Prose>{"Apply the causal/padding mask, then softmax over the legal memory positions. The resulting weights mix "}<InlineMath>{"U_{V,i}c_s"}</InlineMath>{"."}</Prose>

<Prose>{"The rotary branch is not a stored integer position or a content-free tag: it is a projected input vector rotated using a position. Likewise, “NoPE/content” means that this particular branch receives no direct rotary transform. Its input hidden state may already contain positional information from earlier layers. Do not interpret the name as proof that every latent coordinate contains only semantics and no information about order."}</Prose>

<Prose>{"Decoupling preserves a specified positional mechanism. It does not by itself guarantee good behavior at arbitrary unseen lengths. Context extension still depends on frequency/scaling choices, training and evaluation, as the preceding positional lesson explained. A causal mask itself also carries an ordering constraint; removing an explicit positional branch does not justify the blanket statement that a causal network has no way to distinguish order."}</Prose>

<LatentScoreFigure />

<H2>{"4. Read the latent directly: the exact rearrangement"}</H2>

<H3>{"Absorb the key up-projection into the query"}</H3>

<Prose>{"For any fixed head and query,"}</Prose>

<div className="neural-equation"><MathBlock>{"(q^C)^T U_K c=(U_K^Tq^C)^Tc."}</MathBlock></div>

<Prose>{"Define "}<InlineMath>{"\\widetilde q=U_K^Tq^C"}</InlineMath>{", a vector with "}<InlineMath>{"d_c"}</InlineMath>{" coordinates. The content score can now be computed directly against the cached c. No historical content key needs to be expanded for that comparison. Keep the separate rotary dot product unchanged."}</Prose>

<Prose>{"This is ordinary associativity of matrix multiplication. It does not discard singular directions or approximate a probability. In a row-vector implementation, the same operation is "}<code>{"effective_query = content_query @ key_up"}</code>{", with the up-projection stored as output-by-input."}</Prose>

<H3>{"Mix latents before expanding values"}</H3>

<Prose>{"Once the weights "}<InlineMath>{"a_s"}</InlineMath>{" are known,"}</Prose>

<div className="neural-equation"><MathBlock>{"o_i=\\sum_s a_{s,i}U_{V,i}c_s\n=U_{V,i}\\left(\\sum_s a_{s,i}c_s\\right)."}</MathBlock></div>

<Prose>{"Compute a weighted latent "}<InlineMath>{"z_i=\\sum_s a_{s,i}c_s"}</InlineMath>{", then expand it once for this query/head. Each head still has its own weights and therefore its own z. The shared cache is not one shared output."}</Prose>

<Prose>{"If "}<InlineMath>{"W_{O,i}"}</InlineMath>{" is the output-map slice for head i, the residual contribution is"}</Prose>

<div className="neural-equation"><MathBlock>{"y=\\sum_i W_{O,i}U_{V,i}z_i."}</MathBlock></div>

<Prose>{"For fixed weights, the product "}<InlineMath>{"W_{O,i}U_{V,i}"}</InlineMath>{" can be precomputed. It can also be left factorized. A larger precomputed matrix may cost more storage and arithmetic than two smaller multiplications, so “can absorb” is not the same as “every implementation must premerge at export.” The official reference computes useful contractions at runtime."}</Prose>

<H3>{"Keep the original score scale"}</H3>

<Prose>{"The effective query has width "}<InlineMath>{"d_c"}</InlineMath>{", but its score is the original content dot product expressed differently. The divisor remains "}<InlineMath>{"\\sqrt{d_k+d_r}"}</InlineMath>{", or the checkpoint's explicitly modified scale."}</Prose>

<Prose>{"A generic attention call given concatenated latent and rotary features may default to "}<InlineMath>{"1/\\sqrt{d_c+d_r}"}</InlineMath>{". When "}<InlineMath>{"d_c\\ne d_k"}</InlineMath>{", that changes the logits and their concentration. It can produce valid shapes and plausible outputs while implementing the wrong model. For our real example, the intended divisor is "}<InlineMath>{"\\sqrt6"}</InlineMath>{"; the accidental latent-width divisor is "}<InlineMath>{"\\sqrt{10}"}</InlineMath>{"."}</Prose>

<LatentPathsLab />

<H3>{"A complete hand example"}</H3>

<Prose>{"Use two heads, "}<InlineMath>{"d_c=d_k=d_v=d_r=2"}</InlineMath>{", one query at position 2 and memory positions 0,1,2. These are chosen arithmetic inputs, not trained activations. For easy hand rotations use a quarter turn per position; the trained study later uses ordinary RoPE instead."}</Prose>

<NeuralTable caption={"A complete hand example"} headers={[<>{"Memory position"}</>,<>{"Cached c"}</>,<>{"Raw rotary key"}</>,<>{"Rotated rotary key"}</>]} rows={[[<>{"0"}</>,<>{"[1,0]"}</>,<>{"[1,0]"}</>,<>{"[1,0]"}</>],[<>{"1"}</>,<>{"[0,1]"}</>,<>{"[1,0]"}</>,<>{"[0,1]"}</>],[<>{"2"}</>,<>{"[1,1]"}</>,<>{"[1,0]"}</>,<>{"[−1,0]"}</>]]} />

<Prose>{"Content queries are "}<InlineMath>{"q^C_0=[1,0]^T"}</InlineMath>{" and "}<InlineMath>{"q^C_1=[1,1]^T"}</InlineMath>{". Raw rotary queries "}<code>{"[1,0]"}</code>{" and "}<code>{"[0,1]"}</code>{" become "}<code>{"[−1,0]"}</code>{" and "}<code>{"[0,−1]"}</code>{" at position 2. Use"}</Prose>

<div className="neural-equation"><MathBlock>{"U_{K,0}=\\begin{bmatrix}1&0\\\\0&2\\end{bmatrix},\\quad\nU_{K,1}=\\begin{bmatrix}1&1\\\\1&-1\\end{bmatrix},\\quad\nU_{V,0}=I,\\quad\nU_{V,1}=\\begin{bmatrix}2&0\\\\1&-1\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Head 0's effective query is "}<code>{"[1,0]"}</code>{". Its content scores are "}<code>{"[1,0,1]"}</code>{"; rotary scores are "}<code>{"[−1,0,1]"}</code>{". Add and divide by 2, obtaining logits "}<code>{"[0,0,1]"}</code>{". The weights are"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{[1,1,e]}{2+e}\\approx[0.211942,0.211942,0.576117]."}</MathBlock></div>

<Prose>{"The weighted latent is "}<code>{"[0.788058,0.788058]"}</code>{". Because "}<InlineMath>{"U_{V,0}=I"}</InlineMath>{", that is also head 0's value output."}</Prose>

<Prose>{"Head 1's effective content query is "}<code>{"[2,0]"}</code>{", giving content scores "}<code>{"[2,0,2]"}</code>{". The rotary scores are "}<code>{"[0,−1,0]"}</code>{", so scaled logits are "}<code>{"[1,−0.5,1]"}</code>{". Its weights are approximately "}<code>{"[0.449816,0.100368,0.449816]"}</code>{". Its weighted latent is "}<code>{"[0.899632,0.550184]"}</code>{", which its value map transforms into "}<code>{"[1.799265,0.349449]"}</code>{"."}</Prose>

<Prose>{"The two heads read the same latent records but have different weights and different value outputs. To complete the example, combine them with"}</Prose>

<div className="neural-equation"><MathBlock>{"W_O=\\begin{bmatrix}1&0&0.5&0\\\\0&1&0&0.5\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The attention contribution is approximately "}<code>{"[1.687691,0.962783]"}</code>{". This is an attention output vector; a full Transformer block still applies its residual and feedforward computation before a task prediction."}</Prose>

<H3>{"Changing coordinates is different from losing coordinates"}</H3>

<Prose>{"Let S be any invertible change of latent basis. Store "}<InlineMath>{"c'=Sc"}</InlineMath>{", and replace each up-projection U by "}<InlineMath>{"U'=US^{-1}"}</InlineMath>{". Then "}<InlineMath>{"U'c'=Uc"}</InlineMath>{", so every reconstructed key/value and attention result is unchanged. Our hand example checks "}<InlineMath>{"S=\\operatorname{diag}(2,0.5)"}</InlineMath>{" exactly."}</Prose>

<Prose>{"This is why a latent coordinate does not automatically have a unique semantic identity. Its scale and basis can change with compensating maps. The argument applies to the defined cached vector after any normalization; it does not say that an arbitrary basis change commutes through RMSNorm."}</Prose>

<Prose>{"Discarding coordinates is different: a noninvertible projection has no inverse that recovers all possible c. The actual effect depends on what was discarded and what the model needed. Section 6 measures that distinction on real observations."}</Prose>

<H2>{"5. Implement it, then count the right things"}</H2>

<H3>{"A transparent executable reference"}</H3>

<Prose>{"This complete NumPy program implements both paths for the hand example. K/V are reconstructed only in the expanded branch. The rotary vectors are already rotated, making the matrix reassociation visible without hiding it inside a larger model."}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\n\ndef mla_read(query, latent, key_up, value_up, query_rope, key_rope,\n             scale, expanded=False):\n    # query [H,dk], latent [L,dc], up-maps [H,output_width,dc].\n    if expanded:\n        keys = np.einsum(\"lc,hpc->hlp\", latent, key_up)\n        values = np.einsum(\"lc,hvc->hlv\", latent, value_up)\n        content = np.einsum(\"hp,hlp->hl\", query, keys)\n    else:\n        effective_query = np.einsum(\"hp,hpc->hc\", query, key_up)\n        content = effective_query @ latent.T\n    logits = (content + query_rope @ key_rope.T) * scale\n    # All three keys are legal for this final-position hand example.\n    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))\n    weights /= weights.sum(axis=-1, keepdims=True)\n    if expanded:\n        output = np.einsum(\"hl,hlv->hv\", weights, values)\n    else:\n        mixed_latent = weights @ latent\n        output = np.einsum(\"hc,hvc->hv\", mixed_latent, value_up)\n    return output, weights\n\n\nq = np.array([[1., 0.], [1., 1.]])\nc = np.array([[1., 0.], [0., 1.], [1., 1.]])\nuk = np.array([[[1., 0.], [0., 2.]], [[1., 1.], [1., -1.]]])\nuv = np.array([np.eye(2), [[2., 0.], [1., -1.]]])\nqr = np.array([[-1., 0.], [0., -1.]])\nkr = np.array([[1., 0.], [0., 1.], [-1., 0.]])\nwo = np.array([[1., 0., .5, 0.], [0., 1., 0., .5]])\nexpanded, first_weights = mla_read(q, c, uk, uv, qr, kr, .5, True)\nabsorbed, second_weights = mla_read(q, c, uk, uv, qr, kr, .5, False)\nprint(\"head outputs:\", np.round(absorbed, 6))\nprint(\"final output:\", np.round(wo @ absorbed.reshape(-1), 6))\nprint(\"same output:\", np.allclose(expanded, absorbed, atol=1e-12))\nprint(\"same weights:\", np.allclose(first_weights, second_weights, atol=1e-12))"}</CodeBlock>

<Prose>{"It produces the two head outputs and final output calculated above, and both checks print "}<code>{"True"}</code>{". The "}<a href={"/learn-assets/multi-head-latent-attention-mla/mechanism-calculations.py"}>{"complete independent fixture program"}</a>{" additionally supplies editable latent/map inputs, rotations, a consistent basis change, broken-position and nonlinear-map contrasts, and exact byte/work counts. The full neural program in §6 adds normalized learned projections, actual causal masks, incremental cache state and task predictions."}</Prose>

<LatentProgram file="mechanism-calculations.py" />

<H3>{"What exactly is in the compact cache?"}</H3>

<Prose>{"For B requests, N uniform layers, L occupied positions and s bytes per stored number, the unquantized compact payload is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\boxed{\\operatorname{bytes}=BNL(d_c+d_r)s.}"}</MathBlock></div>

<Prose>{"There is one c and one shared rotated key per position/layer. Do not multiply c by H or count it twice because it participates in both the key and value calculations. Queries, score tiles, weighted latents and output vectors are transient work, not additional copies of the persistent history."}</Prose>

<Prose>{"At fixed "}<InlineMath>{"d_c,d_r"}</InlineMath>{", increasing H leaves this formula unchanged. It still increases the number of comparisons and head outputs, as well as relevant parameters and transient tensors. This is a storage property, not free additional heads."}</Prose>

<Prose>{"Different baselines require different denominators. The following are "}<strong>{"calculated representations"}</strong>{", all using H=128, content width 128, value width 128, rotary width 64 and latent width 512 where applicable:"}</Prose>

<NeuralTable caption={"What exactly is in the compact cache?"} headers={[<>{"Representation"}</>,<>{"Stored numbers per token/layer"}</>,<>{"Meaning"}</>]} rows={[[<>{"Plain MHA, 128-wide keys and values"}</>,<>{"32,768"}</>,<>{"RoPE can rotate within its existing key width; this is a different parameterization"}</>],[<>{"GQA with 8 such KV heads"}</>,<>{"2,048"}</>,<>{"Shared heads with those stated widths"}</>],[<>{"MQA with one such KV head"}</>,<>{"256"}</>,<>{"Smallest payload in this particular table"}</>],[<>{"Same MLA function, literal expanded K/V cache"}</>,<>{"40,960"}</>,<>{"128 heads each store a 192-wide key and 128-wide value"}</>],[<>{"Same MLA function, expanded content plus one shared rotary key"}</>,<>{"32,832"}</>,<>{"Avoids duplicating the 64-wide rotary key across heads"}</>],[<>{"Compact MLA"}</>,<>{"576"}</>,<>{"One 512-wide latent plus one 64-wide rotary key"}</>]]} />

<Prose>{"The compact payload is about 56.89 times smaller than the first plain-MHA count and 71.11 times smaller than the literal expanded representation of the "}<strong>{"same MLA function"}</strong>{". These answer different comparisons. It is 2.25 times the MQA payload in this table, not smaller than every alternative. A memory table alone contains no quality ranking."}</Prose>

<Prose>{"For 60 layers, 32,768 positions, one request and two-byte numbers, compact MLA needs 2,264,924,160 bytes, or 2.109375 GiB. The plain-MHA row needs 120 GiB; GQA-8 needs 7.5 GiB. A GiB is "}<InlineMath>{"2^{30}"}</InlineMath>{" bytes. These are payload calculations, not measured allocator memory or statements that a whole model fits on a device."}</Prose>

<Prose>{"The V2 report's headline 93.3% cache reduction compares "}<strong>{"different complete models"}</strong>{", DeepSeek-V2 and the earlier DeepSeek 67B. It is not the rounded value of our 56.89-fold, same-head-width calculation. Keep the stated comparison attached to any percentage. "}<a href={"https://arxiv.org/html/2405.04434v5"}>{"Original report"}</a>{"."}</Prose>

<Prose>{"Actual allocation may include page rounding, reserved slots, position metadata, quantization scales, distributed replicas and cache layouts that materialize expanded heads. The "}<a href={"https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py"}>{"official V3 reference"}</a>{" visibly allocates different buffers for its "}<code>{"naive"}</code>{" and absorbed branches. An architecture supports a compact representation; an implementation must actually store it to realize that payload."}</Prose>

<LatentBudgetLab />

<H3>{"Why less data movement can involve more arithmetic"}</H3>

<Prose>{"At fixed H, T, L and widths, the expanded attention core uses approximately"}</Prose>

<div className="neural-equation"><MathBlock>{"2BHTL(d_k+d_r+d_v)"}</MathBlock></div>

<Prose>{"operations for scores and weighted values, counting a multiply-add as two. The absorbed core uses approximately"}</Prose>

<div className="neural-equation"><MathBlock>{"2BHTL(2d_c+d_r)."}</MathBlock></div>

<Prose>{"It compares with a "}<InlineMath>{"d_c"}</InlineMath>{"-wide latent and also sums a "}<InlineMath>{"d_c"}</InlineMath>{"-wide latent. When "}<InlineMath>{"d_c"}</InlineMath>{" is larger than "}<InlineMath>{"d_k"}</InlineMath>{" and "}<InlineMath>{"d_v"}</InlineMath>{", these core arithmetic terms increase. For the widths in the table, the ratio is "}<InlineMath>{"1088/320=3.4"}</InlineMath>{"."}</Prose>

<Prose>{"For one query and 32,768 memory positions, our formula gives about 2.684 billion expanded-core operations versus 9.127 billion absorbed-core operations. The absorbed representation can nevertheless reduce persistent storage and repeated memory traffic by sharing the latent across heads. Effective hardware reuse, tiling, bandwidth and occupancy determine how these facts translate into time."}</Prose>

<Prose>{"There is another comparison: rebuilding "}<strong>{"all"}</strong>{" historical expanded K/V from a compact cache for every new query costs roughly"}</Prose>

<div className="neural-equation"><MathBlock>{"2BLH d_c(d_k+d_v)"}</MathBlock></div>

<Prose>{"additional operations. Absorption avoids that repeated reconstruction. Instead, it transforms each new query and the resulting weighted latent once per head. In our numerical configuration, full-prefix reconstruction is about "}<InlineMath>{"1.10\\times10^{12}"}</InlineMath>{" operations; each of those two per-query transformations is about 16.78 million. Avoiding reconstruction is useful, but it is not evidence that absorbed dense attention has 57 times fewer operations than an already-expanded-cache attention core."}</Prose>

<Prose>{"Prefill has many known query rows; decode often has one new row per request. A compute-friendly expanded path can be attractive during prefill, with bounded temporary reconstruction, while a compact path can suit decode. "}<a href={"https://docs.vllm.ai/en/v0.20.0/api/vllm/model_executor/layers/attention/mla_attention/"}>{"vLLM's MLA implementation notes"}</a>{" explain both forms and chunked prefill. Use the actual dimensions: "}<InlineMath>{"T/L"}</InlineMath>{" is near 1 for an uncached full prefill and small for one-query decode. Do not reproduce a reversed small/large ratio from a documentation sentence."}</Prose>

<H3>{"Parameter counts are also dimension-dependent"}</H3>

<Prose>{"Ignoring biases and normalization parameters, the factorized design described here has"}</Prose>

<div className="neural-equation"><MathBlock>{"D(d_c+d_q+d_r)\n+Hd_c(d_k+d_v)\n+Hd_q(d_k+d_r)\n+DHd_v"}</MathBlock></div>

<Prose>{"projection parameters. The four terms account for input/down maps, KV up maps, query up maps and the output map. A direct-query design changes the query terms. Learned latent normalization adds its scales."}</Prose>

<Prose>{"There is no universal “only a few percent more than MHA” answer. In particular, the convenient MHA expression "}<InlineMath>{"4D^2"}</InlineMath>{" assumes the usual total head widths equal D. A configuration with "}<InlineMath>{"Hd_k\\ne D"}</InlineMath>{" does not satisfy that assumption. Count the actual maps before comparing architectures or optimizer-state memory."}</Prose>

<H3>{"Use the absorbed representation with an ordinary attention primitive"}</H3>

<Prose>{"The "}<a href={"/learn-assets/multi-head-latent-attention-mla/mla_sdpa_bridge.py"}>{"complete SDPA bridge"}</a>{" supplies a practical tensor API route. Concatenate the absorbed query "}<code>{"q_content @ U_K"}</code>{" and positioned rotary query; concatenate each cached latent and positioned shared rotary key. Use the cached latent as the value. SDPA then returns a latent mixture, which each head's value-up map turns into its output."}</Prose>

<Prose>{"Pass "}<code>{"scale=1/sqrt(P+R)"}</code>{" explicitly. SDPA's default would use the concatenated width C+R, changing the function whenever C differs from P. "}<code>{"enable_gqa=True"}</code>{" makes all query heads read the one-head shared memory. The boolean mask means allowed and uses the logical positions; dropout is zero. The complete program compares direct/API output, all six input/factor gradients and one equal update. Run "}<code>{"python mla_sdpa_bridge.py"}</code>{" with PyTorch. A bounded author probe on Torch 2.14.0 CPU gave output discrepancy 4.44e-16 and largest gradient discrepancy 3.33e-16."}</Prose>

<LatentProgram file="mla_sdpa_bridge.py" />

<Prose>{"This uses the "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html"}>{"PyTorch SDPA contract"}</a>{"; it is not a specialized DeepSeek kernel. Device/backend support and physical allocation for grouped queries with unequal value width need separate measurement. The original tensor path remains a useful fallback. The whole model also has projections, latent normalization, residuals, positions and cache identity; those remain explicit in "}<code>{"LatentForecaster"}</code>{", rather than being inferred from operator agreement."}</Prose>

<Prose>{""}<strong>{"Change the constraint:"}</strong>{" use latent width 7 with P=3 and R=2. Change the latent and both up-projection shapes together. "}<strong>{"Hint:"}</strong>{" absorption changes representation width, not the intended temperature. "}<strong>{"Solution:"}</strong>{" preserve "}<code>{"1/sqrt(5)"}</code>{" in both routes; the equality checks should still pass. Deliberately using "}<code>{"1/sqrt(9)"}</code>{" in only one route creates a shape-valid semantic mismatch."}</Prose>

<H2>{"6. A real model: exact execution and a lossy intervention"}</H2>

<H3>{"Predict the next point of an observed movement"}</H3>

<Prose>{"The task is causal coordinate forecasting using "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI Libras Movement"}</a>{", an openly licensed set of normalized hand-centroid trajectories. Each row contains 45 ordered x/y pairs. At positions 0–43, predict the next coordinate pair at positions 1–44, without reading future points."}</Prose>

<Prose>{"The source has 360 rows, including 30 extra copies of exact trajectories. After checking duplicate labels agree, retain the first occurrence of each distinct trajectory. Reuse the declared seed-73 classwise row split: 220 training, 50 validation and 60 test trajectories. Keep entire trajectories together. Labels determine stratification only; they are not forecast inputs or targets."}</Prose>

<Prose>{"Use fixed "}<InlineMath>{"2x-1"}</InlineMath>{" scaling and convert RMSE back to the original coordinate unit. The source does not provide reliable per-row performer/session IDs, so this is a row-level diagnostic, not a test on a new signer. Its coordinates have already been processed by the dataset creators; the study does not validate an online camera pipeline. "}<a href={"/learn-assets/multi-head-latent-attention-mla/data-provenance.md"}>{"Full source/protocol provenance"}</a>{" accompanies the original offline files."}</Prose>

<Prose>{"Persistence predicts the latest point unchanged. A six-parameter affine map predicts the next two coordinates from the current two, fitted only on training transitions. Smooth trajectories make these meaningful baselines."}</Prose>

<H3>{"The complete small architecture and declared training"}</H3>

<Prose>{"Our model has a 2-to-24 stem and one pre-normalized Transformer block. There are four heads with content width 4, rotary width 2 and value width 4. The KV latent has eight coordinates; the query latent has twelve. Both use RMSNorm with epsilon "}<InlineMath>{"10^{-6}"}</InlineMath>{" and learned scales. Residual/final LayerNorm uses epsilon "}<InlineMath>{"10^{-5}"}</InlineMath>{". The FFN has width 48 with GELU; the final prediction has two unconstrained coordinates."}</Prose>

<Prose>{"Attention maps have no biases. Stem, FFN and forecast maps have biases. Ordinary adjacent-pair RoPE uses base 10000, with the correct logical positions; there is no YaRN or dropout. This is a small dense model designed to expose MLA's mechanism, not a reproduction of DeepSeek's expert network."}</Prose>

<Prose>{"Train once with seed 131 for 200 full-batch Adam updates at learning rate 0.003, no weight decay. Choose the lowest validation MSE, taking the earliest exact tie. The selected checkpoint is update 200, the budget boundary. That is the recorded selection, not a claim that optimization has converged. The model has 4,118 trainable parameters."}</Prose>

<H3>{"Check the same function before changing its information"}</H3>

<Prose>{"Use the "}<strong>{"same weights and inputs"}</strong>{" for reconstructed and absorbed paths. In a float64 check on a small real prefix, their complete model outputs differ by at most "}<InlineMath>{"3.89\\times10^{-16}"}</InlineMath>{". Backpropagating the same squared-output objective through both paths gives corresponding gradients for "}<strong>{"every parameter"}</strong>{", with maximum difference "}<InlineMath>{"7.11\\times10^{-15}"}</InlineMath>{"."}</Prose>

<Prose>{"This directly contradicts the idea that absorbed attention must have unstable gradients because its normalization statistics somehow change. Both paths normalize the same vector and define the same differentiable function. Floating-point order and a kernel's implementation can affect numerical behavior; they do not change the algebraic identity. An exported, detached product must of course be refreshed if its underlying trainable weights change."}</Prose>

<Prose>{"For the 32-point visible prefix, ordinary float32 expanded/absorbed forecasts agree within "}<InlineMath>{"1.79\\times10^{-7}"}</InlineMath>{" in transformed coordinates. Processing the same prefix one point at a time with the compact cache agrees with the full causal pass within "}<InlineMath>{"3.58\\times10^{-7}"}</InlineMath>{". This checks projection, rotary positions, masks, residual computation and forecasts—not just concatenating already computed arrays."}</Prose>

<H3>{"Deliberately remove four latent directions"}</H3>

<Prose>{"Now ask a different question. Stack the trained content-key and value up-projections into a joint output-by-latent matrix M. Its singular value decomposition identifies orthogonal latent directions and their contribution to those linear maps. Let P contain the four right singular vectors with the largest singular values."}</Prose>

<Prose>{"For each new input, first compute the original normalized eight-coordinate c, then cache only"}</Prose>

<div className="neural-equation"><MathBlock>{"c'=P^Tc\\in\\mathbb R^4."}</MathBlock></div>

<Prose>{"Replace each up-projection U by UP. The reconstructed representation is now "}<InlineMath>{"UPP^Tc"}</InlineMath>{", which discards the other four directions. The rotary key and query paths stay unchanged. There is "}<strong>{"no additional training"}</strong>{" and no test-driven choice of rank."}</Prose>

<Prose>{"This intervention reduces persistent content coordinates. It still computes the original eight-coordinate RMSNorm before projection; it is not a separately trained four-coordinate latent architecture. The distinction matters both for arithmetic and for interpreting the result."}</Prose>

<Prose>{"The discarded squared singular values sum to about 2.862722. That matches the squared Frobenius error of the rank-4 joint up-projection. This is the optimum for that specified "}<strong>{"parameter-space reconstruction objective"}</strong>{", not a guarantee of minimal attention error or forecast loss."}</Prose>

<Prose>{"Use all eight singular directions as a control. The complete orthogonal basis changes coordinates but loses none, so the outputs should agree. Across the validation set, the actual maximum difference is "}<InlineMath>{"3.58\\times10^{-7}"}</InlineMath>{" in transformed coordinates. We have separately checked a basis change and a truncation, instead of calling both “compression.”"}</Prose>

<H3>{"Actual outcomes"}</H3>

<Prose>{"RMSE is per original normalized coordinate over all 44 output slots of each trajectory. The held-out set contains 60 whole trajectories."}</Prose>

<NeuralTable caption={"Actual outcomes"} headers={[<>{"Method"}</>,<>{"Validation RMSE"}</>,<>{"Test RMSE"}</>,<>{"Float32 compact payload for one 32-point prefix"}</>]} rows={[[<>{"Selected MLA, 8 content coordinates + 2 rotary"}</>,<>{"0.025576"}</>,<>{"0.024946"}</>,<>{"1,280 bytes"}</>],[<>{"Same model, rank-4 cache intervention + 2 rotary"}</>,<>{"0.095481"}</>,<>{"0.091934"}</>,<>{"768 bytes"}</>],[<>{"Persistence baseline"}</>,<>{"—"}</>,<>{"0.026458"}</>,<>{"Not an MLA cache"}</>],[<>{"Training-fitted affine baseline"}</>,<>{"—"}</>,<>{"0.026222"}</>,<>{"Not an MLA cache"}</>]]} />

<Prose>{"The selected small MLA is only modestly better than these simple baselines. The unadapted rank reduction substantially damages its forecasts. Both findings belong in the lesson. A lower parameter reconstruction error than another factorization would not prove good task performance; an important direction can have a modest singular value."}</Prose>

<Prose>{"The table does not compare this model directly with preceding lessons' differently shaped attention models. It also does not establish that a model trained from scratch at rank four could not learn well, or that further adaptation would never help. Those are different experiments requiring their own declared protocols."}</Prose>

<H3>{"Inspect an actual cached prediction"}</H3>

<Prose>{"The visible input was chosen before training: source row 77, with its first 32 points observed. The actual next point is approximately "}<code>{"[0.593810,0.250000]"}</code>{"."}</Prose>

<Prose>{"The full model predicts "}<code>{"[0.599737,0.263627]"}</code>{"; the rank-4 intervention predicts "}<code>{"[0.613758,0.260927]"}</code>{". The compact tensors have shapes "}<code>{"[1,32,8]"}</code>{" and "}<code>{"[1,32,2]"}</code>{" in the first case, versus "}<code>{"[1,32,4]"}</code>{" and "}<code>{"[1,32,2]"}</code>{" in the second. The two cache fields have different meanings; the rotary field is not a second copy of the latent."}</Prose>

<Prose>{"Reflect observed point 23's x coordinate with "}<InlineMath>{"x\\mapsto1-x"}</InlineMath>{" and recompute the prefix. The full model's final prediction becomes "}<code>{"[0.600519,0.263407]"}</code>{". All outputs before the edited point remain "}<strong>{"exactly unchanged"}</strong>{" in the checked computation. A changed past observation can affect subsequent forecasts, so the cache for the old prefix cannot silently be reused."}</Prose>

<Prose>{"Shift all logical positions by 100 under the same ordinary RoPE rule. Full-model outputs agree within "}<InlineMath>{"1.79\\times10^{-7}"}</InlineMath>{". Moving only queries while retaining old rotated keys is a different operation and generally changes scores. A consistent global position transformation can be a valid equivalence; “a cache can never be reused at another absolute position” is too strong. Reuse must preserve or correctly transform all relevant state and positional conventions."}</Prose>

<Prose>{"The wrong scale "}<InlineMath>{"1/\\sqrt{10}"}</InlineMath>{" changes this full model's final prediction only slightly, to "}<code>{"[0.599745,0.263674]"}</code>{". The small effect is still a changed function. Do not magnify it into a dramatic failure. For the rank-4 intervention, its latent width happens to equal the content-head width, so the accidental default equals the intended scale and gives a null result. A test that covers only equal widths can therefore miss the bug."}</Prose>

<LatentForecastLab />

<H3>{"Reproduce the entire study"}</H3>

<Prose>{"Place "}<a href={"/learn-assets/multi-head-latent-attention-mla/author-calculations.py"}>{"the full CPU program"}</a>{", "}<a href={"/learn-assets/multi-head-latent-attention-mla/movement_libras.data"}>{"original data"}</a>{" and "}<a href={"/learn-assets/multi-head-latent-attention-mla/movement_libras.names"}>{"original metadata"}</a>{" together. With Python, NumPy and PyTorch installed, run:"}</Prose>

<CodeBlock language={"bash"}>{"python author-calculations.py"}</CodeBlock>

<Prose>{"The recorded environment is Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, with one CPU thread and deterministic algorithms. The complete program implements preprocessing, duplicate-aware splitting, both baselines, training/selection, both MLA paths, normalized cache truncation, full-parameter derivative checks and actual forecast/cache interventions. It requires no network or GPU. "}<a href={"/learn-assets/multi-head-latent-attention-mla/author-results.json"}>{"Recorded results"}</a>{" and "}<a href={"/learn-assets/multi-head-latent-attention-mla/forecast-model.json"}>{"saved weights/traces"}</a>{" make the observations inspectable; different numerical environments can change last digits."}</Prose>

<Prose>{"Read "}<code>{"LatentForecaster.forward"}</code>{" in the same order as the diagram: normalize the input state, compute content/query latents and rotary branches, append compact fields and logical IDs, calculate the two score terms with the original scale, mask and normalize, mix latents, expand the current head outputs, then complete the residual/FFN/task path."}</Prose>

<LatentProgram file="author-calculations.py" />

<Prose>{"The website investigation needs only one small model and the selected input. Full author evidence is an optional download. It should not train models, evaluate the entire corpus or eagerly load every experiment on page opening."}</Prose>

<H2>{"7. Deeper connections and practical boundaries"}</H2>

<H3>{"What is low rank, and what is not?"}</H3>

<Prose>{"Before normalization, stacking the content-key and value maps gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{bmatrix}k^C\\\\v\\end{bmatrix}\n=\\underbrace{\\begin{bmatrix}U_K\\\\U_V\\end{bmatrix}}_M\nW^{DKV}h."}</MathBlock></div>

<Prose>{"The effective linear map has rank at most "}<InlineMath>{"d_c"}</InlineMath>{". With a nonlinear normalization before the up-map, the entire h-to-output map is no longer one fixed linear matrix. Nevertheless, the stacked outputs still lie in the column space of M, whose dimension is at most "}<InlineMath>{"d_c"}</InlineMath>{". These are related but different rank statements."}</Prose>

<Prose>{"This restriction explains why a smaller latent can lose useful distinctions. If two cached content vectors are identical, all their reconstructed content keys and values are identical. Their rotary keys can still differ, and the full model has other paths, so do not turn this into a claim that the complete inputs or predictions must be indistinguishable in every setting."}</Prose>

<Prose>{"For one head, the unnormalized content-score matrix factors through the latent coordinates. But row-softmax is nonlinear and does not preserve matrix rank. For example, the three-by-three logit matrix with entries "}<InlineMath>{"\\ell_{ij}=ij"}</InlineMath>{" for i,j in "}<code>{"{0,1,2}"}</code>{" has rank one. Its row-softmax matrix has rank three; the independently computed determinant is about 0.024431."}</Prose>

<LatentSoftmaxFigure />

<Prose>{"MLA therefore does not make the full softmax attention matrix low rank merely by using low-rank projection parameters. Dense full-sequence attention still compares query/key positions; its usual score work grows quadratically with sequence length. The "}<a href={"/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals"}>{"next Sparse and Linear Attention lesson"}</a>{" changes which comparisons happen or which operator is computed. Latent-coordinate compression and sequence-level approximation are separate ideas."}</Prose>

<H3>{"Why a good matrix approximation can be a poor predictor"}</H3>

<Prose>{"For a matrix M with singular values "}<InlineMath>{"\\sigma_1\\ge\\cdots\\ge\\sigma_r"}</InlineMath>{", a rank-k truncated SVD minimizes the squared Frobenius reconstruction error, with error "}<InlineMath>{"\\sum_{j>k}\\sigma_j^2"}</InlineMath>{". One way to understand this is to use orthogonal singular coordinates: each retained direction preserves one independent squared-energy contribution, so retaining the largest contributions minimizes the discarded sum. The full theorem covers all rank-k matrices, not only a particular coordinate deletion."}</Prose>

<Prose>{"This objective weights matrix entries uniformly. Actual inputs need not visit every direction uniformly, and task outputs need not value all errors equally. Take "}<InlineMath>{"M=\\operatorname{diag}(10,1)"}</InlineMath>{". Its best rank-one Frobenius approximation is "}<InlineMath>{"\\operatorname{diag}(10,0)"}</InlineMath>{", with squared error 1. For input "}<code>{"[0,10]"}</code>{", however, the original output is "}<code>{"[0,10]"}</code>{" and the approximation gives "}<code>{"[0,0]"}</code>{". The discarded direction contains the entire useful signal for this input."}</Prose>

<Prose>{"In attention, the consequences can be even less direct: key errors alter normalized weights, value errors alter what those weights mix, and later layers transform the result. A data-aware reconstruction objective might weight directions using input covariance; a task-aware adaptation can optimize prediction loss. Neither is automatically equivalent to minimizing projection-weight distance. Our rank-4 result is a concrete example of the distinction."}</Prose>

<LatentRankLab />

<H3>{"Converting an existing checkpoint is possible, but not a configuration edit"}</H3>

<Prose>{"An arbitrary trained MHA/GQA checkpoint does not already have the required joint latent factorization and shared rotary design. Changing a head-count or rank field cannot create compatible weights. It is also incorrect to claim that useful conversion is impossible."}</Prose>

<Prose>{""}<a href={"https://aclanthology.org/2025.acl-long.1597.pdf"}>{"MHA2MLA, published at ACL 2025"}</a>{", investigates partial-RoPE adaptation and joint low-rank factorization, followed by continued training. Its methods explicitly assess retained rotary subspaces and factorize the relevant key/value maps. This is a studied conversion route with measured tradeoffs, not proof that every checkpoint converts losslessly. A converted variant's retained rotary components and cache accounting must be read from that variant, rather than assumed identical to the original shared-key V2 design."}</Prose>

<Prose>{"A bounded conceptual initialization is straightforward when the relevant K/V maps are linear and no incompatible positional operation intervenes. Stack their output-by-input weights into M, compute a truncated SVD "}<InlineMath>{"M\\approx U_k\\Sigma_kV_k^T"}</InlineMath>{", choose down-map "}<InlineMath>{"\\Sigma_k^{1/2}V_k^T"}</InlineMath>{" and joint up-map "}<InlineMath>{"U_k\\Sigma_k^{1/2}"}</InlineMath>{", then split the up-map into key and value parts. These factors approximate the original parameter matrix. They do not automatically preserve task outputs, compensate for changed rotary dimensions or supply the proper normalization/adaptation recipe."}</Prose>

<Prose>{"Our small study starts from a model already trained with MLA and reduces its post-normalization cache. It is not a reproduction of MHA2MLA's full-to-partial-RoPE checkpoint conversion. The original normalization remains part of the declared model, so the two experiments should not be conflated."}</Prose>

<H3>{"What can move through the value sum?"}</H3>

<Prose>{"The value rearrangement requires a map shared across memory positions that is linear in the summed latent. An affine map "}<InlineMath>{"Uc+b"}</InlineMath>{" can also be handled: if attention weights sum to one, its weighted output is "}<InlineMath>{"U\\sum_s a_sc_s+b"}</InlineMath>{". With attention dropout or other unnormalized weights, the bias contributes "}<InlineMath>{"b\\sum_s a_s"}</InlineMath>{", which must be accounted for explicitly."}</Prose>

<Prose>{"A nonlinear map generally cannot move outside the sum. With latents −1 and 1 and weights one-half each, mixing their ReLU values gives 0.5, while applying ReLU to their mixed latent gives 0. Moving the nonlinearity has changed the function."}</Prose>

<Prose>{"Head-dependent linear maps are fine because each head has its own weighted latent. Position-dependent maps generally cannot be pulled outside a sum over positions as one fixed map. Data-dependent routing or quantization also needs its own exact contract. This reasoning is more useful than memorizing that every operation called an “up-projection” is absorbable."}</Prose>

<H3>{"Backward computation and numerical precision"}</H3>

<Prose>{"For an output "}<InlineMath>{"o=Uz"}</InlineMath>{", the gradient with respect to U is the outer product of the upstream output derivative with z; the gradient with respect to z is "}<InlineMath>{"U^T"}</InlineMath>{" times that derivative. For "}<InlineMath>{"z=\\sum_s a_sc_s"}</InlineMath>{", c receives both a direct value-mixture contribution and an indirect contribution through attention weights when c also influences keys."}</Prose>

<Prose>{"The same dependencies exist in the expanded graph. The chain rule combines them in a different order; it does not create a different intended derivative. Our full-parameter check compares those derivatives rather than relying on a loss curve as evidence that attention is correct."}</Prose>

<Prose>{"Floating-point multiplication and summation are not perfectly associative. Different intermediate precision, softmax accumulation, kernel reductions or quantization can produce different numerical errors. Compare outputs and gradients with a clear tolerance and inspect which operation changes. Exact algebra does not promise bitwise equality across GPU kernels."}</Prose>

<Prose>{"Nor does “FP8 model” mean every tensor, operation and cache field is FP8. The "}<a href={"https://arxiv.org/html/2412.19437v2#S3.SS3"}>{"V3 mixed-precision report"}</a>{" retains higher precision for selected operations, including attention and normalization. Cache storage is a separate serving choice. A currently documented V3.2 sparse FlashMLA format has 512 one-byte latent values, 16 scale bytes and 128 bytes for 64 BF16 rotary coordinates: "}<strong>{"656 bytes per token"}</strong>{", not simply 576 or 512. This is that specific format, not a universal MLA layout. "}<a href={"https://github.com/deepseek-ai/FlashMLA#mla-decoding"}>{"Official FlashMLA cache-format documentation"}</a>{"."}</Prose>

<H3>{"Read empirical claims with their actual comparison"}</H3>

<Prose>{"The original attention ablations are in "}<strong>{"Appendix D"}</strong>{", not the overall model comparison in Table 2. Table 9's small-MoE comparison reports MMLU 48.7 versus 50.0 for MHA/MLA, but C-Eval 51.6 versus 50.9. Its large-MoE comparison uses a different training scale. These are useful scoped results; even this one table does not justify “MLA never loses quality.” "}<a href={"https://arxiv.org/html/2405.04434v5#A4"}>{"DeepSeek-V2 attention ablations"}</a>{"."}</Prose>

<Prose>{"A quality-versus-cache plot needs comparable models, tasks, training budgets and measurements. Combining an MMLU number from one model family with a task-average score from another paper does not create a measured frontier. Our figures instead show exact formula-derived storage, actual operator differences and actual local forecasting outcomes, with their evidence types labelled."}</Prose>

<Prose>{"The "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json"}>{"V2"}</a>{" and "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json"}>{"V3"}</a>{" configurations share several MLA widths, but differ in model width, layer count and other settings. A supported buffer length in a configuration also need not equal a demonstrated effective context length on every task. Use the full positional configuration and training/evaluation evidence before extending an inference length."}</Prose>

<H3>{"Applications and interactions worth recognizing"}</H3>

<Prose>{""}<strong>{"Cached cross-attention."}</strong>{" If encoder outputs are fixed while a decoder generates, their latent content and positional representations can be cached and reused. Source positions and decoder positions need an appropriate cross-attention convention; blindly reusing the self-attention rotary relation may be inappropriate. If the source changes, its cached representation changes. The later cross-attention topic owns the full source/target alignment design."}</Prose>

<Prose>{""}<strong>{"Movement, sensor and event streams."}</strong>{" Our forecast demonstrates an actual non-language use. Compact per-observation state can help when many positions are retained. A dense cache still grows with stream length; a window, reset or another architecture is needed for bounded indefinite memory. Sensor calibration or corrected historical observations also require invalidation policies."}</Prose>

<Prose>{""}<strong>{"MoE and sparse attention."}</strong>{" Expert routing usually changes the feedforward part of a Transformer; MLA changes attention representation. They can coexist without expert count multiplying every attention-cache field. Likewise, selecting fewer legal/retrieved positions can coexist with a latent representation for those positions. The current "}<a href={"https://github.com/deepseek-ai/FlashMLA"}>{"FlashMLA repository"}</a>{" distinguishes dense and sparse kernels and version-specific formats. The next variants lesson and "}<a href={"/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals"}>{"MoE lesson"}</a>{" develop these separate mechanisms."}</Prose>

<Prose>{""}<strong>{"Encoders and short inputs."}</strong>{" The operator can be used without an autoregressive cache. Its main persistent-history saving then may not apply, but representation/parameter choices can still be studied. There is no mathematical ban on encoder use or universal one-billion-parameter cutoff. Compare the actual task, temporary activation requirements and implementation support."}</Prose>

<H3>{"Choose the implementation for the workload"}</H3>

<Prose>{"Start with an explicit reference for masks, shapes, scale, normalization, rotary layout and cache lifetime. Then select an implementation that supports those dimensions and dtypes. Check whether the kernel expects expanded heads or an MQA-like latent layout; that API label can describe how the same MLA computation is executed, rather than a different trained architecture."}</Prose>

<Prose>{"For performance, measure prefill and decode separately at stated lengths, batches, dtypes and hardware, including cache allocation and synchronization. Consider total weights, temporary workspace, communication and scheduling. A cache ratio alone cannot explain a commercial API price or establish how many GPUs serve a complete model under a latency target."}</Prose>

<Prose>{"The published "}<a href={"https://arxiv.org/html/2412.19437v2#S3.SS4"}>{"V3 deployment discussion"}</a>{" describes substantial distributed, workload-specific arrangements. It is not evidence for a universal single-server fit rule. Current kernels may support formats and variants newer than this lesson's V2/V3 core; pin the actual source/version when reproducing them. Our CPU programs teach correctness and the effects of a declared intervention, with no claim of a production latency crossover."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"8. Practice: preserve a computation or change it deliberately"}</H2>

<Prose>{"Use a fresh calculation or prediction before opening the optional help. Exercises 1–5 cover the first-pass route; 6–9 extend the deeper branches."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Read one head through a different latent basis"}</H3>

<Prose>{"A head has content query "}<code>{"[1,2]"}</code>{", latent records "}<code>{"[1,0]"}</code>{" and "}<code>{"[0,1]"}</code>{", key up-map "}<code>{"[[2,0],[0,1]]"}</code>{", value up-map equal to the identity, and no positional contribution. The defined scale is "}<InlineMath>{"1/\\sqrt2"}</InlineMath>{". Compute its effective query, weights and output. Then use basis change "}<InlineMath>{"S=\\operatorname{diag}(2,0.5)"}</InlineMath>{". What must happen to the latent records and up-maps to preserve the output?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compute "}<InlineMath>{"U_K^Tq"}</InlineMath>{" first. A consistent basis change uses "}<InlineMath>{"c'=Sc"}</InlineMath>{" and "}<InlineMath>{"U'=US^{-1}"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The effective query is "}<code>{"[2,2]"}</code>{"; both scaled scores are "}<InlineMath>{"\\sqrt2"}</InlineMath>{", so weights are "}<code>{"[0.5,0.5]"}</code>{" and the output is "}<code>{"[0.5,0.5]"}</code>{". The new latent records are "}<code>{"[2,0]"}</code>{" and "}<code>{"[0,0.5]"}</code>{". Right-multiply both up-maps by "}<InlineMath>{"S^{-1}=\\operatorname{diag}(0.5,2)"}</InlineMath>{". Their reconstructed keys and values then equal the originals, so scores, weights and output are preserved. Changing only the stored coordinates would generally change the function."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Preserve the temperature after absorption"}</H3>

<Prose>{"A model has content-head width 4, rotary width 4 and latent width 12. Its concatenated latent/rotary query has width 16. A library uses its default inverse-square-root feature-width scale. What scale should the model use, what scale did the library choose, and how were the logits changed? Why can a test with latent width 4 miss this mistake?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The algebra preserves the old dot product, so compare the original content-plus-rotary width with the new representation width."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The intended scale is "}<InlineMath>{"1/\\sqrt8"}</InlineMath>{"; the default is "}<InlineMath>{"1/\\sqrt{16}=1/4"}</InlineMath>{". Every finite unmasked logit is multiplied by "}<InlineMath>{"\\sqrt{8/16}=1/\\sqrt2"}</InlineMath>{" relative to the intended value, making unequal logits less separated before softmax. This generally changes weights, although equal-score or other special cases can be unchanged. If latent width also equals 4, both total widths equal 8 and the scales coincide, concealing the bug."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Build the compact payload"}</H3>

<Prose>{"Two requests each retain 2,048 positions at 12 layers. The content latent has 24 coordinates and the shared rotary key has 8, stored at two bytes each. Compute the compact payload in bytes and MiB. If the model doubles its query-head count while preserving these cache widths, what changes in this count and what can still become more expensive?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Multiply requests, layers, occupied positions, coordinates per record and bytes per coordinate. A MiB is "}<InlineMath>{"2^{20}"}</InlineMath>{" bytes."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The payload is "}<InlineMath>{"2\\times12\\times2048\\times32\\times2=3,145,728"}</InlineMath>{" bytes, or 3 MiB. It stays unchanged when only query-head count doubles. Query/head projections, score/value-mixture work and transient tensors can grow, as can communication or kernel overhead. The payload excludes reserved capacity, metadata, replicated copies and other model state."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Distinguish three “compression” operations"}</H3>

<Prose>{"A programmer proposes: A, rearrange the same MLA weights from reconstructed to absorbed execution; B, change to an invertible latent basis and compensate every up-map; C, retain only the first half of some latent coordinates. Which are algebraically function-preserving under the stated conditions? Does an observed agreement on one input prove C is lossless?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Ask whether information or an operation was discarded, rather than whether a tensor has a new name."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"A is exact when masking, scale, positions and all other operations remain equivalent. B is exact for the defined cached vector with the compensating inverse maps. C is generally lossy; it is exact only if the discarded directions have no relevant effect for the domain being claimed. Agreement on one input may mean that input has no discarded component or that effects cancel. It does not prove equality for every possible input. A basis change before an uncompensated nonlinear normalization is a different operation from B."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Diagnose a cache that forgot its positional field"}</H3>

<Prose>{"A developer caches only the normalized content latent, then reconstructs every shared rotary key as if its position were zero. Shapes still match, and next-token outputs look plausible. What information is wrong? Why is a successful one-position test insufficient? Give a useful controlled comparison."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Trace the two additive score terms. Think about both different relative distances and a consistent common shift."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The content term can be correct while the rotary term compares queries with keys rotated at the wrong logical positions. A one-position test can have only one legal key, making softmax equal to one regardless of the score; it therefore cannot establish positional correctness. Compare an explicit multi-position reference with the cache path, using unequal vectors and distances. Also shift all positions consistently under fixed ordinary RoPE as a preservation control, then shift only queries or only cached-key positions as a contrasting operation. Retain or correctly reconstruct each required rotary key and its positional contract."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Find the nonlinear obstruction"}</H3>

<Prose>{"Two cached scalar latents are −2 and 1 with weights 1/3 and 2/3. Values are obtained with ReLU. Compare mixing the values with applying ReLU after mixing latents. Would the rearrangement become valid for a fixed linear value map instead?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Evaluate ReLU on each value first in the original expression. In the second expression, add before applying it."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Mixing ReLU values gives "}<InlineMath>{"(1/3)0+(2/3)1=2/3"}</InlineMath>{". Mixing latents gives "}<InlineMath>{"(1/3)(-2)+(2/3)1=0"}</InlineMath>{", whose ReLU is zero. The two computations differ. A fixed linear map distributes over the weighted sum, so its rearrangement is exact. A position-dependent map, normalization or nonlinear activation needs separate analysis."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Explain why the smaller cache did more arithmetic"}</H3>

<Prose>{"A single-query attention call has H=8, L=1,024, content width 8, rotary width 4, value width 8 and latent width 32. Ignore projection work and count a multiply-add as two operations. Compute expanded-core and absorbed-core operations. Does the larger arithmetic number decide which is faster?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use "}<InlineMath>{"2HL(d_k+d_r+d_v)"}</InlineMath>{" and "}<InlineMath>{"2HL(2d_c+d_r)"}</InlineMath>{" for B=T=1."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Expanded core uses "}<InlineMath>{"2\\times8\\times1024\\times20=327,680"}</InlineMath>{" operations. Absorbed core uses "}<InlineMath>{"2\\times8\\times1024\\times68=1,114,112"}</InlineMath>{", or 3.4 times as many. The absorbed cache can still require much less stored data and support reuse across heads. Hardware bandwidth, arithmetic throughput, kernel layout and other overheads determine latency; neither the FLOP ratio nor the cache ratio alone is a timing result."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Challenge the rank interpretation"}</H3>

<Prose>{"Someone says, “The content logits have rank at most two, so the softmax weights also have rank at most two and the entire attention is linear in sequence length.” Identify the two unsupported steps. What would a genuine linear-attention method need to specify?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate a rank bound on a matrix product from the effect of a nonlinear elementwise transformation and normalization."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Softmax need not preserve matrix rank; the lesson gives rank-one logits whose three-by-three probability matrix has full rank. Also, a latent feature factorization does not by itself remove the pairwise softmax computation over query and memory positions. A linear-attention method must specify its actual kernel/operator, normalization and associative accumulation or approximation, including causal-state behavior and its relationship to ordinary softmax. Those are the next topic's mechanisms."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Design the next experiment honestly"}</H3>

<Prose>{"Our trained rank-eight model worsened after a rank-four cache intervention. A colleague concludes that rank-four MLA can never work. Another concludes that ten more epochs would certainly recover it. What do the recorded results actually establish, and what protocol would investigate either possibility without choosing a result after looking at the test set?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Identify which model was trained, which operation happened afterward, and which data chose the checkpoint."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The result establishes the effect of one predeclared, unadapted post-normalization truncation on one selected small model and this row-level task. It does not test a rank-four model trained from scratch or a declared adaptation schedule. Define that new architecture/intervention, training budget, seeds, simple baselines and validation criterion in advance; preserve group/duplicate boundaries and keep test data outside selection. Report all declared outcomes, including failure to recover. A selected checkpoint at the budget endpoint is not evidence of eventual convergence or certain recovery."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--next" data-lesson-ending="next"><H2>{"What comes next?"}</H2>

<Prose>{"You can now identify the actual cache fields, calculate both equivalent MLA paths, preserve scale and position, and distinguish representation loss from computational rearrangement. Continue to "}<a href={"/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals"}>{"Sparse and Linear Attention Variants"}</a>{", which changes the sequence comparisons or attention operator itself. Carry the same questions forward: what function is defined, what information is retained, what approximation is introduced, and what was actually measured?"}</Prose></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References and another way to learn"}</H2>

<ul><li>{""}<strong>{"Original architecture:"}</strong>{" "}<a href={"https://arxiv.org/html/2405.04434v5"}>{"DeepSeek-V2 report"}</a>{". The full section list, §2.1, practical model settings, context-extension treatment, Appendix C formulas and Appendix D ablations were inspected. Read the operator and cache definitions before the empirical comparisons. Its full model includes other changes, so headline system savings cannot all be attributed to one attention identity."}</li><li>{""}<strong>{"An executable primary implementation:"}</strong>{" "}<a href={"https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py"}>{"DeepSeek-V3 official inference model"}</a>{". The actual rotary function, MLA constructor, RMSNorm/cache branches and complete MLA forward were inspected. Follow the two branches with the same shapes. This is a mutable source branch; pin a revision for reproduction. Our complete small CPU program is independently implemented and executed."}</li><li>{""}<strong>{"A compact alternate explanation:"}</strong>{" "}<a href={"https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/"}>{"Sebastian Raschka's MLA chapter guide"}</a>{". The complete short body was read. It connects the preceding GQA idea to a compressed cache and offers a useful recap; use the equations and source-bound examples here for scale, positional and performance details."}</li><li>{""}<strong>{"Visual attention prerequisite:"}</strong>{" "}<a href={"https://www.3blue1brown.com/lessons/attention/"}>{"3Blue1Brown's illustrated attention article and accompanying video"}</a>{". Its substantive written Q/K/softmax/value explanation was inspected in the preceding packets and is reused as background. It helps visualize the weighted sum that MLA rearranges; it is not an MLA kernel tutorial, and no new video viewing is claimed here."}</li><li>{""}<strong>{"Existing-checkpoint conversion:"}</strong>{" "}<a href={"https://aclanthology.org/2025.acl-long.1597/"}>{"MHA2MLA at ACL 2025"}</a>{" and "}<a href={"https://aclanthology.org/2025.acl-long.1597.pdf"}>{"the paper"}</a>{". The background, partial-RoPE selection, split/joint SVD methods and model/task setup with the first results table were inspected. They establish a concrete conversion research route, with task- and rank-dependent costs. Do not interpret the paper's title as a universal lossless-conversion guarantee."}</li><li>{""}<strong>{"Prefill and decode implementations:"}</strong>{" "}<a href={"https://docs.vllm.ai/en/v0.20.0/api/vllm/model_executor/layers/attention/mla_attention/"}>{"vLLM 0.20.0 MLA notes"}</a>{" were inspected through dimensions, both compute paths and chunked prefill. Match the actual shapes and scale rather than copying informal pseudocode or a reversed ratio description. No vLLM installation or hardware performance test was performed here."}</li><li>{""}<strong>{"Current kernel and precision contracts:"}</strong>{" "}<a href={"https://github.com/deepseek-ai/FlashMLA"}>{"Official FlashMLA"}</a>{". The dense/sparse capability table, API and version-specific cache formats were inspected on 13 September 2026. These distinguish stored bytes, scaling metadata and rotary precision. Kernel/model variants continue to evolve; their reported throughput is not a measurement made by this lesson."}</li><li>{""}<strong>{"Practical model details:"}</strong>{" "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json"}>{"V2 configuration"}</a>{", "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json"}>{"V3 configuration"}</a>{", and "}<a href={"https://arxiv.org/html/2412.19437v2"}>{"V3 report"}</a>{". Relevant architecture, normalization, mixed-precision, deployment and context-extension portions were inspected. They help distinguish actual fields from shorthand architecture names."}</li><li>{""}<strong>{"Reproducible local learning:"}</strong>{" "}<a href={"/learn-assets/multi-head-latent-attention-mla/data-provenance.md"}>{"Data/protocol provenance"}</a>{", "}<a href={"/learn-assets/multi-head-latent-attention-mla/author-calculations.py"}>{"complete CPU forecasting program"}</a>{", "}<a href={"/learn-assets/multi-head-latent-attention-mla/author-results.json"}>{"actual outcomes"}</a>{", "}<a href={"/learn-assets/multi-head-latent-attention-mla/forecast-model.json"}>{"saved model/trace"}</a>{", "}<a href={"/learn-assets/multi-head-latent-attention-mla/mechanism-calculations.py"}>{"independent NumPy fixtures"}</a>{" and "}<a href={"/learn-assets/multi-head-latent-attention-mla/mechanism-fixtures.json"}>{"their exact values"}</a>{". These support the manuscript's observed and constructed examples without requiring a large pretrained-model download."}</li></ul></section>
</div> };
export default lesson;
