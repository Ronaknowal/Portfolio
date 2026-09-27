// Full prepared manuscript rendered by render-sparse-attention-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { SparseFigure } from '../../components/lesson-labs/SparseAttentionFigures.jsx';
import { SparseGraphLab, SparseReadLab, SparseMemoryLab, SparseRandomLab, SparseProjectionLab, SparseTilesLab, SparseForecastLab, SparseProgram } from '../../components/lesson-labs/SparseAttentionLabs.jsx';
import '../../components/lesson-labs/neural-lesson-neutral.css';
export default {
 title: 'Sparse & Linear Attention Variants',
 readTime: '~100 min read + experiments and practice',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson neural-lesson-neutral sparse-attention-lesson"><LessonIntro prerequisites="Attention weights, vector products, matrix multiplication and a recurrent update; the local examples refresh the required shapes and normalization." sections={[["1-locate-the-cost-before-choosing-a-shortcut","1. Locate the cost before choosing a shortcut"],["2-sparse-attention-choose-which-connections-exist","2. Sparse attention: choose which connections exist"],["3-linear-attention-change-the-question-the-memory-can-answer","3. Linear attention: change the question the memory can answer"],["4-approximate-softmax-with-random-features","4. Approximate softmax with random features"],["5-compress-the-sequence-instead-of-its-feature-sums","5. Compress the sequence instead of its feature sums"],["6-make-the-graph-efficient-on-the-actual-machine","6. Make the graph efficient on the actual machine"],["7-compare-three-operators-on-observed-hand-trajectories","7. Compare three operators on observed hand trajectories"],["8-deeper-connections-and-practical-judgment","8. Deeper connections and practical judgment"],["9-practice-and-transfer","9. Practice and transfer"],["10-another-way-to-learn-and-what-comes-next","10. Another way to learn, and what comes next"]]}>Trace what reaches a query, what a memory retains, and which work an implementation actually avoids.</LessonIntro>
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit sparse edges, feature-memory writes/evictions, random-feature settings, compression coefficients, block layout and supported real trajectories. Show removed mass, reachability, normalized summaries, approximation error, future influence and tile occupancy live under a fixed random draw. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose sparsity or approximation by accessible information, numerical error and actual block work, not a single sparsity percentage."}</Prose>

<Prose>{"A long conversation contains many earlier tokens, but the next token may need only a few of them. A stream of hand movements has the opposite possibility: many earlier observations may matter collectively, without needing to retrieve any single observation exactly. These suggest two different ways to reduce attention's work: "}<strong>{"read fewer individual records"}</strong>{", or "}<strong>{"maintain a smaller summary that can answer a particular kind of query"}</strong>{"."}</Prose>

<Prose>{"The distinction matters. Removing connections, approximating a similarity function, compressing the sequence, and executing the same calculation more carefully can all reduce a resource cost. They preserve different things. This lesson gives you a way to inspect those choices rather than memorize a ranking of model names."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" follow §§1–3 and the opening of §4, the worked sequence-compression example in §5, and the real trajectory experiment in §7. Try the graph and memory investigations, then practice 1–5. You should be able to explain what information an operator can use, calculate a small output, and distinguish work from retained state. "}<strong>{"Deeper pass:"}</strong>{" study the random-feature derivation, approximation families, gradients and current sparse systems in §§4–6 and §8, then the remaining practice. Those branches develop implementation and research judgment; they are not hidden requirements for understanding the core story."}</Prose>

<Prose>{"We build on "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention & Multi-Head Attention"}</a>{", "}<a href={"/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals"}>{"Transformer Block Architecture"}</a>{", and the immediately preceding "}<a href={"/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals"}>{"Multi-Head Latent Attention"}</a>{". The refreshers below supply the particular mathematics we need."}</Prose>

<H2>{"1. Locate the cost before choosing a shortcut"}</H2>

<Prose>{"For one head, a "}<strong>{"query"}</strong>{" asks what to read; each "}<strong>{"key"}</strong>{" describes an available record; its "}<strong>{"value"}</strong>{" is the information returned. At position "}<InlineMath>{"t"}</InlineMath>{", let "}<InlineMath>{"J_t"}</InlineMath>{" be the legal key positions. For a causal decoder, "}<InlineMath>{"J_t=\\{0,\\ldots,t\\}"}</InlineMath>{". The head computes"}</Prose>

<div className="neural-equation"><MathBlock>{"s_{tj}=q_t^Tk_j/\\sqrt{d_k},\\qquad\np_{tj}=\\frac{e^{s_{tj}}}{\\sum_{u\\in J_t}e^{s_{tu}}},\\qquad\no_t=\\sum_{j\\in J_t}p_{tj}v_j."}</MathBlock></div>

<Prose>{"The row of weights sums to one. Its denominator depends on every legal score, so the ordinary softmax cannot simply be moved through a matrix multiplication. In general,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{softmax}(QK^T)V\\ne\n\\operatorname{softmax}(Q)(K^TV)."}</MathBlock></div>

<Prose>{"There are several bills to pay, and paying less of one does not cancel the others."}</Prose>

<NeuralTable caption={"1. Locate the cost before choosing a shortcut"} headers={[<>{"Resource"}</>,<>{"What it pays for"}</>,<>{"What changes it"}</>]} rows={[[<>{"Attention arithmetic"}</>,<>{"Key comparisons and value accumulation"}</>,<>{"Fewer edges, fewer summary slots, or a different factorable kernel"}</>],[<>{"Temporary attention storage"}</>,<>{"Scores, probabilities and backward intermediates"}</>,<>{"Tiling, recomputation, checkpointing, fused kernels"}</>],[<>{"Persistent inference state"}</>,<>{"Information retained for future tokens"}</>,<>{"KV sharing, latent compression, eviction, recurrent summaries"}</>],[<>{"Whole-model work"}</>,<>{"Projections, FFNs, routing, normalization and communication"}</>,<>{"Architecture and implementation beyond the attention core"}</>]]} />

<Prose>{"For length "}<InlineMath>{"L"}</InlineMath>{", dense causal attention has "}<InlineMath>{"L(L+1)/2"}</InlineMath>{" legal pairs per head. Computing their dot products and weighted values costs order "}<InlineMath>{"L^2(d_k+d_v)"}</InlineMath>{". A simple implementation allocates a full square score tensor, including cells later masked. At batch one, 32 heads and two bytes per score, that tensor alone is 4 GiB at "}<InlineMath>{"L=8192"}</InlineMath>{", and 64 GiB at "}<InlineMath>{"L=32768"}</InlineMath>{": "}<InlineMath>{"32L^2\\times2"}</InlineMath>{" bytes, with "}<InlineMath>{"1\\text{ GiB}=2^{30}"}</InlineMath>{" bytes. These are allocation calculations, not peak memory measurements or claims about which GPU can train a model."}</Prose>

<Prose>{""}<strong>{"Exact attention need not allocate that square tensor."}</strong>{" FlashAttention tiles the same softmax computation and accumulates it using stable running statistics; it reduces transfers and intermediates while retaining the dense operator and its quadratic pair arithmetic. The relevant comparison for a proposed approximation is a strong exact implementation, not only a deliberately materialized reference. "}<a href={"https://arxiv.org/abs/2205.14135"}>{"FlashAttention paper"}</a>{""}</Prose>

<Prose>{"The previous GQA and MLA lessons reduced what is stored per past token. Sparse attention instead asks which past tokens to read. A recurrent feature method changes how the past is represented. These choices can sometimes be combined, but their equations and costs must still be checked together."}</Prose>

<SparseFigure kind="representations" />

<H2>{"2. Sparse attention: choose which connections exist"}</H2>

<Prose>{"Choose a nonempty subset "}<InlineMath>{"S_t\\subseteq J_t"}</InlineMath>{", then perform ordinary softmax over that subset:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\tilde o_t=\\sum_{j\\in S_t}\n\\frac{e^{s_{tj}}}{\\sum_{u\\in S_t}e^{s_{tu}}}v_j."}</MathBlock></div>

<Prose>{"This is exact attention "}<strong>{"for the specified sparse graph"}</strong>{". It generally differs from the original dense graph. The retained values are the same records, but their probabilities change because the normalization changes."}</Prose>

<H3>{"A removed value changes the other weights too"}</H3>

<Prose>{"Suppose a dense row has weights "}<InlineMath>{"[0.5,0.25,0.25]"}</InlineMath>{" and scalar values "}<InlineMath>{"[2,-1,4]"}</InlineMath>{". Its output is"}</Prose>

<div className="neural-equation"><MathBlock>{"0.5(2)+0.25(-1)+0.25(4)=1.75."}</MathBlock></div>

<Prose>{"Remove the final key. The two retained weights become "}<InlineMath>{"2/3"}</InlineMath>{" and "}<InlineMath>{"1/3"}</InlineMath>{", so the result becomes "}<InlineMath>{"1"}</InlineMath>{", not "}<InlineMath>{"0.75"}</InlineMath>{". The removed probability mass was "}<InlineMath>{"\\delta=0.25"}</InlineMath>{", but the output changed by "}<InlineMath>{"0.75"}</InlineMath>{". Probability mass and output error have different units."}</Prose>

<Prose>{"There is a useful bound. Let "}<InlineMath>{"\\mu_{\\rm keep}"}</InlineMath>{" and "}<InlineMath>{"\\mu_{\\rm drop}"}</InlineMath>{" be the normalized weighted averages within the retained and removed sets, with "}<InlineMath>{"0<\\delta<1"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"o=(1-\\delta)\\mu_{\\rm keep}+\\delta\\mu_{\\rm drop},\\qquad\no-\\tilde o=\\delta(\\mu_{\\rm drop}-\\mu_{\\rm keep})."}</MathBlock></div>

<Prose>{"If all values satisfy "}<InlineMath>{"\\|v_j\\|\\le R"}</InlineMath>{", the triangle inequality gives "}<InlineMath>{"\\|o-\\tilde o\\|\\le2R\\delta"}</InlineMath>{". Here the bound is "}<InlineMath>{"2"}</InlineMath>{", which safely exceeds "}<InlineMath>{"0.75"}</InlineMath>{". It can be loose: if every value equals the same vector, dropping keys changes the probabilities but leaves the output unchanged. A small removed mass is helpful evidence, not a complete end-to-end model guarantee; later layers may amplify or damp the change. Also, calculating the exact removed mass ordinarily requires the full reference row, so it is an evaluation diagnostic rather than a free sparse-selection algorithm."}</Prose>

<SparseFigure kind="mass" />

<H3>{"Windows have a precise reach"}</H3>

<Prose>{"In this lesson, a causal window of width "}<strong>{""}<InlineMath>{"W"}</InlineMath>{" means at most "}<InlineMath>{"W"}</InlineMath>{" total keys, including the current position"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"S_t=\\{j:0\\le j\\le t,\\ t-j<W\\}."}</MathBlock></div>

<Prose>{"This convention prevents a common off-by-one error. With "}<InlineMath>{"W=3"}</InlineMath>{", position 7 can directly read 5, 6 and 7. For "}<InlineMath>{"L=12"}</InlineMath>{", the first rows have 1 and 2 keys; the remaining ten have 3, giving "}<InlineMath>{"33"}</InlineMath>{" edges. In general, with "}<InlineMath>{"m=\\min(L,W)"}</InlineMath>{", the count is"}</Prose>

<div className="neural-equation"><MathBlock>{"m(m+1)/2+(L-m)m."}</MathBlock></div>

<Prose>{"Read the attention mask as a graph: row "}<InlineMath>{"t"}</InlineMath>{", column "}<InlineMath>{"j"}</InlineMath>{" means information can travel "}<strong>{"from input "}<InlineMath>{"j"}</InlineMath>{" to output "}<InlineMath>{"t"}</InlineMath>{""}</strong>{" in one layer. A second layer reads the first layer's representations, allowing information to move again. With this window and only tokenwise operations between attention layers, "}<InlineMath>{"D"}</InlineMath>{" layers can reach at most "}<InlineMath>{"D(W-1)"}</InlineMath>{" positions into the past. Residual connections preserve already reachable information. Parallel heads in one layer do not turn a two-edge route into a two-layer computation."}</Prose>

<Prose>{"This is a statement about possible dependence. A reachable record may receive negligible weight or lose information while passing through intermediate vectors. A missing path, however, gives an exact inability to depend on that record under the stated architecture."}</Prose>

<H3>{"Global, strided and random connections"}</H3>

<Prose>{"A "}<strong>{"global token"}</strong>{" can gather and redistribute information. In a bidirectional encoder, it can read the whole input and other positions can read it in a later layer. Longformer's local-plus-global pattern makes this useful for document classification or question tokens. It also uses separate projections for its global attention. Dilated windows sample nearby offsets more sparsely. "}<a href={"https://arxiv.org/html/2004.05150v2"}>{"Longformer, §3"}</a>{""}</Prose>

<Prose>{"Causality changes that picture. A global token at position 0 cannot collect information from position 2 in a causal decoder. In our 12-position example, adding a causal hub at 6 creates the path "}<InlineMath>{"2\\to6\\to11"}</InlineMath>{"; adding a hub at 0 does not. Every edge must respect time, including edges inside a pooling or routing component. Drawing an undirected star hides this distinction."}</Prose>

<SparseFigure kind="hubs" />

<Prose>{"Alternating local and strided patterns offers another route. A layer can read a nearby range, then another layer can read positions separated by stride "}<InlineMath>{"s"}</InlineMath>{". With "}<InlineMath>{"s"}</InlineMath>{" around "}<InlineMath>{"\\sqrt L"}</InlineMath>{", a construction using local spans of order "}<InlineMath>{"s"}</InlineMath>{" and strided spans of order "}<InlineMath>{"L/s"}</InlineMath>{" costs order "}<InlineMath>{"L\\sqrt L"}</InlineMath>{". The exact routes depend on offsets and layer order. Sparse Transformer developed local/strided and fixed-block factorizations for sequence generation, including images and audio; the factorization changes the available computation rather than reproducing every dense head exactly. "}<a href={"https://arxiv.org/html/1904.10509v1"}>{"Sparse Transformer, §4–5"}</a>{""}</Prose>

<Prose>{"BigBird combines local, global and random connections. With fixed numbers of each, the edge count is linear in sequence length. Its universality result is an existence theorem for sufficiently expressive networks and continuous functions on a fixed-length compact domain, using an appropriate graph containing a global star. It is not a promise that a fixed small model will equal dense attention, nor that a particular number of random edges guarantees task accuracy. The paper also studies genomics, where relevant sequence context extends beyond nearby symbols. "}<a href={"https://arxiv.org/html/2007.14062v2"}>{"BigBird, §2–3 and §5"}</a>{""}</Prose>

<SparseGraphLab /><SparseReadLab />

<H3>{"Content-based selection: find candidates without comparing every full pair"}</H3>

<Prose>{"*Deeper family comparison; continue to §3 on a first pass.*"}</Prose>

<Prose>{"A fixed window cannot know that an old variable definition is relevant to today's query. Content routing first obtains a manageable candidate set, then applies attention within it."}</Prose>

<Prose>{""}<strong>{"Reformer"}</strong>{" uses locality-sensitive hashing. Its actual hash chooses the largest component of concatenated positive and negative random projections, "}<InlineMath>{"h(x)=\\arg\\max[xR;-xR]"}</InlineMath>{". Similar directions tend to share a bucket; this is not the same hash as independently taking every projection's sign. Shared query/key representations, normalized keys, sorting by bucket, bounded chunks and multiple hash rounds make candidate comparisons manageable. Causal masks use original positions after sorting. Repeated candidates across rounds must be handled without accidentally multiplying their contribution. A bounded chunk can miss members of a large bucket; an uncapped all-pairs bucket instead risks quadratic work. "}<a href={"https://arxiv.org/html/2001.04451v2"}>{"Reformer, §2"}</a>{""}</Prose>

<Prose>{""}<strong>{"Routing Transformer"}</strong>{" replaces fixed random partitions with online clustering of query/key representations. It uses normalized representations and balanced candidate budgets. With "}<InlineMath>{"c"}</InlineMath>{" clusters, assignment costs roughly "}<InlineMath>{"Lcd"}</InlineMath>{"; balanced within-cluster comparisons cost roughly "}<InlineMath>{"L^2d/c"}</InlineMath>{". Balancing those terms suggests "}<InlineMath>{"c"}</InlineMath>{" of order "}<InlineMath>{"\\sqrt L"}</InlineMath>{", hence order "}<InlineMath>{"L^{3/2}d"}</InlineMath>{", not automatically linear. Original position masks still matter. "}<a href={"https://aclanthology.org/2021.tacl-1.4.pdf"}>{"Routing Transformer, §4.1"}</a>{""}</Prose>

<Prose>{"These are approximate search mechanisms: a relevant key can be missed. Their cost includes sorting, assignment, selection and gathers. Computing a full dense score matrix and then zeroing small probabilities produces sparse *weights*, but it has already paid for the dense score computation."}</Prose>

<H2>{"3. Linear attention: change the question the memory can answer"}</H2>

<Prose>{"Sparse attention retains individual records but skips some reads. A feature-kernel method can include every earlier record by first combining them into a small state."}</Prose>

<Prose>{"A "}<strong>{"feature map"}</strong>{" transforms a vector into another vector, "}<InlineMath>{"\\phi:\\mathbb R^{d_k}\\to\\mathbb R^m"}</InlineMath>{". A "}<strong>{"kernel"}</strong>{" here is a similarity of the form"}</Prose>

<div className="neural-equation"><MathBlock>{"\\kappa(q,k)=\\phi(q)^T\\phi(k)."}</MathBlock></div>

<Prose>{"For a normalized weighted average, we choose features giving nonnegative similarities and require a positive denominator. Instead of softmax, define"}</Prose>

<div className="neural-equation"><MathBlock>{"o_t=\\frac{\\sum_{j\\le t}\\phi(q_t)^T\\phi(k_j)v_j}\n{\\sum_{j\\le t}\\phi(q_t)^T\\phi(k_j)}."}</MathBlock></div>

<Prose>{"Distribute the multiplication inside the sum:"}</Prose>

<div className="neural-equation"><MathBlock>{"S_t=\\sum_{j\\le t}\\phi(k_j)v_j^T\\in\\mathbb R^{m\\times d_v},\\qquad\nz_t=\\sum_{j\\le t}\\phi(k_j)\\in\\mathbb R^m,"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"o_t^T=\\frac{\\phi(q_t)^TS_t}{\\phi(q_t)^Tz_t}."}</MathBlock></div>

<Prose>{"The same result is computed by a different order of operations. A new record adds an "}<strong>{"outer product"}</strong>{": each key-feature component scales the entire value vector, creating one row contribution to "}<InlineMath>{"S"}</InlineMath>{". The query reads a weighted combination of those rows. The normalizer "}<InlineMath>{"z"}</InlineMath>{" keeps the matching amount of key-feature evidence, so output magnitude does not simply grow with the number of records. This causal recurrence is a central construction in "}<a href={"https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf"}>{"Transformers are RNNs, §3"}</a>{"."}</Prose>

<H3>{"Work one memory update by hand"}</H3>

<Prose>{"Use two key features and scalar values:"}</Prose>

<NeuralTable caption={"Work one memory update by hand"} headers={[<>{"Position"}</>,<>{"Key features"}</>,<>{"Value"}</>,<>{"Contribution to "}<InlineMath>{"S"}</InlineMath>{""}</>,<>{"Contribution to "}<InlineMath>{"z"}</InlineMath>{""}</>]} rows={[[<>{"0"}</>,<>{""}<InlineMath>{"[1,0]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"2"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[2,0]^T"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[1,0]^T"}</InlineMath>{""}</>],[<>{"1"}</>,<>{""}<InlineMath>{"[0,1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"-1"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[0,-1]^T"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[0,1]^T"}</InlineMath>{""}</>],[<>{"2"}</>,<>{""}<InlineMath>{"[1,1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"3"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[3,3]^T"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[1,1]^T"}</InlineMath>{""}</>]]} />

<Prose>{"After all three records, "}<InlineMath>{"S=[5,2]^T"}</InlineMath>{" and "}<InlineMath>{"z=[2,2]^T"}</InlineMath>{". A query with features "}<InlineMath>{"[2,1]"}</InlineMath>{" produces numerator "}<InlineMath>{"12"}</InlineMath>{", denominator "}<InlineMath>{"6"}</InlineMath>{", and output "}<InlineMath>{"2"}</InlineMath>{"."}</Prose>

<Prose>{"Check it by explicitly comparing all keys: their similarities are "}<InlineMath>{"[2,1,3]"}</InlineMath>{", giving weights "}<InlineMath>{"[1/3,1/6,1/2]"}</InlineMath>{". The weighted values again sum to "}<InlineMath>{"2"}</InlineMath>{". The memory route has not approximated this feature-kernel operator."}</Prose>

<SparseFigure kind="memory" />

<SparseMemoryLab />

<H3>{"What “linear” does and does not mean"}</H3>

<Prose>{"For fixed "}<InlineMath>{"m,d_k,d_v"}</InlineMath>{", each write and read costs order "}<InlineMath>{"md_v"}</InlineMath>{", plus the feature-map cost. Across "}<InlineMath>{"L"}</InlineMath>{" positions, the core work is order "}<InlineMath>{"Lmd_v"}</InlineMath>{". Streaming state per head contains "}<InlineMath>{"m(d_v+1)"}</InlineMath>{" scalars. The state size is independent of how many records have arrived, but not independent of the feature width or value width."}</Prose>

<Prose>{"The output is generally "}<strong>{"nonlinear in the input"}</strong>{" because feature maps, normalization and the surrounding network are nonlinear. “Linear attention” refers to sequence-length scaling or the recurrent algebra, not a linear predictive model."}</Prose>

<Prose>{"A frequently used map is applied componentwise:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\phi(x)=\\operatorname{ELU}(x)+1=\n\\begin{cases}x+1&x\\ge0\\\\e^x&x<0.\\end{cases}"}</MathBlock></div>

<Prose>{"It defines its own similarity. It does not approximate "}<InlineMath>{"e^{q^Tk}"}</InlineMath>{" merely because it is positive. In this lesson's model, the map receives "}<InlineMath>{"q/d_k^{1/4}"}</InlineMath>{" and "}<InlineMath>{"k/d_k^{1/4}"}</InlineMath>{"; that scale is part of the declared model, not an identity turning ELU+1 into softmax."}</Prose>

<Prose>{"Strictly positive finite features give a positive denominator for a nonempty prefix in exact arithmetic. Merely nonnegative features can give zero overlap. Floating-point exponentials can also underflow. An implementation must detect or handle the problem; adding a denominator floor changes the mathematical operator where the floor is active. Our measured example stays far above its floor."}</Prose>

<H3>{"A compressed state can forget distinctions"}</H3>

<Prose>{"Let every key have one feature equal to 1. Histories with values "}<InlineMath>{"[1,3]"}</InlineMath>{" and "}<InlineMath>{"[2,2]"}</InlineMath>{" both create "}<InlineMath>{"S=4,z=2"}</InlineMath>{". Every positive query returns their mean, "}<InlineMath>{"2"}</InlineMath>{". No read of this state can answer “what was the first value?” differently for those histories."}</Prose>

<Prose>{"This is a concrete collision, not a claim that all useful information must be lost. More expressive key features can separate other histories. Positional features and gates can make writes depend on order. But a fixed-size state should be evaluated on the retrieval distinctions the task actually needs. A low mean prediction error on a smooth signal does not prove the ability to retrieve an arbitrary earlier identifier."}</Prose>

<Prose>{"For a pure additive state, permuting the "}<strong>{"already formed key-feature/value pairs"}</strong>{" leaves the final sum unchanged. That does not mean a whole causal network ignores order: prefix outputs differ, and adding position to inputs changes the pairs themselves. This distinction connects the algebra to the earlier positional-encoding lesson."}</Prose>

<H2>{"4. Approximate softmax with random features"}</H2>

<Prose>{"*Read the opening distinction on a first pass. The derivation and sampling details are a deeper branch.*"}</Prose>

<Prose>{"We can choose a different kernel deliberately, or approximate the existing exponential kernel. These are different experiments."}</Prose>

<Prose>{"Set "}<InlineMath>{"x=q/d_k^{1/4}"}</InlineMath>{", "}<InlineMath>{"y=k/d_k^{1/4}"}</InlineMath>{", so "}<InlineMath>{"x^Ty=q^Tk/\\sqrt{d_k}"}</InlineMath>{". Draw a fixed random vector "}<InlineMath>{"\\omega\\sim\\mathcal N(0,I)"}</InlineMath>{", and define"}</Prose>

<div className="neural-equation"><MathBlock>{"f_\\omega(x)=\\exp(\\omega^Tx-\\|x\\|^2/2)."}</MathBlock></div>

<Prose>{"The Gaussian identity "}<InlineMath>{"\\mathbb E[e^{\\omega^Ta}]=e^{\\|a\\|^2/2}"}</InlineMath>{" gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[f_\\omega(x)f_\\omega(y)]\n=e^{-\\|x\\|^2/2-\\|y\\|^2/2}e^{\\|x+y\\|^2/2}\n=e^{x^Ty}."}</MathBlock></div>

<Prose>{"With "}<InlineMath>{"m"}</InlineMath>{" sampled vectors, place "}<InlineMath>{"f_{\\omega_r}(x)/\\sqrt m"}</InlineMath>{" in feature coordinate "}<InlineMath>{"r"}</InlineMath>{". The feature dot product estimates the unnormalized exponential similarity. It can then use the same "}<InlineMath>{"S,z"}</InlineMath>{" recurrence. Positive random features and orthogonal constructions are central to "}<a href={"https://arxiv.org/html/2009.14794v4"}>{"Performer and FAVOR+"}</a>{"."}</Prose>

<Prose>{"The result is exact for the "}<strong>{"sampled feature operator"}</strong>{", approximate for the original softmax operator. Keep the projections fixed during a sequence. A state constructed under one set of random features cannot be read as though it had been constructed under another."}</Prose>

<H3>{"Unbiased similarities do not give an unbiased normalized output"}</H3>

<Prose>{"The expectation of a ratio is not generally the ratio of expectations. For a small counterexample, suppose two estimated similarities are equally likely to be "}<InlineMath>{"(1,1)"}</InlineMath>{" or "}<InlineMath>{"(3,1)"}</InlineMath>{". Their means are "}<InlineMath>{"(2,1)"}</InlineMath>{". The expected normalized first weight is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\tfrac12(1/2+3/4)=5/8,"}</MathBlock></div>

<Prose>{"while normalizing the mean similarities gives "}<InlineMath>{"2/3"}</InlineMath>{". An unbiased kernel estimator therefore does not automatically make a finite-feature attention row or output unbiased."}</Prose>

<Prose>{"For the independent Gaussian construction, a single pair has variance"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{Var}(\\widehat\\kappa_m(x,y))\n=\\frac{e^{2x^Ty}}{m}\\left(e^{\\|x+y\\|^2}-1\\right)."}</MathBlock></div>

<Prose>{"You can derive this by applying the same Gaussian identity to the square of "}<InlineMath>{"f_\\omega(x)f_\\omega(y)"}</InlineMath>{", then subtracting the squared mean. It explains both the "}<InlineMath>{"1/m"}</InlineMath>{" variance reduction and sensitivity to vector norms. It is a variance of the unnormalized pair estimator, not an identical formula for task loss or the normalized ratio. Increasing "}<InlineMath>{"m"}</InlineMath>{" improves this expectation-level quantity; one nested random draw can still become less accurate."}</Prose>

<H3>{"Orthogonal directions still need the right radii"}</H3>

<Prose>{"Using mutually orthogonal directions within a block can reduce redundant sampling. To preserve a standard Gaussian marginal for each row, combine a uniformly random orthogonal direction with an independent radius distributed as the length of a "}<InlineMath>{"d_k"}</InlineMath>{"-dimensional standard Gaussian vector. The supplied program obtains directions using a sign-corrected QR decomposition, then samples those independent radii. Independent blocks allow "}<InlineMath>{"m>d_k"}</InlineMath>{"."}</Prose>

<Prose>{"Multiplying every unit direction by the fixed number "}<InlineMath>{"\\sqrt{d_k}"}</InlineMath>{" is a different distribution. In one dimension, take "}<InlineMath>{"x=y=1"}</InlineMath>{". A fixed-radius vector is "}<InlineMath>{"+1"}</InlineMath>{" or "}<InlineMath>{"-1"}</InlineMath>{", so the expected feature product is "}<InlineMath>{"e^{-1}\\cosh2\\approx1.384"}</InlineMath>{"; the Gaussian construction gives "}<InlineMath>{"e\\approx2.718"}</InlineMath>{". Both are valid things to compute, but only the latter follows the Gaussian identity above. The Performer paper also studies a fixed-radius regularized kernel; it should be named as such, not described as an unchanged Gaussian estimator."}</Prose>

<H3>{"Keep the numerical stabilization consistent"}</H3>

<Prose>{"Exponentials may overflow. Multiplying all feature coordinates for one query by a common scalar cancels between that query's numerator and denominator. Multiplying every key-feature vector in the whole memory by the "}<strong>{"same"}</strong>{" scalar also cancels. Multiplying each key by its own unrelated scalar usually changes their relative importance."}</Prose>

<Prose>{"For streaming exponential features, if a newly arrived key requires changing the common key scale, rescale the existing "}<InlineMath>{"S"}</InlineMath>{" and "}<InlineMath>{"z"}</InlineMath>{" by the same factor before adding the new write. Otherwise old and new records use different units. Padding, state reset, query/key feature scaling, and causal prefix boundaries must agree between training and inference."}</Prose>

<SparseFigure kind="random" /><SparseRandomLab />

<H2>{"5. Compress the sequence instead of its feature sums"}</H2>

<Prose>{"Another strategy replaces many keys and values with fewer summary slots before attention. It deserves its own picture because it is not the same operation as skipping keys or accumulating "}<InlineMath>{"\\phi(k)v^T"}</InlineMath>{"."}</Prose>

<H3>{"Linformer: learned combinations along the length axis"}</H3>

<Prose>{"Let "}<InlineMath>{"K\\in\\mathbb R^{L\\times d_k}"}</InlineMath>{", "}<InlineMath>{"V\\in\\mathbb R^{L\\times d_v}"}</InlineMath>{". Define learned sequence projections "}<InlineMath>{"E,F\\in\\mathbb R^{r\\times L}"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\bar K=EK,\\quad\\bar V=FV,\\quad\nO=\\operatorname{softmax}(Q\\bar K^T/\\sqrt{d_k})\\bar V."}</MathBlock></div>

<Prose>{"Each row of "}<InlineMath>{"E"}</InlineMath>{" combines positions into a key summary; each row of "}<InlineMath>{"F"}</InlineMath>{" combines positions into a value summary. Learned coefficients need not be nonnegative or sum to one. The model attends over "}<InlineMath>{"r"}</InlineMath>{" summaries. With fixed "}<InlineMath>{"r\\ll L"}</InlineMath>{", projection and attention cost order "}<InlineMath>{"Lr(d_k+d_v)"}</InlineMath>{". The original work motivates this using empirical low-rank behavior of attention maps and approximation arguments; it does not establish that every query/key matrix or task has a universally small useful rank. "}<a href={"https://arxiv.org/html/2006.04768v3"}>{"Linformer, §3–4"}</a>{""}</Prose>

<Prose>{"Now inspect causality. With one value-summary row "}<InlineMath>{"F=[0.5,0,0.5]"}</InlineMath>{" and values "}<InlineMath>{"[1,2,9]"}</InlineMath>{", the summary is "}<InlineMath>{"5"}</InlineMath>{". There is only one summary slot, so its softmax weight is one. A query at position 0 would receive "}<InlineMath>{"5"}</InlineMath>{", containing future value 9. Changing that future value to 1 changes the supposedly earlier output to 1. Applying a triangular mask to the single summary slot cannot remove the particular future contribution already mixed into it."}</Prose>

<SparseFigure kind="projection" />

<Prose>{"A causal variant can be constructed, but must define a different prefix-dependent operator. For fixed projection columns "}<InlineMath>{"e_j,f_j\\in\\mathbb R^r"}</InlineMath>{", maintain"}</Prose>

<div className="neural-equation"><MathBlock>{"\\bar K_t=\\bar K_{t-1}+e_tk_t^T,\\qquad\n\\bar V_t=\\bar V_{t-1}+f_tv_t^T."}</MathBlock></div>

<Prose>{"The current query attends only to these summaries of positions through "}<InlineMath>{"t"}</InlineMath>{". In the one-slot example, position 0 sees "}<InlineMath>{"0.5"}</InlineMath>{", not the full-sequence 5. The coefficient has not automatically become a prefix-normalized average. Define how unused summary rows participate, how columns are generated beyond the trained length, and whether coefficients depend on future inputs. This construction shows why “the usual full-sequence projection leaks” is precise, while “sequence projection can never be causal” is too strong."}</Prose>

<SparseProjectionLab />

<H3>{"Nyströmformer: use landmark queries and keys"}</H3>

<Prose>{"Nyströmformer selects a smaller set of representative query/key vectors, called "}<strong>{"landmarks"}</strong>{". One option takes means of contiguous segments. Here "}<InlineMath>{"d=d_k"}</InlineMath>{" is query/key width, and the symbols "}<InlineMath>{"F,A,B"}</InlineMath>{" name new factors local to this construction. Define row-softmax matrices"}</Prose>

<div className="neural-equation"><MathBlock>{"F=\\operatorname{softmax}(Q\\tilde K^T/\\sqrt d),\\quad\nA=\\operatorname{softmax}(\\tilde Q\\tilde K^T/\\sqrt d),\\quad\nB=\\operatorname{softmax}(\\tilde QK^T/\\sqrt d)."}</MathBlock></div>

<Prose>{"Approximate the attention output by "}<InlineMath>{"FA^+BV"}</InlineMath>{", where "}<InlineMath>{"A^+"}</InlineMath>{" is the Moore–Penrose pseudoinverse. Compute from the right to avoid forming an "}<InlineMath>{"L\\times L"}</InlineMath>{" matrix. The paper uses an iterative inverse approximation; the teaching fixture uses a direct numerical pseudoinverse. Pseudoinverse coefficients can be negative, so the approximate full matrix is not automatically a nonnegative probability matrix. Full-sequence landmarks also need a separate causal design before use in autoregressive prediction. "}<a href={"https://arxiv.org/html/2102.03902v3"}>{"Nyströmformer, §3"}</a>{""}</Prose>

<SparseFigure kind="nystrom" />

<Prose>{"For "}<InlineMath>{"r"}</InlineMath>{" landmarks, a straightforward implementation pays order "}<InlineMath>{"Lrd_k+Lrd_v+r^2d_v+r^3"}</InlineMath>{", including an SVD-based pseudoinverse. “Linear in "}<InlineMath>{"L"}</InlineMath>{"” assumes "}<InlineMath>{"r"}</InlineMath>{" is held fixed; increasing landmarks to preserve quality changes that tradeoff. Singular values near zero make the pseudoinverse sensitive, so approximation and numerical error must be examined together."}</Prose>

<H2>{"6. Make the graph efficient on the actual machine"}</H2>

<Prose>{"*Deeper practical branch; the real-data core continues in §7.*"}</Prose>

<H3>{"Eight edges can occupy very different amounts of work"}</H3>

<Prose>{"A hardware kernel often processes tiles instead of individual matrix cells. In an "}<InlineMath>{"8\\times8"}</InlineMath>{" mask with "}<InlineMath>{"2\\times2"}</InlineMath>{" tiles, eight diagonal edges occupy four tiles: 16 candidate cell positions inside those tiles. Place one edge per row at columns "}<InlineMath>{"2i\\bmod8"}</InlineMath>{", and the same eight edges occupy eight tiles: 32 candidate positions. Both masks have 12.5% token-edge density. They have different tile occupancy and memory access patterns. This example is a bidirectional layout exercise, not a causal mask."}</Prose>

<Prose>{"An occupied tile may still contain masked cells; a fully empty tile can be skipped. Neither count alone predicts seconds. Gather overhead, head dimensions, precision, hardware, compilation and other layers matter. A dense implementation of a sparse mask still computes the dense scores if it forms "}<code>{"Q @ K.T"}</code>{" first."}</Prose>

<SparseTilesLab />

<Prose>{"PyTorch's FlexAttention provides a way to express custom score modifications and block masks and compile suitable attention kernels. Its "}<code>{"mask_mod"}</code>{" receives batch, head, query index and key index and returns whether that pair is allowed. A block mask can skip fully masked blocks. This does not mean every arbitrary mask is equally fast, or that an unsupported device will execute the same compiled path. "}<a href={"https://pytorch.org/blog/flexattention/"}>{"FlexAttention introduction and examples"}</a>{", "}<a href={"https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html"}>{"current API"}</a>{""}</Prose>

<H3>{"Modern sparse systems also pay to choose the reads"}</H3>

<Prose>{"The field has continued beyond the early fixed patterns. These examples are architecture snapshots checked during 13–27 September 2026, not a leaderboard."}</Prose>

<Prose>{""}<strong>{"Native Sparse Attention (NSA)"}</strong>{" combines three separately normalized branches: compressed blocks, selected fine-grained blocks, and a local window. Compressed-attention scores help select important blocks, with selection shared across grouped heads. Learned sigmoid gates combine branch outputs; their sum is not required to be one. The three-branch design preserves local access while learning coarser and selected long-range reads. With a fixed compression stride, the compressed branch still grows with the number of compressed positions, so a fixed selection budget alone does not establish linear total work. "}<a href={"https://arxiv.org/html/2502.11089v1"}>{"NSA, §3"}</a>{""}</Prose>

<Prose>{""}<strong>{"DeepSeek-V3.2-Exp's DSA"}</strong>{" uses a small lightning indexer to select past positions for its main MLA read. The indexer aggregates query-head scores of the form "}<InlineMath>{"w_{tj}\\operatorname{ReLU}((q^I_{tj})^Tk^I_s)"}</InlineMath>{". It is first trained against a dense attention-derived target, then the main sparse model is adapted. Top-"}<InlineMath>{"k"}</InlineMath>{" indices are discrete; ordinary backpropagation through their integer selection is not the indexer's training method. The report explicitly distinguishes order "}<InlineMath>{"Lk"}</InlineMath>{" main attention from the indexer's still-quadratic sequence comparison, albeit with a smaller cost. "}<a href={"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf"}>{"V3.2-Exp technical report, §1–3"}</a>{""}</Prose>

<Prose>{""}<strong>{"DeepSeek-V4.1's CSA2"}</strong>{" distinguishes layers that build and index new compressed memory, layers that reuse memory but issue fresh index queries, and layers that reuse both memory and selected indices. Its hierarchical decoder indexer obtains a candidate pool from an initial full scan; later reindexing searches that pool. Local sliding-window memory remains a separate source. The architecture's causal encoder–decoder division changes which layer supplies the global memory, so memory reuse and selection reuse must not be conflated. These are separate mechanisms from NSA's three gated outputs. "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf"}>{"V4.1 report, §2.2–2.3"}</a>{""}</Prose>

<Prose>{""}<strong>{"Candidate discovery"}</strong>{", "}<strong>{"main selected reads"}</strong>{", "}<strong>{"local reads"}</strong>{", and "}<strong>{"cross-layer reuse"}</strong>{" incur separate costs, as the connected paths below show. It does not invent a speedup from an edge count. Current "}<a href={"https://github.com/deepseek-ai/FlashMLA"}>{"FlashMLA source documentation"}</a>{" provides concrete examples of dense and sparse kernels with architecture-specific formats; a cache format or benchmark cannot be transplanted unchanged between model versions."}</Prose>

<SparseFigure kind="architectures" />

<H2>{"7. Compare three operators on observed hand trajectories"}</H2>

<Prose>{"An abstract graph tells us whether an effect is possible. A trained example tells us what a particular learned system actually does. We will forecast the next two-dimensional point of an observed hand movement using the openly licensed "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI Libras Movement dataset"}</a>{", credited to Dias, Peres and Bíscaro, under CC BY 4.0."}</Prose>

<Prose>{"The source has 360 trajectories of 45 points and 15 movement labels. Exact coordinate duplicates reduce it to 330 unique trajectories; duplicate labels agree. We retain the first row in each exact-duplicate group "}<strong>{"before splitting"}</strong>{", so identical trajectories do not appear on both sides of the evaluation. A fixed classwise seed-73 split gives 220 training, 50 validation and 60 test trajectories. Movement labels are used only for stratification, not as model inputs. Positions 0–43 predict 1–44, using fixed input scaling "}<InlineMath>{"2x-1"}</InlineMath>{"."}</Prose>

<Prose>{"The source describes four performers and two recording sessions but does not supply row-level performer/session identifiers. This split tests held-out trajectories, not new-person or new-session generalization. The processed coordinate records also do not validate a live image-to-coordinate pipeline. This familiar dataset is intentionally reused from earlier lessons; the new test comparison is not an independent new benchmark across those lessons."}</Prose>

<H3>{"Hold the experiment fixed"}</H3>

<Prose>{"Each model has 4,946 learned parameters: a 2→24 input map, fixed sinusoidal position addition, one pre-norm block with three 8-wide heads, a 24→48→24 GELU FFN, final normalization and a two-coordinate output. There is no dropout or weight decay. The only operator change is dense causal softmax, a five-key causal softmax window, or normalized ELU+1 feature attention. Identical seed-137 initialization and 160 full-batch Adam updates at learning rate 0.003 give an equal-budget local comparison."}</Prose>

<Prose>{"Each model selects its lowest validation MSE, taking the earliest exact tie. All selected update 160, the budget endpoint. We did not extend the run to make a preferred outcome emerge, and this result does not establish convergence. A persistence baseline predicts the current point as the next; a six-parameter affine baseline is fitted using training pairs only. RMSE pools both coordinates and all next-point errors in each split, in the source's original coordinate units."}</Prose>

<NeuralTable caption={"Hold the experiment fixed"} headers={[<>{"Predictor"}</>,<>{"Training RMSE"}</>,<>{"Validation RMSE"}</>,<>{"Test RMSE"}</>]} rows={[[<>{"Persistence"}</>,<>{"0.027529"}</>,<>{"0.026304"}</>,<>{"0.026458"}</>],[<>{"Training-fitted affine"}</>,<>{"0.027334"}</>,<>{"0.026154"}</>,<>{"0.026222"}</>],[<>{"Dense causal softmax"}</>,<>{"0.024152"}</>,<>{"0.023358"}</>,<>{"0.023219"}</>],[<>{"Five-key causal window"}</>,<>{"0.025147"}</>,<>{"0.023907"}</>,<>{"0.024684"}</>],[<>{"ELU+1 feature kernel"}</>,<>{"0.026106"}</>,<>{"0.025327"}</>,<>{"0.024680"}</>]]} />

<Prose>{"All three learned operators improve on these simple baselines in this run. Dense attention has the lowest test RMSE here. The window and kernel test results differ by only about "}<InlineMath>{"0.0000041"}</InlineMath>{", far too little to promote into an architectural verdict from a single split and seed. There is no measured GPU timing comparison in this table."}</Prose>

<H3>{"Look at one actual prefix and a controlled edit"}</H3>

<Prose>{"The worked example uses source row 77, first 32 observed points. The true next point is approximately "}<InlineMath>{"(0.593810,0.250000)"}</InlineMath>{". Keep the weights fixed and reflect the x coordinate at zero-based frame 23 around 0.5: "}<InlineMath>{"x\\mapsto1-x"}</InlineMath>{". This is an artificial sensitivity intervention on a real recorded trajectory, not another observed movement."}</Prose>

<NeuralTable caption={"Look at one actual prefix and a controlled edit"} headers={[<>{"Operator"}</>,<>{"Original next-point forecast"}</>,<>{"Forecast after editing frame 23"}</>]} rows={[[<>{"Dense"}</>,<>{""}<InlineMath>{"(0.598713,0.250658)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(0.601801,0.252266)"}</InlineMath>{""}</>],[<>{"Window"}</>,<>{""}<InlineMath>{"(0.609244,0.247768)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(0.609244,0.247768)"}</InlineMath>{""}</>],[<>{"Feature kernel"}</>,<>{""}<InlineMath>{"(0.601817,0.248507)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(0.603256,0.248870)"}</InlineMath>{""}</>]]} />

<Prose>{"The window result is an exact null in this model: at final input position 31, its single attention layer reads positions 27–31, and all other operations are tokenwise. Frame 23 has no route to that output. The other two models change modestly; a global path does not require a dramatic response. In all three models, outputs before the edited position remain exactly unchanged, as causality requires."}</Prose>

<SparseFigure kind="trajectory" />

<Prose>{"At this 32-point prefix, float32 numeric attention payloads are 6,144 bytes for dense K/V, 960 for window K/V, and 864 for kernel "}<InlineMath>{"S,z"}</InlineMath>{". These exclude weights, outputs, position counters, index metadata and allocator overhead. They are not peak memory measurements. Dense and window incremental forecasts agree with their full-prefix computation to below "}<InlineMath>{"2.4\\times10^{-7}"}</InlineMath>{" maximum absolute error in transformed output coordinates. The kernel's recurrent and explicit pairwise implementations agree below "}<InlineMath>{"2.7\\times10^{-7}"}</InlineMath>{". Separate float64 checks give whole-network gradient agreement below "}<InlineMath>{"3.6\\times10^{-15}"}</InlineMath>{" for this small checked input."}</Prose>

<SparseForecastLab />

<H3>{"Approximate one trained dense head without retraining it"}</H3>

<Prose>{"Take the selected dense model's actual head-0 queries, keys and values from the worked prefix. Compare exact softmax output against positive independent Gaussian features, using seeds 0–7 and nested feature counts 16, 64 and 256. For each run, measure"}</Prose>

<div className="neural-equation"><MathBlock>{"\\text{relative output error}=\\frac{\\|\\widehat O-O\\|_F}{\\|O\\|_F}."}</MathBlock></div>

<Prose>{"The denominator is nonzero for this saved head. The metric compares the whole prefix's head outputs, not model RMSE or probabilities of a downstream label."}</Prose>

<NeuralTable caption={"Approximate one trained dense head without retraining it"} headers={[<>{"Features"}</>,<>{"Mean error over eight seeds"}</>,<>{"Smallest–largest observed error"}</>]} rows={[[<>{"16"}</>,<>{"0.043293"}</>,<>{"0.021708–0.077060"}</>],[<>{"64"}</>,<>{"0.031098"}</>,<>{"0.016960–0.042411"}</>],[<>{"256"}</>,<>{"0.017026"}</>,<>{"0.011095–0.022679"}</>]]} />

<Prose>{"The average falls, but individual nested draws do not all improve at every step. Seed 5 worsens from about 0.02171 to 0.02813 when going from 16 to 64 features. Seed 6 worsens slightly from 64 to 256. Keep those outcomes in the plot. These are operator approximations of a fixed dense head, not results from training a Performer, and not a worst-case guarantee."}</Prose>

<H3>{"Run the complete teaching programs"}</H3>

<Prose>{"The packet includes the original small data files, "}<a href={"/learn-code/sparse-linear-attention-variants/provenance.md"}>{"data attribution and provenance"}</a>{", "}<a href={"/learn-code/sparse-linear-attention-variants/author-calculations.py"}>{"the complete CPU training and evaluation program"}</a>{", "}<a href={"/learn-code/sparse-linear-attention-variants/mechanism-calculations.py"}>{"mechanism calculations"}</a>{", and the actual saved models/results. Use Python with NumPy and PyTorch in your own environment; the recorded execution used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, one CPU thread."}</Prose>

<CodeBlock language={"bash"}>{"python -m venv .venv\n# Activate .venv using your shell's normal activation command.\npython -m pip install numpy==2.3.5 torch==2.14.0\npython mechanism-calculations.py\npython author-calculations.py"}</CodeBlock>

<Prose>{"Run from the downloaded packet directory. The programs use local data and perform no network access. The mechanism program checks sparse gathered/masked equality, graph reach, feature-state reads and the fresh numerical cases. The training program supplies data loading, duplicate handling, split construction, all model layers, training, validation selection, baselines, incremental evaluation, gradient checks, saved forecasts and the 24 random-feature trials. Every named helper is included in the downloadable source."}</Prose>

<SparseProgram file="author-calculations.py" title="Read the complete trainable PyTorch models and retained study" /><SparseProgram file="mechanism-calculations.py" title="Read the independent numerical mechanisms and samplers" />

<Prose>{"Expected recorded model lines are dense/window/kernel selecting update 160 with the RMSE values above; each program ends with a named PASS line. Different supported software or hardware can change last digits. If you alter the data, seed or protocol, label the new result as a new experiment. Training here computes small dense reference masks and, for kernel attention, stores prefix states for automatic differentiation. The code teaches the operator and verifies its recurrence; it is not a high-performance sparse GPU kernel or a constant-memory training implementation."}</Prose>

<Prose>{"For a minimal streaming implementation you can inspect the following complete NumPy example. Inputs are already feature vectors, so it cleanly separates the recurrence from the choice of feature map."}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\ndef causal_feature_attention(query_features, key_features, values):\n    q = np.asarray(query_features, dtype=float)\n    k = np.asarray(key_features, dtype=float)\n    v = np.asarray(values, dtype=float)\n    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or len(v) != len(q):\n        raise ValueError(\"Expected matching (length, features) Q/K and (length, values) V.\")\n    if not all(np.isfinite(x).all() for x in (q, k, v)) or (q < 0).any() or (k < 0).any():\n        raise ValueError(\"Use finite nonnegative features and finite values.\")\n    memory = np.zeros((k.shape[1], v.shape[1]))\n    normalizer = np.zeros(k.shape[1])\n    outputs = []\n    for query, key, value in zip(q, k, v):\n        memory += np.outer(key, value)\n        normalizer += key\n        denominator = query @ normalizer\n        if denominator <= 0:\n            raise ValueError(\"No positive overlap with this prefix.\")\n        outputs.append(query @ memory / denominator)\n    return np.asarray(outputs)\n\nq = [[2, 1], [2, 1], [2, 1]]\nk = [[1, 0], [0, 1], [1, 1]]\nv = [[2], [-1], [3]]\nprint(causal_feature_attention(q, k, v).ravel())\n# [2. 1. 2.]"}</CodeBlock>

<H3>{"Implement the other compression choices, not just name them"}</H3>

<Prose>{"The earlier program owns the trained causal dense/window/kernel comparison. The additional "}<a href={"/learn-code/sparse-linear-attention-variants/attention_compression_bridges.py"}>{"sequence-compression program"}</a>{" opens the bidirectional Linformer and Nyström operations from §5 and supplies an actual gathered-window route. It requires only PyTorch; run "}<code>{"python attention_compression_bridges.py"}</code>{"."}</Prose>

<SparseProgram file="attention_compression_bridges.py" title="Read the complete scratch and SDPA compression bridges" />

<Prose>{""}<code>{"linformer"}</code>{" owns two learned length-axis matrices E and F. It computes EK and FV first, then runs attention over those compressed slots. Its normal tool route calls SDPA on the same compressed arrays. E/F receive gradients alongside Q/K/V; a learned summary is not a fixed downsampling label. The program compares values and all five gradients under identical initial arrays. It intentionally has "}<strong>{"no causal mask"}</strong>{": making a full-sequence summary and applying a later triangle cannot remove future information already mixed into the summary."}</Prose>

<Prose>{""}<code>{"nystrom"}</code>{" forms segment-mean query/key landmarks. Seven positions split into three segments retain the trailing positions. It constructs the three softmax factors, uses a tolerance-controlled pseudoinverse for the small middle matrix, and multiplies from the value side: "}<code>{"front @ (pinv(middle) @ (back @ V))"}</code>{". It never materializes the L×L approximate weight matrix. The cost includes O(L r d) pair/factor work and O(r³) pseudoinversion, with O(Lr+r²) factor storage, in addition to the feature/value widths. A pseudoinverse is a well-defined tool here; implementing SVD again would repeat "}<a href={"/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu?module=math-foundations"}>{"Matrix Decompositions"}</a>{". Near a rank threshold, derivatives can be sensitive: changing "}<code>{"rtol"}</code>{" changes which directions are retained and must be treated as a model/numerical decision."}</Prose>

<Prose>{""}<code>{"gathered_window"}</code>{" only scores the keys actually in a causal window: O(L W (d_k+d_v)) arithmetic and O(W) temporary scores per query, beyond inputs and outputs. Its SDPA comparison deliberately uses a dense mask as an independent semantic reference, "}<strong>{"not"}</strong>{" as evidence of sparse execution. The maintained tool takes responsibility for backend selection; a genuinely sparse accelerator path needs a kernel supporting the chosen block pattern."}</Prose>

<Prose>{"The author ran these small CPU float64 probes: Linformer and gathered-window maximum API differences were each 1.11e-16; using every position as a Nyström landmark reproduced the dense result to 1.45e-15. Three landmarks gave maximum output error 0.25091 for this declared random fixture. That last number is a single approximation example, not a general error guarantee or a trained accuracy result. The random-feature Gaussian-marginal sampling and stabilizations remain owned by "}<code>{"mechanism-calculations.py"}</code>{"; they are different approximations from these learned/landmark summaries."}</Prose>

<Prose>{""}<strong>{"Take control."}</strong>{" Change sequence length to 11, use four landmarks and a window of width 1. Inspect both output and gradient checks. Then reduce "}<code>{"rtol"}</code>{" for nearly duplicate landmarks."}</Prose>

<details><summary>Hint</summary>Width one must return each position's own V. Unequal segment lengths are allowed; the inverse problem remains small.</details>

<details><summary>Solution and success criteria</summary>The window output equals V and has zero Q/K derivative. Linformer's manual/API equality should remain, while approximation quality is a separate measured quantity. Nyström with fewer landmarks need not improve monotonically for each input as count increases. Near duplicate landmarks, record singular values and chosen tolerance before interpreting a large gradient; smaller tolerance is not automatically a better model.</details>

<H2>{"8. Deeper connections and practical judgment"}</H2>

<H3>{"Backpropagation through a recurrent summary"}</H3>

<Prose>{"Forward equivalence is not enough if the two training paths differentiate different calculations. For a local read, write "}<InlineMath>{"a=\\phi(q)"}</InlineMath>{", numerator "}<InlineMath>{"n=S^Ta"}</InlineMath>{", denominator "}<InlineMath>{"b=a^Tz>0"}</InlineMath>{", and "}<InlineMath>{"o=n/b"}</InlineMath>{". If the arriving output gradient is "}<InlineMath>{"g=\\partial\\mathcal L/\\partial o"}</InlineMath>{", ordinary quotient differentiation gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial\\mathcal L}{\\partial S}=\\frac{ag^T}{b},\\qquad\n\\frac{\\partial\\mathcal L}{\\partial z}=-\\frac{a(g^To)}{b},\\qquad\n\\frac{\\partial\\mathcal L}{\\partial a}=\\frac{Sg-z(g^To)}{b}."}</MathBlock></div>

<Prose>{"In a causal sequence, a write at position "}<InlineMath>{"j"}</InlineMath>{" influences every later state. Its gradient therefore collects contributions from reads "}<InlineMath>{"t\\ge j"}</InlineMath>{", which can be accumulated by a reverse scan. For its direct outer-product contribution, if the accumulated matrix gradient is "}<InlineMath>{"G_j"}</InlineMath>{", then the value gradient includes "}<InlineMath>{"G_j^T\\phi(k_j)"}</InlineMath>{"; the key-feature gradient includes "}<InlineMath>{"G_jv_j"}</InlineMath>{" plus the normalizer contribution. The chain rule then differentiates the feature map and input projections."}</Prose>

<Prose>{"You can parallelize prefix operations or process chunks, but the memory layout and backward strategy still matter. A naive "}<code>{"cumsum"}</code>{" over all outer products stores an "}<InlineMath>{"L\\times m\\times d_v"}</InlineMath>{" tensor. A custom scan/recomputation method can trade storage for work. Our saved all-parameter gradient comparison checks the defined small model, including surrounding normalization and FFN, rather than checking only one isolated final sum."}</Prose>

<H3>{"Gates and delta updates change the memory, not just its speed"}</H3>

<Prose>{"The additive update "}<InlineMath>{"S_t=S_{t-1}+k_tv_t^T"}</InlineMath>{" retains every write with equal temporal persistence. A decay changes it to "}<InlineMath>{"S_t=\\gamma_tS_{t-1}+k_tv_t^T"}</InlineMath>{", so earlier contributions are multiplied by later decay factors. Featurewise gates allow different parts of memory to forget differently."}</Prose>

<Prose>{"A delta-style write instead uses the current prediction error for the key, for example"}</Prose>

<div className="neural-equation"><MathBlock>{"S_t=S_{t-1}+\\beta_t k_t(v_t-S_{t-1}^Tk_t)^T."}</MathBlock></div>

<Prose>{"For a unit key and "}<InlineMath>{"\\beta_t=1"}</InlineMath>{", the new read at that key is exactly "}<InlineMath>{"v_t"}</InlineMath>{": multiplying by "}<InlineMath>{"k_t^T"}</InlineMath>{" cancels the old prediction error. It behaves like correcting a stored association, not merely adding another copy. With a nonunit key, the same conclusion does not follow without adjusting the update. This explains why a family can have linear sequence scaling but different overwrite, normalization and retrieval behavior."}</Prose>

<Prose>{"The earlier "}<a href={"/learn/path/full-curriculum/rwkv-linear-attention-models?module=deep-learning-fundamentals"}>{"RWKV & Linear Attention Models"}</a>{" and "}<a href={"/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals"}>{"State Space Models"}</a>{" own the versioned recurrent mechanisms and selective dynamics. A signed recurrence or an mLSTM normalization is not automatically a positive normalized feature kernel, and none becomes exact row-softmax merely because matrix products can be reassociated. Use the update, read and normalization equations to classify a new model."}</Prose>

<H3>{"Position and masking are algebraic constraints"}</H3>

<Prose>{"A feature recurrence works when each write can be computed from the current/past information and each read uses the appropriate state. An arbitrary pairwise relative-position bias need not factor into a fixed-size query/key state. Some distance factors do: a scalar exponential decay corresponds to the recurrent weighting above. Other position schemes require extra features or a changed kernel. Rotating vectors before a nonlinear feature map does not establish the same relative-position identity as rotating the dot-product vectors; check the resulting similarity explicitly."}</Prose>

<Prose>{"Packed documents need state resets or segmented scans at document boundaries. A single global sum over an entire batch of packed text leaks information across samples. During decoding, state updates must match whether the current token is included, the prompt prefix already processed, and the attention layer's positional convention. These are part of the operator, not cosmetic bookkeeping."}</Prose>

<H3>{"Choose an experiment that can disprove your idea"}</H3>

<NeuralTable caption={"Choose an experiment that can disprove your idea"} headers={[<>{"Task need"}</>,<>{"Candidate to investigate"}</>,<>{"A revealing failure test"}</>]} rows={[[<>{"Predict a locally smooth signal"}</>,<>{"Window or recurrent summary, alongside simple baselines"}</>,<>{"Insert a relevant remote change; test new entities, not duplicate rows"}</>],[<>{"Retrieve an earlier exact identifier"}</>,<>{"Individual-key access, possibly selected or hybrid"}</>,<>{"Move the target, add distractors, vary delay and compare missed candidates"}</>],[<>{"Classify a complete structured document"}</>,<>{"Local/global graph or summaries"}</>,<>{"Remove structure labels; vary document length and cross-section dependencies"}</>],[<>{"Compress an existing softmax model"}</>,<>{"Approximation plus adaptation, with exact reference"}</>,<>{"Compare operator error, final task loss and required retraining separately"}</>],[<>{"Serve a long causal prompt"}</>,<>{"Measure prefill, decode, cache and selection costs"}</>,<>{"Include routing/indexing and batch/concurrency effects, not only kernel arithmetic"}</>]]} />

<Prose>{"An interesting scientific use follows from the graph picture. In a DNA sequence, a local motif and a distant regulatory context may both matter. A sparse graph can preserve cheap local comparisons while adding routes between distant regions. But graph connectivity alone does not establish biological relevance; evaluation must preserve meaningful held-out sequence/entity boundaries. Similarly, a protein's amino-acid sequence is one-dimensional while its interactions can be distant along that sequence. Efficient attention offers a way to examine longer contexts; a token-level prediction score is not itself a validated three-dimensional structure or function prediction. The research examples in the references are motivations for careful task design, not permission to infer such downstream capabilities from our hand-motion experiment."}</Prose>

<Prose>{"Retrieval before the model can reduce how much context enters it; efficient attention changes how the supplied context is processed. Neither universally replaces the other. A useful system may use both, and its evaluation should include evidence that retrieval did not discard the needed information."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"9. Practice and transfer"}</H2>

<Prose>{"Work these problems before opening the optional hints or solutions. They use different values and positions from both the worked explanations and the initial investigations."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Count the real connections"}</H3>

<Prose>{"A causal sequence has 20 positions and a window of four total keys including self. How many legal query/key pairs exist in one head? What is the farthest possible input distance after three such layers, assuming all other operations are tokenwise? Does reaching that distance guarantee a useful learned dependence?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Count the growing first rows separately. A single layer moves information at most three positions."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The first four rows contribute "}<InlineMath>{"1+2+3+4=10"}</InlineMath>{"; the next 16 contribute 64, totaling "}<strong>{"74"}</strong>{". Three layers can span at most "}<strong>{"9"}</strong>{" positions. This is possible information flow, not a guarantee that the learned weights preserve or use the information."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Compute a summary read"}</H3>

<Prose>{"Two key-feature vectors are "}<InlineMath>{"[1,2]"}</InlineMath>{" and "}<InlineMath>{"[3,1]"}</InlineMath>{", with scalar values "}<InlineMath>{"-2"}</InlineMath>{" and "}<InlineMath>{"4"}</InlineMath>{". The query features are "}<InlineMath>{"[2,1]"}</InlineMath>{". Compute "}<InlineMath>{"S,z"}</InlineMath>{", the two normalized weights and the output. Then change the second value to "}<InlineMath>{"-2"}</InlineMath>{" without changing any key or query."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The unnormalized key similarities are 4 and 7. A value edit changes the numerator, not the denominator."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{""}<InlineMath>{"S=[10,0]^T,z=[4,3]^T"}</InlineMath>{". The denominator is 11, weights are "}<InlineMath>{"4/11,7/11"}</InlineMath>{", and output is "}<strong>{""}<InlineMath>{"20/11\\approx1.81818"}</InlineMath>{""}</strong>{". With both values "}<InlineMath>{"-2"}</InlineMath>{", every normalized weighted average equals "}<strong>{""}<InlineMath>{"-2"}</InlineMath>{""}</strong>{". This is a useful null even if you later change the query."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Diagnose a misleading speed claim"}</H3>

<Prose>{"An implementation forms a "}<InlineMath>{"4096\\times4096"}</InlineMath>{" score matrix, then masks all but 64 keys per row and calls softmax. Its author says “only 64 keys are visible, so the matrix multiplication is linear in sequence length.” Identify the error and propose a correctness check for a genuinely gathered implementation."}</Prose>

<details><summary>Solution</summary>

<Prose>{"The full matrix multiplication has already computed all pairs; a later mask does not undo that work. Gather the legal key/value rows before comparison or use a kernel that skips masked blocks. For small fixed Q/K/V and a nonempty mask per row, compare the gathered outputs to a dense reference with illegal scores set to negative infinity. Test causal boundaries and a mask with uneven row lengths. Matching output establishes the specified sparse operator, not a speedup; timing requires the actual target implementation."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. A future value inside a summary"}</H3>

<Prose>{"One sequence-summary row has coefficients "}<InlineMath>{"[0.2,0.3,0.5]"}</InlineMath>{", values "}<InlineMath>{"[5,0,6]"}</InlineMath>{", and only one attention slot. At query position 1, compare the full-sequence summary with the prefix-only summary. Change the last value to "}<InlineMath>{"-2"}</InlineMath>{". Which earlier output should remain invariant in a causal implementation?"}</Prose>

<details><summary>Solution</summary>

<Prose>{"The full summary is "}<strong>{"4"}</strong>{", then becomes "}<strong>{"0"}</strong>{" after the future edit. The prefix-only summary is "}<strong>{"1"}</strong>{" before and after. These coefficients are not renormalized over the prefix. The full summary leaks; masking its sole slot cannot selectively remove the future component."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Compare the right memory quantities"}</H3>

<Prose>{"For batch one, four heads, "}<InlineMath>{"d_k=d_v=16"}</InlineMath>{", "}<InlineMath>{"L=2048"}</InlineMath>{" and float32 state, calculate dense K/V payload, a 32-key window payload, and feature state with "}<InlineMath>{"m=24"}</InlineMath>{". Exclude metadata and weights. Which calculation tells you peak training memory?"}</Prose>

<details><summary>Solution</summary>

<Prose>{"Dense: "}<InlineMath>{"4\\times2048\\times(16+16)\\times4=\\mathbf{1,048,576}"}</InlineMath>{" bytes. Window: "}<InlineMath>{"4\\times32\\times32\\times4=\\mathbf{16,384}"}</InlineMath>{" bytes. Kernel state: "}<InlineMath>{"4\\times24\\times(16+1)\\times4=\\mathbf{6,528}"}</InlineMath>{" bytes. "}<strong>{"None"}</strong>{" gives peak training memory; gradients, saved activations, optimizer state and temporary computations are additional quantities."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Test a random-feature claim"}</H3>

<Prose>{"Someone reports that their kernel similarities are unbiased, so “the average normalized attention output must be exactly the original output.” Construct a two-outcome counterexample different from the one in §4. What else should be recorded when showing an error-versus-feature-count plot?"}</Prose>

<details><summary>One solution</summary>

<Prose>{"Let similarity estimates be equally likely "}<InlineMath>{"(2,1)"}</InlineMath>{" or "}<InlineMath>{"(6,1)"}</InlineMath>{", with values 1 and 0. The expected output is "}<InlineMath>{"(2/3+6/7)/2=\\mathbf{16/21}"}</InlineMath>{". Normalizing mean similarities "}<InlineMath>{"(4,1)"}</InlineMath>{" yields "}<strong>{""}<InlineMath>{"4/5"}</InlineMath>{""}</strong>{", a different result. Record the fixed Q/K/V, scale, feature distribution, seeds, counts, whether draws are nested, normalization, error metric and all declared outcomes. Do not discard seeds that worsen as features are added."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Repair a streaming feature bug"}</H3>

<Prose>{"A program computes exponential key features, subtracting each key's own largest log-feature value before writing it into memory. Queries are similarly centered per row. It claims the result is unchanged because “softmax ignores additive constants.” Which centering is safe, and what state repair is needed when the common key scale changes?"}</Prose>

<details><summary>Solution</summary>

<Prose>{"A scalar multiplier shared by all features of one query cancels in that query's numerator and denominator. A scalar shared by "}<strong>{"all keys"}</strong>{" also cancels. Different per-key scalars change relative key contributions. Maintain a common key scale and, when it changes, multiply the existing "}<InlineMath>{"S,z"}</InlineMath>{" by the corresponding conversion factor before adding the new write. The usual row-softmax invariance does not justify unrelated rescaling of separate keys."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Explain an exact local null"}</H3>

<Prose>{"A model has two causal attention layers, each with a three-key window, and tokenwise FFNs. Its last input is at position 14. Can changing raw input position 8 affect output 14? What about position 10? How would you distinguish a missing path from a learned near-zero response?"}</Prose>

<details><summary>Solution</summary>

<Prose>{"The maximum distance is "}<InlineMath>{"2(3-1)=4"}</InlineMath>{". Position 8 is six positions away, so it cannot affect output 14 under these assumptions. Position 10 is reachable, so a dependence is possible but may be weak or absent for particular weights and values. Check graph reach independently of numerical sensitivity; a small observed change cannot prove a missing path. Cross-position normalization, convolution or another global operation would change the assumptions."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Design a fair comparison"}</H3>

<Prose>{"You want to replace dense attention in a code-assistance model. A window model wins on next-token loss averaged over short files. Specify a test that could reveal a meaningful weakness, and separate an operator-level comparison from a trained-model comparison."}</Prose>

<details><summary>One solution</summary>

<Prose>{"Create held-out files or repositories requiring earlier definitions, renamed identifiers and distractors at varied distances; prevent near-duplicate leakage. Compare exact correctness on those dependencies and behavior on ordinary code. For an operator test, hold Q/K/V and weights fixed and measure output differences caused by the replacement. For a model comparison, permit declared adaptation/training and report its budget, validation selection and untouched test results. Measure prefill, decoding, cache and total latency on the target device separately. A short-file average alone does not establish long-range retrieval ability."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"10. Interpret a modern indexer's complexity"}</H3>

<Prose>{"A sparse main attention layer reads a fixed 512 selected keys per query, but its indexer compares each query with every earlier key. Is the complete sequence computation linear in "}<InlineMath>{"L"}</InlineMath>{"? If a later layer reuses a candidate pool, does that erase the first layer's cost?"}</Prose>

<details><summary>Solution</summary>

<Prose>{"The main read is order "}<InlineMath>{"512L"}</InlineMath>{" times its per-pair cost, but the all-prefix indexer contributes order "}<InlineMath>{"L^2"}</InlineMath>{" comparisons. A smaller indexer dimension or precision can make that term practically cheaper without changing its order. Reusing a bounded candidate pool can reduce later selection work; the initial scan still belongs in the total. The selected memory representations, selected indices and fresh queries must each be accounted for."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"10. Another way to learn, and what comes next"}</H2>

<Prose>{"For a visual explanation of graph connectivity, read Google Research's "}<a href={"https://research.google/blog/constructing-transformers-for-longer-sequences-with-sparse-attention-methods/"}>{"Constructing Transformers for Longer Sequences with Sparse Attention Methods"}</a>{". Its graph, sentence/paragraph and blockification explanations are useful companions to §2 and §6. The 2021 hardware limits and broad performance wording describe that historical setting; use this lesson's explicit causal and cost distinctions when interpreting them."}</Prose>

<Prose>{"For a different view of feature factorization, read the authors' "}<a href={"https://research.google/blog/rethinking-attention-with-performers/"}>{"Rethinking Attention with Performers"}</a>{". Its matrix-association and prefix-sum visuals accompany §3–4, and its protein example provides another application. Read its “unbiased attention” shorthand with the kernel-versus-normalized-ratio distinction developed here. Both articles offer a useful visual companion to the self-contained derivations here."}</Prose>

<Prose>{"For implementation practice, the "}<a href={"https://pytorch.org/blog/flexattention/"}>{"FlexAttention tutorial"}</a>{" shows how score modifications and block masks connect to compiled kernels. The current "}<a href={"https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html"}>{"API reference"}</a>{" is the version-sensitive companion. GPU execution and latency comparisons are separate from the CPU programs supplied here."}</Prose>

<Prose>{"Primary reading, by the question it answers:"}</Prose>

<ul><li>{""}<a href={"https://arxiv.org/html/1904.10509v1"}>{"Sparse Transformer"}</a>{": how alternating spatial patterns create routes through layers."}</li><li>{""}<a href={"https://arxiv.org/html/2004.05150v2"}>{"Longformer"}</a>{" and "}<a href={"https://arxiv.org/html/2007.14062v2"}>{"BigBird"}</a>{": local/global graph design and the limits of expressivity claims."}</li><li>{""}<a href={"https://arxiv.org/html/2001.04451v2"}>{"Reformer"}</a>{" and "}<a href={"https://aclanthology.org/2021.tacl-1.4.pdf"}>{"Routing Transformer"}</a>{": candidate discovery by hashing or clustering."}</li><li>{""}<a href={"https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf"}>{"Transformers are RNNs"}</a>{": feature-state recurrence and causal differentiation."}</li><li>{""}<a href={"https://arxiv.org/html/2009.14794v4"}>{"Performer"}</a>{": positive random features, Gaussian versus fixed-radius constructions, and approximation analysis."}</li><li>{""}<a href={"https://arxiv.org/html/2006.04768v3"}>{"Linformer"}</a>{" and "}<a href={"https://arxiv.org/html/2102.03902v3"}>{"Nyströmformer"}</a>{": two distinct routes through a small intermediate dimension."}</li><li>{""}<a href={"https://arxiv.org/html/2502.11089v1"}>{"NSA"}</a>{", "}<a href={"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf"}>{"V3.2-Exp report"}</a>{", and "}<a href={"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf"}>{"V4.1 report"}</a>{": current examples where selection, compression and reuse have separate mechanisms and costs."}</li><li>{""}<a href={"https://arxiv.org/html/2009.06732v3"}>{"Efficient Transformers survey"}</a>{": a historical map of families and evaluation issues, not a current exhaustive ranking."}</li></ul>

<Prose>{"You are ready for the next lesson when you can identify the legal input path, explain the stored state, calculate one sparse and one feature-kernel read, and propose a control that would expose a misleading efficiency claim. You do not need to memorize every architecture's acronym."}</Prose>

<Prose>{"The next topic in the module is "}<a href={"/learn/path/full-curriculum/vision-transformers-vit-deit-swin-dinov2?module=deep-learning-fundamentals"}>{"Vision Transformers: ViT, DeiT, Swin and DINOv2"}</a>{". Image patches give the sparse graph a two-dimensional geometry; shifted windows and learned image representations will make the connection concrete. Later "}<a href={"/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals"}>{"Mixture-of-Experts Transformers"}</a>{" sparsify which expert computations run, a different axis from selecting attention edges."}</Prose></section>
 </div>,
};
