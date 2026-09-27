// Generated from the full prepared manuscript by scripts/generate-hopfield-lesson.mjs.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {LessonIntro} from '../../components/lesson-labs/LessonElements.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {LookupFigure,NormTrapFigure,SignedMemoryFigure,BinaryWorkedFigure,BinaryEnergyFigure,BinaryCapacityFigure,ContinuousWorkedFigure,CobwebFigure,EnergyLandscapeFigure,KeyValueFigure,ModuleOwnershipFigure,QueryTrainingFigure,DigitRolesFigure,DigitArchitectureFigure,DigitMetricsFigure,MarginFigure,CapacityAxesFigure,ParityFigure,BagPoolingFigure,HopularFigure,SensitivityFigure,ChangingBankFigure} from '../../components/lesson-labs/HopfieldFigures.jsx';
import {BinaryMemoryLab,ContinuousMemoryLab,AssociationLab} from '../../components/lesson-labs/HopfieldLabs.jsx';
import {DigitMemoryLab,DigitStoriesFigure,HopfieldProgram} from '../../components/lesson-labs/HopfieldDigitLab.jsx';
export default {title:"Modern Hopfield Networks",readTime:"~70 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral hopfield-lesson"><LessonIntro prerequisites="Vectors, dot products and weighted averages; derivatives and the energy argument are developed here." sections={[["1-a-memory-is-more-than-a-label","1. A memory is more than a label"],["2-classical-hopfield-memory-correct-one-feature-at-a-time","2. Classical Hopfield memory: correct one feature at a time"],["3-continuous-memories-score-distribute-weight-retrieve","3. Continuous memories: score, distribute weight, retrieve"],["4-attention-reads-associations-as-well-as-memories","4. Attention reads associations as well as memories"],["5-a-real-memory-bank-for-handwritten-digits","5. A real memory bank for handwritten digits"],["6-build-a-reliable-association-system","6. Build a reliable association system"],["7-optional-depth-what-capacity-and-energy-really-promise","7. Optional depth: what capacity and energy really promise"],["8-applications-that-make-the-memory-choice-matter","8. Applications that make the memory choice matter"],["9-practice-predict-calculate-and-diagnose","9. Practice: predict, calculate and diagnose"],["references-another-way-to-learn-it","References & another way to learn it"]]}>Turn a partial cue into a memory read, then test what that read actually improves.</LessonIntro>
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Flip cue bits and visit order, move continuous memories, and edit the keys, values and pixels of a real handwriting cue. Follow the resulting energy, retrieval weights and returned values. Compare cases where recall repairs a memory with cases where a cleaner reconstructed image represents the wrong class."}</Prose>

<Prose>{"A smudged handwritten digit still contains clues: the bend of a stroke, an opening in a loop, the position of a vertical line. Suppose we keep examples of handwriting and ask a model to reconstruct something useful from those clues. The interesting question is not only which example receives the highest score. It is whether repeatedly using the retrieved information improves the cue, whether several examples should contribute, and how to recognize an incorrect reconstruction."}</Prose>

<Prose>{"A "}<strong>{"Hopfield network"}</strong>{" is an associative memory: you give it content that resembles a memory, and its dynamics attempt to complete or refine that content. A conventional database uses an address such as record 42. Associative memory uses a cue such as “the shape with a loop and this downward stroke.”"}</Prose>

<Prose>{"This lesson moves from four binary features to continuous memories, then to a small trained handwriting classifier. You will calculate an update, explain its energy change, distinguish a retrieved vector from its associated label, and investigate why an apparently cleaner image can represent the wrong digit."}</Prose>

<Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/spectral-normalization-gradient-penalty?module=deep-learning-fundamentals"}>{"Spectral Normalization & Gradient Penalty"}</a>{" lesson asked how strongly a function can change its output when its input changes. Here that becomes a concrete question about retrieval: does a small cue change shrink after another update, or push the state toward a different memory?"}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" follow §§1–6 and the investigations, then try exercises 1–7. The explicitly optional branches in §7 develop capacity, higher-order memories and energy-derived architectures. The local calculations use vectors, dot products, weighted averages and derivatives, all refreshed where they enter."}</Prose>

<H2>{"1. A memory is more than a label"}</H2>

<Prose>{"Imagine storing the four-feature pattern"}</Prose>

<Prose>{""}<strong>{"[on, on, off, off] = [1, 1, −1, −1]."}</strong>{""}</Prose>

<Prose>{"An arriving cue is [1, −1, −1, −1]: one feature is wrong. We want the two left features to support one another and the two right features to oppose them. Those relationships can correct a feature without being told which feature was corrupted."}</Prose>

<Prose>{"There are three different success criteria:"}</Prose>

<NeuralTable caption={"1. A memory is more than a label"} headers={[<>{"Goal"}</>,<>{"What counts as success?"}</>,<>{"What must be available?"}</>]} rows={[[<>{"Exact stored-pattern recall"}</>,<>{"Output equals a particular stored vector"}</>,<>{"That vector was stored"}</>],[<>{"Reconstruction"}</>,<>{"Output approaches the clean source under a stated distance"}</>,<>{"Clean reference for evaluation"}</>],[<>{"Classification"}</>,<>{"Associated class is correct"}</>,<>{"Labels attached to reference examples"}</>]]} />

<Prose>{"A new person's handwritten “1” was never in our memory bank. Returning the pixels of an older “1” may classify it correctly while changing its handwriting. A convex combination of several examples may improve average reconstruction error without matching any particular stored image."}</Prose>

<LookupFigure />

<Prose>{"A useful geometric refresher: a dot product adds coordinate-wise agreements. For q = [1, 0], memories [1, 0] and [3, 1] score 1 and 3. The second wins by dot product even though the first has Euclidean distance zero. If all memories have equal norm, minimizing squared distance to a fixed cue is equivalent to maximizing dot product, because"}</Prose>

<div className="neural-equation"><MathBlock>{"\\|q-x_i\\|^2=\\|q\\|^2+\\|x_i\\|^2-2q^\\top x_i."}</MathBlock></div>

<Prose>{"When norms differ, the middle term matters. Normalizing nonzero vectors to unit length makes the dot product a cosine similarity. That is a modeling choice: it removes information carried only by magnitude."}</Prose>

<NormTrapFigure />

<H2>{"2. Classical Hopfield memory: correct one feature at a time"}</H2>

<H3>{"Store relationships"}</H3>

<Prose>{"Let P binary memories be rows of X, each containing d values in {−1, +1}. The Hebbian storage rule forms"}</Prose>

<div className="neural-equation"><MathBlock>{"W=\\frac{X^\\top X}{d},\\qquad W_{ii}\\leftarrow0."}</MathBlock></div>

<Prose>{"“Hebbian” here means that features with matching signs contribute a positive connection and opposite signs contribute a negative connection. Summing outer products adds each memory's suggested relationships. The connections are symmetric."}</Prose>

<Prose>{"For our one memory x = [1, 1, −1, −1], the matrix is"}</Prose>

<NeuralTable caption={"Store relationships"} headers={[<>{"W"}</>,<>{"feature 1"}</>,<>{"feature 2"}</>,<>{"feature 3"}</>,<>{"feature 4"}</>]} rows={[[<>{"feature 1"}</>,<>{"0"}</>,<>{"0.25"}</>,<>{"−0.25"}</>,<>{"−0.25"}</>],[<>{"feature 2"}</>,<>{"0.25"}</>,<>{"0"}</>,<>{"−0.25"}</>,<>{"−0.25"}</>],[<>{"feature 3"}</>,<>{"−0.25"}</>,<>{"−0.25"}</>,<>{"0"}</>,<>{"0.25"}</>],[<>{"feature 4"}</>,<>{"−0.25"}</>,<>{"−0.25"}</>,<>{"0.25"}</>,<>{"0"}</>]]} />

<Prose>{"A neuron does not vote for itself: removing the diagonal makes the local field describe the other features' evidence. A nonzero diagonal can change update behavior; it does not simply force every state to become all ones."}</Prose>

<SignedMemoryFigure />

<H3>{"Read relationships"}</H3>

<Prose>{"The "}<strong>{"local field"}</strong>{" at coordinate i is hᵢ = Σⱼ Wᵢⱼsⱼ. If it is positive, set sᵢ to +1; if negative, set it to −1. At exactly zero, retain the current value."}</Prose>

<Prose>{"Update one coordinate, then use the changed state when updating the next coordinate. This is an "}<strong>{"asynchronous update"}</strong>{". A "}<strong>{"sweep"}</strong>{" visits every coordinate once."}</Prose>

<Prose>{"Starting from [1, −1, −1, −1], feature 1 sees field +0.25 and stays +1. Feature 2 then sees +0.75 and changes to +1. Features 3 and 4 already agree with their negative fields. After that sweep the state is the stored pattern."}</Prose>

<BinaryWorkedFigure />

<H3>{"Why this particular update settles"}</H3>

<Prose>{"Define an energy, a scalar score assigned to the whole configuration:"}</Prose>

<div className="neural-equation"><MathBlock>{"E(s)=-\\frac12 s^\\top Ws."}</MathBlock></div>

<Prose>{"This is a mathematical objective, not an amount of electrical energy measured in joules. Positive connections prefer equal signs; negative connections prefer opposite signs. Both preferences lower E."}</Prose>

<Prose>{"For symmetric W with zero diagonal, changing only coordinate i gives"}</Prose>

<div className="neural-equation"><MathBlock>{"E(s_{\\mathrm{after}})-E(s_{\\mathrm{before}})=-(s_{i,\\mathrm{after}}-s_{i,\\mathrm{before}})h_i."}</MathBlock></div>

<Prose>{"Our corrected coordinate changes from −1 to +1 with field +0.75, so ΔE = −2 × 0.75 = −1.5. The complete cue has energy 0; the recovered state has energy −1.5."}</Prose>

<Prose>{"Every actual flip with a nonzero field decreases energy. Unchanged coordinates leave it constant. There are finitely many binary configurations, so with fair repeated coordinate visits and this tie rule the process reaches a state with no energy-lowering single-coordinate flip."}</Prose>

<Prose>{"This is a local guarantee. It does not say the state is the desired memory, the closest memory, or the global minimum."}</Prose>

<BinaryEnergyFigure />

<Prose>{""}<strong>{"Why update order matters."}</strong>{" From [−1, −1, −1, −1], the forward order 1, 2, 3, 4 reaches [1, 1, −1, −1]; the reverse order reaches [−1, −1, 1, 1]. Both have energy −1.5. With no thresholds, E(s) = E(−s), so the inverse pattern is equally plausible to this energy."}</Prose>

<Prose>{"If all coordinates change simultaneously, the proof above no longer applies: each changed field was computed against the old state. For W = [[0, 1], [1, 0]], synchronous updates alternate [1, −1] → [−1, 1] → [1, −1]. A stopping limit is not proof of convergence."}</Prose>

<BinaryMemoryLab />

<H3>{"When memories interfere"}</H3>

<Prose>{"With many stored patterns, W combines competing suggestions. Some errors settle into a "}<strong>{"spurious state"}</strong>{", an attractor that was never deliberately stored. A mixture of correlated memories can be stable. Exact duplicates also alter the strength of their contributions."}</Prose>

<Prose>{"The often quoted 0.138d capacity concerns a particular random-pattern, Hebbian, large-system retrieval regime allowing small errors. It is not a universal hard limit for every learning rule, every finite memory bank, or every definition of successful recall. Exact recovery of most versus every random stored pattern gives different asymptotic conditions. We return to those distinctions in §7."}</Prose>

<Prose>{"A small experiment is easier to interpret than an unexplained theoretical line. With d = 64, eight independently generated banks at each size, and exactly six randomly flipped cue bits, our recorded run gives:"}</Prose>

<NeuralTable caption={"When memories interfere"} headers={[<>{"Patterns per bank"}</>,<>{"Stored patterns tested across 8 banks"}</>,<>{"Exact fixed states"}</>,<>{"Exact recalls from damaged cues"}</>]} rows={[[<>{"2"}</>,<>{"16"}</>,<>{"16"}</>,<>{"16"}</>],[<>{"6"}</>,<>{"48"}</>,<>{"48"}</>,<>{"47"}</>],[<>{"10"}</>,<>{"80"}</>,<>{"61"}</>,<>{"56"}</>],[<>{"16"}</>,<>{"128"}</>,<>{"49"}</>,<>{"29"}</>],[<>{"24"}</>,<>{"192"}</>,<>{"12"}</>,<>{"2"}</>]]} />

<Prose>{"“Exact fixed state” means the original pattern would retain every coordinate under the tie-preserving local rule. Recall starts with damage and uses sequential updates. The two columns ask different questions, and neither eight-bank experiment estimates a universal capacity constant."}</Prose>

<BinaryCapacityFigure />

<H2>{"3. Continuous memories: score, distribute weight, retrieve"}</H2>

<Prose>{"The modern continuous construction keeps the memory vectors explicitly. X now has P rows and d real-valued columns. A cue q has d coordinates."}</Prose>

<ol start={1}><li>{""}<strong>{"Score:"}</strong>{" s = Xq gives one dot product per memory."}</li><li>{""}<strong>{"Sharpen:"}</strong>{" pᵢ = exp(βsᵢ) / Σⱼ exp(βsⱼ)."}</li><li>{""}<strong>{"Read:"}</strong>{" q next = Xᵀp = Σᵢ pᵢxᵢ. We call this complete read operation F(q)."}</li></ol>

<Prose>{"The positive number β is the "}<strong>{"inverse temperature"}</strong>{". Increasing β increases the relative advantage of higher-scoring memories. The softmax weights are nonnegative and sum to one, so retrieval is a weighted average inside the memories' convex hull."}</Prose>

<Prose>{"These normalized weights are an allocation of attention. They are not automatically calibrated probabilities that a memory is correct."}</Prose>

<H3>{"Work through two memories"}</H3>

<Prose>{"Keep x₁ = [1, 0], x₂ = [−1, 0], and start at q = [0.2, 0.4]. The scores are [0.2, −0.2]. With β = 2, the logits are [0.4, −0.4], giving weights approximately [0.689974, 0.310026]. Thus"}</Prose>

<Prose>{"q next = [0.689974 − 0.310026, 0] = [0.379949, 0]."}</Prose>

<Prose>{"The vertical component disappears because neither memory contains one. The horizontal component becomes more positive, but it does not jump to +1."}</Prose>

<ContinuousWorkedFigure />

<Prose>{"Apply the same update again:"}</Prose>

<NeuralTable caption={"Work through two memories"} headers={[<>{"Update count"}</>,<>{"Horizontal coordinate"}</>,<>{"Vertical coordinate"}</>]} rows={[[<>{"0"}</>,<>{"0.200000"}</>,<>{"0.400000"}</>],[<>{"1"}</>,<>{"0.379949"}</>,<>{"0"}</>],[<>{"2"}</>,<>{"0.641017"}</>,<>{"0"}</>],[<>{"3"}</>,<>{"0.857026"}</>,<>{"0"}</>]]} />

<Prose>{"For these two memories the whole horizontal recurrence is"}</Prose>

<Prose>{"qₓ next = tanh(βqₓ)."}</Prose>

<Prose>{"Here tanh(z) = [exp(z) − exp(−z)] / [exp(z) + exp(−z)]. The two opposing softmax contributions reduce to this expression."}</Prose>

<Prose>{"At β = 2 the positive stable fixed point is near 0.9575, not exactly the stored coordinate 1. A "}<strong>{"fixed point"}</strong>{" is a state the update leaves unchanged. A "}<strong>{"stored pattern"}</strong>{" is a row of X. They need not be identical at finite temperature."}</Prose>

<Prose>{"Now reduce β to 0.5. The same cue's horizontal coordinates become 0.099668, 0.049793 and 0.024891. Repetition approaches the middle, averaging the memories instead of selecting one."}</Prose>

<CobwebFigure />

<Prose>{"A cue [0, 0.6] gives equal weights and reaches [0, 0]. It stays there even at β = 2. The exact symmetric state remains fixed although a small horizontal disturbance grows. A balanced cue does not acquire evidence about which memory was intended just because we turn up β."}</Prose>

<H3>{"The energy behind the update"}</H3>

<Prose>{"For fixed X and β > 0, write"}</Prose>

<div className="neural-equation"><MathBlock>{"E(q)=\\frac12\\|q\\|^2-\\frac1\\beta\\log\\sum_i\\exp(\\beta x_i^\\top q)."}</MathBlock></div>

<Prose>{"We omit constants independent of q; adding them changes neither the update nor energy differences. The log-sum-exp is a smooth version of the largest score. Its negative encourages agreement with memories. The quadratic eventually dominates this term as ||q|| grows, keeping the energy bounded below. More directly, after one update q lies in the finite memory bank's convex hull."}</Prose>

<Prose>{"Differentiating gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\nabla E(q)=q-X^\\top\\operatorname{softmax}(\\beta Xq)=q-F(q)."}</MathBlock></div>

<Prose>{"A stationary point therefore satisfies q = F(q). Writing this equality identifies a fixed-point equation; it does not solve it in one step."}</Prose>

<Prose>{"Why does the iteration decrease E? Let g(q) = β⁻¹log Σ exp(βxᵢᵀq). Because g is convex,"}</Prose>

<Prose>{"g(z) ≥ g(q) + ∇g(q)ᵀ(z − q)."}</Prose>

<Prose>{"Negate this inequality and add ½||z||². We have built an upper bound on E(z) that touches E at q. Minimizing that quadratic upper bound gives z = ∇g(q) = F(q). Consequently,"}</Prose>

<div className="neural-equation"><MathBlock>{"E(F(q))\\le E(q)-\\frac12\\|F(q)-q\\|^2."}</MathBlock></div>

<Prose>{"This is a short version of the "}<strong>{"concave-convex procedure"}</strong>{": replace the concave part by a tangent upper bound, minimize, repeat. Its direction matters: log-sum-exp is convex; negative log-sum-exp is concave."}</Prose>

<Prose>{"For the β = 2 example, energies at updates 0–3 are −0.285550, −0.406684, −0.472651 and −0.505746. The second update changes the state substantially. Calling the first read “exact one-step convergence” would contradict the numbers."}</Prose>

<EnergyLandscapeFigure />

<Prose>{"The "}<a href={"https://arxiv.org/abs/2008.02217"}>{"Ramsauer paper"}</a>{" proves convergence properties and much stronger local retrieval results under separation assumptions. “One update” in those retrieval results means reaching a prescribed small error near an associated fixed point, not equality after one update for arbitrary memories and cues."}</Prose>

<ContinuousMemoryLab />

<H3>{"A useful sensitivity connection"}</H3>

<Prose>{"The derivative of retrieval is"}</Prose>

<div className="neural-equation"><MathBlock>{"J_F(q)=\\beta X^\\top[\\operatorname{diag}(p)-pp^\\top]X=\\beta\\operatorname{Cov}_p(x)."}</MathBlock></div>

<Prose>{"The covariance measures how much the currently weighted memories disagree. If almost all weight lies on one memory, this local derivative can be small: nearby cues yield nearly the same retrieval. If conflicting memories share weight, a cue change can have a larger effect."}</Prose>

<Prose>{"In our two-memory example, F′(0) = β. At β = 0.5, small horizontal errors shrink near zero; at β = 2, they grow. This connects the previous lesson's derivative bounds to an observable attraction or repulsion. A small local derivative near one memory does not establish a global contraction over every cue."}</Prose>

<SensitivityFigure />

<H2>{"4. Attention reads associations as well as memories"}</H2>

<Prose>{"A library catalogue separates the description used to search from the information returned. A key might describe a book's subject; the value might be its location. Associative neural memory can use the same separation."}</Prose>

<Prose>{"Let K contain P keys of width dₖ, V contain P associated values of width dᵥ, and Q contain B query rows. The read is"}</Prose>

<div className="neural-equation"><MathBlock>{"A=\\operatorname{softmax}(\\beta QK^\\top),\\qquad Z=AV."}</MathBlock></div>

<NeuralTable caption={"4. Attention reads associations as well as memories"} headers={[<>{"Quantity"}</>,<>{"Shape"}</>,<>{"Meaning"}</>]} rows={[[<>{"Q"}</>,<>{"B × dₖ"}</>,<>{"B requests"}</>],[<>{"K"}</>,<>{"P × dₖ"}</>,<>{"descriptions used to score memories"}</>],[<>{"QKᵀ"}</>,<>{"B × P"}</>,<>{"one score per query-memory pair"}</>],[<>{"A"}</>,<>{"B × P"}</>,<>{"row-normalized allocation of weight"}</>],[<>{"V"}</>,<>{"P × dᵥ"}</>,<>{"payload attached to each memory"}</>],[<>{"Z"}</>,<>{"B × dᵥ"}</>,<>{"retrieved payloads"}</>]]} />

<Prose>{"For one query, K = V = X and β = 1/√d, this is exactly the modern Hopfield update written with row vectors. The transpose convention is the only difference. Our complete CPU comparison against PyTorch scaled dot-product attention returns [0.1688805062, 0.4355881251], with maximum difference 2.78 × 10⁻¹⁷ in float64."}</Prose>

<Prose>{"A learned value projection can then transform the key-space read into another space. Ordinary attention permits keys and values to have independent projections. Its association formula still makes sense, but the output is no longer automatically a state update of the same scalar energy we just derived."}</Prose>

<Prose>{"For example, keys [1, 0] and [0, 1], cue [0.6, −0.2], and β = 1 give weights [0.689974, 0.310026]. Attach scalar payloads 10 and −2. The returned payload is approximately 6.279694, a scalar. It cannot be fed directly into the two-dimensional key-space energy."}</Prose>

<KeyValueFigure />

<H3>{"Classify by combining memory labels"}</H3>

<Prose>{"Attach one-hot class vectors as values. For classes A and B, these are [1, 0] and [0, 1]. The read sums weight from all memories labeled A into one number, and from all memories labeled B into the other."}</Prose>

<Prose>{"Two B memories can jointly outweigh the largest individual A memory. A classifier that chooses the label of the highest-scoring single memory can therefore disagree with one that sums the class weights. Both decisions should be evaluated against the same task."}</Prose>

<Prose>{"If every value equals [4, −2], the output is [4, −2] for every query. The keys can change the weights without changing the retrieved payload. This is why an attention heatmap alone cannot explain all output behavior."}</Prose>

<AssociationLab />

<H3>{"Three useful module designs"}</H3>

<Prose>{""}<strong>{"Associate two sets."}</strong>{" Queries come from one input and keys/values from another. Image patches can query a set of candidate object descriptions; a decoder can query encoded input. This connects to "}<a href={"/learn/path/full-curriculum/interleaved-cross-attention-architectures?module=deep-learning-fundamentals"}>{"Cross-Attention Architectures"}</a>{"."}</Prose>

<Prose>{""}<strong>{"Pool a variable-sized set."}</strong>{" Learn one or several query vectors. Each query searches the input's keys and returns a weighted summary. If the input rows are permuted and their keys/values stay paired, the summary stays the same. This makes set pooling appropriate when order is irrelevant. An empty set still needs an explicit policy: softmax over no memories is undefined."}</Prose>

<Prose>{""}<strong>{"Learn a fixed prototype bank."}</strong>{" Store a trainable parameter matrix rather than every example. Query projections and prototype coordinates can move during training. Such a bank contains learned representations, not necessarily verbatim training records."}</Prose>

<Prose>{"These correspond to the author library's Hopfield, HopfieldPooling and HopfieldLayer abstractions. The "}<a href={"https://github.com/ml-jku/hopfield-layers"}>{"official repository"}</a>{" is useful for configuration and examples; its README describes an older Python/PyTorch development environment. The runnable programs here use ordinary NumPy and PyTorch, so understanding the lesson does not depend on installing that research package."}</Prose>

<ModuleOwnershipFigure />

<H3>{"Learn where to look"}</H3>

<Prose>{"For a single target memory t, the loss L = −log pₜ teaches the query to favor its key. With fixed keys and inverse temperature β,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial q}=\\beta K^\\top(p-e_t)."}</MathBlock></div>

<Prose>{"Here eₜ is one at the target's position and zero elsewhere. The gradient subtracts the target key from the current weighted key average."}</Prose>

<Prose>{"Take K = I₂, q = [0.2, −0.1], β = 1, and target memory 2. We obtain p = [0.574443, 0.425557], loss 0.854355, and gradient [0.574443, −0.574443]. An update q ← q − 0.1∇L gives [0.142556, −0.042556], reducing loss to 0.789980. The second key becomes relatively easier to retrieve."}</Prose>

<Prose>{"This is "}<strong>{"parameter or representation learning across examples"}</strong>{". It differs from "}<strong>{"state refinement for one cue"}</strong>{", which reduces the fixed-bank energy. In a network q is produced by learned parameters, and backpropagation carries this gradient into those parameters."}</Prose>

<Prose>{"For a class represented by several memories, let π_c = Σᵢ:yᵢ=c pᵢ. Training uses −log π_c. Its derivative with respect to scaled logit ℓᵢ = βqᵀkᵢ is pᵢ − rᵢ, where rᵢ = pᵢ/π_c for target-class memories and zero otherwise. The target class receives extra weight without requiring every query to match one arbitrarily selected prototype."}</Prose>

<QueryTrainingFigure />

<H2>{"5. A real memory bank for handwritten digits"}</H2>

<Prose>{"We use the "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI Optical Recognition of Handwritten Digits dataset"}</a>{". Each image has 8 × 8 cells. A cell records how many pixels were on in a 4 × 4 block of a normalized 32 × 32 handwriting bitmap, so its value is an integer from 0 to 16."}</Prose>

<Prose>{"The original split contains 3,823 training images from 30 writers and 1,797 test images from 13 different writers. We preserve that test boundary. The released 65-column files do not include individual writer IDs within each split."}</Prose>

<Prose>{"Our deliberately small experiment chooses, separately within each digit class and using seed 113:"}</Prose>

<ul><li>{"20 training images as memories: 200 in the bank."}</li><li>{"80 different training images as fitting queries: 800."}</li><li>{"30 further training images as validation queries: 300."}</li><li>{"All 1,797 original test images for assessment."}</li></ul>

<Prose>{"The other 2,523 training rows are unused. We checked that all 5,620 images have distinct exact feature vectors, including across the original train/test boundary. The saved source IDs reproduce every role. Validation comes from the original training writers; test evaluates the different-writer split."}</Prose>

<Prose>{"Only memory labels are available to retrieval. Fitting-query labels train the projection; validation labels select settings; test labels score the final comparisons. No validation or test image becomes a stored reference."}</Prose>

<DigitRolesFigure />

<H3>{"Start with a direct memory baseline"}</H3>

<Prose>{"Divide each intensity by 16, then normalize the 64-dimensional image vector to unit length. Compare the query with the 200 normalized memory vectors."}</Prose>

<Prose>{"The first baseline returns the nearest memory's label by cosine similarity. The second uses softmax weights and sums them by class. Its β is selected from {4, 16, 64, 256} using clean validation cross-entropy. This selects β = 64. The β grid is predefined; we do not select settings on the corrupted test images."}</Prose>

<H3>{"Learn a more useful association space"}</H3>

<Prose>{"The learned model maps an image through a shared linear projection Wₑ of shape 16 × 64:"}</Prose>

<Prose>{"kᵢ = normalize(Wₑxᵢ), q = normalize(Wₑx)."}</Prose>

<Prose>{"It has 1,024 trainable parameters, no bias, and a fixed β = 16. The same projection is applied to both memories and queries, so both sides use the same learned geometry. Softmax operates over 200 memories. Summing weights by memory label gives ten class probabilities."}</Prose>

<Prose>{"We fit Wₑ by mean negative log probability of the correct class, using all 800 fitting queries in each update. Adam uses learning rate 0.005 for 100 epochs. After each update we evaluate clean validation cross-entropy and retain the best epoch. We run seeds 17 and 41 to expose initialization variation. Seed 17 is the predefined demonstration run, not a winner chosen from test performance."}</Prose>

<Prose>{"These are learned associations in a small supervised model. They are not a reproduction of the Ramsauer paper's benchmarks or proof that a Hopfield layer outperforms other architectures."}</Prose>

<DigitArchitectureFigure />

<H3>{"Run the complete programs"}</H3>

<Prose>{"Download "}<a href={"/learn-code/modern-hopfield-networks/./digit_memory.py"}>{"digit_memory.py"}</a>{", "}<a href={"/learn-code/modern-hopfield-networks/./optdigits.tra"}>{"optdigits.tra"}</a>{", "}<a href={"/learn-code/modern-hopfield-networks/./optdigits.tes"}>{"optdigits.tes"}</a>{" and "}<a href={"/learn-code/modern-hopfield-networks/./optdigits.names"}>{"optdigits.names"}</a>{" into one directory. "}<a href={"/learn-code/modern-hopfield-networks/./data-provenance.md"}>{"Data provenance and role details"}</a>{" describe attribution and the saved split. In a Python environment with NumPy and PyTorch:"}</Prose>

<CodeBlock language={"text"}>{"python -m pip install numpy torch\npython digit_memory.py"}</CodeBlock>

<Prose>{"The program reads local files, builds the roles, evaluates the baselines, trains both projections, and saves results and selected weights. It uses the CPU and downloads no model. The recorded run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. Small floating-point differences can occur in another environment."}</Prose>

<Prose>{"The core read, written with explicit shapes, is:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nimport torch.nn.functional as F\n\ndef label_read(query_images, memory_images, memory_labels, projection, beta=16.0):\n    # query_images: B x 64; memory_images: P x 64\n    queries = F.normalize(projection(query_images), dim=-1)\n    keys = F.normalize(projection(memory_images), dim=-1)\n    log_weights = F.log_softmax(beta * queries @ keys.T, dim=-1)\n    # Sum memory mass by class in log space, retaining small probabilities.\n    log_classes = torch.stack([\n        torch.logsumexp(log_weights[:, memory_labels == label], dim=1)\n        for label in range(10)\n    ], dim=1)\n    return log_classes, log_weights.exp()"}</CodeBlock>

<Prose>{"This function assumes that all ten classes are represented in the nonempty memory bank, as they are in the experiment. The complete file supplies inputs, the projection, objective, optimizer, role construction and result reporting. The log-space summation avoids turning very small class probabilities into an artificial zero before taking the loss."}</Prose>

<Prose>{"For the smaller exact mechanisms, save "}<a href={"/learn-code/modern-hopfield-networks/./associative_memory.py"}>{"associative_memory.py"}</a>{" and run it with NumPy and PyTorch available. It reproduces the four-feature recovery, continuous energy trajectories, attention equivalence, gradient example and finite binary-memory scan. Its returned traces distinguish unchanged coordinates from actual flips."}</Prose>

<HopfieldProgram filename="associative_memory.py" /><HopfieldProgram filename="digit_memory.py" />

<H3>{"Read the outcomes"}</H3>

<Prose>{"To test sensitivity to missing visual evidence, set columns 4 and 5 of each 8 × 8 query image to zero: 16 cells out of 64. The models were fitted and selected on clean inputs. This is a fixed occlusion stress test, not an alternative training distribution selected after inspecting its errors."}</Prose>

<NeuralTable caption={"Read the outcomes"} headers={[<>{"Model"}</>,<>{"Clean validation errors / 300"}</>,<>{"Clean test errors / 1,797"}</>,<>{"Occluded test errors / 1,797"}</>]} rows={[[<>{"Nearest of 200 memories, cosine"}</>,<>{"25"}</>,<>{"156"}</>,<>{"663"}</>],[<>{"Weighted labels, fixed pixel geometry, β = 64"}</>,<>{"18"}</>,<>{"136"}</>,<>{"667"}</>],[<>{"Learned projection, seed 17, epoch 100"}</>,<>{"13"}</>,<>{"100"}</>,<>{"564"}</>],[<>{"Learned projection, seed 41, epoch 25"}</>,<>{"16"}</>,<>{"113"}</>,<>{"740"}</>]]} />

<Prose>{"The learned geometries improve clean classification in these runs. Occlusion reveals a different story: seed 41 makes more errors than the nearest-memory baseline, even though its clean test result is better. Clean validation quality does not automatically identify the most robust geometry for a new kind of missing input."}</Prose>

<Prose>{"The seed-17 fitting queries have zero clean classification errors, yet the test set has 100. The memory bank contains only the 200 reference images; the remaining trainable capacity lies in the projection and its learned similarity function. Perfect fitting-query classification is not a guarantee of generalization."}</Prose>

<DigitMetricsFigure />

<H3>{"A cleaner image can tell the wrong story"}</H3>

<Prose>{"The association weights can also retrieve a weighted image: x read = Σᵢ pᵢxᵢ, using the original memory pixels as values. This output is inspectable, but the classifier was trained for labels rather than pixel reconstruction."}</Prose>

<Prose>{"Consider validation image "}<strong>{"training-source row 2946"}</strong>{", labeled “1.” The clean image receives class-1 mass 0.989362 and predicts 1. After zeroing the two central columns, it predicts 0, with class-1 mass only 0.000141. The strongest clean memory is row 1487, with weight 0.939320. Under occlusion, the top three memories become rows 1786, 699 and 104, with weights 0.285921, 0.183328 and 0.120278."}</Prose>

<Prose>{"The retrieved occluded image has mean squared pixel error 0.110814 relative to the original. The damaged input's error is 0.203674. The weighted reconstruction is closer in average pixel error while the class prediction is wrong. This is possible because reconstructing common background and stroke regions can reduce many squared errors while the identifying stroke remains incorrect."}</Prose>

<Prose>{"For another image, row 3052, labeled “0,” the same occlusion retains class 0, with mass 0.838620. Its retrieved image error is 0.022476 versus 0.074036 for the damaged cue. A memory method can be helpful on one shape and misleading on another."}</Prose>

<DigitStoriesFigure />

<Prose>{"Across all test images, seed 17's mean squared reconstruction error after occlusion is 0.051198, versus 0.124713 for the damaged input. On clean images, reconstruction error is 0.029621, whereas the original clean input has zero error. Retrieval pulls handwriting toward the memory bank; it is not an identity operation and not an unconditional denoiser."}</Prose>

<DigitMemoryLab />

<H2>{"6. Build a reliable association system"}</H2>

<Prose>{"The most useful diagnostics follow the actual read."}</Prose>

<Prose>{""}<strong>{"First inspect the cue and score geometry."}</strong>{" Are vectors normalized consistently? Do norms dominate? Is a zero-filled region being treated as “unknown” even though the model interprets it as background? Our occlusion sets pixels to zero; it supplies no missingness mask. A model trained with missingness indicators or augmentation would be a different experiment."}</Prose>

<Prose>{""}<strong>{"Then inspect the competing memories."}</strong>{" Are similar keys attached to different labels? Does one class have many more stored examples, gaining aggregate mass from multiplicity? Duplicating one of two equally scored memories changes its side's total weight from one-half to two-thirds. Duplicate entries are not neutral unless the intended weighting accounts for them."}</Prose>

<Prose>{""}<strong>{"Then inspect the payload."}</strong>{" The highest-weight memory can be correct while a label-summed read differs. Identical values can conceal large changes in weights. A visually plausible weighted image can conceal a classification failure. Inspect the task output alongside the attention distribution."}</Prose>

<Prose>{""}<strong>{"Finally inspect the learning and evaluation boundary."}</strong>{" A trainable bank, a fixed example bank and a current-input activation bank have different storage and leakage properties. Keeping held-out labels in values makes classification trivially easier in a way unavailable on new input. Changing the bank after fitting changes the predictor and should trigger a new evaluation."}</Prose>

<Prose>{"High β approaches an argmax over scores when there is a unique maximum; it preserves ties and cannot repair a wrong score ordering. It can also make gradients through losing memories extremely small. Low β gives broad averaging and may reduce query sensitivity too far. The familiar 1/√d attention scale controls dot-product magnitude under particular component-scale assumptions; it is not a universally optimal temperature for unit-normalized memories."}</Prose>

<Prose>{"For implementation, stable softmax subtracts the largest logit. All-masked or empty banks need an explicit result policy. A common additive shift of finite logits leaves the distribution unchanged; masking every logit to negative infinity does not produce a valid distribution."}</Prose>

<Prose>{"These checks are useful beyond this named architecture. The earlier "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention & Multi-Head Attention"}</a>{" lesson develops attention's full projection and masking mechanics. The present energy interpretation adds a tool for reasoning about particular memory dynamics; it does not substitute for that entire model specification."}</Prose>

<H3>{"Keep the memory operation explicit when using a framework"}</H3>

<Prose>{"The complete mechanism program is "}<a href={"/learn-code/modern-hopfield-networks/associative_memory.py"}>{"associative_memory.py"}</a>{": "}<code>{"store_binary"}</code>{" builds the symmetric zero-diagonal weights, "}<code>{"binary_recall"}</code>{" performs one asynchronous coordinate update at a time, and "}<code>{"retrieve"}</code>{"/"}<code>{"iterate"}</code>{" construct stable modern retrieval and its energy trace. "}<code>{"digit_memory.py::read_memory"}</code>{" then implements the practical trainable bank using ordinary PyTorch operations, with a learned "}<code>{"nn.Linear"}</code>{" projection and log-space class-mass aggregation. It learns the addressing representation; query labels are targets of the loss, never keys supplied to retrieval."}</Prose>

<Prose>{"There is an ordinary fused route for the softmax read itself: "}<code>{"scaled_dot_product_attention"}</code>{". The exact call in "}<code>{"associative_memory.py::main"}</code>{" holds Q, K and V fixed and compares it with explicit "}<code>{"softmax(QKᵀ/√d)V"}</code>{", with dropout zero. In our notation the inverse-temperature β must equal the call's score scale: the default is 1/√d, not an arbitrary β. For a different β pass "}<code>{"scale=beta"}</code>{" explicitly. When K=V are stored patterns the read can be the modern memory update; arbitrary distinct values retain the attention computation but do not inherit that energy-descent interpretation. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html"}>{"PyTorch attention contract"}</a>{"."}</Prose>

<Prose>{"For B queries and M stored d-dimensional patterns, an explicit read uses O(BMd) arithmetic and may materialize O(BM) scores. Query chunking bounds that intermediate without changing fixed-bank reads. Normalized log-class sums are intentionally used for a tiny ordinary bank; the optional Hopfield package is not required to expose this operation or to train it. The discrete symmetric memory uses O(d²) weight storage, a different object from an M-by-d continuous pattern bank."}</Prose>

<Prose>{""}<strong>{"Change the contract."}</strong>{" Give the three stored keys different two-coordinate values, set β=.7, and compare the explicit read with "}<code>{"scaled_dot_product_attention(query, keys, values, scale=.7, dropout_p=0.)"}</code>{". Then differentiate the squared returned-value norm with respect to the same query in both routes."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"Use batched tensors with sequence position before the last feature dimension. Create independent cloned query leaves for the two computations, hold keys/values fixed, and use the same loss reduction before "}<code>{"torch.autograd.grad"}</code>{". Both values and query derivatives should agree within a float64 tolerance such as 1e−11 on these small moderate logits; backend summation need not be bitwise equal. Normalizing keys in only one route or leaving the native scale at 1/√d changes the operation. After separating keys from values, do not test the fixed-pattern energy as a claimed monotonic invariant of the returned value: that invariant's assumptions no longer hold."}</Prose>

</details>

<H2>{"7. Optional depth: what capacity and energy really promise"}</H2>

<H3>{"A margin explains when a single read can work"}</H3>

<Prose>{"Let xₜ be the desired memory, and suppose every memory has norm at most M. For the current cue q, define the score gap"}</Prose>

<Prose>{"δ(q) = qᵀxₜ − maxⱼ≠ₜ qᵀxⱼ."}</Prose>

<Prose>{"If δ(q) > 0, the desired memory outranks every competitor. Divide the softmax denominator by exp(βqᵀxₜ):"}</Prose>

<Prose>{"pₜ = 1 / [1 + Σⱼ≠ₜ exp(β(qᵀxⱼ − qᵀxₜ))]."}</Prose>

<Prose>{"Each competing exponential is at most exp(−βδ), so"}</Prose>

<div className="neural-equation"><MathBlock>{"p_t\\ge\\frac1{1+(P-1)e^{-\\beta\\delta}}."}</MathBlock></div>

<Prose>{"Set a = (P − 1)exp(−βδ). The total competing weight is at most a/(1+a), giving"}</Prose>

<div className="neural-equation"><MathBlock>{"\\|F(q)-x_t\\|\\le\\frac{2Ma}{1+a}\\le2M(P-1)e^{-\\beta\\delta}."}</MathBlock></div>

<Prose>{"The last step uses ||xⱼ − xₜ|| ≤ 2M. This derivation explains the three levers: better separation, sharper temperature, and fewer competitors. A high-dimensional memory bank is not enough if the actual query gives the desired memory a poor score."}</Prose>

<Prose>{"For P = 100, δ = 3 and β = 2, the target-weight lower bound is 0.802957. With M = 1, the error upper bound from the tighter expression is 0.394086. These are bounds computed from the assumed gap, not measurements from the handwriting experiment."}</Prose>

<MarginFigure />

<Prose>{"A separation statement about the clean memory can be connected to noisy cues. Let Δₜ = ||xₜ||² − maxⱼ≠ₜ xₜᵀxⱼ. If ||q − xₜ|| ≤ r, then by Cauchy–Schwarz,"}</Prose>

<Prose>{"δ(q) ≥ Δₜ − 2Mr."}</Prose>

<Prose>{"Noise can spend the available margin. This does not prove that every cue in every such ball reaches a unique attractor; that stronger conclusion needs fixed-point conditions too. It does show exactly where cue error enters a one-read error bound."}</Prose>

<H3>{"Three different meanings of “capacity”"}</H3>

<ol start={1}><li>{""}<strong>{"How many vectors can the hardware hold?"}</strong>{" P rows of width d require Pd stored numbers."}</li><li>{""}<strong>{"How many memories are stable under a specified update?"}</strong>{" This is a mathematical property of the patterns and dynamics."}</li><li>{""}<strong>{"How many relevant examples improve the real task?"}</strong>{" This depends on labels, representations, distribution and evaluation."}</li></ol>

<Prose>{"For classical Hebbian random binary patterns, exact recall of most memories and exact recall of all memories have different asymptotic scales, n/(2 log n) and n/(4 log n), in the corresponding "}<a href={"https://authors.library.caltech.edu/records/q92rz-95p89"}>{"McEliece et al. analysis"}</a>{". Allowing small errors changes the criterion behind the familiar roughly 0.138n regime. None of these constants turns our finite binary scan into a calibrated capacity chart."}</Prose>

<Prose>{"For continuous modern Hopfield memory, the canonical paper establishes exponentially growing storage under a specified random-sphere construction and associated attraction regions. One form of its result uses random patterns on a sphere of radius M = K√(d−1), where K > 0 is a scalar radius multiplier (separate from the earlier key matrix), d > 1, inverse temperature β > 0 and failure probability 0 < p ≤ 1. Define"}</Prose>

<div className="neural-equation"><MathBlock>{"a=\\frac{2[1+\\ln(2\\beta K^2p(d-1))]}{d-1},\\quad b=\\frac{2K^2\\beta}{5},\\quad c=\\frac{b}{W_0(\\exp(a+\\ln b))}."}</MathBlock></div>

<Prose>{"W₀ is the principal Lambert W function, defined by W(z)exp(W(z)) = z. Under the theorem's condition c ≥ (2/√p)^(4/(d−1)), the storage lower bound has form √p × c^((d−1)/4), with probability at least 1−p. The point of displaying the parameters is to expose what must be specified: d alone does not produce a universal exp(d/2) guarantee for arbitrary learned keys."}</Prose>

<Prose>{"Our margin calculation is usually the better first diagnostic. The theorem explains why favorable separated configurations can support many memories; it does not certify every geometry learned from real handwriting."}</Prose>

<Prose>{"At P = 1,000,000 and d = 64, an explicit float32 pattern bank needs 256,000,000 bytes, about 244.14 MiB, before other model state. Reading one query against all patterns uses O(Pd) score arithmetic and O(Pd) value aggregation when key/value widths both equal d. B queries cost O(BPd). If the memory and query sets both grow with sequence length T, this becomes quadratic in T."}</Prose>

<Prose>{"Fused exact attention can avoid materializing all scores in device memory while still doing the dense pairwise arithmetic. Approximate retrieval, sparse subsets and alternative kernels change other parts of the contract. They must be assessed on their own retrieval errors and costs; an SSM such as Mamba is not simply a softmax Hopfield approximation."}</Prose>

<CapacityAxesFigure />

<H3>{"Higher-order binary memory and a useful logical example"}</H3>

<Prose>{"Modern associative memory is a family broader than the continuous softmax construction. "}<a href={"https://arxiv.org/html/1606.01164v2"}>{"Krotov and Hopfield"}</a>{" study energies of the form"}</Prose>

<div className="neural-equation"><MathBlock>{"E(s)=-\\sum_\\mu F(x_\\mu^\\top s),"}</MathBlock></div>

<Prose>{"with polynomial or rectified-polynomial F. A binary coordinate update can compare the energy with that coordinate set to +1 and to −1, then choose the lower-energy state. Higher powers change how sharply strong matches dominate weak ones."}</Prose>

<Prose>{"For a concrete parity task, store the four triples"}</Prose>

<Prose>{"[−1, −1, −1], [−1, +1, +1], [+1, −1, +1], [+1, +1, −1]."}</Prose>

<Prose>{"Clamp the first two coordinates as inputs and infer the third, which should equal −ab. With F(z) = z², both candidate outputs have energy −12 for every input pair. This storage construction cannot distinguish the answers. With F(z) = z³, E(a,b,z) = 24abz on the binary cube. Minimization chooses z = −ab, giving energy −24 rather than +24."}</Prose>

<ParityFigure />

<Prose>{"The example shows why changing the interaction function changes representable relationships. It does not mean a quadratic network with additional hidden units can never represent parity, nor that higher degree always trains better."}</Prose>

<Prose>{"The higher-order paper also connects learned memories to feature-like versus prototype-like representations and to a feedforward hidden layer with related nonlinearities. This gives a useful design question: should a memory describe a reusable feature or a whole prototype? The answer depends on the task; our 200-image reference bank deliberately uses actual examples."}</Prose>

<H3>{"Fixed-bank energy, changing memories and stochastic models"}</H3>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/boltzmann-machines-restricted-boltzmann-machines-rbm?module=deep-learning-fundamentals"}>{"Boltzmann Machines & RBM"}</a>{" lesson uses energy to define probabilities over states and learns through model/data statistics. Classical Hopfield recall here deterministically lowers an energy. A stochastic equilibrium distribution, a deterministic local minimum and a softmax distribution over memory scores are three different objects."}</Prose>

<Prose>{"For a growing sequence memory, causal reading uses only the available prefix. An age bias can modify a score to βqᵀxᵢ − γ(t−i): equally matching older memories receive less unnormalized mass. The factor exp(−γ age) acts as a forgetting preference. This does not make an explicit bank occupy constant storage, and changing time or the bank changes the energy being considered."}</Prose>

<Prose>{"There is also a genuine architecture consequence to requiring a shared energy when both queries and keys evolve. "}<a href={"https://arxiv.org/html/2302.07253v1"}>{"Energy Transformer"}</a>{" derives token dynamics from an engineered energy. Differentiating a token's contribution through both its query role and its key role adds terms absent from ordinary one-way attention. Its memory and attention contributions operate together. This is a specific construction, rather than a new name for an arbitrary transformer stack."}</Prose>

<ChangingBankFigure />

<Prose>{"The continuous-time energy argument and a numerical discretization are separate: a large finite step can require its own stability analysis. The later "}<a href={"/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals"}>{"Neural ODE & Continuous-Depth Models"}</a>{" lesson develops that distinction."}</Prose>

<H2>{"8. Applications that make the memory choice matter"}</H2>

<H3>{"Find a rare signal in a large set"}</H3>

<Prose>{"Suppose a bag contains 10,000 short sequence embeddings, and only a small subset carries useful evidence for the bag's label. Plain mean pooling dilutes each instance equally; max pooling forces each feature to use its largest entry. A learned query instead scores instances and builds a weighted summary."}</Prose>

<Prose>{"This is the structure used in "}<a href={"https://arxiv.org/abs/2007.13505"}>{"DeepRC"}</a>{": receptor sequences become embeddings, attention pools the set, and an output network predicts a repertoire-level label. The training labels apply to the bag, so learning must assign useful credit without being given a correct label for every receptor. Reordering instances should not change the summary. Adding duplicates or sampling a subset can change it."}</Prose>

<BagPoolingFigure />

<Prose>{"The broader lesson is useful for document collections and image patches as well: define what one instance is, what the set label means, and whether order or multiplicity matters. Attention weights alone do not establish that a high-weight receptor is a biological cause."}</Prose>

<H3>{"Revisit examples and features during tabular prediction"}</H3>

<Prose>{""}<a href={"https://arxiv.org/abs/2206.00664"}>{"Hopular"}</a>{" uses two memory roles in each block. One reads across stored training examples; another reads across the embedded features of the current input. The representation is refined through successive blocks, and masked attributes are part of training."}</Prose>

<Prose>{"A row with an unknown target can therefore ask both “which earlier examples resemble my current representation?” and “which of my own attributes inform one another?” That differs from a one-off nearest-neighbor lookup and from attending only across columns. The actual training examples and the current query's unavailable target must remain distinguishable."}</Prose>

<HopularFigure />

<Prose>{"The paper's benchmark conclusions belong to its datasets and protocol. For a new table, compare with strong tabular baselines under the same split and preprocessing. Calling a component a memory does not establish that it will help."}</Prose>

<H3>{"Support sets and stored prototypes"}</H3>

<Prose>{"A few-shot classifier can treat labeled support examples as keys and their labels as values. Our handwriting model is already a small instance of that design. Replacing the support set changes which classes and visual styles can receive mass. A learned embedding should be evaluated on the intended episode/class split; a capacity theorem cannot replace that experiment."}</Prose>

<Prose>{"For large text retrieval, an external search system may first shortlist documents before a neural model reads them. The shortlist operation, the attention read and the final generated answer are distinct. A correct attention identity proves neither that the search found the right evidence nor that an answer faithfully uses it."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"9. Practice: predict, calculate and diagnose"}</H2>

<Prose>{"Attempt each prompt before opening its hint. Changed inputs are intentional: the aim is to use the mechanism, not recall a number from the worked example."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Repair a different damaged feature"}</H3>

<Prose>{"Store [1, −1, 1, −1] with the Hebbian rule. Start at [1, −1, −1, −1] and visit coordinates 1 through 4. Which update changes the cue, and by how much does energy change at that update?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Build only the row needed for the damaged third coordinate. Its three neighbors currently agree with the stored pattern."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The third row is [0.25, −0.25, 0, −0.25]. Its field is 0.75, so the third coordinate changes from −1 to +1. ΔE = −(1−(−1))0.75 = −1.5. Coordinates 1 and 2 remain unchanged before that update, and coordinate 4 remains −1 afterward. The stored pattern is recovered."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. A changed continuous cue"}</H3>

<Prose>{"Keep memories [1, 0] and [−1, 0], but use cue [−0.3, 0.7] and β = 1. What is the first read? Does a second read necessarily equal it?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The vertical coordinate disappears; the horizontal update is tanh(βqₓ)."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The first read is approximately [−0.291313, 0]. The second is [tanh(−0.291313), 0], approximately [−0.283342, 0]. They differ. A one-step read can be a useful feedforward operation without being an exact fixed point."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. One high-scoring memory versus a class"}</H3>

<Prose>{"Three memories have unnormalized weights 4, 3 and 3. Their labels are A, B and B. What do nearest-memory classification and label-summed retrieval predict?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Normalize by the sum, then aggregate by label."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The weights are 0.4, 0.3 and 0.3. The highest single memory belongs to A; class masses are A = 0.4 and B = 0.6, so label summation predicts B. Neither rule is intrinsically the correct classifier for every problem; evaluate the rule you intend to use."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Can a shared value hide a changed attention map?"}</H3>

<Prose>{"Keep the three weights from exercise 3, but attach value [2, 5] to every memory. Then change the weights to [0.9, 0.05, 0.05]. What changes?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Factor the common value out of the weighted sum."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The output remains [2, 5], since the weights sum to one in both cases. The attention distribution changed, but the payload did not. A debugging tool should display both."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. An apparently excellent handwritten-digit result"}</H3>

<Prose>{"A colleague puts all 300 validation images and their labels into the memory bank, then reports near-perfect validation accuracy. Why does that not answer our original evaluation question?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Identify which information would be unavailable for a new query."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The bank now contains the evaluation images' correct labels and exact self-matching keys. It evaluates a predictor with information unavailable for a new unlabeled image. Restore the fit-only memory bank and redo the validation protocol. A genuinely transductive task would require a separately stated information boundary; it cannot expose the unknown query labels as values."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. A noisy cue with the wrong winner"}</H3>

<Prose>{"The intended memory's score is 0.8 and an incorrect memory's score is 0.9. What happens to their relative weight as β increases?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write the ratio of their exponentials."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The incorrect-to-intended ratio is exp(0.1β), which grows with β. Sharpening strengthens the wrong winner. Improving the representation, acquiring more cue information or changing the decision rule may help; temperature cannot reverse this score ordering."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Choose the metric before judging the image"}</H3>

<Prose>{"A reconstruction cuts mean squared error from 0.12 to 0.06 but changes a correctly recognized 7 to a 1. Is this improvement?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"State the intended task and distinguish two valid measurements."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"It improves average squared pixel reconstruction on this example and worsens classification. For a recognition system, report the class failure. For a reconstruction task, inspect whether the metric misses a perceptually or semantically important stroke. A useful report gives both measurements and explains the mismatch rather than choosing whichever makes the method look better."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Bound the competing mass"}</H3>

<Prose>{"A bank has 11 unit-norm memories. The desired memory beats every other score by at least 2, and β = 1.5. Give a lower bound for its weight and an upper bound for the read's distance to it."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use a = (P−1)exp(−βδ)."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"a = 10exp(−3) ≈ 0.497871. Target weight is at least 1/(1+a) ≈ 0.667614. The tighter distance bound is 2a/(1+a) ≈ 0.664771. This is a sufficient bound based on the assumed score gap, not the exact error of an unspecified bank."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Test energy reasoning against an update"}</H3>

<Prose>{"A system evolves both queries and keys through unrelated learned projections and residual updates. Its authors plot the fixed-X energy from §3 at each layer and claim it must decrease. What must they establish?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Ask whether the same scalar function is being minimized and whether every update is derived from it."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"They must define a shared state and energy, include dependencies through changing keys as well as queries, and show that the actual update lowers that energy under its assumptions. Recomputing a different fixed-bank energy at every layer does not prove descent of one objective. Residuals, projections, normalization and finite step sizes need their own treatment."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"10. Plan a meaningful extension of the digit experiment"}</H3>

<Prose>{"You can improve robustness by training with masks. Specify a fair experiment and a decision rule without using the existing test results to select a favorable mask."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate the augmentation design, validation criteria and final assessment."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"One acceptable plan predefines a distribution of missing-cell masks using fitting data only; trains both clean and mask-augmented projections with the same memory bank, parameter count and optimization budget; selects epochs using a predefined combination of clean and masked validation losses; and evaluates the frozen models on a separately specified test corruption protocol plus clean test images. Report both seeds, error counts and reconstruction metrics. Preserve the original images and masks so another learner can reproduce the comparison. More robust results on one mask family do not establish robustness to every handwriting distortion."}</Prose>

</details>

<Prose>{"You are ready to move on when you can distinguish state refinement from parameter training, trace keys through weights to values, explain a failed retrieval without appealing to a vague capacity claim, and preserve the learning boundary in a memory-based experiment."}</Prose>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/xlstm-extended-lstm?module=deep-learning-fundamentals"}>{"xLSTM (Extended LSTM)"}</a>{". It returns to recurrent sequence memory and asks how changing gates and scalar or matrix state changes what a model can retain and read. That is a different storage/update contract from retaining every row of an explicit reference bank."}</Prose></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References & another way to learn it"}</H2>

<ul><li>{""}<a href={"https://arxiv.org/html/2008.02217v3"}>{"Ramsauer et al., Hopfield Networks is All You Need"}</a>{" — the main continuous-memory reference. Read §2 after the worked energy calculation, §3 for layer choices, and Appendix A.1.5–A.1.6 for precise stability and capacity assumptions. The introduction's one-update language is made precise in the theorems."}</li><li>{""}<a href={"https://ml-jku.github.io/hopfield-layers/"}>{"Johannes Brandstetter and the JKU authors, Hopfield layers illustrated article"}</a>{" — an alternate visual explanation of memory retrieval, temperature, keys/queries/values and pooling. Its familiar-image examples help establish the geometry; pair its informal convergence wording with the paper's exact statements."}</li><li>{""}<a href={"https://arxiv.org/html/1606.01164v2"}>{"Krotov & Hopfield, Dense Associative Memory for Pattern Recognition"}</a>{" — study §§2–3 for energy-difference updates and the parity example, and §§4–5 for learned features/prototypes and the feedforward interpretation."}</li><li>{""}<a href={"https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/"}>{"Hopfield, Neural networks and physical systems with emergent collective computational abilities"}</a>{" — the 1982 historical starting point. The archive provides the original scanned article and describes content-addressable memory and asynchronous dynamics."}</li><li>{""}<a href={"https://authors.library.caltech.edu/records/q92rz-95p89"}>{"McEliece et al., The capacity of the Hopfield associative memory"}</a>{" — a more mathematical resource separating exact recovery of most memories from exact recovery of all memories. Useful when evaluating a capacity claim."}</li><li>{""}<a href={"https://github.com/ml-jku/hopfield-layers"}>{"Official Hopfield layers code and examples"}</a>{" — compare the three module interfaces and study the bit-pattern and latch-sequence notebook descriptions. Treat its documented older dependency environment as a research-package detail; the notebook experiments were not executed for this lesson."}</li><li>{""}<a href={"https://www.youtube.com/watch?v=nv6oFDp6rNQ"}>{"Yannic Kilcher, Hopfield Networks is All You Need — Paper Explained"}</a>{" — optional advanced paper walkthrough, also linked by the authors' article. The title and author link were verified; the video itself was not reviewed here, so use the paper and checked examples for the technical guarantees."}</li><li>{""}<a href={"https://arxiv.org/abs/2007.13505"}>{"Widrich et al., Modern Hopfield Networks and Attention for Immune Repertoire Classification"}</a>{" — read the Deep Repertoire Classification section to see how the unit of supervision changes from one sequence to a large set of sequences."}</li><li>{""}<a href={"https://arxiv.org/abs/2206.00664"}>{"Schäfl et al., Hopular"}</a>{" — §3 explains the two memory roles in tabular refinement. Follow the distinction between sample-to-sample and feature-to-feature retrieval."}</li><li>{""}<a href={"https://arxiv.org/html/2302.07253v1"}>{"Hoover et al., Energy Transformer"}</a>{" — an advanced extension. §2 explicitly derives dynamics with changing token representations and explains why its energy attention differs from ordinary attention."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI digit dataset"}</a>{" and "}<a href={"/learn-code/modern-hopfield-networks/./data-provenance.md"}>{"local provenance"}</a>{" — original acquisition, count features, train/test writer split, license, exact downloads and our separate experimental roles."}</li></ul></section>
</div>};
