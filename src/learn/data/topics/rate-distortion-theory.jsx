import { H2, H3, Prose, Code } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample as CompleteExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { CompressionChannelFigure, BinaryCodebookLab, BinaryFrontierLab, RateDistortionOptimizerLab, GaussianAllocationLab, FidelityMarginalFigure, LearnedCodecFigure } from '../../components/lesson-labs/RateDistortionLabs.jsx';
import { rateDistortionExamples as examples } from '../rate-distortion-examples.js';
function Example({
  example
}) {
  return <section><Prose><strong>Before running.</strong> {example.question}</Prose><CompleteExample example={example} /></section>;
}
function Practice({
  title,
  prompt,
  hint,
  children
}) {
  return <section className="lesson-check"><h3>{title}</h3><Prose>{prompt}</Prose><details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Show explained solution</summary>{children}</details></section>;
}
export default {
  title: 'Rate-Distortion Theory',
  readTime: '~70 min read + 2–3 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot rate-distortion-lesson">
    <LessonIntro prerequisites="Probability, entropy and mutual information from the preceding lessons; weighted averages, logarithms and basic derivatives. We refresh conditional tables and explain the optimization steps locally. Gaussian variance is reviewed when needed; no compression library or prior codec implementation is assumed." sections={[['1-send-an-index-and-reconstruct-a-block', 'Build a tiny compressor'], ['2-define-the-budget-before-the-bound', 'Rate, error and information'], ['3-derive-the-binary-frontier', 'Binary limits and source bias'], ['4-explain-what-long-blocks-can-achieve', 'Coding theorem and limits'], ['5-compute-a-finite-reconstruction-rule', 'Tradeoff optimization'], ['6-allocate-error-for-gaussian-data', 'Gaussian rate and allocation'], ['7-choose-what-must-survive', 'Fidelity, realism and rare events'], ['8-turn-a-representation-into-a-bitstream', 'Actual bytes and learned codecs'], ['9-practise-the-compression-decision', 'Independent practice']]}>A remote sensor has more readings than it can transmit. Sending fewer bits is easy if we discard the readings; the useful question is how much of their meaning can survive a particular bit budget. Begin with an encoder and decoder you can run, derive a lower limit for their error, then learn when a measured system can fairly be compared with that limit.</LessonIntro>

    <H2>1. Send an index and reconstruct a block</H2>
    <Prose>Suppose a sensor reports three binary readings, such as <Code>011</Code>. An exact message would use three bits to distinguish all eight possible blocks. Instead, agree in advance that the receiver can reconstruct only <Code>000</Code> or <Code>111</Code>. These two allowed outputs form a <strong>codebook</strong>. The encoder chooses the closer one and sends its index: 0 for the first, 1 for the second.</Prose>
    <Prose>For 011, output 111 changes one position; output 000 changes two. Sending index 1 therefore reconstructs 111. The receiver never learns that the first position was originally 0. That missing distinction is the information deliberately lost. The encoder's comparison and the decoder's lookup together form a complete <strong>lossy code</strong>.</Prose>
    <CompressionChannelFigure />
    <Prose>If each input bit is independently 0 or 1 with equal probability, every three-bit block has probability 1/8. Two blocks, 000 and 111, are exact. The other six each acquire one wrong bit. The expected fraction of wrong positions is therefore 6 divided by the 8 ×3 original positions: .25. The payload rate is one transmitted bit divided by three source bits: 1/3.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
R_{\rm code}=\frac{1}{3}\ \text{bits/source bit},\\
D_{\rm code}=\frac{2(0)+6(1)}{8\cdot3}=\frac14.
\end{gathered}`}</MathBlock>
    <Prose>That average does not say that every block has 25% error. Here individual blocks have either zero or 1/3 error. In other codes, a low average can coexist with much worse rare outcomes. Also, the one-bit rate assumes that the codebook and block boundaries are already shared. A file may need additional information to tell the receiver its length and format.</Prose>
    <BinaryCodebookLab />
    <Prose><strong>Run the examples.</strong> Use Python 3.12 or a compatible Python 3 release. Save each complete program in its own <Code>.py</Code> file and run it with <Code>python filename.py</Code>. These examples use only the standard library. The browser's investigations calculate their bounded models here; the Python programs are independent scripts for your own environment.</Prose>
    <Example example={examples.blockCode} />
    <Prose>The printed rows are the entire experiment, not a random sample. Their probabilities are equal for this particular source, so counting is enough. For a biased source, multiply each row's error by that row's probability instead. The same codebook can have a different average distortion without changing its transmitted length.</Prose>

    <H2>2. Define the budget before the bound</H2>
    <Prose>A <strong>source</strong> is the random object being represented. X denotes one source symbol, and X̂, pronounced “X hat,” denotes its reconstruction. A <strong>distortion function</strong> d(x,x̂) assigns a nonnegative cost to a particular mistake. Binary Hamming distortion is 0 for a match and 1 for a mismatch. Squared error, (x−x̂)², penalizes a numerical error of 2 four times as much as an error of 1.</Prose>
    <Prose>Distortion does not have to be a mathematical distance. It can be asymmetric, task-weighted or measured in squared units. The choice defines which mistakes the theory treats as expensive. A distortion budget D normally constrains the <em>expectation</em> of that cost. A maximum-error bound or a bound on the probability of a severe error is a different requirement.</Prose>
    <LessonTable caption="Three quantities that answer different questions." headers={['Quantity', 'Meaning', 'Example']} rows={[['Fixed-length rate', 'ceil(log₂ M)/n transmitted bits per source symbol for an M-word block code of length n', 'Two reconstructions for three bits: 1/3'], ['Average code length', 'Expected encoded length per symbol under a declared coding/framing scheme', 'Variable-length entropy coding of frequently repeated indices'], ['Expected distortion', 'Probability-weighted cost of reconstructing the source', 'Mean wrong-bit fraction .25']]} />
    <Prose>To find the theoretical frontier, consider every possible conditional distribution q(x̂|x). For each input value it gives a probability distribution over reconstructions. This is called a <strong>test channel</strong>: a mathematical description of what is retained. It need not be a physical noisy channel, and a table of these probabilities is not itself an encoder that emits a bitstream.</Prose>
    <Prose>Fix the source probabilities p(x). The joint probability of an input/output pair is p(x)q(x̂|x). Summing a column produces the output probability r(x̂). Unlike ordinary optimal transport, where both marginals were fixed, this optimization generally chooses the output marginal as well as the pairing.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
P(x,\hat x)=p(x)q(\hat x\mid x),\\
r(\hat x)=\sum_xp(x)q(\hat x\mid x),\\
D(q)=\sum_{x,\hat x}p(x)q(\hat x\mid x)d(x,\hat x).
\end{gathered}`}</MathBlock>
    <Prose><strong>Mutual information</strong> measures how much knowing the reconstruction tells us about the source. The preceding lesson derived it as a comparison of the joint law with the product of its marginals. In bits, it is the weighted average of log₂(q/r). A zero-probability pair contributes 0; a reconstruction used by a positive-probability input necessarily has positive marginal probability.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
I(q)=\\
\sum_{x,\hat x}p(x)q(\hat x\mid x)
\log_2\frac{q(\hat x\mid x)}{r(\hat x)},\\
R(D)=\min_{q:\,D(q)\le D}I(q).
\end{gathered}`}</MathBlock>
    <Prose>This finite-alphabet formula defines the <strong>information rate-distortion function</strong>. It asks for the least retained information among all allowed reconstruction rules. For standard independent, identically distributed sources with additive distortion, the coding theorem connects this information minimum to asymptotically attainable coding rates. We will make those assumptions and the endpoint qualification explicit.</Prose>
    <Prose>Three facts follow before solving anything. First, increasing an allowed error budget only adds feasible choices, so R(D) cannot increase. Second, R(D) is nonnegative because mutual information is. Third, zero rate is possible once the decoder can ignore the input and emit a best constant reconstruction.</Prose>
    <MathBlock>{String.raw`D_{\rm zero}
=\min_{\hat x}\sum_xp(x)d(x,\hat x).`}</MathBlock>
    <Prose>For a finite reproduction alphabet, randomizing independently among constants cannot beat the best constant's expected distortion: it merely averages their costs. For fair bits, either constant is wrong half the time, so the zero-rate threshold is .5. If ones occur only 10% of the time, always outputting 0 incurs .1 error and uses no source-dependent bits. It also misses every 1.</Prose>
    <Checkpoint prompt="A binary alarm occurs in1% of readings. A designer proposes sending nothing and always reconstructing0. Does average Hamming error below2% certify that alarms survive?">
      <Prose>No. This reconstruction has 1% overall error and a 100% miss rate conditional on an alarm. It satisfies that average requirement precisely because the requirement assigns small probability weight to alarms. Add an appropriate conditional constraint or cost if preserving them is essential; the original R(D) calculation cannot silently enforce a different requirement.</Prose>
    </Checkpoint>

    <H2>3. Derive the binary frontier</H2>
    <Prose>Let X be a bit with P(X=1)=p, and use Hamming distortion. Write Hᵦ(p) for its binary entropy. For fair bits, Hᵦ(.5)=1 bit. Define the error bit E=X xor X̂: it is 1 exactly when the reconstruction is wrong. If the actual error probability is d, E has entropy Hᵦ(d).</Prose>
    <Prose>Once X̂ is known, specifying E is equivalent to specifying X: flip X̂ when E=1. Therefore H(X|X̂)=H(E|X̂). Conditioning cannot increase entropy, so this uncertainty is at most H(E). The information retained must consequently be at least Hᵦ(p)−Hᵦ(d).</Prose>
    <MathBlock>{String.raw`\begin{gathered}
H(X\mid\hat X)=H(E\mid\hat X)\le H_b(d),\\
I(X;\hat X)\ge H_b(p)-H_b(d).
\end{gathered}`}</MathBlock>
    <Prose>For budgets D below min(p,1−p), we have d≤D≤.5, where binary entropy is increasing. Replacing Hᵦ(d) with Hᵦ(D) gives a lower bound valid for every feasible rule. To prove it is the minimum, we must also construct a rule that reaches it; a lower bound alone is not enough.</Prose>
    <Prose>Construct the variables <em>backward</em>. Draw a reconstruction bit with probability r of being 1. Independently draw an error bit that is 1 with probability D, then set X=X̂ xor E. Independence makes H(E|X̂)=H(E), so the bound becomes equality. We still need the resulting X to have the required source probability p.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
p=r(1-D)+(1-r)D,\\
r=\frac{p-D}{1-2D}.
\end{gathered}`}</MathBlock>
    <Prose>When 0≤D&lt;min(p,1−p), r lies between 0 and 1, so this is a valid construction. At the zero-rate threshold, use a most-probable constant reconstruction directly; this also avoids the indeterminate expression at p=D=.5. The complete result, including large budgets, is:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
D_{\rm zero}=\min(p,1-p),\\
R(D)=H_b(p)-H_b(D)\\
\text{for }0\le D<D_{\rm zero},\\[4pt]
R(D)=0\quad\text{for }D\ge D_{\rm zero}.
\end{gathered}`}</MathBlock>
    <Prose>For p=.5, allowing 10% average error gives 1−Hᵦ(.1)≈.531 bits per source bit. Allowing 25% gives≈.189. Our actual three-bit code used 1/3 bit at 25% error, so it lies above the limit. The gap says that this code is not asymptotically optimal; it does not invalidate the bound.</Prose>
    <Prose>The original program below remains useful on its explicitly limited sample budgets, all between 0 and .5. <strong>Do not extend its expression to budgets above .5.</strong> Hᵦ(D) falls again after .5, so <Code>max(0, 1-H_b(D))</Code> would rise again. The correct rate remains 0 because the constant reconstruction is still allowed.</Prose>
    <Example example={examples.originalBinary} />
    <Prose>A biased source makes the backward/forward distinction visible. With p=.2 and D=.1, r=.125. The joint masses are .7875, .0125, .0875 and .1125 when rows are input 0/1 and columns are output 0/1. They preserve input probabilities .8/.2 and give total error .0125+.0875=.1.</Prose>
    <Prose>But the conditional false-positive probability is .0125/.8=.015625, while the false-negative probability is .0875/.2=.4375. They are not equal. Adding an independent 10% flip to the <em>source</em> would be a forward binary symmetric channel; for this biased source it is generally not the minimum-information rule. “Independent error” must say independent of which variable.</Prose>
    <Example example={examples.biasedChannel} />
    <Prose>A common engineering choice is to minimize R+λD rather than set a hard D first. Here λ is measured in bits per distortion unit; increasing it makes error more expensive. For an interior binary optimum, differentiate Hᵦ(p)−Hᵦ(D)+λD. Since Hᵦ′(D)=log₂((1−D)/D), the minimizing distortion is 1/(1+2^λ), capped by the zero-rate threshold.</Prose>
    <MathBlock>{String.raw`D_\lambda=
\min\left\{D_{\rm zero},\,\frac{1}{1+2^\lambda}\right\}.`}</MathBlock>
    <Prose>For fair bits and λ=2, Dλ=.2. The green marker in the next figure is this operating point; the amber marker is your separately selected error budget. A line R+λD=constant has slope−λ. Moving it down until it touches the convex frontier explains the same choice geometrically. At λ=0, any feasible zero-rate point minimizes the objective; the marker uses the smallest distortion that reaches zero rate.</Prose>
    <BinaryFrontierLab />

    <H2>4. Explain what long blocks can achieve</H2>
    <Prose>Why does mutual information appear in a statement about code length? Let an encoder turn n source symbols Xⁿ into an index W chosen from M possibilities, and let the decoder turn W into X̂ⁿ. No source information can reach the output except through W. Entropy bounds the information carried by that index, and an M-valued variable has entropy at most log₂M.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
X^n\longrightarrow W\longrightarrow\hat X^n,\\
\log_2M\ge H(W)\ge I(X^n;W),\\
I(X^n;W)\ge I(X^n;\hat X^n).
\end{gathered}`}</MathBlock>
    <Prose>The last step is the data-processing inequality. For independent source symbols, the information in the entire reconstructed vector is at least the sum of the component mutual informations. One way to see this is to expand H(Xⁿ) as the sum of H(Xᵢ), then use H(Xⁿ|X̂ⁿ)≤ΣH(Xᵢ|X̂ᵢ). A component's information is at least the rate-distortion minimum for its own error Dᵢ.</Prose>
    <Prose>The function is convex as well as nonincreasing. With a fixed source, mutual information is convex in the conditional channel; mixing two channels gives the weighted average distortion and no more than the weighted average information. Operationally, use one block code for a chosen fraction of blocks and the other for the rest according to a shared schedule. Their long-run rate and distortion average. This time-sharing explains why a straight line between achievable points is achievable asymptotically.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
I(X^n;\hat X^n)\ge\sum_{i=1}^n I(X_i;\hat X_i),\\
\sum_i I(X_i;\hat X_i)\ge\sum_i R(D_i),\\
\sum_iR(D_i)\ge nR\!\left(\frac1n\sum_iD_i\right).
\end{gathered}`}</MathBlock>
    <Prose>Combining the inequalities gives log₂M/n≥R(D) when mean distortion is at most D. This is a <strong>converse</strong>: no code under these assumptions beats the information bound. A finite fixed-width index actually uses ceil(log₂M) bits, which is at least log₂M; entropy coding uses a different expected-length argument.</Prose>
    <Prose>The other direction is <strong>achievability</strong>. For a memoryless source and separable distortion, choose many reconstruction blocks independently from the marginal distribution of a good test channel. A typical source block has an exponentially small chance, roughly 2⁻ⁿᴵ, of finding any particular random reconstruction jointly compatible with it. Having roughly 2ⁿᴿ candidates with R&gt;I makes it increasingly likely that at least one works. The encoder searches for a sufficiently good candidate and sends its index.</Prose>
    <Prose>The probability idea itself is elementary: if each independent candidate succeeds with probability a, all M fail with probability(1−a)ᴹ≤e⁻ᴹᵃ. Typicality supplies the information-theoretic size of a and controls distortion. The full theorem also handles atypical source blocks; for unbounded costs it needs appropriate integrability rather than silently treating them as harmless.</Prose>
    <Prose>In the standard iid setting with nonnegative additive distortion, finite expected cost for some constant reconstruction, and D strictly above the minimum achievable distortion, the theorem identifies the operational asymptotic rate with the information minimum. For a finite alphabet and bounded distortion this is the familiar block-coding result. Rates can approach the frontier with long enough blocks and arbitrarily small slack; it is not a claim that every finite block meets exactly the limiting pair. The proof establishes existence, not a cheap search algorithm: enormous codebooks can be computationally impractical.</Prose>
    <details><summary>Deeper: why exact zero error is a delicate endpoint</summary>
      <Prose>Take biased iid bits with 0&lt;p&lt;1. Every one of the 2ⁿ length-n strings has positive probability. If a finite deterministic fixed-length code must make exactly zero Hamming error, it must distinguish them all: M≥2ⁿ, hence at least 1 bit per source bit. Yet Hᵦ(p) can be less than 1.</Prose>
      <Prose>There is no contradiction. The information expression R(0)=Hᵦ(p) describes the limiting boundary, approached by fixed-rate codes with vanishing allowed distortion/error. It is not the finite exact-zero-error requirement just imposed. Variable-length lossless codes can instead assign long descriptions to rare strings and approach Hᵦ(p) in <em>expected</em> length while remaining exact. Always name the rate convention, error criterion and order of limits.</Prose>
    </details>

    <H2>5. Compute a finite reconstruction rule</H2>
    <Prose>Most source/distortion choices lack the binary closed form. With a finite source and a declared finite reconstruction alphabet, we can minimize I(q)+λD(q) numerically. This is a convex optimization in q for fixed p, fixed costs and λ≥0. It differs from training a neural encoder or the general information-bottleneck objective, whose parameterizations can introduce other optimization difficulties.</Prose>
    <Prose>Introduce a temporary output guess r. Using r in place of q's actual marginal changes I into an upper expression. In natural-log units, let s=λln 2 and write J(q,r) for the average log(q/r) plus s times distortion. Its relation to the true objective follows from the KL chain identity:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
J(q,r)=\\
\sum_{x,j}p_xq_{j|x}
\left(\ln\frac{q_{j|x}}{r_j}+s\,d_{xj}\right),\\
J(q,r)=I_{\rm nats}(q)+sD(q)\\
+D_{\rm KL}(r_q\Vert r).
\end{gathered}`}</MathBlock>
    <Prose>The notation r_q means the actual output marginal produced by q. For a fixed conditional table, the best guess is therefore its own marginal, making the extra KL zero. For a fixed guess r, minimize each conditional row subject to summing to 1. Differentiating with a row-normalization multiplier gives ln(q/r)+1+s d+constant=0. Exponentiating and normalizing yields:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
Z_x=\sum_jr_j e^{-s d_{xj}},\\
q_{j|x}=\frac{r_j e^{-s d_{xj}}}{Z_x},\\
r^{\rm next}_j=\sum_xp_xq_{j|x}.
\end{gathered}`}</MathBlock>
    <Prose>A reconstruction receives more conditional probability when it is already common and when its distortion for this input is low. Replacing the guess by the resulting marginal lets the entire source distribution influence which reconstructions remain useful. Repeating these two minimizations is the rate-distortion form of the <strong>Blahut–Arimoto algorithm</strong>.</Prose>
    <Prose>Use positive starting probabilities for every reconstruction you want considered. If rⱼ starts at 0, the conditional numerator for that column remains 0 and the update cannot revive it. Small nonzero values and exact zeros are different. The implementation uses logarithms, subtracts each row's log-sum-exp normalizer, and retains log marginals, so very small intermediate probabilities need not be discarded prematurely.</Prose>
    <Prose>Neither a small change between iterations nor a fixed loop count proves that the full optimum has been found. The investigation reports an achieved objective and a lower bound for the entire declared alphabet. Their gap bounds how much the objective can improve. It does not separately certify the rate and distortion coordinates, and it does not certify that the chosen distortion describes a useful application.</Prose>
    <RateDistortionOptimizerLab />
    <details><summary>Deeper: derive the displayed optimization bound</summary>
      <Prose>For fixed r, substituting the optimal conditional rows makes J equal f(r)=−ΣpₓlnZₓ. Minimizing f over output distributions is equivalent to the original finite problem. Define gⱼ=Σpₓe⁻ˢᵈˣʲ/Zₓ. For any alternative output distribution u, its corresponding normalizer is Zₓ(u). Jensen's inequality for−ln gives:</Prose>
      <MathBlock>{String.raw`\begin{gathered}
f(u)-f(r)
=-\sum_xp_x\ln\frac{Z_x(u)}{Z_x(r)},\\
f(u)-f(r)\ge-\ln\!\left(\sum_ju_jg_j\right),\\
f(u)\ge f(r)-\ln\max_jg_j.
\end{gathered}`}</MathBlock>
      <Prose>Because the entries of u sum to 1, their weighted average of g cannot exceed the largest g. Thus the last expression is a lower bound even when u puts mass on a column omitted by our initialization. Divide by ln 2 to express it in bits. The achieved conditional table's I+λD is an upper bound. A negative lower bound is allowed and simply loose; the objective itself is nonnegative.</Prose>
      <Prose>In exact arithmetic the alternating objective decreases, and positive-support BA converges to the finite convex optimum under the usual finite-cost setup. Floating-point arithmetic and a finite loop cap need explicit checks. The code uses the computed gap as a numerical stopping certificate, with independent native optimization checks for these examples. It is not a formally interval-arithmetic-certified implementation.</Prose>
    </details>
    <Example example={examples.finiteOptimizer} />
    <Prose>For fair bits at λ=2, the positive start immediately gives D=.2, I≈.278072 and objective≈.678072. Starting with only output 0 gives D=.5, I=0 and objective 1. The state remains unchanged, but its bound gap stays positive. That is a useful diagnostic: numerical stillness is not optimality.</Prose>
    <Prose>Changing λ explores supported points of the convex frontier. At a corner or a linear segment, the minimizing channel may not be unique; do not infer one unique distortion for every λ. Adding more reconstruction symbols can improve the finite-alphabet optimum, but optimizing within a selected grid does not prove optimality over every possible continuous reconstruction. A larger λ also demands more care with numerical scales.</Prose>

    <H2>6. Allocate error for Gaussian data</H2>
    <Prose>Now let X be a Gaussian numerical measurement with mean μ and variance σ². Variance is the average squared deviation from the mean. Under squared-error distortion, reconstructing μ every time costs σ² and requires no source-dependent bits. Thus D≥σ² is again a zero-rate regime.</Prose>
    <Prose>For 0&lt;D&lt;σ², the Gaussian rate-distortion function is half the base 2 logarithm of the variance-to-error ratio. The ratio is dimensionless: numerator and denominator must use the same squared units. This is an information limit for the Gaussian model, not the rate of every scalar quantizer.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
R(D)=\tfrac12\log_2(\sigma^2/D)\\
\text{for }0<D<\sigma^2,\\[4pt]
R(D)=0\quad\text{for }D\ge\sigma^2.
\end{gathered}`}</MathBlock>
    <Prose>Here is the lower-bound reasoning. Write the reconstruction error as E=X−X̂. Given X̂, translating X by that known value does not change its conditional differential entropy, so h(X|X̂)=h(E|X̂)≤h(E). Among continuous distributions with a specified variance, a Gaussian has the largest differential entropy. Also Var(E)≤E[E²]≤D.</Prose>
    <Prose>That maximum-entropy step can be checked using KL. Compare an error density with the Gaussian having its same mean and positive variance. The Gaussian's negative log density is a constant plus a squared deviation, so its expected value depends only on those two moments. The cross-entropy therefore equals the matching Gaussian's entropy. Since KL is cross-entropy minus the error density's entropy and cannot be negative, the error density's entropy cannot be larger.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
h(X)=\tfrac12\log_2(2\pi e\sigma^2),\\
h(E)\le\tfrac12\log_2(2\pi eD),\\
I(X;\hat X)\ge\tfrac12\log_2(\sigma^2/D).
\end{gathered}`}</MathBlock>
    <Prose>The entropy calculation applies where the densities and entropies are well-defined; the information statement handles limiting cases through its KL formulation. To attain equality for positive D below σ², construct X backward as X̂+E, where X̂ is Gaussian with mean μ and variance σ²−D and E is an independent zero-mean Gaussian of variance D. Their sum has exactly the required source law, and the independent error makes both inequalities tight.</Prose>
    <Prose>Independent error is again backward, not a license to add arbitrary independent noise to X. For the attaining joint Gaussian law, the forward conditional mean is μ+(1−D/σ²)(X−μ), and its conditional variance is D(1−D/σ²). The reconstruction shrinks toward the known mean as information is discarded.</Prose>
    <Prose>For σ²=9 and D=2.25, the limit is 1 bit per measurement. A simple one-bit scalar quantizer that sends the sign of X−μ is worse: with its optimal conditional-mean reconstruction levels μ±σ√(2/π), its MSE is σ²(1−2/π), about 3.2704. The Gaussian rate-distortion bound becomes accessible through long-block coding; one scalar decision need not attain it.</Prose>
    <Prose>If σ²&gt;0, demanding exactly zero squared error sends the ideal information rate to infinity. A non-atomic continuous value has arbitrarily fine distinctions. This does not say a computer file already stored in a finite floating-point format has infinite length: that file is a discrete representation of the measurement. A source with variance 0 is already constant and needs zero rate even at zero error.</Prose>
    <H3>Spend a total error budget across independent components</H3>
    <Prose>Suppose each sample is now a vector of independent Gaussian components with variances v₁,…,vₖ. Define D_total as the <em>sum</em> of their mean squared errors, not the average. Rates add in bits per vector. For each component, error above its variance buys no further rate reduction, because that component can already be reconstructed by its mean.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
D_i=\min(v_i,\theta),\\
\sum_iD_i=D_{\rm total},\\
R_{\rm vector}=\frac12
\sum_{i:\,v_i>\theta}\log_2\frac{v_i}{\theta}.
\end{gathered}`}</MathBlock>
    <Prose>For a budget below the sum of variances, choose the common level θ to satisfy the sum. This is <strong>reverse water-filling</strong>. To derive it, minimize the sum of the Gaussian component rates with a multiplier for total error. On each component still receiving positive rate, the derivative is−1/(2ln 2 Dᵢ). Equal marginal benefit requires equal Dᵢ. Components whose variance is below that level cap out at their variance.</Prose>
    <Prose>With variances 9 and 1 and total budget 3, equal unconstrained error 1.5 would waste .5 on the second component: it can reach zero rate already at error 1. Instead allocate 2 to the first and 1 to the second. The rate is .5 log₂(9/2)≈1.084963 bits per vector. Dividing by 2 gives average bits per component; the two component rates themselves are not equal.</Prose>
    <GaussianAllocationLab />
    <Example example={examples.gaussianAllocation} />
    <details><summary>Deeper: why covariance eigenvectors can help, and when they do not suffice</summary>
      <Prose>A jointly Gaussian vector can have correlated coordinates. An orthogonal eigenvector transform of its covariance preserves total squared error and makes its transformed Gaussian components independent. Apply reverse water-filling to the covariance eigenvalues, then transform reconstructions back. This links the lesson to matrix decompositions and transform coding.</Prose>
      <Prose>For a non-Gaussian source, uncorrelated transformed coordinates need not be independent, and an MSE-based allocation can discard low-variance but task-critical information. The Gaussian formula may guide a model or a bound under additional conditions; a covariance matrix alone does not establish that it is the true rate-distortion function of the data.</Prose>
    </details>

    <H2>7. Choose what must survive</H2>
    <Prose>The theory answers the distortion question you give it. If the question is wrong, solving its optimization accurately will not repair the application. Pixel squared error, speech intelligibility, retrieval ranking and preserving a rare sensor event can favour different reconstructions. Choose the criterion by the decision the reconstruction must support, then test its failure cases.</Prose>
    <Prose>Why can minimizing MSE average away a detail? Suppose the received representation is Z and the decoder chooses a number a(Z). Conditional on a particular Z, let m=E[X|Z] be the mean of still-plausible values. Expanding the square around m makes the cross-term zero:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\mathbb E[(X-a)^2\mid Z]\\
=\operatorname{Var}(X\mid Z)
+\bigl(\mathbb E[X\mid Z]-a\bigr)^2.
\end{gathered}`}</MathBlock>
    <Prose>Only the second term depends on the chosen reconstruction, so the MSE-optimal answer is the conditional mean. If a lost detail could plausibly be−1 or+1 with equal probability, the mean is 0, even when 0 is not a plausible original detail. For a vector, the same argument applies coordinate by coordinate under summed squared error; averaging possible images can remove texture.</Prose>
    <FidelityMarginalFigure />
    <Prose>In the figure, always producing 0 incurs squared error 1. Producing an independent fair−1 or+1 looks correct in aggregate: its output distribution exactly matches the source. But it has error 0 on half the pairs and error 4 on the other half, so MSE 2. Both schemes transmit zero source information. Matching a marginal law does not establish that a particular reconstruction matches its actual input.</Prose>
    <Example example={examples.fidelity} />
    <Prose>The rate-distortion-perception literature adds a discrepancy between the source and output distributions as a further constraint. This sharpens one aspect of realism but is not a universal mathematical model of human judgment. Its operational coding statements depend on the chosen formulation, including how randomness and distributional constraints are supplied. Our zero-bit example only needs local randomness at the decoder; it does not assert a general coding theorem for every perception constraint.</Prose>
    <Prose>A complementary issue is rare-event weighting. The optimizer's two three-level scenarios keep the same source probabilities but multiply the high-input row's miss costs by 10. That explicitly changes what the objective rewards. It is not evidence that 10 is an appropriate weight for a real sensor. If a guarantee is essential, impose and evaluate the actual conditional or worst-case requirement instead of assuming a weighted average supplies it.</Prose>
    <Checkpoint prompt="A compressor's overall error falls after deployment, but it begins missing every reading in a rare high-value category. What additional numbers would distinguish an actual improvement from a changed mix of inputs?">
      <Prose>Report the category frequencies and error conditional on each category, using the same error definition and held-out evaluation protocol. Reweighting the category errors to a fixed reference mixture separates within-category changes from a changed mixture. Check the application's actual miss constraint as well as the aggregate; the mean alone cannot reveal whether the rare category was preserved.</Prose>
    </Checkpoint>

    <H2>8. Turn a representation into a bitstream</H2>
    <Prose>An autoencoder with eight latent numbers has eight coordinates, not a declared eight-bit message. Each coordinate may have many possible values. A codec needs a finite representation, an encoding of that representation and a decoder that agrees on all conventions. Counting an in-memory tensor's shape or displaying a training loss does not measure those bytes.</Prose>
    <LearnedCodecFigure />
    <Prose>A common learned-compression pipeline transforms data into latent values, quantizes them, losslessly entropy-codes the discrete indices, and reconstructs with a synthesis transform. If the true index distribution is P but the code uses model Q, the idealized expected log-loss rate is H(P)+KL(P∥Q), in bits when base 2 is used. Actual coding incurs additional framing and finite-code overhead. The prior entropy lesson supplies that mismatch mechanism.</Prose>
    <Prose>Quantization creates a training difficulty: rounding is locally constant almost everywhere, so its ordinary derivative does not provide a useful gradient. Ballé, Laparra and Simoncelli's 2017 formulation used additive uniform noise as a continuous training relaxation, then evaluated actual quantized, entropy-coded bitstreams. That example explains why a training surrogate and a deployed coding operation must be assessed separately; it is not a current codec ranking.</Prose>
    <Prose>For a VAE-style latent model, expected KL from an encoder q(z|x) to a prior r(z) decomposes into I(X;Z)+KL(q(z)∥r(z)) under the relevant finite-information assumptions. It can therefore be an information upper bound. Interpreting it as actual compression requires a coding construction and its overheads; neither a KL penalty nor a latent dimension is a bitstream by itself. Information Bottleneck focuses on preserving information about a target Y, which may intentionally discard detail that reconstruction distortion would retain.</Prose>
    <Prose>Even our tiny code needs a protocol. The next program stores the number of original source bits in a four-byte header, packs one codebook index per block into a payload, and decodes using only those bytes and the fixed shared codebook. For the 24-bit demonstration, the payload is one byte, but the complete file is five bytes. The original bits could be packed into three bytes. This example compresses the <em>payload representation</em> and enlarges this tiny file once its declared header is counted.</Prose>
    <Example example={examples.bitstream} />
    <Prose>The header lets the decoder remove padding from a final partial block. Payload length and unused bits are checked, but this is still a teaching format: it provides no corruption checksum, format negotiation or high-performance entropy coder. A longer stream amortizes this fixed header; a changed codebook or model would require a shared version or additional transmitted description. Always report which costs are included.</Prose>
    <Prose>The related <a href="/learn/path/full-curriculum/sampling-aliasing-and-quantization-in-neural-recordings?module=computational-neuroscience">planned neural-recording lesson on sampling, aliasing and quantization</a> owns the instrument-side distinction between sample timing, value resolution, clipping and noise assumptions. A quantizer's number of levels alone is not its measured fidelity. Here the key comparison is between the source law, actual reconstruction error and declared communication cost.</Prose>
    <Prose>A practical evaluation proceeds from a defined source and decoder task to candidate distortions, complete encoded size, held-out reconstruction errors and computational constraints. Plot actual rate/error pairs with the data split, codec settings and byte-count convention stated. Check representative and rare inputs separately. Choose an operating point that meets the requirements; changing source statistics can alter both entropy-coded rate and error. A theoretical frontier is a model-dependent comparison, not a substitute for that evaluation.</Prose>

    <H2>9. Practise the compression decision</H2>
    <Prose>Attempt these with the explanations closed. Use the hint only after forming a prediction. A satisfactory answer connects the probability law, transmitted representation, distortion criterion and applicable limit; a numerical result without its units or assumptions is incomplete.</Prose>
    <Practice title="1. A different four-word code" prompt="For fair three-bit inputs, use the codebook 000,001,110,111. Choose a nearest reconstruction, breaking ties by codebook order. Find the fixed rate, average distortion and worst distortion." hint="Four inputs are codewords. For each of the other four, determine whether one changed bit suffices.">
      <Prose>Each index uses 2 bits, so the fixed rate is 2/3 bit per source bit. The four codewords have no error; each other word has a nearest codeword one bit away. Expected error is 4/(8×3)=1/6, and worst block error is 1/3. This is a finite operating point, not proof that its rate equals R(1/6).</Prose>
    </Practice>
    <Practice title="2. Repair a curve that rises again" prompt="Someone evaluates 1-Hb(D) for a fair source all the way to D=.9 and reports a positive required rate there. Identify the violated principle and supply the correct value." hint="Which reconstruction was already feasible at D=.5?">
      <Prose>The feasible set only expands when D increases. A constant reconstruction with actual error .5 remains feasible at D=.9, so the minimum is 0. The entropy formula's curved branch ends at .5; applying it beyond that point violates monotonicity and ignores the constant-output feasible choice.</Prose>
    </Practice>
    <Practice title="3. Change the biased source" prompt="Use p=.3 and budget D=.1. Construct the attaining joint table, find both forward conditional error probabilities and calculate the information rate." hint="First obtain output probability r=(p-D)/(1-2D), then generate X backward by an independent error bit.">
      <Prose>r=.25. The joint table, with input rows and output columns, is [[.675,.025],[.075,.225]]. Its rows sum to .7/.3 and its off-diagonal mass is .1. False-positive probability is .025/.7≈.035714; false-negative probability is .075/.3=.25. The rate is Hᵦ(.3)−Hᵦ(.1)≈.412295 bits. A forward channel that flips each source bit with probability .1 would be a different joint law.</Prose>
    </Practice>
    <Practice title="4. Diagnose the exact-zero-error claim" prompt="For p=.1 and a fixed block length n=10, a designer says source entropy below 1 bit proves that a fixed-width exact zero-error code can use fewer than 10 bits per block. What went wrong?" hint="How many length-ten strings have positive probability?">
      <Prose>All 1024 strings have positive probability. Exact zero error requires 1024 distinct reconstructions, so a fixed-width index needs 10 bits. Source entropy constrains expected-length lossless coding and the asymptotic vanishing-error boundary, not this finite exact fixed-width requirement. Rare strings can be long in a variable-length code; a fixed-width code cannot hide their length in an average.</Prose>
    </Practice>
    <Practice title="5. Convert the distortion units" prompt="A solver minimizes I+2D, with D in squared metres. You change the reported error to Dnew=100D but want exactly the same objective and optimizer. What weight should the new solver use?" hint="Set the new weighted distortion term equal to2D.">
      <Prose>Use λnew=.02, because .02×100D=2D. The multiplier has inverse distortion units. Keeping it at 2 would make the same physical error 100 times as expensive and generally select a different operating point.</Prose>
    </Practice>
    <Practice title="6. A stationary but bad optimizer" prompt="The fair-binary solver at λ=2 starts with r=(1,0). After 100 iterations its conditional table is unchanged. Is it optimal? Give a feasible rule that improves its objective." hint="The zero column cannot reappear. Compare the actual zero-rate error with the binary operating point D=.2.">
      <Prose>The constant output has I=0, D=.5 and objective 1. The valid binary channel with D=.2 has I=1−Hᵦ(.2)≈.278072 and objective≈.678072. Its improvement is about .321928. Positive initialization admits that channel; an unchanged support-limited table is insufficient evidence. The reported gap need not equal the actual improvement, because a lower bound can be loose.</Prose>
    </Practice>
    <Practice title="7. Allocate a new Gaussian budget" prompt="Two independent Gaussian components have variances4 and1. Allocate total squared error1.5 and calculate the total rate. Then describe what happens when the budget reaches3." hint="Try a common error level and check whether either component would exceed its variance.">
      <Prose>At total 1.5, θ=.75 lies below both variances, so the errors are (.75,.75). The component rates are about 1.207519 and .207519 bits, adding to 1.415037 bits per vector before rounding. At budget 3, the smaller component caps at 1 and the other gets 2; rate becomes .5 bit per vector. Dividing the total by 2 gives average rate per component, not either component's individual rate.</Prose>
    </Practice>
    <Practice title="8. A plausible output that is not faithful" prompt="With an unknown fair sign X∈{−1,+1}, compare reconstructing0 with sampling an independent fair sign. Can the second scheme prove that the original sign was preserved because its output histogram matches?" hint="Enumerate the four equally likely input/output pairs of the random scheme.">
      <Prose>No. Constant 0 has MSE 1. Independent signs agree on two pairs and differ by 2 on two pairs, giving MSE 2. The output marginal matches, but independence means I=0. Preserving the actual sign would require an input-dependent message or other informative side information. Marginal similarity and paired fidelity measure different properties.</Prose>
    </Practice>
    <Practice title="9. Build and validate a changed byte format" prompt="Adapt the complete serializer to the four-word codebook from task1, using two index bits per block and the same four-byte source-length header. Encode 001011101111. Define tests for partial blocks and malformed data." hint="Select the nearest codeword, emit its index as exactly two bits, concatenate indices, and pad only the final byte. The decoder derives the index count from the original length.">
      <Prose>The indices are 1,1,1,3, so the payload bits are 01010111, hex 57. The complete five-byte message is hex 0000000c 57. Decoding reconstructs 001001001111, with two wrong positions out of 12. The padding and length protocol must state that a partial source block is zero-padded before encoding and truncated after reconstruction.</Prose>
      <Prose>Acceptance tests: exhaust all binary strings of lengths 0 through 10; preserve the original decoded length; emit only the declared index width; check the expected payload byte count; reject a truncated header, missing/extra payload bytes and nonzero unused bits. For full fair three-bit blocks, verify average error 1/6 and at most one changed bit per block. Report both payload and full-message bits per original bit; an empty input has a valid header but no meaningful per-symbol rate. These tests verify the declared toy protocol, not resilience to arbitrary bit corruption.</Prose>
    </Practice>
    <Practice title="10. Choose a useful audio operating point" prompt="You must compress 30-second call recordings. Propose a cheap distortion check, a more task-relevant check and an evaluation process that does not let a good average conceal lost critical words." hint="Distinguish paired waveform fidelity, intelligibility of words and the actual encoded bytes. State what would make a candidate fail.">
      <Prose>One defensible starting answer uses aligned waveform MSE as a cheap paired signal check, while acknowledging its sensitivity to small timing changes and its incomplete relationship to intelligibility. A task-relevant evaluation uses blinded listening/transcription of held-out calls, checking important words or numbers separately. Include realistic noise and speaker conditions with category-level results. A specific intelligibility metric could supplement these checks if its validity for the conditions is established.</Prose>
      <Prose>Encode and decode every candidate setting through the real format; count complete bytes, measure latency on stated hardware and evaluate the same held-out recordings. Specify an acceptable error criterion before comparing settings, including a separate critical-word requirement. A candidate fails when it violates that requirement even if aggregate MSE improves. Document the chosen bitrate range and what source changes would trigger reevaluation. The theory organizes the tradeoff; it cannot decide which lost words are acceptable.</Prose>
    </Practice>
    <Prose>You are ready to move on when you can build a decodable code, distinguish its finite rate from an information frontier, derive a simple limit, diagnose an optimizer's missing support and explain why the distortion criterion changes the answer. The next module topic, <a href="/learn/path/full-curriculum/f-divergences-integral-probability-metrics?module=math-foundations">f-Divergences &amp; Integral Probability Metrics</a>, studies other ways to compare probability laws. The fidelity example explains why the choice of comparison matters.</Prose>
    <Sources alternatives={<p>For a lecture-based second pass, use <a href="https://www.youtube.com/watch?v=qmCTnAoMSc8" target="_blank" rel="noreferrer">S.N.Merchant's NPTEL lecture 31: Introduction to Rate-Distortion Theory</a>, from <a href="https://nptel.ac.in/courses/117101053" target="_blank" rel="noreferrer">IIT Bombay's Information Theory and Coding course</a>. It is an older foundational lecture best approached after entropy and mutual information; the creator/title were verified, but this revision did not review full playback or a transcript.</p>}>
      <li><a href="https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/resources/mit6_441s16_chapter_23/" target="_blank" rel="noreferrer">MIT 6.441, chapter 23: Rate-Distortion Theory</a> — graduate notes connecting quantization, finite codes and the information bound; useful for the assumptions behind familiar engineering approximations.</li>
      <li><a href="https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/aaa8d18ddecde45f97134d3f3dcee4a3_MIT6_441S16_chapter_24.pdf" target="_blank" rel="noreferrer">MIT 6.441, chapter 24: Achievability Bounds</a> — theorem 24.2 and its random-codebook argument provide the formal next step after our finite example and converse.</li>
      <li><a href="https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec16.pdf" target="_blank" rel="noreferrer">CMU 10-704, lecture 16, sections 16.3–16.4</a> — a compact Gaussian derivation and finite rate-distortion BA calculation. These are course scribe notes, intended as a supplementary mathematical route.</li>
      <li><a href="https://www.comm.utoronto.ca/~weiyu/2024_ISITW_Sadaf.pdf" target="_blank" rel="noreferrer">Qian and colleagues: Gaussian Vector Rate-Distortion-Perception</a> — sectionIII reviews classical reverse water-filling; later sections are an advanced extension with additional perception assumptions.</li>
      <li><a href="https://proceedings.mlr.press/v97/blau19a.html" target="_blank" rel="noreferrer">Blau &amp; Michaeli,2019: Rethinking Lossy Compression</a> — why paired distortion and output-distribution realism can impose different constraints; read its formulation and operational qualifications.</li>
      <li><a href="https://www.cns.nyu.edu/pub/lcv/balle17a-final.pdf" target="_blank" rel="noreferrer">Ballé, Laparra &amp; Simoncelli,2017: End-to-End Optimized Image Compression</a> — sections 2–3 and the entropy-code appendix connect transform/quantization training to actual encoded rates. This is a mechanism reference, not a current performance recommendation.</li>
    </Sources>
  </div>
};
