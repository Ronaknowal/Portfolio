import { H2, H3, Prose, Callout } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { OutcomePushforwardLab, MeanSquaredLossLab, JointCovarianceLab, SharedNoiseLab, ConditionalMomentsLab, SquaredUniformLab, SampleMeanLawLab } from '../../components/lesson-labs/RandomVariableLabs.jsx';
import { randomVariableExamples } from '../random-variables-examples.js';
function Example({
  name
}) {
  const example = randomVariableExamples[name];
  return <section><Prose><strong>Before running.</strong> {example.question}</Prose><RunnableExample example={example} /><Prose>{example.interpretation}</Prose></section>;
}
function Practice({
  title,
  prompt,
  hint,
  children
}) {
  return <section className="lesson-check"><H3>{title}</H3><Prose>{prompt}</Prose><details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Reasoned solution</summary>{children}</details></section>;
}
export default {
  title: 'Random Variables, Expectation & Covariance',
  readTime: '~80 min read + 3 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="random-variable-lesson">
    <LessonIntro prerequisites="Start with sets, functions, fractions and elementary algebra. We introduce finite probability from the beginning. The continuous branch uses the preceding Single-Variable Calculus lesson; the finite core does not need integration." sections={[['1-give-an-outcome-a-number', 'Outcomes and random variables'], ['2-follow-probability-into-a-distribution', 'PMFs and CDFs'], ['3-average-the-right-quantity', 'Expectation and transformations'], ['4-measure-spread-and-choose-a-prediction', 'Variance and squared loss'], ['5-keep-the-pairings', 'Joint laws and dependence'], ['6-interpret-centered-products', 'Covariance and correlation'], ['7-combine-quantities-with-shared-noise', 'Linear combinations'], ['8-predict-with-available-information', 'Conditional moments'], ['9-transform-continuous-values', 'Continuous distributions'], ['10-check-whether-a-moment-exists', 'Moment boundaries'], ['11-distinguish-a-population-from-a-sample', 'Sampling and estimation'], ['12-practise-on-changed-models', 'Independent practice']]}>Two measurements each fluctuate by the same amount. Will averaging them halve the variance? That depends on whether their errors move together. This lesson develops the tools needed to answer that question from a probability model, while keeping the individual outcomes, their distribution and the quantity you want to predict distinct.</LessonIntro>

    <H2>1. Give an outcome a number</H2>
    <Prose>Imagine tossing two labelled coins: first and second. Before tossing, the possible outcomes are TT, TH, HT and HH. The <strong>sample space</strong> Ω is this collection of outcomes. A particular outcome is written ω. An <strong>event</strong> is a set of outcomes answering a yes-or-no question: “at least one head” is the event {'{TH,HT,HH}'}.</Prose>
    <Prose>A probability model gives each possible outcome a nonnegative weight, with all weights adding to one. For two independent fair coins, each outcome has probability 1/4. “Fair” describes each coin's individual chance; <strong>independent</strong> says one coin's result does not change the other's chance. Together those assumptions justify multiplying 1/2 by 1/2. Two fair coins need not be independent if, for example, one result is copied from the other.</Prose>
    <Prose>Now ask for the number of heads. The rule X assigns TT→0, TH→1, HT→1, HH→2. A <strong>random variable</strong> is a numerical function on the outcome space. The function itself is fixed; its input is uncertain. If the result is TH, the <strong>realized value</strong> is x=X(TH)=1. Random variables often use capitals and realized numbers lowercase.</Prose>
    <LessonTable caption="Keep the objects separate" headers={['Object', 'In the coin example', 'What it describes']} rows={[['Outcome ω', 'TH', 'The full result of the declared experiment'], ['Random variable X', 'Count the heads', 'A fixed rule applied to every possible outcome'], ['Realization x', '1 after observing TH', 'One observed numerical value'], ['Distribution of X', 'Masses 1/4,1/2,1/4 at 0,1,2', 'How probability is spread over numerical values']]} />
    <Prose>Several random variables can describe the same outcome. X can count heads while Y indicates whether the first coin was heads. On TH, X=1 and Y=0. Replacing X with a different rule does not require another toss. Their shared outcome is also what makes it meaningful to ask how X and Y behave together.</Prose>
    <Prose>For finite spaces any such numerical rule is allowed. In general probability theory, a random variable must be <strong>measurable</strong>: inverse images of the relevant numerical events must be events to which the model assigns probabilities. That is the precise extension developed in <a href="/learn/path/full-curriculum/measure-theory-probability-spaces?module=math-foundations">Measure Theory &amp; Probability Spaces</a>. Here the finite event sets make this automatic.</Prose>

    <H2>2. Follow probability into a distribution</H2>
    <Prose>To find P(X=1), collect <em>every</em> outcome that X sends to 1. TH and HT each contribute 1/4, so the total is 1/2. It would be wrong to give 0,1,2 equal probabilities merely because there are three different values. Probability belongs to the original model and is carried through the rule.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      p_X(x)&=P(X=x)\\
      &=\sum_{\omega:\,X(\omega)=x}P(\{\omega\}).
    \end{aligned}`}</MathBlock>
    <Prose>The <strong>probability mass function</strong>, or PMF, records these point probabilities for a discrete variable. Its positive-mass values form the support in this finite setting. Summing the PMF gives one because every original outcome contributes to exactly one value. The same operation is sometimes called pushing a probability law forward through a function.</Prose>
    <OutcomePushforwardLab />
    <H3>Accumulating mass answers threshold questions</H3>
    <Prose>The <strong>cumulative distribution function</strong> is F(t)=P(X≤t). For the fair head count it is 0 below 0, 1/4 from 0 up to but excluding 1, 3/4 from 1 up to but excluding 2, and 1 from 2 onward. The equality in “≤” includes mass at the threshold, so the CDF jumps when the threshold reaches a supported value.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      P(a<X\le b)&=F(b)-F(a),\\
      P(a\le X\le b)&=F(b)-F(a^-).
    \end{aligned}`}</MathBlock>
    <Prose>F(a⁻) means the cumulative mass strictly below a. Thus P(X=a)=F(a)−F(a⁻). The CDF never decreases, approaches 0 and 1 at the two extremes and is right-continuous. Unlike a PMF or density formula of a specific kind, a CDF describes discrete, continuous and mixed real-valued laws.</Prose>
    <Prose>For the optional programs, save any complete Python block as <code>random_variables.py</code> and run <code>python random_variables.py</code> with Python 3.12 or a compatible Python 3. They use only the standard library and need no data files or earlier program. Each block includes its imports, inputs, expected output and interpretation; nothing executes automatically in this page.</Prose>
    <Example name="mapping" />
    <Checkpoint prompt="Two fair coins always agree because the second result copies the first. Is the head-count PMF still 1/4,1/2,1/4?"><Prose>No. TT and HH each have probability 1/2, while TH and HT have probability 0. The head count is 0 or 2 with equal probability; its mean will still be 1. Fair marginals do not specify the joint outcome law.</Prose></Checkpoint>

    <H2>3. Average the right quantity</H2>
    <Prose>The <strong>expectation</strong> E[X] is the probability-weighted average of a random variable. For a finite model, multiply each value by the probability attached to it and add. The result has the same units as X. It need not be a possible observation: a fair die has mean 3.5, although it never lands on 3.5.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      E[X]&=\sum_\omega X(\omega)P(\{\omega\})\\
      &=\sum_x x\,p_X(x).
    \end{aligned}`}</MathBlock>
    <Prose>The second line regroups the first line by equal X-values. This explains why we can compute from either the outcome table or the induced distribution. The mean describes a model, not a guarantee about the next value or the exact average of a particular finite sample. We will state the conditions for repeated averages later.</Prose>
    <H3>Transform first, then average</H3>
    <Prose>If the relevant cost is g(X), apply g to each possible value before weighting it. You do not have to derive the whole distribution of g(X) first. Regrouping the original outcome sum proves the useful rule sometimes called <strong>LOTUS</strong>, the law of the unconscious statistician:</Prose>
    <MathBlock>{String.raw`E[g(X)]=\sum_x g(x)p_X(x).`}</MathBlock>
    <Prose>For X equally likely −1 or 1, E[X]=0 but E[X²]=1. Squaring the mean gives 0, a different quantity. A nonlinear transformation generally does not commute with expectation. The distinction matters when a loss, energy or financial payoff is nonlinear in the uncertain input.</Prose>
    <H3>Linearity does not require independence</H3>
    <Prose>For constants a,b and two variables on the same outcome space, expand the finite sum for aX+bY. Distribute the probability weight over the two terms and collect them: E[aX+bY]=aE[X]+bE[Y]. This uses no factorization of joint probabilities, so it does not require independence. Adding a constant shifts the mean by that constant.</Prose>
    <Prose>An <strong>indicator</strong> I_A is 1 on event A and 0 elsewhere. Its expectation is P(A). To count events, add their indicators: the expected count is the sum of their individual probabilities, whether or not those events are independent. For three repeated copies of the same bit with success probability p, the expected number of ones is 3p. The distribution of that count is not generally binomial.</Prose>
    <Example name="indicators" />
    <Prose>For infinite spaces these operations need the relevant integrability conditions. Finite E|X| and E|Y| suffice for ordinary signed linearity. Nonnegative expectations can be infinite, but expressions such as infinity minus infinity are undefined. We return to this limitation after the continuous examples.</Prose>

    <H2>4. Measure spread and choose a prediction</H2>
    <Prose>The mean alone does not describe fluctuation. Center X by subtracting μ=E[X]. The signed deviations average to zero, so simply averaging them cannot measure spread. <strong>Variance</strong> averages their squares. <strong>Standard deviation</strong> takes its square root to return to the original units.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      \operatorname{Var}(X)&=E[(X-\mu)^2]\\
      &=E[X^2]-2\mu E[X]+\mu^2\\
      &=E[X^2]-\mu^2.
    \end{aligned}`}</MathBlock>
    <Prose>In the ordinary finite-variance setting, every term is defined and variance is nonnegative. It is zero exactly when X equals its mean with probability one. For a finite model this allows exceptional values only on zero-probability outcomes.</Prose>
    <Prose>If X is a time in seconds, Var(X) is in seconds². For Y=aX+b, its centered value is a(X−μ), hence Var(Y)=a²Var(X) and SD(Y)=|a|SD(X). A translation does not change spread. A negative scaling reverses order but still gives a nonnegative variance.</Prose>
    <H3>Why the mean is the best constant under squared error</H3>
    <Prose>Suppose you must announce one prediction c before seeing X, and your penalty is (X−c)². Split X−c into (X−μ)+(μ−c). On expansion, the cross term has expectation 2(μ−c)E[X−μ]=0. Therefore:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &E[(X-c)^2]\\
      &=\operatorname{Var}(X)+(\mu-c)^2.
    \end{aligned}`}</MathBlock>
    <Prose>The first term is fixed by the population. The second is nonnegative and vanishes at c=μ. Thus the mean is the unique best constant prediction under this loss when the second moment is finite. This conclusion depends on the loss: absolute error has a median as a minimizer, a different optimization question.</Prose>
    <MeanSquaredLossLab />
    <Example name="moments" />
    <Prose>Spread is not a full risk description. Equal mass at −1 and 1 has mean 0 and variance 1. Masses 1/8,3/4,1/8 at −2,0,2 have the same moments, but P(|X|≥1.5) changes from 0 to 1/4. A standard-deviation interval is not automatically a stated-coverage probability interval. A distributional assumption or a valid inequality is needed.</Prose>

    <H2>5. Keep the pairings</H2>
    <Prose>To describe two variables together, retain the probability of each <em>pair</em>. The <strong>joint PMF</strong> p(x,y)=P(X=x,Y=y) is a table: rows can index X-values and columns Y-values. Summing a row removes the question about Y and gives the <strong>marginal</strong> P(X=x); column sums give P(Y=y).</Prose>
    <Prose>Suppose X is equally likely −1,0,1. In one model Y=X; in a second Y=−X; in a third Y is an independent uniform draw. Every model gives the same individual distribution to Y. The pairings are different, so the outcome of X+Y can be very different: under Y=−X it is always 0.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      p_X(x)&=\sum_y p(x,y),\\
      p_{Y|X}(y|x)&=\frac{p(x,y)}{p_X(x)}
      \quad\text{for}\\
      &p_X(x)>0.
    \end{aligned}`}</MathBlock>
    <Prose>The second line describes the distribution left after the information X=x is known. Divide the retained joint row by its total so it sums to one. For a zero-probability row the division does not identify a conditional law. A software choice on such a row must not be presented as something learned from the model.</Prose>
    <Prose><strong>Independence</strong> means that the joint law factors: p(x,y)=p_X(x)p_Y(y) for every pair in the finite table. Equivalently, each positive-mass conditional row gives the unchanged marginal law of Y. One mismatched cell refutes independence; one matching cell does not prove it. For general variables independence is defined through all suitable event pairs, not just point masses that may all be zero.</Prose>
    <Prose>Expectations of pair-dependent quantities use the joint law: E[h(X,Y)]=Σ_xΣ_y h(x,y)p(x,y). In particular E[XY] needs pairings. If X,Y are independent and integrable, the factorized sum or integral gives E[XY]=E[X]E[Y]. The converse for this one product is not generally true.</Prose>

    <H2>6. Interpret centered products</H2>
    <Prose>When both variables tend to be above their means together, their centered product is positive. When one is above while the other is below, it is negative. <strong>Covariance</strong> averages these products. This captures a particular linear association rather than every form of dependence.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &\operatorname{Cov}(X,Y)\\
      &=E[(X-\mu_X)(Y-\mu_Y)]\\
      &=E[XY]-\mu_X\mu_Y.
    \end{aligned}`}</MathBlock>
    <Prose>Assume finite second moments throughout this covariance section. They also ensure that the product is integrable. Covariance is symmetric; Cov(X,X)=Var(X); and linearity gives Cov(aX+b,cY+d)=acCov(X,Y). Its units are the product of X's and Y's units. A positive change of measurement scale changes its magnitude, while a negative one reverses its sign.</Prose>
    <JointCovarianceLab />
    <H3>Normalize only when both spreads are positive</H3>
    <Prose><strong>Correlation</strong> divides covariance by both standard deviations. The result is dimensionless. If either variable is constant almost surely, its standard deviation is zero and this correlation is undefined, rather than zero.</Prose>
    <MathBlock>{String.raw`\rho=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y},
      \qquad \sigma_X,\sigma_Y>0.`}</MathBlock>
    <Prose>To see why |ρ|≤1, write A=X−μ_X and B=Y−μ_Y. For any t, E[(B−tA)²] is nonnegative. With Var(X)&gt;0, choose t=Cov(X,Y)/Var(X). Expanding gives Var(Y)−Cov(X,Y)²/Var(X)≥0, which is the claimed bound after dividing by Var(Y)&gt;0. Equality requires B=tA almost surely: under these positive-variance conditions, perfect correlation is an affine relation with positive or negative slope.</Prose>
    <H3>A nonlinear dependency can cancel every centered product</H3>
    <Prose>Take X uniform on −1,0,1 and Y=X². Then E[X]=0, E[Y]=2/3 and E[XY]=E[X³]=0 by cancellation, so covariance is 0. Yet P(X=0,Y=1)=0 differs from P(X=0)P(Y=1)=2/9. In fact Y is completely determined by X. The issue is the limited summary, not a lack of dependence.</Prose>
    <Example name="joint" />
    <Callout title="Correlation is not a causal effect">A joint distribution describes association under its data-generating process. It does not alone say what happens if you intervene on one variable. A shared cause, selection or a change in groups can alter the association. Causal conclusions need additional assumptions and design; the <a href="/learn/path/full-curriculum/causal-inference-do-calculus?module=math-foundations">Causal Inference</a> lesson studies those requirements.</Callout>
    <Prose>One useful special case has extra structure: jointly Gaussian variables with zero cross-covariance are independent. Having individually normal marginal distributions is not the same assumption as having a jointly Gaussian vector. Do not apply that exception to the arbitrary finite tables above.</Prose>

    <H2>7. Combine quantities with shared noise</H2>
    <Prose>Center aX+bY, square it and keep the cross term. This gives the rule needed for sums, differences and weighted averages:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &\operatorname{Var}(aX+bY)\\
      &=a^2\operatorname{Var}(X)\\
      &\quad+b^2\operatorname{Var}(Y)\\
      &\quad+2ab\operatorname{Cov}(X,Y).
    \end{aligned}`}</MathBlock>
    <Prose>Only uncorrelated cross terms disappear. Independence is sufficient, but not necessary, for that simplification. A negative covariance reduces the variance of a sum and increases the variance of a difference; changing one coefficient's sign changes which cancellations occur.</Prose>
    <H3>A complete common-error example</H3>
    <Prose>In a synthetic measurement model, A=10+S+e_A and B=20+S+e_B millivolts. The shared disturbance S is equally likely −2 or 2. Independent local disturbances e_A,e_B are each equally likely −1 or 1, and all three disturbance signs are independent. These assumptions give eight equally likely outcomes.</Prose>
    <Prose>The means are 10 and 20. Each variance is 4+1=5 mV². Their covariance is 4 mV² because only the same shared S term remains after all independent centered cross products vanish. Correlation is 4/5. Thus the average has variance (5+5+2·4)/4=9/2 mV², not 5/2. The difference B−A has variance 5+5−2·4=2 mV².</Prose>
    <SharedNoiseLab />
    <Example name="noise" />
    <Prose>These quantities have different baselines: the average is centered at 15 mV; the difference at 10 mV. A lower variance does not automatically make one a better estimator of the other. If both channels measure a common unknown target, first state how calibration removes their known offsets and what quantity the combination must preserve. A sum-to-one weight constraint can preserve a common target; an unqualified difference would remove it along with any shared component.</Prose>
    <H3>Package every pairwise covariance in one matrix</H3>
    <Prose>A <strong>random vector</strong> collects variables into a column, such as Z=(X,Y)ᵀ. Its mean is the column of means. Its covariance matrix Σ has variances on the diagonal and pairwise covariances off it. The superscript T denotes transposition: turn a column into a row.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      \Sigma=E[(Z-\mu)(Z-\mu)^T],\\
      \Sigma_{ij}=\operatorname{Cov}(Z_i,Z_j),\\
      \operatorname{Var}(w^TZ)=w^T\Sigma w.
    \end{gathered}`}</MathBlock>
    <Prose>For two components, the last line expands to the variance rule above. Since every variance is nonnegative, Σ is <strong>positive semidefinite</strong>: wᵀΣw≥0 for every real coefficient vector w. It need not be invertible; if a nonzero combination wᵀZ is constant almost surely, that direction has zero variance.</Prose>
    <Prose>A linear map V=AZ+b gives Cov(V)=AΣAᵀ. If Z has d entries and V has k, A is k×d and the new covariance is k×k. To derive it, center V to get A(Z−μ), form its outer product and pull the fixed matrices outside the expectation. This concerns linear transformations; replacing A with a Jacobian for a nonlinear map is a local approximation requiring additional error analysis.</Prose>
    <Example name="matrix" />
    <details><summary>Deeper application: the best affine prediction</summary>
      <Prose>To predict Y with c+aX, first choose c=μ_Y−aμ_X; this makes the prediction error centered. Its expected square becomes Var(Y)−2aCov(X,Y)+a²Var(X). For Var(X)&gt;0, completing the square gives the optimum slope a=Cov(X,Y)/Var(X). If Var(X)=0, X supplies no varying linear feature and the best constant is μ_Y; slope is not uniquely identified.</Prose>
      <Prose>This is a population squared-loss calculation under finite second moments. A fitted regression estimates quantities from data and has additional uncertainty. The best affine prediction is not always the full conditional mean, as the next program shows.</Prose><Example name="prediction" />
    </details>

    <H2>8. Predict with available information</H2>
    <Prose>Suppose you learn which group G an outcome belongs to before predicting Y. Within a positive-probability group g, normalize its masses and average Y. This gives E[Y|G=g], a number for that fixed group. Before the group is observed, E[Y|G] is itself a random variable: it selects the mean associated with the group that occurs.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      m(g)&=E[Y\mid G=g],\\
      m(G)(\omega)&=m(G(\omega)),\\
      E[m(G)]&=E[Y].
    \end{aligned}`}</MathBlock>
    <Prose>The last identity is <strong>total expectation</strong>. For a finite partition, multiplying each conditional average by its group mass cancels the normalization denominator; summing groups counts every outcome once. A null group's chosen value contributes no mass. For nested information, averaging a finer conditional prediction inside each coarser group gives the coarser conditional mean: the <strong>tower property</strong>. The nesting premise matters.</Prose>
    <H3>Residual error versus changing group means</H3>
    <Prose>Write Y−E[Y]=(Y−m(G))+(m(G)−E[Y]). The first term has conditional mean zero within every positive-mass group. Multiplying it by the second, which is fixed inside each group, still gives conditional mean zero. Expanding the square therefore yields:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      \operatorname{Var}(Y)
      &=E[\operatorname{Var}(Y\mid G)]\\
      &\quad+\operatorname{Var}(E[Y\mid G]).
    \end{aligned}`}</MathBlock>
    <Prose>This is the <strong>law of total variance</strong> under finite second moments. The first term is average remaining spread within groups; the second is spread between group means. Do not average conditional variances without the correct group probabilities.</Prose>
    <Prose>The same decomposition for both X and Y gives <strong>total covariance</strong>. Both residual-versus-group-mean cross terms vanish by the same conditional averaging argument, so:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &\operatorname{Cov}(X,Y)\\
      &=E[\operatorname{Cov}(X,Y\mid G)]\\
      &\quad+\operatorname{Cov}(E[X\mid G],E[Y\mid G]).
    \end{aligned}`}</MathBlock>
    <Prose>For a concrete example, G is equally likely −2 or 2 and independent U is equally likely −1 or 1. Let X=G+U and Y=G−U. Within a group, increasing U raises X and lowers Y, so conditional covariance is −1. But both conditional means equal G, whose variance is 4. The pooled covariance is −1+4=3. The relation within groups and the pooled relation answer different questions.</Prose>
    <ConditionalMomentsLab />
    <Example name="conditioning" />
    <H3>Why additional information can improve average prediction</H3>
    <Prose>For any alternative prediction a(G) using only the same group information, expand Y−a(G)=(Y−m(G))+(m(G)−a(G)). The cross term vanishes, giving E[(Y−a(G))²]=E[(Y−m(G))²]+E[(m(G)−a(G))²]. Thus the conditional mean minimizes expected squared loss among those predictions. A richer information set permits every old prediction, so its minimum <em>average</em> risk cannot increase.</Prose>
    <Prose>This does not say every observation improves or every subgroup has smaller conditional variance than the pooled variance. A rare high-variability group can have a larger variance. Nor does it say a finite fitted predictor automatically attains the population optimum. Hidden outcome information would change the permitted prediction problem rather than prove better learning.</Prose>
    <Prose>The finite partition proof shows the mechanism. <a href="/learn/path/full-curriculum/measure-theory-probability-spaces?module=math-foundations#8-use-information-to-form-a-conditional-expectation">The measure-theory conditional-expectation branch</a> extends this to general information sets, almost-sure versions and continuous observations where P(X=x)=0. The <a href="/learn/path/full-curriculum/linear-logistic-regression?module=classical-ml">regression lesson</a> develops fitted models; it does not remove the need to distinguish population risk from finite-data performance.</Prose>

    <H2>9. Transform continuous values</H2>
    <Prose>A continuous variable can spread probability over intervals instead of isolated points. A <strong>probability density</strong> f_X, when it exists, integrates to 1 and gives interval probability by area. Its height is not a point probability and can exceed 1. A uniform delay on [0,.2] seconds has density 5 per second; a .1-second interval inside the support has probability .5.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      P(a<X\le b)=\int_a^b f_X(x)\,dx,\\
      E[g(X)]=\int g(x)f_X(x)\,dx.
    \end{gathered}`}</MathBlock>
    <Prose>The expectation integral has the same weighted-value meaning as the finite sum, subject to its existence conditions. For a density-only law a singleton has probability 0. A CDF jump instead signals point mass. A mixed law can combine a point mass and a density component, so no ordinary density alone accounts for all its probability. The earlier <a href="/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations#6-read-area-and-cumulative-probability">Probability Distributions lesson</a> works a mixed-delay example in detail.</Prose>
    <H3>A non-injective rule needs every preimage branch</H3>
    <Prose>Let U be uniform on [−1,1] and Y=U². For 0≤y≤1, the event U²≤y is the interval −√y≤U≤√y. Its length is 2√y; multiplying by uniform density 1/2 gives F_Y(y)=√y. The CDF is 0 for y&lt;0 and 1 for y≥1.</Prose>
    <MathBlock>{String.raw`f_Y(y)=\frac{1}{2\sqrt y},\qquad 0<y<1.`}</MathBlock>
    <Prose>Differentiating the CDF gives this density on the open interval. Its unbounded height near 0 is integrable: the mass from 0 toε is√ε, which tends to 0. Therefore there is no point mass at 0. Forgetting the negative U branch would account for only half the mass.</Prose>
    <SquaredUniformLab />
    <Prose>We can compute Y's moments directly from U without a new density calculation: integrate u²/2 from −1 to 1 to get E[Y]=1/3. Similarly E[Y²]=E[U⁴]=1/5, so Var(Y)=1/5−1/9=4/45. These agree with integrating against f_Y; substitution y=u² on the positive branch verifies the same calculation.</Prose>
    <Example name="continuous" />
    <Prose>For a differentiable strictly monotone transformation, an inverse derivative corrects the density for changes of width. For a many-to-one transformation, contributions from all appropriate inverse branches must be added, under their regularity conditions. The CDF/preimage method used here often avoids mistakes before a general change-of-variables formula is introduced.</Prose>

    <H2>10. Check whether a moment exists</H2>
    <Prose>A finite weighted table always gives finite moments. An unbounded distribution need not. For a signed variable, a finite ordinary expectation requires E|X|&lt;∞. More generally its positive and negative parts may define an extended expectation if they are not both infinite. A symmetric-looking expression does not justify subtracting two divergent contributions.</Prose>
    <Prose>For example, the standard Cauchy density is 1/[π(1+x²)]. Its symmetric integral of x f(x) over [−R,R] is 0 for every finite R, but each of the positive and negative absolute contributions grows without bound. That principal-value cancellation is not an ordinary mean. A simulation or finite integration window cannot repair this distinction.</Prose>
    <H3>A finite mean can coexist with an infinite second moment</H3>
    <Prose>Consider the Pareto density f(x)=αx<sup>−α−1</sup> for x≥1, with α&gt;0. It integrates to 1. To find the k-th nonnegative moment, integrate αx<sup>k−α−1</sup> from 1 to infinity. The power integral converges exactly when k&lt;α, and then equals α/(α−k). Thus the mean is finite only for α&gt;1; the second moment only for α&gt;2.</Prose>
    <Example name="tails" />
    <Prose>For α&gt;2, subtracting the squared mean gives variance α/[(α−1)²(α−2)]. At α=3/2 the mean is 3, but the second moment diverges; finite-variance covariance formulas and the Chebyshev bound below cannot be used as if a finite variance had been supplied. A finite sample variance can still be computed, because a finite list and the underlying law are different objects.</Prose>
    <Prose>These examples are the boundary check, not a replacement for measure and convergence theory. The lesson on <a href="/learn/path/full-curriculum/real-analysis-sequences-modes-of-convergence?module=math-foundations">Real Analysis, Sequences &amp; Modes of Convergence</a> explains why passing limits through averages requires hypotheses.</Prose>

    <H2>11. Distinguish a population from a sample</H2>
    <Prose>The distribution's mean μ is a population quantity. Before collecting n readings, their sample mean X̄=(X₁+⋯+Xₙ)/n is a random variable too. After the readings x₁,…,xₙ are known, x̄ is one realized estimate. A probability statement about X̄ concerns repeated possible datasets under a specified sampling model, not several interpretations of an already fixed number.</Prose>
    <Prose>If all Xᵢ have mean μ, linearity gives E[X̄]=μ even if they are dependent. Its variance, however, retains every pair covariance:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &\operatorname{Var}(\bar X)\\
      &=\frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\operatorname{Cov}(X_i,X_j).
    \end{aligned}`}</MathBlock>
    <Prose>For independent identically distributed readings with variance σ², only the n diagonal terms remain, giving σ²/n. “Identically distributed” means the marginal laws agree; “independent” is a separate claim about their joint law. If every reading copies one variable Z, all n² covariance terms equal Var(Z), so the average keeps that full variance.</Prose>
    <Prose>A <strong>Bernoulli</strong> variable B is 1 with probability p and 0 otherwise. Its mean is p. Because B²=B, its variance is p−p²=p(1−p). For n independent copies, a particular arrangement with k ones has probability pᵏ(1−p)ⁿ⁻ᵏ. There are “n choose k” such arrangements: choose which k positions contain the ones. Their count has a <strong>binomial</strong> law. The sample mean takes value k/n with that same mass. Python's comb(n,k) supplies this number of arrangements.</Prose>
    <SampleMeanLawLab />
    <Example name="sample_mean" />
    <H3>A precise finite-variance averaging guarantee</H3>
    <Prose>For any variable W with mean μ and finite variance, the event |W−μ|≥ε contributes at least ε² per unit of its probability to E[(W−μ)²]. Therefore P(|W−μ|≥ε)≤Var(W)/ε² for ε&gt;0: <strong>Chebyshev's inequality</strong>. Under the iid finite-variance model, substituting W=X̄ gives σ²/(nε²), capped at 1 as a probability bound.</Prose>
    <Prose>For fixed ε this upper bound tends to 0 with n, proving convergence in probability of the iid finite-variance average to μ. A broader iid law of large numbers works under finite absolute first moment, with a different proof; it does not create a finite variance when one is absent. Neither result promises monotone improvement for one sample path or an exact finite-sample error. Sharper assumptions and bounds belong to <a href="/learn/path/full-curriculum/concentration-inequalities-hoeffding-bernstein-chernoff?module=math-foundations">Concentration Inequalities</a>.</Prose>
    <H3>What the n−1 denominator corrects</H3>
    <Prose>For iid paired observations (Xᵢ,Yᵢ), a usual sample covariance is s_XY=Σᵢ(Xᵢ−X̄)(Yᵢ−Ȳ)/(n−1), n≥2. Both entries from the same pair stay together. Independence is across pairs, not a requirement that X and Y within each pair be independent.</Prose>
    <Prose>Let the true means be μ_X,μ_Y. Expanding the numerator gives Σᵢ(Xᵢ−μ_X)(Yᵢ−μ_Y)−n(X̄−μ_X)(Ȳ−μ_Y). The first expected value is nCov(X,Y). The second is nCov(X̄,Ȳ)=Cov(X,Y), using iid pairs and finite second moments. The expected numerator is therefore (n−1)Cov(X,Y). Dividing by n−1 makes this estimator unbiased under those conditions. Taking Y=X proves the sample-variance version.</Prose>
    <Example name="sample_covariance" />
    <Prose>Dividing by n instead has a different legitimate use: it is the variance of the empirical distribution that places mass 1/n at each observed row. “Unbiased estimator of a population variance” and “variance of this finite empirical distribution” are different targets. Unequal weights, missing pair entries, dependence and selection mechanisms need their own analysis. There is no universal denominator correction that fixes them all.</Prose>
    <H3>Compute centered moments carefully</H3>
    <Prose>E[X²]−E[X]² is a useful algebraic identity, but a floating-point implementation can lose a small variance when subtracting two nearly equal huge values. A centered two-pass calculation or an online update can avoid that particular cancellation. Welford's update maintains a count, mean and sum M₂ of squared deviations from the current mean.</Prose>
    <Prose>When a new value x arrives, let δ=x−oldMean, set newMean=oldMean+δ/n, and add δ(x−newMean) to M₂. To derive the update, recenter the old n−1 values around newMean; their old signed deviations sum to zero. Their new squared deviations add (n−1)(oldMean −newMean)², then the new point adds(x−newMean)². Substitution yields δ(x−newMean). Choose the final denominator according to the intended population or sample statistic.</Prose>
    <Example name="stable" />
    <Prose>This example uses exactly representable finite inputs around 10¹². It is not a claim that every floating-point range or stream is safe: input rounding, overflow and accumulated error still matter. Our interactive models use small declared ranges, centered sums and explicit degenerate states.</Prose>

    <H2>12. Practise on changed models</H2>
    <Prose>The complete programs use the Python standard library and were executed with Python 3.12. Solve the questions before opening the reasoned solutions. Finite code checks support the calculations; general laws rely on their derivations and assumptions.</Prose>
    <Practice title="1. Reconstruct the induced law" prompt="Four outcomes have probabilities 1/8,3/8,1/4,1/4. A rule sends them to −1,−1,2,4. Find the PMF, E[X], F(2), and P(−1<X≤2)." hint="Combine the first two masses before computing the distribution."><Prose>The PMF is −1 with 1/2, 2 with 1/4, 4 with 1/4. Mean is −1/2+1/2+1=1. F(2)=3/4. Excluding −1 leaves P(−1&lt;X≤2)=1/4. Equal numerical values collect mass; they do not create extra independent draws.</Prose></Practice>
    <Practice title="2. Change the loss and the units" prompt="For that law, compute variance and E[(X−2)²]. Then transform to Y=−3X+5." hint="Compute E[X²], then use the centered identities."><Prose>E[X²]=1/2+1+4=11/2; Var(X)=11/2−1=9/2. Predicting 2 has squared loss 9/2+(2−1)²=11/2. The new mean is 2, variance 81/2 and standard deviation 3√(9/2). The negative scale reverses order but cannot create a negative variance.</Prose></Practice>
    <Practice title="3. Refute independence with a real witness" prompt="Take X uniform on −2,0,2 and Y=X². Compute covariance and show why it is not an independence test." hint="Use symmetry for E[X] and E[X³], then inspect X=0,Y=4."><Prose>E[X]=E[X³]=0, so covariance is 0. But the joint mass at(0,4) is 0 while the marginal product is (1/3)(2/3)=2/9. The dependence is nonlinear and completely specified by Y=X².</Prose></Practice>
    <Practice title="4. Two dice, two new variables" prompt="For two independent fair dice A,B, define S=A+B and D=A−B. Find both means, variances and covariance; decide whether S,D are independent." hint="Expand using Var(A)=Var(B)=35/12, then check one joint cell."><Prose>E[S]=7, E[D]=0; both variances are 35/6. Cov(S,D)=Var(A)−Var(B)=0. Yet P(S=2,D=0)=1/36, whereas P(S=2)P(D=0)=(1/36)(1/6)=1/216. The variables are dependent.</Prose></Practice>
    <Example name="dice" />
    <Practice title="5. Calibrate the task before comparing variance" prompt="Now A=θ+S+e_A and B=θ+S+e_B measure the same target, with independent zero-mean S,e_A,e_B of variances 9,1,4. For T=aA+(1−a)B, find the variance-minimizing weight. What does B−A estimate?" hint="The common term always has coefficient 1; minimize a²+4(1−a)²."><Prose>Var(T)=9+a²+4(1−a)². Completing the square gives 9+5(a−4/5)²+4/5. The optimum is a=4/5, variance 49/5. It remains unbiased for θ under the stated zero-mean model. B−A has mean 0 and variance 5; it cancels θ as well as S, so it is not an estimator of θ. Its smaller variance answers a different question.</Prose></Practice>
    <Practice title="6. An impossible covariance proposal" prompt="A proposed three-variable correlation matrix has every diagonal 1 and every off-diagonal −0.8. Can it be valid?" hint="Calculate the variance of the sum of all three standardized variables."><Prose>The sum would have variance 3+2·3·(−.8)=−1.8. A variance cannot be negative, so the matrix is not positive semidefinite and cannot be a correlation matrix. Having every individual entry in [−1,1] is not enough.</Prose></Practice>
    <Practice title="7. Weight the conditional cells" prompt="Group A has probability 3/4, conditional Y mean 2 and variance 1. Group B has probability 1/4, conditional mean 6 and variance 9. Find the overall mean and variance. Does each group's variance have to be smaller?" hint="Add the average within-group variance to the variance of the two means."><Prose>The overall mean is 3. Average within variance is (3/4)1+(1/4)9=3. Between variance is (3/4)(2−3)²+(1/4)(6−3)²=3. Total variance is 6. Group B's variance 9 exceeds 6, so the law controls an average of conditional variances, not a bound for each subgroup.</Prose></Practice>
    <Practice title="8. Keep both continuous branches" prompt="For U uniform on [−1,1], calculate P(.04≤U²≤.49) and E[(U²)²]. What is P(U²=0)?" hint="Translate the interval into positive and negative U values."><Prose>The preimages are[−.7,−.2] and[.2,.7]. Total length 1 at density 1/2 gives probability 1/2. E[U⁴]=1/5. The singleton has probability 0; an unbounded density near 0 does not create an atom.</Prose></Practice>
    <Practice title="9. Check the theorem's input" prompt="A Pareto law has alpha=1.8. A sample returns variance 12. Can you put 12 into the known-variance Chebyshev formula for its iid mean?" hint="Distinguish the population moment condition from a finite statistic."><Prose>No. The mean exists because 1.8&gt;1, but the second moment diverges because 1.8≤2. A finite sample variance does not supply a finite population variance or a guaranteed upper bound. Even for a finite-variance population, substituting an observed variance into that theorem needs separate justification.</Prose></Practice>
    <Practice title="10. Diagnose copied observations" prompt="A Bernoulli(1/4) bit is copied into every row of a dataset of size 100. Find the expected sample mean, its variance and the realized corrected sample variance." hint="Every row is the same random variable."><Prose>The sample mean is the original bit: expectation 1/4 and variance 3/16. In either realized dataset all rows are equal, so the sample variance is 0 even with denominator 99. Independence across rows is absent; more copies do not produce 100 independent trials.</Prose></Practice>
    <Practice title="11. Identify what a zero covariance does permit" prompt="X,Y have means 0, variances 4 and 9, and covariance 0. What is Var(2X−Y)? Can you conclude E[X²Y²]=E[X²]E[Y²]?" hint="Use the second-moment identity only for the quantity it covers."><Prose>Var(2X−Y)=4·4+9−4·0=25. The product of squared variables is a fourth-order question, not determined by zero covariance; those fourth moments may not even exist. Independence plus suitable integrability would justify a factorization, but it has not been supplied.</Prose></Practice>
    <Prose><strong>Readiness check.</strong> Explain where each probability mass came from; calculate a transformed expectation and a paired covariance; supply an explicit independence witness; preserve units and the estimand in a noise combination; derive a conditional decomposition; and identify which sampling or moment assumption a claimed guarantee requires.</Prose>
    <Prose><strong>Next in this module:</strong> <a href="/learn/path/full-curriculum/sampling-measurement-experimental-design?module=math-foundations">Sampling, Measurement &amp; Experimental Design</a> asks how data collection produces the observations these calculations use. Carry forward the distinction between a new independent unit, a repeated reading and a copied value. Publication status does not change that syllabus sequence.</Prose>
    <Sources alternatives={<p>For a spoken route, the <a href="https://stat110.hsites.harvard.edu/youtube" target="_blank" rel="noreferrer">official Harvard Stat 110 guide</a> links lectures 8–9 on random variables/expectation, 21 on covariance and 25–29 on conditioning and averages. The course map and recording identities were checked; full-video viewing is not claimed. <a href="https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/resources/derivation-of-the-law-of-total-variance/" target="_blank" rel="noreferrer">John Tsitsiklis's MIT total-variance derivation</a> is a focused alternative after the finite grouping example; its official resource identity was checked.</p>}>
      <li><a href="https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class04-prep-b.pdf" target="_blank" rel="noreferrer">MIT 18.05: expected value</a> — selected weighted-average, linearity and infinite-mean discussion; a concise undergraduate written route with additional problems.</li>
      <li><a href="https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class07-prep-b.pdf" target="_blank" rel="noreferrer">MIT 18.05: covariance and correlation</a> — selected finite definitions, sum-variance identities and independence counterexample. Use the finite sections first; all numbers in this lesson were checked independently.</li>
      <li><a href="https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-a.pdf" target="_blank" rel="noreferrer">MIT 18.05: continuous moments</a> — selected definition and uniform/exponential moment passages connect sums to integrals. The integral branch assumes basic calculus.</li>
      <li><a href="https://web.mit.edu/18.06/www/Spring21/Lecture%20notes.pdf" target="_blank" rel="noreferrer">MIT 18.06, lecture 31: covariance matrices</a> — equations 288–291 and the following degeneracy explanation connect weighted outer products to positive semidefiniteness and the variance of linear combinations. The full notes were not reviewed.</li>
      <li><a href="https://www2.stat.duke.edu/courses/Fall19/sta721/lectures/NormalTheory/multnorm.pdf" target="_blank" rel="noreferrer">Duke STA 721: multivariate normal theory</a> — the block-independence construction explains why the zero-covariance exception requires a jointly Gaussian vector. This is a specialist follow-up, not an assumption needed for our finite examples.</li>
    </Sources>
  </div>
};
