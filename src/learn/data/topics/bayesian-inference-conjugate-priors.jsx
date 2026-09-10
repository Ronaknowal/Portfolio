import { H2, H3, Prose, Code } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample as CompleteExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { BayesianUpdateLab, BetaEvidenceFigure, BatchPredictionLab, SharedParameterFigure, GammaExposureLab, DirichletCompositionFigure, NormalPrecisionLab, PredictivePatternLab } from '../../components/lesson-labs/BayesianInferenceLabs.jsx';
import { bayesianInferenceExamples as examples } from '../bayesian-inference-examples.js';
function Practice({
  question,
  prompt,
  hint,
  children
}) {
  return <section className="lesson-check"><h3>{question}</h3>{prompt}<details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Show explained solution</summary>{children}</details></section>;
}

function RunnableExample({ example }) {
  return <section><Prose><strong>Before running.</strong> {example.question}</Prose><CompleteExample example={example} /></section>;
}
export default {
  title: 'Bayesian Inference & Conjugate Priors',
  readTime: '~70 min read + 2–3 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot bayesian-inference-lesson">
    <LessonIntro prerequisites="Conditional probability, probability mass versus density, expectation and variance from Probability Distributions. MLE/MAP introduced fitting a parameter; Hypothesis Testing distinguished repeated-sampling intervals. We recap the required distinctions before using them. The optional unknown-variance branch uses completing a quadratic square and an additional distribution family." sections={[['1-eight-visitors-do-not-determine-a-rate', 'Specify the model'], ['2-derive-the-update-and-keep-its-uncertainty', 'Beta updating and intervals'], ['3-predict-a-batch-with-shared-uncertainty', 'Future counts and dependence'], ['4-add-only-new-evidence-and-question-the-prior', 'Sequential updates and sensitivity'], ['5-count-events-and-the-time-you-watched', 'Gamma rates and exposure'], ['6-move-from-two-outcomes-to-many', 'Categorical conjugacy'], ['7-combine-noisy-measurements-by-precision', 'Normal means and predictions'], ['8-check-what-the-model-fails-to-describe', 'Predictive model checks'], ['9-compare-models-and-choose-an-action', 'Evidence, decisions and computation'], ['10-practise-the-complete-inference-loop', 'Independent practice']]}>Eight of ten sampled visitors try a new feature. Should you expect eight successes among the next ten? How confident should you be that the underlying adoption rate exceeds 70%? Keep the original visitor example, but follow it all the way from assumptions to a posterior distribution, future outcomes, model checks and a decision.</LessonIntro>

    <H2>1. Eight visitors do not determine a rate</H2>
    <Prose>The observed fraction is .8. The <strong>long-run adoption probability</strong>, written θ (“theta”), is unknown. Ten visitors provide limited information: several rates could plausibly produce eight successes. A Bayesian model represents uncertainty about θ with a distribution, then updates that distribution when observations arrive. It does not declare that a particular physical probability must itself fluctuate between visitors.</Prose>
    <Prose>Specify the experiment before calculating. Our synthetic example assumes independently sampled binary outcomes <em>conditional on the same fixed θ</em>. Every visitor has that adoption chance; the sample was not filtered to show enthusiastic users. Repeat visits, targeted recruitment, time drift and correlated households can require a different likelihood. A mathematically correct update cannot repair those assumptions automatically.</Prose>
    <LessonTable caption="Three distributions answer three different questions" headers={['Object', 'What varies?', 'Question']} rows={[['Likelihood L(θ; D)', 'Candidate parameter, observed data D fixed', 'How compatible are these observations with this rate?'], ['Posterior π(θ | D)', 'Parameter under the model after data', 'How is our remaining uncertainty distributed?'], ['Predictive distribution', 'Future observable outcomes', 'What could happen next, averaging over that uncertainty?']]} />
    <Prose>A <strong>prior</strong> π(θ) expresses the uncertainty before this dataset. The likelihood supplies the data's relative support for candidate rates. Multiplying gives an unnormalized posterior. Divide by the total weighted likelihood so its area is one:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\pi(\theta\mid D)=\frac{L(\theta;D)\pi(\theta)}{Z(D)},\\
Z(D)=\int_0^1 L(u;D)\pi(u)\,du.
\end{gathered}`}</MathBlock>
    <Prose>The integration variable u is a placeholder. The denominator Z is called the <strong>marginal likelihood</strong> or <strong>evidence</strong>. It must be finite and positive for this posterior formula to work. It is a probability for discrete data and a data density for continuous data; it is not generally a probability density over θ. Later we will need its actual value when comparing models.</Prose>
    <Prose>The preceding MLE/MAP lesson chooses particular parameter values. Here the distribution remains central: a point summary alone cannot give an interval or a future-batch tail probability. All examples are synthetic teaching calculations. Ten programs use Python's standard library; the explicitly marked unknown-variance program uses SciPy for Student-t quantiles. Save one complete block as <Code>bayesian_example.py</Code> and run <Code>python bayesian_example.py</Code>. Browser investigations evaluate separate bounded analytic models.</Prose>

    <H2>2. Derive the update and keep its uncertainty</H2>
    <Prose>A <strong>Beta distribution</strong> describes a number between zero and one. Its positive shape parameters α and β determine how its density spreads across possible rates. Beta(1, 1) is uniform; Beta(2, 2) prefers the middle; Beta(20, 20) has a much stronger preference near one half. Beta(3, 7) favors lower rates. The parameters specify a distribution, not extra actual visitors.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\pi(\theta)=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},\\
0<\theta<1,\\
B(\alpha,\beta)=\int_0^1 u^{\alpha-1}(1-u)^{\beta-1}\,du.
\end{gathered}`}</MathBlock>
    <Prose>The Beta function B is simply the area needed to normalize those powers. Positive α and β make the integral finite. For positive integer shapes, B(α, β) = (α−1)!(β−1)!/(α+β−1)!, which supplies exact fractions for our modest examples.</Prose>
    <Prose>Let s count successes and f count failures. The likelihood of one particular ordered sequence is θˢ(1−θ)ᶠ. If only the total count was recorded, multiply by the number of possible arrangements, choose(s+f, s). That multiplier is constant in θ and cancels when normalizing the posterior. It still belongs in the probability of the count event.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
L(\theta;D)\pi(\theta)\\
\propto\
\theta^{s+\alpha-1}(1-\theta)^{f+\beta-1},\\
\theta\mid D\sim\operatorname{Beta}(a,b),\\
a=\alpha+s,\qquad b=\beta+f.
\end{gathered}`}</MathBlock>
    <Prose>The posterior is another Beta distribution because multiplying powers adds their exponents. This is <strong>conjugacy</strong>: a prior family stays within that family under the chosen likelihood. It makes this calculation exact and convenient. It says nothing about whether the likelihood describes real visitors well.</Prose>
    <BetaEvidenceFigure />
    <Prose>The mean and variance follow by integrating θ or θ² against the posterior. For example, E[θ|D] = B(a+1,b)/B(a,b) = a/(a+b). Applying the same ratio to θ², then subtracting the squared mean, gives:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\mathbb E[\theta\mid D]=\frac{a}{a+b},\\
\operatorname{Var}(\theta\mid D)\\ =
\frac{ab}{(a+b)^2(a+b+1)}.
\end{gathered}`}</MathBlock>
    <Prose>With the Beta(2, 2) prior and 8/10 data, the posterior is Beta(10, 4). Its mean is 5/7 ≈ .714286. For a,b &gt; 1, differentiating the log density gives the mode (a−1)/(a+b−2), here 3/4. The mean, mode and observed fraction .8 are different summaries with different meanings.</Prose>
    <H3>An interval contains probability, not a guarantee about every study</H3>
    <Prose>A <strong>95% credible interval</strong> contains 95% of posterior probability under the stated model and prior. An equal-tailed interval leaves .025 below its lower endpoint and .025 above its upper endpoint. It need not be the shortest interval. A highest-density region answers another geometric question and can be disconnected for a multimodal posterior; density-based regions also depend on the parameter coordinate, as the MAP lesson explained.</Prose>
    <BayesianUpdateLab />
    <Prose>For Beta(10, 4), the equal-tailed 95% interval is approximately [.461868, .909080]. The probability that θ exceeds .70 is about .579394. These are probabilities about the unknown <em>parameter</em> under this posterior, not statements that the next visitor's binary outcome lies between .46 and .91.</Prose>
    <RunnableExample example={examples.betaUpdate} />
    <Prose>The program's integer-shape Beta CDF equals a binomial upper tail; its survival function uses the other tail directly. A bisection repeatedly halves a bracket until its CDF reaches the requested probability. This avoids installing a library for the original example. General positive real shapes, extreme probabilities or very large counts deserve a validated numerical library, with direct survival methods when subtracting from one would lose a tiny tail.</Prose>
    <Prose>A 95% <strong>confidence interval</strong> instead describes repeated-sampling coverage of a procedure at a fixed parameter. Similar endpoints do not make these interpretations interchangeable. A Bayesian credible procedure has its corresponding average coverage under the joint prior-and-data model; that does not promise 95% coverage at every particular fixed θ or under a misspecified model.</Prose>
    <Checkpoint prompt="Increase the sample from 8/10 to 80/100 successes while keeping Beta(2,2). What happens to the mean, and why?"><Prose>The posterior becomes Beta(82,22), with mean 82/104 = 41/52 ≈ .788462. It moves closer to the observed fraction .8 because the fixed prior weight is now smaller relative to the new dataset. This compares two sample sizes; it does not count the original ten observations twice.</Prose></Checkpoint>
    <Checkpoint prompt="Under a uniform prior, ten failures and no successes give which next-success probability? Is a boundary MAP of zero the same thing?"><Prose>The posterior is Beta(1, 11), whose mean and next-success probability are 1/12. Its density mode is at zero, but a continuous posterior assigns zero probability to any individual point. A finite sequence of failures has not proved that success is impossible.</Prose></Checkpoint>
    <details><summary>Positive shapes below one are valid too</summary><Prose>Beta(.5, .5) has an integrable density that diverges toward both endpoints. A plot that clips those heights does not make the clipped points finite modes. The update still adds observed counts, but the familiar interior-mode formula needs a,b &gt; 1. Beta(1,1) is flat; if one shape is one and the other exceeds one, its maximum is at the corresponding boundary. Our browser's positive integer priors keep all displayed heights finite; this restriction belongs to the demonstration, not to Bayesian inference.</Prose></details>

    <H2>3. Predict a batch with shared uncertainty</H2>
    <Prose>To predict a next success, average its chance θ across the posterior: P(Y=1|D) = E[θ|D] = a/(a+b). For m future visitors, the expected number of successes is m times that number. But a mean is not a distribution. All future visitors share the same uncertain θ.</Prose>
    <SharedParameterFigure />
    <Prose>At a fixed θ, the future count K is Binomial(m, θ). Integrating its complete probability mass against Beta(a,b) gives the <strong>Beta-Binomial predictive distribution</strong>:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
P(K=k\mid D)\\
={m\choose k}\int_0^1\\
\theta^k(1-\theta)^{m-k}
\pi(\theta\mid D)\,d\theta\\
={m\choose k}\frac{B(a+k,b+m-k)}{B(a,b)},\\
k=0,\ldots,m.
\end{gathered}`}</MathBlock>
    <Prose>The integral is another Beta normalizer. This is a useful recurring trick: integrate a prediction by recognizing the normalized family, rather than replace all uncertainty with a single best rate.</Prose>
    <BatchPredictionLab />
    <Prose>For ten future visitors after Beta(10,4), both the integrated model and Binomial(10, 5/7) expect 50/7 ≈ 7.142857 successes. The integrated variance is 160/49 ≈ 3.265306; the plug-in variance is only 100/49 ≈ 2.040816. The probability of at least nine successes is about .250736 versus .172858. Shared uncertainty changes the risk of an extreme batch.</Prose>
    <H3>Derive where the extra variance comes from</H3>
    <Prose>Write p = a/(a+b). For two distinct future indicators Yᵢ,Yⱼ, conditional independence gives E[YᵢYⱼ|θ] = θ². After averaging over θ, their covariance is E[θ²|D]−E[θ|D]² = Var(θ|D), which is positive here. Each single future indicator still has variance p(1−p). Add the m individual variances and the m(m−1) ordered pair covariances:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\operatorname{Var}(K\mid D)\\
=mp(1-p)\\
{}+m(m-1)\operatorname{Var}(\theta\mid D)\\
=mp(1-p)\frac{a+b+m}{a+b+1}.
\end{gathered}`}</MathBlock>
    <Prose>The correlation of distinct future indicators is 1/(a+b+1). This is dependence from sharing an uncertain rate, not an added story that one visitor persuades another. A different model with a fresh independently drawn θ for every visitor would remove that shared uncertainty and produce the plug-in Binomial count distribution.</Prose>
    <RunnableExample example={examples.batchPrediction} />
    <Prose>With m=1 the two distributions agree. With the same posterior mean but increasing concentration a+b, their difference shrinks for a fixed future batch. If a rare high count has a large operational cost, use the distribution appropriate to the shared-rate model rather than infer safety from matching means.</Prose>

    <H2>4. Add only new evidence and question the prior</H2>
    <Prose>After Beta(10,4), five new visitors give three successes and two failures. Reusing the posterior as the next prior gives Beta(13,6), mean 13/19 ≈ .684211. Starting again from Beta(2,2) and pooling all eleven successes/four failures gives the same result. Multiplication is associative, and the sufficient totals add.</Prose>
    <Prose>The order of genuinely new observations does not matter under this fixed common-rate likelihood. Counting the original ten rows twice instead gives Beta(18,6), a different and unjustified answer. Identical observed values can be new evidence; duplicated records are not. If the rate itself changes with time, updating one fixed θ forever can also become the wrong model.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\frac{\alpha+s}{\alpha+\beta+n}
=\frac{\alpha+\beta}{\alpha+\beta+n}\frac{\alpha}{\alpha+\beta}\\
{}+\frac{n}{\alpha+\beta+n}\frac{s}{n},\qquad n>0.
\end{gathered}`}</MathBlock>
    <Prose>This expresses the posterior mean as a weighted compromise between prior mean and observed fraction. In this particular mean formula, α+β acts like a prior weight. The analogy does not create historical observations. The MAP uses different exponents; a prior's meaning also depends on its family, coordinate and the quantity being predicted.</Prose>
    <LessonTable caption="The same 8/10 evidence under four declared priors" headers={['Prior', 'Posterior', 'Mean']} rows={[['Beta(1,1)', 'Beta(9,3)', '.75'], ['Beta(2,2)', 'Beta(10,4)', '5/7 ≈ .714286'], ['Beta(20,20)', 'Beta(28,22)', '.56'], ['Beta(3,7)', 'Beta(11,9)', '.55']]} />
    <Prose>With little data, credible alternatives for the prior can meaningfully change the result. Compare their implications for observable outcomes before fitting, then report sensitivity in the decisions that matter. A tight posterior may reflect strong assumptions rather than many independent observations.</Prose>
    <H3>Learning does not make every realized variance decrease</H3>
    <Prose>Beta(100,1) is very concentrated near one. One surprising failure gives Beta(100,2), moving probability away from that boundary and increasing the variance from about .00009611 to .00018663. There is no rule that each realized posterior must get narrower. Under a coherent model, expected conditional variance decreases on average as information is added; an individual surprising outcome can widen it.</Prose>
    <RunnableExample example={examples.sequentialSensitivity} />
    <Prose>A prior that gives zero mass to a whole region cannot acquire positive mass there merely by multiplying by a finite likelihood. The program illustrates that issue in a finite candidate model with rate .8 excluded. In a continuous model, zero probability at a <em>single point</em> is normal and does not exclude its neighborhood. Inspect support, not just a density's value at one point.</Prose>
    <Prose>An <strong>improper prior</strong> has infinite total mass and cannot itself be sampled as a probability distribution. Some yield proper posteriors after enough suitable data, but that must be proved for the model. The formal density 1/[θ(1−θ)] gives kernel θˢ⁻¹(1−θ)ᶠ⁻¹: it normalizes only when both s and f are positive. Calling it Beta(0,0) does not make it a proper Beta distribution. Its arbitrary normalization constant also prevents ordinary marginal-likelihood comparison. A diffuse proper prior and an improper expression are different objects.</Prose>

    <H2>5. Count events and the time you watched</H2>
    <Prose>A device log records three events in half an hour and six events in two hours. A probability between zero and one is no longer the parameter: we need a nonnegative <strong>event rate</strong> λ, in events per hour. Under a homogeneous Poisson model, disjoint observation intervals are conditionally independent given λ, and count xᵢ in exposure tᵢ has mean λtᵢ. The observation periods and event-detection process must be trustworthy.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
P(x_i\mid\lambda,t_i)=\frac{(\lambda t_i)^{x_i}e^{-\lambda t_i}}{x_i!},\\
L(\lambda;D)\propto
\lambda^{\sum_i x_i}e^{-\lambda\sum_i t_i}.
\end{gathered}`}</MathBlock>
    <Prose>Exposure is the amount of time during which events could have been counted. An hour with no events is informative exposure, not a missing row to discard. Zero exposure with a positive count would be impossible under this model. Our interactive windows are strictly positive.</Prose>
    <Prose>A <strong>Gamma(a,b)</strong> prior has density bᵃλᵃ⁻¹exp(−bλ)/Γ(a) for λ&gt;0, with a,b&gt;0. Here b is the Gamma <em>rate parameter</em>, not its scale; Γ extends factorial, with Γ(k)=(k−1)! for positive integers. Since bλ is dimensionless, b has units of hours while λ has units of inverse hours. Mean λ is a/b and variance is a/b². Writing both quantities merely as “rate” without units invites an error.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\lambda^{a-1}e^{-b\lambda}
\lambda^{\sum x_i}e^{-\lambda\sum t_i}\\
=\lambda^{a+\sum x_i-1}e^{-(b+\sum t_i)\lambda},\\
\lambda\mid D\\
\sim\operatorname{Gamma}(a+\textstyle\sum x_i,\ b+\sum t_i).
\end{gathered}`}</MathBlock>
    <GammaExposureLab />
    <Prose>Start with Gamma(2,1 hour). The two intervals add nine events and 2.5 hours, producing Gamma(11,3.5 hours), mean 22/7 ≈ 3.142857 events/hour. The likelihood-only estimate is total events / total time = 9/2.5 = 3.6. Averaging the interval rates, (6+3)/2 = 4.5, would incorrectly weight half an hour as heavily as two hours.</Prose>
    <H3>Predict a future count, not just the rate</H3>
    <Prose>For future exposure h, integrate a Poisson count over the Gamma posterior. Its powers again produce a Gamma integral. With posterior shape a and rate b, the resulting mass is:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
P(K=k\mid D,h)\\ =
\frac{\Gamma(a+k)}{\Gamma(a)\,k!}\\
{}\times
\left(\frac b{b+h}\right)^a
\left(\frac h{b+h}\right)^k,\\
\mathbb E[K\mid D,h]=\frac{ah}{b},\\
\operatorname{Var}(K\mid D,h)\\
=\frac{ah}{b}\left(1+\frac hb\right).
\end{gathered}`}</MathBlock>
    <Prose>This Gamma-Poisson mixture is a negative-binomial count distribution under a specified parameter convention. The mean uses expected λh. The variance adds the Poisson noise ah/b and uncertainty h²Var(λ|D); plugging in the posterior mean rate loses the latter. For the next hour in our example, the expected count is 22/7, variance 198/49 and probability of zero events about .063010.</Prose>
    <RunnableExample example={examples.gammaExposure} />
    <Prose>Changing hours to minutes divides the numerical event rate by 60 and multiplies b and exposure by 60. Probability of the same next-hour event is unchanged. The model can support inspection scheduling or error-count planning, but changing detection thresholds, missing logs and clustered failures challenge the common-rate Poisson assumptions. A conjugate update is not evidence those problems are absent.</Prose>

    <H2>6. Move from two outcomes to many</H2>
    <Prose>Suppose a log classifies a request into three declared outcomes A, B and C. The unknown parameter is now a vector (θ₁,θ₂,θ₃): nonnegative category probabilities adding to one. The set of such compositions is a <strong>simplex</strong>. Increasing one component must be balanced by the others; they cannot all vary independently while preserving the sum.</Prose>
    <Prose>The <strong>Dirichlet</strong> distribution generalizes Beta to K categories. Positive shapes α₁,…,αₖ give density proportional to the product of θⱼ raised to αⱼ−1 on that simplex. If conditional trials are independent and cⱼ counts category j, the likelihood contributes θⱼᶜʲ. Multiplying adds each category's count to its shape:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\boldsymbol\theta\mid D
\sim\operatorname{Dirichlet}(A_1,\ldots,A_K),\\
A_j=\alpha_j+c_j,\qquad A=\sum_j A_j,\\
\mathbb E[\theta_j\mid D]=\frac{A_j}{A},\\
P(Y_{\rm next}=j\mid D)=\frac{A_j}{A}.
\end{gathered}`}</MathBlock>
    <DirichletCompositionFigure />
    <Prose>For prior (1,1,1) and counts (6,3,1), posterior shapes are (7,4,2), giving next-category probabilities 7/13, 4/13 and 2/13. The strip is their mean composition, not the whole posterior distribution. Integrating products of components gives:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\operatorname{Var}(\theta_j\mid D)=\frac{A_j(A-A_j)}{A^2(A+1)},\\
\operatorname{Cov}(\theta_i,\theta_j\mid D)\\
=-\frac{A_iA_j}{A^2(A+1)},\quad i\ne j.
\end{gathered}`}</MathBlock>
    <Prose>The negative covariance expresses the sum constraint. The marginal distribution of one θⱼ is Beta(Aⱼ,A−Aⱼ), so a single-category credible interval can use a Beta calculation. Separate 95% marginal intervals are not automatically a joint 95% region for the entire vector.</Prose>
    <details><summary>Where the multivariate normalizer and count prediction come from</summary><Prose>The Dirichlet normalizer is B(α⃗) = ∏Γ(αⱼ)/Γ(Σαⱼ), the simplex integral of its power product. Increasing one exponent by one gives the mean through B(A⃗+eⱼ)/B(A⃗); increasing two gives second moments and the covariance above. For a future count vector k⃗ with total m, integrating the multinomial mass gives m!/∏kⱼ! times B(A⃗+k⃗)/B(A⃗). This is the Dirichlet-multinomial predictive law; components and future trials retain their shared-parameter dependence.</Prose></details>
    <RunnableExample example={examples.dirichletCategories} />
    <Prose>With counts (6,4,0) under the same prior, category C still has next-event probability 1/13. That is useful smoothing for a <em>declared but unseen</em> category. It does not create probabilities for outcomes absent from the category universe. An unknown label, newly discovered species or changing taxonomy requires an explicit expanded or open-category model. K=2 reduces to the Beta case, providing a useful consistency check.</Prose>

    <H2>7. Combine noisy measurements by precision</H2>
    <Prose>The measurements [2,3,4,7] return from MLE/MAP. Their mean is four. Assume independent normal observation noise with known variance σ²=4 and a Normal(m₀=0,v₀=1) prior on the unknown shared mean μ. Here Normal is parameterized by <em>mean and variance</em>; σ is a standard deviation.</Prose>
    <Prose>The negative log posterior contains two quadratic costs: Σ(xᵢ−μ)²/(2σ²) and (μ−m₀)²/(2v₀). Expand their μ² and μ terms. The μ² coefficient determines posterior precision, the reciprocal of variance; the linear coefficient determines its center. Completing the square yields:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\tau_0=1/v_0,\qquad \tau_D=n/\sigma^2,\\
v=(\tau_0+\tau_D)^{-1},\\
m=v(\tau_0m_0+\tau_D\bar x),\\
\mu\mid D\sim\operatorname{Normal}(m,v).
\end{gathered}`}</MathBlock>
    <Prose>Each genuinely independent observation contributes 1/σ² precision. Our prior and four-reading sample each contribute one, so the posterior mean is two and variance is one half. More precise sources receive more weight. If observation variances differ and are known, replace n/σ² with Σ1/σᵢ² and n x̄/σ² with Σxᵢ/σᵢ²; the same quadratic argument proves the update. Correlated measurements require their joint covariance instead of this scalar sum.</Prose>
    <NormalPrecisionLab />
    <Prose>A future reading is Xnew = μ + εnew. Under the model, its fresh noise is independent of μ and has variance σ². Integrating the posterior therefore gives Normal(m, v+σ²), not Normal(m,v). In our example the mean's 95% credible interval is [.614096,3.385904], while the next-reading 95% predictive interval is [−2.157711,6.157711]. Learning the mean does not remove observation noise.</Prose>
    <RunnableExample example={examples.normalPrecision} />
    <Prose>This distinction matters when calibrating a sensor. Estimating its common offset can become precise even though any one reading remains noisy. If readings share an unmodeled drift, the apparent added precision may be fictional. The prior also needs interpretable units: changing volts to millivolts multiplies means by 1000 and variances by a million, preserving corresponding event probabilities.</Prose>
    <details><summary>Deeper: when both mean and noise variance are unknown</summary>
      <Prose>Replacing σ² with a fitted estimate and pretending it is known discards uncertainty. One conjugate joint model instead uses V=σ²&gt;0, an Inverse-Gamma(α₀,β₀) prior with density proportional to V⁻ᵅ⁰⁻¹ exp(−β₀/V), and μ|V ~ Normal(m₀,V/κ₀). Take α₀,β₀,κ₀&gt;0. Here β₀ is the inverse-Gamma scale in this explicit density convention. The prior on μ and V is coupled, not two independent priors.</Prose>
      <Prose>Let S = Σ(xᵢ−x̄)². Combine the data's quadratic n(μ−x̄)² with κ₀(μ−m₀)². Completing the square gives a new centered quadratic with coefficient κ₀+n plus an extra disagreement term κ₀n(x̄−m₀)²/(κ₀+n). Matching the remaining powers of V gives:</Prose>
      <MathBlock>{String.raw`\begin{gathered}
\kappa_n=\kappa_0+n,\\
m_n=\frac{\kappa_0m_0+n\bar x}{\kappa_n},\\
\alpha_n=\alpha_0+n/2,\\
\beta_n=\beta_0+S/2\\ {}+
\frac{\kappa_0n(\bar x-m_0)^2}{2\kappa_n}.
\end{gathered}`}</MathBlock>
      <Prose>The posterior has V|D ~ Inverse-Gamma(αn,βn) and μ|V,D ~ Normal(mn,V/κn). Integrating V gives Student-t distributions with 2αn degrees of freedom. The mean parameter's t scale squared is βn/(αnκn); a future reading's scale squared is βn(κn+1)/(αnκn). A t scale is not its standard deviation: for ν&gt;2, variance is ν/(ν−2) times scale squared. Very small degrees of freedom can make moments fail to exist even though the distribution is proper.</Prose>
      <RunnableExample example={examples.unknownVariance} />
      <Prose>For the declared prior (m₀,κ₀,α₀,β₀)=(0,1,2,2), the data produce (3.2,5,4,15.4). The predictive t has eight degrees of freedom, scale about 2.149419 and variance 6.16. This is a different joint prior/model from the earlier known-variance example, so its interval is not simply an “improved version” of that interval. The two-dimensional integral supplies the heavier tails.</Prose>
    </details>

    <H2>8. Check what the model fails to describe</H2>
    <Prose>Conjugate inference can be computationally exact and still miss the important pattern. The two sequences SSSSSSSSFF and SSFSSSFSSS both have eight successes and two failures. Their common-rate likelihoods and Beta posteriors are identical. The first sequence may suggest a time change; the totals discard that order.</Prose>
    <Prose>A <strong>prior predictive check</strong> asks what data the prior and observation model imply before using the current data: draw θ from the prior, then generate a complete dataset conditional on that θ. A <strong>posterior predictive check</strong> draws θ from the posterior and generates a replicated dataset under the same observation design. Compare a relevant pattern—such as variation, extremes or ordering—with the observed data.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
p(D^{\rm rep}\mid D)\\
=\int p(D^{\rm rep}\mid\theta)\pi(\theta\mid D)\,d\theta.
\end{gathered}`}</MathBlock>
    <Prose>Do not use only a statistic the fitted model almost automatically matches. A one-rate Bernoulli model is fitted largely through its count; a run statistic instead asks about ordering. A run is a consecutive block of identical outcomes. The clustered sequence has two runs, whereas the separated-failure sequence has five.</Prose>
    <PredictivePatternLab />
    <Prose>The exact posterior predictive probability of at most two runs here is about .131628. If we instead condition on exactly eight successes, all 45 possible orderings are equally likely under the model and only two have at most two runs, giving 2/45 ≈ .044444. These probabilities differ because the reference experiments differ. The predictive experiment allows different totals; the conditional ordering experiment fixes the total.</Prose>
    <RunnableExample example={examples.predictivePatterns} />
    <Prose>Posterior predictive tail probabilities generally are <em>not</em> uniformly calibrated classical p-values, because the data help fit the predictive reference. A small value can locate a model concern; a moderate one cannot certify every assumption. Noticing a pattern and then selecting the statistic also changes the assessment context. Use domain knowledge and additional data to investigate temporal dependence or drift rather than turn one displayed tail into an automatic verdict.</Prose>
    <Prose>Prior predictive checks reveal whether a “weak” prior produces plausible observable behavior. With a uniform Beta prior, the prior predictive count of successes among m trials is uniform over 0,…,m—not a Binomial concentrated near m/2. Before observing anything, that prior permits many rates, making extreme batches substantially more plausible than fixing θ=.5 would. Similar checks can expose a Gamma rate prior that implies implausibly many failures per hour.</Prose>
    <Prose>Posterior predictive checking and held-out prediction answer related but different questions. Replicating the training design asks whether fitted models reproduce chosen aspects of it; genuinely new withheld observations test predictive performance under a specified future-data process. A neat replicated plot cannot rule out selection bias or distribution shift outside that process.</Prose>

    <H2>9. Compare models and choose an action</H2>
    <H3>The normalizing constant can be the quantity you need</H3>
    <Prose>Consider two explicit models: M₀ fixes θ=.5; M₁ gives θ a proper Beta prior. Model uncertainty is represented by a separate discrete model indicator. This is how a point-null model can receive positive posterior probability; an ordinary continuous Beta posterior by itself gives P(θ=.5|D)=0.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\frac{P(M_1\mid D)}{P(M_0\mid D)}\\
=\frac{p(D\mid M_1)}{p(D\mid M_0)}
\frac{P(M_1)}{P(M_0)},\\
p(D\mid M_1)=\frac{B(\alpha+s,\beta+f)}{B(\alpha,\beta)}.
\end{gathered}`}</MathBlock>
    <Prose>The evidence ratio is a <strong>Bayes factor</strong>. For the 8/10 sequence, a uniform alternative prior gives factor about 2.068687 against the fair model. With equal model priors, the alternative's posterior model probability is about .674128. This is moderate evidence in this particular comparison, not proof the feature is useful or that every conceivable model was tested.</Prose>
    <RunnableExample example={examples.modelEvidence} />
    <Prose>The binomial arrangement factor cancels between these models only because both use the same observed count event. In general, constants can differ between models; dropping them before computing evidence gives the wrong comparison. Evidence also depends on prior spread and support, which is why the program compares several declared proper alternatives. An improper prior's arbitrary scale cannot supply an ordinary model evidence without additional theory.</Prose>
    <H3>Translate uncertainty into a stated loss</H3>
    <Prose>Suppose the decision is whether to launch when the true rate exceeds .70. Let q = P(θ&gt;.70|D). If an incorrect launch costs 19 and missing a beneficial launch costs 1, posterior expected loss is 19(1−q) for launch and q for waiting. Launch is preferred only when q&gt;19/20=.95. Our q≈.579394 does not pass.</Prose>
    <Prose>Now change the question: each adoption brings 100 units of value and serving a visitor costs 70 regardless of outcome; waiting yields zero. Expected net value per visitor is 100E[θ|D]−70≈1.428571, so that declared linear utility favors launching. The two actions differ because the loss functions differ, not because the posterior arithmetic changed. These synthetic values demonstrate decision reasoning, not a recommendation for a real financial or clinical decision.</Prose>
    <RunnableExample example={examples.decisionLoss} />
    <Prose>Decision thresholds should follow costs, constraints and the population of interest; inference does not choose them automatically. Record the sampling and stopping process too. Bayesian computation does not make arbitrary repeated thresholds, chosen models or selected reports inherit a frequentist error guarantee.</Prose>
    <H3>When conjugacy stops, keep the probability question</H3>
    <Prose>Suppose a detector reports success with probability .9 for a true adopter and .1 for a nonadopter, with these rates known. If θ is the underlying adoption probability, the reported-success probability is .1+.8θ. Eight reported successes now give likelihood (.1+.8θ)⁸(.9−.8θ)². Multiplying by a Beta prior does not generally produce another Beta density. Updating the shapes as though reports were true outcomes would fit the wrong variable.</Prose>
    <Prose>A small one-dimensional grid can approximate the posterior: evaluate prior×likelihood at cell midpoints, multiply by cell width, normalize the resulting masses, then sum the desired quantity. Work in logs and subtract the largest log weight before exponentiating to avoid unnecessary underflow. Equal cell widths cancel from normalized weights, but they remain necessary for the approximate evidence integral.</Prose>
    <RunnableExample example={examples.nonconjugateGrid} />
    <Prose>Refining 100→1000→10000 cells stabilizes this example near mean .727949 and P(θ&gt;.7)≈.614373. Refinement is a numerical check, not an automatic error certificate for arbitrary densities: narrow peaks, discontinuities, endpoint singularities, nonuniform cells and truncated support need care. A grid over d dimensions grows exponentially in d. Later MCMC and Variational Inference lessons develop other computational approaches while keeping the distinction between target posterior, numerical approximation and model adequacy.</Prose>
    <Prose>The four conjugate pairs here share a principle: the likelihood changes coefficients or exponents already represented by the prior. Exponential Families & Sufficient Statistics develops the general theory later. Hierarchical models can learn how related groups share information instead of assigning unrelated fixed prior strengths; their dependency structure and inference deserve their own treatment. Convenience is a reason to consider a prior family, not a sufficient reason to impose it.</Prose>

    <H2>10. Practise the complete inference loop</H2>
    <Prose>Work these with the corresponding examples closed. State the observation process, parameter support and units before calculating. A useful solution contains both numbers and the interpretation that makes those numbers relevant.</Prose>
    <Practice question="1 · Change the prior, then add a fresh batch" hint="Update two shape parameters separately. The second batch must contain new observations." prompt={<Prose>Start with Beta(3,2), observe two successes/four failures, then three new successes/one failure. Derive both sequential posteriors and the final mean, interior mode and next-success probability.</Prose>}>
      
      <Prose>First Beta(5,6), then Beta(8,7). Pooling the fresh data adds five successes/five failures to the original prior and agrees. Final mean and prediction are 8/15; mode is 7/13. Multiplying the first batch again would double-count it.</Prose>
    </Practice>
    <Practice question="2 · Predict two visitors without discarding uncertainty" hint="Use a Beta(2,2) posterior and integrate θ² for two successes." prompt={<Prose>Before looking at the solution, find P(K=0), P(K=1), P(K=2) for two future visitors under Beta(2,2), then compare a Binomial(2,.5).</Prose>}>
      
      <Prose>The integrated masses are 3/10,2/5,3/10; plug-in masses are 1/4,1/2,1/4. Both have mean 1, but integrated variance 3/5 exceeds 1/2. The two future outcomes have covariance Var(θ)=1/20 and correlation 1/5. One uncertain rate is shared.</Prose>
    </Practice>
    <Practice question="3 · Keep event counts and exposure together" hint="Zero observed events still contribute observation time." prompt={<Prose>Use Gamma(3,2 hours), then observe no events in half an hour and four in 1.5 hours. Find the posterior, expected count in the next two hours, predictive variance and probability of no events.</Prose>}>
      
      <Prose>Posterior Gamma(7,4 hours). Over h=2 hours, mean is 7/2 and variance 21/4; the zero-count probability is (4/6)⁷=(2/3)⁷≈.058528. Ignoring the empty half hour incorrectly changes the denominator and exaggerates the rate. Converting all time quantities to minutes must preserve these count probabilities.</Prose>
    </Practice>
    <Practice question="4 · A new label is not an unseen declared label" hint="List the complete category universe before using a Dirichlet vector." prompt={<Prose>With prior (2,1,1) and counts (1,2,0), find the posterior means. What happens if the next observation uses a fourth label never included in the model?</Prose>}>
      
      <Prose>The posterior is Dirichlet(3,3,1), giving means 3/7,3/7,1/7. Category 3 remains possible although unseen. A fourth label is outside this model's declared sample space; do not silently insert a zero prior shape or renormalize after hiding the observation. Reconsider the taxonomy, observation mapping or an explicit model supporting additional categories.</Prose>
    </Practice>
    <Practice question="5 · Separate a mean interval from an outcome interval" hint="Prior precision is one fourth; data precision is two." prompt={<Prose>A Normal(0,4) mean prior meets two independent measurements 1 and 3 with known observation variance 1. Find posterior mean/variance and the variance of one new reading.</Prose>}>
      
      <Prose>Data mean 2 and total precision 1/4+2=9/4 give posterior variance 4/9 and mean 16/9. The new reading's variance is 13/9. Normal 95% intervals are 16/9 ±1.959964×2/3 for μ and 16/9 ±1.959964×√13/3 for a new reading. Treating the observations as correlated would require a different precision calculation.</Prose>
    </Practice>
    <Practice question="6 · Diagnose a falsely reassuring model check" hint="Which feature does the common-rate likelihood actually use?" prompt={<Prose>An analyst rearranges the eight-success/two-failure sequence, sees the same posterior mean and claims the model has passed an ordering check. Explain the mistake and reproduce the conditional probability of at most two runs.</Prose>}>
      
      <Prose>All such orderings have the same likelihood and posterior by construction; the equal mean checks no temporal pattern. There are choose(10,8)=45 orderings; only all-successes-then-failures and the reverse have at most two runs, so the conditional probability is 2/45. A posterior predictive reference allows other totals and gives another number. Neither reference should be silently substituted for the other.</Prose>
    </Practice>
    <Practice question="7 · Make the decision rule explicit" hint="Compare posterior expected losses, not just whether a point estimate exceeds the threshold." prompt={<Prose>Suppose P(θ&gt;.7|D)=.8. An incorrect launch costs 3 and a missed beneficial launch costs 1. Compare actions, then change the incorrect-launch cost to 9. Does inference need to change?</Prose>}>
      
      <Prose>With cost 3, launch risk is 3×.2=.6 and waiting risk is .8, so launch. With cost 9, launch risk becomes 1.8 and waiting remains .8, so wait. The posterior is unchanged; the preferred action follows the loss. Equality at a threshold gives tied risks unless another rule breaks the tie.</Prose>
    </Practice>
    <Practice question="8 · Rebuild and test the numerical update" hint="Retain cell widths for evidence; recognize improper normalization and prior-excluded support." prompt={<Prose>Modify the grid program to use four reported successes/six failures. Predict the direction of the posterior change. Compare two grid resolutions and check normalization, bounds, a constant-likelihood case and the true-outcome limit with perfect sensitivity/specificity.</Prose>}>
      
      <Prose>The evidence shifts toward lower θ than the 8/10 example. At 10,000 cells, the mean is about .420543, P(θ&gt;.7) about .036292, and the ordered-sequence evidence about .000681820. These agree with a separate continuous integral. When sensitivity equals the false-positive rate, reports carry no information about θ and the posterior stays at the prior. Required checks: weights sum to one, mean lies in [0,1], threshold probability lies in [0,1], and refinement stabilizes the reported quantities. With no observations the posterior equals the chosen Beta prior; with perfect reporting the grid approaches Beta(6,8), mean 3/7. An improper prior with zero shape values cannot be rescued by calling an unnormalized finite grid “the exact posterior.” Record approximation limits rather than force an answer.</Prose>
    </Practice>
    <Prose>You are ready to continue when you can build a posterior from its likelihood and prior, distinguish a parameter interval from an outcome prediction, carry shared uncertainty into a batch, use exposure/precision correctly and diagnose a model assumption a convenient update cannot check. The next module topic is <a href="/learn/path/full-curriculum/concentration-inequalities-hoeffding-bernstein-chernoff?module=math-foundations">Concentration Inequalities (Hoeffding, Bernstein, Chernoff)</a>: finite-sample deviation guarantees under explicit assumptions. MCMC then develops numerical expectations when the posterior cannot be handled by these small conjugate calculations.</Prose>
    <Sources alternatives={<ul>
      <li><a href="https://stat110.hsites.harvard.edu/youtube">Harvard Stat 110 official video index</a> and its <a href="https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo">YouTube playlist</a> — lectures 23–25 develop Beta/Gamma intuition; lecture 27 develops conditional expectation and variance. Official index/playlist metadata were checked; full recordings were not watched.</li>
      <li><a href="https://seeing-theory.brown.edu/bayesian-inference/index.html">Brown University: Seeing Theory, Bayesian Inference</a> — another visual route through conditional evidence, likelihood and sequential Beta updates. Relevant written sections and described controls were inspected; this lesson does not claim to have independently verified that site's implementation.</li>
      <li><a href="https://www.maths.dur.ac.uk/stats/stats2/practical_1-4.html">Durham: Bayesian analysis of Poisson data</a> — a separate R-based exercise using historical volcano counts. Use it to practise Gamma-Poisson modeling and prior comparisons, while questioning the event-rate assumptions. It is historical data, not current volcano monitoring.</li>
    </ul>}>
      <li><a href="https://statproofbook.github.io/P/bin-post.html">The Book of Statistical Proofs: Binomial posterior</a> — the normalized Beta update derivation. Our exact program verifies the numerical examples separately.</li>
      <li><a href="https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf">Kevin Murphy: Conjugate Bayesian analysis of the Gaussian distribution</a> — sections 2 and 6 are useful deeper derivations for known and unknown variance. Variance, precision, inverse-Gamma and Student-t scale conventions differ across texts; use the explicit definitions in this lesson. Relevant sections were read, not every equation in the report.</li>
      <li><a href="https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html">Stan User's Guide: posterior and prior predictive checks</a> — model replication, choice of diagnostic statistics and the non-classical meaning of predictive tail probabilities; version 2.39 was inspected.</li>
      <li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html">SciPy Beta</a> and <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html">Student t</a> — numerical distributions and parameter conventions. The optional native t example uses SciPy; browser computations do not load it.</li>
    </Sources>
  </div>
};
