import { Code, H2, H3, Prose } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { ArrivalCountWaitLab, BayesPopulationLab, DensityAreaLab, EventConditionLab, MomentTailFigure, NormalUnitsFigure, ReusedEvidenceLab, UrnCountLab } from '../../components/lesson-labs/ProbabilityDistributionsLabs.jsx';
import { probabilityDistributionExamples } from '../probability-distributions-examples.js';
export default {
  title: "Probability Distributions & Bayes' Theorem",
  readTime: '~60 min read + 2–3 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot probability-distributions-lesson">
    <LessonIntro exampleKind="Python standard library" prerequisites="Fractions, sums, powers and reading simple Python. We introduce events, random variables, conditioning and moments here. Basic integration helps with the continuous branch; rectangle areas establish its main idea first." sections={[['1-decide-what-can-happen', 'Outcomes and conditioning'], ['2-reverse-the-question-with-bayes', 'Bayes and base rates'], ['3-check-whether-evidence-is-new', 'Repeated evidence'], ['4-build-count-distributions-from-the-experiment', 'Counts and sampling'], ['5-summarize-without-losing-the-distribution', 'Mean, variance and tails'], ['6-read-area-and-cumulative-probability', 'Density, units and point masses'], ['7-connect-counts-waits-and-continuous-models', 'Poisson, exponential and normal'], ['8-update-from-a-continuous-observation', 'Continuous evidence and modelling'], ['9-practise-on-changed-experiments', 'Independent practice']]}>A detector flags a case. It correctly flags 95% of cases with a particular condition. How confident should you be that this flagged case has it? The answer needs more than the detector's success rate. Build the probability model, keep track of which population each number describes, and learn to ask the same careful questions about counts, timings and measurements.</LessonIntro>

    <H2>1. Decide what can happen</H2>
    <Prose><strong>Probability assigns weights to possibilities under a stated model.</strong> The weights let us calculate how much of the model supports an event. They may describe a repeatable random mechanism or uncertainty about an unknown case. A coherent calculation does not, by itself, establish that the model matches the world.</Prose>
    <Prose>Begin with one fair six-sided die. An <strong>outcome</strong> is the actual face, and the <strong>sample space</strong> Ω={'{1,2,3,4,5,6}'} lists the possibilities. An <strong>event</strong> is a set of outcomes answering a question. Let A be “the face is even,” so A={'{2,4,6}'}. Fairness gives each face probability 1/6; adding the three relevant weights gives P(A)=3/6=1/2.</Prose>
    <Prose>Equal labels do not imply equal probabilities. A loaded die needs six stated weights. Two independent fair dice have 36 equally likely <em>ordered pairs</em>, but their sums are not equally likely: sum 2 has one pair, while sum 7 has six. Choose elementary outcomes before counting them.</Prose>
    <Prose>Probabilities are nonnegative; the whole sample space has probability 1; and disjoint events add. <strong>Disjoint</strong> means they cannot occur together. For a finite model these rules say the outcome weights sum to 1. For countably many disjoint events, the sum rule extends to the infinite series. The complement Aᶜ means “not A,” so P(Aᶜ)=1−P(A).</Prose>
    <H3>“And,” “or,” and “given” do different work</H3>
    <Prose>Let B mean “the face is at least 4,” or {'{4,5,6}'}. The <strong>intersection</strong> A∩B contains {'{4,6}'}: both events happen. The <strong>union</strong> A∪B contains {'{2,4,5,6}'}: at least one happens. Adding P(A)+P(B) counts the intersection twice, so subtract it once:</Prose>
    <MathBlock>{String.raw`\begin{aligned}P(A\cup B)&=P(A)+P(B)\\&\quad-P(A\cap B).\end{aligned}`}</MathBlock>
    <Prose>Here that is 1/2+1/2−1/3=2/3. Now suppose someone tells you B occurred. Face 2 is no longer possible, even though it belongs to A. Among the three retained faces, two are even. The <strong>conditional probability</strong> P(A|B), read “probability of A given B,” is 2/3.</Prose>
    <MathBlock>{String.raw`\begin{gathered}P(A\mid B)=\frac{P(A\cap B)}{P(B)},\\P(B)>0.\end{gathered}`}</MathBlock>
    <Prose>The numerator is the original mass satisfying both statements. The denominator is all the mass still possible after conditioning. Dividing renormalizes that retained mass to 1. Counting retained faces works here because the faces are equally weighted; with unequal weights, use their probability sums.</Prose>
    <EventConditionLab />
    <Prose><strong>Independence</strong> means learning one event does not change the other's probability. The definition P(A∩B)=P(A)P(B) also handles zero-probability events without division. When P(B)&gt;0 it is equivalent to P(A|B)=P(A). Our original A and B are not independent: 1/3≠1/4. They are not disjoint either.</Prose>
    <Prose>Two disjoint events with positive probabilities cannot be independent: their joint probability is zero, but the product of their probabilities is positive. “Independent” does not mean “different,” “unrelated-looking,” or “unable to occur together.” It is a precise property of their joint model.</Prose>
    <RunnableExample example={probabilityDistributionExamples.eventRules} />
    <Prose>This exact-fraction program verifies the set operations. The two conditional probabilities happen to agree in this particular example because P(A)=P(B). Do not generalize that coincidence: Bayes' theorem will show how the two directions normally differ. A condition with probability zero has no elementary ratio here; returning zero would invent an answer.</Prose>
    <Checkpoint prompt="For the fair die, A={6} and B={4,5,6}. Calculate P(A|B) and P(B|A).">
      <Prose>P(A|B)=1/3 because one of B's three faces is 6. P(B|A)=1 because observing 6 guarantees B. Both use the same joint event {'{6}'}, but divide its mass 1/6 by different denominators: 1/2 and 1/6.</Prose>
    </Checkpoint>

    <H2>2. Reverse the question with Bayes</H2>
    <Prose>Return to the detector. H means “the case has the condition”; + means “the detector flags it.” Use an imagined population with P(H)=.01, P(+|H)=.95 and P(−|not H)=.90. These are synthetic teaching values, not measurements of a real diagnostic test. The last number is <strong>specificity</strong>; its complement P(+|not H)=.10 is the false-positive rate. The .95 is <strong>sensitivity</strong>.</Prose>
    <Prose>P(+|H) asks about flagged cases <em>inside H</em>. P(H|+) asks about H cases <em>inside the flagged group</em>. To reverse direction, first reconstruct both ways a positive flag can arise. In 100,000 expected cases, 1,000 are H; 950 of those are flagged. The other 99,000 are not H; 9,900 of those are flagged. Thus the positive group contains 10,850 cases, only 950 of which are H.</Prose>
    <MathBlock>{String.raw`\begin{aligned}P(H\mid +)&=\frac{950}{950+9900}\\&=\frac{19}{217}\approx0.087558.\end{aligned}`}</MathBlock>
    <Prose>The flag raises H's probability from 1% to about 8.76%. It is evidence for H, but it does not make H more likely than not. A large ordinary population can contribute many false positives even when its false-positive <em>rate</em> is modest.</Prose>
    <BayesPopulationLab />
    <H3>Derive the rule from the same joint mass</H3>
    <Prose>The product rule is just the conditional-probability formula rearranged: P(H∩E)=P(E|H)P(H). Writing the joint mass in the other order gives P(H|E)P(E). Equate the two and divide by P(E)&gt;0. For the denominator, split the population into H and not H; the two contributions are disjoint and exhaustive.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
P(E)&=P(E\mid H)P(H)\\
&\quad+P(E\mid H^c)P(H^c),\\[2pt]
P(H\mid E)&=\frac{P(E\mid H)P(H)}{P(E)}.
\end{aligned}`}</MathBlock>
    <Prose>This is <strong>Bayes' theorem</strong>. The <strong>prior</strong> P(H) describes uncertainty before this evidence. The <strong>likelihood</strong> P(E|H) tells how compatible the evidence is with H. The <strong>posterior</strong> P(H|E) describes uncertainty after conditioning on it. The denominator is the evidence's total probability, which normalizes all explanations. For several mutually exclusive, exhaustive hypotheses Hᵢ, sum P(E|Hᵢ)P(Hᵢ) over them instead.</Prose>
    <Prose>A negative result also carries information. Its H mass is .01×.05=.0005 and its not-H mass is .99×.90=.891. Therefore P(H|−)=.0005/.8915=1/1783≈.000561, about .0561%. Always identify which result was actually observed.</Prose>
    <RunnableExample example={probabilityDistributionExamples.baseRates} />
    <Prose>The second fixture preserves the same detector and changes the prior to .2. Now 19,000 H cases and 8,000 not-H cases are flagged, so the posterior is 19/27≈.703704. A different population changes the result's interpretation even with identical sensitivity and specificity. This calculation assumes those conditional rates really transfer; changing the population can also change the detector's behavior.</Prose>
    <Prose>As a consistency check, average the two posteriors using how often their results occur: P(H|+)P(+)+P(H|−)P(−)=P(H). This is the same partition identity again. It does not say each observation leaves your belief unchanged; different observations move it in different directions.</Prose>
    <Checkpoint prompt="If both classes assign probability zero to a positive flag, what is P(H|+) when a positive arrives?">
      <Prose>The model assigns the observed event probability zero, so the elementary Bayes ratio is 0/0 and undefined. Inspect the model, data or event definition. A program should expose this contradiction rather than report a confident posterior of zero or one.</Prose>
    </Checkpoint>

    <H2>3. Check whether evidence is new</H2>
    <Prose>Suppose two monitors report a positive flag. If they use independent observations <em>conditional on the actual class</em>, the H likelihood is .95²=.9025 and the not-H likelihood is .10²=.01. With prior .01, Bayes gives .009025/(.009025+.0099)≈.476882. That is stronger evidence than one flag. But if one monitor merely forwards the other's message, the second message adds no new information.</Prose>
    <Prose>The always-valid chain rule uses the evidence you already know. Let E₁ and E₂ be the two observations:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
P(E_1,E_2\mid H)\\
=P(E_1\mid H)\,P(E_2\mid E_1,H).
\end{gathered}`}</MathBlock>
    <Prose>Replacing the second factor by P(E₂|H) requires conditional independence. You need the corresponding assumption within not H too. For an exact copy, P(E₂=+|E₁=+,H)=1, so the joint positive likelihood is .95, not .95². The same holds with .10 under not H.</Prose>
    <ReusedEvidenceLab />
    <Prose>The lab's partial-copy model makes the assumption inspectable. Within either class, let a single positive probability be p. With probability c, draw once and copy; with probability 1−c, draw twice independently. Then P(++)=cp+(1−c)p², P(+-)=P(-+)=(1−c)p(1−p), and P(--)=c(1−p)+(1−c)(1−p)². These four masses sum to 1 and keep each individual positive marginal equal to p.</Prose>
    <RunnableExample example={probabilityDistributionExamples.pairedEvidence} />
    <Prose>The program enumerates the latent copy/fresh branches before combining them. At c=.5, two positives imply a posterior of about .145380. At c=1, the posterior returns to the one-flag result .087558. Mixed-sign pairs have zero probability at that endpoint; they reveal a contradiction rather than a reason to divide by zero.</Prose>
    <H3>Odds expose the evidence multiplier</H3>
    <Prose>For 0&lt;P(H)&lt;1, <strong>odds</strong> compare H mass to not-H mass: O(H)=P(H)/(1−P(H)). Divide Bayes' formula for H by the formula for not H. The common evidence denominator cancels:</Prose>
    <MathBlock>{String.raw`\frac{P(H\mid E)}{P(H^c\mid E)}
=\frac{P(H)}{P(H^c)}
\frac{P(E\mid H)}{P(E\mid H^c)}.`}</MathBlock>
    <Prose>The final ratio is a <strong>likelihood ratio</strong>. One positive multiplies prior odds 1/99 by .95/.10=9.5. Two conditionally independent positives multiply by 9.5 twice. A copied second positive multiplies by 1 after the first result. These finite ratios assume positive denominators; impossible or decisive evidence needs its boundary case handled directly.</Prose>
    <Prose>Even the fresh branch does not make the two reports independent <em>overall</em>. Both are more often positive in H cases, so mixing the two populations creates dependence. Conditional independence is about the specified condition; do not erase that qualifier. This distinction also explains why a Naive Bayes classifier's product of feature likelihoods is a modelling assumption, not a general probability rule.</Prose>

    <H2>4. Build count distributions from the experiment</H2>
    <Prose>A <strong>random variable</strong> assigns a number to each outcome. In a three-draw experiment, X might be the number of marked objects drawn. The underlying outcome includes which objects appeared; several outcomes may produce the same X. The <strong>distribution of X</strong> collects their total probability.</Prose>
    <Prose>For discrete X, its <strong>probability mass function</strong> is p<sub>X</sub>(k)=P(X=k). Every mass is nonnegative and the masses over its <strong>support</strong> sum to 1. Here the discrete support lists values with positive probability. Its <strong>cumulative distribution function</strong>, or CDF, is F<sub>X</sub>(x)=P(X≤x); add all masses at values up to x. A CDF never decreases and tends to 0 as x→−∞ and to 1 as x→+∞. An unbounded distribution need not reach either limit at a finite x.</Prose>
    <H3>One trial, several categories, repeated trials</H3>
    <Prose>A <strong>Bernoulli</strong> variable encodes one yes/no event: X=1 with probability p and X=0 with probability 1−p. The numeric encoding matters for moments, not for the event's name. A <strong>categorical</strong> variable selects one label from several possibilities, such as red/green/blue with weights .2/.5/.3. Its weights sum to 1, but a mean of arbitrary label numbers usually has no meaningful interpretation.</Prose>
    <Prose>For n independent Bernoulli trials with the same success probability p, the number of successes is <strong>binomial</strong>. A particular sequence with k successes has probability pᵏ(1−p)ⁿ⁻ᵏ. There are C(n,k) ways to choose their positions. Here C(n,k), also written “n choose k,” is n!/[k!(n−k)!]; n! multiplies the positive integers up to n, with 0!=1.</Prose>
    <MathBlock>{String.raw`\begin{gathered}P(X=k)=\binom nk p^k(1-p)^{n-k},\\k=0,\ldots,n.\end{gathered}`}</MathBlock>
    <Prose>The formula counts arrangements; it does not make the trials independent. That assumption must come from the experiment or model. If success probabilities vary independently, the count generally is not binomial with their average p. Dependence can change it even more.</Prose>
    <UrnCountLab />
    <H3>Keeping a selected object out changes the next draw</H3>
    <Prose>Take six individually labelled objects, A through F, with A and B marked. Draw three. With replacement and independent remixing, each draw has p=2/6=1/3. There are 6³=216 equally likely ordered selections, and X is binomial. P(X=3)=(1/3)³=1/27: the same marked object can appear more than once.</Prose>
    <Prose>Without replacement, a chosen object cannot appear again. There are C(6,3)=20 equally likely unordered subsets. Each subset has the same number 3! of possible draw orders, so ignoring order is valid for this count. Exactly k marked objects can be chosen in C(2,k) ways, and the other 3−k plain objects in C(4,3−k) ways.</Prose>
    <MathBlock>{String.raw`P(X=k)=\frac{\binom Kk\binom{N-K}{n-k}}{\binom Nn}.`}</MathBlock>
    <Prose>This <strong>hypergeometric</strong> model uses population size N, marked count K and draw count n, sampled uniformly without replacement. Feasible counts satisfy max(0,n−(N−K))≤k≤min(n,K). Here masses at k=0,1,2,3 are 1/5,3/5,1/5,0. Three marked draws are impossible because only two marked objects exist.</Prose>
    <RunnableExample example={probabilityDistributionExamples.urnCounts} />
    <Prose>The code derives masses by enumerating labelled selections and separately checks the formulas. In the no-replacement case P(X≤1)=1/5+3/5=4/5. The PMF chart shows individual masses; the staircase shows their running total. At an integer, the filled CDF value includes that integer's mass. Drawing all six fixes X=2, so its variance becomes zero.</Prose>
    <Checkpoint prompt="A collection has six objects, three marked. Draw two without replacement. Is P(X=2) equal to (1/2)²?">
      <Prose>No. The first marked probability is 3/6, but after a marked draw only two marked objects remain among five. Thus P(X=2)=(3/6)(2/5)=1/5. Equivalently C(3,2)/C(6,2)=3/15. Replacement would give 1/4 under independent remixing.</Prose>
    </Checkpoint>

    <H2>5. Summarize without losing the distribution</H2>
    <Prose>The <strong>expectation</strong> or mean is a probability-weighted average. It need not be a possible outcome. For X in {'{0,1,2}'} with masses .2,.5,.3, the mean is 0(.2)+1(.5)+2(.3)=1.1. It describes the distribution's balance point, not a prediction that the next count is 1.1.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
\mu=\mathbb E[X]&=\sum_x x\,p_X(x),\\
\operatorname{Var}(X)&=\sum_x(x-\mu)^2p_X(x)\\
&=\mathbb E[X^2]-\mu^2.
\end{aligned}`}</MathBlock>
    <Prose>Variance is the expected squared distance from the mean. Expanding (X−μ)² gives E[X²]−2μE[X]+μ²=E[X²]−μ². Our second moment is 1.7, so variance is 1.7−1.1²=.49. The <strong>standard deviation</strong> is its square root, .7. If X is measured in seconds, variance has seconds² units and standard deviation has seconds units. These formulas require the relevant moments to be finite.</Prose>
    <Prose>Expectation is linear: E[aX+b]=aE[X]+b. Variance scales quadratically: Var(aX+b)=a²Var(X), because subtracting the new mean leaves a(X−μ). Adding a constant shifts location without changing spread. For Y=2X+3, mean is 5.2 and variance 1.96.</Prose>
    <MomentTailFigure />
    <RunnableExample example={probabilityDistributionExamples.moments} />
    <Prose>The last two examples have identical mean and variance but different tail events. Distribution A puts half its mass at each of −1 and 1. Distribution B puts masses 1/8,3/4,1/8 at −2,0,2. Both have mean 0 and second moment 1, but P(|X|≥1.5) is 0 versus 1/4. A mean/variance summary cannot replace a risk-relevant tail calculation or identify a distribution family.</Prose>
    <H3>Why a binomial count has mean np</H3>
    <Prose>Write X=I₁+⋯+Iₙ, where Iᵢ is 1 for a success on trial i and 0 otherwise. Each E[Iᵢ]=p, so linearity gives E[X]=np. Linearity does not require independence. Since Iᵢ²=Iᵢ, Var(Iᵢ)=p−p²=p(1−p). For independent trials the cross terms vanish, giving Var(X)=np(1−p).</Prose>
    <Prose>Without replacement, each draw still has marginal marked probability K/N, so the mean is nK/N. But successes make later successes less likely. For N&gt;1, let p=K/N. Two distinct draw indicators have E[IᵢIⱼ]=K(K−1)/[N(N−1)], so their covariance E[IᵢIⱼ]−p² is −p(1−p)/(N−1). Expanding the variance of their sum gives:</Prose>
    <MathBlock>{String.raw`\operatorname{Var}(X)=np(1-p)\frac{N-n}{N-1}.`}</MathBlock>
    <Prose>The factor (N−n)/(N−1) is the finite-population correction. With six objects, two marked and three draws, both count models have mean 1. Replacement variance is 2/3; no-replacement variance is (2/3)(3/5)=2/5. The changing draw relationships explain the different spreads. Full covariance theory has a later lesson; this calculation is the bridge needed here.</Prose>

    <H2>6. Read area and cumulative probability</H2>
    <Prose>A measured delay need not take only integer values. Suppose, as a synthetic model, a delay is uniformly distributed between 0 and .2 seconds. “Uniform” means intervals of equal width within that range have equal probability. To give the whole width .2 total mass 1, its constant <strong>probability density</strong> must be 1/.2=5 per second.</Prose>
    <Prose>Density height is not probability. The interval [.05,.15] seconds has width .10, so its probability is 5×.10=.5. Density has reciprocal units: “per second” times “seconds” leaves a dimensionless probability. In milliseconds the same range is [0,200], its density is .005 per millisecond, and [.05,.15] seconds becomes [50,150] milliseconds. The area is still .005×100=.5.</Prose>
    <MathBlock>{String.raw`P(a\le X\le b)=\int_a^b f_X(x)\,dx.`}</MathBlock>
    <Prose>A <strong>PDF</strong> f<sub>X</sub> is a nonnegative density whose integral over the whole line is 1. For a general shape, integration adds thin area slices instead of one rectangle. For a distribution with a PDF, an exact point has zero probability: an interval of zero width has zero area. That does not mean an observed real number is logically impossible; an uncountable set of zero-mass points can carry total probability 1. Recorded measurements also usually stand for finite precision intervals.</Prose>
    <DensityAreaLab />
    <H3>The CDF works for both discrete and continuous models</H3>
    <Prose>F(x)=P(X≤x) is defined regardless of whether a density exists. For the uniform delay, F(x)=0 below 0, F(x)=x/.2 within the interval, and F(x)=1 above .2. Differences in cumulative probability give interval mass. In particular P(a&lt;X≤b)=F(b)−F(a). A closed left endpoint needs its possible point mass too:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
P(a\le X\le b)&=F(b)-F(a^-),\\
P(X=a)&=F(a)-F(a^-).
\end{aligned}`}</MathBlock>
    <Prose>F(a⁻) means the limit just below a, excluding mass at a. Every CDF is right-continuous; jumps are point probabilities. For a density-only model there are no jumps, so the distinction between open and closed endpoints does not change probability. For a discrete or mixed model it can be decisive.</Prose>
    <H3>A cache-style delay model has a point mass</H3>
    <Prose>Imagine a request completes immediately with probability q=.3. Otherwise its extra delay is uniform on (0,.2) seconds. This simplified extra-delay model places a mass .3 at exactly zero and spreads the remaining .7 continuously. The continuous part's density is .7/.2=3.5 per second; its area over the full range is .7, not 1. There is no ordinary finite-height PDF that accounts for the point mass as well.</Prose>
    <Prose>The CDF jumps from 0 to .3 at zero, then rises linearly to 1. For [.05,.15], only the continuous area contributes: 3.5×.10=.35. For [0,.10], add the atom: .3+3.5×.10=.65. For the singleton interval [0,0], the probability is .3 even though its width is zero. Changing time units does not alter this point probability.</Prose>
    <RunnableExample example={probabilityDistributionExamples.mixedDelay} />
    <Prose>This is a <strong>mixed distribution</strong>. “Continuous” is not simply a synonym for “stored as a decimal”: the distribution's mathematical mass structure matters. Conversely, rounding a continuous measurement to milliseconds creates discrete recorded values even if the underlying model has a density.</Prose>
    <Checkpoint prompt="In the q=.3 mixture, compare P(0<X≤.1) with P(0≤X≤.1). What is P(X=.1)?">
      <Prose>The first probability is .35; the second adds the atom at zero and is .65. The point .1 has no atom, so its probability is zero. Only endpoints carrying actual mass change the answer.</Prose>
    </Checkpoint>

    <H2>7. Connect counts, waits and continuous models</H2>
    <H3>Poisson counts and exponential waits describe one process</H3>
    <Prose>A <strong>homogeneous Poisson process</strong> is a model of event arrivals at a constant rate λ. Counts in disjoint time intervals are independent, and the count distribution depends on interval length. For a short interval Δt, one arrival has probability λΔt plus a smaller-order error, while two or more arrivals have probability smaller than order Δt. This excludes scheduled bursts or simultaneous batches as part of this simple model.</Prose>
    <Prose>Write N(t) for the number of arrivals in a window of length t. It has a <strong>Poisson distribution</strong> with dimensionless mean m=λt:</Prose>
    <MathBlock>{String.raw`\begin{gathered}P(N(t)=k)=e^{-\lambda t}\frac{(\lambda t)^k}{k!},\\k=0,1,2,\ldots.\end{gathered}`}</MathBlock>
    <Prose>A useful construction is to split the window into n small slots, use independent Bernoulli arrivals with probability λt/n, and let n grow. For fixed k, the binomial mass tends to the expression above: its combinatorial factor contributes (λt)ᵏ/k! and its no-arrival factor tends to <Code>exp(−λt)</Code>. The Poisson mean and variance are both λt. This limit motivates the model; it does not establish that a particular traffic stream satisfies it.</Prose>
    <Prose>For λ&gt;0, let T be the wait to the first arrival. The event T&gt;t is exactly “no arrival by time t,” so P(T&gt;t)=P(N(t)=0)=<Code>exp(−λt)</Code>. Its complement gives the CDF, and differentiating gives the exponential density. A zero-rate process has no arrivals; the finite exponential-wait formulas below do not apply to it.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
F_T(t)&=1-e^{-\lambda t},\\
f_T(t)&=\lambda e^{-\lambda t}\quad(t\ge0),\\
\mathbb E[T]&=1/\lambda,\\
\operatorname{Var}(T)&=1/\lambda^2.
\end{aligned}`}</MathBlock>
    <Prose>The density and CDF are zero for t&lt;0. λ is a <strong>rate</strong>, such as events per minute; 1/λ is a <strong>scale</strong>, measured in minutes. Some libraries parameterize exponentials by scale, so passing a rate where a scale is expected reverses the intended change. Under this process, successive interarrival waits are independent exponentials with the same rate.</Prose>
    <ArrivalCountWaitLab />
    <Prose>At λ=2 per minute and t=1.5 minutes, mean count is 3, but zero arrivals still has probability e⁻³≈.049787. “At least one” has probability 1−e⁻³≈.950213. The mean first wait is .5 minutes; its median solves 1−e⁻²ᵗ=.5 and is log(2)/2≈.346574 minutes. A skewed distribution's mean and median need not agree.</Prose>
    <Prose>The <strong>p-quantile</strong> is a threshold with cumulative probability p; more generally use the smallest x with F(x)≥p. Solving the exponential CDF gives t<sub>p</sub>=−log(1−p)/λ for 0&lt;p&lt;1. This also gives inverse-CDF sampling: feed a uniform value u in (0,1) into that formula to obtain an exponential wait. The lab uses disclosed fixed u values for reproducibility; it is not claiming those twelve values estimate the distribution.</Prose>
    <RunnableExample example={probabilityDistributionExamples.countsAndWaits} />
    <Prose>The code uses <Code>expm1(z)</Code> for eᶻ−1 and <Code>log1p(z)</Code> for log(1+z), which avoid cancellation for small z. The PMF chart shows only counts 0–24 and reports its omitted tail instead of rescaling the visible bars to sum to 1. It is a calculated distribution, distinct from the single event timeline.</Prose>
    <H3>Memorylessness is a model property, not a rule for every wait</H3>
    <Prose>For s,t≥0, condition on having already waited s minutes. Divide the probability of waiting past s+t by the probability of waiting past s:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
P(T>s+t\mid T>s)
&=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}\\
&=e^{-\lambda t}.
\end{aligned}`}</MathBlock>
    <Prose>Surviving s minutes without an event does not change the distribution of the additional wait. A scheduled bus, an ageing part or a rate that changes with time generally does not have this property.</Prose>
    <Prose>The discrete counterpart is a <strong>geometric</strong> wait for the first success in independent Bernoulli trials with 0&lt;p≤1. If T counts trials <em>including</em> the successful one, P(T=k)=(1−p)ᵏ⁻¹p for k≥1 and E[T]=1/p. At p=1, T=1 with certainty; at p=0, success never arrives and the finite-mean formula does not apply. At p=.25, P(T=3)=.75²(.25)=.140625 and mean wait is four trials. Another convention counts failures before success and starts at zero; always inspect the support before using a library result.</Prose>
    <H3>Normal error uses a location and a scale</H3>
    <Prose>A <strong>normal</strong> or Gaussian model X∼N(μ,σ²), with σ&gt;0, has density:</Prose>
    <MathBlock>{String.raw`\begin{gathered}z=\frac{x-\mu}{\sigma},\\
f_X(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-z^2/2}.\end{gathered}`}</MathBlock>
    <Prose>μ sets the center and σ sets the standard deviation; the support is the whole real line. Subtract μ and divide by σ to form Z=(X−μ)/σ, which has the standard normal distribution N(0,1). An event a≤X≤b becomes (a−μ)/σ≤Z≤(b−μ)/σ. This is a change of coordinates for the same event, not a change in probability.</Prose>
    <Prose>For a synthetic sensor-error model with μ=0 and σ=.2 volts, the interval [−.1,.1] volts becomes [−.5,.5] standard deviations. Its probability is about .382925. The density at zero is about 1.994711 per volt, which is allowed: its integral, not its height, must equal 1.</Prose>
    <NormalUnitsFigure />
    <RunnableExample example={probabilityDistributionExamples.normalAreas} />
    <Prose>The normal CDF has no elementary closed form, so the program evaluates it numerically with Python's standard library. For a very small right tail, subtracting a rounded CDF from 1 can lose accuracy; the complementary error function <Code>erfc</Code> evaluates that tail directly. P(Z&gt;8) is approximately 6.22×10⁻¹⁶, not zero.</Prose>
    <Prose>Gaussian noise is an assumption to check. A central-limit theorem explains why appropriately standardized sums of many independent, identically distributed variables with finite nonzero variance approach normality. It does not say individual measurements, bounded delays, multimodal populations or arbitrary dependent data are Gaussian. Extreme-tail approximations may require much more care than the center of a distribution.</Prose>
    <LessonTable caption="Choose the model from what is sampled and which assumptions hold" headers={['Question', 'Candidate model', 'Check before using it']} rows={[['One yes/no outcome', 'Bernoulli(p)', 'Define what success means.'], ['One of several labels', 'Categorical(weights)', 'Mutually exclusive, exhaustive labels; normalized weights.'], ['Success count in n trials', 'Binomial(n,p)', 'Fixed n, equal p and independent trials.'], ['Marked count in a finite selection', 'Hypergeometric', 'Uniform selection without replacement.'], ['Trial of first success', 'Geometric(p)', 'Independent equal-p trials; specify whether success is counted.'], ['Count in a time window', 'Poisson(λt)', 'For the arrival interpretation: homogeneous Poisson assumptions.'], ['Wait to next arrival', 'Exponential(rate λ)', 'Constant-rate memoryless model; distinguish rate and scale.'], ['Continuous symmetric error', 'Normal(μ,σ²)', 'Assess support, shape, dependence and tails against the application.']]} />

    <H2>8. Update from a continuous observation</H2>
    <Prose>How can Bayes use an exact measured time when a density model assigns every exact time probability zero? Start with an observation interval, where ordinary conditional probability works, then examine what happens as the interval narrows. Do not put two zero point probabilities into the discrete formula.</Prose>
    <Prose>Suppose a machine has two equally likely operating modes, fast and slow. As a synthetic model, its completion time is exponential with rate 3 per minute in fast mode and 1 per minute in slow mode. For a narrow interval [t,t+h], each mode assigns probability e⁻ʳᵗ(1−e⁻ʳʰ). For small h this is approximately f(t)h. Multiply by each prior, normalize, and let h shrink. The common h cancels.</Prose>
    <MathBlock>{String.raw`\begin{gathered}P(H_i\mid T=t)\\=
\frac{P(H_i)f_{T\mid H_i}(t)}
{\sum_j P(H_j)f_{T\mid H_j}(t)}.\end{gathered}`}</MathBlock>
    <Prose>This density expression gives the conditional model at points with positive marginal density, using compatible density conventions. Here the smooth positive densities make the shrinking-interval argument direct. General conditional probability at zero-mass events needs additional measure-theoretic care; a ratio of point probabilities remains undefined.</Prose>
    <Prose>At t=.2 minutes, the two densities are 3e⁻⁰·⁶ and e⁻⁰·². Weighting each by .5 gives a fast-mode posterior of about .667880. At t=2 minutes, the posterior is only .052085: a long wait is much more compatible with the slow mode. This is a useful application of Bayes beyond pass/fail tests.</Prose>
    <RunnableExample example={probabilityDistributionExamples.continuousEvidence} />
    <Prose>For [.2,.3] the exact interval posterior is .646101. As the width decreases to .001 and .00001 minutes, it approaches .667880. Converting to seconds divides both density likelihoods by 60, so that common factor cancels from Bayes' ratio. A density can change its numerical height with units while the posterior remains invariant.</Prose>
    <H3>A calculation, a fitted parameter and a useful prediction are different things</H3>
    <Prose>This lesson starts with specified probability models and computes their consequences. Estimating an unknown p, λ, μ or σ from data is the next task. The <strong>likelihood as a function of a parameter</strong> evaluates how the fixed observations vary in compatibility across parameter choices; it is not automatically a normalized probability distribution over those parameters. Maximum likelihood and MAP will make that distinction concrete next.</Prose>
    <Prose>A classifier may output a score that orders cases usefully without being a trustworthy numerical probability. <strong>Calibration</strong> concerns probabilistic interpretation: among cases assigned probability near .8 for a specified event, its observed frequency should be near .8 in the population being evaluated. Finite samples, bins and population shifts affect that assessment. A model that always predicts the base rate can be calibrated while failing to distinguish individual cases.</Prose>
    <Prose>Use held-out evidence to assess predictions and assumptions. Compare more than a mean: inspect support, tails, subpopulations and dependence. Gaussian convenience does not justify negative probabilities, negative waiting times or a missed second mode. Bayes coherently updates the model you supplied; it cannot repair an omitted explanation, double-counted evidence or incorrect likelihood automatically.</Prose>
    <Checkpoint prompt="Two binary features are independent within each class in a Naive Bayes model. Must they be independent after combining the classes?">
      <Prose>No. The earlier fresh-detector model is a counterexample: both features are more likely positive within H, so the hidden class couples them marginally. Multiplication is justified inside the stated class-conditional model, not after dropping the condition.</Prose>
    </Checkpoint>

    <H2>9. Practise on changed experiments</H2>
    <Prose>Write the event, sample space or density model before calculating. For each answer, check its units, support and denominator, then explain one assumption that would change it. These programs use only Python's standard library and were executed with Python 3.12; copy each complete example into a file and run it with <Code>python filename.py</Code>. Small final-digit differences can occur in floating-point functions.</Prose>
    <H3>1. A conditional count with overlapping events</H3>
    <Prose>Roll two independent fair six-sided dice. A means the sum is 7. B means the first die is even. Calculate P(A), P(B), P(A∩B), P(A|B), and decide whether A and B are independent. Then replace B with “at least one die is 6.”</Prose>
    <details><summary>Hint: use ordered pairs, not equally weighted sums</summary><Prose>There are 36 pairs. List A's six pairs, then count which also satisfy each B.</Prose></details>
    <details><summary>Worked solution and changed-condition check</summary><Prose>P(A)=6/36=1/6. For an even first die, P(B)=18/36=1/2 and the intersection is (2,5),(4,3),(6,1), giving 3/36=1/12. Thus P(A|B)=1/6=P(A), and the events are independent. For “at least one 6,” B has 6+6−1=11 pairs, while the intersection is (1,6),(6,1). Now P(A|B)=2/11, so they are not independent. Acceptance check: enumerate all 36 pairs and verify both results.</Prose></details>
    <H3>2. A changed detector, both outcomes and an odds threshold</H3>
    <Prose>Use prior P(H)=.04, sensitivity .9 and false-positive rate .05. Find P(H|+) and P(H|−). Keeping those detector rates fixed, find the prior above which a positive makes H more likely than not.</Prose>
    <details><summary>Hint: build four joint masses first</summary><Prose>Positive masses are .04×.9 and .96×.05. For the threshold solve .9p&gt;.05(1−p), rather than trying arbitrary priors.</Prose></details>
    <details><summary>Worked solution and boundary</summary><Prose>The joint masses are .036,.004,.048,.912. P(H|+)=.036/.084=3/7≈.428571; P(H|−)=.004/.916=1/229≈.004367. A positive gives posterior above .5 exactly when .95p&gt;.05, or p&gt;1/19≈.052632. Equality gives .5. The likelihood ratio is 18, so the odds check is 18p/(1−p)&gt;1.</Prose></details>
    <H3>3. Recognize copied evidence</H3>
    <Prose>A prior is .1, and one signal has positive probabilities .8 within H and .2 within not H. Compute the posterior after one positive, two conditionally independent positives and an exactly copied second positive. Explain which product changes for the copy.</Prose>
    <details><summary>Hint: compare joint likelihoods .8²/.2² with .8/.2</summary><Prose>Do not use the posterior as though it were a likelihood. Either normalize prior-weighted pair likelihoods or update odds with genuinely new evidence.</Prose></details>
    <details><summary>Worked solution and contradiction case</summary><Prose>One positive gives .08/(.08+.18)=4/13≈.307692. Two fresh positives give .064/(.064+.036)=.64. A copied positive stays at 4/13 because its second conditional likelihood is 1 in either class after the first positive. A +− pair is impossible under an exact-copy-only model; report the contradiction instead of a posterior.</Prose></details>
    <H3>4. Derive a count distribution and its uncertainty</H3>
    <Prose>Six objects contain three marked ones. Draw two. Give the complete PMF, mean, variance and P(X≥1), first without replacement and then with independent replacement. Predict what happens if all six are drawn without replacement.</Prose>
    <details><summary>Hint: the no-replacement sample space has 15 subsets</summary><Prose>For k=0,1,2, count C(3,k)C(3,2−k). With replacement use a binomial with n=2 and p=.5.</Prose></details>
    <details><summary>Worked solution and full-population check</summary><Prose>Without replacement, masses are 3/15,9/15,3/15, or .2,.6,.2. Mean is 1, variance .4, and P(X≥1)=.8. With replacement, masses are .25,.5,.25; mean remains 1, variance is .5, and P(X≥1)=.75. Drawing all six without replacement makes X=3 with probability 1 and variance 0. Enumeration of subsets/sequences should reproduce these values.</Prose></details>
    <H3>5. A mixed delay and a unit conversion</H3>
    <Prose>A delay is zero with probability .2 and otherwise uniform between 0 and .4 seconds. Find its continuous density, F(.1), P(.1≤X≤.3), P(X=0), mean and variance. Re-express the interval probability in milliseconds.</Prose>
    <details><summary>Hint: the continuous part has total mass .8</summary><Prose>Its density is .8/.4. Conditional on a nonzero delay, the first and second moments are .4/2 and .4²/3.</Prose></details>
    <details><summary>Worked solution and normalization</summary><Prose>Density is 2 per second on the continuous interval, integrating to .8. F(.1)=.2+2(.1)=.4. The positive interval [.1,.3] has area 2(.2)=.4, while P(X=0)=.2. Mean is .8(.2)=.16 seconds. Second moment is .8(.16/3)=.042666…, so variance is .042666…−.0256=.017066… seconds². In milliseconds, density is .002 per millisecond and interval width is 200, giving the same .4 probability. The atom and continuous area sum to 1.</Prose></details>
    <H3>6. A count window is not a schedule</H3>
    <Prose>Assume a homogeneous Poisson process with rate 3 per minute. Find the chance of no arrival in 20 seconds, at least one arrival in that time, and the 90th percentile of the first wait. If there has been no arrival for a minute, what is the chance of waiting another 20 seconds? Explain why the same answer need not hold for scheduled arrivals.</Prose>
    <details><summary>Hint: convert seconds to minutes before multiplying by the rate</summary><Prose>Twenty seconds is 1/3 minute, so λt=1. Use −log(.1)/3 for the 90th percentile in minutes.</Prose></details>
    <details><summary>Worked solution and assumptions</summary><Prose>No arrival has probability e⁻¹≈.367879; at least one has 1−e⁻¹≈.632121. The 90th percentile is log(10)/3≈.767528 minutes, or 46.0517 seconds. Conditional on no arrival for a minute, another 20-second wait still has probability e⁻¹ by exponential memorylessness. A scheduled arrival's remaining wait depends on the schedule and elapsed time, so the homogeneous Poisson model is not justified merely by knowing an average rate.</Prose></details>
    <H3>7. Continuous evidence and a decision boundary</H3>
    <Prose>In the fast/slow timing model, keep equal priors and rates 3 and 1 per minute. At what observed time are the posterior probabilities equal? Which mode is more likely before and after it? Would converting to seconds move the physical boundary?</Prose>
    <details><summary>Hint: compare weighted densities and cancel equal priors</summary><Prose>Solve 3e⁻³ᵗ=e⁻ᵗ, or 3e⁻²ᵗ=1.</Prose></details>
    <details><summary>Worked solution and changed-unit check</summary><Prose>The boundary is log(3)/2≈.549306 minutes, or 32.9584 seconds. Fast is more likely before it and slow after it. The two density heights are equal at the boundary. Both divide by 60 when expressed per second, so the physical boundary stays the same. This is a model-based classification boundary, not proof of which mode generated a particular case.</Prose></details>
    <H3>8. Challenge a useful-looking summary</H3>
    <Prose>A team chooses a Gaussian model because measured errors have mean near zero and variance near one. Another team reports that a classifier is calibrated because it always predicts the observed base rate. Explain what each statement establishes, what it does not, and what you would inspect next.</Prose>
    <details><summary>Hint: revisit the same-moments figure and separate probability reliability from discrimination</summary><Prose>Two moments do not determine tail shape. A calibrated constant score cannot rank cases differently.</Prose></details>
    <details><summary>Explained assessment</summary><Prose>Near-zero mean and unit variance are compatible with many distributions, including our two discrete counterexamples. Inspect the measurement process, support, multimodality, tails and dependence, then assess predictions on held-out observations. A constant score equal to a population's event frequency can be calibrated in that population, but it has no case-level discrimination. Inspect useful ranking/decision performance and calibration on the relevant population, with uncertainty from finite data. Neither property grants the other automatically.</Prose></details>
    <Prose><strong>Ready to move on?</strong> You should be able to build a joint table, reverse a conditional with its denominator, justify or reject a product of likelihoods, choose a count model from a sampling rule, read PMF/PDF/CDF correctly, and change units without changing the event. Next, <strong>Maximum Likelihood &amp; MAP Estimation</strong> asks how data can determine the model's unknown parameters. Later lessons deepen random-variable theory, Bayesian inference and stochastic processes; these connections do not reorder the module's reading sequence.</Prose>
    <Sources alternatives={<ul>
      <li><a href="https://stat110.hsites.harvard.edu/youtube">Harvard Stat 110 video index</a> and its <a href="https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo">officially linked YouTube playlist</a>: use lectures 4–5 for conditioning/Bayes, 7–9 for distributions/expectation, 11–14 for Poisson/normal/CDFs and 16–18 for exponentials and mixed models. The index and playlist metadata were checked; this is not a claim that the recordings were watched end to end.</li>
      <li><a href="https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/lecture-2/">MIT 6.041SC Lecture 2</a>: another route through conditioning and Bayes with slides, video and recitation work. <a href="https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-ii/lecture-8/">Lecture 8</a> develops densities/CDFs, and <a href="https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-iii/lecture-14/">Lecture 14</a> connects Poisson counts with waits. Course pages and relevant written slides were inspected; no full-video viewing is implied.</li>
    </ul>}>
      <li>NIST distribution references: <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm">binomial</a>, <a href="https://itl.nist.gov/div898/handbook/eda/section3/eda366j.htm">Poisson</a>, <a href="https://itl.nist.gov/div898/handbook/eda/section3/eda3667.htm">exponential</a> and <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm">normal</a>. Useful formula/support/moment references after the derivations. Check the exponential page's scale convention against this lesson's rate λ.</li>
      <li>SciPy's official <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html">binom</a> and <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html">hypergeom</a> documentation: numerical PMF/CDF/survival methods, support and parameter conventions. This lesson's complete programs use the standard library; SciPy provides an independent numerical verification route.</li>
    </Sources>
  </div>
};
