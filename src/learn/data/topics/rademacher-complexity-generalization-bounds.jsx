import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  BestResponseLab, BoundedNormLab, MarginBoundLab, SignedGeometryLab,
} from '../../components/lesson-labs/RademacherLabs.jsx';
import {
  BestResponseMatrixFigure, BoundComponentsFigure, ConvexHullFigure, DataRolesFigure, DirectionVersusEnergyFigure,
  GhostSampleFigure, KernelGramFigure, L1VersusL2Figure, LossSlopesFigure, MarginDistributionFigure,
  MonteCarloFigure, NestedClassesFigure, PredictorVersusLossFigure, RampFigure, SupportingPointFigure,
  TheoremTermsFigure, ThresholdRestrictionFigure,
} from '../../components/lesson-labs/RademacherFigures.jsx';
import { rademacherExcerpts, rademacherPrograms } from '../rademacher-examples.js';
import {
  duplicateFeatureGroups, experimentSettings, fittedModels, majorityBaseline, provenance, recordedFixtures,
  recordedMonteCarlo, representation, selection,
} from '../rademacher-data.js';
import {
  absoluteComplexity, empiricalComplexity, kernelComplexity, linearComplexity, massartSignBound, mistakeRows,
  rampLoss, sauerCount, signPatterns, thresholdRows,
} from '../rademacher-models.js';

/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.sqrt` here would resolve to that component and silently
   yield `undefined` rather than a number. Nothing below reaches for the global
   `Math`; every derived value comes from `rademacher-models.js`. */
/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');
/** A rounded decimal, at the precision the surrounding sentence claims. */
const to = (value, digits) => String(Number(value.toFixed(digits))).replace('-', '−');

const THREE = [-1, 0, 1];
const thresholdModel = empiricalComplexity(thresholdRows(THREE));
const constantModel = empiricalComplexity([[-1, -1, -1], [1, 1, 1]]);
const singletonModel = empiricalComplexity([[1, 1, 1]]);
const bothOrientations = thresholdRows(THREE, { bothOrientations: true });
const bothOrientationsModel = empiricalComplexity(bothOrientations);
const cubeModel = empiricalComplexity(signPatterns(3));
const lossModel = empiricalComplexity(mistakeRows(thresholdRows(THREE), [1, -1, 1]));
const absoluteSingleton = absoluteComplexity([[1, 1, 1]]);

const parallel = linearComplexity([[1, 0], [1, 0]], 1);
const perpendicular = linearComplexity([[1, 0], [0, 1]], 1);
const oneSample = linearComplexity([[0]], 1);
const twoSample = linearComplexity([[0], [1]], 1);
const practiceGeometry = linearComplexity([[3, 0], [0, 4]], 2);
const kernelAtZero = kernelComplexity([[1, 0], [0, 1]], 1);
const kernelAtNine = kernelComplexity([[1, 0.9], [0.9, 1]], 1);
const kernelAtOne = kernelComplexity([[1, 1], [1, 1]], 1);
const practiceThresholds = empiricalComplexity(thresholdRows([2, 5]));
const practiceThresholdsPlus = empiricalComplexity([...thresholdRows([2, 5]), [1, -1]]);
const sauerExample = sauerCount(3, 100);

const defaultRamp = rampLoss(recordedFixtures.scalarMargins, 0.5);
const practiceRamp = rampLoss([-0.1, 0.2, 0.8], 0.4);
const selected = fittedModels.find(model => model.radius === selection.chosenRadius);
const workedModel = fittedModels.find(model => model.radius === 2);
const workedBound = workedModel.bounds[1];
const smallestAtOne = fittedModels.reduce((best, model) =>
  (model.bounds[1].rawUpper < best.bounds[1].rawUpper ? model : best));

const headings = [
  '1. A three-input noise-matching game',
  '2. The definition, and the three things held fixed',
  '3. Turn a noise calculation into a risk statement',
  '4. Why random signs appear in the proof',
  '5. Compute or bound complexity using geometry',
  '6. Connect score geometry to losses and margins',
  '7. Estimate without turning a lower estimate into an upper guarantee',
  '8. A real-data experiment: what a norm budget buys and costs',
  '9. Deeper: ensembles, covering numbers, neural bounds and the alternatives',
  '10. Practice with changed problems',
  '11. Other ways to learn this, and the next connection',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Excerpt({ excerpt, children }) {
  return <section className="rad-excerpt">
    <H3>{excerpt.title}</H3>
    <CodeBlock language={excerpt.language}>{excerpt.code}</CodeBlock>
    <Prose>Lifted out of <Code>{excerpt.file}</Code> by parsing it, so this is the code that actually
      ran. {children}</Prose>
  </section>;
}

function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="rad-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const rademacherContent = {
  title: 'Rademacher Complexity & Generalization Bounds',
  readTime: '~55 min first pass · ~110 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot rad-lesson">
    <LessonIntro
      prerequisites={<>Dot products, averages, and the idea of an expectation. <Math>{'\\sup'}</Math> means the largest
        achievable value — for the finite tables here it is an ordinary maximum, and the lesson says so again where it
        first matters. The preceding <a href="/learn/path/full-curriculum/pac-learning-vc-dimension?module=classical-ml">PAC
        Learning &amp; VC Dimension</a> lesson supplies empirical risk, population risk and what a guarantee over
        repeated samples means; those are refreshed here rather than assumed.
        The <a href="/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml">Calibration
        &amp; Conformal Prediction</a> lesson asked what a probability promises. This one asks what choosing a
        predictor from a family costs statistically.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      You tried several classifiers and kept the one that made the fewest training mistakes. Some of that improvement
      may be real structure. Some may come from having enough choices to accommodate accidental details of the sample.
      You will put a number on that second possibility: replace the labels with coin flips, ask how well the best
      allowed rule can match them, and get exactly {num(thresholdModel.complexity)} for a threshold class on three
      points. Then you will turn that number into a risk bound, derive it geometrically for a norm ball, and run it on
      480 rows of real banknote measurements — where the classifier gets
      to {selected.assessment.errors} mistakes out of 80 while every bound expression on the page stays
      above {to(smallestAtOne.bounds[1].rawUpper, 2)} and therefore says nothing at all. Both facts are true, and the
      lesson is about why. Every investigation updates its topic-specific results from valid control changes, with no expected-answer input.
    </LessonIntro>

    <div className="rad-route"><Prose><strong>First pass.</strong> Read sections 1–8 and do practice 1–7. That route
      gets you the exact calculation, the theorem it feeds, the geometry that makes it computable, the margin
      construction that makes a hard classification analysable, and one real experiment you can judge. Run both
      programs on the way: the first in section 7, the second in section 8. Section 9 is a deeper branch — convex
      hulls, covering numbers and chaining, neural norm bounds, PAC-Bayes and stability — and is best read once the
      small sums feel natural.</Prose></div>

    <Prose>Try a deliberately impossible prediction task: keep the inputs, replace the answers with independent coin
      flips, and ask how well the best allowed rule can match them. Repeat with new flips. A class with many ways to
      adapt can obtain a large average match even though there is no signal to
      discover. <strong>Rademacher complexity measures this freedom to correlate with random signs.</strong> It helps
      bound the difference between performance on a training sample and performance on the population that generated
      it.</Prose>

    <Callout title="Which numbers here are measurements, and what a bound actually promises">
      <p>The threshold tables, vectors, kernels and margins in sections 1–7 are <strong>constructed examples</strong>,
      chosen so every value can be checked by hand. The banknote experiment in section 8 is the one place real
      observations appear, and it is attributed there.</p>
      <p>Three quantities are kept apart throughout, and mixing them is the main way a generalisation claim gets
      overstated. The <strong>empirical</strong> complexity is an exact average over sign patterns for one fixed
      sample. The <strong>expected</strong> complexity averages that over fresh samples too, and is generally unknown.
      A <strong>bound</strong> built from either holds with probability at least <Math>{'1-\\delta'}</Math> over the
      draw of the sample, simultaneously for every member of the class — it is never a deterministic promise about the
      dataset in front of you, and <Math>{'1-\\delta'}</Math> is not a confidence attached to any individual
      prediction. This caution is stated once here; the rest of the lesson refers back to it rather than repeating
      it.</p>
    </Callout>

    {/* ============================================================ §1 */}
    <H2 id={headingId(headings[0])}>{headings[0]}</H2>
    <Prose>Put three inputs in increasing order: <Math>{'-1, 0, 1'}</Math>. A <strong>positive threshold rule</strong>
      {' '}predicts <Math>{'-1'}</Math> below a cutoff and <Math>{'+1'}</Math> at or above it. Whatever real cutoff we
      choose, its predictions on these three inputs must be one of four rows.</Prose>

    <ThresholdRestrictionFigure />

    <Prose>The infinitely many cutoffs have only four distinct effects here. We must include both constant outcomes;
      restricting cutoffs to observed values would miss the all-negative rule.</Prose>

    <Prose>Now flip three fair coins. Represent their results by signs <Math>{'\\sigma=(+1,-1,+1)'}</Math>. For a
      prediction row <Math>{'h'}</Math>, calculate</Prose>
    {/* The three-term sum cannot wrap inside a \frac, so the division is
        written as a leading 1/3 over a bracketed sum that can. Same
        expression; it fits a 320px column. */}
    <MathBlock>{'\\begin{gathered}\\operatorname{match}(h,\\sigma)=\\\\[2pt]\\tfrac13\\bigl[\\sigma_1h(x_1)+\\sigma_2h(x_2)\\\\+\\;\\sigma_3h(x_3)\\bigr].\\end{gathered}'}</MathBlock>
    <Prose>Each agreement contributes <Math>{'+1'}</Math> and each disagreement contributes <Math>{'-1'}</Math>. The
      result is twice the fraction of agreements, minus one. For this sign pattern the best threshold agrees twice and
      disagrees once, giving <Math>{'1/3'}</Math>. No threshold can produce the alternating
      row <Math>{'(+1,-1,+1)'}</Math>.</Prose>
    <Prose>There are eight equally likely sign patterns. For the four patterns that already <em>are</em> threshold
      rows, the best match is 1. For the other four, it is <Math>{'1/3'}</Math>. The average best match is
      therefore</Prose>
    <MathBlock>{'\\frac{4(1)+4(1/3)}8=\\frac23.'}</MathBlock>
    <Prose>That is the empirical Rademacher complexity of this threshold class on these three inputs, using the
      convention developed below.</Prose>

    <Prose><strong>The order of operations matters.</strong> For each new coin-flip pattern, choose its best rule; then
      average those best scores. If you fix one rule first and average its correlation over fair signs, the answer
      is {num(thresholdModel.maxAfterAveraging)}. Averaging before taking the maximum erases the freedom to adapt that
      we are trying to measure.</Prose>

    <BestResponseMatrixFigure />

    <BestResponseLab />

    <H3>Compare classes without inventing a universal ranking</H3>
    <Prose>On these same three distinct inputs, exact enumeration gives five nested answers.</Prose>

    <NestedClassesFigure />

    <Prose>Reading the figure outward: the fixed all-positive rule alone gives {num(singletonModel.complexity)}, either
      constant sign gives {num(constantModel.complexity)}, positive thresholds
      give {num(thresholdModel.complexity)}, thresholds in either orientation — all {bothOrientations.length} distinct
      rows — give {num(bothOrientationsModel.complexity)}, and every possible sign row
      gives {num(cubeModel.complexity)}.</Prose>
    <Prose>Every row in one class is available in the next, so the best achievable match cannot decrease. This is a
      justified comparison of nested classes on one sample. “Trees are always more complex than linear models” is not
      justified without specifying outputs, constraints and inputs.</Prose>
    <Prose>The fixed all-positive predictor has complexity {num(singletonModel.complexity)} even if it predicts the
      real labels terribly. Complexity describes freedom to adapt, not whether the permitted predictions are
      appropriate. A broad class can also contain an excellent predictor: high capacity does not force the algorithm to
      select a bad one. These observations are why a useful risk bound needs a training-loss term as well as a
      complexity term.</Prose>

    {/* ============================================================ §2 */}
    <H2 id={headingId(headings[1])}>{headings[1]}</H2>
    <Prose>Let <Math>{'F'}</Math> be a nonempty class of real-valued functions.
      On inputs <Math>{'S=(x_1,\\ldots,x_n)'}</Math>, it produces a set of vectors</Prose>
    <MathBlock>{'\\begin{gathered}F|_S=\\\\\\{(f(x_1),\\ldots,f(x_n)):f\\in F\\}.\\end{gathered}'}</MathBlock>
    <Prose>The vertical bar means “restricted to this sample.” We can study these vectors without deciding how each
      function behaves elsewhere. A <strong>Rademacher variable</strong> is a fair
      sign: <Math>{'P(\\sigma_i=-1)=P(\\sigma_i=+1)=1/2'}</Math>. Draw the <Math>{'n'}</Math> signs independently. Our
      convention is</Prose>
    <MathBlock>{'\\begin{gathered}\\widehat{\\mathfrak R}_S(F)\\\\=\\mathbb E_\\sigma\\left[\\sup_{f\\in F}\\frac1n\\sum_{i=1}^n\\sigma_i f(x_i)\\right].\\end{gathered}'}</MathBlock>
    <Prose>“Supremum” means the largest achievable value, or its limiting value if the class approaches it without
      attaining it. It becomes an ordinary maximum for our finite prediction table. The expectation is just the average
      over all <Math>{'2^n'}</Math> sign patterns, or an approximation to that average when enumeration is too
      large.</Prose>
    <Prose>During this calculation, hold fixed the <strong>sample, function class and output definition</strong>. Only
      the auxiliary signs change. The true target labels are not needed for complexity of a predictor class. They are
      needed for complexity of a loss class, which is the object in the generalization theorem.</Prose>
    <Prose>The expected complexity adds another average:</Prose>
    <MathBlock>{'\\mathfrak R_n(F)=\\mathbb E_{S\\sim D^n}\\widehat{\\mathfrak R}_S(F).'}</MathBlock>
    <Prose>Here a fresh sample of <Math>{'n'}</Math> independent observations is drawn from <Math>{'D'}</Math> before
      the signs are drawn. Increasing the number of sign draws on one fixed sample estimates its empirical complexity
      more accurately; it does not average over new datasets.</Prose>

    <H3>Why a convention must accompany a number</H3>
    <Prose>Some references put an absolute value inside the supremum, multiply by 2, or both. These are useful related
      definitions, but the constants and properties must follow the chosen version. The original Bartlett–Mendelson
      paper uses an absolute-value, <Math>{'2/n'}</Math> normalization; the definition and theorem here
      follow <a href="https://cs.nyu.edu/~mohri/mls/lecture_3.pdf">Mohri&rsquo;s lecture on infinite hypothesis
      sets</a>.</Prose>
    <Prose>For our singleton all-positive rule, the displayed definition
      gives {num(singletonModel.complexity)}. Inserting an absolute value
      gives <Math>{'\\mathbb E|\\sigma_1+\\sigma_2+\\sigma_3|/3='}</Math> {num(absoluteSingleton.complexity)}. That is
      not a harmless formatting change: it measures the enlarged symmetric set consisting of the rule and its negative,
      which is exactly the two-constant class with
      complexity {num(constantModel.complexity)}. For classes already closed under negation, taking the absolute value
      does not change the per-draw supremum.</Prose>
    <Prose>Our complexity is nonnegative whenever the expectations exist: the expected maximum is at least the
      expectation for any fixed member, which is zero. If outputs lie in <Math>{'[-1,1]'}</Math>, it is at most 1. For
      unrestricted real-valued scores there is no universal upper limit of 1; multiplying every output by 10 multiplies
      complexity by 10.</Prose>

    <Prose>Several properties follow directly from the noise game. Each is an exact null you can reproduce in the
      investigation above.</Prose>
    <LessonTable caption="Properties of the quantity, and what each one means for the class"
      headers={['Change to the class', 'Effect on the complexity', 'Why']}
      rows={[
        ['Enlarge the class', 'cannot decrease', 'every old row is still available for each pattern'],
        ['Duplicate a prediction vector', 'exactly unchanged', 'a second name for an option you already had'],
        ['Add the same fixed vector to every member', 'exactly unchanged', 'the shift changes each score by a common amount whose expectation is zero'],
        ['Multiply all outputs by c', 'multiplied by |c|', 'every correlation scales by the same factor'],
        ['Add convex averages of existing vectors', 'exactly unchanged', 'an average correlation cannot exceed the largest original correlation'],
      ]} />

    <Prose>Adding the same <strong>fixed</strong> output shift is different from allowing the learner to choose any
      intercept. An unbounded freely chosen intercept can make the supremum infinite. An intercept must be fixed,
      separately bounded, or included in a norm constraint on an augmented feature vector — which is exactly what
      section 8 does.</Prose>
    <Prose>Nor must the empirical number decrease every time a new observation is appended.
      For <Math>{'f_w(x)=wx'}</Math> with <Math>{'|w|\\leq1'}</Math>, the sample <Math>{'[0]'}</Math> has
      complexity {num(oneSample.complexity)}. The sample <Math>{'[0,1]'}</Math> has
      complexity {num(twoSample.complexity)}. The usual rates concern expected behaviour under stated sampling and
      boundedness conditions, not a monotonicity promise for every observed sample sequence.</Prose>

    {/* ============================================================ §3 */}
    <H2 id={headingId(headings[2])}>{headings[2]}</H2>
    <Prose>A training observation is <Math>{'z=(x,y)'}</Math>. A predictor <Math>{'f'}</Math> incurs
      loss <Math>{'\\ell(f(x),y)'}</Math>. Its population risk is the expected loss on a new observation from the same
      population; its empirical risk is the average on the training sample:</Prose>
    <MathBlock>{'\\begin{gathered}R(f)=\\mathbb E_{(X,Y)\\sim D}\\,\\ell(f(X),Y),\\\\[4pt]\\widehat R_S(f)=\\frac1n\\sum_i\\ell(f(x_i),y_i).\\end{gathered}'}</MathBlock>
    <Prose>Define the <strong>loss class</strong></Prose>
    <MathBlock>{'\\begin{gathered}G=\\{g_f:\\;(x,y)\\mapsto\\\\\\ell(f(x),y):f\\in F\\}.\\end{gathered}'}</MathBlock>
    <Prose>Each member is now a function whose output is an error cost, rather than a prediction score. This
      distinction is essential. An arbitrary bounded loss need not preserve the complexity of raw predictions. A tiny
      positive score and a tiny negative score can produce different hard classifications even when their numerical
      difference is arbitrarily small.</Prose>

    <Prose>Assume <Math>{'G'}</Math> is fixed before drawing <Math>{'S'}</Math>, every <Math>{'g'}</Math> takes values
      in <Math>{'[0,1]'}</Math>, and <Math>{'S'}</Math> consists of <Math>{'n'}</Math> independent and identically
      distributed observations from <Math>{'D'}</Math>. Under the usual measurability conditions, for
      any <Math>{'\\delta\\in(0,1)'}</Math>, with probability at least <Math>{'1-\\delta'}</Math>
      over <Math>{'S'}</Math>, simultaneously for every <Math>{'f\\in F'}</Math>,</Prose>
    <MathBlock>{'\\begin{gathered}R(f)\\leq\\widehat R_S(f)+2\\widehat{\\mathfrak R}_S(G)\\\\[4pt]+\\;3\\sqrt{\\frac{\\ln(2/\\delta)}{2n}}.\\end{gathered}'}</MathBlock>
    <Prose>This is the empirical-complexity version. A separate expected-complexity version is</Prose>
    <MathBlock>{'\\begin{gathered}R(f)\\leq\\widehat R_S(f)+2\\mathfrak R_n(G)\\\\[4pt]+\\;\\sqrt{\\frac{\\ln(1/\\delta)}{2n}}.\\end{gathered}'}</MathBlock>
    <Prose>The first adapts to the observed sample but pays for that additional randomness — note the coefficient 3
      rather than 1. The second involves an expectation over datasets that is generally unknown. They are not
      interchangeable plug-in formulas. <a href="https://cs.nyu.edu/~mohri/mls/lecture_3.pdf">The two statements and
      their proof appear on slides 6–9 of Mohri&rsquo;s lecture</a>.</Prose>

    <TheoremTermsFigure />

    <Prose>The three terms answer different questions: how much loss was observed, how much selection freedom the
      class has on the sample, and how rare an unusually unrepresentative sample we are willing to
      allow. <Math>{'\\delta=.05'}</Math> describes the probability of failure of the simultaneous statement over
      repeated datasets.</Prose>
    <Prose>The word <strong>simultaneously</strong> lets us choose <Math>{'f'}</Math> using the training data. The
      chosen model remains one of the members for which the event holds. It does not let us inspect the data, invent an
      unrestricted new class containing only our chosen model, and claim its singleton complexity is zero. The class
      was supposed to be fixed before the sample. A representation learned on an independent sample may be conditioned
      on and frozen; one learned on the same observations requires an analysis covering that learning step.</Prose>

    <H3>A useful exact shortcut for binary classification</H3>
    <Prose>If <Math>{'h(x)\\in\\{-1,+1\\}'}</Math>, <Math>{'y\\in\\{-1,+1\\}'}</Math>, and the loss is a classification
      mistake, then</Prose>
    <MathBlock>{'\\ell(h(x),y)=\\frac{1-yh(x)}2.'}</MathBlock>
    <Prose>The fixed <Math>{'1/2'}</Math> term disappears after averaging over signs. Multiplying each fair sign by the
      fixed <Math>{'-y_i'}</Math> produces another independent fair sign. Therefore</Prose>
    <MathBlock>{'\\widehat{\\mathfrak R}_S(\\ell\\circ H)=\\tfrac12\\widehat{\\mathfrak R}_{X}(H).'}</MathBlock>

    <PredictorVersusLossFigure />

    <Prose>Our threshold example consequently has loss-class complexity {num(lossModel.complexity)} for any fixed
      binary target labels, compared with predictor complexity {num(thresholdModel.complexity)}. This exact identity
      concerns binary-valued hypotheses. It does not say that the hard-thresholded predictions of a bounded-norm score
      class have half the complexity of its real scores.</Prose>
    <Prose>If a bound&rsquo;s right side is 1.24 while the loss is in <Math>{'[0,1]'}</Math>, the trivial upper bound 1
      is better. The result is <strong>vacuous</strong> numerically. That may reveal loose inequalities, insufficient
      data, a class that is too broad, or a mismatch between what the theorem measures and the algorithm&rsquo;s useful
      structure. It does not prove that the model performs badly — section 8 shows a model that is
      demonstrably good and a bound that says nothing about it.</Prose>

    {/* ============================================================ §4 */}
    <H2 id={headingId(headings[3])}>{headings[3]}</H2>
    <Prose>The proof is easier to follow as an information flow: unknown population average → independent comparison
      sample → random pair swaps → two noise-matching problems → a concentration statement. The ghost sample is a
      mathematical device, not additional data that an implementation must collect.</Prose>
    <Prose>Write <Math>{'\\Phi(S)=\\sup_g(\\mathbb Eg-\\widehat{\\mathbb E}_Sg)'}</Math>, the largest optimism of any
      permitted loss function on <Math>{'S'}</Math>. Introduce <Math>{'S\''}</Math>, another independent sample of the
      same size. For a fixed <Math>{'g'}</Math>, its average on <Math>{'S\''}</Math> has
      expectation <Math>{'\\mathbb Eg'}</Math>. Allowing the choice of <Math>{'g'}</Math> to depend on the realized
      ghost sample can only increase the expected maximum, giving</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E_S\\Phi(S)\\\\\\leq\\mathbb E_{S,S\'}\\sup_g\\frac1n\\sum_i\\bigl[g(z\'_i)-g(z_i)\\bigr].\\end{gathered}'}</MathBlock>
    <Prose>Pair <Math>{'z_i'}</Math> with <Math>{'z\'_i'}</Math>. Since both are drawn independently from the same
      distribution, swapping their positions does not change the joint distribution. Independently choose to swap each
      pair using a fair sign. This gives the same expected supremum with each difference multiplied
      by <Math>{'\\sigma_i'}</Math>.</Prose>

    <GhostSampleFigure />

    <Prose>Now split one supremum into two:</Prose>
    <MathBlock>{'\\begin{gathered}\\sup_g\\sum_i\\sigma_i[g(z\'_i)-g(z_i)]\\\\\\leq\\sup_g\\sum_i\\sigma_i g(z\'_i)\\\\+\\sup_g\\sum_i(-\\sigma_i)g(z_i).\\end{gathered}'}</MathBlock>
    <Prose>The inequality may be loose because the right side lets two different functions win. Each term, after
      division by <Math>{'n'}</Math> and averaging, is the expected Rademacher complexity. Thus the expected largest
      generalization gap is at most twice the expected complexity of <Math>{'G'}</Math>. The factor 2 first appears
      here; it should not be inserted again at the ghost-sample step.</Prose>
    <Prose>Finally use <strong>bounded differences</strong>. Replacing one observation changes the average of
      a <Math>{'[0,1]'}</Math>-valued function by at most <Math>{'1/n'}</Math>. Taking a supremum preserves that bound,
      so <Math>{'\\Phi(S)'}</Math> changes by at most <Math>{'1/n'}</Math>. McDiarmid&rsquo;s inequality then gives a
      deviation above its expectation of at most <Math>{'\\sqrt{\\ln(1/\\delta)/(2n)}'}</Math>, except on an event of
      probability <Math>{'\\delta'}</Math>.</Prose>
    <Prose>The empirical complexity itself also changes by at most <Math>{'1/n'}</Math> when one observation changes.
      Apply a second concentration bound, allocating <Math>{'\\delta/2'}</Math> to each event. Replacing its population
      expectation by its observed value contributes two copies of the deviation term because the complexity is
      multiplied by 2; concentrating <Math>{'\\Phi'}</Math> contributes the third. That is where the empirical
      theorem&rsquo;s coefficient 3 comes from — and <strong>this is the same number appearing by two routes</strong>:
      the 2 in front of the complexity term and the 3 in front of the confidence term are not independent constants,
      but a consequence of the same replacement argument applied twice.</Prose>
    <Prose>The assumptions now have visible jobs. If samples have different distributions, pair swapping need not
      preserve the distribution. If the loss has no range or tail control, the <Math>{'1/n'}</Math> change argument
      fails. If the class changes with the sample, the same replacement proof does not automatically apply. Extensions
      exist, but require the corresponding theorem.</Prose>

    {/* ============================================================ §5 */}
    <H2 id={headingId(headings[4])}>{headings[4]}</H2>
    <H3>A Euclidean norm ball has a closed-form best response</H3>
    <Prose>Consider real scores <Math>{'f_w(x)=w^Tx'}</Math> with <Math>{'\\|w\\|_2\\leq B'}</Math>. The Euclidean norm
      is the vector&rsquo;s length, and the dot product measures alignment. For one sign vector,</Prose>
    <MathBlock>{'\\begin{gathered}\\sup_{\\|w\\|_2\\leq B}\\frac1n\\sum_i\\sigma_i w^Tx_i\\\\[4pt]=\\frac1n\\sup_{\\|w\\|_2\\leq B}w^Tv=\\frac{B}{n}\\|v\\|_2,\\end{gathered}'}</MathBlock>
    <Prose>where <Math>{'v=\\sum_i\\sigma_i x_i'}</Math>. The best vector points along <Math>{'v'}</Math> with
      length <Math>{'B'}</Math>: <Math>{'w^*=Bv/\\|v\\|_2'}</Math> if <Math>{'v\\neq0'}</Math>.
      If <Math>{'v=0'}</Math>, every allowed <Math>{'w'}</Math> gives zero. We have solved the inner optimization
      exactly; training a classifier on artificial labels would be unnecessary and would generally solve a different
      objective.</Prose>

    <SupportingPointFigure />

    <Prose>Taking the average over signs gives the exact empirical
      quantity <Math>{'B\\,\\mathbb E\\|\\sum_i\\sigma_ix_i\\|_2/n'}</Math>. For a convenient upper bound, an average
      length is at most the square root of the average squared length. Expand that square. Cross terms
      contain <Math>{'\\mathbb E[\\sigma_i\\sigma_j]=0'}</Math> for <Math>{'i\\neq j'}</Math>,
      while <Math>{'\\mathbb E\\sigma_i^2=1'}</Math>. Consequently,</Prose>
    <MathBlock>{'\\begin{gathered}\\widehat{\\mathfrak R}_S(F_B)\\leq\\frac{B}{n}\\sqrt{\\sum_i\\|x_i\\|_2^2}\\\\[4pt]\\leq\\frac{B\\max_i\\|x_i\\|_2}{\\sqrt n}.\\end{gathered}'}</MathBlock>
    <Prose>The first bound uses the observed feature energy; the second uses the largest observed row norm.
      If <Math>{'\\|X\\|_2\\leq R'}</Math> holds throughout the population, averaging
      yields <Math>{'\\mathfrak R_n(F_B)\\leq BR/\\sqrt n'}</Math>. A maximum observed in a sample is not automatically
      a bound on every future observation.</Prose>

    <H3>Same lengths, different directions</H3>
    <Prose>Take <Math>{'B=1'}</Math> and two observations. If both are <Math>{'(1,0)'}</Math>, the signed sum has
      length 2 for equal signs and zero for opposite signs. Complexity
      is {num(parallel.complexity)} after dividing each sum by <Math>{'n=2'}</Math>.</Prose>
    <Prose>If the observations are <Math>{'(1,0)'}</Math> and <Math>{'(0,1)'}</Math>, every signed sum has
      length <Math>{'\\sqrt2'}</Math>. Complexity is {num(perpendicular.complexity)}. Both samples have the same
      feature energy and the same energy upper bound {num(parallel.energyUpper)}. The exact quantity captures
      directional geometry that this upper bound discards.</Prose>

    <DirectionVersusEnergyFigure />

    <SignedGeometryLab />

    <Prose>A plot of model coefficients alone can mislead. Scaling all inputs by a positive <Math>{'c'}</Math> and
      scaling the entire coefficient budget by <Math>{'1/c'}</Math> leaves the set of possible predictions unchanged.
      Shrinking coefficients by changing feature units is not free capacity reduction.</Prose>

    <H3>Kernels keep the same calculation in a feature space</H3>
    <Prose>A kernel gives inner products of feature
      vectors: <Math>{'K_{ij}=k(x_i,x_j)=\\langle\\varphi(x_i),\\varphi(x_j)\\rangle'}</Math>. For the feature-space
      norm ball, the same squared-length calculation yields</Prose>
    <MathBlock>{'\\begin{gathered}\\widehat{\\mathfrak R}_S(F_B)=\\frac{B}{n}\\mathbb E_\\sigma\\sqrt{\\sigma^TK\\sigma}\\\\[4pt]\\leq\\frac{B}{n}\\sqrt{\\operatorname{tr}K}.\\end{gathered}'}</MathBlock>
    <Prose>The trace is the sum of diagonal entries. It measures total feature-space squared length. We do not need to
      materialize an infinite feature vector. With an RBF kernel normalized so <Math>{'k(x,x)=1'}</Math>, this bound
      is <Math>{'B/\\sqrt n'}</Math>. Infinite feature dimension therefore does not itself imply an infinite
      bounded-norm complexity.</Prose>

    <KernelGramFigure />

    <Prose>For a two-point Gram matrix with diagonal 1 and off-diagonal
      similarity <Math>{'r'}</Math>, <Math>{'r=0'}</Math> gives {num(kernelAtZero.complexity)}, <Math>{'r=.9'}</Math>
      {' '}gives {num(kernelAtNine.complexity)}, and <Math>{'r=1'}</Math> gives {num(kernelAtOne.complexity)}
      {' '}when <Math>{'B=1'}</Math>. The diagonal-based upper bound stays at {num(kernelAtZero.traceUpper)}
      {' '}throughout. These are exact four-sign calculations, not measured SVM accuracies.</Prose>
    <Prose>The function space associated with the kernel is called a reproducing kernel Hilbert space, or RKHS. Its
      score family must include its constraints: allowing an unbounded norm has no such finite score bound. A free
      intercept is still a separate issue. A fitted kernel SVM&rsquo;s norm is obtained from its signed dual
      coefficients <Math>{'a'}</Math> by <Math>{'\\|w\\|^2=a^TKa'}</Math>, not from the number of support vectors.
      Selecting a radius after seeing data also needs to be covered by the selection argument.</Prose>

    <H3>Finite classes, VC theory and sparse coefficients</H3>
    <Prose>For <Math>{'M'}</Math> distinct prediction vectors <Math>{'a^1,\\ldots,a^M\\in\\mathbb R^n'}</Math> with
      length at most <Math>{'A'}</Math>, <strong>Massart&rsquo;s finite-class bound</strong> is</Prose>
    <MathBlock>{'\\widehat{\\mathfrak R}_S(F)\\leq\\frac{A\\sqrt{2\\ln M}}n.'}</MathBlock>
    <Prose>For sign-valued functions <Math>{'A=\\sqrt n'}</Math>, so the familiar expression
      is <Math>{'\\sqrt{2\\ln M/n}'}</Math>. Sixteen fixed binary rules on 100 observations give an upper bound
      of {num(massartSignBound(16, 100))}. <Math>{'M=1'}</Math> gives {num(massartSignBound(1, 100))}. Counting
      repeated copies separately only weakens this bound; the actual quantity is unchanged.</Prose>
    <Prose>One proof explains why a logarithm appears. The exponential of a maximum is at most the sum of
      exponentials. For a fixed vector <Math>{'a'}</Math>, independence of signs
      and <Math>{'\\cosh(u)\\leq\\exp(u^2/2)'}</Math> bound <Math>{'\\mathbb E\\exp(\\lambda\\sigma^Ta)'}</Math>
      {' '}by <Math>{'\\exp(\\lambda^2A^2/2)'}</Math>. Taking a logarithm gives an upper
      bound <Math>{'\\ln(M)/\\lambda+\\lambda A^2/2'}</Math> on the expected maximum.
      Choose <Math>{'\\lambda=\\sqrt{2\\ln M}/A'}</Math> and divide by <Math>{'n'}</Math>. Zero-radius and singleton
      cases follow directly rather than dividing by zero.</Prose>
    <Prose>If a binary class has VC dimension <Math>{'1\\leq d\\leq n'}</Math>, Sauer&rsquo;s lemma bounds the number
      of distinct sample predictions
      by <Math>{'\\sum_{j=0}^dC(n,j)\\leq(en/d)^d'}</Math>. For <Math>{'d=3'}</Math> and <Math>{'n=100'}</Math> the
      exact count is {sauerExample.exactCount.toLocaleString('en-US')} and the simplified expression
      gives {sauerExample.simplified.toLocaleString('en-US', { maximumFractionDigits: 0 })}. Combining that count with
      Massart recovers a bound <Math>{'\\sqrt{2d\\ln(en/d)/n}'}</Math>, here {num(sauerExample.bound)}.
      For <Math>{'d=0'}</Math> the nonempty class has only one prediction pattern on every sample, so its complexity is
      zero. When <Math>{'d>n'}</Math>, use the bound <Math>{'2^n'}</Math> on the number of labelings instead; do not
      apply the simplified Sauer expression outside its range. Rademacher and VC analyses connect through restrictions
      to the sample. This particular derivation is not a claim that every Rademacher bound is strictly tighter than
      every VC result.</Prose>
    <Prose>For an <Math>{'\\ell_1'}</Math> coefficient budget <Math>{'\\|w\\|_1\\leq B'}</Math>, the inner optimum
      instead selects the largest absolute coordinate of <Math>{'v=\\sum_i\\sigma_ix_i'}</Math>:</Prose>
    <MathBlock>{'\\sup_{\\|w\\|_1\\leq B}w^Tv=B\\|v\\|_\\infty.'}</MathBlock>

    <L1VersusL2Figure />

    <Prose>You can spend the whole absolute-weight budget on that coordinate with the useful sign. Treating
      the <Math>{'d'}</Math> feature columns and their negatives as <Math>{'2d'}</Math> vectors, Massart gives</Prose>
    <MathBlock>{'\\begin{gathered}\\widehat{\\mathfrak R}_S(F_{\\ell_1,B})\\\\\\leq B\\max_i\\|x_i\\|_\\infty\\sqrt{\\frac{2\\ln(2d)}n}.\\end{gathered}'}</MathBlock>
    <Prose>This is a useful connection to sparse high-dimensional models. The logarithmic dimension term comes with a
      specific <Math>{'\\ell_1'}</Math> constraint and coordinate bound; it is not a promise that adding arbitrary
      features has no cost. <a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">Understanding
      Machine Learning, chapter 26, develops the finite, Euclidean and <Math>{'\\ell_1'}</Math> calculations</a>.</Prose>

    {/* ============================================================ §6 */}
    <H2 id={headingId(headings[5])}>{headings[5]}</H2>
    <Prose>A function <Math>{'\\varphi'}</Math> is <Math>{'L'}</Math>-Lipschitz if changing its input by a
      distance <Math>{'d'}</Math> changes its output by at most <Math>{'Ld'}</Math>. It has a bounded steepness, even
      when it has corners. For our no-absolute-value convention, coordinatewise <Math>{'L'}</Math>-Lipschitz maps
      satisfy the contraction inequality</Prose>
    <MathBlock>{'\\widehat{\\mathfrak R}_S(\\ell\\circ F)\\leq L\\,\\widehat{\\mathfrak R}_{X}(F),'}</MathBlock>
    <Prose>provided each map <Math>{'a\\mapsto\\ell(a,y_i)'}</Math> has that Lipschitz constant over the score range in
      question. Different observations may have different maps because their <Math>{'y_i'}</Math> differ. A fixed value
      at zero need not vanish for this convention; fixed offsets disappear under the sign expectation. Absolute-value
      versions may have different constants and centering
      requirements. <a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">The
      coordinatewise proof is lemma 26.9 in Understanding Machine Learning</a>.</Prose>
    <Prose>For intuition, condition on every sign except one. The remaining fair sign compares the two best responses
      obtained by adding and subtracting one coordinate. A Lipschitz map cannot separate those alternatives more
      than <Math>{'L'}</Math> times their original separation. Repeat this replacement coordinate by coordinate. This
      argument bounds a supremum over a class; it does not assert that the fitted model&rsquo;s losses individually
      improve.</Prose>

    <LossSlopesFigure />

    <LessonTable caption="Each map with its input stated, a valid Lipschitz constant, and what its range does to the [0, 1] risk theorem"
      headers={['Map, with its input stated', 'Valid Lipschitz constant', 'Range issue for the [0,1] risk theorem']}
      rows={[
        ['Hinge margin loss max(0, 1 − m)', '1', 'unbounded as m → −∞; bound or truncate it before using that theorem'],
        ['Logistic margin loss ln(1 + exp(−m))', '1', 'also unbounded on unrestricted margins'],
        ['Sigmoid σ(s) = 1/(1 + exp(−s))', '1/4', 'a probability map in [0, 1], not the logistic loss'],
        ['Squared error (a − y)², a, y ∈ [−M, M]', '4M in a', 'range [0, 4M²]; normalize or use the range-dependent bound'],
        ['Clipped margin loss below', '1/ρ', 'always in [0, 1]'],
      ]} />

    <Prose>For logistic loss the derivative in <Math>{'m'}</Math> is <Math>{'-1/(1+\\exp(m))'}</Math>, whose magnitude
      approaches 1. For squared loss it is <Math>{'2(a-y)'}</Math>, which can have magnitude <Math>{'4M'}</Math>. This
      is why borrowing the sigmoid&rsquo;s <Math>{'1/4'}</Math> constant for logistic loss, or calling squared
      loss &ldquo;2-Lipschitz&rdquo; without a domain, leads to incorrect numbers. A smaller upper bound for one loss
      also does not automatically make it the better learning objective for another task.</Prose>

    <H3>Margins make a hard classification decision analyzable</H3>
    <Prose>For <Math>{'y\\in\\{-1,+1\\}'}</Math>, the <strong>margin</strong> <Math>{'m=yf(x)'}</Math> is positive when
      the score has the correct sign, negative when it has the wrong sign, and zero at the boundary. A large positive
      margin means more score change is needed to reverse the decision. Pick a fixed positive
      threshold <Math>{'\\rho'}</Math> and define</Prose>
    <MathBlock>{'\\begin{gathered}\\phi_\\rho(m)=\\\\\\begin{cases}1,&m\\leq0,\\\\1-m/\\rho,&0<m<\\rho,\\\\0,&m\\geq\\rho.\\end{cases}\\end{gathered}'}</MathBlock>

    <RampFigure />

    <Prose>This ramp upper-bounds a classification error, including either consistent tie decision at score zero. It
      also charges partially for correct predictions close to the boundary.
      Margins <Math>{'[-.2,.1,.4,1.2]'}</Math> with <Math>{'\\rho=.5'}</Math> produce
      losses <Math>{'['}</Math>{defaultRamp.map(value => num(value)).join(', ')}<Math>{']'}</Math>, whose mean
      is {num(defaultRamp.reduce((sum, value) => sum + value, 0) / defaultRamp.length)}. Only one of the four is an
      outright wrong sign, but two more are fragile at this chosen scale.</Prose>
    <Prose>Contraction and the empirical theorem give, simultaneously for <Math>{'f'}</Math> in the fixed score
      family,</Prose>
    <MathBlock>{'\\begin{gathered}P(Yf(X)\\leq0)\\\\[2pt]\\leq\\frac1n\\sum_i\\phi_\\rho(y_i f(x_i))\\\\[4pt]+\\;\\frac{2}{\\rho}\\widehat{\\mathfrak R}_{X}(F)\\\\[4pt]+\\;3\\sqrt{\\frac{\\ln(2/\\delta)}{2n}}.\\end{gathered}'}</MathBlock>
    <Prose>The event on the left counts all zero margins as errors, so it is an upper bound for a classifier with a
      fixed tie rule. For a norm ball, substitute its geometric complexity bound.
      Increasing <Math>{'\\rho'}</Math> makes more training observations count as small-margin cases but decreases
      the <Math>{'1/\\rho'}</Math> complexity multiplier. The learner must balance both terms.</Prose>

    <MarginBoundLab />

    <H3>Model selection needs its own accounting</H3>
    <Prose>If you predeclare <Math>{'K'}</Math> candidate combinations of norm budget and margin threshold, give each
      bound failure allowance <Math>{'\\delta/K'}</Math>. A union bound says all <Math>{'K'}</Math> hold together
      except on an event of probability at most <Math>{'\\delta'}</Math>. The confidence term
      becomes <Math>{'3\\sqrt{\\ln(2K/\\delta)/(2n)}'}</Math>. Now choosing among these candidates does not invalidate
      their simultaneous bounds.</Prose>
    <Prose>This is a simple form of <strong>structural risk minimization</strong>: compare empirical fit plus a
      complexity allowance across specified classes. For a countable collection, allocate failure budgets that sum
      to <Math>{'\\delta'}</Math>. Choosing among an unrestricted continuum after the fact requires an appropriate
      uniform theorem, discretization argument or independent selection procedure. A plain regularization sweep is
      useful validation; it is not itself a computed Rademacher certificate.</Prose>
    <Prose>This clarifies the connection to a soft-margin SVM. Its usual objective is one-half the squared coefficient
      norm plus <Math>{'C'}</Math> times the sum of hinge losses. A smaller <Math>{'C'}</Math> makes margin violations
      less costly relative to a large norm; it does not enforce a smaller tolerated violation.
      Changing <Math>{'C'}</Math> changes that tradeoff, while the resulting norm and margin distribution determine
      quantities useful to a bound. The objective does not directly minimize the exact Rademacher complexity, and a
      large support-vector count alone does not diagnose either overfitting or excessive regularization. Compare the
      actual losses, coefficients, margins and held-out performance.</Prose>

    {/* ============================================================ §7 */}
    <H2 id={headingId(headings[6])}>{headings[6]}</H2>
    <Prose>The complete calculation program enumerates all signs for the small tables and computes every quantity
      above. Download <a href={provenance.calculationProgram} download>complexity_calculations.py</a>, save it
      locally, and run it with Python and NumPy.</Prose>

    <Program example={rademacherPrograms.runCalculations}>
      <Prose>Two of those six you already knew without running anything: the singleton is zero because a fixed vector
        has zero average correlation with fair signs, and the full cube is 1 because some row matches every pattern
        exactly. The mistake-class value {num(lossModel.complexity)} is exactly half
        the {num(thresholdModel.complexity)} above it, which is the identity from section 3 appearing again — the
        same number by a second route, and a sign that the convention held all the way through.</Prose>
    </Program>

    <Excerpt excerpt={rademacherExcerpts.finiteComplexity}>
      Every dot product, the maximum across hypotheses for each sign pattern, then the average. It deliberately
      contains no <Code>abs</Code>: the convention is in the code, not only in the prose.
    </Excerpt>

    <Excerpt excerpt={rademacherExcerpts.thresholdValues}>
      The two infinities are the point. Restricting cutoffs to observed values would drop the all-negative row, and
      the answer would come out wrong rather than merely incomplete. Equal inputs are grouped, because no threshold
      can separate two identical observations.
    </Excerpt>

    <Prose>For <Math>{'n'}</Math> observations, full sign enumeration has <Math>{'2^n'}</Math> rows. The program limits
      it to <Math>{'n\\leq12'}</Math>. At larger <Math>{'n'}</Math>, draw <Math>{'T'}</Math> independent sign patterns
      and average their exact best-response values. For a Euclidean norm ball each value lies
      in <Math>{'[0,Q]'}</Math>, where</Prose>
    <MathBlock>{'Q=\\frac{B}{n}\\sum_i\\|x_i\\|_2.'}</MathBlock>
    <Prose>Hoeffding&rsquo;s inequality gives a useful upper correction. With probability at
      least <Math>{'1-\\eta'}</Math> over the sign draws, conditional on the fixed sample,</Prose>
    <MathBlock>{'\\widehat{\\mathfrak R}_S(F)\\leq\\widehat{\\mathfrak R}_{MC}+Q\\sqrt{\\frac{\\ln(1/\\eta)}{2T}}.'}</MathBlock>
    <Prose>For the two duplicate unit vectors whose exact complexity is {num(recordedMonteCarlo.exact)}, one executed
      sequence with seed {recordedMonteCarlo.table[0].seed} produced the trajectory below.</Prose>

    <MonteCarloFigure />

    <Prose>More draws shrink the deterministic correction but do not make a particular estimate
      approach {num(recordedMonteCarlo.exact)} monotonically. The displayed endpoint is a guarantee for each
      predeclared draw count separately. Inspecting many endpoints and reporting the most favourable one needs
      simultaneous or sequential accounting.</Prose>
    <Prose>If a risk theorem fails with probability <Math>{'\\delta'}</Math> and a separately computed Monte Carlo
      upper correction fails with probability <Math>{'\\eta'}</Math>, a union bound permits total failure
      allowance <Math>{'\\delta+\\eta'}</Math>. Omitting <Math>{'\\eta'}</Math> silently turns an estimate into an
      asserted upper bound. When an analytic upper bound is already smaller than the corrected Monte Carlo result, use
      the analytic one.</Prose>
    <Prose>There is another source of error: the inner optimization. A fitted neural network that achieves
      correlation .3 proves that the supremum is <strong>at least</strong> .3. Failure to optimize it further does not
      prove that the class cannot achieve .9. Repeating imperfect optimization many times does not repair the
      inequality direction. A certificate needs an exact supremum or a justified upper bound on it; an experiment with
      random labels remains useful evidence about the particular training procedure.</Prose>
    <Prose>For dense linear scores, one sign draw costs <Math>{'O(nd)'}</Math>; <Math>{'T'}</Math> draws
      cost <Math>{'O(Tnd)'}</Math>. Process draws in bounded chunks for large matrices. A kernel Gram matrix
      requires <Math>{'O(n^2)'}</Math> storage if materialized and <Math>{'O(n^2)'}</Math> work per dense quadratic
      form, while the trace bound needs only diagonal values. Finite classes cost <Math>{'O(TnM)'}</Math> using the
      prediction table; threshold-specific cumulative sums can reduce repeated work after sorting. None of these
      formulas implies a hardware-independent runtime in seconds.</Prose>

    {/* ============================================================ §8 */}
    <H2 id={headingId(headings[7])}>{headings[7]}</H2>
    <Prose>We now use the <a href={provenance.doi}>UCI Banknote Authentication dataset</a>, attributed
      to {provenance.creator} and available under <a href={provenance.licenseUrl}>{provenance.license}</a>. Its four
      inputs are wavelet-derived statistics; the target is a binary class code. The
      retained <a href={provenance.file} download>{provenance.retainedRows}-row subset</a> preserves source row IDs. We
      use it to compare constrained predictors and inspect a bound calculation, without inferring unverified
      operational meaning from the class codes.</Prose>

    <H3>Declare the information flow before fitting</H3>
    <DataRolesFigure />

    <Prose>The file is the same fixed subset used in neighbouring lessons, but this experiment assigns four explicit
      roles. The CSV&rsquo;s historical <Code>split</Code> labels remain in the file; the program&rsquo;s explicit
      index allocation governs this experiment. A source row belongs to one role only.</Prose>
    <Prose><strong>Distinct row IDs do not ensure distinct inputs.</strong> This subset
      has {duplicateFeatureGroups.length} pairs with identical feature
      vectors: {duplicateFeatureGroups.map(group => `rows ${group.sourceRows[0]} and ${group.sourceRows[1]} (${group.roles.join(' / ')})`).join('; ')}. The
      validation result therefore includes two feature vectors already seen during fitting. We retain this declared
      finite-corpus study to inspect its mathematics, without presenting the validation score as performance on wholly
      new feature vectors. A new experiment intended to assess that question should group identical inputs before
      assigning roles, as the AutoML lesson does.</Prose>
    <Prose>Freeze the representation fitted on the first 80 observations. For every
      later input, subtract each coordinate&rsquo;s fitted mean and divide by its fitted standard deviation, then
      divide by {experimentSettings.clipStandardDeviations} and clip to <Math>{'[-1,1]'}</Math>. Append a constant 1
      coordinate for the intercept. The resulting five-dimensional vector has norm at
      most <Math>{'\\sqrt5'}</Math> on every input. Its intercept weight is constrained together with the other
      weights. Convert class 0 to label <Math>{'-1'}</Math> and class 1 to label <Math>{'+1'}</Math>. Clipping limits
      extreme feature influence and changes the available predictor family.</Prose>
    <Prose>For each <Math>{'B'}</Math> in <Math>{'\\{.25,.5,1,2,4\\}'}</Math>, fit</Prose>
    <MathBlock>{'\\begin{gathered}\\min_{\\|w\\|_2\\leq B}\\;\\frac1{240}\\sum_i\\\\[2pt]\\ln(1+\\exp(-y_iw^T\\tilde x_i)).\\end{gathered}'}</MathBlock>

    <Excerpt excerpt={rademacherExcerpts.fitBall}>
      A numerical constrained optimizer for this convex objective, with an analytic gradient, a feasibility check and a
      stationarity residual. The tiny inward projection repairs solver roundoff at the radius <em>before</em> the
      objective is reported, so the number described is the number the returned model actually achieves. It optimizes
      logistic loss but the bound <strong>evaluates the bounded ramp loss</strong>: contraction does not grant
      unbounded logistic loss the <Math>{'[0,1]'}</Math> theorem for free.
    </Excerpt>

    <Prose>Choose the candidate with the fewest validation classification errors, breaking ties by validation log loss
      and then smaller <Math>{'B'}</Math>. The assessment labels are not used in that choice. Two margin
      thresholds, <Math>{'\\rho=.5'}</Math> and <Math>{'\\rho=1'}</Math>, are predeclared for the theory comparison,
      giving <Math>{'K=' + experimentSettings.comparisons}</Math> budget/threshold pairs.</Prose>

    <H3>Run the complete experiment</H3>
    <Prose>Download <a href={provenance.experimentProgram} download>bounded_norm_experiment.py</a>,{' '}
      <a href={provenance.calculationProgram} download>complexity_calculations.py</a> and{' '}
      <a href={provenance.file} download>banknote-subset.csv</a> into one directory. Then run:</Prose>

    <Program example={rademacherPrograms.runExperiment}>
      <Prose>The author run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. Small numerical
        differences across versions are possible; the program checks the stated constraints rather than trusting
        formatted output.</Prose>
    </Program>

    <BoundComponentsFigure />

    <Prose>The validation rule selects <Math>{'B=' + selection.chosenRadius}</Math>. Its assessment error
      is {selected.assessment.errors}/80, compared with {majorityBaseline.assessmentErrors}/80 for the majority class
      chosen from the fit set. Larger budgets improve measured classification performance throughout this particular
      sweep. We do not manufacture a turn upward in the error curve to make the regularization story look more
      familiar.</Prose>

    <MarginDistributionFigure />

    <Prose>For <Math>{'B=2'}</Math> and <Math>{'\\rho=1'}</Math>, the calculation is completely inspectable:</Prose>
    <MathBlock>{'\\begin{gathered}\\underbrace{' + to(workedBound.empiricalRamp, 6) + '}_{\\text{training ramp}}\\\\+\\;\\underbrace{2(2)(' + to(experimentSettings.energyFactor, 7) + ')}_{\\text{complexity addend}}\\\\+\\;\\underbrace{' + to(experimentSettings.confidenceAddend, 6) + '}_{\\text{confidence},\\ K=10,\\ \\delta=.05}\\\\\\approx' + to(workedBound.rawUpper, 6) + '.\\end{gathered}'}</MathBlock>
    <Prose>Here {to(experimentSettings.energyFactor, 7)} is <Math>{'\\sqrt{\\sum\\|\\tilde x_i\\|^2}/n'}</Math> for
      the {experimentSettings.fitRows} fit vectors. Every candidate bound expression exceeds 1, so this calculation
      gives no informative numerical certificate. That remains true even though the measured classifier is useful. The
      expression favours <Math>{'B=' + smallestAtOne.radius}</Math> at <Math>{'\\rho=1'}</Math>, whereas validation
      favours <Math>{'B=' + selection.chosenRadius}</Math>; an upper bound is not a prediction of the assessment
      error.</Prose>

    <BoundedNormLab />

    <Prose>The data were sampled without replacement from a fixed source corpus. That finite-corpus design and any
      acquisition dependence are different from the iid population model of our stated theorem. The table evaluates its
      expression as a theory diagnostic; it does not certify that a future banknote population is iid or that its risk
      has a particular bound. A formal deployment certificate would need a justified sampling model and the matching
      theorem. Independently of that issue, these expressions are already numerically vacuous.</Prose>
    <Prose>The program also estimates the <strong>unit-ball</strong> empirical score complexity on the fixed fit
      vectors: {num(recordedMonteCarlo.unitBall.estimate)} with{' '}
      {recordedMonteCarlo.unitBall.draws.toLocaleString('en-US')} sign draws. Its conservative one-sided Monte Carlo
      endpoint is {num(recordedMonteCarlo.unitBall.endpoint)}, while the analytic feature-energy upper bound
      is {num(experimentSettings.energyFactor)}. The analytic result is better here. This comparison concerns the same
      fixed geometry; neither number is the model&rsquo;s assessment error.</Prose>

    {/* ============================================================ §9 */}
    <H2 id={headingId(headings[8])}>{headings[8]}</H2>
    <Prose>This whole section is a deeper branch. Nothing below is needed for the core route; come back once the small
      sums in sections 1–8 feel natural.</Prose>

    <H3>Why averaging many learners can preserve score complexity</H3>
    <Prose>Suppose a boosting or voting score is a convex average of base
      predictions: <Math>{'f=\\sum_j\\alpha_jh_j'}</Math>, <Math>{'\\alpha_j\\geq0'}</Math>
      and <Math>{'\\sum_j\\alpha_j=1'}</Math>. For any fixed noise signs, its correlation is a weighted average of the
      base correlations and cannot exceed their maximum. Since each base learner is itself an allowed average, the two
      suprema are equal. The convex hull has the same empirical score complexity as the base class.</Prose>

    <ConvexHullFigure />

    <Prose>This is an interesting reason to track margins rather than only the number of fitted components. A large
      ensemble can improve its margin distribution without automatically increasing the normalized score-class
      complexity. Thresholding the average into a hard label is discontinuous, so the same conclusion does not transfer
      directly to its 0/1 loss; the margin argument supplies the missing connection. Unnormalized or signed coefficient
      sums need their own budget.</Prose>

    <H3>Covering numbers, chaining and local complexity</H3>
    <Prose>Counting all functions is wasteful when many make nearly identical predictions. A covering set approximates
      the prediction vectors within a chosen distance. Coarse covers identify major distinctions; finer covers account
      for smaller residual differences. <strong>Chaining</strong> combines bounds over multiple scales instead of
      paying the finest-scale count for every distinction. This can sharpen a simple one-scale Massart/Sauer
      analysis.</Prose>
    <Prose>Global complexity also measures functions the learning procedure is unlikely to consider near a good
      solution. <strong>Local Rademacher analysis</strong> restricts attention to a region, often defined by an
      excess-loss or variance condition, and solves a relation between that region&rsquo;s radius and its complexity.
      Faster rates can emerge under additional noise or curvature conditions. Arbitrarily deleting poor-looking
      hypotheses after observing the sample is not a proof of
      localization. <a href="https://arxiv.org/abs/math/0508275">Bartlett, Bousquet and Mendelson&rsquo;s
      local-complexity paper</a> is a deeper route for this distinction.</Prose>
    <Prose>Gaussian complexity replaces fair signs with independent standard-normal multipliers. It supports related
      geometric and comparison arguments but is a different quantity with its own normalization and tail
      behaviour. <a href="https://jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf">Bartlett and Mendelson&rsquo;s
      structural-results paper</a> develops both measures and applications to kernels, networks and trees.</Prose>

    <H3>Neural networks need the class and normalization to be explicit</H3>
    <Prose>A sufficiently rich neural architecture may fit many random labelings, yet a particular training procedure
      can still generalize on structured data. A coarse uniform bound over every permitted network may miss the
      constraints or preferences that matter. It is equally inaccurate to declare every finite network&rsquo;s VC
      dimension infinite or to declare every Rademacher-based neural bound useless.</Prose>
    <Prose>Norm-based results specify network depth, activation properties and layer constraints. Neyshabur, Tomioka
      and Srebro study group and path norms, including cases where width dependence disappears and cases where it
      cannot. The bound is not universally &ldquo;product of norms divided by <Math>{'\\sqrt n'}</Math>&rdquo; with all
      other factors dropped. <a href="https://proceedings.mlr.press/v40/Neyshabur15.pdf">Their theorem 1 and its
      conditions</a> show why the precise norm and depth matter.</Prose>
    <Prose>Later spectral-margin results combine products of layer operator norms with additional complexity factors
      and normalize by margins. Scaling successive layers or the final scores can make an unnormalized norm or margin
      look larger without the simple generalization interpretation suggested by that number
      alone. <a href="https://papers.neurips.cc/paper/7204-spectrally-normalized-margin-bounds-for-neural-networks.pdf">Bartlett,
      Foster and Telgarsky&rsquo;s spectral-margin paper</a> makes this comparison on stated architectures and
      datasets. Its empirical associations do not prove that every increase in a trained network&rsquo;s norm causes
      worse future performance.</Prose>

    <H3>Compare frameworks by what they control</H3>
    <LessonTable caption="Four frameworks, what each one constrains, and what an application of it must supply"
      headers={['Framework', 'Object constrained', 'What must accompany an application']}
      rows={[
        ['VC/growth analysis', 'possible binary prediction patterns', 'defined class, sampling assumptions and the applicable finite-sample result'],
        ['Rademacher analysis', 'average best noise correlation of score or loss classes', 'exact convention, class constraints, loss range and justified computation'],
        ['PAC-Bayes', 'a distribution Q over predictors relative to a prior P', 'a valid prior choice, KL term, empirical randomized loss and the specific theorem'],
        ['Algorithmic stability', 'change in loss when training data change', 'algorithm, neighbouring-data definition, expectation versus high-probability distinction, and smoothness/step assumptions'],
      ]} />

    <Prose>PAC-Bayes lets the posterior <Math>{'Q'}</Math> depend on the sample under a bound that is uniform over
      posteriors. The KL divergence measures how <Math>{'Q'}</Math> redistributes probability relative
      to <Math>{'P'}</Math>; it is not merely the distance between their means. In the basic theorem the
      prior <Math>{'P'}</Math> must be independent of that sample; selecting a prior using the same observations needs
      explicit accounting. Dziugaite and Roy optimize a stochastic network posterior, use a prior mean fixed at random
      initialization, and account for choosing a prior variance from a discrete family. They also bound the error in
      estimating the randomized classifier&rsquo;s empirical
      loss. <a href="https://arxiv.org/pdf/1703.11008">Sections 3.1–3.3 describe those separate steps</a>. Merely
      moving a prior onto a trained network and declaring the KL small is not valid.</Prose>
    <Prose>For stability, changing one observation is a perturbation of the <strong>training procedure</strong>, not a
      random-label fit. A uniform stability guarantee bounds the change in loss over all neighbouring datasets and
      query examples; measuring one leave-one-out change is a diagnostic, not the supremum. Under
      convex, <Math>{'\\beta'}</Math>-smooth, <Math>{'L'}</Math>-Lipschitz losses and suitable
      steps <Math>{'\\eta_t\\leq2/\\beta'}</Math>, Hardt, Recht and Singer obtain stability at
      most <Math>{'(2L^2/n)\\sum_t\\eta_t'}</Math>. Their nonconvex result uses different conditions and dependence on
      steps. This is not a universal high-probability gap of <Math>{'2\\varepsilon'}</Math> for arbitrary SGD-trained
      networks. <a href="https://proceedings.mlr.press/v48/hardt16.pdf">The convex and nonconvex results are theorems
      3.7 and 3.8</a>.</Prose>
    <Prose>These tools answer related questions under different constraints. There is no universal ranking that makes
      one framework the best certificate for every transformer, kernel model or dataset. A meaningful certificate needs
      an actual theorem, valid information boundaries and computed quantities; a useful learning curve needs a sound
      evaluation protocol. Each can contribute without being mislabeled as the other.</Prose>

    {/* ============================================================ §10 */}
    <H2 id={headingId(headings[9])}>{headings[9]}</H2>

    <Practice title="1. Two inputs, three thresholds"
      question={<>For inputs 2 and 5, the positive-threshold prediction rows
        are <Math>{'(-1,-1)'}</Math>, <Math>{'(-1,+1)'}</Math>, <Math>{'(+1,+1)'}</Math>. Compute the exact empirical
        complexity over all four sign patterns. Would adding <Math>{'(+1,-1)'}</Math> change it?</>}
      hint={<>For each pattern, take the largest of the three normalized dot products. Do not average each
        hypothesis&rsquo;s four values first.</>}>
      <Prose>The maxima for signs <Math>{'(-,-)'}</Math>, <Math>{'(-,+)'}</Math>, <Math>{'(+,-)'}</Math>
        , <Math>{'(+,+)'}</Math> are {practiceThresholds.maxima.map(value => num(value)).join(', ')}. Their average
        is {num(practiceThresholds.complexity)}. Adding the missing prediction row makes every maximum 1, so complexity
        becomes {num(practiceThresholdsPlus.complexity)}. Duplicating any existing row would
        preserve {num(practiceThresholds.complexity)}.</Prose>
    </Practice>

    <Practice title="2. A mysterious absolute value"
      question={<>A colleague reports complexity {num(absoluteSingleton.complexity)} for a single fixed all-positive
        rule on three observations. Your calculation gives {num(singletonModel.complexity)}. Explain how both numbers
        might have been obtained, and why one cannot be pasted into the other&rsquo;s theorem without checking
        conventions.</>}
      hint={<>List the possible sums of three signs and ask whether the sign of that sum was retained.</>}>
      <Prose>The ordinary expected signed sum is zero. The absolute sum has average 1.5,
        giving {num(absoluteSingleton.complexity)} after division by 3. Taking the absolute value effectively permits
        the rule&rsquo;s negative as well — which is why it returns the two-constant class&rsquo;s
        value {num(constantModel.complexity)}. The definition has changed, so factors, centering properties and
        associated risk statements must be checked rather than mixed.</Prose>
    </Practice>

    <Practice title="3. Geometry with a different scale"
      question={<>For <Math>{'x_1=(3,0)'}</Math>, <Math>{'x_2=(0,4)'}</Math>, and <Math>{'\\|w\\|_2\\leq2'}</Math>,
        compute the exact score complexity and the feature-energy bound. Does an answer greater than 1 indicate a
        bug?</>}
      hint={<>Every signed sum has the same length. Include <Math>{'B'}</Math> and the division
        by <Math>{'n'}</Math>.</>}>
      <Prose>Every signed sum is <Math>{'(\\pm3,\\pm4)'}</Math>, length 5. The inner optimum
        is <Math>{'2\\times5/2='}</Math>{num(practiceGeometry.maxima[0])} for every sign pattern, so the average
        is {num(practiceGeometry.complexity)}. The energy bound is
        also <Math>{'2\\sqrt{9+16}/2='}</Math>{num(practiceGeometry.energyUpper)}. These are unrestricted real scores,
        not outputs constrained to <Math>{'[-1,1]'}</Math>; the unit ceiling does not apply. A risk theorem still
        requires its stated loss range or a valid margin transformation.</Prose>
    </Practice>

    <Practice title="4. Compare a finite-class bound correctly"
      question={<>There are eight distinct sign-valued hypotheses on 200 observations. Compute Massart&rsquo;s bound.
        If 1,000 duplicate copies of those same rows are added to the file, what changes?</>}
      hint={<>Use natural logarithms and the number of distinct prediction vectors. Separate a bound from the exact
        quantity.</>}>
      <Prose>The bound is <Math>{'\\sqrt{2\\ln 8/200}\\approx'}</Math>{num(massartSignBound(8, 200))}. The exact
        complexity may be smaller. Duplicate rows leave every maximum and the exact complexity unchanged. Counting
        copies inside the logarithm would give {num(massartSignBound(1008, 200))} — a valid but unnecessarily looser
        upper bound; deduplicating restores the original calculation.</Prose>
    </Practice>

    <Practice title="5. Repair a loss comparison"
      question={<>Margins are <Math>{'[-.1,.2,.8]'}</Math> and <Math>{'\\rho=.4'}</Math>. Calculate the ramp loss, then
        explain what is wrong with &ldquo;logistic loss is 1/4-Lipschitz, therefore its classification guarantee is
        always four times better than hinge.&rdquo;</>}
      hint={<>Distinguish a sigmoid probability from <Math>{'\\ln(1+\\exp(-m))'}</Math>, and distinguish a gap bound
        from a risk comparison.</>}>
      <Prose>Ramp losses are <Math>{'['}</Math>{practiceRamp.map(value => num(value)).join(', ')}<Math>{']'}</Math>,
        with mean {num(practiceRamp.reduce((sum, value) => sum + value, 0) / practiceRamp.length)}. Logistic margin
        loss is 1-Lipschitz, while the sigmoid probability map is 1/4-Lipschitz. Hinge and logistic loss are unbounded
        over unrestricted margins; the <Math>{'[0,1]'}</Math> theorem cannot be applied unchanged. Even valid different
        loss bounds concern different empirical objectives, ranges and potentially fitted models, so a ratio of one
        term is not a universal classification comparison.</Prose>
    </Practice>

    <Practice title="6. The apparently perfect singleton certificate"
      question={<>An agent memorizes a training set with a flexible model, defines <Math>{'F'}</Math> afterward as just
        that fitted function, computes complexity zero and claims that only the confidence term is needed. Identify the
        missing condition and give two valid ways to proceed.</>}
      hint={<>Ask when <Math>{'F'}</Math> was fixed relative to the sample used for the theorem.</>}>
      <Prose>The class is sample-dependent, so the fixed-class proof does not apply. One option is to analyze a
        predeclared class covering all possible selected models, with its actual complexity. Another is to freeze the
        model and evaluate it on genuinely independent data with a fixed-predictor concentration result. An appropriate
        theorem for data-dependent classes is a further route, but cannot be assumed from the ordinary
        statement.</Prose>
    </Practice>

    <Practice title="7. Diagnose the real experiment"
      question={<>The <Math>{'B=4'}</Math> candidate has {selected.assessment.errors}/80 assessment errors, but
        its <Math>{'\\rho=1'}</Math> bound expression is about {to(fittedModels[4].bounds[1].rawUpper, 4)}. Does this
        disprove the theorem? Should we choose <Math>{'B=2'}</Math> merely because its expression is smaller? Would
        10,000 more sign draws solve the problem?</>}
      hint={<>Distinguish the population assumptions, the raw upper expression, the declared validation rule and which
        part uses Monte Carlo.</>}>
      <Prose>No. An upper expression above 1 is uninformative, not contradicted by a small measured error; the real
        fixed-corpus experiment also does not establish the iid deployment assumptions. The supplied selection rule
        chooses <Math>{'B=' + selection.chosenRadius}</Math> by validation, and the bound is not a test-error
        prediction. Its displayed value uses the analytic energy upper bound, so more sign draws would not alter that
        calculation at all. A refined complexity analysis could change a bound, while better validation or additional
        independent data could change the practical evidence; those are separate tasks.</Prose>
    </Practice>

    <Practice title="8. Audit a random-label “upper bound”"
      question={<>A neural optimizer reaches mean signed correlation .35 over 200 random-label fits. Its author
        calls .35 an upper bound on the whole architecture&rsquo;s Rademacher complexity and inserts it into a 95% risk
        bound. Name the two distinct issues, even if the underlying sample were iid.</>}
      hint={<>One issue is the direction of inner optimization error. The other is finite simulation
        uncertainty.</>}>
      <Prose>The achieved correlation is no larger than the supremum; an imperfect optimizer supplies lower evidence
        about capacity, not a certified upper value. Even if the supremum were exact, averaging only 200 random-sign
        draws would estimate empirical complexity and need a justified upper correction with its own failure allowance.
        The class, output bounds and loss transformation must also match the theorem. Increasing the number of
        imperfect fits addresses neither the missing optimization certificate nor the loss-class distinction
        automatically.</Prose>
    </Practice>

    {/* ============================================================ §11 */}
    <H2 id={headingId(headings[10])}>{headings[10]}</H2>
    <Prose>You are ready to move on when you can compute a small empirical complexity exactly, say why the maximum
      comes before the average, name which quantity a stated theorem needs and over what its probability is taken,
      derive a norm-ball best response, choose a margin threshold and account for the comparisons you made, and look at
      a vacuous number and say precisely what it does and does not rule out.</Prose>

    <LessonTable caption="Readiness check"
      headers={['you should be able to', 'where it was taught']}
      rows={[
        ['Compute an exact empirical complexity by enumerating sign patterns', 'Section 1, figures 1–2, investigation 1'],
        ['Explain why maximum-then-average differs from average-then-maximum', 'Section 1, figure 2'],
        ['Reproduce the duplicate, translation, convex-average and scaling nulls', 'Section 2, investigation 1'],
        ['State the convention in force, and what an absolute value changes', 'Section 2, figure 3, practice 2'],
        ['Build a loss class and use the exact one-half identity for binary hypotheses', 'Section 3, figure 5, practice 5'],
        ['Say what 1 − δ ranges over, and what "simultaneously" licenses', 'Section 3, figure 4'],
        ['Follow the pair-swap argument and locate where the factor 2 enters', 'Section 4, figure 6'],
        ['Derive the norm-ball best response and its two upper bounds', 'Section 5, figures 7–8, investigation 2'],
        ['Compute a kernel complexity from inner products alone', 'Section 5, figure 9, investigation 2'],
        ['Apply Massart and Sauer inside their stated ranges', 'Section 5, practice 4'],
        ['Distinguish a margin loss from a probability map, and use the right constant', 'Section 6, figure 11'],
        ['Assemble a margin bound term by term and see the scale null', 'Section 6, figure 12, investigation 3'],
        ['Account for K predeclared comparisons with a union bound', 'Section 6, investigation 3'],
        ['Correct a Monte Carlo estimate, and say which failure allowance it spends', 'Section 7, figure 13, practice 8'],
        ['Apply a declared selection rule before looking at assessment', 'Section 8, investigation 4'],
        ['Report a vacuous bound honestly beside a model that works', 'Section 8, figures 15–16, practice 7'],
      ]} />

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; each of these offers
      a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://cs.nyu.edu/~mohri/mls/lecture_3.pdf">Mohri&rsquo;s lecture slides</a> — the shortest route
        through the exact convention and the two high-probability bounds used here. Reconstruct the pair-swap proof
        alongside slides 6–9, then use the growth-function section to connect back to PAC and VC theory. Slides only;
        no recording was reviewed.</li>
      <li><a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">Understanding
        Machine Learning, chapter 26</a>, Shalev-Shwartz and Ben-David — a longer proof route: Rademacher calculus,
        contraction, Euclidean and <Math>{'\\ell_1'}</Math> classes, and SVM guarantees. Chapter 27 then introduces
        covering numbers. Keep each theorem&rsquo;s normalization and range assumptions together when comparing it with
        another source; this book&rsquo;s conventions are not identical to Mohri&rsquo;s.</li>
      <li><a href="https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/">Patrick Rebeschini&rsquo;s Oxford course</a>
        {' '}— links notes, slides and recordings for lecture 2, &ldquo;Maximal Inequalities and Rademacher
        Complexity,&rdquo; and lecture 3, &ldquo;Rademacher Complexity. Examples.&rdquo;
        The <a href="https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/material/lecture03.pdf">lecture 3 notes</a>
        {' '}are particularly useful for comparing <Math>{'\\ell_2'}</Math> and <Math>{'\\ell_1'}</Math> geometry. The
        course page lists recordings; their public playback access was not established, and no video was watched.</li>
    </ul></>}>
      <li><a href="https://jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf">Rademacher and Gaussian Complexities</a>,
        Bartlett and Mendelson, 2002 — the structural results and kernel applications. Note its definition uses an
        absolute value and a <Math>{'2/n'}</Math> normalization, so its constants are not interchangeable with the ones
        on this page.</li>
      <li><a href="https://arxiv.org/abs/math/0508275">Local Rademacher complexities</a>, Bartlett, Bousquet and
        Mendelson; <a href="https://proceedings.mlr.press/v40/Neyshabur15.pdf">Norm-based capacity control</a>,
        Neyshabur, Tomioka and Srebro; <a href="https://papers.neurips.cc/paper/7204-spectrally-normalized-margin-bounds-for-neural-networks.pdf">Spectrally-normalized
        margin bounds</a>, Bartlett, Foster and Telgarsky; <a href="https://arxiv.org/pdf/1703.11008">Computing
        nonvacuous generalization bounds</a>, Dziugaite and
        Roy; <a href="https://proceedings.mlr.press/v48/hardt16.pdf">Train faster, generalize better</a>, Hardt, Recht
        and Singer — the four deeper routes section 9 points at. Each changes a specific part of the analysis; start
        with the question you want answered rather than treating them as interchangeable certificates.</li>
      <li><a href={provenance.doi}>UCI Banknote Authentication</a>, {provenance.creator}, licensed{' '}
        <a href={provenance.licenseUrl}>{provenance.license}</a> — the actual observations. This page serves{' '}
        <a href={provenance.file} download>its own unchanged copy</a>, {provenance.bytes.toLocaleString('en-US')} bytes,
        SHA-256 <Code>{provenance.sha256}</Code>, beside its <a href={provenance.attribution}>attribution</a>, which
        records the extraction, the added columns and what the source record does not supply. It also
        serves <a href={provenance.calculationProgram} download>complexity_calculations.py</a> and{' '}
        <a href={provenance.experimentProgram} download>bounded_norm_experiment.py</a>, the two programs whose output
        appears above.</li>
    </Sources>

    <Prose>Next in this module is <a href="/learn/path/full-curriculum/ml-problem-formulation-baselines-data-leakage?module=classical-ml">ML
      Problem Formulation, Baselines &amp; Data Leakage</a>. It returns from these guarantees to choosing the
      prediction unit, target, baseline and information boundaries of a real project. The connection is direct: a
      sophisticated complexity calculation cannot repair a mislabeled target, a leaked feature or a test population
      different from the one named in the claim.</Prose>

    <Prose>Sections 1–7 use constructed inputs: the small sign tables are exactly enumerated, while section 7's
      Monte Carlo table records seeded estimates and their upper corrections. Both are reproducible by the program above.
      Section 8&rsquo;s numbers are measurements on the identified real dataset under one declared protocol, with
      the {experimentSettings.radii.length} budgets and {experimentSettings.rhoValues.length} margin thresholds fixed
      before the run and the {80} assessment rows scored after the selection was committed. The tables in figures 15
      and 16 are recomputed in your browser from the served rows and the recorded coefficients, and reproduce the
      program&rsquo;s own numbers exactly; they are not a second experiment. None of this is a benchmark, and none of
      it is a certificate about any future dataset.</Prose>
  </div>,
};

export default rademacherContent;
