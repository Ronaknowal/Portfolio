import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { BoundaryStripLab, FiniteWorldLab, WitnessLab } from '../../components/lesson-labs/PacLabs.jsx';
import {
  BoundCurveFigure, BoundaryStripFigure, CandidateBandFigure, EliminationFigure, GhostSampleFigure,
  GrowthFigure, HalfPlaneFigure, LearningCurveFigure, SimulationFigure, SineFigure, TwoLevelsFigure,
} from '../../components/lesson-labs/PacFigures.jsx';
import { asInput, fixed } from '../../components/lesson-labs/PacShared.jsx';
import { pacExamples } from '../pac-examples.js';
import { pacData } from '../pac-data.js';
import {
  candidateBandGeometry, finiteFamilyRadii, finiteRadius, finiteWorldProbability, fixtures, growthTable,
  intervalExperiment, intervalPatterns, realizableVcSampleBound, sauerSum, sineWitness, twoStripBound, vcRadius,
} from '../pac-models.js';

/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.sqrt` here would resolve to that component and yield
   undefined rather than a number. Every quantity on this page therefore comes
   from `pac-models.js` or from the recorded data, never from arithmetic typed
   into the prose. */

const bandFigure = candidateBandGeometry();
const worked = intervalExperiment(fixtures.interval);
const workedNearEdges = intervalExperiment(fixtures.intervalCloserEdges);
const workedNegatives = intervalExperiment(fixtures.intervalNegativeNull);
const workedEmpty = intervalExperiment(fixtures.intervalNoPositives);
const practiceFit = intervalExperiment(fixtures.intervalPractice);
const practiceWithPositive = intervalExperiment(fixtures.intervalPracticePositive);
const practiceWithNegative = intervalExperiment(fixtures.intervalPracticeNegativeNull);
const patternsOnThree = intervalPatterns(3);
const patternsOnFour = intervalPatterns(4);
/* Derived, not typed: the pattern the class cannot make is whatever is absent
   from the enumeration, and saying so in a literal would make the claim
   unfalsifiable if the enumeration ever changed. */
const allThreePointPatterns = Array.from({ length: 8 }, (_unused, code) =>
  [0, 1, 2].map(index => (code >> (2 - index)) & 1).join(''));
const realizedOnThree = new Set(patternsOnThree.map(pattern => pattern.join('')));
const missingOnThree = allThreePointPatterns.filter(pattern => !realizedOnThree.has(pattern));
const worldAtFour = finiteWorldProbability({ target: [0, 0, 1, 1], n: 4, epsilon: 0.25 });
const worldAtTwentyFour = finiteWorldProbability({ target: [0, 0, 1, 1], n: 24, epsilon: 0.25 });
const sine = sineWitness(fixtures.sineLabels);
const simulationRows = pacData.simulation.rows;
const curveRows = pacData.learningCurves.rows;
const curveLabels = pacData.learningCurves.labels;
const provenance = pacData.provenance;

const headings = [
  '1. Two different kinds of randomness',
  '2. The PAC contract, with its assumptions visible',
  '3. Why a finite class can be learned',
  '4. When zero error is impossible: uniform convergence and agnostic learning',
  '5. Infinite choices can still produce few label patterns',
  '6. The growth function and the price of searching',
  '7. What a VC guarantee actually says',
  '8. Compute the constructions and run an honest experiment',
  '9. Use capacity without turning it into a model-selection shortcut',
  '10. Deeper: parameter count, other tools, and computation',
  '11. Practice: state the claim before calculating',
  '12. What you can now do, and where it goes next',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Practice({ title, question, hint, children }) {
  return <section className="pac-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>Show the explained solution</summary>{children}</details>
  </section>;
}

const pacLearningContent = {
  title: 'PAC Learning & VC Dimension',
  readTime: '~55 min first pass · ~105 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot pac-lesson">
    <LessonIntro
      prerequisites={<>Probability of an event, independence, and the idea of an average. <Math>{'\\Pr(A)'}</Math> is
        how often <Math>{'A'}</Math> happens; independent draws are ones whose outcomes do not inform each
        other. Natural and base-2 logarithms both appear and are always named. The
        preceding <a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">Evaluation
        Metrics</a> lesson supplies the empirical 0–1 loss and the discipline of naming a denominator; hypothesis
        classes, shattering, growth functions and VC dimension are all introduced here.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A classifier can get every training example right and still fail on the next one. That is easy to say. The
      harder question is <strong>what would make success on a finite sample trustworthy?</strong> You will build
      a four-input world whose failure probability is an exact fraction — {worldAtFour.failureExact} at four
      draws — fit an interval whose true risk you can calculate rather than estimate, find the one labelling of
      three points that no interval can produce, watch a distribution-free bound come out
      at {fixed(vcRadius(2, 100, 0.05), 6)} and say clearly why a number above 1 is not a disaster, and finish
      on {provenance.subsetRows} real banknote measurements where a model that fits all 80 training labels loses
      to one that does not. Every investigation updates its topic-specific results from valid control changes, with no expected-answer input.
    </LessonIntro>

    <div className="pac-route"><Prose><strong>First pass.</strong> Read sections 1–7 and do practice 1–7. That
      route gets you both meanings of “probably approximately correct”, the union-bound argument in full, the
      difference between a gap and an excess risk, the one labelling intervals cannot make, and what a VC bound
      does and does not promise. Run the three short programs on the way and do the small constructions by hand
      before opening them. Section 8 turns the whole thing into reproducible experiments with real data;
      sections 9 and 10 are deeper branches — 9 on using capacity in model choice without cheating, 10 on
      parameter counts, Rademacher complexity, stability, PAC-Bayes and computation.</Prose></div>

    <Prose>Imagine learning which temperature range makes a simple laboratory indicator turn on. You observe a
      few temperatures and their on/off outcomes, then choose an interval that fits them. There are infinitely
      many possible interval endpoints, but the observations constrain those intervals in a highly organized
      way. Compare that with a lookup table allowed to assign an arbitrary answer at every unseen temperature.
      Both can fit the sample; their ability to make arbitrary choices beyond it is very different.</Prose>
    <Prose><strong>PAC learning</strong> gives a precise language for the accuracy a learning procedure can
      achieve from random examples. <strong>VC dimension</strong> measures one kind of flexibility of a binary
      prediction family. Together they connect the examples we observe, the choices a learner is allowed to
      make, and the error it may make on new examples. Our core loss is deliberately simple: 0 for a correct
      binary prediction and 1 for an incorrect one.</Prose>

    <Callout title="Which numbers here are theorems, which are constructions, and which are measurements">
      Three kinds of quantity appear on this page and are never mixed. <strong>Exact constructions</strong> —
      pattern counts, the four-input probabilities, the sine witnesses — are integer or rational arithmetic with
      no tolerance. <strong>Computed theorem expressions</strong> — every radius and sufficient sample size — are
      formulas evaluated at stated arguments; a bound is not a benchmark. <strong>Measurements</strong> appear in
      exactly two places: the retained simulation of section 8, whose failure counts are Monte Carlo estimates,
      and the banknote development curves, which are attributed there. Each figure states which kind it is.
      This caution is stated once; the rest of the lesson refers back to it rather than repeating it.
    </Callout>

    {/* ============================================================ §1 */}
    <H2 id={headingId(headings[0])}>{headings[0]}</H2>
    <Prose>Let <Math>{'x'}</Math> be an input and <Math>{'y'}</Math> its observed class, either 0 or 1.
      A <strong>hypothesis</strong> <Math>{'h'}</Math> is one prediction rule; a <strong>hypothesis
      class</strong> <Math>{'H'}</Math> is the collection of rules the learner is allowed to choose from. A
      threshold, an interval, and an arbitrary lookup table define different classes.</Prose>
    <Prose>The training sample is <Math>{'S=((x_1,y_1),\\dots,(x_n,y_n))'}</Math>. Assume for now that these
      pairs are independent draws from the same fixed distribution <Math>{'D'}</Math>. A learning
      algorithm <Math>{'A'}</Math> takes the sample and returns <Math>{'h_S=A(S)'}</Math>.</Prose>
    <Prose>For a fixed rule <Math>{'h'}</Math>, its <strong>population risk</strong> is the chance of making an
      error on a fresh draw:</Prose>
    <MathBlock>{'R_D(h)=\\Pr_{(X,Y)\\sim D}\\bigl(h(X)\\ne Y\\bigr).'}</MathBlock>
    <Prose>Its <strong>empirical risk</strong> is the fraction of observed sample errors:</Prose>
    <MathBlock>{'\\hat R_S(h)=\\frac1n\\sum_{i=1}^{n}\\mathbf1[h(x_i)\\ne y_i].'}</MathBlock>
    <Prose>The indicator is 1 when the bracketed statement is true and 0 otherwise. Three errors among twenty
      examples give empirical risk 3/20. Population risk averages over the underlying distribution, which
      normally is not known exactly.</Prose>
    <Prose>There are two probability levels. Once a model is fitted, its population risk concerns <strong>a new
      example</strong>. Before training, different random samples can produce different fitted models and
      therefore different risks. PAC's confidence concerns <strong>that training-sample draw</strong>.</Prose>

    <TwoLevelsFigure />

    <Prose>For example, “error at most 0.1 with probability at least 0.95” means that the learning procedure
      produces a rule with population error no greater than 10% on at least 95% of training draws under the
      stated setting. It does not say 95% of predictions are correct, nor that a particular unlucky sample can
      be identified from its training score. A randomized learner also contributes its internal randomness to
      the outer probability when the guarantee includes it.</Prose>

    {/* ============================================================ §2 */}
    <H2 id={headingId(headings[1])}>{headings[1]}</H2>
    <Prose><strong>Probably Approximately Correct</strong> separates two tolerances. <Math>{'\\varepsilon'}</Math> is
      how much population error is acceptable: the “approximately correct”
      part. <Math>{'\\delta'}</Math> bounds how often the learning procedure may fail to achieve that target
      over its random training input: the “probably” part.</Prose>
    <Prose>In <strong>realizable</strong> binary PAC learning, labels come from some target
      rule <Math>{'h^*'}</Math> in <Math>{'H'}</Math>. Therefore a rule with zero population classification
      error exists inside the chosen class. A learner PAC-learns <Math>{'H'}</Math> if, for
      every <Math>{'\\varepsilon'}</Math> and <Math>{'\\delta'}</Math> in <Math>{'(0,1)'}</Math>, there is a
      sufficient sample size <Math>{'n_H(\\varepsilon,\\delta)'}</Math> such that, for every allowed input
      distribution and every target <Math>{'h^*\\in H'}</Math>, using at least that many independent examples
      gives</Prose>
    <MathBlock>{'\\Pr_S\\bigl(R_D(A(S))\\le\\varepsilon\\bigr)\\ge1-\\delta.'}</MathBlock>
    <Prose>The sample-size guarantee cannot be selected afterward to suit the particular unknown distribution or
      target. That is the <strong>distribution-free</strong> part of this definition. It does not mean that the
      realized error is the same for all distributions.</Prose>
    <Prose>A <strong>consistent</strong> learner chooses a rule that makes no training errors.
      An <strong>empirical risk minimizer</strong>, or ERM, chooses a rule with the smallest training error
      in <Math>{'H'}</Math>. Under realizability, an exact ERM is consistent because <Math>{'h^*'}</Math> itself
      makes no errors. Zero training error is a property of the fit; the PAC argument is what connects it to
      population error.</Prose>
    <Prose>Realizability can be a useful model for learning an exact threshold or a deterministic labeling rule.
      It is an assumption to examine, not something certified by fitting a flexible model until training error
      vanishes. In noisy or misspecified problems, no rule in <Math>{'H'}</Math> may have zero risk. We
      introduce the <strong>agnostic</strong> contract in section 4.</Prose>

    <LessonTable caption="Four distinctions this lesson keeps apart. Each is stated here and referred back to."
      headers={['Distinction', 'What it separates']}
      rows={[
        ['Sufficient versus minimum', 'Every sample size on this page is an upper bound that suffices under a stated theorem. None is a claim that fewer examples cannot work.'],
        ['Realizable versus agnostic', 'Realizable assumes some class member has zero risk. Agnostic compares against the best member and bounds the excess above it.'],
        ['Statistical versus computational', 'Finite VC dimension characterizes when a suitable sample size and learning rule exist. It supplies no fast optimization algorithm.'],
        ['Gap versus excess risk', 'A uniform bound limits |population − empirical| for every member. ERM pays that radius twice, so an excess-risk promise is the weaker one.'],
      ]} />

    <Prose>That third row matters enough to state in its own words. <strong>Statistical
      learnability</strong> asks whether a suitable sample size and learning rule exist. <strong>Efficient
      learning</strong> additionally constrains computation and representation size, usually by polynomial
      bounds. Finite VC dimension characterizes the standard statistical binary setting under suitable
      regularity conditions; it does not automatically provide a fast optimization algorithm. The original
      Blumer–Ehrenfeucht–Haussler–Warmuth paper explicitly separates these
      questions. <a href="https://mwarmuth.bitbucket.io/pubs/J14.pdf">Original paper, introduction and sections
      2–3</a>.</Prose>

    {/* ============================================================ §3 */}
    <H2 id={headingId(headings[2])}>{headings[2]}</H2>
    <Prose>Suppose <Math>{'H'}</Math> contains <Math>{'K'}</Math> rules fixed before we see the sample, and the
      setting is realizable. Take one <strong>bad</strong> rule <Math>{'h'}</Math> whose true error
      exceeds <Math>{'\\varepsilon'}</Math>. It can still make zero training errors if every sampled example
      happens to miss its error region.</Prose>
    <Prose>For one independent example, the probability of missing that region is at
      most <Math>{'1-\\varepsilon'}</Math>. For <Math>{'n'}</Math> independent examples, it is at most</Prose>
    <MathBlock>{'(1-\\varepsilon)^n\\le e^{-n\\varepsilon}.'}</MathBlock>
    <Prose>This is already the main intuition: repeated independent observations make it difficult for a
      genuinely bad fixed rule to keep hiding all its mistakes.</Prose>
    <Prose>But the learner chooses among <Math>{'K'}</Math> rules. We must protect
      against <strong>any</strong> bad member surviving. The union bound says the probability of at least one
      event is no larger than the sum of the individual event probabilities:</Prose>
    <MathBlock>{'\\begin{gathered}\\Pr(\\text{some bad }h\\in H\\\\\\text{fits every sample label})\\le K e^{-n\\varepsilon}.\\end{gathered}'}</MathBlock>
    <Prose>No independence between the <Math>{'K'}</Math> rules' failure events is required. They can make
      overlapping mistakes on the same examples. Independence of the sampled observations was used earlier to
      multiply the per-example probabilities; these are different assumptions.</Prose>
    <Prose>If the right side is at most <Math>{'\\delta'}</Math>, then every consistent output is approximately
      correct on the protected event. Solving gives the sufficient sample condition</Prose>
    <MathBlock>{'n\\ge\\frac{\\ln K+\\ln(1/\\delta)}{\\varepsilon}.'}</MathBlock>
    <Prose>Here <Math>{'\\ln'}</Math> means natural logarithm.
      For <Math>{'K=32'}</Math>, <Math>{'\\varepsilon=.05'}</Math> and <Math>{'\\delta=.01'}</Math>, the right
      side is about 161.42, so {finiteFamilyRadii.sufficientRealizable} examples suffice under this particular
      bound. This is an <strong>upper bound on sufficient sample size</strong>, not a claim that 161 examples
      cannot work or that every real dataset needs {finiteFamilyRadii.sufficientRealizable}.</Prose>

    <EliminationFigure />

    <H3>An exact four-point world</H3>
    <Prose>Take four equally likely inputs 0, 1, 2, 3. The true labels are [0, 0, 1, 1].
      Let <Math>{'H'}</Math> contain all 16 binary labelings of these four points. Our fixed learner returns the
      lexicographically first consistent label vector: it uses observed labels at seen points and predicts 0 at
      unseen points.</Prose>
    <Prose>With <Math>{'\\varepsilon=.25'}</Math>, the learner fails only if it misses <strong>both</strong> positive
      inputs. Seeing even one positive leaves at most one wrong label, risk .25, which satisfies “at most ε.”
      Each draw misses both positives with probability 1/2, so</Prose>
    <MathBlock>{'\\Pr\\bigl(R_D(h_S)>.25\\bigr)=(1/2)^n.'}</MathBlock>
    <Prose>At <Math>{'n=4'}</Math>, the exact failure probability
      is {worldAtFour.failureExact} = {fixed(worldAtFour.failure, 6)}. The generic finite-class upper bound
      is <Math>{'16e^{-1}'}</Math> = {fixed(worldAtFour.bound.raw, 6)}, larger than 1, so after clipping to the
      trivial probability limit it says only “at most 1.” The loose bound did not make a false prediction; it
      simply supplied no useful numerical restriction at that sample size.</Prose>
    <Prose>At <Math>{'n=24'}</Math>, the exact failure probability is {worldAtTwentyFour.failureExact}. The
      finite bound is <Math>{'16e^{-6}\\approx'}</Math>{' '}{fixed(worldAtTwentyFour.bound.raw, 6)}, sufficient
      for <Math>{'\\delta=.05'}</Math> and still much larger than the exact failure probability for this learner
      in this distribution. Exact behavior, a general sufficient bound, and the smallest possible sample
      requirement are three different quantities.</Prose>

    <FiniteWorldLab />

    {/* ============================================================ §4 */}
    <H2 id={headingId(headings[3])}>{headings[3]}</H2>
    <Prose>Suppose a sensor sometimes gives an ambiguous result even at exactly the same input. Or suppose the
      true positive region consists of two separated intervals while <Math>{'H'}</Math> allows only one. We
      should then compare the learner with the best achievable risk <strong>inside <Math>{'H'}</Math></strong>,
      rather than demand absolute error approaching zero.</Prose>
    <Prose>An agnostic PAC guarantee has the form</Prose>
    <MathBlock>{'R_D(h_S)\\le\\inf_{h\\in H}R_D(h)+\\varepsilon'}</MathBlock>
    <Prose>with probability at least <Math>{'1-\\delta'}</Math>. The infimum is the best risk attainable or
      approached within the class. The guarantee concerns <strong>excess risk above that reference</strong>. If
      the best class member has risk .12, a guarantee of excess risk at most .03 means a risk ceiling .15,
      not .03.</Prose>
    <Prose>For a fixed rule with independent losses in <Math>{'[0,1]'}</Math>, Hoeffding's inequality
      gives</Prose>
    <MathBlock>{'\\Pr\\bigl(|R(h)-\\hat R(h)|>r\\bigr)\\le2e^{-2nr^2}.'}</MathBlock>
    <Prose>For <Math>{'K'}</Math> fixed candidate rules, allocate failure
      probability <Math>{'\\delta/K'}</Math> to each and apply the union bound:</Prose>
    <MathBlock>{'\\begin{gathered}\\text{With probability }\\ge1-\\delta,\\\\[4pt]'
      + '\\text{for every }h\\in H:\\\\[4pt]|R(h)-\\hat R(h)|\\le r_K,\\\\[4pt]'
      + 'r_K=\\sqrt{\\tfrac{\\ln(2K/\\delta)}{2n}}.\\end{gathered}'}</MathBlock>
    <Prose>This is <strong>uniform convergence</strong>: one event protects every member simultaneously. It
      covers a member selected after seeing the evaluation scores, provided it belongs to that protected
      family. The Concentration Inequalities lesson developed the individual event and the finite union; here
      uniform protection is what allows selection.</Prose>
    <Prose>For <Math>{'K=25'}</Math>, <Math>{'n=500'}</Math> and <Math>{'\\delta=.05'}</Math>,
      {' '}<Math>{'r_K\\approx'}</Math>{' '}{fixed(finiteFamilyRadii.selection, 8)}. For a single fixed rule the
      radius is {fixed(finiteFamilyRadii.single, 8)}. Selecting among 25 incurs a larger allowance. The models
      can all be evaluated on the same 500 examples; their prediction errors need not be independent of one
      another. If they were fitted on a separate training sample, condition on that training sample before
      applying the held-out bound.</Prose>
    <Prose>The family must be fixed independently of the evaluation outcomes, or a larger appropriately
      protected class must contain all possible candidates. Keeping only the final three models after inventing
      and testing hundreds of outcome-dependent alternatives does not justify paying
      for <Math>{'K=3'}</Math> — which here would be {fixed(finiteRadius(3, 500, 0.05), 8)}, a radius that was
      never earned. A final independent assessment or an analysis covering the selection procedure is
      needed.</Prose>

    <CandidateBandFigure />

    <H3>Why ERM pays the radius twice</H3>
    <Prose>On the uniform event, compare an ERM <Math>{'h_S'}</Math> with a best class
      member <Math>{'h^*'}</Math> when the minimum exists:</Prose>
    <MathBlock>{'\\begin{gathered}R(h_S)\\le\\hat R(h_S)+r_K\\\\\\le\\hat R(h^*)+r_K\\\\\\le R(h^*)+2r_K.\\end{gathered}'}</MathBlock>
    <Prose>The first step moves from the chosen rule's empirical to population risk. The middle step uses ERM's
      defining property. The last moves the comparator's empirical risk back to its population risk. If a
      minimum does not exist, compare with increasingly near-optimal rules and take the infimum. If
      optimization stops <Math>{'\\eta'}</Math> above the best empirical risk, add <Math>{'\\eta'}</Math> to the
      result.</Prose>
    <Prose>To make <Math>{'2r_K\\le\\varepsilon'}</Math>, a sufficient condition
      is <Math>{'n\\ge2\\ln(2K/\\delta)/\\varepsilon^2'}</Math>. Contrast
      the <Math>{'\\varepsilon^{-2}'}</Math> dependence with the <Math>{'\\varepsilon^{-1}'}</Math> finite-class
      realizable argument. They concern different promises. The asymptotic exponents do not say a particular
      noisy dataset needs exactly 100 times as many examples at <Math>{'\\varepsilon=.01'}</Math>; constants,
      class structure, noise assumptions and the reference error matter.</Prose>

    {/* ============================================================ §5 */}
    <H2 id={headingId(headings[4])}>{headings[4]}</H2>
    <Prose>An interval's endpoints are real numbers, so there are infinitely many interval rules.
      Putting <Math>{'K=\\infty'}</Math> into the finite union bound is useless. Yet on a finite ordered set of
      points, many different intervals make <strong>exactly the same predictions</strong>.</Prose>
    <Prose>Take <Math>{'x_1=.2'}</Math>, <Math>{'x_2=.5'}</Math>, <Math>{'x_3=.8'}</Math>. An interval can label
      a consecutive block of points positive, or label all negative. There
      are {patternsOnThree.length} patterns:</Prose>
    <LessonTable caption="Every labelling one closed interval can produce on (.2, .5, .8), with a witness"
      headers={['Labels at (.2, .5, .8)', 'One witnessing interval']}
      rows={patternsOnThree.map(pattern => {
        const points = [0.2, 0.5, 0.8].filter((_x, index) => pattern[index] === 1);
        return [
          pattern.join(''),
          points.length ? `[${points[0]}, ${points[points.length - 1]}]` : 'Empty positive region',
        ];
      })} />
    <Prose>The missing pattern is {missingOnThree[0]}. If an interval contains .2 and .8, it contains .5 as
      well. Trying more random endpoints will not help. The obstruction is structural.</Prose>
    <Prose>A class <strong>shatters</strong> a set of points if it realizes every possible binary labeling on
      that same set. Three points have <Math>{'2^3=8'}</Math> potential labelings. Intervals realize
      only {patternsOnThree.length}, so they do not shatter this set.</Prose>
    <Prose>The <strong>VC dimension</strong> is the largest size of a set that the class can shatter, or
      infinity if arbitrarily large finite sets can be shattered. Its quantifiers matter: find <strong>some</strong> set
      of <Math>{'d'}</Math> points; show that <strong>every</strong> binary labeling of that set has a witness
      in <Math>{'H'}</Math>; and to prove the dimension is exactly <Math>{'d'}</Math>,
      show <strong>every</strong> set of <Math>{'d+1'}</Math> points has at least one impossible
      labeling.</Prose>
    <Prose>The witnessing rule can change when the requested labeling changes. One fixed rule is not expected to
      realize every labeling simultaneously. Conversely, showing that one unfortunate point configuration cannot
      be shattered does not upper-bound the entire class's VC dimension.</Prose>

    <H3>Thresholds, intervals and half-planes</H3>
    <Prose>For increasing thresholds <Math>{'h_a(x)=\\mathbf1[x\\ge a]'}</Math>, one point can receive either
      label. For <Math>{'x_1<x_2'}</Math>, the labeling 10 is impossible: <Math>{'a\\le x_1'}</Math> implies
      {' '}<Math>{'a\\le x_2'}</Math>. Thresholds therefore have VC dimension 1. Allowing either threshold
      direction defines a different class; the convention is part of the statement.</Prose>
    <Prose>Intervals shatter any two distinct points: neither, left only, right only, or both. For any three
      ordered points, 101 is impossible. Their VC dimension is 2. Including the empty positive set makes the
      all-negative rule explicit and gives a consistent default when no positives are observed.</Prose>
    <Prose>An affine half-plane in the plane predicts 1
      when <Math>{'w_1x_1+w_2x_2+b\\ge0'}</Math>. Three noncollinear points can be shattered: the
      all-positive and all-negative cases use constant signs, and a line can separate each chosen vertex from
      the other two. Complementing a separating line gives the complementary patterns. This supplies all 8
      labelings.</Prose>
    <Prose>No four-point set can be shattered. If one point lies in the triangle formed by the other three,
      making those three positive forces the interior point positive too. If the four points form a convex
      quadrilateral, alternate the labels around its boundary: the positive diagonal and negative diagonal
      intersect, so no line can strictly separate the two groups. Degenerate collinear configurations already
      contain an ordered triple with an impossible alternating pattern. Thus the VC dimension of affine
      half-planes is 3.</Prose>

    <HalfPlaneFigure />

    <Prose>The program checks all four configurations drawn above: 8 feasible triangle patterns, 14 feasible
      square patterns, 6 feasible collinear-triple patterns and 14 for the interior-point configuration, using
      linear-program feasibility and verified signed margins. Those finite
      numerical checks support the drawings. The geometry above supplies the universal upper-bound reasoning; an
      optimizer failing to find a separator by random search would not be a proof.</Prose>

    <WitnessLab />

    {/* ============================================================ §6 */}
    <H2 id={headingId(headings[5])}>{headings[5]}</H2>
    <Prose>The <strong>growth function</strong> <Math>{'\\Pi_H(n)'}</Math> is the maximum number of distinct
      label patterns <Math>{'H'}</Math> can realize on any <Math>{'n'}</Math>-point set. It is not the number of
      parameter settings, and a count on one arbitrary configuration is only a lower bound on that
      maximum.</Prose>
    <Prose>For increasing thresholds on <Math>{'n'}</Math> distinct ordered points, there
      are <Math>{'n+1'}</Math> patterns: choose where the positive suffix starts, including before all points or
      after all points. For intervals, count consecutive positive blocks. There are <Math>{'n'}</Math> choices
      for a one-point block, <Math>{'n-1'}</Math> for a two-point block, and so on, plus the all-negative
      pattern:</Prose>
    <MathBlock>{'\\begin{gathered}\\Pi_{\\text{intervals}}(n)\\\\[4pt]'
      + '=1+n+(n-1)+\\cdots+1\\\\[4pt]=1+\\tfrac{n(n+1)}2.\\end{gathered}'}</MathBlock>

    <GrowthFigure />

    <Prose>At <Math>{'n=5'}</Math>, intervals realize {growthTable[4].intervals}/{growthTable[4].allBinary} of
      all patterns. Read the numerator and denominator together: {growthTable[4].intervals} realizable patterns
      out of {growthTable[4].allBinary} possible ones. The finite-dimensional class has polynomially many
      patterns; their fraction among all <Math>{'2^n'}</Math> patterns becomes small because the denominator is
      exponential.</Prose>
    <Prose><strong>Sauer's lemma</strong> generalizes this counting fact. If <Math>{'\\mathrm{VC}(H)=d'}</Math>,
      then</Prose>
    <MathBlock>{'\\Pi_H(n)\\le\\sum_{i=0}^{\\min(d,n)}\\binom ni.'}</MathBlock>
    <Prose>For <Math>{'1\\le d\\le n'}</Math>, this is at most <Math>{'(en/d)^d'}</Math>. For <Math>{'d=0'}</Math> the
      class realizes at most one pattern and the expression with division by <Math>{'d'}</Math> should not be
      used. For <Math>{'n\\le d'}</Math>, the maximum is <Math>{'2^n'}</Math>. The bound is an upper bound, not
      a claim that every VC-<Math>{'d'}</Math> class attains it.</Prose>
    <Prose>The recurrence behind the result is instructive. Remove one point from a set. Some patterns on the
      remaining points admit only one choice for the removed point; others admit both. Count every distinct
      restricted pattern once using capacity <Math>{'d'}</Math>, then add one extra copy for each pattern
      admitting both labels. That second collection has capacity at most <Math>{'d-1'}</Math>, because
      shattering <Math>{'d'}</Math> remaining points with both choices would shatter <Math>{'d+1'}</Math> original
      points. This gives the same recurrence as binomial sums. It explains why a missing ability to realize all
      patterns constrains the later growth.</Prose>
    <Prose>For <Math>{'n=100'}</Math>, <Math>{'d=5'}</Math>, the exact binomial sum
      is {sauerSum(100, 5).toLocaleString('en-US')}, compared
      with <Math>{'2^{100}\\approx1.27\\times10^{30}'}</Math>. This upper bound alone does not establish that
      100 examples yield a useful error guarantee; it must still enter a probabilistic
      analysis. <a href="https://web.uvic.ca/~nmehta/ml_theory_fall2021/lecture12.pdf">Mehta's lecture notes,
      sections 1–4</a>.</Prose>

    <H3>Why we cannot simply replace K with the observed pattern count</H3>
    <Prose>The patterns realized on the training inputs are themselves sample-dependent. Plugging their observed
      number into a fixed-family bound without further argument skips the reason the theorem works.</Prose>
    <Prose>The usual proof introduces an independent <strong>ghost sample</strong> used only in the analysis. It
      relates population-versus-sample discrepancies to discrepancies between two samples, then controls the
      finitely many patterns on their combined inputs. Random exchanges between the samples and concentration
      make the counting argument legitimate. The ghost sample is a proof device; a learner need not secretly
      obtain a second labeled dataset to run ERM.</Prose>

    <GhostSampleFigure />

    {/* ============================================================ §7 */}
    <H2 id={headingId(headings[6])}>{headings[6]}</H2>
    <Prose>For well-behaved binary hypothesis classes with finite VC dimension <Math>{'d'}</Math>, uniform
      convergence gives distribution-free statistical learning guarantees. “Well-behaved” includes the
      measurability conditions needed for the relevant random events; the ordinary finite, threshold, interval
      and half-plane examples here satisfy the standard conditions. The original theorem states this
      qualification explicitly. It is not a license to extend the result to arbitrary nonmeasurable
      constructions. <a href="https://mwarmuth.bitbucket.io/pubs/J14.pdf">Blumer and colleagues, Theorem 2.1 and
      Appendix A1</a>.</Prose>
    <Prose>One deliberately conservative explicit uniform bound, with <Math>{'n\\ge d\\ge1'}</Math> and
      {' '}<Math>{'\\delta\\in(0,1)'}</Math>, is</Prose>
    <MathBlock>{'\\begin{gathered}\\text{With probability }\\ge1-\\delta,\\\\[4pt]'
      + '\\text{for every }h\\in H:\\\\[4pt]|R(h)-\\hat R_S(h)|\\le r,\\\\[4pt]'
      + 'r=\\sqrt{\\tfrac{32}{n}\\bigl(d\\ln\\tfrac{en}{d}+\\ln\\tfrac8\\delta\\bigr)}.\\end{gathered}'}</MathBlock>
    <Prose>This is the convention used in the program and in the bound curves below. Sharper inequalities can
      use different constants, so identify the stated bound before comparing numerical values. Applying the
      result to 0–1 losses is valid because fixed labels simply flip the corresponding prediction bits and do
      not increase the maximum number of
      patterns. <a href="https://web.uvic.ca/~nmehta/ml_theory_fall2021/lecture12.pdf">Mehta, section 4</a>.</Prose>

    <BoundCurveFigure />

    <Prose>A 0–1 risk gap is already bounded by 1. A radius above 1 is <strong>vacuous</strong> for that
      comparison: it adds nothing to the trivial range restriction. It is not evidence that the actual error
      exceeds 1 or that learning is impossible. For <Math>{'d=2'}</Math>, <Math>{'\\delta=.05'}</Math>, the raw
      radius falls from {fixed(vcRadius(2, 100, 0.05), 6)} at {(100).toLocaleString('en-US')} examples
      to {fixed(vcRadius(2, 100000, 0.05), 6)} at {(100000).toLocaleString('en-US')}.</Prose>
    <Prose>For agnostic ERM, apply the earlier three-step argument to get excess risk at most twice the uniform
      radius. In particular, a gap bound of .1 is not automatically an excess-risk bound of .1. With sharper
      analyses, the familiar distribution-free agnostic sample scale is proportional
      to <Math>{'(d+\\ln(1/\\delta))/\\varepsilon^2'}</Math> up to constants; the displayed elementary VC route
      retains an additional logarithmic factor. That sharper sample scale and our conservative displayed bound
      are different results. <a href="https://vatsalsharan.github.io/fall23/lec3.pdf">Sharan, lecture 3, page 3,
      note 1</a>.</Prose>
    <Prose>Realizability permits stronger bounds because a surviving rule must avoid every training error. A
      classical explicit sufficient condition from Blumer and colleagues asks the sample size to satisfy both
      of these at once — equivalently, to be at least the larger of the two:</Prose>
    {/* Written as two conditions rather than one max, because `max{A, B}` on one
        line overflows a 320 px column and a bracket pair cannot be split across
        rows. `n >= max{A, B}` and `n >= A and n >= B` say the same thing. */}
    <MathBlock>{'\\begin{gathered}n\\ge\\tfrac4\\varepsilon\\log_2\\tfrac2\\delta'
      + '\\\\[4pt]\\text{and}\\quad n\\ge\\tfrac{8d}\\varepsilon\\log_2\\tfrac{13}\\varepsilon.\\end{gathered}'}</MathBlock>
    <Prose>The logarithms in this formula are <strong>base 2</strong>, as in that paper. For
      intervals <Math>{'d=2'}</Math>, <Math>{'\\delta=.05'}</Math>, the rounded-up sufficient sizes
      at <Math>{'\\varepsilon=.2,.1,.05'}</Math> are {realizableVcSampleBound(2, 0.2, 0.05).toLocaleString('en-US')},
      {' '}{realizableVcSampleBound(2, 0.1, 0.05).toLocaleString('en-US')} and
      {' '}{realizableVcSampleBound(2, 0.05, 0.05).toLocaleString('en-US')}. These conservative general-class
      guarantees differ from a bound exploiting the particular interval-learning algorithm or a particular input
      distribution.</Prose>
    <Prose>The two main lessons are structural. Finite VC dimension prevents unrestricted fitting of all
      sufficiently large binary patterns, enabling statistical learning. Infinite VC dimension prevents a
      uniform distribution-free sample guarantee in this standard binary setting. It does <strong>not</strong> mean
      that no particular distribution or target in an infinite-VC family can be learned; alternative assumptions
      or target-dependent guarantees are different questions.</Prose>
    <Prose>There is a useful connection to ensembles here. Later work removes the classical extra logarithmic
      factor in realizable sample complexity using carefully constructed votes of consistent learners. A vote
      can lie outside the original class, so a result about such a learner is not automatically a result about
      every ERM that must return a member of <Math>{'H'}</Math>. Hanneke's construction and Larsen's later
      bagging analysis connect abstract sample efficiency to combining fits on subsamples. They do not certify
      arbitrary random-forest defaults or noisy applications under a realizable
      theorem. <a href="https://jmlr.org/papers/volume17/15-389/15-389.pdf">Hanneke's optimal-learning
      result</a>, <a href="https://proceedings.mlr.press/v195/larsen23a.html">Larsen's bagging result</a>.</Prose>

    <H3>The unseen-label argument behind the limitation</H3>
    <Prose>If a class shatters a large set, imagine a distribution supported on that set with labels chosen
      independently across its points. A training sample reveals only some labels. At an unobserved point, both
      labels are still compatible with what was seen; without additional structure, no learner can infer which
      one was chosen better than chance averaged over these targets.</Prose>
    <Prose>For <Math>{'d'}</Math> equally likely shattered points and at most <Math>{'n'}</Math> distinct
      observed points, averaging over target labelings leaves an expected error of at
      least <Math>{'(d-n)/(2d)'}</Math>. This is an intuition-building lower-bound step, not a complete
      high-probability sample-complexity theorem. It explains why an arbitrarily large shattered set can defeat
      any proposed fixed sample size. The full lower-bound argument converts this remaining uncertainty into the
      appropriate failure probability.</Prose>

    {/* ============================================================ §8 */}
    <H2 id={headingId(headings[7])}>{headings[7]}</H2>
    <Prose>Three short programs below show the mechanisms; each one's algorithm is lifted byte for byte from the
      complete downloadable program, so nothing on the page can drift away from what actually runs. Both
      complete programs are offered whole at the end of this section.</Prose>

    <Program example={pacExamples.patterns} />
    <Prose>The seven strings are every labelling one interval can make on three ordered points, and 101 is not
      among them. That is a fact about the class, not about how long the search ran.</Prose>

    <Program example={pacExamples['finite-world']} />
    <Prose>Each printed row is the exact failure fraction, then the raw union bound, then the same bound clipped
      to a probability. Read the first two rows: the bound is above 12 and above 9, so the clipped column
      says 1 and the bound restricts nothing. By <Math>{'n=24'}</Math> the exact probability
      is {worldAtTwentyFour.failureExact} while the bound is {fixed(worldAtTwentyFour.bound.raw, 6)} — both
      correct, five orders of magnitude apart. The last line is the all-zero target: the same sixteen rules, the
      same learner, and failure probability exactly 0. Class size alone does not determine task
      difficulty.</Prose>

    <H3>A fitted interval whose true risk we can calculate</H3>
    <Prose>Let <Math>{'X'}</Math> be uniform on <Math>{'[0,1]'}</Math> and the target turn on exactly
      inside <Math>{'[.3,.7]'}</Math>. The learner returns the smallest closed interval containing all observed
      positive inputs. If no positive input is observed, it returns the empty positive region. This default is
      essential: inventing a central positive interval would not necessarily be consistent with an all-negative
      sample.</Prose>
    <Prose>For observations [{worked.points.map(asInput).join(', ')}], the positive inputs
      are {worked.positives.map(asInput).join(', ')}, so the learned interval
      is [{worked.interval[0]}, {worked.interval[1]}]. Training error is {fixed(worked.empiricalRisk, 6)}. It
      misses [.3, .35) and (.65, .7], total length {fixed(worked.segmentTotal, 6)}. Under the uniform
      distribution, length equals probability, so its exact population risk is {fixed(worked.risk, 6)}.</Prose>
    <Prose>Add observations .31 and .69: the learned interval expands
      to [{workedNearEdges.interval[0]}, {workedNearEdges.interval[1]}] and exact risk falls
      to {fixed(workedNearEdges.risk, 6)}. Add only negative observations .01 and .99 instead: the interval and
      risk stay unchanged at [{workedNegatives.interval[0]}, {workedNegatives.interval[1]}]
      and {fixed(workedNegatives.risk, 6)}. With observations [{workedEmpty.points.map(asInput).join(', ')}],
      there are no positives; the empty rule is consistent and its true risk
      is {fixed(workedEmpty.risk, 6)}.</Prose>

    <Program example={pacExamples['interval-risk']} />

    <BoundaryStripFigure />

    <Prose>A useful algorithm-specific bound comes from the two interior boundary strips, each of
      width <Math>{'\\varepsilon/2'}</Math>, for <Math>{'0<\\varepsilon\\le.4'}</Math>. If at least one sample
      lands in each strip, the tight interval misses at most <Math>{'\\varepsilon'}</Math> of probability. Each
      strip is missed with probability <Math>{'(1-\\varepsilon/2)^n'}</Math>; union-bound the two events:</Prose>
    <MathBlock>{'\\begin{gathered}\\Pr\\bigl(R(h_S)>\\varepsilon\\bigr)\\\\[4pt]'
      + '\\le\\min\\{1,\\,2(1-\\varepsilon/2)^n\\}.\\end{gathered}'}</MathBlock>
    <Prose>This explains <strong>how</strong> observations near both edges control error. It is a sufficient
      event: a sample can achieve small error even if one chosen strip is empty. For this derivation we used the
      specified uniform interval world, not an arbitrary banknote distribution.</Prose>

    <BoundaryStripLab />

    <H3>What a thousand repeated draws do and do not show</H3>
    <Prose>Now repeat independent training draws. At each <Math>{'n'}</Math>, the author
      ran {simulationRows[0].repetitions.toLocaleString('en-US')} samples with random generator
      seed {pacData.simulation.seed} and <Math>{'\\varepsilon=.1'}</Math>.</Prose>

    <SimulationFigure />

    <Prose>The true risk for each fitted interval is calculated analytically, not estimated from a second large
      Monte Carlo test set. The failure fractions across training runs are simulation estimates. Zero observed
      failures at <Math>{'n=200'}</Math> does not prove failure probability 0 — the two-strip bound at that size
      is {fixed(twoStripBound(0.1, 200).raw, 10)}, small and strictly positive. The mathematical bound was
      derived independently; this experiment illustrates its setting and looseness rather than proving it. An
      individual draw's error need not decrease when a different, larger sample is drawn.</Prose>

    <H3>A real development learning curve</H3>
    <Prose>The Banknote Authentication data provides four wavelet-derived features and a binary class code. The
      included subset and source IDs match preceding lessons. Here all {provenance.poolRows} designated training
      rows have labels; the {provenance.developmentRows} development rows are fixed. The
      final {provenance.testRows} test rows are <strong>not evaluated by this program</strong>. The task is
      predicting the source class code, without assigning an unverified genuine/forged
      meaning. <a href={provenance.record}>UCI data record</a>, <a href={provenance.licenseUrl}>{provenance.license}</a>.</Prose>
    <Prose>The program uses a fixed random permutation of training rows, seed {pacData.learningCurves.seed}, and
      nested prefixes of {pacData.learningCurves.sizes.join(', ')} examples. At each size it fits three
      predeclared procedures: {Object.values(curveLabels).join('; ')}. Each scaler is fitted only on its current
      training prefix. All four features — {provenance.features.join(', ')} — are used.</Prose>

    <LearningCurveFigure />

    <Prose>These are actual observations from one nested sequence and one shared development set. They show how
      empirical performance changes for these procedures on these data. They do not measure each family's VC
      dimension, prove a theorem, or establish that validation accuracy must rise at every increment. The shared
      development observations are not independent replicate estimates and do not justify invented confidence
      bands.</Prose>
    <Prose>The practical question is what experiment to run next: collect more representative data, change the
      representation, investigate errors, or adjust model capacity. Development curves can inform that choice.
      Final performance assessment belongs after those choices, using an appropriate untouched evaluation
      set.</Prose>

    <H3>Run the whole thing yourself</H3>
    <Prose>Two complete programs are offered as downloads rather than printed in full, because each writes a
      results file rather than a handful of
      lines. <a href={provenance.calculationProgram}>pac-calculations.py</a> contains the exact interval and
      threshold pattern enumeration, the checked two-dimensional separators, the exact finite-world
      probabilities, every bound calculation, the one-parameter construction of section 10 and the repeated
      interval-learning simulation; the three programs above are slices of
      it. <a href={provenance.curveProgram}>banknote-learning-curves.py</a> produces the measured development
      table, and <a href={provenance.file}>banknote-subset.csv</a> is the data it reads. Put the three files in
      one directory.</Prose>
    <CodeBlock language="powershell">{'python -m venv .venv\n'
      + '.venv\\Scripts\\python.exe -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1\n'
      + '.venv\\Scripts\\python.exe pac-calculations.py\n'
      + '.venv\\Scripts\\python.exe banknote-learning-curves.py'}</CodeBlock>
    <CodeBlock language="sh">{'python3 -m venv .venv\n'
      + '.venv/bin/python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1\n'
      + '.venv/bin/python pac-calculations.py\n'
      + '.venv/bin/python banknote-learning-curves.py'}</CodeBlock>
    <Prose><Code>pac-calculations.py</Code> writes <Code>checked-results.json</Code> beside itself; every number
      on this page that is not a measurement comes from that file or is recomputed from the same definitions in
      the browser. <Code>banknote-learning-curves.py</Code> prints one line per fitted model and
      writes <Code>banknote-learning-curve-results.json</Code>. Here is exactly what it printed when this page's
      numbers were produced, on Python {pacExamples.curves.environment.python} with
      NumPy {pacExamples.curves.environment.numpy} and
      scikit-learn {pacExamples.curves.environment['scikit-learn']}:</Prose>
    <CodeBlock language="output">{pacExamples.curves.expected}</CodeBlock>
    <Prose>The columns are the procedure, the training size, the training labels it got right, and the
      development labels it got right out of {provenance.developmentRows}.</Prose>

    {/* ============================================================ §9 */}
    <H2 id={headingId(headings[8])}>{headings[8]}</H2>
    <Prose>A smaller class can make estimation easier while excluding the rule the task needs. A larger class
      may reduce approximation error and increase the challenge of choosing among its members. Uniform
      convergence controls the estimation side; it does not tell us the best risk inside every competing
      family.</Prose>
    <Prose>Suppose a simple family cannot achieve risk below .2 while a richer family contains a rule with
      risk .02. A wider bound for the richer family does not prove the simple family will predict better.
      Conversely, a very flexible family that perfectly fits twenty labels has not supplied evidence about
      unseen examples merely by achieving zero training error. Compare empirical evidence, task structure and a
      guarantee's actual assumptions together.</Prose>
    <Prose>For a finite set of predeclared families, allocate an overall <Math>{'\\delta'}</Math> across their
      simultaneous bounds before choosing. More generally, a countable sequence can use positive
      weights <Math>{'\\pi_j'}</Math> summing to 1 and confidence budgets <Math>{'\\delta\\pi_j'}</Math>. This is
      the starting idea behind <strong>structural risk minimization</strong>: compare empirical error plus a
      justified class penalty, while accounting for the family search. Choosing a narrow family only after
      inspecting outcomes and then charging for that narrow family alone is not the same procedure — the
      difference is exactly the {fixed(bandFigure.radius, 6)} against {fixed(bandFigure.shrunkRadius, 6)} of
      figure 3.</Prose>
    <Prose>The distinction also matters in transfer learning. A representation chosen using independent prior
      data can help make a target task simpler. If it is adapted using the target evaluation labels, the
      analysis must include that adaptation. “Pretrained” is not a mathematical exemption from selection or
      distribution assumptions.</Prose>
    <Prose>A useful statement records the loss, sampling unit, hypothesis family, algorithm, sample size,
      confidence, theorem and numerical result. If assumptions such as independence or matching deployment
      distribution fail, the theorem does not directly apply to that deployment question. A bound that is valid
      but numerically larger than the trivial range is vacuous; an inapplicable bound is a different problem.
      PAC theory by itself is not a certification of clinical, financial or safety outcomes.</Prose>

    {/* ============================================================ §10 */}
    <H2 id={headingId(headings[9])}>{headings[9]}</H2>
    <Prose>This section is a deeper branch. It answers questions the core route raises but does not need.</Prose>

    <H3>Parameter count is not VC dimension</H3>
    <Prose>The class of all affine half-spaces in <Math>{'p'}</Math>-dimensional input space has VC
      dimension <Math>{'p+1'}</Math>; homogeneous separators through the origin form a different class.
      Axis-aligned boxes in <Math>{'p'}</Math> dimensions have VC dimension <Math>{'2p'}</Math>. A union of at
      most <Math>{'k'}</Math> intervals on the line has VC dimension <Math>{'2k'}</Math>: it can realize all
      patterns on <Math>{'2k'}</Math> points, while <Math>{'2k+1'}</Math> alternating points starting and ending
      positive require <Math>{'k+1'}</Math> separate positive runs.</Prose>
    <Prose>These formulas concern the specified unrestricted mathematical classes. They do not license assigning
      a depth-limited tree a VC dimension from leaf count alone while ignoring input dimension, split family and
      representation. An RBF SVM's finite number of support vectors in one fit also is not the VC dimension of
      the unrestricted kernel hypothesis family. Norm and margin restrictions and algorithm-specific analyses
      need their own definitions.</Prose>
    <Prose>Here is a surprising exact example. The
      family <Math>{'h_\\theta(x)=\\mathbf1[\\sin(\\theta x)\\ge0]'}</Math> uses one real parameter but has
      infinite VC dimension when allowed the inputs and precision below.
      Take points <Math>{'1,2,4,\\dots,2^{n-1}'}</Math>. Given desired
      labels <Math>{'y_1,\\dots,y_n'}</Math>, form a binary fraction <Math>{'r'}</Math> whose
      first <Math>{'n'}</Math> bits are <Math>{'1-y_1,\\dots,1-y_n'}</Math> and append bits 01.
      Set <Math>{'\\theta=2\\pi r'}</Math>.</Prose>

    <SineFigure />

    <Prose>For labels [{sine.labels.join(', ')}], <Math>{'r='}</Math>{sine.rText}
      {' '}and <Math>{'\\theta\\approx'}</Math>{' '}{fixed(sine.theta, 6)}. The fractional cycles
      at <Math>{'x=[1,2,4,8]'}</Math> are [{sine.cycles.join(', ')}], giving
      exactly [{sine.predicted.join(', ')}]. The complete program checks all 16 four-point labelings using exact
      fractions.</Prose>
    <Prose>For ReLU networks, precise architecture-dependent bounds involve weights, layers and activation
      structure. Bartlett and colleagues give an upper bound <Math>{'O(WL\\log W)'}</Math> and related lower
      bounds for piecewise-linear networks. This does not make “all neural networks have VC dimension equal to
      parameter count” correct, nor establish that ordinary training always selects a provably small-capacity
      subclass. <a href="https://jmlr.org/papers/v20/17-612.html">Primary neural-network VC-dimension
      result</a>.</Prose>

    <H3>Why other generalization tools exist</H3>
    <Prose><strong>Rademacher complexity</strong> asks how well a fixed function family can correlate with
      independent random signs on sampled inputs. Its empirical form adapts to the sample geometry, which can
      provide information that a worst-case shattering count omits. It still requires a clearly defined family
      and loss; taking the single already-trained model as a new outcome-dependent class does not automatically
      justify a tiny penalty. The upcoming Rademacher Complexity &amp; Generalization Bounds lesson develops
      this mechanism and its computation.</Prose>
    <Prose><strong>Algorithmic stability</strong> studies how changing one training observation changes the
      learned predictor's loss. This can exploit how an algorithm chooses among hypotheses rather than control
      the entire class uniformly. Different stability definitions and bounds have different assumptions; neither
      every use of SGD nor every regularizer automatically supplies a useful
      bound. <a href="https://jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf">Bousquet and Elisseeff</a>.</Prose>
    <Prose><strong>PAC-Bayes</strong> uses a distribution over predictors and penalizes its divergence from a
      suitable prior. The simplest statements require a prior independent of the training sample; data-dependent
      priors need an appropriate extension or separate data. A distribution over predictors is not automatically
      the deterministic trained model's guarantee. Dziugaite and Roy demonstrated nonvacuous bounds for
      particular deep stochastic networks; that is an existence result under a studied setup, not a claim that
      every modern model receives a tight bound. <a href="https://arxiv.org/abs/1703.11008">Their primary
      paper</a>.</Prose>
    <Prose>For real-valued prediction, <strong>pseudo-dimension</strong> introduces a separate comparison
      threshold at each input and asks which above/below patterns functions can
      realize. <strong>Fat-shattering</strong> adds a positive separation scale around those thresholds. They
      extend the capacity question, but meaningful regression guarantees also need assumptions on output and
      loss ranges or tails. Substituting a dimension into a bounded binary formula does not control arbitrary
      unbounded squared losses.</Prose>

    <H3>Computation is a separate question</H3>
    <Prose>Our interval enumeration requires only counting contiguous blocks. Trying every labeling
      of <Math>{'n'}</Math> points takes <Math>{'2^n'}</Math> requests; random trials can find witnesses but
      cannot generally certify nonexistence. For a fixed finite half-plane labeling, a linear feasibility
      problem provides a more principled computational check. Establishing a class-wide VC upper bound still
      requires an argument covering all point configurations.</Prose>
    <Prose>General capacity computation depends on how a class is represented. Do not attach a single
      complexity-class label to every neural network, finite table or geometric family. The statistical theorem
      separates sample sufficiency from finding an ERM efficiently. A useful author or practitioner preserves
      that separation rather than promising a scalable generic “VC calculator.”</Prose>

    {/* ============================================================ §11 */}
    <H2 id={headingId(headings[10])}>{headings[10]}</H2>

    <Practice title="1. Interpret the two tolerances"
      question="A procedure promises population error at most .08 with probability at least .99 over independent training draws. Explain what ε and δ mean, and whether a particular fitted model is promised 99% accuracy."
      hint="One probability concerns new examples given a fitted rule; the other concerns which fitted rule training produces.">
      <Prose>ε = .08 and δ = .01. At least 99% of training draws produce a rule whose population classification
        error is at most 8%, under the specified setting. The success target is at least 92% population
        accuracy, not 99%. The statement permits a small fraction of training draws to miss that target and does
        not identify them from their training scores.</Prose>
    </Practice>

    <Practice title="2. A family selected on shared validation examples"
      question="Twelve candidate predictors are fitted independently of 800 evaluation outcomes. For bounded 0–1 losses and δ = .02, compute the simultaneous two-sided radius. Must the twelve predictors' errors be independent? May we replace 12 with 1 after selecting the best observed rule?"
      hint="Use the radius formula of section 4. Distinguish independence of observations from dependence across rules.">
      <Prose>The radius is <Math>{'\\sqrt{\\ln(1200)/1600}'}</Math> = {fixed(finiteFamilyRadii.practice, 10)}.
        The union bound does not require independence across predictors. Independent sampled observations are
        needed by the fixed-rule concentration argument. Replacing 12 with 1 after selecting on the same
        outcomes would omit the search; the simultaneous event already covers the selected member.</Prose>
    </Practice>

    <Practice title="3. Find the missing interval patterns"
      question="Four points have coordinates [.1, .3, .6, .9]. How many labelings can a single interval realize? List the impossible patterns. Does shifting every coordinate right by 2 change the answer for unrestricted intervals on the real line?"
      hint="Positive labels must form one contiguous run.">
      <Prose>There are 1 + 4×5/2 = {patternsOnFour.length} realizable patterns. The five impossible ones are
        0101, 1001, 1010, 1011 and 1101. Each has separated positive runs. A common translation preserves point
        order and the available interval witnesses, so the count and feasibility of each ID-attached pattern are
        unchanged. Investigation 2's “translate everything by +2” button is exactly this check.</Prose>
    </Practice>

    <Practice title="4. A counterexample is not the whole VC proof"
      question="Three collinear plane points cannot be shattered by affine half-planes. Does that prove the class has VC dimension at most 2? Supply the missing reasoning for its actual dimension."
      hint="The lower-bound part of VC dimension is existential over point sets.">
      <Prose>No. A noncollinear triangle can be shattered, giving a lower bound 3. An upper bound requires ruling
        out every four-point configuration: an interior point versus the surrounding triangle, alternating
        vertices of a convex quadrilateral, and degenerate cases. The combination gives VC dimension 3. A
        failure on one triple says only that that triple is not shattered.</Prose>
    </Practice>

    <Practice title="5. A box proof without a false geometric claim"
      question="Show that axis-aligned rectangles in the plane shatter the four points (−1, 0), (1, 0), (0, −1), (0, 1), but cannot shatter any set of five distinct points."
      hint="For the upper bound choose representatives attaining minimum and maximum x and y. A fifth point lies in their bounding box; it need not lie in their convex hull.">
      <Prose>For any nonempty chosen subset of the four cross-shaped points, its tight bounding rectangle
        includes exactly that subset; use the empty positive set for the empty subset. For any five-point set,
        choose at most four representatives for the x and y extrema. At least one remaining point lies inside or
        on their bounding box. Label the representatives positive and that remaining point negative. Any
        rectangle containing the representatives contains the bounding box and therefore the negative point.
        This impossible labeling proves the upper bound 4, including coordinate ties. The bounding-box argument
        does not require the point to lie in the convex hull of the chosen extrema.</Prose>
    </Practice>

    <Practice title="6. Agnostic risk versus absolute risk"
      question="A uniform event bounds all empirical/population gaps by .04. An exact ERM searches a class whose best population risk is .15. What does the standard ERM comparison guarantee? What changes if its empirical optimization is .01 suboptimal?"
      hint="The comparison crosses the empirical/population boundary twice.">
      <Prose>Excess risk is at most 2 × .04 = .08, giving population risk at most .23. With optimization
        error .01, the ceiling becomes .24. Neither result promises absolute error .08 or .04, and neither says
        the ceiling equals the actual error.</Prose>
    </Practice>

    <Practice title="7. Repair a guarantee report"
      question="An author says: “The sufficient VC radius is 1.2, therefore the classifier cannot learn. A deeper model has twice as many parameters, so it requires exactly twice as much data. We verified the theorem because no simulated run violated it.” Rewrite the claims correctly."
      hint="Separate vacuity, representation capacity, sufficient bounds and empirical evidence.">
      <Prose>A radius 1.2 adds nothing beyond the 0–1 gap range, so that bound is numerically vacuous; it does
        not prove poor actual performance or unlearnability. Parameter count alone does not determine VC
        dimension or a task-specific sample ratio. A finite simulation checks the examples and can expose
        implementation errors, but cannot prove a distribution-free theorem or zero failure probability. State
        the exact inequality, family, assumptions and observed simulation counts separately.</Prose>
    </Practice>

    <Practice title="8. Change the interval experiment"
      question="Use target [.25, .75] and observed inputs [.05, .3, .4, .7, .95]. Determine the tight learned interval and exact uniform-input risk. Compare adding .26 with adding .99. Then explain why the same lengths need not equal risk under a nonuniform input distribution."
      hint="Add lengths of missed target pieces. An exterior negative does not move the fitted positive extrema.">
      <Prose>The learned interval is [{practiceFit.interval[0]}, {practiceFit.interval[1]}], missing .05 at each
        end, risk {fixed(practiceFit.risk, 6)}. Adding .26 changes it
        to [{practiceWithPositive.interval[0]}, {practiceWithPositive.interval[1]}], risk .01 + .05
        = {fixed(practiceWithPositive.risk, 6)}. Adding only .99
        leaves [{practiceWithNegative.interval[0]}, {practiceWithNegative.interval[1]}] and
        risk {fixed(practiceWithNegative.risk, 6)} unchanged. For nonuniform <Math>{'X'}</Math>, integrate the
        probability mass of the disagreement regions rather than their geometric lengths; a short high-density
        region can matter more than a long low-density one. Reproduce all three in investigation 3 by changing
        the target and then using the two preset buttons.</Prose>
    </Practice>

    <Practice title="9. A meaningful learning-curve follow-up"
      question={`The measured tree fits all 80 training labels but gets ${curveRows.find(row => row.model === 'depth5_tree' && row.n === 80).developmentCorrect}/80 development labels right. The RBF SVC gets ${curveRows.find(row => row.model === 'rbf_svc' && row.n === 80).trainCorrect}/80 training and ${curveRows.find(row => row.model === 'rbf_svc' && row.n === 80).developmentCorrect}/80 development labels right. What can you conclude, and what should remain undecided?`}
      hint="These counts come from one shared development set and different algorithms, not measured VC dimensions.">
      <Prose>On this particular development set the SVC makes one error and the tree six. Perfect training fit
        did not imply the better observed development result. The comparison can motivate paired error
        inspection and further development experiments. It does not identify either class's VC dimension,
        establish a universal ranking, quantify independent-run uncertainty, or give an untouched final estimate
        after selecting a procedure on these same outcomes.</Prose>
    </Practice>

    {/* ============================================================ §12 */}
    <H2 id={headingId(headings[11])}>{headings[11]}</H2>
    <Prose>You can now separate the two randomness levels behind “probably approximately correct”; run the
      union-bound argument over a predeclared finite class and say what breaks when the class is chosen after
      the fact; distinguish a gap bound from an excess-risk bound and explain the factor of two; decide whether
      a class shatters a given set, and say why one unshatterable configuration proves nothing about the class;
      count the growth function for thresholds and intervals and read Sauer's lemma as the upper bound it is;
      and read a numerical VC radius, including one above 1, without either dismissing the theorem or
      overstating it.</Prose>
    <Prose>Check your readiness on three questions. Why does the finite-class argument need independent
      observations but not independent rules? A class realizes {growthTable[9].intervals} patterns on 10 points
      while {growthTable[9].allBinary.toLocaleString('en-US')} exist — which of those two numbers is the one
      Sauer's lemma bounds? And if a bound's radius is {fixed(vcRadius(2, 100, 0.05), 6)}, what exactly have you
      learned about the classifier?</Prose>
    <Prose>Continue in module order
      to <a href="/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml">Calibration
      &amp; Conformal Prediction</a>, where the question changes from aggregate classification error to reliable
      uncertainty statements about individual predictions — a different promise on the same fitted model.
      Rademacher Complexity &amp; Generalization Bounds then returns to the capacity argument with a closer view
      of the sampled geometry, replacing the worst-case shattering count with a quantity computed on the sample
      you actually have.</Prose>

    <Callout title="Where this page's numbers come from">
      <Prose>The dataset served here is <a href={provenance.file}>banknote-subset.csv</a>, {provenance.bytes.toLocaleString('en-US')} bytes,
        SHA-256 <Code>{provenance.sha256}</Code>, with
        its <a href={provenance.attribution}>attribution and full provenance</a>. It
        is {provenance.subsetRows} rows drawn from the {provenance.sourceRows.toLocaleString('en-US')}-row UCI
        record by {provenance.selection}, retrieved {provenance.retrieved}. This lesson serves its own copy:
        two other lessons use the same extract under their own directories, and none of them reads another's.
        The {provenance.testRows} test rows are not evaluated anywhere on this page.</Prose>
      <Prose>Every other number is either an exact construction or a theorem expression evaluated in the
        browser from the same definitions the downloadable program uses. The three displayed programs are
        assembled from that program: each one's <em>algorithm</em> is a byte-exact slice of it, pinned by
        SHA-256, with only an adapted import line and a short printing block added so the snippet prints a few
        lines instead of writing a 60 KB file. Running the whole program fresh reproduces its recorded results
        file exactly.</Prose>
    </Callout>

    <Sources alternatives={<>
      <p>For a visual lecture route, Caltech's Learning From Data course
        has <a href="https://www.youtube.com/watch?v=6FWRijsmLtE">Lecture 6: Theory of Generalization</a> and
        {' '}<a href="https://www.youtube.com/watch?v=Dc0sr0kdBVI">Lecture 7: The VC Dimension</a>.
        The <a href="https://work.caltech.edu/telecourse.html">official course page</a> identifies their topics
        and links the recordings. Use them after our finite-family argument in section 3, then reconstruct the
        interval and triangle witnesses yourself. They are extended lectures rather than substitutes for
        calculating the examples, and the course metadata rather than the video content was checked here.</p>
      <p>For concise mathematical
        notes, <a href="https://web.uvic.ca/~nmehta/ml_theory_fall2021/lecture12.pdf">Mehta's lectures 12–13</a> connect
        concrete shattering examples, Sauer's lemma and an explicit uniform-convergence inequality — the one
        this page displays. Six pages, suitable immediately after section
        6. <a href="https://www.stat.berkeley.edu/~bartlett/courses/2014spring-cs281bstat241b/lectures/04-notes.pdf">Bartlett's
        lecture 4</a> explains why selecting a rule requires more than a fixed-rule concentration statement;
        read its notation alongside our candidate-row figure in section 4.</p>
    </>}>
      <li><a href="https://mwarmuth.bitbucket.io/pubs/J14.pdf">Blumer, Ehrenfeucht, Haussler and Warmuth,
        <em> Learnability and the Vapnik–Chervonenkis dimension</em></a> — the deeper source for the learnability
        characterization, the sufficient sample bounds quoted in section 7, the geometric algorithms and the
        separation of statistical from computational learning. Its logarithm convention is base 2 and its
        regularity assumptions matter.</li>
      <li><a href="https://vatsalsharan.github.io/fall23/lec3.pdf">Sharan, lecture 3</a> — page 3, note 1 gives
        the sharper agnostic sample scale without the elementary logarithmic factor, and pages 4–5 state the
        Sauer union/intersection count.</li>
      <li><a href="https://jmlr.org/papers/volume17/15-389/15-389.pdf">Hanneke, <em>The optimal sample
        complexity of PAC learning</em></a> and <a href="https://proceedings.mlr.press/v195/larsen23a.html">Larsen,
        <em> Bagging is an optimal PAC learner</em></a> — the ensemble connection in section 7. Abstract-level
        results; neither certifies a production ensemble default.</li>
      <li><a href="https://jmlr.org/papers/v20/17-612.html">Bartlett, Harvey, Liaw and Mehrabian,
        <em> Nearly-tight VC-dimension bounds for piecewise linear neural networks</em></a> — the exact
        architecture-dependent result quoted in section 10.</li>
      <li><a href="https://jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf">Bousquet and Elisseeff,
        <em> Stability and generalization</em></a> — stability definitions and their framework.</li>
      <li><a href="https://arxiv.org/abs/1703.11008">Dziugaite and Roy</a> — nonvacuous PAC-Bayes bounds for
        particular stochastic networks; an existence result under a studied setup.</li>
      <li><a href={provenance.record}>UCI Banknote Authentication</a>, {provenance.creator}, DOI
        {' '}<a href={provenance.doi}>{provenance.doi.replace('https://doi.org/', '')}</a>,
        {' '}<a href={provenance.licenseUrl}>{provenance.license}</a>.</li>
    </Sources>
  </div>,
};

export default pacLearningContent;
