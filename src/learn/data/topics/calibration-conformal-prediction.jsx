import { Callout, H2, H3, Prose, Code } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  IntervalLab, MonotoneLab, RankLab, ReliabilityLab,
} from '../../components/lesson-labs/CalibrationLabs.jsx';
import {
  ApsFigure, ConditioningForkFigure, GroupMosaicFigure, LabelRolesFigure, MapComparisonFigure,
  MeasuredClassificationFigure, MeasuredIntervalsFigure, ProbabilityLayersFigure, ResidualGeometryFigure,
  ResolutionFigure, ScoreSetFigure, TemperatureFigure,
} from '../../components/lesson-labs/CalibrationFigures.jsx';
import { KindTag, Table, round } from '../../components/lesson-labs/CalibrationShared.jsx';
import { calibrationExamples } from '../calibration-examples.js';
import { calibrationData } from '../calibration-data.js';
import {
  brierLoss, classSets, conformalRank, conformalThreshold, conformalRank as rankOf, cqrInterval,
  fitPav, fixtures, floatingBoundaryCheck, isotonicPredict, mosaicMarginal, normalizedInterval,
  positivePredictiveValue, reliability, resolutionComparison, rocAuc, softmaxAt,
} from '../calibration-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');
/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.round` here resolves to that component and silently
   yields `undefined` rather than a number. Anything that would reach for the
   global `Math` gets an explicit, shadow-proof helper instead. */
const percent = (value, digits = 2) => String(Number((100 * value).toFixed(digits)));

const classification = calibrationData.classification;
const regression = calibrationData.regression;
const provenance = calibrationData.provenance;
const record = calibrationData.constructedRecord;

const forecasts = fixtures.cards.map(card => card.forecast);
const outcomes = fixtures.cards.map(card => card.outcome);
const twoBins = reliability(forecasts, outcomes, fixtures.twoBinEdges);
const oneBin = reliability(forecasts, outcomes, fixtures.oneBinEdges);
const repaired = reliability(fixtures.repairedForecasts, outcomes, fixtures.twoBinEdges);
const brierBefore = brierLoss(forecasts, outcomes);
const brierRepaired = brierLoss(fixtures.repairedForecasts, outcomes);
const aucBefore = rocAuc(forecasts, outcomes);
const aucRepaired = rocAuc(fixtures.repairedForecasts, outcomes);

const resolution = resolutionComparison(fixtures.resolution);
const pav = fitPav(fixtures.pavScores, fixtures.pavLabels);
const pavTies = fitPav(fixtures.tiedScores, fixtures.tiedLabels);
const practicePav = fitPav(fixtures.practicePavScores, fixtures.practicePavLabels);
const coolTemperature = softmaxAt(fixtures.temperatureLogits, 1);
const warmTemperature = softmaxAt(fixtures.temperatureLogits, 2);

const workedThreshold = conformalThreshold(fixtures.calibrationScores, fixtures.defaultAlpha);
const tinyAlphaRank = conformalRank(fixtures.calibrationScores.length, 0.05);

const normalizedScores = fixtures.residuals.map((residual, index) => residual / fixtures.localScales[index]);
const absoluteThreshold = conformalThreshold(fixtures.residuals, fixtures.intervalAlpha);
const normalizedThreshold = conformalThreshold(normalizedScores, fixtures.intervalAlpha);
const changedThreshold = conformalThreshold(
  fixtures.changedResiduals.map((residual, index) => residual / fixtures.localScales[index]),
  fixtures.intervalAlpha);
const easyInterval = normalizedInterval(fixtures.queries[0].centre, fixtures.queries[0].localScale, normalizedThreshold.q);
const hardInterval = normalizedInterval(fixtures.queries[1].centre, fixtures.queries[1].localScale, normalizedThreshold.q);
const easyAbsolute = normalizedInterval(fixtures.queries[0].centre, 1, absoluteThreshold.q);

const mosaic = mosaicMarginal(fixtures.mosaic);
const ppvHalf = positivePredictiveValue(fixtures.labelShift.sensitivity, fixtures.labelShift.falsePositiveRate, 0.5);
const ppvTenth = positivePredictiveValue(fixtures.labelShift.sensitivity, fixtures.labelShift.falsePositiveRate, 0.1);
const boundary = floatingBoundaryCheck(fixtures.floatingBoundary.positiveProbability);

const practiceRankTwenty = rankOf(14, 0.2);
const practiceRankTwo = rankOf(14, 0.02);
const practiceCqr = conformalThreshold(fixtures.cqrScores, fixtures.cqrAlpha);
const practiceCqrInterval = cqrInterval(fixtures.cqrBase.lower, fixtures.cqrBase.upper, practiceCqr.q);
const practiceVectorSets = classSets(fixtures.practiceVector, fixtures.practiceQ);
const practiceVectorNarrower = classSets(fixtures.practiceVector, 0.6);

const sigmoidRow = classification.methods.sigmoid;
const isotonicRow = classification.methods.isotonic;
const ridgeRow = regression.methods.ridge_absolute;
const cqrRow = regression.methods.cqr;
const rawQuantileRow = regression.methods.raw_quantiles;

const headings = [
  '1. A probability is a statement about a conditioning population',
  '2. Build a reliability diagram from the actual observations',
  '3. Fit a probability map to independent predictions',
  '4. Decide who is allowed to see each label',
  '5. Conformal prediction starts with an ordering of mistakes',
  '6. Adapt the set to the quantity you want to predict',
  '7. Run two complete offline experiments',
  '8. Interpret coverage without changing the probability statement',
  '9. Deeper: change the score or the guarantee deliberately',
  '10. Practice: calculate, change, and interpret',
  '11. What you can now do, and where it goes next',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="cal-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const calibrationContent = {
  title: 'Calibration & Conformal Prediction',
  readTime: '~55 min first pass · ~110 min complete read + 70–110 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot cal-lesson">
    <LessonIntro
      prerequisites={<>Conditional probability, averages, sorting, and the idea of a held-out sample.{' '}
        <Math>{'\\mathbb E[Y\\mid P]'}</Math> means “among the inputs that received a given forecast, how often
        did the event happen”. The preceding <a href="/learn/path/full-curriculum/pac-learning-vc-dimension?module=classical-ml">PAC
        Learning &amp; VC Dimension</a> lesson concerned population error and variation across training samples;
        this one asks what a single predicted number means.{' '}
        <a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">Evaluation
        Metrics</a> and logistic regression help, and the operations they supply are refreshed where they are
        needed. Reliability diagrams, the pool-adjacent-violators algorithm, temperature scaling, nonconformity
        scores and the split-conformal rank are all introduced here.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A classifier assigns an unfamiliar recording a probability of .9 for “mechanical fault”. Another system
      returns a set: {'{'}loose bearing, rubbing belt{'}'}. A third predicts tomorrow's sound level as 120 dB with
      an interval from 113 to 127 dB. All three communicate uncertainty and all three make different promises. You
      will build a reliability diagram from ten cards and watch its error vanish without a single forecast
      changing, pool a monotone map by hand, calculate the exact rank {' '}
      <Math>{'k=\\lceil (n+1)(1-\\alpha)\\rceil'}</Math> that reserves a place for an example you have not seen,
      turn a residual scale into a physical interval, and then read two complete experiments on real data in which
      the best-covering method is not the most useful one. The investigations update their calculations and visual explanations as you change valid inputs.
    </LessonIntro>

    <div className="cal-route"><Prose><strong>First pass.</strong> Read sections 1–8 and do practice 1–6. That
      route gets you the two meanings of calibration, a reliability diagram you built yourself, a fitted
      probability map, the four separate jobs a label can do, the exact finite rank and why it has an{' '}
      <Math>{'n+1'}</Math> in it, intervals in real units, and a measured comparison you can judge. Run the three
      short programs on the way and download the two complete ones for section 7. Section 9 is a deeper branch:
      adaptive sets, full conformal and jackknife+, covariate shift, risk control and the library route. Come back
      to it once the rank calculation feels automatic.</Prose></div>

    <Prose><strong>Calibration</strong> asks whether forecasts carrying a stated probability have the
      corresponding outcome frequency in the population of interest. <strong>Conformal prediction</strong> turns a
      fixed prediction procedure and fresh labelled examples into sets with a particular coverage guarantee.
      Neither operation makes an inaccurate model accurate by definition. The useful result is a prediction whose
      meaning, construction and limits you can explain.</Prose>

    <Callout title="Three kinds of number live on this page, and they are never the same thing">
      An <strong>exact constructed population</strong> quantity is a designed teaching value: the two forecast
      groups in section 1, the group mosaic in section 8 and the prevalence example carry no sampling error at
      all. A <strong>calibration</strong> quantity is learned from held-out rows — a fitted map, a rank, a
      threshold — and would be a different number on another sample. An <strong>assessment</strong> quantity is a
      finite count of recorded outcomes with a denominator, such as 71 of 80. Conformal validity is{' '}
      <em>marginal</em> and rests on exchangeability, so an assessment count read as a conditional or future
      guarantee is the single most consequential mistake available here. Every number below carries its kind, and
      the rest of the lesson refers back to this rather than repeating the warning.
    </Callout>

    {/* ============================================================ §1 */}
    <H2>{headings[0]}</H2>
    <Prose>Suppose a weather service issues “70% chance of rain” on many days. Among the days receiving that
      forecast, the rain frequency should approach .7 in the population the service describes. This is not a
      demand that any one day contain 70% rain, or that exactly seven of the next ten forecasts succeed.</Prose>
    <Prose>Let <Math>{'Y'}</Math> be 1 when the event happens and 0 otherwise, and
      let <Math>{'P=p(X)'}</Math> be the model's probability for that event. Binary class-probability
      calibration means</Prose>
    <MathBlock>{'\\mathbb E[Y\\mid P]=P.'}</MathBlock>
    <Prose>The left side is the positive rate among inputs receiving a given forecast; the right side is the
      stated forecast. With continuously varying probabilities the conditioning is understood through a
      conditional expectation, and we estimate it from nearby scores because an exact decimal may never repeat. A
      finite sample estimates this population relationship and can disagree with it through sampling noise.</Prose>

    <H3>Class probability and confidence of being correct</H3>
    <Prose>For a binary prediction <Math>{'p=.2'}</Math>, the model assigns .2 to class 1 and .8 to class 0. If it
      predicts class 0, its top-class confidence is .8. A plot of <Math>{'p'}</Math> against the frequency
      of <strong>class 1</strong> is therefore a different plot from one of maximum confidence
      against <strong>whether the selected class was correct</strong>.</Prose>
    <Prose>That distinction can conceal real errors. Take two equally common forecast groups. One
      receives {num(fixtures.forecastGroups[0].forecast)} but has a true positive rate
      of {num(fixtures.forecastGroups[0].positiveRate)}; the other
      receives {num(fixtures.forecastGroups[1].forecast)} with a true positive rate
      of {num(fixtures.forecastGroups[1].positiveRate)}. Class-1 probabilities are too low in both groups. Yet
      top-class confidence is .8 for everyone, and the average correctness is .8 as well. A confidence-only
      diagram looks perfect because it merges two different mistakes.</Prose>

    <ConditioningForkFigure />

    <Prose>For several classes, useful questions include calibration of each class's probability against that
      class's indicator, calibration of top confidence against correctness, and calibration of the entire
      predicted probability vector. These condition on different information. To examine the forecast for
      class <Math>{'k'}</Math>, use the pairs <Math>{'(p_k,\\mathbf 1\\{Y=k\\})'}</Math> across the relevant
      evaluation population; restricting to examples with <Math>{'Y=k'}</Math> leaves only positive outcomes and
      is not a class-<Math>{'k'}</Math> reliability diagram at all.{' '}
      <a href="https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf">Vaicenavicius and
      colleagues</a> set out the definitions and the evaluation framework.</Prose>

    <H3>Calibration can discard useful information</H3>
    <Prose>Imagine two equally common production lines whose true fault rates
      are {num(fixtures.resolution.rates[0])} and {num(fixtures.resolution.rates[1])}. A model
      assigns {num(resolution.coarseForecast)} to both. It is calibrated when you condition only on that score,
      because the combined rate is {num(resolution.coarseForecast)}. A model that keeps the line information and
      reports {num(fixtures.resolution.rates[0])} or {num(fixtures.resolution.rates[1])} is also calibrated, with
      more <strong>resolution</strong>: it separates groups whose outcome frequencies differ.</Prose>
    <Prose>Suppose releasing a faulty item costs {fixtures.resolution.releaseCost} units, releasing a sound item
      costs nothing, and quarantining any item costs {fixtures.resolution.quarantineCost}. Release has expected
      cost <Math>{'80p'}</Math> and quarantine costs 20, so the threshold
      is <Math>{'p=' + num(resolution.decideThreshold)}</Math>. The coarse score releases everyone, costing{' '}
      {num(resolution.coarseCost)} on average. Using the two rates releases the lower group and quarantines the
      higher one, costing {num(resolution.fullCost)}.</Prose>
    <Prose>Recall that Brier loss is the average squared probability error, <Math>{'(p-y)^2'}</Math>. In a group
      whose positive rate is <Math>{'r'}</Math>, forecasting <Math>{'p'}</Math> has expected
      loss <Math>{'r(1-p)^2+(1-r)p^2'}</Math>: weight the error for each possible outcome by how often it occurs.
      Averaging across the two equally common groups gives {num(resolution.coarseBrier)} for the constant score
      and {num(resolution.fullBrier)} for the two exact group rates. Both forecasts are calibrated. The better
      proper score reflects additional useful information, not a calibration repair.</Prose>

    <ResolutionFigure />

    <Prose>Calibration supports sensible decisions using the information in a score; it does not guarantee that
      the score preserves everything useful for the decision. That is the distinction between ranking, probability
      quality and operational cost developed
      in <a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">Evaluation
      Metrics</a> and put to work
      in <a href="/learn/path/full-curriculum/decision-theory-risk-cost-sensitive-decisions?module=math-foundations">Decision
      Theory</a>.</Prose>

    {/* ============================================================ §2 */}
    <H2>{headings[1]}</H2>
    <Prose>Use this small <strong>constructed sample</strong>. There are five observations with
      forecast {num(fixtures.cards[0].forecast)}, of which {twoBins.bins[0].positive} are positive, and five with
      forecast {num(fixtures.cards[5].forecast)}, of which {twoBins.bins[1].positive} are positive.</Prose>

    <Table caption="Ten forecast cards in two bins, with the signed gap in the last column"
      headings={['forecast', 'count', 'positive outcomes', 'observed positive fraction', 'observed − forecast']}
      rows={twoBins.bins.map(bin => [
        num(bin.meanP), String(bin.count), String(bin.positive),
        num(bin.fractionPositive), num(bin.gap),
      ])}
      kind={twoBins.kind} />

    <Prose>A reliability diagram places the mean predicted class-1 probability on the horizontal axis and the
      observed class-1 fraction on the vertical axis. The points are{' '}
      ({num(twoBins.bins[0].meanP)}, {num(twoBins.bins[0].fractionPositive)})
      and ({num(twoBins.bins[1].meanP)}, {num(twoBins.bins[1].fractionPositive)}). The diagonal represents matching
      values. At <Math>{'x=' + num(twoBins.bins[1].meanP)}</Math>, {' '}
      <Math>{'y=' + num(twoBins.bins[1].fractionPositive)}</Math> is <strong>below</strong> the diagonal: the
      positive probability is overestimated. At <Math>{'x=' + num(twoBins.bins[0].meanP)}</Math> the point is
      above: it is underestimated. On a top-confidence diagram, below the diagonal specifically means
      overconfidence about the selected class.</Prose>
    <Prose>The accompanying count strip matters. Two positives out of five are weak evidence about a precise
      population rate; two hundred out of five hundred are substantially more precise. A dot without its
      denominator hides that difference. Empty bins have no estimated fraction and should not be mistaken for bins
      with zero positive outcomes.</Prose>

    <H3>What binned Expected Calibration Error measures</H3>
    <Prose>For bins <Math>{'B_b'}</Math>, define the sample summary</Prose>
    <MathBlock>{'\\begin{gathered}\\widehat{\\mathrm{ECE}}\\\\[4pt]=\\sum_b\\frac{n_b}{n}\\left|\\bar y_b-\\bar p_b\\right|.\\end{gathered}'}</MathBlock>
    <Prose>With one bin for each forecast group, ECE is {num(twoBins.ece)}. Merge the entire sample into one bin
      and both means become {num(oneBin.bins[0].meanP)}, giving an ECE of exactly {num(oneBin.ece)}. The model did
      not change. The binning erased two opposing local discrepancies.</Prose>
    <Prose>Binning creates two competing problems: wide bins hide variation, while small bins have noisy outcome
      counts. Even a perfectly calibrated population can produce a nonzero empirical ECE, and a zero empirical ECE
      for a particular binning does not prove calibration. Report the population, the forecast definition, the
      boundaries, the bin counts and the sampling uncertainty along with the summary, and choose the analysis
      before using it to select a model. Repeatedly searching bin schemes for the smallest ECE is another form of
      selection.{' '}
      <a href="https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf">Section 4 of the primary
      evaluation study</a> discusses these estimation effects.</Prose>
    <Prose>For our two-bin sample, replacing {num(fixtures.cards[0].forecast)} and {num(fixtures.cards[5].forecast)} with
      the observed fractions {num(repaired.bins[0].meanP)} and {num(repaired.bins[1].meanP)} gives
      zero <strong>in-sample</strong> ECE. Brier loss falls from {num(brierBefore)} to {num(brierRepaired)}, while
      ROC AUC stays at {num(aucBefore.value)} before and {num(aucRepaired.value)} after, because replacing each
      group's forecast by its own observed fraction leaves the ordering untouched. It is a useful calculation, but
      fitting and judging the repair on the same ten outcomes would not establish generalisation.</Prose>

    <Program example={calibrationExamples.reliability}>
      <Prose>The guard block at the top of <Code>reliability</Code> is the complete program's own, and it earns
        its place here: a learner who edits the boundaries can easily supply a list that does not span the whole
        probability scale, which would drop observations in silence rather than raising anything. Note the last
        line of the output. The boundary fixture has forecasts 0, .5 and 1; a rule that excluded forecasts exactly
        equal to 1 would report two cards instead of three, and the ones it lost would be the most confident.</Prose>
    </Program>

    <ReliabilityLab />

    {/* ============================================================ §3 */}
    <H2>{headings[2]}</H2>
    <Prose>A raw model score may rank examples well without being a probability. A support-vector classifier's
      decision score is one example. Applying a sigmoid merely puts that score between zero and one; it does not
      establish that .8 means an 80% event rate.</Prose>
    <Prose>We keep the base predictor fixed and learn a map from its score to a probability, using examples that
      did not fit that predictor. The map has its own parameters and can overfit, so its performance has to be
      assessed on further observations.</Prose>

    <H3>Sigmoid calibration: fit both slope and offset</H3>
    <Prose>Write the logistic function as <Math>{'\\sigma(z)=1/(1+e^{-z})'}</Math>. A sigmoid calibrator uses</Prose>
    <MathBlock>{'q(s)=\\sigma(as+b).'}</MathBlock>
    <Prose>The slope controls how quickly probabilities change with the score; the offset moves the probability
      scale. At <Math>{'q=.5'}</Math>, <Math>{'as+b=0'}</Math>, so changing <Math>{'b'}</Math> can change
      predicted classes under a .5 threshold. A positive <Math>{'a'}</Math> preserves the score order, a
      negative <Math>{'a'}</Math> reverses it, and a zero <Math>{'a'}</Math> produces a constant. Multiclass
      one-versus-rest mappings followed by normalisation have additional interactions, and their effects cannot be
      read off one binary curve.</Prose>
    <Prose>For binary labels, fit <Math>{'a'}</Math> and <Math>{'b'}</Math> by minimising average log loss. A
      numerically stable contribution is <Code>logaddexp(0, z) - target*z</Code>,
      with <Math>{'z=as+b'}</Math>; its gradients are the averages of <Math>{'(q-\\text{target})s'}</Math> and{' '}
      <Math>{'q-\\text{target}'}</Math>. This is the same residual mechanism as logistic regression, applied to
      one already-computed score.</Prose>
    <Prose>Platt's method uses smoothed targets: positives
      receive <Math>{'(N_+{+}1)/(N_+{+}2)'}</Math> and negatives <Math>{'1/(N_-{+}2)'}</Math>. Smoothing
      discourages infinite logits on a separable calibration sample. It is part of the stated fit, not a promise
      of exact calibration.</Prose>
    <Prose>A calibration sample with only one outcome class supplies no evidence about the missing class.
      This lesson's sigmoid fitter therefore requires both classes and declines to fit otherwise; investigation
      2's “make every label 1” button demonstrates that evidence policy. It is not a claim that smoothing has no
      numerical solution. With a constant smoothed target <Math>{'t\\in(0,1)'}</Math>, the constant map
      <Math>{'a=0,\\ b=\\log(t/(1-t))'}</Math> minimises the loss; when scores vary, the two parameters are
      uniquely determined. For six positive labels, for example, <Math>{'t=7/8'}</Math> and
      <Math>{'b=\\log 7'}</Math>. <a href="https://www.csie.ntu.edu.tw/~htlin/paper/doc/plattprob.pdf">Lin, Lin and
      Weng's analysis</a> gives the convexity condition. ROC AUC has a different problem: with only one class
      there is no positive–negative pair to compare, so AUC itself is undefined. A rare class missing from a
      calibration split is an ordinary way to encounter both issues.</Prose>
    <Prose>For scores <Math>{'[-3,-2,-1,0,1,2,3,4]'}</Math> and
      labels <Math>{'[0,1,0,0,1,0,1,1]'}</Math>, the fit
      obtains <Math>{'a\\approx' + num(record.sigmoid.a)}</Math> and <Math>{'b\\approx' + num(record.sigmoid.b)}</Math>.
      These are fitted values for a constructed example, not an independently measured improvement.</Prose>

    <H3>Isotonic calibration: pool a local contradiction</H3>
    <Prose>Sometimes a sigmoid is too restrictive, but higher scores should still mean no lower
      probability. <strong>Isotonic regression</strong> fits non-decreasing probabilities by minimising the sum of
      squared differences from the observed labels. Sort the observations by their scores while keeping the labels
      attached; do not sort the labels separately.</Prose>
    <Prose>Start with each distinct score as a block. A block's fitted probability is its fraction of positive
      outcomes. If a left block's mean exceeds its right neighbour's, the required monotonicity is violated: merge
      them and replace both estimates by their combined, count-weighted mean, then continue backwards if the merge
      creates a new violation. That is the pool-adjacent-violators algorithm.</Prose>
    <Prose>In the eight-score example the sequence of raw label means
      is <Math>{'[0,1,0,0,1,0,1,1]'}</Math>. The 1 followed by 0 merges to <Math>{'\\tfrac12'}</Math>, then that
      block and the next 0 merge to <Math>{'\\tfrac13'}</Math>; the later 1, 0 pair merges
      to <Math>{'\\tfrac12'}</Math>. The fitted values become</Prose>
    <MathBlock>{'[0,\\tfrac13,\\tfrac13,\\tfrac13,\\tfrac12,\\tfrac12,1,1].'}</MathBlock>
    <Prose>Equal raw scores must receive equal fitted values.
      For scores <Math>{'[-2,-2,0,1]'}</Math> with labels <Math>{'[1,0,0,1]'}</Math>, group the
      two <Math>{'-2'}</Math> observations first: their mean is {num(0.5)} with weight 2. Pooling with the zero at
      score 0 gives {num(pavTies.fitted[0])} with weight {pavTies.blocks[0].weight}. The fitted values at the
      distinct scores <Math>{'[-2,0,1]'}</Math> are{' '}
      <Math>{'[' + pavTies.fitted.map(value => num(value)).join(',') + ']'}</Math>. Averaging block means without
      their counts would give {num((0.5 + 0) / 2)}, which is the wrong answer.</Prose>

    <MapComparisonFigure />

    <Program example={calibrationExamples.monotone}>
      <Prose>The stack implementation is linear after sorting; sorting ordinarily
        costs <Math>{'O(n\\log n)'}</Math>. The fitted values are defined at the observed knots and are constant
        within pooled blocks. A library must additionally specify how to predict between knots and outside the
        observed range: our practical example uses scikit-learn's interpolation with endpoint clipping, so halfway
        between the first two knots the value
        is {num(isotonicPredict(pav.knots, pav.fitted, (pav.knots[0] + pav.knots[1]) / 2))} rather than either
        neighbour. The result need not be a staircase between every pair of distinct knots.</Prose>
    </Program>

    <Prose>More flexibility requires enough informative calibration observations, especially in rare score
      regions. There is no universal sample count at which isotonic suddenly becomes preferable. It can introduce
      ties and change AUC despite being monotone, so assess held-out proper scores, reliability counts and the
      relevant decisions together.{' '}
      <a href="https://scikit-learn.org/stable/modules/calibration.html">Scikit-learn's calibration guide</a> sets
      out both methods.</Prose>

    <MonotoneLab />

    <H3>Temperature scaling: change confidence while keeping the winning logit</H3>
    <Prose>A multiclass model often produces logits <Math>{'z_1,\\dots,z_K'}</Math>, and softmax turns them into
      positive numbers summing to one. <strong>Temperature scaling</strong> uses one positive <Math>{'T'}</Math>:</Prose>
    <MathBlock>{'q_k=\\frac{\\exp(z_k/T)}{\\sum_j\\exp(z_j/T)}.'}</MathBlock>
    <Prose>For logits <Math>{'[3,1,0]'}</Math>, <Math>{'T=1'}</Math> produces
      approximately <Math>{'[' + coolTemperature.map(value => num(value)).join(',') + ']'}</Math>.
      With <Math>{'T=2'}</Math> it becomes <Math>{'[' + warmTemperature.map(value => num(value)).join(',') + ']'}</Math>.
      The leading class stays first. Dividing all logits by the same positive number preserves their
      within-example order and their ties, so the top-1 class and its accuracy are unchanged under the same tie
      rule. Probability thresholds and cost-sensitive decisions can still change.</Prose>

    <TemperatureFigure />

    <Prose>Fit <Math>{'T'}</Math> by minimising held-out multiclass log loss. Using inverse
      temperature <Math>{'\\beta=1/T'}</Math>, the objective is the average
      of <Code>logsumexp(beta*z) - beta*z_true</Code>, whose derivative compares the probability-weighted mean
      logit with the true-class logit. Our instructional fit uses a declared positive search range and reports
      whether the answer lands near its boundary: on the four constructed rows it
      returns <Math>{'T\\approx' + num(record.temperatureFit.temperature)}</Math>, moving the log loss
      from {num(record.temperatureFit.nllBefore)} to {num(record.temperatureFit.nllAfter)} with the search
      boundary {record.temperatureFit.nearSearchBoundary ? 'reached' : 'not reached'}. The base model's logits are
      cached, so fitting <Math>{'T'}</Math> need not repeatedly run the whole network.</Prose>
    <Prose>The method can soften or sharpen probabilities, but one scalar cannot repair arbitrary class-specific
      biases; vector or matrix scaling adds parameters and can change winning classes.{' '}
      <a href="https://proceedings.mlr.press/v70/guo17a/guo17a.pdf">Guo and colleagues</a> found temperature
      scaling effective on the models and datasets they studied. That empirical finding is not a rule that every
      later network must be overconfident, or must improve under calibration.</Prose>

    {/* ============================================================ §4 */}
    <H2>{headings[3]}</H2>
    <Prose>A clear data-flow diagram is more valuable than a method name when you are looking for leakage. In our
      classification experiment, labels have four separate jobs.</Prose>

    <LabelRolesFigure />

    <Prose>The third role is separate because probability calibration is itself training. Reusing its outcomes to
      compute an ordinary split-conformal threshold treats those observations differently from a fresh test
      example. Holding out only the original classifier's training data is not enough. Likewise, hyperparameter
      selection and early stopping consume label information: either account for them within a suitable nested
      procedure, or complete them before using genuinely separate calibration observations.{' '}
      <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation
      &amp; Hyperparameter Tuning</a> develops the nested version.</Prose>
    <Prose>Scikit-learn {provenance.environment.scikitLearn} supports{' '}
      <Code>CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method="sigmoid")</Code>. The wrapper holds
      the fitted pipeline fixed and uses the observations you supply to fit calibration. It cannot determine
      whether you supplied an honest independent split. The estimator's score interface matters: sigmoid
      calibration uses <Code>decision_function</Code> when one is available and otherwise
      uses <Code>predict_proba</Code>. GaussianNB has no decision function, so this is not automatically a sigmoid
      of its generative log odds, and saturated probabilities and unsaturated log odds carry different usable
      numerical information — a point developed
      in <a href="/learn/path/full-curriculum/naive-bayes-probabilistic-classifiers?module=classical-ml">Naive
      Bayes</a>.</Prose>
    <Prose>The current API also supports <Code>method="temperature"</Code>; for probability-only estimators it
      constructs log-probability logits with a numerical safeguard. SVC's <Code>probability</Code> parameter is
      deprecated in 1.9 and scheduled for removal in 1.11, so our example uses an explicit calibration wrapper
      instead — see <a href="/learn/path/full-curriculum/support-vector-machines-svm?module=classical-ml">Support
      Vector Machines</a>. These are versioned software contracts, not timeless properties of the mathematics.{' '}
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html">CalibratedClassifierCV</a> and{' '}
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVC</a> carry the current
      signatures.</Prose>

    <Checkpoint prompt={'A team fits a text classifier with a TF-IDF vocabulary, then builds calibration folds '
      + 'over the same documents. Name the step that has already leaked, and the two ways out.'}>
      <Prose>Fitting the vocabulary — or any supervised feature selection — before the folds are constructed lets
        each fold's calibrator see information derived from its own held-out rows. The extraction must live{' '}
        <em>inside</em> the estimator that is cloned for every fold, so each clone builds its vocabulary from that
        fold's training part only. The alternative is a genuinely independent calibration split scored by a frozen
        already-fitted complete pipeline, which is what section 7 does. Calibration folds also do not undo
        hyperparameter selection that already used their outcomes.</Prose>
    </Checkpoint>

    {/* ============================================================ §5 */}
    <H2>{headings[4]}</H2>
    <Prose>We now freeze the complete predictor and its score function. A <strong>nonconformity score</strong> is
      a number that is large when a proposed answer fits poorly. It need not be a calibrated probability.</Prose>
    <Prose>For classification, one choice is <Math>{'s(x,y)=1-p_y(x)'}</Math>: a class given a small model
      probability receives a large score. For regression, <Math>{'s(x,y)=|y-m(x)|'}</Math> measures the distance
      from a point prediction, in the target's units. The prediction set will contain those candidate answers
      whose scores are not unusually large relative to fresh labelled calibration examples.</Prose>
    <Prose>Suppose nine calibration scores, sorted from <strong>smallest to largest</strong>, are</Prose>
    <MathBlock>{'\\begin{gathered}[.05,.10,.15,.20,.25,\\\\.30,.40,.60,.90].\\end{gathered}'}</MathBlock>
    <Prose>For a desired coverage of <Math>{'1-\\alpha=' + num(1 - fixtures.defaultAlpha)}</Math>,
      calculate <Math>{'k=\\lceil (n+1)(1-\\alpha)\\rceil=\\lceil 10\\times.8\\rceil=' + workedThreshold.k}</Math>.
      The threshold is the {workedThreshold.k}th smallest score, <Math>{'q=' + num(workedThreshold.q)}</Math>. For a
      new input, include every candidate <Math>{'y'}</Math> satisfying <Math>{'s(x,y)\\le ' + num(workedThreshold.q)}</Math>.</Prose>

    <ScoreSetFigure />

    <Prose>Equality is included. The empty row is not a contradiction: this particular score can return an empty
      set when no class reaches the learned threshold. A larger set is not automatically produced for every
      difficult input. Empty, singleton and large sets all deserve interpretation, and a singleton is not
      automatically a label with {percent(1 - fixtures.defaultAlpha, 0)}% conditional probability of correctness —
      the theorem concerns a different probability.</Prose>

    <H3>Why <Math>{'n+1'}</Math> appears</H3>
    <Prose>Imagine including the next example's true-label score alongside the <Math>{'n'}</Math> calibration
      scores. If these <Math>{'n+1'}</Math> scores are exchangeable and have no ties, the next score is equally
      likely to occupy any of their <Math>{'n+1'}</Math> ranks. For <Math>{'k\\le n'}</Math>, the event that it is
      no greater than the <Math>{'k'}</Math>th calibration score corresponds to its combined rank being at
      most <Math>{'k'}</Math>, whose probability is <Math>{'k/(n+1)'}</Math> — at least <Math>{'1-\\alpha'}</Math> by
      our choice of <Math>{'k'}</Math>.</Prose>
    <Prose>The next example's true label is unknown when predicting. The algorithm therefore tests each possible
      label and retains those that would pass this same score comparison. Exactly when the true label passes, the
      set covers it.</Prose>
    <Prose>Here <strong>exchangeable</strong> means that permuting the observations leaves their joint
      distribution unchanged. Independent draws from one distribution are a common sufficient condition, but
      exchangeability can also hold for dependent observations, such as a uniformly randomised ordering of a fixed
      finite collection. It is not a property created by shuffling an arbitrary time series before forecasting its
      future.</Prose>
    <Prose>Conditional on any independent base-training and probability-calibration work, if the conformal
      observations and the new example are exchangeable and are scored by the same fixed rule,</Prose>
    <MathBlock>{'\\begin{gathered}\\Pr\\{Y_{\\rm new}\\in C(X_{\\rm new})\\}\\\\[4pt]\\ge 1-\\alpha.\\end{gathered}'}</MathBlock>
    <Prose>The probability averages over the conformal calibration data <em>and</em> the new example. It is not a
      claim about each individual input, nor about every realised calibration set. With ties, including equality
      preserves the lower guarantee and can produce extra coverage; the familiar upper
      limit <Math>{'1-\\alpha+1/(n+1)'}</Math> additionally needs a no-ties condition or a suitable randomised
      construction. If every possible score equals .2, the non-randomised set rule covers every true label:
      coverage is 1, even when the target was .8.{' '}
      <a href="https://arxiv.org/html/2107.07511v6#A4">The split-conformal theorem and its rank proof</a> are in
      appendix D of the tutorial.</Prose>

    <H3>The finite rank is not an interpolated percentile</H3>
    <Prose>If <Math>{'k\\le n'}</Math>, use <Code>sorted_scores[k-1]</Code> in zero-based indexing.
      If <Math>{'k=n+1'}</Math>, use infinity. For our nine scores and <Math>{'\\alpha=.05'}</Math>,{' '}
      <Math>{'k=' + tinyAlphaRank}</Math> and the answer is infinity, so every possible answer is included.
      Clipping <Math>{'k'}</Math> to <Math>{'n'}</Math> would discard exactly the protection a target beyond the
      resolution of this small calibration set was asking for.</Prose>

    <Program example={calibrationExamples.rank}>
      <Prose>On these nine scores NumPy's default linear quantile at <Math>{'8/9'}</Math> returns
        about {num(0.633333)}, while <Code>method="higher"</Code> at the same level returns {num(0.9)}. Neither is
        the eighth order statistic {num(workedThreshold.q)}: quantile conventions use different indexing rules.
        Rather than rely on a remembered recipe, the function computes <Math>{'k'}</Math> and selects the exact
        indexed value, which also lets it handle the infinity case explicitly. The last two lines are the rotation
        the investigation below asks you to predict.</Prose>
    </Program>

    <RankLab />

    <Prose>The rotations illustrate the symmetry behind the proof. They are an exact enumeration over a fixed
      multiset of cards, not a simulation claiming to verify all exchangeable populations.</Prose>

    {/* ============================================================ §6 */}
    <H2>{headings[5]}</H2>

    <H3>Constant-width regression intervals</H3>
    <Prose>With absolute residual scores, <Math>{'s(x,y)=|y-m(x)|\\le q'}</Math> means</Prose>
    <MathBlock>{'C(x)=[m(x)-q,\\ m(x)+q].'}</MathBlock>
    <Prose>Every interval has width <Math>{'2q'}</Math>. The centres vary; the widths do not. These are intervals
      for a <strong>future response</strong>, including its variability around the predictor. They are not
      confidence intervals for an unknown mean, and they are not credible intervals from a Bayesian
      posterior.</Prose>

    <H3>Normalise by a local scale</H3>
    <Prose>Suppose an independent training procedure supplies a positive scale <Math>{'u(x)'}</Math>, large in
      regions expected to have bigger errors. Use <Math>{'s(x,y)=|y-m(x)|/u(x)'}</Math>. The calibration scores
      are now dimensionless and the final interval is</Prose>
    <MathBlock>{'[m(x)-q\\,u(x),\\ m(x)+q\\,u(x)].'}</MathBlock>
    <Prose>The local scale need not be the true standard deviation for marginal conformal validity; its quality
      affects usefulness. It must be fixed before ordinary conformal calibration, and finite and positive.
      Training outcomes may be used to fit it, but tuning it with the conformal calibration outcomes changes the
      procedure the ordinary split-conformal proof covers. Zero or negative scales are invalid here and this
      lesson's controls refuse them rather than substituting a small number.</Prose>
    <Prose>Consider residual magnitudes{' '}
      <Math>{'[' + fixtures.residuals.map(value => num(value)).join(',') + ']'}</Math> and corresponding scales{' '}
      <Math>{'[' + fixtures.localScales.map(value => num(value)).join(',') + ']'}</Math>, in the same physical
      units. At <Math>{'\\alpha=' + num(fixtures.intervalAlpha)}</Math>, the absolute-residual threshold
      is {num(absoluteThreshold.q)} and the normalised threshold is {num(normalizedThreshold.q)}. Predictions
      of {num(fixtures.queries[0].centre)} with scale {num(fixtures.queries[0].localScale)}
      and {num(fixtures.queries[1].centre)} with scale {num(fixtures.queries[1].localScale)} receive intervals
      [{num(easyInterval.lower)}, {num(easyInterval.upper)}] and
      [{num(hardInterval.lower)}, {num(hardInterval.upper)}]. The absolute method gives
      [{num(easyAbsolute.lower)}, {num(easyAbsolute.upper)}] and the same
      [{num(hardInterval.lower)}, {num(hardInterval.upper)}]. This construction explains how useful scale
      information can avoid giving easy cases unnecessarily large intervals; it does not claim that this scale
      model was learned accurately from data.</Prose>

    <ResidualGeometryFigure />

    <H3>Conformalised quantile regression</H3>
    <Prose>A different approach learns lower and upper conditional quantile
      estimates <Math>{'L(x)'}</Math>, <Math>{'U(x)'}</Math>. One standard definition of the quantile at
      level <Math>{'\\tau'}</Math> is the smallest cutoff <Math>{'q'}</Math> for
      which <Math>{'P(Y\\le q)\\ge\\tau'}</Math>; using the infimum handles distributions whose support needs it. In
      a continuous distribution with a strictly increasing cumulative probability near <Math>{'q'}</Math>, that
      probability equals <Math>{'\\tau'}</Math>. <strong>Pinball loss</strong> weights an underprediction
      by <Math>{'\\tau'}</Math> and an overprediction by <Math>{'1-\\tau'}</Math>, so fitting it targets a quantile
      rather than a mean.</Prose>
    <Prose>Approximate .05 and .95 quantiles supply an initial 90% interval, but fitted endpoints do not
      automatically have finite-sample coverage. Conformalised quantile regression, or <strong>CQR</strong>,
      calibrates the score</Prose>
    <MathBlock>{'s(x,y)=\\max\\{L(x)-y,\\ y-U(x)\\}.'}</MathBlock>
    <Prose>If <Math>{'y'}</Math> lies outside the initial interval, this is the distance beyond the nearer
      violated endpoint; inside, it is non-positive. The final set
      is <Math>{'[L(x)-q,\\ U(x)+q]'}</Math>. A positive <Math>{'q'}</Math> expands the endpoints; a
      negative <Math>{'q'}</Math> can shrink an over-wide initial interval. The score definition admits an empty
      result if shrunken endpoints cross. Expanding such a set for a declared practical policy preserves
      inclusion, but its width and coverage should then be evaluated as that enlarged procedure.</Prose>
    <Prose>Our training code resolves crossed <strong>base</strong> quantile estimates by a fixed pointwise
      minimum/maximum rearrangement before computing any scores, and applies the same transformation at
      prediction. It never sorts each endpoint differently using the true answer. CQR adapts widths through its
      initial quantile fits while keeping a marginal guarantee under the same exchangeability conditions; it does
      not guarantee correct coverage at every <Math>{'x'}</Math>.{' '}
      <a href="https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf">Romano,
      Patterson and Candès</a> give the score and the construction.</Prose>

    <IntervalLab />

    {/* ============================================================ §7 */}
    <H2>{headings[6]}</H2>
    <Prose>Download <a href={provenance.files['calibration_calculations.py'].file} download>calibration_calculations.py</a>,{' '}
      <a href={provenance.files['uncertainty_experiments.py'].file} download>uncertainty_experiments.py</a>,{' '}
      <a href={provenance.files['banknote-subset.csv'].file} download>banknote-subset.csv</a> and{' '}
      <a href={provenance.files['airfoil-subset.csv'].file} download>airfoil-subset.csv</a> into one folder. The
      first program contains the complete numerical mechanisms — including the three excerpts shown above — and
      the second imports them and runs both data experiments. Neither downloads data or needs access to this site.
      The <a href={provenance.attribution}>attribution file</a> gives the creators, licences, original row
      identifiers, selection rules, units and hashes.</Prose>
    <Prose>Use Python {provenance.environment.python} and a virtual environment. On Windows:</Prose>
    <Code>{'py -3.12 -m venv .venv'}</Code>
    <Prose>then install the pinned versions and run the two files in order — the first
      writes <Code>checked-results.json</Code>, the second imports from it and
      writes <Code>experiment-results.json</Code>:</Prose>
    <Code>{'.venv\\Scripts\\python -m pip install numpy==' + provenance.environment.numpy
      + ' scipy==' + provenance.environment.scipy + ' scikit-learn==' + provenance.environment.scikitLearn}</Code>
    <Code>{'.venv\\Scripts\\python calibration_calculations.py'}</Code>
    <Code>{'.venv\\Scripts\\python uncertainty_experiments.py'}</Code>
    <Prose>On macOS or Linux, create the environment with <Code>python3 -m venv .venv</Code> and
      use <Code>.venv/bin/python</Code> in place of <Code>{'.venv\\Scripts\\python'}</Code>. The first program's summary
      line is its own check that nothing drifted:</Prose>
    <Code>{calibrationExamples.calculations.expected}</Code>
    <Program example={calibrationExamples.experiments}>
      <Prose>Under NumPy {provenance.environment.numpy}, SciPy {provenance.environment.scipy} and
        scikit-learn {provenance.environment.scikitLearn}, both programs reproduce this lesson's recorded results
        byte for byte; that equality is rechecked whenever the page's verifiers run. A different library version
        can move the last digits of a fitted optimum, and no claim is made about those. Read the four label roles
        off <Code>classification_experiment</Code>: the four index ranges are assigned before any model is fitted,
        and the same score computation is used for calibration and for prediction so that the tied boundary below
        cannot go missing.</Prose>
    </Program>

    <H3>Classification: separate two meanings of calibration</H3>
    <Prose>The Banknote Authentication data contain four wavelet-derived image features and binary class codes.
      We keep codes 0 and 1 as supplied; the dataset metadata does not say which physical condition each names, so
      we do not infer one. Each row is one observation. Our retained{' '}
      {provenance.banknote.retainedRows}-row subset uses the
      first {classification.roleSizes.train} for the base pipeline, the
      next {classification.roleSizes.probability_calibration} for probability calibration, the
      next {classification.roleSizes.conformal} for conformal calibration, and the
      last {classification.roleSizes.test} for assessment. These allocations precede the comparisons.</Prose>
    <Prose>The base pipeline is <Code>StandardScaler</Code> followed by an RBF SVC
      with <Math>{'C=' + classification.baseParameters.C}</Math>. We predeclare sigmoid calibration as the primary
      procedure, and also report isotonic, temperature, a naive sigmoid of the raw decision score, and the
      training-prior constant. All the mappings use the same frozen base predictor where applicable. We do not
      select the best method using these final assessment results.</Prose>
    <Prose>For each method, the true-class conformal scores on its
      separate {classification.roleSizes.conformal} examples
      determine <Math>{'k=\\lceil 81\\times.9\\rceil=' + classification.rank}</Math>. The
      last {classification.roleSizes.test} observations assess both its probabilities and its sets.</Prose>

    <MeasuredClassificationFigure />

    <Prose>The sigmoid procedure has {sigmoidRow.sizeCounts['0']} empty sets
      and {sigmoidRow.sizeCounts['1']} singletons, and its observed coverage
      is {percent(sigmoidRow.covered / 80)}%, below 90%. That observation is not logically inconsistent with the
      marginal theorem: this is one realised conformal split and a finite assessment sample. Conversely
      isotonic's {isotonicRow.covered} of 80 does not prove a population guarantee or establish it as the
      universally better calibrator. Its threshold is {num(isotonicRow.q)} in this run because many conformal
      scores tie at zero, and ties are precisely why a no-ties upper bound is inapplicable.</Prose>
    <Prose>All four SVC-derived scores have an AUC of {num(sigmoidRow.auc)} on
      these {classification.roleSizes.test} observations, despite different Brier losses and different set
      behaviour. The prior baseline returns both labels for every observation: coverage is perfect and the sets
      convey no class distinction at all. That baseline makes the usefulness question explicit.</Prose>
    <Prose>There is also a numerical lesson in that constant baseline. Its class-1 probability
      is <Math>{'103/240'}</Math>. Computing <Code>1 - probability &lt;= q</Code> includes the equality
      at <Math>{'q=1-103/240'}</Math>. Rearranging it to <Code>probability &gt;= 1 - q</Code> can fail by one
      floating-point rounding unit, because the second subtraction is not exact: on this machine the first
      comparison is {String(boundary.scoreComparison)} and the second
      is {String(boundary.rearrangedComparison)}. We use the same score computation for calibration and for
      prediction, and the small program retains this boundary fixture. Algebraic equivalence does not excuse
      dropping a tied boundary in an implementation.</Prose>

    <H3>Regression: sound pressure in an airfoil experiment</H3>
    <Prose>The Airfoil Self-Noise dataset records wind-tunnel measurements. The inputs
      are {provenance.airfoil.features.join(', ')}. The target
      is <strong>{provenance.airfoil.target}</strong>, the quantity the dataset supplies; an interval width in dB
      is not a linear acoustic-pressure difference. UCI lists {provenance.airfoil.sourceRows} observations. We
      retain {provenance.airfoil.retainedRows} randomly ordered
      rows: {regression.roleSizes.train} for fitting, {regression.roleSizes.conformal} for conformal calibration
      and {regression.roleSizes.test} for final assessment.{' '}
      <a href={provenance.airfoil.record}>The UCI record</a> carries the data description and
      attribution.</Prose>
    <Prose>We compare a constant predictor with absolute-residual intervals, a standardised ridge model with
      absolute-residual intervals, raw {regression.parameters.quantileLevels.join('/')} gradient-boosted quantile
      estimates, and CQR around those same quantile estimates. The quantile models
      have {regression.parameters.trees} trees of depth {regression.parameters.maxDepth} and fixed seeds; no test
      outcomes selected these settings. CQR uses rank {regression.rank} of {regression.roleSizes.conformal} scores,
      so its <Math>{'q'}</Math> is {num(regression.qCqr)} dB and it expands both raw endpoints by that
      amount.</Prose>

    <MeasuredIntervalsFigure />

    <Prose>CQR improves this sample's coverage over the unadjusted quantiles, from {rawQuantileRow.covered} to{' '}
      {cqrRow.covered} of {cqrRow.n}, and remains slightly narrower on average than the ridge intervals. It also
      covers fewer observed responses than the ridge procedure does. The table preserves that tradeoff: shorter
      intervals are not automatically better if they miss the outcomes the task needs to cover. Ridge's point mean
      absolute error is about {num(regression.pointMae)} dB compared with {num(regression.constantMae)} for the
      training-mean baseline, and point-error metrics and interval metrics answer different questions.</Prose>
    <Prose>The randomised allocation supports an exchangeability interpretation for predicting a randomly selected
      held-out row from this fixed corpus, conditional on the separately fitted model. Measurements share
      experimental conditions, so this does not establish an iid sample of future wind-tunnel campaigns, or
      performance on a new airfoil design. An intended deployment on new experimental settings would need that
      grouping to determine the split and the validation unit. This is a concrete distinction between a
      reproducible corpus experiment and its possible real-world generalisation.</Prose>
    <Prose>Frequency groups below {regression.frequencySplitHz} Hz and at least {regression.frequencySplitHz} Hz
      are declared diagnostic slices. For ridge the two coverage counts
      are {ridgeRow.frequencyGroups.below_2000_hz.covered}/{ridgeRow.frequencyGroups.below_2000_hz.n}{' '}
      and {ridgeRow.frequencyGroups.at_least_2000_hz.covered}/{ridgeRow.frequencyGroups.at_least_2000_hz.n},
      despite an overall {ridgeRow.covered}/{ridgeRow.n}. The same overall score hides quite different local
      behaviour.</Prose>

    {/* ============================================================ §8 */}
    <H2>{headings[7]}</H2>

    <ProbabilityLayersFigure />

    <Prose>At a .9 target, a reported 92 of 100 is an observation, not a proof. A reported 87 of 100 is a reason
      to examine uncertainty, data roles, dependence, score versions and shifts — not by itself a mathematical
      refutation. Under iid continuous scores and a fixed independently trained model, the conditional coverage
      for the <Math>{'k'}</Math>th order statistic follows a <Math>{'\\mathrm{Beta}(k,n+1-k)'}</Math> distribution
      when <Math>{'k\\le n'}</Math>, with mean <Math>{'k/(n+1)'}</Math>; that explains variation between
      calibration samples. A finite iid assessment set adds binomial
      variation <strong>conditional on the fixed threshold</strong>. Marginally, its indicators share a random
      threshold and should not be treated as independent Bernoulli trials with parameter
      exactly <Math>{'1-\\alpha'}</Math>. None of these continuous-score statements describes tied isotonic scores
      or our finite-corpus sampling scheme.{' '}
      <a href="https://arxiv.org/html/2107.07511v6">The tutorial</a> develops the calibration-size and assessment
      analysis.</Prose>

    <H3>Marginal coverage does not protect every group</H3>
    <Prose>If {percent(fixtures.mosaic[0].share, 0)}% of a population belongs to group A with coverage{' '}
      {num(fixtures.mosaic[0].coverage)}, and {percent(fixtures.mosaic[1].share, 0)}% belongs to group B with
      coverage {num(fixtures.mosaic[1].coverage)}, the marginal coverage is {num(mosaic.marginal)}. Half of group
      B's answers are missed.</Prose>

    <GroupMosaicFigure />

    <Prose>A <strong>group-conditional</strong> or Mondrian construction can compute a separate rank threshold
      within each predeclared group. The group definition and the score must be fixed independently of that
      group's conformal outcomes, and the calibration and future examples need the appropriate within-group
      exchangeability. Small groups can require infinity to support a demanding coverage target. A guarantee for
      several specified groups is not a guarantee conditional on every possible <Math>{'x'}</Math>, nor on every
      overlapping slice chosen after inspecting failures.</Prose>
    <Prose>Class-conditional conformal prediction is a related construction:
      compute <Math>{'q_k'}</Math> from calibration observations whose true class is <Math>{'k'}</Math>, then test
      candidate class <Math>{'k'}</Math> against <Math>{'q_k'}</Math> at prediction. The future label is unknown,
      but each candidate uses its own class threshold. This is quite different from the mistaken class-calibration
      plot of section 1, which discards all the negative outcomes for a class.</Prose>

    <H3>A singleton is a selection event</H3>
    <Prose>Keeping only singleton sets changes the population under discussion. Ordinary marginal coverage does
      not guarantee the same rate among those retained predictions. A selective system should report how many
      observations it acts on, the error among acted-on observations, and how empty or multiple-label sets are
      handled. If that selected population needs its own guarantee, use a construction that targets the
      corresponding conditional risk; the original marginal claim cannot simply be renamed.</Prose>

    <H3>Changing the population or the procedure</H3>
    <Prose>A simple label-shift example makes the calibration issue tangible. Suppose a binary test has
      sensitivity {num(fixtures.labelShift.sensitivity)} and false-positive
      rate {num(fixtures.labelShift.falsePositiveRate)}, unchanged between populations. With event
      prevalence {num(0.5)}, the positive predictive value is {num(ppvHalf.value)}. With prevalence {num(0.1)}, it
      becomes about {num(ppvTenth.value)}. The class-conditional test behaviour stayed fixed; the posterior
      meaning of its positive result changed.{' '}
      <a href="/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations">Probability
      Distributions &amp; Bayes' Theorem</a> develops that algebra.</Prose>
    <Prose>Conformal guarantees also depend on the relevant joint sampling structure. A new population, a changed
      preprocessing rule, a refitted model, an updated probability map, or a score selected using conformal labels
      can each invalidate the old threshold's argument. Re-estimating a threshold on recent data helps only if the
      intended sampling assumptions and selection procedure are appropriate. For sequential data, a time-aware
      method must state its own guarantee; repeatedly shuffling observations does not turn next-month forecasting
      into the original exchangeable setting.</Prose>

    <Table caption="What an acceptable uncertainty report names"
      headings={['item', 'why it is in the report']}
      rows={[
        ['the outcome and the unit of analysis', 'a guarantee about rows is not a guarantee about campaigns'],
        ['the data-role allocation', 'which observations fitted, which calibrated, which only measured'],
        ['model and calibration versions', 'a refit changes the score function and retires the threshold'],
        ['the score definition', 'the same target coverage under two scores is two different procedures'],
        ['alpha, the rank and the threshold', 'the rank is exact and reproducible; a percentile is a convention'],
        ['the equality policy', 'ties can only enlarge coverage under the weak comparison'],
        ['coverage counts, not rates alone', 'a rate without a denominator cannot be judged'],
        ['the set-size or width distribution', 'perfect coverage with uninformative sets is not useful'],
        ['important slices', 'marginal coverage never promised any of them'],
        ['the sampling interpretation', 'what exchangeability is being assumed, and over what'],
        ['what will trigger a new evaluation', 'a threshold is only as current as the procedure it came from'],
      ]}
      footnote={'No universal ECE threshold, percentage drop or mandatory calibration method replaces that '
        + 'reasoning.'} />

    {/* ============================================================ §9 */}
    <H2>{headings[8]}</H2>
    <Prose><strong>This section is a deeper branch.</strong> It assumes sections 1–8 and changes one specific part
      of the procedure at a time: the score, the data reuse, the sampling assumption, or the loss being
      controlled.</Prose>

    <H3>Adaptive classification sets and regularisation</H3>
    <Prose>The score <Math>{'1-p_y'}</Math> tests each class against a common probability
      cutoff. <strong>Adaptive prediction sets</strong>, or APS, instead use the cumulative probability mass up to
      a candidate class after sorting classes by descending probability. The score incorporates the competition
      among labels rather than only one probability.</Prose>

    <ApsFigure />

    <Prose>RAPS adds a non-negative penalty to cumulative scores for labels beyond a chosen rank, encouraging
      efficiency in large label spaces. Penalty and rank choices consume tuning information: freeze them before
      ordinary conformal calibration, or use an analysis that accounts for their selection. Better adaptivity is a
      design aim, not a universal pointwise coverage guarantee.{' '}
      <a href="https://arxiv.org/html/2107.07511v6">The tutorial</a> discusses the construction and its boundary
      choices.</Prose>

    <H3>Reuse data with a different conformal procedure</H3>
    <Prose>Full conformal prediction treats a proposed test label symmetrically with the training observations and
      recomputes the scores in that augmented sample. It can avoid a single held-out split, but naive
      implementations may refit for every test-and-candidate-label combination. Continuous regression labels
      require additional computational structure or a justified approximation, not a finite class loop.</Prose>
    <Prose>Jackknife+ fits leave-one-out models and combines each omitted observation's residual with that model's
      prediction at the new <Math>{'x'}</Math>. This is not merely adding leave-one-out residuals around one final
      fit. Under exchangeability and a symmetric fitting algorithm, its basic worst-case guarantee
      is <Math>{'1-2\\alpha'}</Math> in the original parameterisation, although performance can be closer
      to <Math>{'1-\\alpha'}</Math> under further conditions. CV+ substitutes fold-specific fits and has its own
      finite-sample correction. Neither should inherit the split-conformal <Math>{'1-\\alpha'}</Math> formula by
      analogy. Linear least-squares shortcuts need their own algebra, and ordinary logistic regression does not
      have a general exact ridge-style hat-matrix leave-one-out formula.{' '}
      <a href="https://stat.cmu.edu/~ryantibs/papers/jackknife.pdf">Barber and colleagues</a> give the
      construction and the theorems.</Prose>

    <H3>Shift, unusual applications, and what the guarantee targets</H3>
    <Prose>Under covariate shift, <Math>{'P(X)'}</Math> changes while <Math>{'P(Y\\mid X)'}</Math> stays fixed.
      Weighted conformal methods can account for known density ratios and support conditions using a weighted
      score distribution, including weight associated with the new input. Estimated ratios add an estimation
      problem, and arbitrary dataset shifts are not repaired by putting weights on the old quantile. The original
      weighted-conformal study itself uses airfoil measurements, which makes it a useful next experiment after
      this lesson's unweighted protocol.{' '}
      <a href="https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf">Tibshirani
      and colleagues</a> set out the setting.</Prose>
    <Prose>One useful extension is anomaly detection. If larger scores indicate more unusual behaviour, a new
      score <Math>{'s'}</Math> can be compared with <Math>{'n'}</Math> exchangeable reference scores
      using <Math>{'p=(1+\\#\\{\\text{reference scores}\\ge s\\})/(n+1)'}</Math>. Flagging a small{' '}
      <Math>{'p'}</Math> controls a marginal false-flag probability for a new in-distribution observation under
      that setup. It does not establish that a flagged item is faulty, and many simultaneous or repeated flags
      require a corresponding multiple-testing or sequential analysis. The extra one and the tie direction echo
      the same rank logic as prediction sets.</Prose>
    <Prose>For structured predictions, the cost of an omission may matter more than whether an entire set contains
      an exact label. <strong>Conformal risk control</strong> replaces the binary miscoverage loss with a
      specified bounded monotone loss over a nested family, under its stated conditions. A segmentation task might
      care about the fraction of missed relevant pixels. A selective classifier might instead care about errors
      among accepted cases; that loss need not satisfy the basic monotonicity condition and requires an
      appropriate procedure for the selected population, potentially a more general risk-control framework. Define
      the loss and the randomness being controlled before translating “90%” into a product claim. Venn predictors
      are another probabilistic framework, and are not a synonym for ordinary group-conditional conformal sets.{' '}
      <a href="https://arxiv.org/html/2107.07511v6#S4">Section 4 of the tutorial</a> covers the risk-control and
      outlier-detection branches.</Prose>

    <H3>Practical libraries and cost</H3>
    <Prose>MAPIE 1.5 provides <Code>SplitConformalClassifier</Code> and <Code>SplitConformalRegressor</Code>. A
      fitted estimator with <Code>prefit=True</Code> is followed by <Code>conformalize</Code> on the separate data
      and then <Code>predict_set</Code> or <Code>predict_interval</Code>. Choose the conformity score explicitly
      and inspect the output shape: a classification set array has axes for observation, class and confidence
      level, and its classes must be aligned with the estimator's class order rather than assumed to be
      consecutive integers. The <a href="https://mapie.readthedocs.io/en/stable/api/classification/">current
      classifier API</a> and <a href="https://mapie.readthedocs.io/en/v1.5.0/content/getting-started/quick-start/">quick
      start</a> offer a library route after the exact from-scratch rank calculation. This page's executed programs
      use NumPy and scikit-learn only; the MAPIE route is documentation-based guidance and is not a claimed
      executed benchmark here.</Prose>
    <Prose>A frozen sigmoid or temperature map is small, but fitting involves optimisation iterations. Isotonic
      needs sorting and stored knots, and a fold ensemble retains several base models. Split conformal requires
      calibration predictions and an order statistic, plus storing or summarising the scores according to the
      application. Class-set construction evaluates <Math>{'K'}</Math> candidates per observation for the simple
      score, while APS additionally sorts the classes. Local scale or quantile models add their own training and
      inference costs. The overhead can be modest compared with the base model, but it is measurable and should
      not be described as always zero.</Prose>

    {/* ============================================================ §10 */}
    <H2>{headings[9]}</H2>
    <Prose>Work through 1–6 before the deeper problems. Each one changes a quantity or an assumption rather than
      repeating a trace above it.</Prose>

    <Practice title="1. A misleading confidence diagram"
      question={<>In two equally common groups, forecasts for class 1
        are {num(fixtures.practiceForecastGroups[0].forecast)} and {num(fixtures.practiceForecastGroups[1].forecast)},
        but the true positive rates
        are {num(fixtures.practiceForecastGroups[0].positiveRate)} and {num(fixtures.practiceForecastGroups[1].positiveRate)}.
        What does a top-confidence diagram show? What does a class-1 diagram reveal?</>}
      hint={<>The first group predicts class 0, so its correctness is one minus its positive rate.</>}>
      <Prose>Both groups have top confidence {num(0.9)}. Their correctness rates are {num(0.8)} and {num(1)},
        averaging {num(0.9)}, so the confidence-only diagram matches the diagonal exactly. The class-1 points
        are ({num(0.1)}, {num(0.2)}) and ({num(0.9)}, {num(1)}), each {num(0.1)} above the diagonal. The different
        conditioning quantities conceal miscalibration when merged. Figure 1 has this exact pair as its second
        setting, so you can compare both readings side by side.</Prose>
    </Practice>

    <Practice title="2. Construct a different isotonic fit"
      question={<>Scores <Math>{'[1,2,3,4,5,6]'}</Math> have labels <Math>{'[0,1,0,1,0,1]'}</Math>. Find the
        monotone least-squares probabilities. Explain whether permuting the input row order changes the
        result.</>}
      hint={<>Pool adjacent violations after sorting by score, and keep each block's count.</>}>
      <Prose>The fit is <Math>{'[' + practicePav.fitted.map(value => num(value)).join(',') + ']'}</Math>. Each
        1-then-0 pair averages to {num(0.5)}; adjacent blocks whose means are equal may remain separately
        represented without changing the predictions, because a tie is not a violation. Row order has no effect
        when the scores and labels stay attached and ties are handled correctly. This fit is not an independent
        demonstration of population calibration. Investigation 2 has this fixture as a preset, and also a
        “reverse the row order” button whose result is an exact null.</Prose>
    </Practice>

    <Practice title="3. Compute a rank without a percentile shortcut"
      question={<>Fourteen calibration scores are the integers 1 through 14.
        For <Math>{'\\alpha=.2'}</Math>, compute <Math>{'k'}</Math> and <Math>{'q'}</Math>.
        For <Math>{'\\alpha=.02'}</Math>, what changes? Is the <Math>{'k'}</Math>th score counted from the largest
        or from the smallest?</>}
      hint={<>Include the unseen next observation in the rank denominator.</>}>
      <Prose><Math>{'k=\\lceil 15\\times.8\\rceil=' + practiceRankTwenty}</Math>,
        so <Math>{'q=' + practiceRankTwenty}</Math> — the {practiceRankTwenty}th <em>smallest</em> score,
        which on these particular data happens to equal the rank itself.
        At <Math>{'\\alpha=.02'}</Math>, <Math>{'k=\\lceil 14.7\\rceil=' + practiceRankTwo}</Math>, larger than
        the fourteen available scores, so <Math>{'q'}</Math> is infinity and every candidate answer is included.
        Clipping to 14 is a different procedure and loses the rank argument that justified the target.</Prose>
    </Practice>

    <Practice title="4. Turn class scores into a set"
      question={<>A three-class predictor
        returns <Math>{'[' + fixtures.practiceVector.map(value => num(value)).join(',') + ']'}</Math>. A separately
        computed threshold is <Math>{'q=' + num(fixtures.practiceQ)}</Math> for the
        score <Math>{'1-p_y'}</Math>. Which classes are included? What happens
        if <Math>{'q'}</Math> decreases to {num(0.6)}? Does a singleton then
        have {percent(1 - fixtures.defaultAlpha, 0)}% conditional correctness?</>}
      hint={<>Use the weak inequality on the
        scores <Math>{'[' + practiceVectorSets.scores.map(value => num(value)).join(',') + ']'}</Math>.</>}>
      <Prose>At {num(fixtures.practiceQ)} the set
        is {'{' + practiceVectorSets.included.map((keep, index) =>
          (keep ? fixtures.classNames[index] : null)).filter(Boolean).join(', ') + '}'}, including the class whose
        score is exactly equal to the threshold. At {num(0.6)} only{' '}
        {'{' + practiceVectorNarrower.included.map((keep, index) =>
          (keep ? fixtures.classNames[index] : null)).filter(Boolean).join(', ') + '}'} remains. Its singleton
        status does not establish {percent(1 - fixtures.defaultAlpha, 0)}% correctness conditional on that status
        or on that input. The nominal conformal coverage belongs to the sampling procedure and the specified
        score, not to each displayed set.</Prose>
    </Practice>

    <Practice title="5. A negative CQR adjustment"
      question={<>An initial interval
        is <Math>{'[' + fixtures.cqrBase.lower + ',' + fixtures.cqrBase.upper + ']'}</Math>. Five calibration CQR
        scores are <Math>{'[' + fixtures.cqrScores.join(',') + ']'}</Math> and <Math>{'\\alpha=' + num(fixtures.cqrAlpha)}</Math>.
        Find the conformalised interval. Why can the score be negative?</>}
      hint={<><Math>{'k=\\lceil 6\\times.6\\rceil'}</Math>, and the score measures how far a label lies beyond — or
        within — the endpoints.</>}>
      <Prose><Math>{'k=' + practiceCqr.k}</Math> and <Math>{'q=' + num(practiceCqr.q)}</Math>. The final interval
        is <Math>{'[10-(-2),\\,20+(-2)]=[' + num(practiceCqrInterval.lower) + ',' + num(practiceCqrInterval.upper) + ']'}</Math>.
        Negative scores arise when outcomes fall inside their initial intervals, so calibration can shrink excess
        width. This constructed sample does not prove that every future response lies in the shrunken interval,
        and if the endpoints had crossed the answer would be an empty set rather than a negative width.
        Investigation 4's CQR branch has a preset that makes exactly that happen.</Prose>
    </Practice>

    <Practice title="6. A data-flow bug"
      question={<>A team fits a neural network, chooses an epoch using validation labels, fits temperature on
        those same labels, and computes a split-conformal threshold from the same observations. A new independent
        test set is untouched. Which step needs redesign for the ordinary split-conformal argument?</>}
      hint={<>Consider the entire final score function, not only the original network weights.</>}>
      <Prose>The conformal observations helped select the model and fit the temperature, so they are not scored
        symmetrically with new test examples by an independently fixed rule. Complete the selection and the
        probability fitting first, then use separate conformal observations — or use a specifically justified
        alternative procedure. An untouched test set measures the resulting system, but it does not retroactively
        validate the threshold construction. Figure 5 is the four-lane version of exactly this argument.</Prose>
    </Practice>

    <Practice title="7. Evaluate the acoustic intervals"
      question={<>Ridge intervals cover {ridgeRow.covered} of {ridgeRow.n} observations,
        with {ridgeRow.frequencyGroups.below_2000_hz.covered}/{ridgeRow.frequencyGroups.below_2000_hz.n} below
        2,000 Hz and {ridgeRow.frequencyGroups.at_least_2000_hz.covered}/{ridgeRow.frequencyGroups.at_least_2000_hz.n} above.
        Write a two-sentence interpretation and a useful next development action, without calling either empirical
        fraction a theorem.</>}
      hint={<>Separate the overall measurement, the subgroup measurement, and the sampling interpretation.</>}>
      <Prose>The observed overall coverage is {percent(ridgeRow.covered / ridgeRow.n)}%, while the lower-frequency
        slice is about {percent(ridgeRow.frequencyGroups.below_2000_hz.covered / ridgeRow.frequencyGroups.below_2000_hz.n)}%
        and the higher-frequency slice
        is {percent(ridgeRow.frequencyGroups.at_least_2000_hz.covered / ridgeRow.frequencyGroups.at_least_2000_hz.n)}%
        on this fixed assessment sample. These measurements identify uneven behaviour; they neither prove exact
        population coverage nor supply a conditional guarantee for either slice.</Prose>
      <Prose>A useful next development experiment examines residual patterns and the relevant experimental groups,
        designs a scale-normalised, quantile-based or group-specific procedure using development data, and then
        obtains an appropriate new independent assessment — rather than tuning on this final table.</Prose>
    </Practice>

    <Practice title="8. Can a narrower set be worse?"
      question={<>One system always returns all five labels. Another returns one label per observation but omits
        the true label on 30% of them. Which uncertainty report is more useful, and what is missing from that
        question?</>}
      hint={<>There is no single scalar objective without a task and an error tolerance.</>}>
      <Prose>The first has perfect coverage and no discrimination; the second is selective but misses many
        answers. Compare coverage, size, the consequence of an omission, the downstream action and the requested
        guarantee under a declared task. Neither perfect coverage nor minimal set size alone determines
        usefulness. A method that targets an appropriate loss or acceptance policy may be needed — and the
        training-prior row of section 7 is the first system, measured: it covers all {classification.roleSizes.test} rows
        and returns both labels every time.</Prose>
    </Practice>

    {/* ============================================================ §11 */}
    <H2>{headings[10]}</H2>
    <Prose>You are ready to move on when you can construct and read a reliability diagram with its counts, explain
      why merging bins can drive a binned ECE to zero without changing a model, fit a monotone map by pooling
      adjacent violators with their weights, name the four jobs a label can do in one experiment, calculate the
      exact split-conformal rank including its infinity case, form a class set or a physical interval from a
      threshold, and separate a marginal guarantee from a measured count.</Prose>

    <Table caption="Readiness check"
      headings={['you should be able to', 'where it was taught']}
      rows={[
        ['Say what a probability forecast conditions on, and read a reliability diagram in the stated direction',
          'Section 1, figure 1; section 2'],
        ['Distinguish a class-1 diagram from a top-confidence diagram, and say what merging them hides',
          'Section 1, figure 1, practice 1'],
        ['Separate calibration from resolution, and price the difference against a stated cost',
          'Section 1, figure 2'],
        ['Report an empty bin as having no observed fraction, rather than a fraction of zero',
          'Section 2, investigation 1'],
        ['Make a binned ECE vanish by rebinning, and explain why the model did not change',
          'Section 2, investigation 1'],
        ['Pool adjacent violators with their counts, and group equal scores before merging',
          'Section 3, figure 3, investigation 2, practice 2'],
        ['Say what a sigmoid offset can change, and what temperature scaling cannot',
          'Section 3, figure 4'],
        ['Trace four label roles and name which step a leak entered', 'Section 4, figure 5, practice 6'],
        ['Compute the exact rank including its infinity case, and not reach for a percentile',
          'Section 5, figure 6, investigation 3, practice 3'],
        ['Form a class set under the weak comparison, including at exact equality',
          'Section 5, investigation 3, practice 4'],
        ['Explain the n+1 by the combined rank of the unseen example', 'Section 5, investigation 3'],
        ['Turn a dimensionless threshold into a physical half-width, and say what a change of units does not do',
          'Section 6, figure 7, investigation 4'],
        ['Read a negative CQR adjustment, and recognise an empty set rather than a negative width',
          'Section 6, investigation 4, practice 5'],
        ['Read a measured comparison where the best-covering method is not the most useful',
          'Section 7, figures 8 and 9, practice 8'],
        ['Keep marginal, conditional and observed coverage apart', 'Section 8, figures 10 and 11, practice 7'],
        ['Say why a singleton-only report is a selection event', 'Section 8'],
      ]} />

    <Prose>The next topic in this module
      is <a href="/learn/path/full-curriculum/rademacher-complexity-generalization-bounds?module=classical-ml">Rademacher
      Complexity &amp; Generalization Bounds</a>. It returns to the generalisation question from PAC and VC with a
      complexity measure sensitive to the sampled inputs. The connection is the habit you have just practised:
      define the random object, say what was fitted using which information, and name the exact event that a
      probability statement controls.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; each of these
      offers a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://www.youtube.com/watch?v=nql000Lu_iE">A Tutorial on Conformal Prediction</a> and{' '}
        <a href="https://www.youtube.com/watch?v=TRx4a2u-j7M">Part 2: Conditional Coverage and Diagnostics</a>,
        the authors' own recorded talks, identified from their{' '}
        <a href="https://stephenbates19.github.io/videos.html">official video listing</a>. Reconstruct the ten-card
        rotation in investigation 3 before moving to the second talk, then compare its conditioning language with
        the frequency slices of section 7. Video, roughly an hour each; the recordings were identified from the
        authors' page rather than watched through, so treat the timings as unverified.</li>
      <li><a href="https://scikit-learn.org/stable/modules/calibration.html">Scikit-learn's calibration guide</a>.
        Read its definition and proper-score discussion before copying a wrapper. It is the reference for the
        sigmoid, isotonic and temperature contracts this lesson uses, and it is version-sensitive: the page
        documents the current release, not the one you may have installed.</li>
      <li><a href="https://arxiv.org/html/2107.07511v6">A Gentle Introduction to Conformal Prediction</a>,
        Angelopoulos and Bates. Diagrams, applications, evaluation and the finite-rank proof. Use the formal
        order-statistic definition in appendix D and attend to its footnotes about ties; a code convention and a
        theorem's assumptions have to agree. The APS, risk-control and shift branches of section 9 are its
        sections 2 and 4.</li>
    </ul></>}>
      <li><a href="https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf">Evaluating model
        calibration in classification</a>, Vaicenavicius and colleagues — why a single binned statistic misses
        part of calibration, and the estimation effects of section 2's binning discussion.</li>
      <li><a href="https://proceedings.mlr.press/v70/guo17a/guo17a.pdf">On Calibration of Modern Neural
        Networks</a>, Guo and colleagues — the temperature method and an actual empirical study. Its findings
        describe the models and datasets examined there; they are not a property of every later architecture.</li>
      <li><a href="https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf">Conformalized
        Quantile Regression</a>, Romano, Patterson and Candès, and{' '}
        <a href="https://stat.cmu.edu/~ryantibs/papers/jackknife.pdf">Predictive inference with the jackknife+</a>,
        Barber, Candès, Ramdas and Tibshirani, and{' '}
        <a href="https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf">Conformal
        prediction under covariate shift</a>, Tibshirani and colleagues — the three papers behind section 6 and
        section 9, each changing one specific part of the procedure.</li>
      <li><a href="https://mapie.readthedocs.io/en/stable/api/classification/">MAPIE 1.5 classifier API</a> —
        a library route after the from-scratch rank. Documentation inspected; the library is not installed or
        executed anywhere on this page.</li>
      <li><a href={provenance.banknote.doi}>UCI Banknote Authentication</a>, {provenance.banknote.creator}, and{' '}
        <a href={provenance.airfoil.doi}>UCI Airfoil Self-Noise</a>, {provenance.airfoil.creator}, both
        licensed <a href={provenance.airfoil.licenseUrl}>{provenance.airfoil.license}</a> — the actual
        observations. This page serves its own copies,{' '}
        <a href={provenance.files['banknote-subset.csv'].file} download>banknote-subset.csv</a>{' '}
        ({provenance.files['banknote-subset.csv'].bytes.toLocaleString('en-US')} bytes,
        SHA-256 <Code>{provenance.files['banknote-subset.csv'].sha256}</Code>) and{' '}
        <a href={provenance.files['airfoil-subset.csv'].file} download>airfoil-subset.csv</a>{' '}
        ({provenance.files['airfoil-subset.csv'].bytes.toLocaleString('en-US')} bytes,
        SHA-256 <Code>{provenance.files['airfoil-subset.csv'].sha256}</Code>), beside
        their <a href={provenance.attribution}>attribution</a>.</li>
    </Sources>

    <Prose>The forecast cards, the eight-score fixture, the nine calibration scores, the residual-and-scale pairs,
      the group mosaic and every practice fixture on this page
      are <strong>exact constructed calculations</strong>, not measurements. The banknote and airfoil results
      are calculations on the two identified real datasets under one declared protocol, with
      the {classification.roleSizes.test} and {regression.roleSizes.test} reserved assessment rows never reachable
      from any investigation on this page — every lab above runs on constructed fixtures only. Every coverage
      figure in section 7 is a count with its denominator attached, on one realised split. None of it is a
      benchmark, and none of it is a claim about any future dataset.</Prose>
    <KindTag kind="assessment"
      extra={`${classification.roleSizes.test} banknote rows and ${regression.roleSizes.test} airfoil rows, each `
        + 'scored exactly once, after every method and threshold had been fixed.'} />
  </div>,
};

export default calibrationContent;
