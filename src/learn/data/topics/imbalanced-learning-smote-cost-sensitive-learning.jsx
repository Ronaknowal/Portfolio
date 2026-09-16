import { Callout, H2, H3, Prose, Code } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  CostCrossingLab, ScoreQueueLab, SmoteGeometryLab, TuningQueueLab, WeightedScoreLab,
} from '../../components/lesson-labs/ImbalanceLabs.jsx';
import {
  CaseFlowFigure, LossMassFigure, OutcomesFigure, PipelineFigure, PrevalenceFigure, WeightedStepFigure,
} from '../../components/lesson-labs/ImbalanceFigures.jsx';
import { imbalanceExamples } from '../imbalance-examples.js';
import { inspectionRecords, methods, provenance, roles, study } from '../imbalance-data.js';
import {
  actionRisks, averagePrecisionOf, balancedWeights, confusion, costOf, countsAt, fixtures, focalMass,
  inverseWeightedOptimum, populationFlow, priorShift, rankedQueue, rowExpansion, smoteConstruction,
  weightedOptimum, weightedStep,
} from '../imbalance-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');

const model = confusion(fixtures.modelCounts);
const baseline = confusion(fixtures.baselineCounts);
const ladderQueue = averagePrecisionOf(fixtures.averagePrecisionQueue);
const flowA = populationFlow(fixtures.flowA);
const flowB = populationFlow(fixtures.flowB);
const risk = actionRisks(fixtures.casePosterior, fixtures.costs.costFP, fixtures.costs.costFN);
const practiceRisk = actionRisks(fixtures.practiceCosts.posterior,
  fixtures.practiceCosts.costFP, fixtures.practiceCosts.costFN);
const step = weightedStep({ rows: fixtures.stepRows, rate: fixtures.stepRate, penalty: fixtures.stepPenalty });
const balanced = balancedWeights([study.fittingNegatives, study.fittingPositives]);
const optimum = weightedOptimum(fixtures.weighted.probability,
  fixtures.weighted.positiveWeight, fixtures.weighted.negativeWeight);
const practiceOptimum = weightedOptimum(fixtures.weightedPractice.probability,
  fixtures.weightedPractice.positiveWeight, fixtures.weightedPractice.negativeWeight);
const smote = smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup });
const focal = focalMass(fixtures.focal);
const prior = priorShift(fixtures.prior);
const practicePrior = priorShift(fixtures.priorPractice);
const expansion = rowExpansion(fixtures.expansion.majority, fixtures.expansion.minority);
const practiceExpansion = rowExpansion(fixtures.expansionPractice.majority, fixtures.expansionPractice.minority);

const outcomes = methods.map(method => ({
  method,
  tuned: countsAt(inspectionRecords.labels, method.inspectionScores, method.chosenThreshold),
  atHalf: countsAt(inspectionRecords.labels, method.inspectionScores, 0.5),
  queue: rankedQueue(inspectionRecords.sourceIds, inspectionRecords.labels, method.inspectionScores, 10),
}));
const costOfRow = row => costOf(row.tuned, study.costFalsePositive, study.costFalseNegative);
const bestCost = outcomes.reduce((best, row) => (costOfRow(row) < costOfRow(best) ? row : best));
const bestAp = outcomes.reduce((best, row) => (row.method.averagePrecision > best.method.averagePrecision ? row : best));
/* `Math` in this module is the KaTeX component imported above, not the global
   object: using `Math.max` here resolves to that component and throws on first
   paint. These two helpers keep the extremes explicit and shadow-proof. */
const largest = values => values.reduce((best, value) => (value > best ? value : best));
const smallest = values => values.reduce((best, value) => (value < best ? value : best));
const bestTopTen = largest(outcomes.map(row => row.queue.positives));
const topTenWinners = outcomes.filter(row => row.queue.positives === bestTopTen);
const originalRow = outcomes.find(row => row.method.name === 'original');
const smoteRow = outcomes.find(row => row.method.name === 'smote');
const weightedRow = outcomes.find(row => row.method.name === 'balanced_weight');

const headings = [
  '1. A rare class is a description, not a diagnosis',
  '2. Read the mistakes and the ranked queue',
  '3. Choose an action from its expected consequences',
  '4. Reweighting changes what the fitted score means',
  '5. SMOTE creates a geometric assumption',
  '6. Fit the sampler only where learning is allowed',
  '7. An observed-data study: rare protein localization',
  '8. Deeper choices: where the simple picture changes',
  '9. Practice: make the decision yourself',
  '10. Readiness and the next lesson',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}
function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="imb-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const imbalancedLearningContent = {
  title: 'Imbalanced Learning: SMOTE, Cost-Sensitive Learning & Rare-Event Decisions',
  readTime: '~60 min first pass · ~110 min complete read + 60–100 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot imb-lesson">
    <LessonIntro prerequisites={<>The train/validation distinction from <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a>, an average, and a logistic prediction. Confusion counts, decision costs and the weighted-score notation are introduced here. The preceding <a href="/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml">Bias&ndash;Variance &amp; Learning Curves</a> lesson supplies the sense in which a fitted model changes with its training data.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A laboratory has time to investigate only a small number of candidate proteins. A classifier that labels every protein &ldquo;not our target&rdquo; can be accurate most of the time and contribute nothing. You will move a gate through an editable queue of scored records and watch precision get <em>worse</em> as the gate rises, cross two expected-cost lines, turn a weighted score back into the probability it came from, generate a synthetic point that lands exactly on an observed member of the other class, and then read a real five-procedure comparison on {provenance.studyRows.toLocaleString('en-US')} yeast proteins in which the cost winner, the ranking winner and the top-ten winner are three different answers. Every investigation asks for a recorded prediction before it calculates anything, and retires that prediction the moment an input changes.
    </LessonIntro>
    <div className="imb-route"><Prose><strong>First pass.</strong> Follow sections 1&ndash;7, including the small score queue, the cost calculation, the weighted-probability example and the geometric SMOTE investigation. Section 7 joins them in a complete observed-data study with actual outcomes. Try practice 1&ndash;7 before opening solutions. Section 8 and practice 8&ndash;10 are deeper branches. You need the earlier train/validation distinction, averages and a logistic prediction; confusion counts, costs and the new probability notation are introduced here.</Prose></div>

    <Prose>A laboratory has time to investigate only a small number of candidate proteins. Most belong to common cellular locations; the location of interest is rare. A classifier that labels every protein &ldquo;not our target&rdquo; can be accurate most of the time and still contribute nothing to the investigation. Yet flagging everything is not a solution either: it consumes the entire laboratory budget.</Prose>
    <Prose>The preceding <a href="/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml">Bias&ndash;Variance &amp; Learning Curves</a> lesson asked how a fitted model changes with its training data. Here we add another question: <strong>which observations and mistakes should influence the learning procedure, and which action should follow a score?</strong> Changing training data, changing a loss and changing a decision threshold are three different operations.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>In <strong>class imbalance</strong>, the target classes occur at different frequencies in the dataset. We call the event of interest positive and the other class negative. Positive does not mean good, and it need not always be the smaller class. The <strong>prevalence</strong> <Math>{'\\pi'}</Math> is the positive fraction in the population or sample being discussed.</Prose>
    <Prose>If 20 of 1,000 examples are positive, an always-negative baseline gets {baseline.tn} correct: {num(100 * baseline.accuracy)}% accuracy, zero detected positives. Now consider a model that detects {model.tp} positives, misses {model.fn}, and raises {model.fp} false alarms. It gets {model.tp + model.tn} correct &mdash; slightly less accurate &mdash; but may be more useful if detecting the event matters enough.</Prose>
    <LessonTable caption="The four cells of one confusion table, for the model that detects fourteen of twenty positives" headers={['Actual class', 'Predicted negative', 'Predicted positive']} rows={[
      ['Negative', `TN = ${model.tn}`, `FP = ${model.fp}`],
      ['Positive', `FN = ${model.fn}`, `TP = ${model.tp}`],
    ]} />
    <CaseFlowFigure />
    <Prose>Accuracy still has a precise meaning: the fraction of correct class decisions. It is a valid objective when every error has the same cost and the assessment population matches the intended task. It simply does not answer every rare-event decision problem. A constant predictor is a real baseline, and in a problem with indistinguishable classes and equal error costs it can even be optimal.</Prose>
    <Prose>Nor does the ratio tell you how much information is available. A 99:1 dataset might contain one positive and 99 negatives, or 1,000 positives and 99,000 negatives. The ratio is the same; the possibilities for fitting, validation and discovering positive subgroups differ greatly. Important questions include class overlap, label quality, rare subtypes, sample dependence, feature availability and the cost of each action. No universal 10%, 1% or 0.1% boundary selects the correct algorithm.</Prose>
    <Callout title="Two different problems that share a word">
      Imbalance in training and a change in deployment prevalence are also different problems. Cross-entropy can learn the correct posterior from naturally imbalanced data under suitable model, sampling and optimization conditions. Rebalancing is a modeling choice to investigate; it is not a required repair to a probability law.
    </Callout>

    <H2>{headings[1]}</H2>
    <H3>Counts become decision-specific measurements</H3>
    <Prose>From the four cells above:</Prose>
    <MathBlock>{'\\begin{gathered}\\text{precision}=\\frac{TP}{TP+FP}\\\\[4pt]=\\frac{14}{32}=.4375,\\\\[8pt]\\text{recall}=\\frac{TP}{TP+FN}\\\\[4pt]=\\frac{14}{20}=.7.\\end{gathered}'}</MathBlock>
    <Prose>Precision asks how many selected cases were positive. Recall asks how many actual positives were selected. Specificity is TN/(TN+FP), which is {model.tn}/{model.negatives} = {num(model.specificity)}; the false-positive rate, FPR, is 1&minus;specificity, here {num(model.falsePositiveRate)}. <strong>Balanced accuracy</strong> is the average of positive recall and specificity for this binary task, {num(model.balancedAccuracy)}. It gives the two actual classes equal aggregate weight, unlike ordinary accuracy&rsquo;s prevalence weighting.</Prose>
    <Prose>The harmonic summary is</Prose>
    <MathBlock>{'\\begin{gathered}F_1=\\frac{2TP}{2TP+FP+FN},\\\\[8pt]F_\\beta=\\\\[2pt]\\frac{(1+\\beta^2)TP}{(1+\\beta^2)TP+\\beta^2FN+FP}.\\end{gathered}'}</MathBlock>
    <Prose>For the example, F1 = 28/52 &asymp; {num(model.f1)}. Increasing <Math>{'\\beta'}</Math> emphasizes missed positives more strongly in this formula. It does not mean &ldquo;a false negative costs <Math>{'\\beta'}</Math> currency units&rdquo; or supply a universal monetary-cost conversion. F scores also ignore true negatives. If actual costs or capacity are available, evaluate those directly rather than assuming a particular <Math>{'F_\\beta'}</Math> encodes them.</Prose>
    <Prose>When nothing is selected, precision&rsquo;s denominator is zero. That mathematical quantity is undefined; software may report zero by a declared convention. Recall is undefined if the assessment contains no actual positives. Show the counts and the convention rather than silently adding a tiny denominator and pretending the number has its ordinary interpretation.</Prose>

    <H3>A higher threshold does not guarantee higher empirical precision</H3>
    <Prose>A score-based classifier selects cases whose score is at least a threshold <Math>{'t'}</Math>. As <Math>{'t'}</Math> rises, the selected set shrinks. The number of true positives cannot increase, so recall cannot increase on the same labeled cases. Precision can move either way because the removed cases might be positive or negative.</Prose>
    <Prose>Consider descending scores .9, .8, .7 with actual labels 0, 1, 1:</Prose>
    <LessonTable caption="One ranking, three gates: precision falls as the gate rises" headers={['Threshold, with score≥t selected', 'Selected actual labels', 'Precision', 'Recall']} rows={[
      ['.7', '0, 1, 1', '2/3', '1'],
      ['.8', '0, 1', '1/2', '1/2'],
      ['.9', '0', '0', '0'],
    ]} />
    <Prose>The highest-ranked case is a false alarm. Increasing the threshold makes precision worse throughout this particular queue. That is not an implementation error; it is what the actual ranking does.</Prose>
    <ScoreQueueLab />
    <Prose>The <a href="https://scikit-learn.org/stable/modules/classification_threshold.html">threshold-tuning guide</a> separates fitted scores from actions. Moving a threshold alone leaves the scores, their ranking and their ROC/PR curves unchanged; it selects an operating point on those curves.</Prose>

    <H3>Why prevalence changes what a false-positive rate means operationally</H3>
    <Prose>ROC plots recall/TPR against FPR. Precision&ndash;recall plots recall against precision. If the same population has prevalence <Math>{'\\pi'}</Math>, then</Prose>
    <MathBlock>{'\\begin{gathered}\\text{precision}=\\\\[2pt]\\frac{\\pi\\,TPR}{\\pi\\,TPR+(1-\\pi)FPR}.\\end{gathered}'}</MathBlock>
    <Prose>The numerator is the fraction of all cases that are true positives; the second denominator term is the fraction that are false positives. At <Math>{'\\pi=.01'}</Math>, TPR = {flowA.tpr} and FPR = {flowA.fpr}, {flowA.population.toLocaleString('en-US')} cases contain {flowA.positives} positives: {flowA.tp} are detected, and {flowA.fp} of {flowA.negatives.toLocaleString('en-US')} negatives raise false alarms. Precision is {flowA.tp}/{flowA.alerts} &asymp; {num(flowA.precision)}. A 1% false-positive rate can create more false alarms than true detections.</Prose>
    <Prose>If prevalence falls to {flowB.prevalence} <strong>while the within-class score distributions stay fixed</strong>, the same TPR/FPR gives precision &asymp; {num(flowB.precision)}. The ROC operating point stays the same under that assumption; the workload composition changes. More negatives do not mechanically inflate ROC-AUC if class-conditional score distributions are unchanged. ROC and PR emphasize different quantities, and a useful report may include both plus counts at the chosen operating point. There is no mathematically privileged 10% prevalence cutoff between them. <a href="https://mark.goadrich.com/articles/davisgoadrichpr.pdf">Davis and Goadrich</a> explain their relationship and why interpolation in PR space needs care.</Prose>
    <PrevalenceFigure />
    <Prose><strong>Average precision</strong>, AP, summarizes a scored ranking by weighting precision at each distinct score threshold by the increment in recall:</Prose>
    <MathBlock>{'AP=\\sum_k (R_k-R_{k-1})P_k.'}</MathBlock>
    <Prose>For descending labels 1, 0, 1, 0 with distinct scores, recall increases at ranks 1 and 3, each by 1/2. AP = (1/2)&middot;1 + (1/2)&middot;(2/3) = 5/6 &asymp; {num(ladderQueue.value)}. Ties are handled as a score group, not by inventing an order using labels. Scikit-learn&rsquo;s <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">AP implementation</a> uses this noninterpolated convention; a trapezoidal area under a drawn PR curve is generally different. Name the computation rather than using &ldquo;PR-AUC&rdquo; ambiguously.</Prose>
    <Prose>A constant-score predictor has AP equal to sample prevalence when positives are present. A random independent ranking has precision equal to prevalence at the population level; an individual finite ranking&rsquo;s AP need not equal prevalence exactly. AP is a ranking summary, not performance at a particular review budget, a calibration measure, or a direct cost objective.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>Suppose <Math>{'p=P(Y=1\\mid x)'}</Math> is the posterior for the intended deployment population. For a simple binary decision, declare zero cost for correct decisions, cost <Math>{'c_{FP}'}</Math> for selecting a negative and <Math>{'c_{FN}'}</Math> for missing a positive. Then</Prose>
    <MathBlock>{'\\begin{gathered}R(\\text{select}\\mid x)=(1-p)\\,c_{FP},\\\\[6pt]R(\\text{skip}\\mid x)=p\\,c_{FN}.\\end{gathered}'}</MathBlock>
    <Prose>Select when the first risk is no greater than the second. With positive costs,</Prose>
    <MathBlock>{'p\\geq\\frac{c_{FP}}{c_{FP}+c_{FN}}.'}</MathBlock>
    <Prose>At <Math>{'c_{FP}=1'}</Math>, <Math>{'c_{FN}=12'}</Math>, the threshold is 1/13 &asymp; {num(risk.cutoff)}. For a case with <Math>{'p=.1'}</Math>, selecting costs {num(risk.selectRisk)} in expectation and skipping costs {num(risk.skipRisk)}, so selecting is preferable under this declared model. The numerical costs are teaching assumptions, not laboratory prices or medical guidance. If costs are equal, .5 is the correct threshold for the true posterior, however rare the positive class is.</Prose>
    <CostCrossingLab />
    <Prose>The full rule is broader: for action <Math>{'a'}</Math> and actual class <Math>{'y'}</Math>, choose the action minimizing <Math>{'\\sum_y C(a,y)P(y\\mid x)'}</Math>. Correct actions may have nonzero costs; &ldquo;send to a human reviewer&rdquo; can be another action; costs can vary by case. Use one consistent accounting baseline. The <a href="https://cseweb.ucsd.edu/~elkan/rescale.pdf">Elkan paper</a>, especially section 1, explains why casually mixing lost opportunities and expenditures can create an incoherent cost matrix.</Prose>
    <Prose>The optimality calculation assumes the probabilities used in it match the information and population under discussion. A calibration chart checks average observed frequency among similar scores; it does not prove that a score equals the full-feature posterior for every subgroup. With estimated or misspecified scores, a validation-selected decision rule can be useful, but a theorem about the true posterior is not a guarantee for that estimator.</Prose>

    <H3>A review budget is not the same as an error-cost ratio</H3>
    <Prose>If exactly <Math>{'k'}</Math> cases can be reviewed and each detected positive has equal value, selecting the <Math>{'k'}</Math> largest <strong>true posterior probabilities</strong> maximizes expected detections, since the expectation is the sum of their probabilities. With estimated scores, evaluate the resulting top-<Math>{'k'}</Math> procedure on appropriate assessment cases. If value, harm or review time differs by case, rank by the relevant expected benefit and solve the actual constrained allocation problem; a plain probability ranking need not be optimal.</Prose>
    <Prose>Top-<Math>{'k'}</Math> and a fixed threshold differ. A fixed threshold can produce varying daily workload. Top-<Math>{'k'}</Math> fixes capacity but its cutoff moves with the day&rsquo;s scores. Declare what happens when scores tie across the capacity boundary &mdash; such as a reproducible label-independent tie breaker or a randomized policy. Do not use unseen true labels to break ties. Precision at <Math>{'k'}</Math> and recall at <Math>{'k'}</Math> describe the resulting workload, while AP aggregates many possible cutoffs.</Prose>

    <H2>{headings[3]}</H2>
    <H3>Follow the weighted loss to its gradient</H3>
    <Prose>Logistic prediction uses margin <Math>{'z=b+x^{\\top}w'}</Math> and score <Math>{'q=\\sigma(z)=1/(1+e^{-z})'}</Math>. For <Math>{'y\\in\\{0,1\\}'}</Math>, binary log loss is <Math>{'-y\\log q-(1-y)\\log(1-q)'}</Math>. Its derivative with respect to <Math>{'z'}</Math> is <Math>{'q-y'}</Math>. Weighting observation <Math>{'i'}</Math> by a positive <Math>{'a_i'}</Math> gives the normalized objective</Prose>
    <MathBlock>{'\\begin{gathered}J(b,w)=\\frac{1}{A}\\sum_i a_i\\,\\ell_i\\\\[4pt]+\\frac{\\lambda}{2}\\lVert w\\rVert_2^2,\\\\[6pt]\\ell_i=\\log(1+e^{z_i})-y_iz_i,\\\\[4pt]A=\\sum_i a_i.\\end{gathered}'}</MathBlock>
    <Prose>The intercept is unpenalized. Therefore</Prose>
    <MathBlock>{'\\begin{gathered}\\nabla_w J=\\frac1A\\sum_i a_i(q_i-y_i)x_i\\\\[4pt]+\\lambda w,\\\\[6pt]\\partial_b J=\\frac1A\\sum_i a_i(q_i-y_i).\\end{gathered}'}</MathBlock>
    <Prose>The same residual still appears; its contribution is scaled. For two observations <Math>{'x=0,y=0'}</Math> and <Math>{'x=2,y=1'}</Math>, initial <Math>{'b=w=0'}</Math> gives <Math>{'q=.5'}</Math> for both. With weights 1 and 3, the intercept gradient is ({num(step.rows[0].interceptContribution)} {num(step.rows[1].interceptContribution)})/{num(step.totalWeight)} = {num(step.interceptGradient)} and the coefficient gradient is (0 &minus; 3)/{num(step.totalWeight)} = {num(step.coefficientGradient)}. A step of size {fixtures.stepRate} gives <Math>{'b=.1'}</Math>, <Math>{'w=.3'}</Math>, with new scores about {num(step.scoresAfter[0])} and {num(step.scoresAfter[1])}. These are a single illustrative step, not a converged model or a new accuracy claim.</Prose>
    <WeightedStepFigure />
    <Prose>The common balanced-class rule uses <Math>{'a_c=n/(K n_c)'}</Math>, where <Math>{'K'}</Math> is the number of observed classes and <Math>{'n_c'}</Math> the count of class <Math>{'c'}</Math> in the <strong>fitting</strong> data. Each class then contributes equal total weight. This is a convention, not an estimate of real-world error costs. Weighting after a resampler has already created equal class counts may make &ldquo;balanced&rdquo; weights all one; using old weights afterward instead creates another objective. Check which counts the estimator actually receives.</Prose>
    <Prose>Our normalization divides by total weight <Math>{'A'}</Math>, so multiplying every <Math>{'a_i'}</Math> by the same factor leaves the full objective unchanged. A library that divides by <Math>{'n'}</Math> or uses a summed loss can change its effective regularization under that scaling unless its penalty is adjusted. The <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization lesson</a> explains why these objective conventions matter when comparing <Math>{'C'}</Math> or <Math>{'\\lambda'}</Math>.</Prose>

    <H3>Derive the population score instead of calling it calibrated</H3>
    <Prose>At an input with true positive probability <Math>{'p'}</Math>, positive class weight <Math>{'w_+'}</Math> and negative weight <Math>{'w_-'}</Math> give expected loss</Prose>
    <MathBlock>{'\\begin{gathered}\\ell(q)=-w_+p\\log q\\\\[4pt]-w_-(1-p)\\log(1-q).\\end{gathered}'}</MathBlock>
    <Prose>For <Math>{'0<p<1'}</Math> and positive weights, set the derivative to zero:</Prose>
    <MathBlock>{'\\begin{gathered}-\\frac{w_+p}{q}+\\frac{w_-(1-p)}{1-q}=0\\\\[6pt]\\Longrightarrow\\\\[4pt]q^*=\\frac{w_+p}{w_+p+w_-(1-p)}.\\end{gathered}'}</MathBlock>
    <Prose>The second derivative is positive, so this is the unique interior minimum. Endpoint probabilities give corresponding endpoints. At <Math>{'p=.1'}</Math>, <Math>{'w_+=9'}</Math>, <Math>{'w_-=1'}</Math>, <Math>{'q^*=.5'}</Math>. It does <strong>not</strong> mean a 50% event probability in the original population. Solving backward gives</Prose>
    <MathBlock>{'p=\\frac{w_-q^*}{w_+(1-q^*)+w_-q^*}.'}</MathBlock>
    <Prose>Thresholding the ideal weighted score at .5 is equivalent to thresholding <Math>{'p'}</Math> at <Math>{'w_-/(w_++w_-)'}</Math>, which is {num(optimum.equivalentProbabilityCutoff)} here. If the weights equal the intended FP/FN cost roles &mdash; <Math>{'w_-=c_{FP}'}</Math>, <Math>{'w_+=c_{FN}'}</Math> &mdash; this reproduces the simple cost decision under these population assumptions. Inverse-frequency weights generally express another cost preference. Restricted models, regularization and finite optimization can also change the ranking, not merely the intercept; inverse algebra does not automatically calibrate an arbitrary fitted estimator.</Prose>
    <WeightedScoreLab />

    <H3>Duplication and weights share a limited equivalence</H3>
    <Prose>Repeating a row <Math>{'c_i'}</Math> times is algebraically identical to integer weight <Math>{'c_i'}</Math> in an additive full-data loss <strong>with matching normalization and regularization</strong>. Random oversampling gives random repetition counts, so it does not exactly equal uniform class weighting in every run. Mini-batch composition, early stopping and stateful operations can break a practical training equivalence even when full-data objectives agree. Weighting can save storage, but neither weighting nor duplication creates independent new evidence about an unseen positive subtype.</Prose>

    <H2>{headings[4]}</H2>
    <Prose><strong>Random oversampling</strong> samples existing minority rows with replacement. <strong>Random undersampling</strong> retains a chosen subset of majority rows. The first changes multiplicities; the second can discard valuable coverage. Neither requires generating a physically new input. A target ratio is tunable and need not be 1:1.</Prose>
    <Prose><strong>SMOTE</strong>, Synthetic Minority Over-sampling Technique, adds interpolated feature vectors. For a minority anchor <Math>{'x_i'}</Math>, find <Math>{'k'}</Math> other minority neighbors under a declared distance, choose one <Math>{'x_j'}</Math>, and generate</Prose>
    <MathBlock>{'\\begin{gathered}x_{\\mathrm{new}}=x_i+u(x_j-x_i),\\\\[4pt]u\\sim\\operatorname{Uniform}[0,1].\\end{gathered}'}</MathBlock>
    <Prose>Use <strong>one scalar <Math>{'u'}</Math> for the entire vector</strong> in this line-segment construction. Independent fractions per coordinate generally generate a different shape. Label the synthetic vector as the targeted minority class, then train the chosen classifier on the augmented training data. Ordinary SMOTE&rsquo;s neighbor search does not consult a fitted classifier or ask whether majority examples occupy the segment. See the <a href="https://arxiv.org/pdf/1106.1813">original paper, section 4</a> and the <a href="https://imbalanced-learn.org/stable/over_sampling.html#mathematical-formulation">current sample-generation definition</a>.</Prose>
    <Prose>For <Math>{'x_i=(0,0)'}</Math>, <Math>{'x_j=(2,0)'}</Math>, <Math>{'u=.5'}</Math>, the generated point is ({num(smote.generated.x)}, {num(smote.generated.y)}). If an observed majority point already sits at that coordinate &mdash; and in the investigation below it does &mdash; the interpolation produces a conflicting label at exactly that location. The two positive endpoints do not prove that the segment is positive. This is why synthesis can help one geometry and hurt another; many generated rows are not many independent confirmations.</Prose>
    <SmoteGeometryLab />
    <Prose>Distance is part of the model. One large-scale input can dominate Euclidean neighbors; fit a suitable scaler inside the training boundary. Interpolating standardized coordinates and then applying the inverse affine scaler produces a segment in source coordinates, but scaling can change <strong>which neighbor is chosen</strong>. More neighbors may cross separate minority clusters; fewer can make synthesis narrow or repetitive. With <Math>{'m'}</Math> minority rows, <Math>{'k'}</Math> must be at most <Math>{'m-1'}</Math> when self-neighbors are excluded. A fold with only one minority observation cannot support ordinary distinct-neighbor SMOTE.</Prose>
    <Prose>Categories, one-hot constraints and biological validity require care. A halfway category code is not a new valid category. SMOTENC uses different categorical handling for mixed data, and SMOTEN targets all-categorical data; neither guarantees that every combined feature pattern is physically realizable. Domain-valid augmentations can be preferable when there is a meaningful mechanism for generating input variations.</Prose>

    <H3>A complete-label row must remain a complete-label row</H3>
    <Prose>In a multi-label task, suppose observed feature 0 has labels (A=1, B=0), while feature 2 has (A=1, B=1). SMOTE for A might place a synthetic feature at 1 and assign A=1. What is its B label? The interpolation has supplied no answer. Setting B=0, copying one endpoint, taking the union, or using a missing-label policy are different assumptions. Concatenating independently synthesized per-label arrays and pretending their rows still identify the same entities is invalid.</Prose>
    <Prose>Randomly duplicating an observed <strong>whole row with its entire label vector</strong> preserves that observation&rsquo;s label alignment, though it changes the frequency of every co-occurring label. A justified multilabel augmentation needs an explicit joint label/missing-label rule and appropriate validation. The earlier <a href="/learn/path/full-curriculum/multi-label-multi-output-learning?module=classical-ml">Multi-Label &amp; Multi-Output Learning</a> supplies the task semantics; this lesson supplies the resampling boundary.</Prose>

    <H3>Complete teaching functions</H3>
    <Prose>Save the following as <Code>{imbalanceExamples.models.file}</Code>. It requires NumPy and SciPy. The optimizer minimizes the exact normalized objective from section 4; <Code>np.logaddexp</Code> avoids unstable direct exponentials, and <Code>expit</Code> supplies the sigmoid. The interpolation routine uses a small, explicit distance matrix so its neighbor identities can be checked. It is appropriate for this lesson&rsquo;s small minority set, not a claim of a scalable million-row implementation.</Prose>
    <Program example={imbalanceExamples.models}>
      <Prose>The printed midpoint is the declared construction from above. The three seeded samples are generated by running this program, not copied from another RNG implementation: each one lies on a segment between two of the three supplied minority points, and all three sit inside the triangle those points span. Inputs to <Code>fit_logistic</Code> in the complete study below are finite, aligned binary data with positive weights and both classes present. The short teaching function relies on that setup; a reusable public library should enforce its complete input contract.</Prose>
    </Program>

    <H2>{headings[5]}</H2>
    <Prose>A protected assessment should contain appropriate <strong>observed</strong> cases from the population and unit you want to evaluate. Do not balance it just to make metrics look pleasant. A case-control evaluation sample can still be useful with an explicit design and valid reweighting, but its raw precision is not automatically deployment precision.</Prose>
    <Prose>For each training/validation split, fit learned preprocessing on the training portion, transform it, generate or select training rows, and fit the classifier. Transform validation inputs using the fitted preprocessing and predict those original rows. No synthetic validation rows, validation-neighbor search or validation-driven cleaning enters that fit.</Prose>
    <PipelineFigure />
    <Prose>Using an <Code>imblearn.pipeline.Pipeline</Code> is a convenient way to implement training-only resampling in cross-validation. A correct manual fold loop can also do it. Standard sklearn pipelines require compatible transform interfaces and do not magically make a <Code>fit_resample</Code> object work. A pipeline cannot repair duplicated entities split across partitions or a target that was unavailable at prediction time. The <a href="https://imbalanced-learn.org/stable/common_pitfalls.html">resampling pitfalls example</a> illustrates both information leakage and the changed evaluation population caused by resampling before a split.</Prose>
    <Prose>Thresholds, resampling ratios, <Math>{'k'}</Math>, class weights, model settings and preprocessing are all choices if selected using scores. Keep them within development. A threshold can be tuned on a separate set or cross-validated out-of-fold predictions. Do not tune it on a set later described as an untouched test. With few positives, one moved case can make a large difference; stratification helps preserve class counts but does not produce independent positive evidence or replace a needed group/time split.</Prose>

    <H2>{headings[6]}</H2>
    <H3>Define the source and the learning task</H3>
    <Prose>The <a href={provenance.doi}>UCI Yeast collection</a> contains {provenance.sourceRows.toLocaleString('en-US')} rows describing protein localization, with eight numeric descriptors and a sequence identifier. We define the positive target as <strong>{provenance.positiveLabel}</strong>, membrane protein with an uncleaved signal, versus the other recorded locations. The source has {provenance.positives} {provenance.positiveLabel} rows. These are historical engineered sequence descriptors, not modern raw-sequence embeddings or a wet-laboratory trial. This page serves its own unchanged copy, <a href={provenance.file} download>{provenance.file.split('/').pop()}</a>, {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a>, beside its <a href={provenance.attribution}>attribution</a>.</Prose>
    <Prose>The actual source contains {provenance.duplicateIdentifiers} repeated sequence IDs. Each repeated ID has exactly identical descriptors and label. We preserve the whole offline source file, verify this identity, and retain the first occurrence of each ID for the derived analysis. That leaves {provenance.studyRows.toLocaleString('en-US')} distinct protein IDs and {provenance.positives} positives. This documented duplicate rule prevents the same observed protein from crossing a split; it does not discard errors after seeing predictions. Protein-family similarity may still create dependence, and the source supplies no family grouping for assessing a new family or species.</Prose>
    <Prose>We predeclare six score inputs: {provenance.names.map(name => <Code key={name}>{name}</Code>).reduce((all, item, index) => (index === 0 ? [item] : [...all, ', ', item]), [])}. Respectively, they concern signal-sequence recognition by two methods, membrane-spanning-region prediction, mitochondrial versus nonmitochondrial amino-acid content, vacuolar/extracellular content, and nuclear-localization signals. We leave out binary HDEL indicator <Code>erl</Code> and targeting-signal field <Code>pox</Code> to keep this <strong>continuous-score interpolation experiment</strong> explicit. Their omission is not a claim of predictive uselessness. Interpolating these scores creates a feature-space training example; it does not manufacture a biologically valid protein sequence. The source does not supply physical measurement units for these descriptor scores.</Prose>
    <Prose>The deterministic split retains source-row IDs:</Prose>
    <LessonTable caption="Four disjoint roles covering every retained protein" headers={['Role', 'Distinct protein records', `${provenance.positiveLabel} positives`, 'What it may influence']} rows={Object.entries(roles).map(([name, entry]) => [
      name.charAt(0).toUpperCase() + name.slice(1), String(entry.records), String(entry.positives), entry.influences,
    ])} />
    <Prose>The outer development split uses seed {study.splitSeeds.development}, fitting split {study.splitSeeds.fitting} and tuning/inspection split {study.splitSeeds.roles}, with stratification at each step. We fit five declared logistic procedures: original data, balanced class weights, random oversampling, random undersampling and ordinary SMOTE. The six inputs, <Math>{'\\lambda=.01'}</Math>, unpenalized intercept, optimizer and preprocessing fit remain fixed. The final comparison contains five fits, with no search for a favorable model family. The scaler fits the original {roles.fitting.records} fitting records for all methods; resampling changes the classifier&rsquo;s training input afterward.</Prose>
    <Prose>There are {study.fittingNegatives} negative and {study.fittingPositives} positive fitting records. Balancing by random duplication or SMOTE adds {study.addedMinorityRows} positives, yielding {study.balancedRows.toLocaleString('en-US')} rows. Random undersampling keeps {study.fittingPositives} negatives and all {study.fittingPositives} positives, yielding {study.undersampledRows} rows. Balanced weights leave {roles.fitting.records} stored rows and assign positive weight 600/42 = {num(balanced.weights[1])} and negative weight 600/1158 = {num(balanced.weights[0])}.</Prose>
    <Prose>For a concrete decision comparison, use <strong>hypothetical</strong> FP cost {study.costFalsePositive} and FN cost {study.costFalseNegative}, with zero correct-decision cost. For each fitted model, evaluate all distinct tuning-score thresholds plus a no-alert policy, choose minimum tuning cost, and break ties toward the higher threshold. This selected score threshold is not claimed to be the posterior formula 1/13: the scores are imperfect and some were fitted to different objectives.</Prose>

    <H3>Run the complete study</H3>
    <Prose>Save as <Code>{imbalanceExamples.study.file}</Code> beside <Code>{imbalanceExamples.models.file}</Code> and the provided <Code>yeast.data</Code>. Setup: Python, NumPy, SciPy and scikit-learn. The author calculation ran with Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and sklearn 1.9.1. Exact split, convergence, donor choices, predictions and thresholds are retained in the author record.</Prose>
    <Program example={imbalanceExamples.study}>
      <Prose>The reserved IDs are returned to make ownership explicit but never passed to prediction. Each procedure keeps its original fitted coefficients while its threshold is tuned; we do not refit on tuning data afterward and silently assume the score scale stays identical. A later production refit needs a compatible complete threshold/calibration protocol. Every number this program prints is recomputed independently from the saved scores by this lesson&rsquo;s verifiers, and the table below is read from those same saved scores.</Prose>
    </Program>

    <H3>Read the outcomes that actually occurred</H3>
    <Prose>The following numbers are from the retained author calculation, recomputed here from the saved per-record scores:</Prose>
    <LessonTable caption="Five procedures, three different winners" headers={['Procedure', 'Selected score threshold', 'Inspection TP / FP / FN / TN', 'Inspection cost FP+12 FN', 'AP', 'Top 10 positives']} rows={outcomes.map(row => [
      row.method.label,
      num(Number(row.method.chosenThreshold.toFixed(6))),
      `${row.tuned.tp} / ${row.tuned.fp} / ${row.tuned.fn} / ${row.tuned.tn}`,
      String(costOfRow(row)),
      num(Number(row.method.averagePrecision.toFixed(6))),
      String(row.queue.positives),
    ])} />
    <Prose>The always-negative baseline has {roles.inspection.records - roles.inspection.positives}/{roles.inspection.records} = {num(100 * study.baselineAccuracy)}% accuracy, recall 0 and cost {study.baselineCost}. Its precision is undefined. At threshold .5, the original fitted model makes {originalRow.atHalf.fp} false alarm and detects {originalRow.atHalf.tp === 0 ? 'none' : originalRow.atHalf.tp}, for cost {costOf(originalRow.atHalf, study.costFalsePositive, study.costFalseNegative)}; the tuned threshold improves its realized cost to {costOfRow(originalRow)}. The SMOTE procedure&rsquo;s tuned result has cost {costOfRow(smoteRow)} rather than its default-threshold cost {costOf(smoteRow.atHalf, study.costFalsePositive, study.costFalseNegative)}. Threshold selection and resampling have played different roles.</Prose>
    <OutcomesFigure />
    <Prose>{bestCost.method.label} has the lowest realized tuned cost among these five; {bestAp.method.label.toLowerCase()} has the highest AP; {topTenWinners.map(row => row.method.label.toLowerCase()).join(', ')} find {bestTopTen} positives in their top ten, while the other two find {smallest(outcomes.map(row => row.queue.positives))}. Those are different questions with different winners. Neither the AP ranking nor the cost ranking certifies a universally best method. With only {roles.inspection.positives} inspection positives, a single additional detection changes recall by 1/{roles.inspection.positives} &asymp; {num(1 / roles.inspection.positives)}. The method/settings were not changed after seeing this table, and the {roles.reserve.records} reserved proteins remain unscored.</Prose>
    <Prose>The original model&rsquo;s inspection Brier score, mean(<Math>{'q-y'}</Math>)&sup2;, is {num(Number(originalRow.method.brierScore.toFixed(6)))}; balanced weighting gives {num(Number(weightedRow.method.brierScore.toFixed(6)))} and SMOTE {num(Number(smoteRow.method.brierScore.toFixed(6)))}. The smaller original value does not prove perfect calibration: Brier also reflects resolution and the event prevalence. It does show why a larger rare-class score or better chosen action rule is not automatically a better original-population probability estimate. The later <a href="/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml">Calibration &amp; Conformal Prediction</a> develops probability assessment and calibration in full.</Prose>
    <TuningQueueLab />
    <Checkpoint prompt={`SMOTE reaches the lowest realised cost, ${costOfRow(bestCost)}, while random oversampling reaches the highest average precision, ${num(Number(bestAp.method.averagePrecision.toFixed(6)))}. A colleague proposes reporting average precision as the headline result, because it is the measure on which their favoured method wins. What has changed about the comparison if you agree?`}>
      <Prose>The objective was declared before the fits: minimum realized cost at {study.costFalsePositive} per false alarm and {study.costFalseNegative} per missed positive. Choosing a different objective <em>after</em> reading the outcome table is a new selection step performed with knowledge of the results, so the inspection partition is no longer an untouched assessment of that newly chosen rule. The honest report names the declared objective, gives all three columns, and says that the three questions disagree. If average precision is genuinely the right objective for the task, that has to be settled before the assessment, and a fresh protocol is needed before presenting the newly selected procedure as independently assessed. The {roles.reserve.records} reserved proteins exist precisely so that such a protocol remains possible; they stay unscored here.</Prose>
    </Checkpoint>
    <Prose>For an API-oriented implementation, a training-only sampler can be composed as below. Save as <Code>{imbalanceExamples.pipeline.file}</Code> beside the prior files. This additionally requires <Code>imbalanced-learn</Code>. The current documentation was inspected during writing, and <strong>the content phase deliberately did not execute this optional package program</strong>; this implementation installed imbalanced-learn 0.14.2 and ran it, so the output below is real. It is a separate threefold development-CV illustration with sklearn&rsquo;s stated <Code>C=1</Code> convention, not a reproduction of the custom <Math>{'\\lambda'}</Math>-normalized five-model table.</Prose>
    <Program example={imbalanceExamples.pipeline}>
      <Prose>Every fold&rsquo;s scaler and sampler fit anew inside that fold. A shared random seed does not make its synthesized rows identical to our from-scratch routine: random draw order, neighbor ties and algorithm conventions can differ. That is why none of these three fold scores coincides with any inspection AP in the table above, and none should: the protocol, the penalty convention and the evaluated records are all different. Validate the learning procedure and synthetic geometry rather than expecting matching stdout.</Prose>
    </Program>

    <H2>{headings[7]}</H2>
    <Prose>The core workflow is already usable: define the decision, protect assessment data, compare a baseline with justified alternatives, and inspect counts at the intended operating point. The branches below explain what changes when ordinary interpolation or a single class weight is insufficient. They are further study, not prerequisites for the first-pass exercises.</Prose>

    <H3>Where should synthetic points be placed?</H3>
    <Prose>Ordinary SMOTE chooses neighbors within the minority class. It does not inspect majority labels when drawing a point on the chosen segment. This is why moving the majority point in the SMOTE investigation changes the apparent conflict while leaving the synthetic coordinates unchanged. The algorithm&rsquo;s construction and our judgment of that construction are different operations.</Prose>
    <Prose><strong>Borderline-SMOTE</strong> first identifies minority observations in mixed-class neighborhoods and directs synthesis toward that boundary region. It can concentrate effort where a classifier is uncertain, but a mixed neighborhood can also contain mislabeled observations or genuine class overlap. A boundary is not automatically missing positive coverage.</Prose>
    <Prose><strong>ADASYN</strong> assigns more synthetic examples near minority observations whose neighborhoods contain a larger proportion of other-class observations. In a simplified binary description, let <Math>{'r_i'}</Math> be that fraction for minority observation <Math>{'i'}</Math>. Normalize these fractions and allocate an intended synthetic budget <Math>{'G'}</Math> roughly as <Math>{'g_i=G r_i/\\sum_j r_j'}</Math>, then use minority neighbors for interpolation. Integer allocation and implementation rules mean the final count need not hit exact parity. If every <Math>{'r_i=0'}</Math>, the normalization is undefined; this is a case to handle explicitly, not evidence that an algorithm has learned how to create useful examples there. Giving difficult regions more points can amplify label noise as well as useful boundary information.</Prose>
    <Prose><strong>KMeans-SMOTE</strong> uses a clustering step to restrict and allocate synthesis across suitable clusters. The cluster count and geometry become additional assumptions. <strong>SVM-SMOTE</strong> uses a fitted support-vector boundary to guide candidates. Its behavior depends on that boundary model. These methods replace one geometric assumption with a richer one; they do not eliminate the need to check whether the resulting features and labels represent possible observations. Compare a variant only when its mechanism addresses a visible failure of the simpler procedure.</Prose>

    <H3>When removing data is useful &mdash; and what is actually removed</H3>
    <Prose>Random undersampling reduces the number of majority observations used in a fit. In our study it retained only {study.fittingPositives} of {study.fittingNegatives} fitting negatives. That speeds some fits and changes the empirical class contribution, but may discard a rare <em>negative</em> subtype that is crucial for avoiding false alarms.</Prose>
    <Prose>Distance-based undersampling makes that choice depend on geometry. The NearMiss variants are distinct:</Prose>
    <LessonTable caption="Three different retention rules, not three phrasings of one" headers={['Rule', 'Majority observations preferentially retained']} rows={[
      ['NearMiss-1', 'Smallest average distance to a specified number of nearest minority observations'],
      ['NearMiss-2', 'Smallest average distance to a specified number of farthest minority observations'],
      ['NearMiss-3', 'First collect a specified number of nearest majority candidates around each minority observation; from these, favor the largest average distances to a specified number of nearest minority observations'],
    ]} />
    <Prose>These are rules for retaining observations, not interchangeable descriptions of &ldquo;remove points near the boundary.&rdquo; Scaling and outliers affect the choices. An implementation&rsquo;s neighbor counts and sampling strategy are part of the algorithm specification.</Prose>
    <Prose>Cleaning methods answer another question: which local configurations should be deleted? A <strong>Tomek link</strong> is an opposite-class pair whose members are each other&rsquo;s nearest neighbor, subject to a tie convention. A common policy removes the majority member; an explicit all-class policy may remove both. <strong>Edited nearest neighbors</strong> deletes selected observations whose labels disagree with a specified neighbor vote. The vote may require unanimity or a majority, and the classes eligible for removal must be stated. Disagreement is observable; an incorrect label is not established by disagreement alone.</Prose>
    <Prose>SMOTE followed by Tomek or neighbor cleaning can first add coverage and then remove selected overlap. Cleaning changes the class counts again, so the result need not remain balanced. Inspect the retained/deleted identities and the decision cost. A tidier scatterplot is not a validation criterion.</Prose>

    <H3>Ensembles can distribute the discarded information</H3>
    <Prose>The <a href="/learn/path/full-curriculum/decision-trees-random-forests?module=classical-ml">decision-trees and random-forests lesson</a> showed how multiple fitted models can reduce dependence on one sample. In a balanced ensemble, each learner can receive a different resampled training subset. Across learners, more majority observations may participate than in a single undersampled fit. This can preserve useful variety while limiting the imbalance seen by each learner.</Prose>
    <Prose>Balanced bagging and balanced random forests apply such sampling around individual learners or trees. EasyEnsemble combines ensembles fitted on different undersampled subsets; RUSBoost combines random undersampling with boosting. Their voting/averaging rules, sampling replacement, per-learner counts and loss remain substantive choices. Different learners seeing a point does not create a new independent protein, and the entire ensemble must stay within each training fold. Use the earlier ensemble mechanisms to reason about these procedures rather than memorizing a league table of sampler names.</Prose>

    <H3>Focal loss changes which examples dominate an update</H3>
    <Prose>A large training set can contain many correctly classified, easy negatives. Even small individual losses may add up. <strong>Focal loss</strong> was introduced for dense object detection, where a detector evaluates many potential object locations and most are background. That setting gives a concrete reason to reduce the influence of already-easy cases.</Prose>
    <Prose>For a binary example, let <Math>{'p_t=q'}</Math> when its true label is 1 and <Math>{'p_t=1-q'}</Math> when its true label is 0. With a class-specific positive factor <Math>{'\\alpha_t'}</Math>, focal loss is</Prose>
    <MathBlock>{'\\begin{gathered}L_{\\mathrm{focal}}=-\\alpha_t(1-p_t)^\\gamma\\log p_t,\\\\[4pt]\\gamma\\ge0.\\end{gathered}'}</MathBlock>
    <Prose>At <Math>{'\\gamma=0'}</Math>, this is weighted cross-entropy. For an easy example with <Math>{'p_t=.9'}</Math>, <Math>{'\\gamma=2'}</Math> multiplies its cross-entropy loss by {num(focal.rows[0].modulator)}; for a difficult example with <Math>{'p_t=.2'}</Math>, the multiplier is {num(focal.rows[1].modulator)}. With <Math>{'\\alpha_t=1'}</Math>, ten thousand easy examples contribute about {num(Number(focal.rows[0].crossEntropyTotal.toFixed(2)))} total cross-entropy versus {num(Number(focal.rows[1].crossEntropyTotal.toFixed(2)))} from ten difficult examples. Under focal loss those totals become about {num(Number(focal.rows[0].focalTotal.toFixed(2)))} and {num(Number(focal.rows[1].focalTotal.toFixed(2)))}. The balancing comes from the current prediction difficulty, not just the class label. These totals are a constructed loss calculation, not a training benchmark.</Prose>
    <LossMassFigure />
    <Prose>Differentiating the focal objective also differentiates the factor <Math>{'(1-p_t)^\\gamma'}</Math>. For a positive example,</Prose>
    <MathBlock>{'\\begin{gathered}\\frac{dL}{dp_t}=\\alpha_t\\gamma(1-p_t)^{\\gamma-1}\\log p_t\\\\[4pt]-\\alpha_t\\frac{(1-p_t)^\\gamma}{p_t}.\\end{gathered}'}</MathBlock>
    <Prose>The derivative with respect to a model&rsquo;s logit additionally multiplies by <Math>{'p_t(1-p_t)'}</Math>. A claim that the training gradient is simply cross-entropy&rsquo;s gradient times {num(focal.rows[0].modulator)} would miss the first term. Difficult examples can include annotation errors; emphasizing them is not always beneficial. Focal loss is a changed objective and does not generally retain ordinary log loss&rsquo;s original-posterior optimum. The later <a href="/learn/path/full-curriculum/loss-functions-ce-mse-focal-contrastive-triplet?module=deep-learning-fundamentals">loss-functions lesson</a> develops network losses, gradients and training comparisons. It is not needed to use the cost-sensitive logistic workflow above.</Prose>

    <H3>A changed prior has an exact correction under a specific assumption</H3>
    <Prose>Suppose sampling changes only the positive-class proportion from a deployment value <Math>{'\\pi'}</Math> to a training value <Math>{'\\rho'}</Math>, while preserving both class-conditional feature distributions. Let <Math>{'q(x)'}</Math> be the true posterior in that sampled population. Bayes&rsquo; rule gives</Prose>
    <MathBlock>{'\\begin{gathered}\\frac{p(x)}{1-p(x)}\\\\[6pt]=\\frac{q(x)}{1-q(x)}\\cdot\\frac{\\pi/(1-\\pi)}{\\rho/(1-\\rho)}.\\end{gathered}'}</MathBlock>
    <Prose>To see why, write each posterior odds as the same likelihood ratio <Math>{'f(x\\mid Y=1)/f(x\\mid Y=0)'}</Math> times its population&rsquo;s prior odds, then divide. This is a prior-shift calculation, not a universal repair of a model score. It requires nondegenerate priors and the unchanged-conditional-distribution assumption; the displayed finite-odds calculation also assumes scores strictly between 0 and 1, with endpoints handled by limits where meaningful.</Prose>
    <Prose>If a balanced sample has <Math>{'\\rho=.5'}</Math>, deployment has <Math>{'\\pi=.01'}</Math>, and its posterior is <Math>{'q=.8'}</Math>, the sampled odds are {num(prior.sampledOdds)}. Multiply by 1/99 to obtain deployment odds 4/99, hence <Math>{'p=4/103'}</Math> &asymp; {num(prior.posterior)}. A sample-posterior value of .8 can correspond to less than 4% deployment probability. That is a change of population, not a contradiction.</Prose>
    <Prose>Random case-control sampling can plausibly preserve class-conditionals when sampling is independent of features within each class. Ordinary SMOTE alters the minority feature distribution by interpolation. Its change is therefore not, in general, just a prior change. A finite regularized classifier can also be misspecified. Use representative held-out probability assessment and an appropriate calibration protocol when decisions require probabilities; do not apply an odds correction and declare the problem solved. Calibration and threshold selection have different roles even when both use development data.</Prose>
    <Prose>This distinction is useful beyond fraud or diagnosis. In a materials screening experiment, scientists may deliberately measure many more promising candidates than their natural prevalence would provide. A score describing that enriched sample is not automatically the probability that a randomly chosen candidate will succeed. In industrial fault monitoring, the deployment class-conditionals themselves may change with a new sensor or operating regime, so a prior-only correction may be insufficient.</Prose>

    <H3>More rows do not create more independent evidence</H3>
    <Prose>{roles.inspection.positives} inspection positives leave very limited information about sensitivity. Stratification can place positives in each fold, but the same positive observed across repeated splits is still one underlying observation. Repeated folds measure a procedure&rsquo;s sensitivity to particular resplits; their standard deviation is not automatically a confidence interval for future recall. Use uncertainty methods suited to the sampling design, keeping repeated subjects or related units together. Exact duplicate removal in this study addresses one concrete leakage route; it does not reveal unknown protein families or guarantee transfer to a new organism.</Prose>
    <Prose>A multiclass problem introduces multiple class-specific errors and possibly a full action-cost matrix. A macro-average weights classes equally; a frequency-weighted average answers a different question. Resampling every class to the largest count is a candidate training design, not a universal target. In a multi-label problem the full label vector and missing-label status remain attached to each observation, as the conflicting-label example in section 5 demonstrated. If rare positives are unlabelled rather than confirmed negative, the task also requires a label-observation model; class balancing cannot turn unknown truth into negative truth.</Prose>

    <H3>Budget the actual operations</H3>
    <Prose>For <Math>{'m'}</Math> minority observations in <Math>{'d'}</Math> dimensions, our explicit all-pairs neighbor calculation needs <Math>{'O(m^2d)'}</Math> arithmetic and <Math>{'O(m^2)'}</Math> stored distances. Sorting every distance row costs <Math>{'O(m^2\\log m)'}</Math>; more specialized selection/search structures can change that work, with effectiveness depending on dimension and geometry. Producing <Math>{'G'}</Math> interpolated vectors then costs <Math>{'O(Gd)'}</Math>, plus storage for the generated data. The neighbor search is over the minority points here, not automatically every point in the dataset.</Prose>
    <Prose>If the majority count is <Math>{'M\\ge m'}</Math>, balancing by adding minority rows changes total rows from <Math>{'M+m'}</Math> to <Math>{'2M'}</Math>. The expansion factor is <Math>{'2M/(M+m)'}</Math>, below 2. A 99-to-1 dataset becomes {expansion.oversampledRows} rows from {expansion.originalRows}, a factor of {num(expansion.factor)}, not a hundred times larger. Balancing by undersampling leaves <Math>{'2m'}</Math> rows. Class weights avoid materializing duplicate feature rows, although they can change conditioning, convergence and the training path. There is no zero-overhead or equal-runtime guarantee.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>Try each question before opening its hint or solution. Questions 1&ndash;7 use the first-pass route. Questions 8&ndash;10 transfer the deeper ideas. A calculator or a short script is welcome; the target is a defensible explanation, not mental arithmetic speed.</Prose>

    <Practice title="1. What can 99% accuracy hide?"
      question="There are 1,200 observations, 12 positive. Construct two confusion matrices with 99% accuracy: one with recall 0 and another with recall 1. Explain how the same accuracy can describe both."
      hint="Both matrices need 1,188 correct predictions and 12 errors. Assign those errors to different cells.">
      <Prose>Always predicting negative gives TP 0, FP 0, FN 12, TN 1,188. Detecting all positives while making 12 false alarms gives TP 12, FP 12, FN 0, TN 1,176. Both have 1,188/1,200 accuracy. Their recalls are 0 and 1. Accuracy reports a particular equal-error-cost aggregate; the aggregate alone does not identify which errors occurred.</Prose>
    </Practice>

    <Practice title="2. Ties are whole score groups"
      question={<>Four records have scores <Code>[.95, .8, .8, .4]</Code> and labels <Code>[0, 1, 0, 1]</Code>. With selection rule score&ge;threshold, compute precision and recall at threshold .8. If the review budget permits exactly two records, why is &ldquo;take everything at least .8&rdquo; not the same policy?</>}
      hint="The threshold includes both records tied at .8. For an exact budget, state how the tie is broken before inspecting its labels.">
      <Prose>At .8, TP 1, FP 2, FN 1, TN 0, so precision 1/3 and recall 1/2. The threshold selects three records. Taking exactly two must select the .95 record and one of the tied records. A fixed record-ID ordering or a predeclared random tie rule can do that; choosing the positive because its held-out truth is known leaks the outcome into the policy. The resulting top-two precision can differ depending on the legitimate tie rule. The &ldquo;Practice 2&rdquo; setup in the score-queue investigation above loads exactly these four records.</Prose>
    </Practice>

    <Practice title="3. Choose by expected cost"
      question={<>A false alarm costs 2 units and a missed positive costs 7. Correct decisions cost 0. A well-specified posterior gives <Math>{'p=.2'}</Math>. Which action has lower expected cost? Derive the cutoff rather than applying .5 automatically.</>}
      hint={<>Compare <Math>{'2(1-p)'}</Math> with <Math>{'7p'}</Math>.</>}>
      <Prose>Selecting costs {num(practiceRisk.selectRisk)} in expectation and skipping costs {num(practiceRisk.skipRisk)}, so skip. Selection is better when <Math>{'2(1-p)<7p'}</Math>, or <Math>{'p>2/9'}</Math> &asymp; {num(practiceRisk.cutoff)}. At exactly 2/9 the actions tie under this cost model. The answer depends on a probability for the relevant population and on the stated costs; rarity alone did not decide it.</Prose>
    </Practice>

    <Practice title="4. What does the weighted score mean?"
      question="At a feature value, the population positive probability is .2. Train unrestricted weighted binary log loss with positive weight 4 and negative weight 1. Find the population-optimal score, then invert it back to the original probability. What happens to the optimum if both weights are multiplied by 3?"
      hint="Insert the values into the optimum derived in section 4. The ratio of weights controls that optimum.">
      <Prose>The optimum is <Math>{'q=(4\\times.2)/(4\\times.2+1\\times.8)='}</Math>{num(practiceOptimum.optimum)}. Inverting gives <Math>{'p=w_-q/[w_+(1-q)+w_-q]='}</Math>{num(inverseWeightedOptimum(practiceOptimum.optimum, 4, 1))}. A weighted score .5 is not a claim that the original event probability is .5. Multiplying both weights by 3 leaves this population optimum unchanged; it scales the unnormalized expected loss. Our normalized finite-sample objective also cancels a common weight factor, while an objective normalized differently can change its effective regularization.</Prose>
    </Practice>

    <Practice title="5. Interpolate a vector, then question its label"
      question={<>A minority anchor is <Math>{'(1,2)'}</Math>, its chosen minority neighbor is <Math>{'(5,4)'}</Math>, and <Math>{'u=.25'}</Math>. Calculate the SMOTE point. If the training fold contains only three minority observations, can the usual &ldquo;five other minority neighbors&rdquo; setting be used? Finally, explain why a majority point near the generated location matters even though it did not enter the interpolation formula.</>}
      hint="Use the same scalar fraction for both coordinates. Count neighbors after excluding the anchor itself.">
      <Prose>The point is <Math>{'(1,2)+.25(4,2)=(2,2.5)'}</Math>. There are at most two other minority observations, so five-neighbor SMOTE is undefined in that fold; choose a justified smaller setting or a different procedure before evaluation. A nearby majority observation suggests overlap or an implausible minority-label assumption. Ordinary SMOTE does not resolve that conflict merely by creating the point. The &ldquo;Practice 5&rdquo; setup in the SMOTE investigation loads this geometry; the investigation refuses <Math>{'k=5'}</Math> with three minority points rather than silently reducing it.</Prose>
    </Practice>

    <Practice title="6. Preserve the whole observation"
      question={<>Two training observations have feature values 0 and 4 and known label vectors <Code>(A=1, B=0, C=1)</Code> and <Code>(A=1, B=1, C=0)</Code>. You interpolate feature 2 while balancing label A. A colleague proposes assigning the midpoint label <Code>(1, 1, 1)</Code> because all three labels occur among the endpoints. Is that label established by the input? Give one coherent alternative that does not invent a label.</>}
      hint="The feature interpolation establishes no rule for how B and C behave between endpoints.">
      <Prose>No. The endpoints do not establish B or C at feature 2, and the proposed label combination was observed at neither endpoint. A domain-supported label-generation model could justify a new observation, but it must be stated and checked. Alternatively, oversample an entire existing row with its unchanged complete label vector, or use a fitting weight with an explicitly defined multi-label loss. Independent per-label synthetic matrices cannot be silently joined into one aligned dataset.</Prose>
    </Practice>

    <Practice title="7. Choose the question before the winner"
      question="Using the observed Yeast table, identify the method with lowest declared inspection cost, the one with highest AP, and all methods with the most positives in their top ten. Would replacing the declared goal with AP after seeing these results preserve an untouched comparison? What does one additional detected positive change in recall?"
      hint={`Read three different columns. The inspection partition contains ${roles.inspection.positives} positives.`}>
      <Prose>{bestCost.method.label} has cost {costOfRow(bestCost)}; {bestAp.method.label.toLowerCase()} has AP {num(Number(bestAp.method.averagePrecision.toFixed(6)))}; {topTenWinners.map(row => row.method.label.toLowerCase()).join(', ')} each find {bestTopTen} positives in their top ten. Changing the goal after inspecting outcomes is a new exploratory decision, not the original locked assessment. A new evaluation protocol is needed before presenting a newly selected procedure as independently assessed. One additional detection changes recall by 1/{roles.inspection.positives} &asymp; {num(1 / roles.inspection.positives)}. None of the three result columns removes that small-positive-count uncertainty.</Prose>
    </Practice>

    <Practice title="8. Correct the sampling odds — deeper"
      question="Assume unchanged class-conditional feature distributions. A sampled population has positive prevalence .2, deployment prevalence is .02, and the sampled-population posterior is .5. Compute the deployment posterior. Explain why the same calculation is not automatically justified for SMOTE scores."
      hint="The sampled posterior odds are 1. Multiply by deployment prior odds divided by sampling prior odds.">
      <Prose>The odds multiplier is <Math>{'(.02/.98)/(.2/.8)=4/49'}</Math> &asymp; {num(practicePrior.multiplier)}. Thus deployment odds are 4/49 and probability is <Math>{'4/53'}</Math> &asymp; {num(practicePrior.posterior)}. SMOTE alters the distribution of minority features, and a fitted score may not equal the sampled population&rsquo;s true posterior. Those issues violate steps used in the derivation; knowing the two class proportions is insufficient.</Prose>
    </Practice>

    <Practice title="9. Count storage and information separately — deeper"
      question="A fitting set has 960 majority and 40 minority observations. How many rows result from random oversampling to parity? From random undersampling to parity? Does the first procedure provide 960 independent minority examples?"
      hint="Keep the original count of unique minority observations separate from the materialized row count.">
      <Prose>Oversampling gives {practiceExpansion.oversampledRows.toLocaleString('en-US')} rows, a {num(practiceExpansion.factor)}-fold expansion from {practiceExpansion.originalRows.toLocaleString('en-US')}. Undersampling gives {practiceExpansion.undersampledRows} rows. The duplicated minority rows still come from {practiceExpansion.independentMinorityObservations} underlying observations, so treating them as 960 independent events for an uncertainty calculation would exaggerate the evidence. The larger training matrix may change an optimization procedure while leaving the number of independently observed proteins unchanged.</Prose>
    </Practice>

    <Practice title="10. Design a review queue under changing conditions — deeper"
      question="A factory can investigate ten sensor alerts per shift. One false alarm takes the same review time as one real fault. The classifier was trained on deliberately fault-enriched data, and next month a new sensor model will be installed. Describe a defensible development/evaluation plan. Explain where class weights, prior correction and a top-ten policy each fit, and name information that the prompt does not supply."
      hint="Separate fitting, probability interpretation, resource allocation and transfer to the new sensor. A prior adjustment assumes more than knowing the fault percentage."
      revealLabel="Assessment and example conclusion">
      <Prose>Keep related machine histories and time boundaries intact; fit preprocessing and any sampler within the training role. Compare a baseline with justified weights or resampling using representative development data. Select a review policy there, with a label-independent tie rule and an explicit response when fewer than ten alerts have worthwhile expected value. A top-ten score policy enforces capacity; probability/cost thresholds answer an additional question about whether reviewing a candidate is worthwhile. Prior correction could apply if enrichment preserved class-conditionals and the fitted probabilities represent that enriched population. A new sensor can change feature distributions within each class, so collect or otherwise justify representative new-sensor assessment rather than assuming that correction suffices. Reserve a final future/group-separated assessment for the selected full procedure. The prompt leaves fault costs, enrichment mechanism, annotation completeness, sensor compatibility and population prevalence unspecified; identify these as design inputs, not arbitrary defaults. Weighted fitting alone does not supply them.</Prose>
    </Practice>

    <H2>{headings[9]}</H2>
    <Prose>You are ready to continue when you can construct a confusion table, explain why an accurate model can miss every rare event, derive an action cutoff from stated costs, distinguish a weighted score from an original-population probability, generate and question a SMOTE point, and place sampling and threshold choice on the correct side of the assessment boundary. Questions 1&ndash;7 check those skills. The advanced branches let you reason about richer samplers, changed priors and resource constraints when a task requires them.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Read a confusion table and say which quantity is undefined and why', 'Section 1, figure 1, practice 1'],
      ['Move a gate through a ranking and explain a falling precision', 'Section 2, the score-queue investigation, practice 2'],
      ['Compute average precision as grouped recall increments, ties included', 'Section 2, the score-queue investigation'],
      ['Say what changes and what does not when prevalence moves', 'Section 2, figure 2'],
      ['Derive an action cutoff from two stated costs, and know when it is undefined', 'Section 3, the cost investigation, practice 3'],
      ['Follow a weighted residual into two gradient sums', 'Section 4, figure 3'],
      ['Turn a weighted score back into the probability it came from', 'Section 4, the weighted-score investigation, practice 4'],
      ['Generate a SMOTE point and say what its rule never looked at', 'Section 5, the SMOTE investigation, practice 5'],
      ['Refuse an unsupported synthetic label on a multi-label row', 'Section 5, practice 6'],
      ['Put the sampler on the fitting branch and keep assessment observed', 'Section 6, figure 4'],
      ['Read a real comparison in which three questions have three winners', 'Section 7, figure 5, practice 7'],
      ['Tune a threshold on tuning records without touching an inspection result', 'Section 7, the tuning-queue investigation'],
      ['Correct a posterior for a changed prior, and say when you may not', 'Section 8, practice 8'],
      ['Separate materialized rows from independent evidence', 'Section 8, practice 9'],
    ]} />
    <Prose>The next module topic is <a href="/learn/path/full-curriculum/automl-neural-architecture-search-nas?module=classical-ml">AutoML &amp; Neural Architecture Search</a>. We will let a procedure search over modeling choices. That only helps if the search is asked the right question: the objective, fold ownership, data geometry and decision cost you specified here must travel with it. Automating a leaky or irrelevant comparison makes it easier to repeat the same mistake at scale.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html">imbalanced-learn &mdash; comparison of oversampling methods</a>. Actual input clouds, generated samples and fitted decision boundaries for duplication, SMOTE, ADASYN and variants, plus mixed and all-categorical examples. Use the pictures to compare geometric assumptions; they are examples, not a universal performance ranking.</li>
      <li><a href="https://arxiv.org/pdf/1106.1813">Chawla et al. &mdash; SMOTE: Synthetic Minority Over-sampling Technique</a>, especially &sect;4&rsquo;s construction and &sect;6&rsquo;s categorical discussion. The paper motivates interpolated minority features and also reports limits, including an unfavorable Adult-dataset case. Its pseudocode&rsquo;s placement of the random gap can be read coordinate-wise; this lesson explicitly uses one scalar fraction for a whole vector, and checks that in its implementation.</li>
      <li><a href="https://scikit-learn.org/stable/modules/classification_threshold.html">scikit-learn &mdash; tuning the decision threshold</a>. Separates score fitting from threshold tuning and illustrates CV-based threshold selection and fixed-threshold use. Its APIs are an alternative to our explicit threshold sweep; choose the score/cost objective appropriate to the task.</li>
    </ul></>}>
      <li><a href="https://imbalanced-learn.org/stable/over_sampling.html">imbalanced-learn &mdash; oversampling</a>, <a href="https://imbalanced-learn.org/stable/under_sampling.html">undersampling</a>, <a href="https://imbalanced-learn.org/stable/combine.html">combined samplers</a> and <a href="https://imbalanced-learn.org/stable/ensemble.html">balanced ensembles</a> &mdash; which observations are generated, retained or removed. Consult the exact sampler and version before treating two API names as equivalent.</li>
      <li><a href="https://imbalanced-learn.org/stable/common_pitfalls.html">imbalanced-learn &mdash; common pitfalls</a> &mdash; why preprocessing and sampling belong within each training split, and how resampling before a split changes the evaluated population.</li>
      <li><a href="https://cseweb.ucsd.edu/~elkan/rescale.pdf">Elkan &mdash; The Foundations of Cost-Sensitive Learning</a>, &sect;&sect;1&ndash;3 &mdash; cost-based decisions and the assumptions behind class-prior rescaling. Read after the two-action derivation here; finite, constrained learners need not behave like unrestricted population-optimal rules.</li>
      <li><a href="https://mark.goadrich.com/articles/davisgoadrichpr.pdf">Davis and Goadrich &mdash; The Relationship Between Precision-Recall and ROC Curves</a> &mdash; their fixed-population relationship and why PR interpolation needs care. For the exact noninterpolated score used in our table, consult <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">average_precision_score</a>.</li>
      <li><a href="https://arxiv.org/pdf/1708.02002">Lin et al. &mdash; Focal Loss for Dense Object Detection</a>, &sect;3 and its loss plots &mdash; the many-background-locations motivation and the modulating factor. The inspected mechanism supports our loss calculation; its detection benchmarks were not reproduced here.</li>
      <li><a href={provenance.page}>UCI &mdash; Yeast</a>, {provenance.creator}, <a href={provenance.doi}>{provenance.doi}</a>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a> &mdash; the actual observations. This page serves <a href={provenance.file} download>its own unchanged copy</a>, {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, beside its <a href={provenance.attribution}>attribution</a>, which records the class definition, the exact-duplicate repair and the fitting/assessment ownership.</li>
    </Sources>
    <Prose>The 1,000-case confusion table, the three-record score queue, the two population flows, the two action costs, the single weighted gradient step, the weighted-loss optimum, the small SMOTE geometry, the focal loss-mass totals and every practice matrix are explicitly <strong>constructed calculations</strong>, not measurements. The decision costs of {study.costFalsePositive} and {study.costFalseNegative} units are a declared teaching assumption, not a laboratory price. The Yeast results are calculations on the identified real dataset under one declared row-level protocol, with the {roles.reserve.records} reserved proteins never predicted or scored. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>,
};

export default imbalancedLearningContent;
