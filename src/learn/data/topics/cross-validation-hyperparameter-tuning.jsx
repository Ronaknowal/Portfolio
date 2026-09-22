import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { FoldBuilderLab, SelectionLab, NestedLab, HalvingLab } from '../../components/lesson-labs/ValidationLabs.jsx';
import {
  LoopFigure, SplitQuestionFigure, NestedRoomsFigure, RealExperimentExplorer, CoverageFigure, RiskFigure, ImprovementFigure,
} from '../../components/lesson-labs/ValidationFigures.jsx';
import { validationExamples } from '../validation-examples.js';
import { NESTED_EXPERIMENT, PENGUIN_SOURCE, RUNTIME_VERSIONS } from '../validation-data.js';
import { bootstrapDistinctShare, drawsForHitProbability, fitBudget, hitProbability, meanPredictorRisk } from '../validation-models.js';

const experiment = NESTED_EXPERIMENT;
const decimals = (value, digits) => value.toFixed(digits);
const budget = fitBudget({ candidates: 6, outerFolds: 3, innerFolds: 3 });
const bootstrap = bootstrapDistinctShare(1000000);

const headings = [
  '1. What exactly are we trying to estimate?',
  '2. Build cross-validation one held-out prediction at a time',
  '3. Choose splits that match the future use',
  '4. Selection can learn the validation answers',
  '5. Search the settings you actually mean to compare',
  '6. A complete nested experiment with real observations',
  '7. Deeper branch: what cross-validation uncertainty does and does not mean',
  '8. Deeper branch: adaptive search and spending resources',
  '9. Run a useful comparison within an honest budget',
  '10. Practice: identify the decision before computing the score',
  '11. Readiness and what follows',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Practice({ title, question, hint, children }) {
  return <section className="cv-practice"><H3>{title}</H3><Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>Show the explained solution</summary>{children}</details>
  </section>;
}

const crossValidationContent = {
  title: 'Cross-Validation & Hyperparameter Tuning',
  readTime: '~55 min first pass · ~100 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot cv-lesson">
    <LessonIntro
      prerequisites={<>The fit/transform boundary from <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">Feature Scaling, Encoding &amp; Imputation</a>, Python indexing, and the idea of a nearest-neighbour prediction. Scores and split roles are defined locally here, so <a href="/learn/path/full-curriculum/ml-problem-formulation-baselines-data-leakage?module=classical-ml">ML Problem Formulation, Baselines &amp; Data Leakage</a> is useful background but not a gate.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A model can improve its score by learning the subject, or by becoming unusually well adapted to the examples we use to judge it. Those are
      different kinds of progress. Cross-validation gives each observation a turn as held-out evidence; hyperparameter tuning uses such evidence
      to choose a learning procedure. This lesson combines them by deciding, every time, <strong>what is being chosen, what is being assessed,
      and which information each decision is allowed to use</strong>. Four investigations show how each valid change reaches the fitted model, score and information flow immediately.
    </LessonIntro>
    <Prose className="cv-route">
      <strong>First pass.</strong> Read sections 1–6 and try core practice questions 1–6, doing the fold-building, candidate-selection and
      nested-fold investigations where they appear, and running the two Python programs in sections 2 and 6. Sections 7–9 deepen uncertainty,
      adaptive search and computational budgeting; they do not become hidden prerequisites for the core readiness check, and practices 7–10
      belong with them. Allow roughly 55 minutes for the core route and another 60–90 for the code and practice.
    </Prose>

    <Callout title="Selection evidence and assessment evidence, once for the whole lesson">
      A score that was used to <strong>choose</strong> a setting is not an estimate of that setting&rsquo;s performance. Throughout this lesson
      the number that picked something is called a <strong>selection score</strong> and the number produced by rows that were kept out of that
      choice is called an <strong>assessment</strong>. They are never averaged together, never plotted on the same axis, and never quoted
      interchangeably &mdash; including inside the investigations, where the two always occupy separate rows. Everything else here follows from
      keeping that boundary visible.
    </Callout>

    {/* ============================== 1 ============================== */}
    <H2>{headings[0]}</H2>
    <Prose>
      Imagine building the penguin species classifier from <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">Feature Scaling, Encoding &amp; Imputation</a>.
      The program learns medians and scales from training rows, then predicts a species from nearby examples. Three questions can all sound like
      &ldquo;how good is the model?&rdquo;:
    </Prose>
    <ol>
      <li>How well does this particular fitted model predict new observations from the intended population?</li>
      <li>How well does a fixed learning recipe usually work when fitted to a new training sample of a specified size?</li>
      <li>How well does the whole process of trying settings, selecting one, and refitting usually work?</li>
    </ol>
    <Prose>
      The first question concerns a fitted object. The second concerns a learning algorithm. The third includes <strong>selection</strong> as
      part of that algorithm. A score should name which question it answers.
    </Prose>
    <Prose>
      A <strong>parameter</strong> is learned during ordinary fitting, such as a regression coefficient. A <strong>hyperparameter</strong>{' '}
      specifies how that fitting or prediction works: a neighbour count, a penalty strength, a maximum tree depth, a preprocessing choice. The
      distinction depends on the procedure. A neighbour count supplied by the programmer becomes a choice learned from data the moment we
      compare counts using validation scores.
    </Prose>
    <Prose>
      For classification here, <strong>accuracy</strong> is the number of correct predictions divided by the number assessed. For regression,{' '}
      <strong>mean squared error</strong> averages <Math>{'(y-\\hat y)^2'}</Math>; it has squared target units and lower is better. A scoring
      rule should reflect the practical question: if rare failures matter much more than common successes, accuracy alone can be unsuitable, and
      the later <a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">evaluation-metrics</a>{' '}
      and <a href="/learn/path/full-curriculum/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml">imbalanced-learning</a>{' '}
      lessons develop alternatives.
    </Prose>
    <LessonTable caption="Names for data roles. The names refer to use, not to permanent properties of rows." headers={['Role', 'What it can influence']} rows={[
      ['Training', 'Fit preprocessing and model parameters'],
      ['Validation', 'Compare candidate settings or stop training'],
      ['Test / outer assessment', 'Assess a procedure whose choices were made elsewhere'],
    ]} />
    <Prose>
      A row can be a training example in one fold and a validation example in another. Within a particular assessment, it must not influence the
      fitted or selected procedure that is being judged on it.
    </Prose>
    <LoopFigure />

    {/* ============================== 2 ============================== */}
    <H2>{headings[1]}</H2>
    <H3>A fold is a role assignment</H3>
    <Prose>
      In K-fold cross-validation, partition the available development rows into K nonoverlapping <strong>folds</strong>, as equal in size as
      possible. For each fold, fit a fresh copy of the entire procedure using the other folds, predict the held-out rows, and save their results.
      Every row is assessed once in that K-fold run.
    </Prose>
    <Prose>
      &ldquo;Fresh copy&rdquo; includes any learned imputer, scaler, feature selector and model. The previous fold&rsquo;s trained object does
      not continue learning into the next fold. The folds also need not be exactly equal: seven observations can form folds of sizes 3, 2 and 2
      without dropping the remainder.
    </Prose>
    <Prose>Consider this constructed one-dimensional dataset:</Prose>
    <LessonTable caption="Seven constructed rows" headers={['Row ID', 'Feature x', 'Label y']} rows={[
      ['0', '0', '0'], ['1', '1', '0'], ['2', '2', '0'], ['3', '3', '1'], ['4', '4', '1'], ['5', '5', '1'], ['6', '6', '1'],
    ]} />
    <Prose>
      Use three consecutive folds and a one-nearest-neighbour classifier. A new x receives the label of the closest training x; equal distances
      use the smaller source row ID in this example.
    </Prose>
    <LessonTable caption="Three fits, and what each one was allowed to use" headers={['Held-out rows', 'Available training rows', 'Held-out predictions', 'Correct / assessed']} rows={[
      ['0, 1, 2', '3, 4, 5, 6', '1, 1, 1', '0 / 3'],
      ['3, 4', '0, 1, 2, 5, 6', '0, 1', '1 / 2'],
      ['5, 6', '0, 1, 2, 3, 4', '1, 1', '2 / 2'],
    ]} />
    <Prose>
      For the first assessment, every training label is 1, so all three predictions are 1. In the second, x = 3 is closest to training x = 2 and
      receives the wrong label 0; x = 4 is closest to x = 5 and receives 1. The third assessment correctly labels both remaining points.
    </Prose>
    <Prose>
      This ordered split is useful for understanding the mechanics. It also reveals a design problem: ordering by class creates very different
      training problems. For approximately independent classification examples we will often distribute classes across folds. For a
      future-in-time question, however, shuffling away the order could destroy the evaluation we actually need.{' '}
      <strong>Choose the split from the prediction question, not from whichever arrangement gives the largest number.</strong>
    </Prose>
    <FoldBuilderLab />

    <H3>An average over folds is not always an average over people or rows</H3>
    <Prose>
      The fold accuracies above are 0, 0.5 and 1. Their unweighted mean is 0.5. But only three of the seven row predictions were correct, giving
      pooled accuracy <Math>{'3/7\\approx0.4286'}</Math>.
    </Prose>
    <Prose>
      Neither arithmetic operation is mysterious. They assign different weights. The unweighted fold mean gives each fold one third of the
      weight; pooled accuracy gives each row one seventh. Write <Math>{'\\ell_i'}</Math> for the loss of the single held-out prediction made for
      row <Math>{'i'}</Math>, and <Math>{'\\widehat R_k'}</Math> for fold k&rsquo;s mean loss over its held-out set <Math>{'V_k'}</Math>. For a
      loss that can be added per row,
    </Prose>
    <MathBlock>{'\\begin{gathered}\\widehat R_{\\text{rows}}=\\frac1n\\sum_{i=1}^{n}\\ell_i\\\\[4pt]=\\sum_{k=1}^{K}\\frac{|V_k|}{n}\\,\\widehat R_k.\\end{gathered}'}</MathBlock>
    <Prose>
      Here <Math>{'\\ell_i=L(y_i,\\hat f_{-k}(x_i))'}</Math>, where <Math>{'\\hat f_{-k}'}</Math> is the procedure fitted without fold k. Equal
      fold sizes make this equal to the unweighted fold mean. Unequal group sizes may motivate another question: should each patient count
      equally, or should every visit count equally? State the intended unit and weighting.
    </Prose>
    <Prose>
      For a nonlinear summary such as F1 or AUC, pooling predictions can change the quantity even with equal fold sizes. AUC, for example,
      compares positive&ndash;negative score pairs; pooling may compare scores emitted by different fitted models. Do not assume every metric can
      be averaged or pooled interchangeably.
    </Prose>

    <H3>A complete splitter that keeps every row</H3>
    <Prose>
      This NumPy-only program recreates the small table. Install NumPy in your Python environment if needed
      (<Code>python -m pip install numpy</Code>). Save and run it as <Code>cv_tiny.py</Code>.
    </Prose>
    <Program example={validationExamples.tinySplitter}>
      <Prose>
        The executed arithmetic gives <Code>[0,1,2] → [1,1,1], 0/3</Code>; <Code>[3,4] → [0,1], 1/2</Code>; <Code>[5,6] → [1,1], 2/2</Code>, then
        fold mean 0.5 and pooled 0.42857142857142855. Shuffling is optional because a splitter cannot know whether rows may be exchanged without
        changing the problem.
      </Prose>
    </Program>

    {/* ============================== 3 ============================== */}
    <H2>{headings[2]}</H2>
    <H3>Independent examples and rare classes</H3>
    <Prose>
      If examples can reasonably be treated as independent draws from a common population, shuffled K-fold is a useful baseline.{' '}
      <strong>Stratified K-fold</strong> approximately preserves class proportions in each fold. This helps avoid folds that omit a rare class
      and prevents certain model and metric failures.
    </Prose>
    <Prose>
      It cannot create missing examples: with two positive observations and five validation folds, at least three folds contain no positive
      observation. Nor does stratification make uncertainty disappear. Making folds more homogeneous can hide some variability caused by rare
      classes. It is a practical allocation choice, not a universal statistical correction. The{' '}
      <a href="https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-with-stratification-based-on-class-labels">cross-validation guide</a>{' '}
      describes these engineering and interpretation limits.
    </Prose>
    <Prose>
      Five or ten folds are common practical starting points, not universal optima. More folds use a larger training fraction in each fit and
      require more fits. Leave-one-out uses n folds, each with one assessed row and n&nbsp;&minus;&nbsp;1 training rows. Its error variance
      depends on the learning procedure and the data; &ldquo;the training sets overlap, therefore variance must be highest&rdquo; is not a valid
      general proof. Section 7 develops this distinction.
    </Prose>
    <Prose>
      Repeated K-fold reruns different partitions of the <strong>same dataset</strong>. It can expose split sensitivity, but it does not collect
      more independent people. Repeated random holdouts let the training fraction vary independently of the number of repetitions; some rows may
      be assessed several times and others not at all. Leave-P-out enumerates all choices of P held-out rows, with{' '}
      <Math>{'\\binom nP'}</Math> fits, which quickly becomes expensive. These are different resampling plans, not stronger and stronger
      guarantees.
    </Prose>

    <H3>New groups versus later records from known groups</H3>
    <Prose>
      Suppose each person contributes several sensor windows. If the intended use is on people absent from training, all windows from a held-out
      person must stay out of that fold&rsquo;s training set. <Code>GroupKFold</Code> and <Code>LeaveOneGroupOut</Code> express this requirement.{' '}
      <Code>StratifiedGroupKFold</Code> also tries to balance classes, but exact balance may be impossible when groups are indivisible.
    </Prose>
    <Prose>
      If the intended use is to predict later records from already known people, a forward-time evaluation may legitimately include their
      earlier records. Calling that automatically wrong would change the question to unseen-person generalization. The problem is using
      information that would not be available at the actual prediction point, or reporting one deployment setting as evidence for another.
    </Prose>
    <Prose>
      For a system serving new people in future calendar periods, both group and time constraints can matter. A standard named splitter may not
      express both; construct and inspect explicit index pairs. <strong>A splitter name is not a substitute for writing the required
      boundary.</strong>
    </Prose>
    <SplitQuestionFigure />

    <H3>Future prediction and label availability</H3>
    <Prose>
      For forecasting, train on information available before the prediction time and assess a later horizon. An expanding window keeps
      accumulating history; a sliding window limits how much old history remains. A deliberate gap may be necessary because labels mature late,
      features use overlapping windows, or the deployment pipeline has a delay.
    </Prose>
    <Prose>
      For example, if a row formed at day t predicts an outcome measured through day t + 7, a training row dated yesterday may not have a known
      target today. Removing rows solely by their feature timestamp is insufficient. Also ensure any rolling features use only permitted past
      observations.
    </Prose>
    <Prose>
      <Code>TimeSeriesSplit</Code> is a useful index-based building block with options such as <Code>gap</Code>, <Code>test_size</Code> and{' '}
      <Code>max_train_size</Code>; a gap counts rows, not elapsed days. Irregular observations and prediction horizons require explicit date
      logic. A timestamp column alone does not force every retrospective task to use forward validation. The evaluation must match the claimed
      use. The later <a href="/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml">Time-Series Validation &amp; Forecasting Baselines</a>{' '}
      develops complete forecasting protocols.
    </Prose>

    <H3>What an out-of-fold prediction table requires</H3>
    <Prose>
      <Code>cross_val_score</Code> assesses the provided splits. <Code>cross_val_predict</Code> additionally requires that each supplied row
      appear in a held-out set <strong>exactly once</strong>, so it can return one prediction per row. Ordinary K-fold and GroupKFold can meet
      that contract. A forward-time plan leaves an initial training prefix with no held-out prediction; repeated holdouts can assess rows
      multiple times. Those are valid evaluation plans, but they do not satisfy this API&rsquo;s partition requirement.
    </Prose>
    <Prose>
      For a manual forward out-of-fold table, retain unpredicted entries as missing and use only rows with legitimate predictions in a
      downstream stacking model. Do not fill the prefix with predictions from a model trained on that same prefix. The preceding{' '}
      <a href="/learn/path/full-curriculum/ensemble-methods-stacking?module=classical-ml">ensemble topic</a> owns the full stacking construction;
      here the distinction is between an evaluation plan and a complete once-per-row prediction table. See the{' '}
      <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html">cross_val_predict contract</a>.
    </Prose>
    <Prose>
      Predefined benchmark splits can be represented explicitly too. Respect what the benchmark&rsquo;s held-out partition is intended to
      measure rather than reshuffling it merely to obtain a convenient score.
    </Prose>

    {/* ============================== 4 ============================== */}
    <H2>{headings[3]}</H2>
    <H3>A tiny example with no uncertain &ldquo;true performance&rdquo;</H3>
    <Prose>
      Imagine four validation cases whose labels are independent fair coin flips. Features contain no information about those labels. Two
      candidate rules predict either <Code>[0,0,0,0]</Code> or <Code>[1,1,1,1]</Code> on those cases. Each rule has expected accuracy 0.5 on
      independent future fair labels.
    </Prose>
    <Prose>
      For validation labels <Code>[1,1,1,0]</Code>, choose the all-one rule: its validation accuracy is 0.75. That choice still has expected
      future accuracy 0.5. Across all sixteen equally likely validation-label patterns, the selected validation accuracy averages 0.6875:
    </Prose>
    <LessonTable caption="Exhaustive enumeration over the sixteen equally likely label patterns" headers={['Number of ones', 'Number of patterns', 'Best correct count']} rows={[
      ['0 or 4', '2', '4'], ['1 or 3', '8', '3'], ['2', '6', '2'],
    ]} />
    <Prose>
      Thus the average selected score is <Math>{'(2\\cdot4+8\\cdot3+6\\cdot2)/(16\\cdot4)=0.6875'}</Math>. Nothing learned a predictive signal.
      We used validation labels to choose the rule that happened to match them.
    </Prose>
    <Prose>
      If sixteen candidate rules cover every possible four-bit prediction pattern, one scores 1 on every validation set. Its expected accuracy on
      independent future labels remains 0.5. Adding a duplicate of an existing rule, however, does not increase the best score.{' '}
      <strong>The number, dependence and flexibility of candidates matter; there is no universal fixed percentage of optimism per trial.</strong>
    </Prose>
    <SelectionLab />
    <Prose>
      Real hyperparameter searches are less artificial, but the same selection mechanism can exploit noise in an estimated score. Its size
      depends on the task. Reporting a selected validation score as though it were untouched assessment evidence is the mistake; an individual
      selected score need not exceed every later test result. Cawley and Talbot&rsquo;s{' '}
      <a href="https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf">primary model-selection study</a> demonstrates how optimizing a
      finite-data criterion can overfit it.
    </Prose>

    <H3>Two valid ways to separate choosing and assessing</H3>
    <Prose>
      <strong>Development plus a separate final assessment:</strong> set aside an appropriate test set, perform all search using the development
      data, refit the selected pipeline on those development data, then assess it on the reserved set. A single split can be useful when it
      provides enough relevant independent assessment units; there is no universal row-count threshold that makes every test reliable.
    </Prose>
    <Prose><strong>Nested cross-validation:</strong> repeat the entire selection process inside each outer training set. For one outer fold:</Prose>
    <ol>
      <li>Protect its outer assessment rows from all fitting and selection.</li>
      <li>Split only the outer training rows into inner training/validation folds.</li>
      <li>Evaluate every candidate pipeline in those inner folds.</li>
      <li>Select a candidate using the declared inner score and tie rule.</li>
      <li>Refit that candidate on all outer training rows.</li>
      <li>Predict the protected outer rows and record the result.</li>
    </ol>
    <Prose>
      Repeat for the other outer folds. The outer score assesses the <strong>selection-and-fitting procedure at the outer training size</strong>,
      under the split&rsquo;s assumptions. It is not a guarantee about an oracle-best setting or an exactly unbiased estimate of one final
      all-data model.
    </Prose>
    <Prose>
      After this assessment, run the declared selection procedure on all available development data and fit the final model. Do not vote among
      outer-fold settings solely because they appeared there most often; each setting was selected using a different training sample. Do not
      select the outer fold with the highest score as the model to deploy.
    </Prose>
    <NestedRoomsFigure />
    <Prose>
      Early stopping is also a selection decision. If an outer assessment set picks the epoch, it is no longer untouched. Inside an inner
      candidate evaluation, choosing an epoch from that same inner validation set makes its score adaptive; a protected outer assessment can
      still assess that complete rule. For a cleaner inner comparison, use an additional stopping subset within inner training and reserve inner
      validation for comparison. What matters is keeping the claimed assessment boundary intact, including preprocessing of any stopping set.
    </Prose>

    {/* ============================== 5 ============================== */}
    <H2>{headings[4]}</H2>
    <H3>Grid search: a finite, inspectable comparison</H3>
    <Prose>
      Suppose the candidates are neighbour counts <Code>[3,5,11]</Code> and scalers <Code>[standard,robust]</Code>. A grid contains all{' '}
      <Math>{'3\\times2=6'}</Math> combinations. With three inner folds, that requires eighteen candidate fits, followed by a refit of the
      selected candidate if requested.
    </Prose>
    <Prose>
      A small grid is useful when the choices themselves are meaningful and affordable. A grid with ten values for each of five independent
      choices has <Math>{'10^5=100{,}000'}</Math> combinations. At five folds and one minute per fit, that is 500,000 fit-minutes before refits
      and overhead &mdash; not the time of one model training. The number of jobs is exact; wall time depends on training sizes, resources and
      parallelism.
    </Prose>
    <Prose>
      Some choices are conditional. An RBF kernel has a bandwidth parameter; a linear kernel does not need it. A list of separate parameter
      dictionaries can express those branches without wasting evaluations on irrelevant combinations. Likewise, do not search impossible layer
      shapes or a neighbour count larger than an inner training set.
    </Prose>

    <H3>Random search: specify a distribution, not just a range</H3>
    <Prose>
      Random search draws candidates from a declared distribution. If a positive parameter spans orders of magnitude, a log-uniform distribution
      gives equal probability to equal multiplicative ranges. On <Code>[0.001,1000]</Code>, each decade has probability one sixth. Uniform
      sampling on the original numeric scale instead assigns almost all probability to large values.
    </Prose>
    <Prose>
      Suppose a satisfactory region has probability mass p under the chosen sampling distribution. For T independent draws, the probability of at
      least one hit is
    </Prose>
    <MathBlock>{'P(\\text{hit})=1-(1-p)^T.'}</MathBlock>
    <Prose>
      If p = 0.05, sixty draws give about {decimals(hitProbability(0.05, 60), 7)}. If p = 0.01, the same sixty give only{' '}
      {decimals(hitProbability(0.01, 60), 7)}. This is a coverage calculation, <strong>not</strong> a theorem that sixty trials reach within 5%
      of the best score. A region containing 5% of the sampling probability is different from a score within 5% of an optimum.
    </Prose>
    <Prose>
      A grid can repeatedly test the same few values along an important dimension while varying unimportant ones. Independent continuous random
      draws explore more distinct values along each coordinate. That is useful when effective importance is concentrated in a few unknown
      dimensions, but it does not make random search universally dominate a well-chosen small grid. The{' '}
      <a href="https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf">Bergstra&ndash;Bengio paper</a> provides the original empirical
      and geometric argument.
    </Prose>
    <CoverageFigure />
    <Prose>
      Adaptive search and early resource allocation are useful extensions in section 8. They change how candidates receive attention, while
      leaving the selection/assessment boundary in place.
    </Prose>

    {/* ============================== 6 ============================== */}
    <H2>{headings[5]}</H2>
    <Prose>
      We reuse the preceding lesson&rsquo;s {PENGUIN_SOURCE.rows}-row Palmer Penguins CSV so the model and features stay familiar. Its CC0 data,
      measurement units and provenance are supplied with this lesson: the served file is{' '}
      <a href={PENGUIN_SOURCE.served}>penguins.csv</a> (SHA-256 <Code>{PENGUIN_SOURCE.sha256.slice(0, 16)}…</Code>, {PENGUIN_SOURCE.bytes} bytes)
      with its <a href={PENGUIN_SOURCE.provenance}>provenance note</a> and the complete{' '}
      <a href="/learn-assets/cross-validation/nested-experiment.json">split record</a> — every fold’s row IDs, all eighteen candidate scores and
      all 344 held-out predictions, so the run below can be checked without this page. This is a documented demonstration of a selection protocol on an already
      familiar dataset, not newly independent validation of a winner from the earlier page. The program protects every outer row from the inner
      decisions used to make its recorded prediction; generalization claims remain confined to the random-row setting represented by that
      experiment.
    </Prose>
    <Prose>
      We compare six candidates: neighbour counts 3, 5 and 11, each with standard or robust scaling. Numeric medians, categorical imputation and
      the one-hot vocabulary are learned inside every candidate training fold &mdash; the source file still carries{' '}
      {PENGUIN_SOURCE.missing.bill_length_mm} missing cells in each numeric measurement and {PENGUIN_SOURCE.missing.sex} missing sex entries, and
      they are imputed only within the relevant training fold. Three stratified outer folds use seed {experiment.outerSeed}; each inner
      three-fold split uses seed {experiment.innerSeed}. Accuracy is the declared selection metric. Exact ties use the first candidate in the
      declared enumeration, making the procedure reproducible rather than silently choosing a favourable tie afterwards.
    </Prose>
    <Prose>
      Save this complete program as <Code>cv_penguins.py</Code> beside <Code>penguins.csv</Code>. The calculation was executed through equivalent
      operations in the supplied author script with Python {RUNTIME_VERSIONS.python}, NumPy {RUNTIME_VERSIONS.numpy}, pandas{' '}
      {RUNTIME_VERSIONS.pandas} and scikit-learn {RUNTIME_VERSIONS.sklearn}, and re-executed verbatim for this page in the same environment. For
      a new environment, install those pinned versions using <Code>python -m pip install</Code>. They are pins for reproducibility rather than
      a recommendation to stay behind: newer releases can move these digits. It runs serially and requires no network data
      request.
    </Prose>
    <CodeBlock language="bash">{'python -m pip install "numpy==2.3.5" "pandas==3.0.1" "scikit-learn==1.9.1"'}</CodeBlock>
    <Program example={validationExamples.nestedPenguins}>
      <Prose>
        The nested parameter name <Code>prepare__numeric__scale</Code> follows the pipeline into its numeric branch and changes that scaler. The
        estimator selection object clones and fits the entire pipeline for each candidate and fold. Its final refit uses the selected candidate
        on all rows supplied to that search, which are the outer training rows inside the loop.
      </Prose>
    </Program>
    <LessonTable caption="Recorded results" headers={['Outer fold', 'Selected neighbour count / scaler', 'Correct / held-out', 'Majority baseline correct']} rows={experiment.folds.map(fold => [
      String(fold.fold + 1),
      `${fold.selected.k} / ${fold.selected.scaler}`,
      `${fold.correct} / ${fold.testRows.length}`,
      `${fold.baselineCorrect} / ${fold.testRows.length}`,
    ])} />
    <Prose>
      The unweighted outer-fold mean is {decimals(experiment.foldMean, 7)}; pooled accuracy is{' '}
      <Math>{`${experiment.pooledCorrect}/${experiment.assessedRows}=${decimals(experiment.pooledAccuracy, 7)}`}</Math>. The difference is small
      because fold sizes differ by only one row, but the weighting distinction is still real. The changing selected settings show that
      near-performing choices can depend on the training sample.
    </Prose>
    <Prose>
      The final search on all {PENGUIN_SOURCE.rows} rows selects {experiment.finalSelected.k} neighbours with standard scaling and has inner
      selection score {decimals(experiment.finalSelected.selectionScore, 7)}. That score is used to choose the final configuration. It does not
      replace the outer assessment or become a new untouched test result.
    </Prose>
    <Prose>
      That selection is a <strong>tie</strong>, and the tie rule is what breaks it: {experiment.finalSelected.tiedWith.join(' and ')} reach the
      same inner mean {decimals(experiment.finalSelected.selectionScore, 10)}, and the declared enumeration order takes the first. In fact all
      six candidates collapse into three tied pairs here, so the scaler in the model you would deploy is settled by the order the candidates
      were written in, not by any evidence. That is not a defect of the procedure — it is what a declared tie rule is for, and it is visible
      only because the rule was declared in advance. The explorer below has a fourth view showing that search’s own candidate table.
    </Prose>
    <Prose>
      One coincidence is worth naming, because this lesson’s whole thesis is that the two must not be confused. Outer folds 1 and 2 each score
      114/115 = {decimals(experiment.folds[0].accuracy, 7)}, and the final selection score prints the same seven digits. They are different
      quantities: the first was produced by rows protected from every choice, the second chose a setting using every row. The equality is
      arithmetic — 114/115 and 228/230 are the same ratio — and not evidence of anything.
    </Prose>
    <RealExperimentExplorer />
    <NestedLab />
    <Prose>
      In the small experiment above, x values are 0 through 15, with labels 0 below 8 and 1 from 8 upward. The first outer fold assesses even row
      IDs and trains on odd IDs; inner validation alternates positions within that training list. Both neighbour counts initially average 0.875
      in the inner comparison, so the declared smaller-count tie rule chooses 1. Change only row 3&rsquo;s label from 0 to 1: inner means become
      0.5 for count 1 and 0.625 for count 3, selecting 3. The change has altered a legitimate selection input. Editing row 2&rsquo;s label
      instead cannot affect this outer fold&rsquo;s selection, because row 2 is protected assessment data.
    </Prose>
    <Prose>
      In ordinary projects, also retain the split indices, preprocessing specification, candidate space, selection metric, tie rule, seed policy
      and failed fits. A score without its selection procedure is difficult to reproduce and easy to misinterpret. <Code>cv_results_</Code>{' '}
      contains candidate scores and timing information; <Code>cross_validate</Code> can return additional metrics, fitted estimators and split
      indices. These are useful records, not evidence that an invalid split became valid.
    </Prose>

    {/* ============================== 7 ============================== */}
    <H2>{headings[6]}</H2>
    <H3>Training size is part of the target quantity</H3>
    <Prose>
      For a fixed recipe A trained on m independent examples, write <Math>{'\\hat f_m=A(D_m)'}</Math> for the model it produces and define its
      expected new-example loss as
    </Prose>
    <MathBlock>{'R(m)=\\mathbb E\\bigl[L(\\hat f_m,Z_{\\text{new}})\\bigr].'}</MathBlock>
    <Prose>
      The expectation averages both the training sample <Math>{'D_m'}</Math> and an independent new observation{' '}
      <Math>{'Z_{\\text{new}}'}</Math>. A balanced K-fold estimate under an independent, identically distributed sampling setup targets
      performance at approximately <Math>{'m=n(K-1)/K'}</Math> training rows. Its folds do not train on all n rows, so it need not target{' '}
      <Math>{'R(n)'}</Math> exactly. Stratified and structured splits introduce additional conditions; do not transfer this simple
      independent-sampling statement to every design without checking them.
    </Prose>
    <Prose>
      A calculation shows why this matters. Suppose observations <Math>{'Y_i'}</Math> have mean μ and variance σ², and the model predicts their
      training mean <Math>{'\\bar Y_m'}</Math> for every new case. Since the new observation is independent of that mean,
    </Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E\\bigl[(Y_{\\text{new}}-\\bar Y_m)^2\\bigr]\\\\[4pt]=\\operatorname{Var}(Y_{\\text{new}})+\\operatorname{Var}(\\bar Y_m)\\\\[4pt]=\\sigma^2+\\frac{\\sigma^2}{m}.\\end{gathered}'}</MathBlock>
    <Prose>
      With n = 12 and σ² = 4, a three-fold fit trains on m = 8 and has expected new loss {decimals(meanPredictorRisk(4, 8).expectedNewLoss, 1)}.
      A full twelve-row fit has expected new loss <Math>{'4+4/12\\approx4.3333'}</Math>. The difference is training size, not evidence that the
      splitter leaked or that an implementation failed.
    </Prose>
    <Prose>
      Now compare the same full-data mean predictor&rsquo;s training loss. The expected average squared residual on its own n training values is{' '}
      <Math>{'\\sigma^2(1-1/n)'}</Math>. Its expected new loss is <Math>{'\\sigma^2(1+1/n)'}</Math>, a gap of <Math>{'2\\sigma^2/n'}</Math>. This
      is a precise simple example of <strong>training optimism</strong>: the model was fitted using the observations being scored. More
      complicated models can have different optimism; the mean-predictor formula is not an all-purpose correction.
    </Prose>
    <RiskFigure />

    <H3>Why overlap is not a variance formula</H3>
    <Prose>For arbitrary fold losses <Math>{'E_1,\\ldots,E_K'}</Math>,</Prose>
    <MathBlock>{'\\begin{gathered}\\operatorname{Var}\\!\\Bigl(\\tfrac1K\\textstyle\\sum_k E_k\\Bigr)\\\\[4pt]=\\frac1{K^2}\\sum_k\\operatorname{Var}(E_k)\\\\[4pt]+\\frac{2}{K^2}\\sum_{j<k}\\operatorname{Cov}(E_j,E_k).\\end{gathered}'}</MathBlock>
    <Prose>
      The covariance terms are about <strong>losses</strong>, not just the fraction of training rows in common. If all variances equal τ² and
      every pairwise correlation equals ρ, this simplifies to <Math>{'\\tau^2[1+(K-1)\\rho]/K'}</Math>. Those assumptions explain the formula;
      they do not tell us that ρ equals the training-set overlap.
    </Prose>
    <Prose>
      A useful counterexample is a learning rule that ignores training and always predicts zero. Its leave-one-out losses on independent
      observations are independent functions of their respective held-out observations. The training sets overlap almost completely, but the
      overlap has no influence on those predictions. Conversely, an unstable fitted model can make cross-validation losses depend strongly on
      shared observations. The learning rule and data both matter.
    </Prose>
    <Prose>
      This is also why the standard deviation across a few fold scores, divided by the square root of K, is not automatically a valid standard
      error for the CV estimate. A neat interval drawn as &ldquo;mean ± 1.96 fold standard error&rdquo; can substantially misstate uncertainty
      when dependence and training variation are ignored. Bengio and Grandvalet&rsquo;s{' '}
      <a href="https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf">primary result</a> rules out a universal unbiased variance
      estimator based on the usual K-fold error measurements across all distributions. It does not say that uncertainty analysis is impossible
      under additional assumptions.
    </Prose>
    <Prose>
      Report the actual folds, sizes and scores, and call their spread a descriptive spread. Use an uncertainty method matched to the unit,
      sampling assumptions and estimand when an interval is required. For a fixed model assessed on genuinely independent future cases,
      uncertainty in its mean loss is a simpler conditional problem than uncertainty in retraining and selecting a new model. Repeating
      partitions of the same data cannot replace collecting new independent units.
    </Prose>
    <Prose>
      The later <a href="/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml">Bias&ndash;Variance Tradeoff &amp; Learning Curves</a>{' '}
      decomposes variation in model predictions over repeated training samples. That is distinct from spread among CV fold scores and from one
      observed train/validation gap.
    </Prose>

    <H3>Bootstrap and analytic criteria answer related questions</H3>
    <Prose>
      The bootstrap draws a new sample of n rows <strong>with replacement</strong> from the observed dataset, then recomputes an estimator. One
      particular row is absent from a bootstrap sample with probability <Math>{'(1-1/n)^n'}</Math>, approaching{' '}
      <Math>{'e^{-1}\\approx0.368'}</Math>. Thus a bootstrap sample contains about {decimals(100 * bootstrap.distinctShare, 1)}% distinct
      original rows on average, despite having n sampled positions. This differs from K-fold&rsquo;s sampling without replacement and its
      exactly-once validation partition.
    </Prose>
    <Prose>
      Bootstrap estimates of an estimator&rsquo;s variability and out-of-bag prediction assessment have different constructions. A question about
      the variability of a complete selected pipeline may require repeating selection, not merely resampling its final predictions. Grouped data
      require resampling meaningful units; time dependence requires another design. A naïve bootstrap of rows is not a universal solution to the
      dependence problem above. The familiar .632 error estimator combines apparent and out-of-bag error with specific weights; it is a
      particular estimator with limitations, not a consequence that every bootstrap score should be multiplied by .632.
    </Prose>
    <Prose>
      Analytic criteria such as AIC, BIC and complexity penalties are other ways to compare models under specified statistical assumptions. They
      are not equivalent to one another or guaranteed substitutes for the evaluation question above. The next{' '}
      <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization lesson</a> compares
      predictive AIC, evidence-oriented BIC and minimum description length with their assumptions; this lesson&rsquo;s core contribution is to
      make data use and selection explicit. The canonical{' '}
      <a href="https://link.springer.com/content/pdf/bfm:978-0-387-84858-7/1">Elements of Statistical Learning front matter</a> lists these
      neighbouring chapter-7 branches and the conditional-versus-expected-error distinction; it is a section map, not a claim that the linked
      front matter contains those derivations.
    </Prose>

    {/* ============================== 8 ============================== */}
    <H2>{headings[7]}</H2>
    <H3>Search can learn where to look next</H3>
    <Prose>
      Bayesian or sequential model-based optimization fits a <strong>surrogate</strong>: a cheaper model of the relation between candidate
      settings and their observed validation outcomes. An <strong>acquisition rule</strong> uses that model to choose the next costly
      evaluation. After the actual evaluation, the history is updated and the surrogate is refitted.
    </Prose>
    <Prose>For minimization, <strong>expected improvement</strong> at candidate λ is</Prose>
    <MathBlock>{'\\begin{gathered}\\operatorname{EI}(\\lambda)\\\\[4pt]=\\mathbb E\\bigl[\\max(\\ell_\\star-Y,0)\\bigr].\\end{gathered}'}</MathBlock>
    <Prose>
      Here <Math>{'\\ell_\\star'}</Math> is the incumbent best loss, and the expectation is over the surrogate&rsquo;s belief about the outcome{' '}
      <Math>{'Y'}</Math> at λ given the history so far. That random variable describes uncertainty in a belief about a candidate outcome; it is
      not the target label. If the current best loss is 0.20, a candidate believed certain to achieve 0.18 has EI 0.02. Another candidate with
      equal believed probabilities of loss 0.05 and 0.45 has mean loss 0.25 but EI <Math>{'0.5(0.20-0.05)=0.075'}</Math>. A larger possible
      improvement can justify an uncertain trial even when its mean prediction is worse.
    </Prose>
    <Prose>
      These are constructed beliefs for understanding the acquisition rule, not fitted results from a real optimizer. The quality of a surrogate
      and its uncertainty matters. Adaptive search can waste effort or overfit a noisy validation criterion; there is no universal trial count
      after which it beats random search.
    </Prose>
    <ImprovementFigure />

    <H3>What TPE models</H3>
    <Prose>
      A Gaussian-process surrogate commonly models loss conditional on settings. The <strong>tree-structured Parzen estimator</strong> instead
      separates observed settings into a better-loss group and the remaining group, fits densities <Math>{'l(\\lambda)'}</Math> and{' '}
      <Math>{'g(\\lambda)'}</Math>, and seeks settings likely under the better group relative to the other. In its original construction,
      expected improvement is proportional to
    </Prose>
    <MathBlock>{'\\left[\\gamma+(1-\\gamma)\\frac{g(\\lambda)}{l(\\lambda)}\\right]^{-1},'}</MathBlock>
    <Prose>
      where γ is the probability mass assigned to the better-loss group. This motivates seeking a high <Math>{'l/g'}</Math> ratio. It is not the
      same as fitting one Gaussian process, and its tree structure can express conditional choices such as parameters for an optional second
      layer. The <a href="https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf">original TPE paper</a>{' '}
      derives the relation and describes the density construction.
    </Prose>
    <Prose>
      For an optional practical extension, save this as <Code>cv_optuna.py</Code> beside the complete <Code>cv_penguins.py</Code> and CSV above.
      It deliberately imports the already defined loader and full preprocessing pipeline. Install Optuna 5.0.0 in addition to that
      program&rsquo;s packages. This supplementary program was checked against the current API documentation but{' '}
      <strong>not executed during the content phase</strong>, and no exact best settings or scores were invented for it there. The output below
      is a later execution in the pinned environment plus Optuna 5.0.0; a different sampler version can move the trial sequence, so treat those
      two lines as one reproducible run rather than a property of TPE.
    </Prose>
    <Program example={validationExamples.optunaStudy}>
      <Prose>
        <Code>distance_power=1</Code> uses absolute-coordinate differences in the Minkowski metric; 2 gives Euclidean distance. Uniform neighbour
        weights count neighbours equally; distance weighting gives nearer ones greater influence according to the estimator&rsquo;s rule. These
        choices change the search space from the earlier six-candidate grid, so comparing the two selected scores is not a controlled claim
        that one search algorithm is superior &mdash; and the printed number is a selection score on a development split, not an assessment of
        anything.
      </Prose>
    </Program>
    <Prose>
      The two spaces are not incomparable, though, and the more useful fact is what actually happened. This study scores candidates on{' '}
      <em>the same three folds</em> the final grid search uses, and its space <em>contains</em> the grid’s selected setting: three neighbours,
      standard scaling, Euclidean distance and uniform weights all lie inside it. On those same folds that point scores{' '}
      {decimals(experiment.finalSelected.selectionScore, 7)}, against this run’s best of 0.9884058. Twenty trials of this sequential sampler did
      not reach a strictly better point that was inside their own space, and none of the twenty evaluated it. That is a fact about one seeded run
      at one budget &mdash; not about the tree-structured estimator, and not a reason to prefer grids. It is the reason a selected score is
      evidence about a <em>search</em>, and never about the space the search was given.
    </Prose>
    <Prose>
      The program performs development search only. It also logs one line per trial to standard error, which the two lines above do not show;
      they are the program’s own standard output. To assess this adaptive recipe, put a new study entirely inside each outer training set, or
      use a separately reserved final assessment set. A fixed sampler seed improves reproducibility of a sequential run; distributed completion
      order and implementation versions can still matter.{' '}
      <a href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html">Optuna&rsquo;s TPE documentation</a>{' '}
      specifies its startup and sampling behaviour.
    </Prose>

    <H3>Successive halving: spend more only after an initial comparison</H3>
    <Prose>
      Sometimes we can assess a candidate cheaply with fewer training rows or fewer optimization steps. Successive halving starts many candidates
      at a small budget, retains a fraction, and gives survivors larger budgets. With nine candidates, budgets 10, 30 and 90, and survival factor
      3, a simple schedule is:
    </Prose>
    <LessonTable caption="One halving schedule for nine candidates" headers={['Stage', 'Candidates assessed', 'Budget per candidate', 'Total nominal resource']} rows={[
      ['1', '9', '10', '90'], ['2', '3', '30', '90'], ['3', '1', '90', '90'],
    ]} />
    <Prose>
      If each stage retrains from scratch, this costs 270 resource units, compared with 810 to give all nine candidates 90 units. If training
      genuinely resumes from saved state, incremental cost can instead be{' '}
      <Math>{'9\\cdot10+3\\cdot20+1\\cdot60=210'}</Math>. Treating every resource unit as equal wall time is an additional approximation.
      Three-fold evaluation would repeat corresponding fits; software overhead and refitting the winner still need a budget.
    </Prose>
    <Prose>
      The key risk is <strong>early ranking</strong>, not a requirement that every validation curve be monotone. A candidate that starts slowly
      may eventually win. In a constructed example, A has losses <Code>[.30,.25,.24]</Code> at budgets <Code>[10,30,90]</Code>, while B has{' '}
      <Code>[.40,.20,.10]</Code>. Eliminating B after budget 10 discards the eventual winner. Both curves improve monotonically; monotonic
      improvement alone did not make the ranking safe.
    </Prose>
    <HalvingLab />
    <Prose>
      <strong>Hyperband</strong> runs multiple halving schedules, called brackets, with different initial candidate counts and starting budgets.
      This explores the tradeoff between examining many candidates briefly and fewer candidates more thoroughly. It is not merely a second name
      for one successive-halving run. The <a href="https://www.jmlr.org/papers/volume18/16-558/16-558.pdf">primary Hyperband paper</a> explains
      the distinction and the assumptions behind its analysis.
    </Prose>
    <Prose>
      Scikit-learn&rsquo;s halving search remains experimental. A resource can be sample count or an eligible estimator parameter; it cannot
      simultaneously be a searched parameter in the same grid. The final stage need not reach the nominal maximum resource, because the schedule
      depends on candidate count, factor and starting budget. Inspect <Code>n_resources_</Code>, <Code>n_candidates_</Code> and{' '}
      <Code>cv_results_</Code> rather than assuming every finalist received all data. The{' '}
      <a href="https://scikit-learn.org/stable/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving">halving guide</a>{' '}
      documents these contracts. Native full-state resumption should not be assumed just because the high-level schedule has increasing budgets.
    </Prose>

    {/* ============================== 9 ============================== */}
    <H2>{headings[8]}</H2>
    <Prose>
      For G candidates and K folds, a grid makes GK candidate fits, plus one final refit if enabled. A nested plan with O outer folds and I inner
      folds makes <Math>{'O(GI+1)'}</Math> fits for assessment, plus <Math>{'GI+1'}</Math> if the same inner configuration is then searched and
      refitted on all development data. This counts fits; inner, outer and full-data fits have different training sizes and costs.
    </Prose>
    <Prose>
      Our six-candidate, three-by-three experiment uses <Math>{`3(6\\cdot3+1)=${budget.assessment}`}</Math> pipeline fits for outer assessment
      and {budget.final} for final selection and refit, totalling {budget.total}. The three majority baselines are separate cheap fits. This is
      small enough to inspect serially. The displayed main script does not need all CPU cores for a useful lesson.
    </Prose>
    <Prose>
      Parallelism reduces wall time only within available compute and memory. Parallelizing both folds and an estimator&rsquo;s own native
      threads can oversubscribe a machine. Large worker pools may duplicate datasets or create too many pending jobs. Begin with a bounded worker
      count, record actual fit times, and decide whether to parallelize trials or model internals. <Code>pre_dispatch</Code> and pipeline caching
      can help in appropriate cases; caching is useful only when repeated transformations genuinely share inputs and parameters. Cluster
      schedulers and database-backed studies extend this idea, but do not remove validation or resource-accounting requirements.
    </Prose>
    <Prose>
      There is no universal threshold such as &ldquo;Bayesian search is better once one fit takes ten seconds.&rdquo; Measure the real cost of
      fitting, scoring, surrogate updates and scheduling for your task. A small grid, a documented random budget, or a representative holdout can
      be better justified than an elaborate search whose results cannot be assessed reliably.
    </Prose>
    <Prose>
      When two candidates are close, inspect paired results on the same splits and the size of the practical difference. Choosing the highest
      average is a selection rule, not automatically a collection of independent hypothesis tests. Overlapping fold intervals do not form a valid
      general significance test, and a multiple-testing correction does not repair dependent or incorrectly constructed evidence. If a formal
      comparison is required, choose an inferential procedure whose assumptions match the paired data and complete search history.
    </Prose>
    <Prose>
      If a fit fails, keep that failure visible. Invalid settings, insufficient minority examples, numerical problems and unavailable features
      call for different fixes. Silently removing failures or replacing them with convenient scores changes the selection procedure. During this
      small lesson <Code>error_score=&quot;raise&quot;</Code> exposes failures immediately; a large search may log failed trials and follow a
      predeclared handling rule.
    </Prose>

    {/* ============================== 10 ============================== */}
    <H2>{headings[9]}</H2>
    <Prose>
      Try the first six using only the core route. The remaining questions extend the deeper branches. Attempt each task before opening its hint
      or solution.
    </Prose>

    <Practice title="1. Unequal folds" question="Three folds assess 4, 3 and 3 rows and get 3, 1 and 2 correct. Find the unweighted mean fold accuracy and the pooled accuracy. Which gives each row equal weight?"
      hint="Average the three fractions for one answer; add correct counts before dividing for the other.">
      <Prose>
        The fold mean is <Math>{'(3/4+1/3+2/3)/3=7/12\\approx.5833'}</Math>. Pooled accuracy is 6/10 = .6 and gives each row equal weight. Equal
        weighting of folds is a different declared summary.
      </Prose>
    </Practice>

    <Practice title="2. A remainder is still a learner's data" question="A splitter uses fold_size = n // k and slices exactly that many rows for each of k validation folds. What happens at n = 11, k = 3? How does the provided splitter repair it?">
      <Prose>
        Only nine rows receive a validation turn; two are omitted. Depending on how training indices are constructed, those omitted rows may be
        permanently in training or dropped altogether. <Code>np.array_split</Code> creates folds 4, 4 and 3 so every row belongs to one
        validation fold and the remaining folds form its training set.
      </Prose>
    </Practice>

    <Practice title="3. New patient or known patient?" question="A wearable model will predict tomorrow's measurements for people who already provided a week of history. A second product must work on entirely new wearers. Describe an assessment boundary for each. What extra question arises if the second product also launches in a future season?">
      <Prose>
        The first task can use each person&rsquo;s available earlier history but must respect prediction time and target availability. The second
        needs held-out people. A future season introduces a time-distribution boundary as well, so a custom group-and-time assessment may be
        needed. A random visit split does not by itself establish either claimed setting.
      </Prose>
    </Practice>

    <Practice title="4. An impossible out-of-fold array" question="A forward plan trains on rows 0–3 and assesses 4–5, then trains 0–5 and assesses 6–7. Why can it be used for fold scoring but not passed directly to cross_val_predict on all eight rows? What should a manual prediction table contain at rows 0–3?">
      <Prose>
        Rows 0&ndash;3 never appear in a held-out set, so the required exactly-once partition is missing. A manual table should retain their
        predictions as absent, not fit on them and label in-sample outputs out-of-fold. Score the valid held-out rows, or train a stacking stage
        only where legitimate predictions exist.
      </Prose>
    </Practice>

    <Practice title="5. A new candidate pattern" question="Validation labels are [0,1,0,1]. Initially the candidates predict all zeros or all ones. Add a candidate predicting [0,1,0,1]. Under the independent fair-label model, what changes in best validation accuracy and expected future accuracy? What if you add only another all-zero candidate?">
      <Prose>
        The best validation score rises from .5 to 1. Every fixed prediction remains independent of future fair labels, so expected future
        accuracy remains .5. A duplicate all-zero candidate changes neither the original best validation score nor future accuracy. Candidate
        diversity, and how selection uses the labels, are what matter. Both cases are saved setups in the selection investigation above.
      </Prose>
    </Practice>

    <Practice title="6. Which rows selected the epoch?" question="An outer-fold model chooses its number of epochs using outer assessment loss, then reports accuracy on that same outer fold. Identify the violated boundary and give two repairs.">
      <Prose>
        The assessed rows selected a training setting. Move stopping into the outer training data, either using a stopping subset inside each
        inner training partition or treating inner-validation-based stopping as part of the complete rule assessed by protected outer rows.
        Another valid design uses development data for all such choices and a genuinely separate final test set. Renaming the used assessment set
        does not restore independence.
      </Prose>
    </Practice>

    <Practice title="7. Probability mass is not score distance — deeper" question="A satisfactory parameter region has probability .02 under your sampler. How many independent draws give at least a 95% chance of hitting it? Would this guarantee a score within 2% of the global optimum?"
      hint={<>Solve <Math>{'(1-.02)^T\\le.05'}</Math> and round upward.</>}>
      <Prose>
        <Math>{'T\\ge\\log(.05)/\\log(.98)\\approx148.28'}</Math>, so {drawsForHitProbability(0.02, 0.95)} draws suffice under the assumed
        independent sampling model. The 2% is sampling mass, not score proximity. The satisfactory region itself must be defined and its assumed
        mass justified for a practical guarantee.
      </Prose>
    </Practice>

    <Practice title="8. An unchanged acquisition value — deeper" question="The incumbent loss is .30. Candidate C has equal predicted probabilities of losses .10 and .50. Find EI. If only the worse outcome changes from .50 to .90, does EI change? Does expected loss change?">
      <Prose>
        EI is <Math>{'.5(.30-.10)=.10'}</Math>. The worse outcome contributes zero improvement in both cases, so EI remains .10. Expected loss
        changes from .30 to .50. This does not prove the surrogate is calibrated; it distinguishes the summaries of its stated distribution.
      </Prose>
    </Practice>

    <Practice title="9. Account for the whole schedule — deeper" question="A halving plan starts 27 candidates with budgets 5, 15, 45 and 135 and retains one third after each stage. Find the nominal from-scratch resource cost, and compare it with giving every candidate 135. If each stage instead resumes genuine saved state, what is the incremental resource cost?">
      <Prose>
        Candidate counts 27, 9, 3 and 1 each consume 135 nominal units per stage, totalling 540, versus 3,645 for all candidates at 135.
        Resuming costs <Math>{'27\\cdot5+9\\cdot10+3\\cdot30+1\\cdot90=405'}</Math>. Multiply appropriate evaluations by fold count and account
        separately for scoring and refits. Real time need not be linear in this resource.
      </Prose>
    </Practice>

    <Practice title="10. A fixed rule tests an overlap story — deeper" question="A classifier always predicts class 0 and ignores training. Labels on n observations are independent fair bits. What is the variance of its leave-one-out accuracy? Why does this contradict equating loss correlation with training overlap?">
      <Prose>
        The correctness indicators are independent Bernoulli(.5), so their average has variance <Math>{'(.5)(.5)/n=1/(4n)'}</Math>. The
        leave-one-out training sets overlap heavily, but those sets do not influence this rule&rsquo;s predictions. Thus overlap alone does not
        determine loss correlation or imply variance stays near 1/4.
      </Prose>
    </Practice>

    {/* ============================== 11 ============================== */}
    <H2>{headings[10]}</H2>
    <Prose>
      You are ready to continue when you can build a complete fold assignment, identify what each score was allowed to influence, place learned
      preprocessing inside that assignment, select a split matching a concrete future use, and explain why a selected inner score differs from
      protected assessment evidence. You do not need to memorize a preferred number of folds or implement a Bayesian optimizer to demonstrate
      those core skills.
    </Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Build a fold assignment in which every row is assessed exactly once', 'Section 2, the fold-building investigation, practice 2'],
      ['Say which rows a score was allowed to see before it was read', 'Section 1, the loop figure, the nested investigation'],
      ['Keep every learned transformation inside the fold that fitted it', 'Sections 2 and 6, the penguin program'],
      ['Choose a split from a concrete future use, including groups and time', 'Section 3, its figure, practice 3'],
      ['Explain why a selected inner score differs from protected evidence', 'Sections 4 and 6, the selection investigation, practice 5'],
      ['Separate sampling mass from score proximity, and hindsight from decision', 'Sections 5 and 8, practices 7 and 9'],
    ]} />
    <Prose>
      The next topic is <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization: L1, L2, Elastic Net &amp; Dropout</a>.
      It asks what a penalty changes about a fitted model. This lesson supplies the procedure for choosing penalty strength without confusing the
      score that selected it with an untouched assessment. Later{' '}
      <a href="/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml">feature selection</a> and{' '}
      <a href="/learn/path/full-curriculum/automl-as-meta-learning?module=classical-ml">AutoML</a> reuse the same boundary around increasingly
      broad choices.
    </Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://scikit-learn.org/stable/modules/cross_validation.html">Scikit-learn cross-validation guide</a>: current splitters, multiple metrics, prediction-table contracts and structured-data choices. Use its diagrams to inspect what a splitter actually assigns, then check that assignment against your task.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html">Visualizing cross-validation behavior</a>: an inspected visual and code alternative showing class, group and training/test stripes together. Compare <Code>KFold</Code>, <Code>GroupKFold</Code> and <Code>TimeSeriesSplit</Code>; explain the different information boundaries before changing a model.</li>
      <li><a href="https://scikit-learn.org/stable/modules/grid_search.html">Scikit-learn parameter-search guide</a>: grids, random distributions, halving schedules, nested parameter names and selection/assessment separation. Its result fields are useful for making a reproducible search record.</li>
    </ul></>}>
      <li><a href="https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf">Cawley and Talbot, <em>On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation</em></a>: primary study of overfitting the criterion used to choose a model. Read its contrast between expected and particular-sample selection curves; do not copy its benchmark magnitudes as universal effects.</li>
      <li><a href="https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf">Bengio and Grandvalet, <em>No Unbiased Estimator of the Variance of K-Fold Cross-Validation</em></a>: deeper primary reading on the distinction between a fitted model&rsquo;s prediction error and error averaged over training samples, and on dependence in CV uncertainty.</li>
      <li><a href="https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf">Bergstra and Bengio, <em>Random Search for Hyper-Parameter Optimization</em></a>: the coordinate-projection argument and original experiments motivating random search as a strong baseline.</li>
      <li><a href="https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf">Bergstra and colleagues, <em>Algorithms for Hyper-Parameter Optimization</em></a>: sequential model-based search, expected improvement and the original TPE derivation. Read sections 2&ndash;4 after the acquisition example rather than treating their notation as a first exposure.</li>
      <li><a href="https://www.jmlr.org/papers/volume18/16-558/16-558.pdf">Li and colleagues, <em>Hyperband</em></a>: sections 3.1&ndash;3.2 show why one halving bracket and Hyperband are different, and why early resource allocation can eliminate a slow starter.</li>
      <li><a href="https://link.springer.com/content/pdf/bfm:978-0-387-84858-7/1">Elements of Statistical Learning front matter</a>: the chapter-7 section list used to place the optimism, bootstrap and conditional-versus-expected-error branches. A section map only; the derivations are not in the front matter.</li>
      <li><a href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html">Optuna TPESampler documentation</a>: startup trials, sampler seeding and current sampling semantics for the optional adaptive program.</li>
      <li><a href="https://allisonhorst.github.io/palmerpenguins/">Palmer Penguins project</a>, released <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0</a>: the real input and measurement context used in the worked program. Credit Allison Horst, Alison Hill and Kristen Gorman, and the original Palmer Station measurements by Gorman and colleagues. The <a href={PENGUIN_SOURCE.provenance}>accompanying provenance note</a> preserves the license, byte hash and exact experimental split; the original measurement context differs from our instructional species-classification task.</li>
    </Sources>
    <Prose>
      The seven-row splitter, the four fair-label cases, the sixteen-row nested fixture, the coverage draws, the mean-predictor curves, the
      acquisition beliefs and the halving trajectories are constructed fixtures with declared inputs. The penguin results are calculations on the
      identified real dataset under one seed pair, one candidate grid and one declared tie rule. None of them is a benchmark, a guarantee about
      an oracle-best setting, or a claim about any future dataset or deployment.
    </Prose>
  </div>
};

export default crossValidationContent;
