import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { CapacityLab, TimelineLab } from '../../components/lesson-labs/FormulationLabs.jsx';
import {
  ContractFlowFigure, FittingBoundaryFigure, LineageFigure, ResultsFigure, ThresholdCostFigure,
  UnitSplitFigure, UpliftFigure,
} from '../../components/lesson-labs/FormulationFigures.jsx';
import { asInput, fixed, round } from '../../components/lesson-labs/FormulationShared.jsx';
import { formulationExamples } from '../formulation-examples.js';
import { formulationData } from '../formulation-data.js';
import {
  costThreshold, criticalQuantile, expectedLosses, latestKnown, majorityBaseline, policyOutcome,
  practiceTimelineFixture, rankByScore, selectionMetrics, supportFixture, withArrival,
} from '../formulation-models.js';

/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.round` here would resolve to that component and yield
   undefined rather than a number. Every quantity on this page therefore comes
   from `formulation-models.js` or from the recorded data, never from arithmetic
   typed into the prose. */

const partition = formulationData.partition;
const provenance = formulationData.provenance;
const procedures = Object.fromEntries(formulationData.procedures.map(entry => [entry.id, entry]));
const validationIds = formulationData.validation.ids;
const validationTargets = formulationData.validation.targets;
const targetById = Object.fromEntries(validationIds.map((id, index) => [id, validationTargets[index]]));

/* Practice answers, derived rather than typed, so a changed model or a changed
   fixture changes the answer instead of contradicting it. */
const candidateRanking = rankByScore({
  ids: validationIds, scores: formulationData.validation.scores.candidate,
});
const practiceAtTwentyFive = selectionMetrics({
  order: candidateRanking,
  capacity: 25,
  targetById,
  totalPositives: partition.validationPositives,
});
const practiceTimeline = latestKnown(practiceTimelineFixture);
const practiceTimelineEarlier = latestKnown(
  withArrival(practiceTimelineFixture, { event: 4, version: 1, available: 4 }));
/* Named once. The prose below computes the oldest admissible event time from
   this, and the `1` used to be written independently in three places -- the
   two fixtures and the sentence describing them -- so a changed fixture would
   have left the prose behind. It is the only place on the page where that was
   true. */
const PRACTICE_TIGHT_AGE = 1;
const practiceTimelineTight = latestKnown({
  ...practiceTimelineFixture, maximumAge: PRACTICE_TIGHT_AGE,
});
const practiceTimelineTightEarlier = latestKnown({
  ...withArrival(practiceTimelineFixture, { event: 4, version: 1, available: 4 }),
  maximumAge: PRACTICE_TIGHT_AGE,
});
const policyA = policyOutcome({ ...supportFixture, policy: supportFixture.policies[0] });
const policyB = policyOutcome({ ...supportFixture, policy: supportFixture.policies[1] });
const practiceThreshold = costThreshold(3, 9);
const practiceLosses = expectedLosses(0.2, 3, 9);
const prevalence = majorityBaseline({ negatives: 90, positives: 10 });

const headings = [
  '1. Begin with a decision, then define the prediction',
  '2. A row is an observation with a history',
  '3. Three clocks determine what was knowable',
  '4. Establish what a simple system already achieves',
  '5. Real records: a higher score can answer the wrong question',
  '6. Locate the shortcut, not just the suspicious score',
  '7. Turn the study into a reviewable next step',
  '8. Deeper: when a good predictor still chooses the wrong action',
  '9. Practice: repair a question before improving a score',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Practice({ title, question, hint, children }) {
  return <section className="formulation-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>Show the explained solution</summary>{children}</details>
  </section>;
}

const formulationContent = {
  title: 'ML Problem Formulation, Baselines & Data Leakage',
  readTime: '~50 min first pass · ~90 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot formulation-lesson">
    <LessonIntro
      prerequisites={<>A table's rows and columns, the idea of fitting a model on examples, and what a
        probability is. The
        preceding <a href="/learn/path/full-curriculum/sampling-measurement-experimental-design?module=math-foundations">Sampling,
        Measurement &amp; Experimental Design</a> lesson supplies the sampling vocabulary, and
        the <a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">Evaluation
        Metrics</a> lesson develops precision, recall, average precision and log loss. Every term used here is
        defined here as well; nothing is assumed from those pages beyond the idea of scoring a prediction
        against a recorded answer.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A team has a model that predicts whether a customer will accept an offer. Its strongest input is how long
      the conversation lasted. The score looks promising. There is one problem: the team wants to choose whom to
      call <strong>before the conversation begins</strong>. You will reconstruct what a system could have known
      at a cutoff and watch the admissible answer change from {asInput(formulationData.timeline.cases[0].selected.value)} to
      {' '}{asInput(formulationData.timeline.cases[1].selected.value)} when an <em>arrival</em> moves and no
      event time does; fit the real thing on {partition.sourceRows.toLocaleString('en-US')} historical call
      records and find the candidate model making <em>fewer</em> correct decisions than a constant while
      concentrating {procedures.candidate.top50Positives} subscriptions into a selected 50
      against {procedures.prior.top50Positives}; and watch a third model beat both on every number using a
      feature that does not exist until after the call. Change the controls to inspect availability,
      chosen actions and their measured consequences together.
    </LessonIntro>

    <div className="formulation-route"><Prose><strong>First pass.</strong> Read sections 1–7 and do practice 1–5.
      That route gets you the target/unit/cutoff contract, the three clocks that decide what was knowable, a
      baseline worth beating, the difference between a classification score and a limited-capacity decision, and
      the five different things people call leakage. Run the two programs on the way and do the small timeline
      by hand before opening the first one. Section 8 is a deeper branch — on selection, intervention effects
      and losses that preserve what the decision needs — and is not required to finish the core.</Prose></div>

    <Prose>The model may have learned a real relationship. It still cannot perform the requested job using that
      input. Defining a useful prediction requires more than choosing an algorithm and measuring its
      accuracy.</Prose>
    <Prose>This lesson connects the methods you have studied to a complete question: <strong>what decision will
      this prediction support, what information could the system actually know, and what comparison would
      establish that the prediction helps?</strong> We will use real historical call records, build a small
      measured comparison, and reconstruct what a system could know at an earlier moment.</Prose>

    <Callout title="Which numbers here are measured, which are constructed, and which are arithmetic">
      Three kinds of quantity appear on this page and are never mixed. <strong>Measurements</strong> are the
      recorded outputs of one fixed experiment on {partition.sourceRows.toLocaleString('en-US')} real bank
      records: every average precision, log loss, correct count and selected-set count in sections 5 and 7, and
      everything investigation 2 reads. <strong>Constructions</strong> are invented teaching fixtures with no
      value taken from those records: the calibration history of section 3, the parcels of section 2, the
      support costs and the two groups of section 8. <strong>Arithmetic</strong> is exact evaluation of a stated
      formula — the decision threshold, the stocking costs, the precision and recall of any selected set. Each
      figure and each investigation states which kind it is, in a line beneath its title. This caution is stated
      once; the rest of the lesson refers back to it rather than repeating it.
    </Callout>

    {/* ============================================================ §1 */}
    <H2 id={headingId(headings[0])}>{headings[0]}</H2>
    <Prose>“Use machine learning on our customer data” leaves nearly everything important unspecified. A more
      useful starting statement is:</Prose>
    <blockquote><Prose>Before the next day's calls are scheduled, rank eligible contact opportunities by the
      chance of a recorded subscription outcome, using information available at scheduling time.</Prose></blockquote>
    <Prose>This is a proposed operational question, not a claim that the supplied historical dataset can fully
      validate it. Writing it reveals what evidence we would need.</Prose>
    <Prose>There are three separate objects. The <strong>outcome</strong> is what we want to improve, such as
      useful contacts completed under a limited calling capacity. The <strong>prediction</strong> is an
      estimate, such as a subscription probability for an eligible opportunity. The <strong>decision
      rule</strong> is how an estimate becomes an action, such as selecting the highest-ranked 50 eligible
      opportunities.</Prose>
    <Prose>The model's output is not automatically the decision, and the measured target is not automatically
      the outcome we ultimately care about. A probability model can support different capacities without
      retraining, provided its meaning and applicable population remain appropriate.</Prose>

    <ContractFlowFigure />

    <Prose>An arrow from the final outcome back into tomorrow's training data is legitimate only after that
      outcome exists. An arrow taking tomorrow's outcome into today's prediction is the shortcut we must
      prevent.</Prose>

    <H3>A compact experiment contract</H3>
    <Prose>Before fitting, fill in these fields in ordinary language.</Prose>
    <LessonTable caption="Nine fields that turn a wish into an experiment"
      headers={['Field', 'Question to answer', 'Example for a proposed pre-call system']}
      rows={[
        ['Population', 'Which cases is this meant for?', 'Eligible contact opportunities under a stated campaign policy'],
        ['Unit', 'What does one prediction represent?', 'One customer opportunity at a specific scheduling time'],
        ['Cutoff', 'When must the prediction be ready?', 'Before the call list is finalized'],
        ['Target', 'What observed answer will train and evaluate it?', 'A precisely defined subscription outcome and observation window'],
        ['Features', 'Which information is available by the cutoff?', 'Versioned customer and history records with documented availability'],
        ['Action', 'What happens to the prediction?', 'Rank eligible opportunities; select at most 50'],
        ['Baseline', 'What would happen without the proposed learner?', 'Existing policy and a simple probability or ranking baseline'],
        ['Evaluation', 'What future situation does the split imitate?', 'Later eligible opportunities, keeping repeated units appropriately separated'],
        ['Success', 'What improvement and constraints matter?', 'Useful outcomes at the fixed capacity, with latency and subgroup checks'],
      ]} />
    <Prose>This contract prevents a project from silently changing its goal when a convenient metric improves.
      It also makes a legitimate change visible: a post-call reporting model is a different task from a pre-call
      scheduling model.</Prose>
    <Prose>Machine learning may be unnecessary. A reliable explicit rule is attractive when the task is
      deterministic, the relevant information is already known, or a simple process change solves the problem.
      For a learned system, we need a learnable relationship, suitable examples, a usable feedback process and
      enough practical benefit to justify its
      cost. <a href="https://developers.google.com/machine-learning/problem-framing">Google's problem-framing
      course</a> offers another guided route through that decision.</Prose>

    {/* ============================================================ §2 */}
    <H2 id={headingId(headings[1])}>{headings[1]}</H2>
    <H3>Unit, target and observation window</H3>
    <Prose>Suppose a delivery service asks, “Will this parcel arrive late?” One row might represent a parcel
      when dispatched, that parcel every hour, or a customer order containing several parcels. Those choices
      create different training examples, dependence and decisions.</Prose>
    <Prose>Define the label relative to a cutoff. If the question is whether delivery will occur more than one
      day after dispatch, a row collected six hours after dispatch may not yet have a known answer. Treating “no
      recorded late delivery yet” as “on time” creates false negatives.</Prose>
    <Prose>A useful label record includes the event or observation window that defines the answer, when the
      answer became ascertainable, and the rule applied to missing or censored outcomes. The
      earlier <a href="/learn/path/full-curriculum/survival-analysis-cox-regression-kaplan-meier-hazard-models?module=classical-ml">survival-analysis</a> lesson
      explains why incomplete follow-up is not automatically a negative event.</Prose>
    <Prose>For repeated hourly rows from one parcel, a random row split can place nearly identical states from
      that parcel on both sides. That might assess interpolation among already represented parcels. It does not
      automatically assess predictions for entirely new parcels. The <strong>evaluation unit</strong> should
      match the generalization question.</Prose>

    <UnitSplitFigure />

    <H3>The name of a column is not its meaning</H3>
    <Prose>An integer called <Code>customer_id</Code> might be a harmless join key, an accidental timestamp, or
      a shortcut to a customer who appears in both training and validation. A column
      called <Code>previous_outcome</Code> might mean the previous campaign's outcome, or it might have been
      overwritten with the current outcome.</Prose>
    <Prose>Write feature definitions with their time and provenance, not only their names:</Prose>
    <blockquote><Prose>Count of successfully completed earlier contacts for this customer, using only events and
      versions available before this opportunity's cutoff.</Prose></blockquote>
    <Prose>That is more informative than “previous.” The historical file's documentation and actual data must
      establish whether its field has that meaning.</Prose>

    <H3>Targets can be proxies</H3>
    <Prose>A recorded subscription is an observable event. Customer benefit, satisfaction or the incremental
      effect of calling are different quantities. Similarly, a click is not identical to a useful
      recommendation, and a short service time is not identical to a well-resolved problem.</Prose>
    <Prose>A <strong>proxy target</strong> is a measurable substitute for something harder to observe. Explain
      why it is informative and where it can diverge from the intended outcome. Improving a proxy is evidence
      about that proxy; check the downstream outcome
      separately. <a href="https://developers.google.com/machine-learning/problem-framing/ml-framing">Google's
      framing discussion</a> distinguishes model outputs, proxy labels and success measures.</Prose>

    {/* ============================================================ §3 */}
    <H2 id={headingId(headings[2])}>{headings[2]}</H2>
    <Prose>For a feature record, distinguish three different times. The <strong>event time</strong> is when the
      underlying measurement or event happened. The <strong>available-at time</strong> is when the prediction
      system could use this particular value. The <strong>prediction cutoff</strong> is the moment at which we
      must reconstruct the available information.</Prose>
    <Prose>A measurement can happen before a prediction yet arrive afterward. A database's insertion timestamp
      can help only if its contract actually represents availability to the serving system. A delayed sync,
      publication schedule or feature computation can add another delay.</Prose>
    <Prose>Use this constructed calibration history. Times are simple abstract units; values are calibration
      offsets.</Prose>
    <LessonTable caption="A constructed calibration history: four records, two of them versions of one event"
      headers={['Sensor', 'Event time', 'Available at', 'Version', 'Value']}
      rows={formulationData.timeline.records.map(record => [
        record.entity.replace('sensor_', ''), asInput(record.event), asInput(record.available),
        asInput(record.version), asInput(record.value),
      ])} />
    <Prose>A prediction for sensor A is required at time 5. Which value can it use?</Prose>
    <Prose>The event at time 4 looks newest, but value 20 does not arrive until time 8. The correction to
      event 1 arrives at time 6. At time 5, the appropriate value in this fixture is
      therefore <strong>{asInput(formulationData.timeline.cases[0].selected.value)}</strong>. Sensor B's
      available measurement belongs to the wrong entity.</Prose>
    <Prose>If the event-4 value arrives at time 4 instead, the answer
      becomes {asInput(formulationData.timeline.cases[1].selected.value)}. If the cutoff moves to 7 under the
      original history, event 4 is still unavailable but event 1's correction is known, so the answer
      becomes {asInput(formulationData.timeline.cases[2].selected.value)}. At cutoff 9, event 4's
      value {asInput(formulationData.timeline.cases[3].selected.value)} is known and is the newest eligible
      event.</Prose>
    <Prose>We have changed knowledge, not historical event order.</Prose>

    <H3>A precise reconstruction rule</H3>
    <Prose>For this particular “latest eligible calibration” policy: match the entity; keep records whose event
      time is no later than the cutoff; keep versions whose available-at time is no later than the cutoff;
      enforce the maximum allowed event age; and choose the newest eligible event, taking among its eligible
      versions the latest available one.</Prose>
    <Prose>If none qualifies, represent a missing calibration and follow the application's fallback policy. Do
      not silently substitute a future value. Equal-time ambiguity requires a documented version ordering; our
      fixture has an integer version for that purpose.</Prose>
    <Prose>Here is a complete small implementation. It illustrates the rule; a production historical join should
      use an appropriately indexed or vectorized implementation.</Prose>

    <Program example={formulationExamples['latest-known']} />
    <Prose>The results are {formulationData.timeline.cases[0].selected.value},
      {' '}{formulationData.timeline.cases[2].selected.value},
      {' '}{formulationData.timeline.cases[3].selected.value}, followed
      by <Code>None</Code>. The last query rejects event 1 as too old and event 4 as not yet available. A
      maximum-age tolerance constrains freshness; it does not make unavailable information available.</Prose>

    <TimelineLab />

    <H3>Why a backward join alone is insufficient</H3>
    <Prose>Pandas <Code>merge_asof</Code> with backward direction finds the last suitable key no greater than a
      query key, optionally within a tolerance and entity group. If that key is event time, this operation alone
      does not also enforce the available-at predicate above. Its documented behavior is a key-matching rule,
      not a guarantee about the meaning of your
      data. <a href="https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html">Pandas API</a>.</Prose>
    <Prose>Feature stores can implement historical retrieval, but inspect the selected version and backend.
      Feast's July 2026 merged change adds an opt-in created-timestamp cutoff with documented store support
      limitations. This is separate from event-time TTL matching; do not assume every historical join enables
      both conditions. It also requires that created time suitably represents availability for the
      task. <a href="https://github.com/feast-dev/feast/pull/6617">Feast change and support contract</a>.</Prose>
    <Prose>Keep historical versions if future reconstruction matters. Overwriting event 1's original value
      with {asInput(formulationData.timeline.cases[2].selected.value)} erases the information needed to
      reproduce the cutoff-5 answer.</Prose>

    {/* ============================================================ §4 */}
    <H2 id={headingId(headings[3])}>{headings[3]}</H2>
    <Prose>“Better than random guessing” can be an extremely weak target. In a dataset
      with {asInput(90)} negative and {asInput(10)} positive cases, always predicting negative
      has {prevalence.accuracy * 100}% accuracy and {prevalence.positiveRecall} recall of the positive
      class.</Prose>
    <LessonTable caption="Baselines, and the question each one answers"
      headers={['Baseline', 'What it tests']}
      rows={[
        ['Existing operational rule', 'Does the proposed system improve the decision people currently make?'],
        ['Majority class or training class prior', 'Is there useful discrimination beyond class prevalence?'],
        ['Training mean or median', 'Does a regression model improve on a constant matched to the loss?'],
        ['A transparent feature rule or small model', 'Does additional complexity earn its cost?'],
        ['Last observed or seasonal value', 'Does a forecasting model improve on temporal persistence?'],
      ]} />
    <Prose>Fit data-dependent baselines on training information. The constant minimizing training squared error
      is the training mean; the constant minimizing absolute error is a training median. To see the first,
      expand the sum around the mean:</Prose>
    <MathBlock>{'\\begin{gathered}\\sum_i(y_i-c)^2\\\\[4pt]'
      + '=\\sum_i(y_i-\\bar y)^2+n(c-\\bar y)^2.\\end{gathered}'}</MathBlock>
    <Prose>The first term does not depend on <Math>{'c'}</Math>, and the second is smallest
      at <Math>{'c=\\bar y'}</Math>. Baselines have assumptions too.</Prose>
    <Prose>A probability baseline predicts the training positive fraction for every case. Its scores all tie, so
      it supplies no ranking information. Its validation average precision equals the validation positive
      fraction; selecting a particular top 50 among tied scores depends on the tie rule and is not a learned
      advantage.</Prose>

    <H3>Match the metric to the decision</H3>
    <Prose>If capacity is {partition.capacity} calls, inspect the first {partition.capacity} ranked cases.
      Write <Math>{'T_k'}</Math> for the number of positive outcomes among the selected <Math>{'k'}</Math>,
      and <Math>{'P'}</Math> for all positive outcomes in the evaluated set:</Prose>
    <MathBlock>{'\\begin{gathered}\\mathrm{precision@}k=\\tfrac{T_k}{k}\\\\[4pt]'
      + '\\mathrm{recall@}k=\\tfrac{T_k}{P}.\\end{gathered}'}</MathBlock>
    <Prose>These answer different questions. Precision describes concentration in the selected set; recall
      describes how much of the observed positive population that set captures. Neither alone estimates how many
      outcomes calling actually causes.</Prose>
    <Prose>For probability quality, log loss penalizes assigning very low probability to outcomes that occur.
      Average precision summarizes a precision–recall ranking with its defined threshold convention. The
      earlier <a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">evaluation-metrics
      lesson</a> develops those calculations; here we use them to check a stated decision.</Prose>
    <Prose>For a binary action with fixed false-positive cost <Math>{'C_{\\rm FP}'}</Math>, false-negative
      cost <Math>{'C_{\\rm FN}'}</Math>, zero correct-action costs, and a probability <Math>{'p'}</Math> valid
      for the current case, acting has expected loss <Math>{'C_{\\rm FP}(1-p)'}</Math>, while not acting has
      loss <Math>{'C_{\\rm FN}p'}</Math>. Acting is preferable when</Prose>
    <MathBlock>{'p>\\frac{C_{\\rm FP}}{C_{\\rm FP}+C_{\\rm FN}}.'}</MathBlock>

    <ThresholdCostFigure />

    <Prose>This derivation assumes the action does not change the target's meaning and omits capacity or
      action-specific effects. With a hard capacity, decisions become coupled: selecting one case may displace
      another. Document the actual decision rule instead of treating .5 as a universal threshold.</Prose>

    {/* ============================================================ §5 */}
    <H2 id={headingId(headings[4])}>{headings[4]}</H2>
    <Prose>Our offline <a href={provenance.file}>bank-additional.csv</a> is the provider's
      unchanged {partition.sourceRows.toLocaleString('en-US')}-row random subset of historical Portuguese bank
      marketing records. It has 20 input columns and a binary recorded subscription
      target; {partition.positiveRows} rows have the positive label. The source is Moro, Rita and Cortez's Bank
      Marketing dataset under {provenance.license}. Its own description explicitly identifies final call
      duration as unavailable before a call. <a href={provenance.record}>UCI source and license</a>;
      {' '}<a href={provenance.description}>retained variable description</a>.</Prose>
    <Prose>The supplied subset lacks the complete entity and availability history needed to establish a
      deployable pre-call policy or an independent future-customer test. Our measured question is
      narrower: <strong>how do a simple recorded-feature pipeline and the same pipeline with an unavailable
      feature compare on matching held-out rows?</strong></Prose>
    <Prose>We reserve {partition.reservedRows} rows without scoring them. Within the
      other {partition.developmentRows.toLocaleString('en-US')},
      use {partition.trainRows.toLocaleString('en-US')} to fit and {partition.validationRows} for the stated
      diagnostic comparison, with fixed stratified seeds. This is development evidence, not a final operational
      acceptance result.</Prose>
    <Prose>Choose a small, explicit candidate feature set: age, previous-contact count, prior-contact
      availability and days, and seven customer and history categories. Exclude current-call duration in the
      candidate pipeline. Exclude campaign, scheduling and economic columns here rather than quietly asserting
      their exact pre-call availability. Their usefulness and historical versions would need a separate
      contract.</Prose>
    <Prose>The provider uses {formulationData.features.sentinel} in <Code>pdays</Code> to mean no previous
      contact. Treating {formulationData.features.sentinel} as an actual elapsed time would invent a distance.
      We create a contacted-before indicator and replace that sentinel with missing in the elapsed-days feature.
      Medians and scaling are fitted inside the training pipeline; categories retain the provider's explicit
      unknown values.</Prose>

    <FittingBoundaryFigure />

    <H3>Complete experiment</H3>
    <Prose>Place the CSV beside this program. It requires NumPy, pandas and scikit-learn; the recorded
      calculation used versions {formulationData.software.numpy}, {formulationData.software.pandas} and
      {' '}{formulationData.software.scikitLearn} respectively. The model is a regularized logistic regression
      with fixed settings, not a search for a winning seed.</Prose>

    <Program example={formulationExamples.experiment} />

    <Prose>The recorded validation set
      contains {partition.validationPositives} positive and {partition.validationRows - partition.validationPositives} negative
      outcomes.</Prose>

    <ResultsFigure />

    <Prose>The candidate has one fewer correct class decision than the majority baseline at threshold .5, while
      concentrating more positives in the top {partition.capacity}. This is why the operational question
      matters: classification accuracy and limited-capacity ranking assess different behavior.</Prose>
    <Prose>The duration model appears better on several numbers. That does not make final duration usable before
      a call. Its validation rows can be entirely separate from training and still contain the wrong information
      for the intended cutoff. A correct train-only scaler cannot repair that feature definition.</Prose>
    <Prose>The prior baseline's {procedures.prior.top50Positives} positives among the first {partition.capacity} tied
      cases are a consequence of the fixed source-row tie rule; all baseline probabilities are identical. It did
      not discover a ranking signal.</Prose>

    <CapacityLab />

    <Prose>The retained calculation inputs include split identities, each validation target and probability,
      actual rankings and confusion matrices; this page serves
      the <a href={provenance.program}>program that produces them</a> beside the data. The attribution file
      separates raw source facts, derived features and author calculations.</Prose>

    {/* ============================================================ §6 */}
    <H2 id={headingId(headings[5])}>{headings[5]}</H2>
    <Prose><strong>Data leakage</strong> is an information path that gives a fitting or evaluation procedure
      access to information disallowed by the intended prediction or evaluation contract. It often improves a
      score, but leakage need not produce a spectacular metric, and a high score alone does not prove
      leakage.</Prose>
    <LessonTable caption="Five leakage mechanisms and a deployment mismatch, with the repair each needs"
      headers={['Failure', 'Concrete shortcut', 'Appropriate repair']}
      rows={[
        ['Target or temporal leakage', 'Final call duration used before calling', 'Redefine features at the actual cutoff; recollect historical versions if needed'],
        ['Preprocessing leakage', 'Select features using all labels before splitting', 'Fit selection inside each training fold'],
        ['Unit contamination', "Same parcel's near-duplicate rows on both sides of a new-parcel test", 'Split at the relevant independent unit'],
        ['Selection leakage', 'Repeatedly inspect a final test to choose variants', 'Use development selection; obtain valid new evaluation for the selected procedure'],
        ['Availability or revision leakage', 'Latest corrected value used in an earlier snapshot', 'Reconstruct as-known versions'],
        ['Deployment mismatch', 'Training on completed historical calls, applying to all eligible future customers', 'Establish population coverage and evaluate the new question'],
      ]} />
    <Prose>The last row need not involve a hidden information channel. It can be a distribution or policy
      mismatch. Naming the mechanism determines the repair.</Prose>
    <Prose>A pipeline is useful because each fold can fit its imputer, scaler, selector and model together on
      that fold's training portion. It does not decide whether an input is from the future, whether two rows
      represent the same parcel, or whether the target measures the desired
      outcome. <a href="https://scikit-learn.org/stable/common_pitfalls.html#data-leakage">scikit-learn's worked
      pitfalls</a> shows train-only transformation and feature-selection examples.</Prose>

    <H3>Follow the information path</H3>
    <Prose>For a suspicious result, trace each influential feature backward. Which raw records produced it?
      Which entity and time does each record describe? When could the system use that particular version? Which
      rows and labels determined its fitted transformation? What model or reporting choice used the evaluation
      results?</Prose>

    <LineageFigure />

    <Prose>The same outcome can reach a feature through an aggregate, a status code, a filename or a human
      workflow.</Prose>
    <Prose>A useful diagnostic removes or delays the suspected feature and repeats a controlled development
      comparison. A collapse in score may support the suspected dependency; it does not prove all remaining
      inputs are valid. Conversely, little score change does not legalize an unavailable feature. The
      admissibility rule comes from the information contract, not its measured importance.</Prose>
    <Prose>Negative controls can reveal shortcuts. An entity-identity-only model that predicts supposedly unseen
      entities deserves investigation. A timestamp-only model can expose a temporal regime or a split artifact.
      Permuting labels can help diagnose some pipeline errors when the permutation respects the relevant
      structure, but it is not a universal leakage detector and does not replace provenance.</Prose>

    {/* ============================================================ §7 */}
    <H2 id={headingId(headings[6])}>{headings[6]}</H2>
    <Prose>For the real example, a defensible conclusion is:</Prose>
    <blockquote><Prose>On this fixed historical row-level development split, the candidate feature pipeline
      improves average precision from {fixed(procedures.prior.averagePrecision, 6)} to
      {' '}{fixed(procedures.candidate.averagePrecision, 6)} and top-{partition.capacity} concentration
      from {procedures.prior.top50Positives} to {procedures.candidate.top50Positives} relative to a constant
      prior, while threshold-.5 accuracy does not improve
      ({procedures.candidate.correct} against {procedures.prior.correct} of {partition.validationRows}). Adding
      final duration improves the diagnostic scores but violates a pre-call information contract. These results
      justify examining a properly timestamped, entity-aware pre-call dataset; they do not establish a deployed
      policy's benefit.</Prose></blockquote>
    <Prose>That conclusion names an actual finding and a next piece of evidence. It does not ask a larger model
      to fix missing history.</Prose>
    <Prose>A useful saved experiment packet contains the target, unit, prediction cutoff, population and
      label-maturity rules; the data identity, source and availability definitions and deterministic row, group
      and time assignments; the baseline and candidate pipelines with their fitting scope, fixed settings and
      any selection performed; the per-case outputs and the decision rule, alongside aggregate and relevant
      slice metrics; and a conclusion, the concrete unresolved assumptions and the next discriminating
      experiment.</Prose>
    <Prose>Do not freeze a misleading problem forever. If writing the contract reveals that the task should be
      “predict unresolved requests after 30 seconds,” revise the target, cutoff and available partial-history
      features together. A final conversation duration from old data cannot stand in for elapsed duration
      observed at 30 seconds. Reconstruct or collect the correct training examples.</Prose>
    <Prose>This connects naturally to the
      next <a href="/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml">Time-Series
      Validation &amp; Forecasting Baselines</a> lesson: once the deadline and label window are explicit, we can
      move them forward in time and build a rolling evaluation that respects them.</Prose>

    {/* ============================================================ §8 */}
    <H2 id={headingId(headings[7])}>{headings[7]}</H2>
    <Prose>This section is a deeper branch. It answers questions the core route raises but does not need.</Prose>

    <H3>Prediction is not an intervention effect</H3>
    <Prose>Suppose two original hypothetical customer groups have the probabilities in figure 7 under two
      actions. Ranking by the first column favors one group; ranking by the incremental effect favors the other.
      These probabilities are invented to expose the distinction; the bank file does not identify both potential
      outcomes for each customer.</Prose>

    <UpliftFigure />

    <Prose>Observed outcomes under a historical calling policy estimate associations in that collected
      population. Estimating the effect of changing the policy needs experimental or justified causal evidence.
      The
      preceding <a href="/learn/path/full-curriculum/bayesian-networks-causal-graphical-models?module=classical-ml">causal-graphical-models</a> lesson
      supplies that framework. No amount of ordinary holdout accuracy supplies an unobserved counterfactual by
      itself.</Prose>

    <H3>Missing labels can be selected by the old system</H3>
    <Prose>If only inspected machine parts receive a precise defect label, the labeled table describes inspected
      parts. An old inspection rule may preferentially select unusual parts. A model trained and evaluated
      within that selected table can perform well yet be poorly assessed for uninspected production.</Prose>
    <Prose>Write the selection process into the contract: who was observed, who was omitted, which outcomes
      became known, and whether there is support for the intended population. Weighting or extrapolation needs
      assumptions; it is not a way to create evidence in a region with no relevant observations.</Prose>

    <H3>The loss should preserve information needed by the action</H3>
    <Prose>Consider provisioning spare parts for a one-period demand of 10 or 20 with equal probability.
      Predicting mean demand gives 15. But if each missing part costs three units and each unused part costs
      one, the expected costs of stocking 10 and 20 are the two values in figure 7's second table. Under this
      simplified cost model, mean prediction plus an automatic “stock the mean” rule is not the optimal
      decision.</Prose>
    <Prose>For underage cost <Math>{'c_u'}</Math> and overage cost <Math>{'c_o'}</Math>, minimizing expected
      asymmetric absolute loss selects a quantile at level</Prose>
    <MathBlock>{'\\tau=\\frac{c_u}{c_u+c_o}.'}</MathBlock>
    <Prose>For a continuous demand distribution with CDF <Math>{'F'}</Math>, increasing stock slightly adds
      overage cost on the fraction <Math>{'F(q)'}</Math> below the current level and removes underage cost on
      the fraction <Math>{'1-F(q)'}</Math> above it. The derivative is therefore</Prose>
    <MathBlock>{'c_oF(q)-c_u\\bigl(1-F(q)\\bigr),'}</MathBlock>
    <Prose>which vanishes at that quantile; at a point mass use the corresponding one-sided condition. In this
      example the {round(criticalQuantile(3, 1), 6)} quantile is 20.</Prose>
    <Prose>This derivation is not new here, and its two nearest owners are both behind
      you. <a href="/learn/path/full-curriculum/decision-theory-risk-cost-sensitive-decisions?module=classical-ml">Decision
      Theory (Risk &amp; Cost-Sensitive Decisions)</a> gives this same critical fraction under “absolute error
      requests a median; asymmetric error requests a quantile”, with a worked underage and overage order and
      the discrete-quantile subtlety this section
      skips. <a href="/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml">Calibration
      &amp; Conformal Prediction</a>, an earlier lesson in this module, defines <strong>pinball loss</strong> by
      name where it conformalises a quantile regressor. What neither supplies — and what this curriculum has no
      dedicated lesson for — is the <em>fitting</em> technique itself;
      for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html">that,
      scikit-learn's QuantileRegressor</a> is an implementation to read against those two pages rather than
      instead of them.</Prose>
    <Prose>This is why “classification for fixed thresholds, regression for changing thresholds” is a heuristic,
      not a theorem. A conditional distribution, calibrated probability or quantile may preserve more useful
      information than a hard category. Choose outputs and losses for the actual decisions and information
      available.</Prose>
    <Prose>Core readiness does not require a causal estimator or a quantile proof. The deeper skill is
      recognizing when the target, observed labels and intended action no longer describe the same
      problem.</Prose>

    {/* ============================================================ §9 */}
    <H2 id={headingId(headings[8])}>{headings[8]}</H2>

    <Practice title="1. A changed historical snapshot"
      question={`For sensor A, records are (event, available, version, value): (2, 2, 1, 8), (4, 7, 1, 11), `
        + `(2, 5, 2, 9). At cutoff ${practiceTimelineFixture.cutoff} with maximum age `
        + `${practiceTimelineFixture.maximumAge}, which value qualifies? What changes if the event-4 value `
        + `arrives at time 4? What if maximum age becomes 1?`}
      hint="Apply entity, event, availability and age conditions before choosing a version.">
      <Prose>Originally choose {asInput(practiceTimeline.value)}: event 2's correction is known
        by {practiceTimelineFixture.cutoff}, while event 4 is unavailable. Moving its arrival to 4
        makes {asInput(practiceTimelineEarlier.value)} eligible and newest. With maximum age 1, event times must
        be at least {practiceTimelineFixture.cutoff - PRACTICE_TIGHT_AGE}, so
        {' '}{practiceTimelineTight.selected === null && practiceTimelineTightEarlier.selected === null
          ? 'none qualifies in either arrival case'
          : 'the answer depends on the arrival'}. Keep missingness visible rather than inventing a future
        calibration. Apply the same eligibility and version-selection steps as investigation 1 to these
        changed records.</Prose>
    </Practice>

    <Practice title="2. A convincing but invalid delivery model"
      question="A model predicts whether a parcel will arrive late using a field set by the final delivery scan. Its test parcels are all different from training parcels. Explain why the split is insufficient and write a valid cutoff and feature rule.">
      <Prose>Holding out parcel identities prevents one contamination mechanism, but a post-delivery field is
        still unavailable at dispatch. Specify a dispatch-time prediction and reconstruct only fields, events
        and versions available then. Define “late” and label maturity independently. If the goal changes to
        post-delivery reporting, acknowledge that it is a different task and test whether prediction is even
        needed.</Prose>
    </Practice>

    <Practice title="3. Accuracy, service cost and capacity"
      question={`There are ${supportFixture.requests} support requests, ${supportFixture.urgent} truly urgent. `
        + `Policy A escalates none. Policy B escalates ${supportFixture.policies[1].escalated}: three urgent `
        + `and one ordinary. An unnecessary escalation costs ${supportFixture.unnecessaryCost} units; missing `
        + `an urgent request costs ${supportFixture.missedCost}. Compute accuracy and cost for both. If `
        + `capacity is only ${supportFixture.capacity} escalations, is B feasible as written?`}
      hint="Count errors by type before using a single total.">
      <Prose>A is correct on {policyA.correct}/{policyA.total} with cost {policyA.cost}. B is correct
        on {policyB.correct}/{policyB.total} with cost {policyB.cost}, but
        {policyB.withinCapacity ? ' stays within' : ' exceeds'} capacity {policyB.capacity}. A revised
        rank-and-select rule needs to state which {supportFixture.capacity} are selected; B's four-case
        confusion counts alone do not determine the new result. This is an action constraint, not something an
        accuracy number encodes.</Prose>
    </Practice>

    <Practice title="4. Change the capacity on real data"
      question={`Using the saved validation probabilities or rerunning the displayed fixed experiment, change `
        + `capacity from ${partition.capacity} to 25. Calculate the candidate's precision and `
        + `recall changes and explain them from the selected rows. Use the same source-row tie rule and leave model settings `
        + `unchanged.`}
      hint="The top 25 are a subset of the top 50 under the fixed ranking. Recall cannot increase; precision can move either way.">
      <Prose>Sort by descending probability and then source-row index, count positives among the first 25, and
        divide by 25 for precision and by {partition.validationPositives} for recall. The retained candidate
        ranking contains {practiceAtTwentyFive.positivesFound} positives in its first 25:
        precision {round(practiceAtTwentyFive.precision, 6)} and
        recall <Math>{`${practiceAtTwentyFive.positivesFound}/${partition.validationPositives}`}</Math> =
        {' '}{round(practiceAtTwentyFive.recall, 6)}, compared with {round(procedures.candidate.precisionAt50, 6)} and
        {' '}<Math>{`${procedures.candidate.top50Positives}/${partition.validationPositives}`}</Math> =
        {' '}{round(procedures.candidate.recallAt50, 6)} at capacity {partition.capacity}. Precision rises here
        while recall falls; a higher precision is not guaranteed by the word “top.” Include the selected
        rows and the actual result; do not search for a favorable capacity and report it as untouched
        evaluation. Investigation 2 lets you inspect exactly this change.</Prose>
    </Practice>

    <Practice title="5. Write the missing contract"
      question="A colleague says, “Our model detects failed equipment with 97% accuracy on randomly split sensor rows.” List at least five missing definitions, then propose an evaluation for predicting failure of a new machine within the next day.">
      <Prose>Specify the machine or forecast-origin unit, the failure event and 24-hour window, feature and
        label availability, the population of machines, repeated-measurement grouping, a baseline, meaningful
        costs and metrics, and the split time. Hold out the intended new machines and respect prediction
        deadlines and mature labels within each training snapshot. Describe how many independent machines and
        events support the evaluation; thousands of rows need not mean thousands of independent units.</Prose>
    </Practice>

    <Practice title="6. Derive a changed decision threshold"
      question="For the fixed binary-cost setting in section 4, let false-positive cost be 3 and false-negative cost 9. Compute the threshold. At p = .2, compare both expected losses. Name a condition that would invalidate this simple decision calculation."
      hint="The threshold is one ratio of the two costs; the comparison at a given p is two multiplications.">
      <Prose>The threshold is <Math>{'3/12'}</Math> = {round(practiceThreshold, 6)}. Acting
        costs {round(practiceLosses.acting, 6)}; not acting costs {round(practiceLosses.waiting, 6)}, so
        do {practiceLosses.preferred === 'act' ? '' : 'not '}act under these assumptions. A hard shared
        capacity, a changed outcome under intervention, different per-case costs or probabilities invalid for
        the deployment population requires a revised calculation. Figure 3's middle setting is this exact
        case.</Prose>
    </Practice>

    <Practice title="7. A delayed label is not a negative label"
      question="You predict an event within seven days of signup. At a dataset snapshot on day 20, a person signed up on day 18 and has no event recorded. Explain why assigning a negative label can be wrong. How would the answer differ if the event already occurred and was reliably recorded on day 19?">
      <Prose>Only two days of the seven-day window have elapsed, so absence so far does not establish a
        complete-window negative. Wait for maturity or use a method that explicitly represents incomplete
        follow-up. A reliably observed event on day 19 establishes a positive within the window; delays in
        recording would need their own rule. The exact training policy should state which known positives and
        incomplete negatives are eligible.</Prose>
    </Practice>

    <Practice title="8. Deeper transfer: rank propensity or impact?"
      question="In an invented outreach example, group C has outcome probabilities .6 with action and .55 without; D has .4 with action and .1 without. Which group has greater observed-action propensity, which greater increment, and why can't a standard classifier on acted-upon cases alone settle the second question?">
      <Prose>C has higher probability under action, .6 versus .4. D has greater increment, .3 versus .05. A
        classifier on acted-upon cases does not directly observe their no-action counterfactuals; the increment
        needs an appropriate experimental or causal identification strategy. These are constructed
        probabilities, not an effect estimate from the bank data.</Prose>
    </Practice>

    <Prose>You are ready for the next lesson when you can write a target, unit and cutoff contract; reconstruct
      a changed as-known snapshot; select and interpret a meaningful baseline; identify an actual information
      shortcut; and state what your evaluation does and does not answer.</Prose>

    <Callout title="Where this page's numbers come from">
      <Prose>The dataset served here is <a href={provenance.file}>bank-additional.csv</a>,
        {' '}{provenance.bytes.toLocaleString('en-US')} bytes,
        SHA-256 <Code>{provenance.sha256}</Code>, with
        its <a href={provenance.attribution}>attribution and full provenance</a> and
        the <a href={provenance.description}>provider's own variable description</a>. It
        is {provenance.subset}, retrieved {provenance.retrieved}. No other lesson in this curriculum serves this
        dataset, and this page reads only its own copy. The {partition.reservedRows} reserved rows are not
        scored anywhere on this page.</Prose>
      <Prose>Both programs above are shown exactly as they ran: each one's bytes are lifted from the frozen
        content packet's manuscript, pinned by SHA-256, and executed on
        Python {formulationExamples.experiment.environment.python} with
        NumPy {formulationExamples.experiment.environment.numpy},
        pandas {formulationExamples.experiment.environment.pandas} and
        scikit-learn {formulationExamples.experiment.environment['scikit-learn']}. The output blocks are
        captured standard output, not transcriptions. The
        complete <a href={provenance.program}>calculation program</a> is served beside the data; run it in a
        directory holding the CSV and it rewrites the {formulationExamples.calculations.producesBytes.toLocaleString('en-US')}-byte
        results file it was originally recorded from, byte for byte.</Prose>
      <Prose>Every other number is exact arithmetic on a stated formula, evaluated in the browser from the same
        definitions the verifiers use. The calibration history, the parcels, the support costs, the spare-parts
        demand and the two customer groups are constructed teaching fixtures; no value in them comes from the
        bank records.</Prose>
    </Callout>

    <Sources alternatives={<>
      <p>For a short guided route through outcomes, outputs and proxy
        labels, <a href="https://developers.google.com/machine-learning/problem-framing">Google's Introduction
        to ML Problem Framing</a> is an interactive reading course, especially
        {' '}<a href="https://developers.google.com/machine-learning/problem-framing/problem">understanding the
        problem</a> and <a href="https://developers.google.com/machine-learning/problem-framing/ml-framing">framing
        outputs and success</a>. Read it after section 1. The core text and section structure were inspected;
        its categorical model-choice and feature-correlation rules should be treated as heuristics, with the
        qualifications section 4 gives here.</p>
      <p>For executable explanations of the preprocessing and feature-selection
        cases, <a href="https://scikit-learn.org/stable/common_pitfalls.html">scikit-learn's common pitfalls</a> is
        a short article with runnable examples. Read it beside section 6. The relevant text and code were read;
        its example scores are separate from the measured study on this page.</p>
    </>}>
      <li><a href="https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html">Pandas
        — <Code>merge_asof</Code></a> — the backward matching, grouping, sorting and tolerance contract to read
        alongside section 3. An event-time key alone does not express a second availability condition.</li>
      <li><a href="https://docs.feast.dev/getting-started/concepts/point-in-time-joins">Feast — point-in-time
        joins</a>, with <a href="https://github.com/feast-dev/feast/pull/6617">the merged created-time filtering
        change</a> — a systems-oriented alternative. Event freshness, known-time filtering and backend support
        are separate concerns; the change record was inspected, and no Feast installation was executed.</li>
      <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html">scikit-learn
        — <Code>QuantileRegressor</Code></a> — the pinball-loss implementation behind section 8's quantile
        connection.</li>
      <li><a href={provenance.record}>UCI — Bank Marketing</a>, {provenance.creator}, DOI
        {' '}<a href={provenance.doi}>{provenance.doi.replace('https://doi.org/', '')}</a>,
        {' '}<a href={provenance.licenseUrl}>{provenance.license}</a>. The provider requests citation of Moro,
        Cortez and Rita, <em>A Data-Driven Approach to Predict the Success of Bank Telemarketing</em>, Decision
        Support Systems (2014). The packet preserves the original variable description and the source paper
        citation; the full subscription-paper text was not reviewed.</li>
    </Sources>
  </div>,
};

export default formulationContent;
