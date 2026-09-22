import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  EligibilityLab, ForecastRequestLab, SeasonalDonorLab,
} from '../../components/lesson-labs/TimeSeriesLabs.jsx';
import {
  ArrivalFigure, ComparisonFigure, ContinuationFigure, IssueTimeFigure, StaircaseFigure,
} from '../../components/lesson-labs/TimeSeriesFigures.jsx';
import { asInput, fixed, rentals } from '../../components/lesson-labs/TimeSeriesShared.jsx';
import { timeSeriesExamples } from '../timeseries-examples.js';
import { timeSeriesData } from '../timeseries-data.js';
import {
  METHOD_LABELS, baselineForecasts, errorSummary, fixtures, horizonFit, informationSetAudit,
  naiveErrorScale, recursiveTrace, seasonalDonorIndex, seasonalNaiveCoincidence, splitterGap, toyComparison,
  updatedTrace, weekdayName,
} from '../timeseries-models.js';

/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.round` here would resolve to that component and yield
   undefined rather than a number. Every quantity on this page therefore comes
   from `timeseries-models.js` or from the recorded data, never from arithmetic
   typed into the prose. */

const toy = toyComparison;
const toyChanged = baselineForecasts({ history: fixtures.toyEdited.history, horizon: 4, period: 2 });
const toyChangedScore = errorSummary(fixtures.toy.future, toyChanged.seasonal);
const recursion = recursiveTrace({
  lastObserved: fixtures.recursion.lastObserved, step: fixtures.recursion.step, horizon: 4,
});
const updated = updatedTrace({
  lastObserved: fixtures.recursion.lastObserved, step: fixtures.recursion.step,
  outcomes: fixtures.recursion.outcomes,
});
const updatedChanged = updatedTrace({
  lastObserved: fixtures.recursionChanged.lastObserved, step: fixtures.recursionChanged.step,
  outcomes: fixtures.recursionChanged.outcomes,
});
const recursionScore = errorSummary(fixtures.recursion.outcomes, recursion.predictions);
const updatedScore = errorSummary(fixtures.recursion.outcomes, updated.predictions);
const recursionChangedScore = errorSummary(fixtures.recursionChanged.outcomes, recursion.predictions);
const updatedChangedScore = errorSummary(fixtures.recursionChanged.outcomes, updatedChanged.predictions);

const development = Object.fromEntries(timeSeriesData.development.map(row => [row.key, row]));
const final = Object.fromEntries(timeSeriesData.final.map(row => [row.key, row]));
const selected = timeSeriesData.selectedMethod;
const protocol = timeSeriesData.protocol;
const provenance = timeSeriesData.provenance;
const weekdays = protocol.targetWeekdays;
/* The history the FIRST FINAL ORIGIN actually sees, not a round number chosen
   for illustration: the page states this position out loud, so it has to be a
   position this experiment really has. */
const finalHistoryLength = protocol.finalOriginIndices[0] + 1;
const h7 = seasonalNaiveCoincidence({ historyLength: finalHistoryLength, period: 7, horizon: 7 });

/* The exact numbers investigation 3 uses, computed here so the prose and the
   lab cannot drift apart. */
const requestCases = [fixtures.request, fixtures.requestSecond].map(setup => {
  const counts = timeSeriesData.series.counts;
  const dates = timeSeriesData.series.dates;
  const history = counts.slice(0, setup.origin + 1);
  const rules = baselineForecasts({ history, horizon: 7, period: setup.period });
  const actual = counts.slice(setup.origin + 1, setup.origin + 8);
  return {
    ...setup,
    date: dates[setup.origin],
    seasonal: errorSummary(actual, rules.seasonal).mae,
    naive: errorSummary(actual, rules.naive).mae,
  };
});

/* Practice 8's scale, from its own history rather than from a typed answer. */
const practiceScale = naiveErrorScale([2, 4, 3, 5], 1);
const practiceScaled = 1.5 / practiceScale.scale;
const practiceConstantScale = naiveErrorScale([4, 4, 4, 4], 1);

/* Practice 2's changed cycle. */
const practiceCycle = baselineForecasts({ history: [3, 9, 6, 4, 10, 8], horizon: 5, period: 3 });
const practiceCycleChanged = baselineForecasts({ history: [3, 9, 6, 4, 12, 8], horizon: 5, period: 3 });
const practiceDonor = seasonalDonorIndex({ historyLength: 6, period: 3, horizon: 5 });

/* Practice 4's two error profiles. */
const profileA = errorSummary([0, 0, 0, 0], [0, 0, 4, 4]);
const profileB = errorSummary([0, 0, 0, 0], [2, 2, 2, 2]);

/* Practice 1 and practice 7's eligibility arithmetic, derived. */
const practiceOneAtTen = [4, 5, 6, 7, 8, 9, 10].filter(origin => origin + 3 + 1 <= 10);
const practiceOneAtTwelve = [4, 5, 6, 7, 8, 9, 10].filter(origin => origin + 3 + 1 <= 12);
const practiceSeven = splitterGap({ horizon: 3, delay: 0 });
const practiceSevenLatest = 20 - 3;

/* One rehearsal of the real schedule, audited, so section 4's claim is a
   computed result rather than a sentence. */
const sampleFit = horizonFit({ issueOrigin: 364, horizon: 7, firstTrainOrigin: 6 });
const sampleAudit = informationSetAudit({
  origin: sampleFit.issueOrigin,
  trainingRows: sampleFit.rows.map(row => ({ origin: row.origin, features: row.features, label: row.label })),
  predictionFeatures: sampleFit.predictionFeatures,
});

const headings = [
  '1. Put a clock on the prediction',
  '2. Build alternatives that a complicated model must beat',
  '3. Turn history into features without importing the answer',
  '4. Rehearse the deployment schedule',
  '5. Forecast a real week of bicycle rentals',
  '6. Practice on changed requests',
  '7. Deeper connections that change the design',
  'Where to go next',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Practice({ title, question, hint, children }) {
  return <section className="ts-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>Show the explained solution</summary>{children}</details>
  </section>;
}

const timeSeriesValidationContent = {
  title: 'Time-Series Validation & Forecasting Baselines',
  readTime: '~50 min first pass · ~95 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot ts-lesson">
    <LessonIntro
      prerequisites={<>Arrays, a train/validation split, and the idea that a fitted scaler learns from training
        data. The preceding <a href="/learn/path/full-curriculum/ml-problem-formulation-baselines-data-leakage?module=classical-ml">ML
        Problem Formulation, Baselines &amp; Data Leakage</a> lesson separates an event's time from the time we
        learned about it; forecasting adds a second separation and this lesson develops it. Forecast notation
        and error summaries are defined here.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      It is Saturday evening. You know how many bicycles were rented each day so far, and must plan for the next
      seven days. A spreadsheet downloaded months later contains the weather and rental totals for that entire
      week. Those columns make prediction easy — after the week has happened. You will reconstruct the harder
      moment: {provenance.rows} real days of rentals, forecasts issued every Saturday, and a comparison against
      rules so simple they cannot be accused of cheating. The selected model wins on aggregate
      at {fixed(final[selected].mae, 2)} rentals a day against {fixed(final.naive.mae, 2)} for naive — and loses
      at three of seven horizons. Every investigation updates its topic-specific results from valid control changes, with no expected-answer input.
    </LessonIntro>

    <div className="ts-route"><Prose><strong>First pass.</strong> Read sections 1–6 and do practice 1–6. That
      route gets you the three clocks a forecast carries, the four baseline rules and which observation each one
      copies, the one inequality that decides whether a training row may be used, the rolling-origin rehearsal
      and its refit schedule, and a real measured comparison you can judge. Run the three short programs on the
      way. Section 7 is a deeper branch on overlapping targets, uncertainty, scaling and changing environments;
      practice 7 and 8 belong with it. Its specialist branches are not extra entry requirements.</Prose></div>

    <Prose>A forecasting experiment must recreate the moment when the answer was still unknown. That requires
      more than putting a date column into a model. We need to reconstruct the information available when a
      prediction was made, specify how far ahead it reached, and compare it with a useful alternative at the
      same moment.</Prose>

    <Callout title="Which numbers here are constructed, which are measured, and which are counterfactual">
      Three kinds of quantity appear on this page and are never mixed. <strong>Exact constructed
      fixtures</strong> — the six-value operating cycle of section 2, the recursion traces of section 3, the
      label-arrival clock of investigations 2 — are small-integer arithmetic with no tolerance and no
      claim to be observations. <strong>Measurements</strong> are the development and final results of
      section 5: {protocol.developmentRidgeFits + protocol.finalRidgeFits} model fits on
      the {provenance.rows}-day recorded series, attributed where they appear. <strong>Counterfactual
      arithmetic</strong> appears only inside investigation 3, when you replace a count; it is labelled the
      moment you do, and it is an exploration on a real series rather than a correction to it. This caution is
      stated once; the rest of the lesson refers back to it rather than repeating it.
    </Callout>

    {/* ============================================================ §1 */}
    <H2 id={headingId(headings[0])}>{headings[0]}</H2>
    <Prose>The preceding <a href="/learn/path/full-curriculum/ml-problem-formulation-baselines-data-leakage?module=classical-ml">problem-formulation
      lesson</a> separated an event's time from the time we learned about it. Forecasting adds a second
      separation: <strong>when the forecast is issued</strong> versus <strong>when the predicted outcome
      occurs</strong>.</Prose>

    <LessonTable caption="The terms this lesson uses, in the bicycle example"
      headers={['Term', 'Meaning in the bicycle example']}
      rows={[
        [<>Observed value <Math>{'y_t'}</Math></>, <>Rentals recorded on day <Math>{'t'}</Math></>],
        [<>Forecast origin <Math>{'t'}</Math></>,
          <>The time we issue the forecast; in this immediate-reporting example, the end of day <Math>{'t'}</Math></>],
        [<>Horizon <Math>{'h'}</Math></>, 'Number of days from that origin to the target day'],
        [<>Forecast <Math>{'\\hat y_{t+h\\mid t}'}</Math></>,
          <>Prediction of day <Math>{'t+h'}</Math>, made using information available at origin <Math>{'t'}</Math></>],
        [<>Forecast error <Math>{'e_{t,h}'}</Math></>,
          <>Actual value minus that forecast: <Math>{'y_{t+h}-\\hat y_{t+h\\mid t}'}</Math></>],
        ['Refit schedule', 'When the learning procedure is allowed to update its fitted parameters'],
      ]} />

    <Prose>The vertical bar means “given information through this origin.” It does not mean division. At the end
      of Saturday, Sunday's count is a horizon-one target; next Saturday's count is a horizon-seven
      target.</Prose>
    <Prose>Two requests can target the same date with different difficulty. A prediction for next Saturday made
      today and a prediction for that Saturday made on Friday have different information. Store both origin and
      target date; grouping only by target date loses the distinction.</Prose>

    <IssueTimeFigure />

    <Prose>Our decision contract is: predict each of the next seven daily rental totals every Saturday after
      that day's count is available. Give each forecast equal weight in the initial comparison. Calendar dates
      are known; future observed weather and rental outcomes are not. A later example assumes counts arrive at
      the end of their recorded day because the source does not supply actual reporting timestamps. A deployment
      using delayed counts would need a different availability rule.</Prose>
    <Prose>Forecasting is one kind of temporal prediction. Estimating today's outcome from measurements already
      available today is a <strong>nowcasting</strong> or contemporaneous prediction problem, depending on the
      setting. Filling a missing historical value after observing later dates is a retrospective reconstruction
      problem. These can be legitimate tasks, but their scores do not answer the Saturday forecasting
      question.</Prose>

    {/* ============================================================ §2 */}
    <H2 id={headingId(headings[1])}>{headings[1]}</H2>
    <Prose>A <strong>baseline</strong> is a complete prediction rule that gives the model a meaningful
      comparison. “Predict zero” is computationally cheap, but often answers little about whether learning was
      useful. A recent value or the corresponding day of the previous week is more informative.</Prose>
    <Prose>Start with a constructed six-day history:</Prose>
    <MathBlock>{'10,\\;20,\\;10,\\;20,\\;12,\\;22.'}</MathBlock>
    <Prose>Imagine a two-day operating cycle, with quieter and busier days. We need four future predictions from
      this one origin. Four rules summarize different beliefs:</Prose>
    <ul>
      <li><strong>Historical mean:</strong> repeat the average of the available history. This treats the overall
        level as useful and ignores order.</li>
      <li><strong>Naive:</strong> repeat the most recent observed value. This treats the current level as the
        best simple guide.</li>
      <li><strong>Seasonal naive:</strong> repeat the last observed cycle. “Seasonal” can mean a weekday cycle
        or a machine's repeating operating schedule; it need not mean summer and winter.</li>
      <li><strong>Drift:</strong> continue the average change from the first to the last observation. This is a
        deliberately simple trend rule.</li>
    </ul>

    <LessonTable caption="Four rules from the same origin, on the same six observations"
      headers={['Rule', 'Four forecasts from this origin', 'What it carries forward']}
      rows={[
        ['Mean', toy.rules.mean.map(value => fixed(value, 3)).join(', '), 'Average of all six values'],
        ['Naive', toy.rules.naive.map(asInput).join(', '), 'Last level'],
        ['Seasonal, period 2', toy.rules.seasonal.map(asInput).join(', '), 'Last two-day pattern'],
        ['Drift', toy.rules.drift.map(asInput).join(', '),
          `Last level plus ${asInput(toy.rules.driftSlope)} per day`],
      ]} />

    <Prose>The drift slope is <Math>{'(22-10)/(6-1)='}</Math>{asInput(toy.rules.driftSlope)}: six observations
      span five intervals. Its horizon-<Math>{'h'}</Math> prediction
      is <Math>{'22+'}</Math>{asInput(toy.rules.driftSlope)}<Math>{'h'}</Math>, not the mean of the
      observations.</Prose>
    <Prose>For a history <Math>{'y_1,\\ldots,y_T'}</Math>, write the mean, naive and drift rules as</Prose>
    <MathBlock>{'\\begin{gathered}\\hat y^{\\text{mean}}_{T+h\\mid T}=\\frac1T\\sum_{i=1}^T y_i,\\\\[4pt]'
      + '\\hat y^{\\text{naive}}_{T+h\\mid T}=y_T,\\\\[4pt]'
      + '\\hat y^{\\text{drift}}_{T+h\\mid T}=y_T+h\\frac{y_T-y_1}{T-1}.\\end{gathered}'}</MathBlock>
    <Prose>For season length <Math>{'m'}</Math>, cycle through the final <Math>{'m'}</Math> observed values. In
      zero-based array notation, the selected historical index
      is <Math>{'T-m+((h-1)\\bmod m)'}</Math>, where <Math>{'T'}</Math> is the number of history entries. This
      works even when the requested horizon exceeds one cycle; the rule repeats <strong>observed
      history</strong>, not future answers. The <a href="https://otexts.com/fpp3/simple-methods.html">FPP3
      baseline chapter</a> gives further examples and the equivalent mathematical indexing.</Prose>
    <Prose>Suppose the four future outcomes turn out to
      be {fixtures.toy.future.map(asInput).join(', ')}. <strong>Mean absolute error</strong> (MAE) averages the
      sizes of the misses, ignoring their signs. Mean and naive each have
      MAE {fixed(toy.summaries.mean.mae, 0)}, seasonal naive has {fixed(toy.summaries.seasonal.mae, 0)}, and
      drift has {fixed(toy.summaries.drift.mae, 0)}. Seasonal naive wins this constructed continuation because
      its assumed cycle matches it. If the continuation changes, the ranking may change.</Prose>

    <Program example={timeSeriesExamples['baseline-rules']} />
    <Prose>The printed rows are the four rules on exactly the history above. The last line is the contrast the
      investigation below is built on: changing the fifth history value
      from {asInput(fixtures.toy.history[4])} to {asInput(fixtures.toyEdited.history[4])} moves the seasonal
      forecast to {toyChanged.seasonal.map(asInput).join(', ')} — MAE {fixed(toyChangedScore.mae, 0)} on the
      same revealed future — while the naive and drift forecasts do not move at all, because the endpoint values
      they need did not change.</Prose>

    <SeasonalDonorLab />

    <Prose>This small mechanism has a practical application beyond transport. A server with a weekly maintenance
      cycle can have a better baseline from the same weekday than from yesterday. Conversely, after a lasting
      level change, a long historical mean may adapt too slowly. A baseline is a hypothesis about what
      persists.</Prose>

    {/* ============================================================ §3 */}
    <H2 id={headingId(headings[2])}>{headings[2]}</H2>

    <H3>Name the row by its origin</H3>
    <Prose>A lag is an earlier value used as a feature. For a training row whose origin
      is <Math>{'s'}</Math>, useful features might be <Math>{'y_s'}</Math>, <Math>{'y_{s-1}'}</Math>,
      {' '}<Math>{'y_{s-6}'}</Math>, and the mean of <Math>{'y_{s-6},\\ldots,y_s'}</Math>. Its horizon-three
      label is <Math>{'y_{s+3}'}</Math>.</Prose>
    <Prose>That row can enter a fit at current origin <Math>{'t'}</Math> only when its features <strong>and its
      label</strong> are available by <Math>{'t'}</Math>. If counts arrive immediately at day end, the label
      condition is <Math>{'s+3\\le t'}</Math>. With a two-day reporting delay it
      becomes <Math>{'s+3+2\\le t'}</Math>. The historical feature snapshot must also change: a forecast issued
      at day <Math>{'s'}</Math> can then use counts only through day <Math>{'s-2'}</Math>,
      not <Math>{'y_s'}</Math>. Reconstruct the inputs as they were available at that row's own issue time. A
      later-completed CSV must not backfill future knowledge into historical feature snapshots.</Prose>

    <ArrivalFigure />

    <Prose>At cutoff {fixtures.eligibility.cutoff}, horizon {fixtures.eligibility.horizon} and
      delay {fixtures.eligibility.delay}, eligible origins from the offered
      list {fixtures.eligibility.origins[0]},&nbsp;…,&nbsp;{fixtures.eligibility.origins[fixtures.eligibility.origins.length - 1]} are
      only {fixtures.eligibility.origins
        .filter(origin => origin + fixtures.eligibility.horizon + fixtures.eligibility.delay <= fixtures.eligibility.cutoff)
        .join(' and ')}. At zero delay, origins {fixtures.eligibilityNoDelay.origins
        .filter(origin => origin + fixtures.eligibilityNoDelay.horizon <= fixtures.eligibilityNoDelay.cutoff)[0]} through
      {' '}{fixtures.eligibilityNoDelay.origins
        .filter(origin => origin + fixtures.eligibilityNoDelay.horizon <= fixtures.eligibilityNoDelay.cutoff)
        .at(-1)} are eligible. At cutoff {fixtures.eligibilityEmpty.cutoff} with the original horizon and delay,
      none are eligible.</Prose>
    <Prose>This derives the omitted recent rows from an actual information constraint. A gap is not a magic
      constant copied from a library example. In an equally spaced, origin-indexed array, a splitter that ends
      training at <Math>{'t-g-1'}</Math> before test origin <Math>{'t'}</Math>
      needs <Math>{'g\\ge h+d-1'}</Math> to enforce <Math>{'s+h+d\\le t'}</Math>. The minus one comes from the
      splitter's indexing convention. If predictions are issued before the day's count arrives, use that earlier
      cutoff instead. With irregular dates or different label durations, evaluate the timestamp inequalities
      directly.</Prose>

    <EligibilityLab />

    <H3>Rolling statistics must face backward from the actual origin</H3>
    <Prose>If the row predicts tomorrow from today's end, today's count is a valid feature. If the row is
      indexed by tomorrow's target date instead, the same feature is a one-day shift. Many off-by-one mistakes
      come from switching these conventions mid-program.</Prose>
    <Prose>For target-indexed daily prediction, the historical seven-day mean is conceptually</Prose>
    <CodeBlock language="python">{timeSeriesExamples['rolling-illustration'].code}</CodeBlock>
    <Prose>For origin-indexed prediction, the seven-day window may include the origin's count. Neither formula
      is universally correct without naming the prediction time. A centered rolling mean uses observations on
      both sides and is unsuitable for an online forecast unless those later values were genuinely available
      under the stated task.</Prose>
    <Prose>Fit medians, scalers, dimensionality reduction and feature selectors inside the eligible training
      set. Deterministic calendar features can be computed in advance because next Tuesday's weekday is already
      known. A weather <strong>forecast issued before the origin</strong> could also be valid, but the weather
      later measured on Tuesday is a different feature. Use the archived forecast version, not a
      retrospectively corrected observation.</Prose>
    <Prose>Missing dates need an explicit policy. In a daily series, “previous row” means “previous day” only
      when the calendar is complete. Reindex to the intended calendar, distinguish a missing report from zero
      events, and choose a causal fill or a model that handles missingness. Interpolating across a future
      observation may be useful for historical visualization while leaking information into a forecast
      experiment.</Prose>

    <H3>Seven-day forecasts are not seven updated one-day forecasts</H3>
    <Prose>Consider the deliberately simple one-step rule “last count
      plus {asInput(fixtures.recursion.step)}.” From the final historical
      count {asInput(fixtures.recursion.lastObserved)}, a <strong>recursive</strong> four-day forecast
      is {recursion.predictions.map(asInput).join(', ')}: each prediction supplies the next input.</Prose>
    <Prose>If the future outcomes are {fixtures.recursion.outcomes.map(asInput).join(', ')}, using each newly
      observed outcome before predicting the next produces {updated.predictions.map(asInput).join(', ')}. That
      is a different procedure. It is legitimate if we issue and update a one-day forecast each evening. It is
      invalid evidence for four forecasts that supposedly were issued together from the original origin.</Prose>
    <Prose>Both sequences happen to have MAE {fixed(recursionScore.mae, 0)} on that continuation. Leakage does
      not have to produce a better score to be a protocol error. For the changed
      continuation {fixtures.recursionChanged.outcomes.map(asInput).join(', ')}, the original recursive forecast
      has MAE {fixed(recursionChangedScore.mae, 0)}, while the updated
      sequence {updatedChanged.predictions.map(asInput).join(', ')} has
      MAE {fixed(updatedChangedScore.mae, 1)}. The attractive number still belongs to the updated task. The
      first pair scored {fixed(recursionScore.mae, 0)} against {fixed(updatedScore.mae, 0)}; equal scores did
      not make the mislabelled claim any less wrong.</Prose>

    <ContinuationFigure />

    {/* ============================================================ §4 */}
    <H2 id={headingId(headings[3])}>{headings[3]}</H2>
    <Prose>A single chronological holdout tells us about one later period. <strong>Rolling-origin
      evaluation</strong> repeats the forecasting task from several historical issue dates. For each
      origin:</Prose>
    <ol>
      <li>Reconstruct eligible history.</li>
      <li>Fit the entire allowed procedure, if the declared schedule calls for refitting.</li>
      <li>Issue all requested horizons without seeing their future outcomes.</li>
      <li>Save origin, target date, prediction and model/procedure identifier.</li>
      <li>Score after the outcomes become available.</li>
    </ol>
    <Prose>An <strong>expanding window</strong> retains all eligible history. A <strong>sliding window</strong> retains
      only a recent eligible interval or number of rows. Expanding history supplies more examples and longer
      cycles; a sliding window can adapt faster after change but discards information and can increase
      estimation noise. Choose the window using development forecasts, not a story invented after the final
      errors are revealed.</Prose>

    <StaircaseFigure />

    <Prose>The claim the staircase makes is checkable, and it is checked rather than asserted. At issue
      origin {sampleFit.issueOrigin}, horizon {sampleFit.horizon}, the expanding fit
      uses {sampleFit.trainCount.toLocaleString('en-US')} training rows running from
      origin {sampleFit.trainStart} to {sampleFit.trainEnd}. Walking every observation index those rows read —
      each row's features, each row's label, and the prediction row's own features — the largest one
      is {sampleAudit.maxObservationIndex}, the issue origin itself,
      with {sampleAudit.violations.length} violations of the rule that no fitted window may contain an
      observation dated later than its own origin.</Prose>

    <Program example={timeSeriesExamples['eligible-rows']} />
    <Prose>Each printed line is one horizon's own boundary. The last training origin moves back by one day for
      every extra day of horizon, so the row count falls with it; the target moves forward. The final column is
      the same every time, and that is the point: whichever horizon is being fitted, the latest day any part of
      that fit reads is the issue origin. The program asserts it as it runs rather than printing a reassurance,
      so editing the origin produces a failure rather than a quietly wrong training range.</Prose>

    <Prose>Choosing the model family, window, features or regularization strength is still model selection. A
      useful arrangement is an earlier set of rolling origins for development, then a later sequence that
      evaluates the locked procedure. If an extensive search needs its own assessment, reproduce the selection
      inside earlier information boundaries, just as nested cross-validation protected outer rows in the earlier
      lesson.</Prose>
    <Prose>A final <strong>rolling</strong> assessment can allow outcomes from its early weeks to enter
      scheduled fits for later weeks after those outcomes arrive. This does not retroactively change an earlier
      forecast. What remains locked is the update/selection policy. By contrast, a frozen-model assessment holds
      one fitted model fixed. Say which one you ran; “test set” by itself does not settle the
      distinction.</Prose>
    <Prose>Randomly shuffling rows usually fails to recreate future-from-past deployment: it can train on later
      regimes and on targets that were unknown at the historical issue date. It is not a theorem that all
      timestamped data require one particular splitter. The contract specifies the use.
      The <a href="https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html">scikit-learn
      lagged-feature example</a> is an alternate Python walkthrough; its next-hour task should not be silently
      relabeled as a whole test-block forecast issued once.</Prose>

    <H3>Preserve the horizon in the score</H3>
    <Prose>For origins <Math>{'\\mathcal T'}</Math> and horizons <Math>{'1,\\ldots,H'}</Math>, calculate</Prose>
    <MathBlock>{'\\begin{gathered}\\operatorname{MAE}_h=\\frac1{|\\mathcal T|}\\sum_{t\\in\\mathcal T}|e_{t,h}|,'
      + '\\\\[4pt]\\operatorname{RMSE}_h=\\sqrt{\\frac1{|\\mathcal T|}\\sum_{t\\in\\mathcal T}e_{t,h}^2}.'
      + '\\end{gathered}'}</MathBlock>
    <Prose>MAE answers average absolute error in the target's units. RMSE gives large errors more influence and
      returns to the same units after the square root. Averaging MAE over equal-sized horizons equals pooling
      all their absolute errors. Averaging per-horizon RMSE generally differs from taking one square root after
      pooling squared errors.</Prose>
    <Prose>Longer horizons often lose information, but an empirical error curve need not increase at every step.
      Here every origin is a Saturday, so horizon and weekday are tied together. A difficult {weekdays[0]} and
      an easier {weekdays[3]} can produce a nonmonotone curve. To separate horizon difficulty from weekday
      effects, design origins that cover different weekdays or analyze both dimensions with enough
      data.</Prose>
    <Prose>If tomorrow matters more than next week, predeclare horizon weights and evaluate that decision. Do
      not switch weights after discovering where your favorite model performs best.</Prose>

    {/* ============================================================ §5 */}
    <H2 id={headingId(headings[4])}>{headings[4]}</H2>
    <Prose>The downloadable <a href={provenance.file}>daily CSV</a> contains {provenance.rows} complete calendar
      days from {timeSeriesData.series.firstDate} to {timeSeriesData.series.lastDate} in the UCI Bike Sharing
      dataset. It is the unchanged daily file, renamed for clarity, with attribution
      and {provenance.license} provenance supplied in <a href={provenance.attribution}>the data record</a>. The
      target is recorded system-wide rentals, not unmet demand, the number of bicycles required at a station, or
      a causal effect of weather.</Prose>
    <Prose>We use counts and deterministic calendar features. The source also contains casual and registered
      rentals; their sum is exactly the target, so using the target day's values would reveal the answer. We do
      not use future observed weather. The source lacks report-arrival timestamps, so this replay explicitly
      assumes counts are known at each day end.</Prose>

    <H3>Declare the comparison before fitting</H3>
    <ul>
      <li>Initial history: {timeSeriesData.series.firstDate} to {provenance.rows > 0
        ? timeSeriesData.series.dates[protocol.developmentOriginIndices[0]] : ''}.</li>
      <li>Issue forecasts every {weekdayName(timeSeriesData.series.dates[protocol.originIndices[0]])} from
        {' '}{timeSeriesData.series.dates[protocol.originIndices[0]]} through
        {' '}{timeSeriesData.series.dates[protocol.originIndices.at(-1)]}, for each of the next seven days.</li>
      <li>Development: first {protocol.developmentOriginIndices.length} origins,
        forecasting {protocol.developmentTargetRange[0]} to {protocol.developmentTargetRange[1]}.</li>
      <li>Final rolling assessment: final {protocol.finalOriginIndices.length} origins,
        forecasting {protocol.finalTargetRange[0]} to {protocol.finalTargetRange[1]}.</li>
      <li>Omit {protocol.unusedTailDates.join(' and ')} from assessment because they do not form another
        complete seven-day block in this file. They are retained in the source.</li>
      <li>Compare mean, naive, seven-day seasonal naive, drift, direct ridge with expanding history, and direct
        ridge with the most recent {protocol.ridgeWindow} eligible training origins.</li>
      <li>Select by development MAE pooled across origins and horizons. Keep the selected procedure and the
        predeclared naive/seasonal baselines for final assessment; do not use final results to choose among all
        six candidates.</li>
    </ul>
    <Prose>For each horizon <Math>{'h'}</Math>, direct ridge fits a separate model to eligible origin
      rows <Math>{'s\\le t-h'}</Math>. Its fourteen inputs are three historical
      counts <Math>{'y_s,y_{s-1},y_{s-6}'}</Math>, the past-seven-day mean, seven target-weekday indicators,
      sine and cosine of target day-of-year, and elapsed calendar time in years. The cyclic year features use a
      disclosed 365.25-day approximation; they are calendar encodings, not learned seasonality
      guarantees.</Prose>
    <Prose>Every horizon fit learns its scaler from its own eligible training rows and uses ridge
      alpha {asInput(protocol.ridgeAlpha)} with an intercept. Counts are nonnegative, so clipping negative ridge
      or drift predictions to zero is part of the predeclared rule. The {protocol.ridgeWindow}-row option means
      the last {protocol.ridgeWindow} eligible origin rows for each horizon, not the
      last {protocol.ridgeWindow} target days regardless of availability.</Prose>

    <H3>Read the measured results</H3>
    <LessonTable caption="Development results, pooled across every origin and horizon"
      headers={['Candidate', 'Development MAE, rentals/day', 'Development RMSE, rentals/day']}
      rows={timeSeriesData.development.map(row => [
        METHOD_LABELS[row.key],
        row.key === selected ? <strong>{fixed(row.mae, 2)}</strong> : fixed(row.mae, 2),
        fixed(row.rmse, 2),
      ])} />
    <Prose>The declared selection chooses {METHOD_LABELS[selected].toLowerCase()}. Refit that fixed procedure at
      each later origin using then-available observations. Its final MAE is {fixed(final[selected].mae, 2)};
      naive gives {fixed(final.naive.mae, 2)} and seasonal naive {fixed(final.seasonal.mae, 2)}. The selected
      model remains better on this aggregate, but its error increases markedly compared with the earlier
      development period.</Prose>
    <Prose>This is evidence about one two-year system and this update policy. The development-to-final
      difference can reflect changing conditions, different dates, selection and sampling variation; it does not
      by itself identify one cause. No repeated independent city sample or universally transferable improvement
      percentage has been measured.</Prose>

    <LessonTable caption="Final horizon-by-horizon MAE, rentals per day"
      headers={['Final horizon', 'Expanding ridge MAE', 'Naive MAE', 'Seasonal naive MAE']}
      rows={timeSeriesData.final[0].maeByHorizon.map((_unused, index) => {
        const values = timeSeriesData.final.map(entry => entry.maeByHorizon[index]);
        const best = values.reduce((lowest, value) => (value < lowest ? value : lowest), values[0]);
        /* EVERY method attaining the minimum is emphasised, never the first one
           found. At horizon 7 two of them tie exactly, and a table that bolded
           one would be hiding the mechanism this section is about. */
        return [
          String(index + 1),
          ...timeSeriesData.final.map(entry => (entry.maeByHorizon[index] === best
            ? <strong>{fixed(entry.maeByHorizon[index], 2)}</strong>
            : fixed(entry.maeByHorizon[index], 2))),
        ];
      })} />

    <Prose>The aggregate winner does not win every horizon. At horizon seven, naive and seven-day seasonal naive
      are <strong>identical rules</strong>: both use the latest {weekdays[6]} count. A seven-day season at
      horizon seven selects history position {h7.donor + 1} of {h7.lastIndex + 1} at the first final
      origin, which is that origin's own count — exactly what the naive rule repeats. Their equal errors
      of {fixed(final.naive.maeByHorizon[6], 4)} are required by the mechanism, not a coincidental numerical
      tie. The table also shows why drawing a smoothly increasing “forecast difficulty” curve would misrepresent
      these measurements.</Prose>

    <ComparisonFigure />

    <ForecastRequestLab />

    <Prose>Two real weeks make the point on their own. At {requestCases[0].date}, the
      period-{requestCases[0].period} MAE is {fixed(requestCases[0].seasonal, 2)} against
      naive {fixed(requestCases[0].naive, 2)}; at {requestCases[1].date} it
      is {fixed(requestCases[1].seasonal, 2)} against {fixed(requestCases[1].naive, 2)}. A plausible weekly rule
      wins one week and loses another. Inspect the exact historical days it copied rather than guessing a
      narrative from the score.</Prose>
    <Prose>The investigation also allows a clearly marked counterfactual edit to an observed historical count or
      a future outcome. A future edit can change the subsequently measured error, but must not change the
      already issued forecast. These edited results are constructed explorations on a real series, not the
      published measured experiment.</Prose>

    <H3>Run the complete experiment</H3>
    <Prose>Save <a href={provenance.program}>forecast-experiments.py</a> beside
      the <a href={provenance.file}>CSV</a> and run it with Python. It contains the full feature construction,
      six-candidate development comparison, locked final replay, exact toy calculations and figure data; there
      are no omitted training helpers or network downloads. It writes <Code>calculated-inputs.json</Code>,
      including every origin's actual values and predictions.</Prose>
    <CodeBlock language="text">{'python -m pip install numpy==2.3.5 pandas==3.0.1 scikit-learn==1.9.1\n'
      + 'python forecast-experiments.py'}</CodeBlock>
    <Prose>The author calculation used Python {timeSeriesData.versions.python} and these library versions, with
      serial deterministic SVD ridge fits. It performed {protocol.developmentRidgeFits} small development fits
      and {protocol.finalRidgeFits} final fits. The following complete, independently runnable baseline program
      exposes the essential forecast loop without hiding it behind model code. Save it beside the same
      CSV.</Prose>

    <Program example={timeSeriesExamples['baseline-loop']} />
    <Prose>The pooled values are development naive {fixed(development.naive.mae, 2)} and
      seasonal {fixed(development.seasonal.mae, 2)}, then final naive {fixed(final.naive.mae, 2)} and
      seasonal {fixed(final.seasonal.mae, 2)}. The code uses each origin's preceding week, not one repeated
      prediction from the beginning of the year. Final-week outcomes can enter a later scheduled forecast only
      after becoming history. Notice the last entry of each printed horizon profile: the two rules agree
      exactly, in development as well as in the final period, for the reason given above.</Prose>

    {/* ============================================================ §6 */}
    <H2 id={headingId(headings[5])}>{headings[5]}</H2>
    <Prose>Try each question before opening its hint or solution.</Prose>

    <Practice title="1. A reporting delay changes the training set"
      question="Offered training origins are days 4–10. At cutoff day 10, labels predict three days ahead and arrive one day after their target day. Which rows have known labels? What changes at cutoff 12?"
      hint="Test the label arrival inequality for each row, not just whether its feature date precedes the cutoff.">
      <Prose>Require <Math>{'s+3+1\\le10'}</Math>, so origins {practiceOneAtTen.join(', ')} qualify. At cutoff
        12, origins {practiceOneAtTwelve[0]}–{practiceOneAtTwelve.at(-1)} qualify. This assumes all features for
        those rows are also available; label eligibility alone does not certify an arbitrary feature
        pipeline.</Prose>
    </Practice>

    <Practice title="2. Repeat the last cycle beyond one season"
      question="History is 3, 9, 6, 4, 10, 8, with period 3. Produce five seasonal-naive forecasts. Which historical value supplies horizon five? Change only that value by adding 2."
      hint="Repeat the final three values in their existing order.">
      <Prose>Forecasts are {practiceCycle.seasonal.map(asInput).join(', ')}. Horizon five copies
        the {practiceDonor + 1}th history value, {asInput(practiceCycle.seasonal[4])}. Changing it
        to {asInput(practiceCycleChanged.seasonal[4])}
        produces {practiceCycleChanged.seasonal.map(asInput).join(', ')}. It does not change horizon
        three, which stays at {asInput(practiceCycle.seasonal[2])}.</Prose>
    </Practice>

    <Practice title="3. Identify the illegal arrow"
      question="A model predicts the next three days from Sunday evening. Its Monday prediction uses Sunday observations, its Tuesday prediction uses Monday's actual count, and its Wednesday prediction uses Tuesday's actual count. The report calls all three “Sunday's three-day forecast.” Repair either the procedure or the claim."
      hint="The same numerical predictions can be legitimate under a different issue schedule.">
      <Prose>For a genuine fixed-origin recursive forecast, feed Monday's prediction into the next step, then
        Tuesday's prediction, with only Sunday-available additional inputs. Or relabel and evaluate the
        procedure as an updated one-day forecast issued each evening. A direct three-model approach can also
        predict each horizon from Sunday-available features without recursion.</Prose>
    </Practice>

    <Practice title="4. Equal MAE can hide different large errors"
      question="Two procedures have four absolute errors: A 0, 0, 4, 4; B 2, 2, 2, 2. Calculate MAE and RMSE. Which metric exposes A's concentrated misses?"
      hint="Square before averaging for RMSE, then take a square root.">
      <Prose>Both have MAE {fixed(profileA.mae, 0)}. A has RMSE <Math>{'\\sqrt8\\approx'}</Math>
        {fixed(profileA.rmse, 3)}; B has RMSE {fixed(profileB.rmse, 0)}. RMSE gives A's two larger misses more
        weight. Whether that is the right preference depends on the cost of large misses; neither arithmetic
        result proves an application policy.</Prose>
    </Practice>

    <Practice title="5. Inspect the real protocol"
      question="The final table's seven-day MAE favors seasonal naive over expanding ridge. May we now claim that a hybrid chosen from that same final table has independently established final performance? What should happen next?"
      hint="Choosing a different model per horizon is itself selection.">
      <Prose>No. The table can motivate a candidate hybrid, but its reported final performance would be
        adaptively selected on that period. Specify the new per-horizon policy using development evidence, or
        treat the observed final results as development for a new experiment and obtain fresh later assessment.
        Keep the original locked-procedure result visible — here, final
        MAE {fixed(final[selected].mae, 2)} for {METHOD_LABELS[selected].toLowerCase()}.</Prose>
    </Practice>

    <Practice title="6. A time-series CSV is not necessarily a daily calendar"
      question="Rows are Monday, Tuesday, Thursday and Friday. A developer labels the preceding row's count “yesterday” for every target. Identify the failure and propose a repair. Should a missing Wednesday be filled with zero?"
      hint="Separate missing observation from observed absence of events.">
      <Prose>For Thursday, the previous row is Tuesday, two days earlier. Build the intended daily calendar and
        distinguish an absent report from a recorded zero. Preserve missingness or use a declared causal fill
        based on available history; do not interpolate from Thursday while pretending to forecast Wednesday. If
        the application truly operates on irregular events, use elapsed time and describe the lag as previous
        event instead.</Prose>
    </Practice>

    <Practice title="7. Overlapping labels and pooled horizons — deeper"
      question="A row at origin s predicts the sum of the next three days, reported immediately after day s + 3. At cutoff 20, what is the latest eligible training origin? If forecasts are issued daily, do adjacent target sums form independent error samples automatically?">
      <Prose>The latest eligible origin is {practiceSevenLatest}. The sums from
        origins {practiceSevenLatest} and {practiceSevenLatest + 1} share days {practiceSevenLatest + 2}
        {' '}and {practiceSevenLatest + 3}, so their targets overlap. Their forecast errors need not be
        independent; shared inputs, parameter fits and serial dynamics can add dependence too. A gap that
        prevents unknown labels from entering training does not establish independent assessment errors. Under
        the index-based splitter of section 3 the same requirement reads g ≥ {practiceSeven.minimumGap}.</Prose>
    </Practice>

    <Practice title="8. A scaled-error denominator — deeper"
      question="For training history 2, 4, 3, 5, calculate the nonseasonal naive training-error scale. A later forecast has absolute error 1.5. What is its scaled error? What if all training values were 4?">
      <Prose>The scale is <Math>{'(|4-2|+|3-4|+|5-3|)/3=5/3'}</Math>, which
        is {fixed(practiceScale.scale, 6)}. The scaled error
        is <Math>{'1.5/(5/3)='}</Math>{fixed(practiceScaled, 1)}. For constant training history the scale
        is {fixed(practiceConstantScale.rawScale, 0)}, so this ratio is undefined; silently adding a tiny
        denominator creates a different metric. Name a suitable alternative and retain the original-unit
        errors.</Prose>
    </Practice>

    <Prose><strong>Core readiness:</strong> given a new request, you can name the issue time and horizon,
      reconstruct legal features and labels, draw the evaluation/refit schedule, compute an informative
      baseline, and report errors without losing the horizon or changing the selection boundary.</Prose>

    {/* ============================================================ §7 */}
    <H2 id={headingId(headings[6])}>{headings[6]}</H2>
    <Prose>This section is a deeper branch. It answers questions the core route raises but does not
      need.</Prose>

    <H3>Direct, recursive and joint multi-step models</H3>
    <Prose>Our real experiment uses <strong>direct</strong> prediction: horizon <Math>{'h'}</Math> has its own
      fitted map from origin features to <Math>{'y_{t+h}'}</Math>. This avoids feeding forecast errors into
      later inputs, but requires several fits and uses fewer eligible recent training origins at longer
      horizons.</Prose>
    <Prose>A <strong>recursive</strong> method fits a one-step rule and repeatedly feeds its own predictions
      back. It can share one model across horizons, but its inputs at long horizons differ from the
      observed-history inputs seen during ordinary training. Approximation errors can feed forward. This is a
      mechanism, not a theorem that recursive prediction must lose to direct prediction on every
      dataset.</Prose>
    <Prose>A <strong>joint multi-output</strong> model predicts a vector of future values. It can share
      information across horizons and can support path-level objectives. Labels for a complete
      length-<Math>{'H'}</Math> vector are available only when the whole vector has matured, unless training
      explicitly handles partial labels. This is another reason not to apply one origin cutoff formula blindly
      to every architecture.</Prose>
    <Prose>The same distinctions matter in a controller planning battery use for the next hour. A forecast that
      will be revised every minute serves a different decision than a schedule that must be committed for the
      full hour. Evaluate the forecast-and-update policy the controller will actually use.</Prose>

    <H3>Overlap is not one problem with one remedy</H3>
    <Prose>Three mechanisms can coexist:</Prose>
    <ol>
      <li><strong>Unknown labels entering training:</strong> enforce each label's actual maturity/arrival
        cutoff.</li>
      <li><strong>Shared information between prediction tasks:</strong> decide whether that reuse is legitimate
        for the intended known-system or unseen-system deployment.</li>
      <li><strong>Dependent assessment errors:</strong> account for serial dependence, overlapping target
        intervals and shared fits when making uncertainty claims.</li>
    </ol>
    <Prose>If every day issues forecasts for horizons 1–7, a particular outcome appears as several distinct
      forecast tasks. Averaging those errors evaluates an origin–horizon distribution. It does not produce seven
      new independent outcomes. A joint weekly staffing decision may instead care about total-week error or the
      probability that capacity is exceeded on any day.</Prose>
    <Prose>Our weekly nonoverlapping target blocks avoid repeatedly scoring the same target day, but adjacent
      weeks can still be dependent. A standard error calculated as sample standard deviation divided by the
      square root of {final[selected].forecasts} treats those errors as independent; that assumption is not
      established here. Dependence-aware inference may use a justified block-resampling or time-series variance
      model, with block length and stationarity conditions examined. More bootstrap replicates cannot repair a
      wrong independence model.</Prose>

    <H3>From point forecasts to calibrated uncertainty</H3>
    <Prose>A point forecast is a single summary. An interval or distribution adds a claim about possible
      outcomes. A band containing roughly 90% of individual days is not automatically a band containing an
      entire seven-day path 90% of the time. For seven independent events each covered with probability 0.9,
      simultaneous coverage would be <Math>{'0.9^7\\approx0.4783'}</Math>; real dependence changes this
      calculation.</Prose>
    <Prose>Residual checks can reveal patterns the model left behind: persistent positive errors suggest
      systematic underprediction on those assessed cases; repeating weekday errors suggest a missing cycle.
      White-looking residuals do not prove future accuracy or correct interval coverage. Residuals from the
      fitted history and genuine horizon-<Math>{'h'}</Math> forecast errors answer different questions.</Prose>
    <Prose>Exchangeability-based conformal guarantees do not become time-series guarantees by changing the
      x-axis to dates.
      The <a href="/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml">Calibration
      &amp; Conformal Prediction</a> topic explains its assumptions; temporal adaptation needs a stated method
      and conditions. The <a href="https://otexts.com/fpp3/distaccuracy.html">FPP3 distributional accuracy
      section</a> is a deeper route to evaluating probabilistic forecasts, not evidence that the point-only
      bicycle experiment produced calibrated intervals.</Prose>

    <H3>Scaling, transformations and changing environments</H3>
    <Prose>MAE in rental counts cannot directly compare a small station with a whole city on equal footing. Mean
      absolute scaled error divides forecast absolute errors by a declared training-only naive-error scale. With
      seasonal period <Math>{'m'}</Math>, that scale is</Prose>
    <MathBlock>{'q_T=\\frac1{T-m}\\sum_{i=m+1}^{T}|y_i-y_{i-m}|.'}</MathBlock>
    <Prose>Then average <Math>{'|e|/q_T'}</Math>, stating whether each rolling origin uses its own training
      scale or one fixed reference scale. A score below one compares with that <strong>training</strong> naive-error
      scale; it does not logically prove a win over the naive forecasts on the later assessment period. Zero
      scale makes the ratio undefined. MAPE has its own problem at zero or near-zero outcomes.
      The <a href="https://otexts.com/fpp3/accuracy.html">forecast-accuracy chapter</a> develops these metric
      choices and their units.</Prose>
    <Prose>A log transformation can stabilize large level-dependent variation, but transforming the mean is not
      the same as averaging transformed predictions back. If <Math>{'\\log Y'}</Math> is modeled as normal with
      mean <Math>{'\\mu'}</Math> and variance <Math>{'\\sigma^2'}</Math>, <Math>{'\\exp(\\mu)'}</Math> is the
      median of that positive lognormal model and its mean
      is <Math>{'\\exp(\\mu+\\sigma^2/2)'}</Math>. The distributional assumption is essential. A simple
      exponential back-transform does not universally give an unbiased mean forecast.</Prose>
    <Prose>Decomposing a series into trend, seasonality and remainder can make patterns easier to reason about.
      A two-sided smoother fitted using the full series, however, cannot supply a historically available feature
      at an earlier origin. Fit the allowed decomposition using only that origin's information, or label the
      full-series decomposition as retrospective analysis. Further ARIMA, GARCH, exponential-smoothing and
      decomposition modeling belongs to the later time-series specialization; this lesson establishes the
      evaluation contract those methods must obey.</Prose>
    <Prose>Drift in the observed error can motivate a shorter window, a new feature or retraining, but
      monitoring and changing the policy create a new adaptive procedure. Record the trigger and evaluate the
      whole response policy. A sudden low rental count could reflect weather, service availability, reporting or
      actual use; this file alone does not identify the cause.</Prose>

    {/* ============================================================ next */}
    <H2 id={headingId(headings[7])}>{headings[7]}</H2>
    <Prose>The next lesson in this
      module, <a href="/learn/path/full-curriculum/end-to-end-supervised-learning-error-analysis?module=classical-ml">End-to-End
      Supervised Learning &amp; Error Analysis</a>, brings task formulation, baseline comparison, disciplined
      selection and error analysis into one complete experiment. Its independent-row example uses a different
      split because its prediction task is different; the information-boundary principle remains the
      same.</Prose>

    <Callout title="Where this page's numbers come from">
      <Prose>The dataset served here is <a href={provenance.file}>bike-sharing-daily.csv</a>,
        {' '}{provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, with
        its <a href={provenance.attribution}>attribution and full provenance</a>. It is the
        unchanged {provenance.rows}-row daily member of the UCI Bike Sharing dataset, retrieved
        {' '}{provenance.retrieved}, and this lesson serves its own copy. The
        complete <a href={provenance.program}>author program</a> is served beside it; a fresh run reproduces its
        recorded results file exactly.</Prose>
      <Prose>Every measured number above comes from that run. Every constructed number is recomputed in the
        browser from the same definitions, and the three displayed programs are byte-exact slices of frozen
        sources — two functions lifted from the author program by their <Code>def</Code> lines, and one block
        lifted whole out of this lesson's own manuscript — each pinned by SHA-256, with only import lines and a
        few printing lines composed around them.</Prose>
    </Callout>

    <Sources alternatives={<>
      <p>For a free, complete textbook route, Hyndman and Athanasopoulos'
        {' '}<a href="https://otexts.com/fpp3/">Forecasting: Principles and Practice</a> chapter 5 covers this
        ground with R and fable examples.
        Its <a href="https://otexts.com/fpp3/simple-methods.html">simple forecasting methods</a> section gives
        the mathematical baseline definitions and contrasting examples — read it after our section 2 to check
        which historical observation each rule copies — and
        its <a href="https://otexts.com/fpp3/tscv.html">time-series cross-validation</a> section supplies
        rolling-origin diagrams and executable code. Translate the issue schedule before borrowing that code for
        a different horizon: it includes one-step and multi-step examples; our request issues seven daily
        forecasts each Saturday.</p>
      <p>For a current Python walkthrough,
        scikit-learn's <a href="https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html">lagged
        features example</a> builds lag columns, evaluates forward rather than randomly, and adds quantile
        prediction. Its hourly next-step task differs from this lesson's weekly direct forecasts, so read its
        evaluation section with our section 4 open beside it.</p>
    </>}>
      <li><a href="https://otexts.com/fpp3/accuracy.html">FPP3, evaluating point forecast accuracy</a> — actual
        error, scale-dependent, percentage and scaled definitions, and the training-denominator distinction
        behind section 7's scaled error.</li>
      <li><a href="https://otexts.com/fpp3/prediction-intervals.html">FPP3, prediction intervals</a>
        {' '}and <a href="https://otexts.com/fpp3/distaccuracy.html">distributional forecast accuracy</a> —
        optional deeper reading on claims a point forecast cannot make. These are additional learning resources,
        not a claim that intervals were fitted in the bicycle experiment.</li>
      <li><a href="https://otexts.com/fpp3/ftransformations.html">FPP3, forecasting with transformations</a> —
        the reverse-transform and bias-adjustment sections behind section 7's lognormal identity.</li>
      <li><a href={provenance.record}>UCI Bike Sharing</a>, {provenance.creator}, DOI
        {' '}<a href={provenance.doi}>{provenance.doi.replace('https://doi.org/', '')}</a>,
        {' '}<a href={provenance.licenseUrl}>{provenance.license}</a> — original data, variable definitions,
        citation and license. The daily member and complete provider description are retained with this
        lesson.</li>
    </Sources>
  </div>,
};

export default timeSeriesValidationContent;
