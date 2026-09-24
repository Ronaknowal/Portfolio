import { Callout, H2, H3, Prose, Code } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import {
  CandidateComparisonFigure, FitAndGeneralisationFigure, InformationLaneFigure, PairedChangeFigure,
  RecallComparisonFigure, StudyHeldOutOutput, StudyProgram,
} from '../../components/lesson-labs/EndToEndFigures.jsx';
import {
  AcceptanceCostLab, DevelopmentErrorLab, FreezeAndReportLab,
} from '../../components/lesson-labs/EndToEndLabs.jsx';
import {
  Count, HeldOutOnly, HeldOutProvider, Score, Table, fixed,
} from '../../components/lesson-labs/EndToEndShared.jsx';
import { endToEndData } from '../endtoend-data.js';
import {
  acceptanceLedger, asSelectionCriterion, candidateByKey, classRecallTable, natsFor, pairedChange,
  scoreRecord, sliceComparison, wilsonInterval,
} from '../endtoend-models.js';

/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.sqrt` here would resolve to that component and yield
   undefined rather than a number. Every quantity on this page therefore comes
   from `endtoend-models.js` or from the recorded data, never from arithmetic
   typed into the prose.

   The same discipline applies to score roles. No number reaches the reader
   except through `<Score>` or `<Count>`, both of which refuse a record without
   one of the four roles, so a quantity cannot arrive on the page unlabelled.
   That is the mistake this lesson teaches a learner to catch, and it would be
   the easiest one for the lesson itself to make. */

const linearTwo = candidateByKey.linear_two;
const linearThree = candidateByKey.linear_three;
const forestTwo = candidateByKey.forest_two;
const majority = candidateByKey.majority;
const change = pairedChange('linear_two', 'linear_three');
const forestChange = pairedChange('linear_two', 'forest_two');
const lowerSlice = sliceComparison({
  reference: 'linear_two', candidate: 'linear_three', cutoffHundredths: 400, side: 'lower',
});
const upperSlice = sliceComparison({
  reference: 'linear_two', candidate: 'linear_three', cutoffHundredths: 400, side: 'upper',
});
const classOneSlice = endToEndData.colourSliceReference.slices;
const twoRecalls = classRecallTable(linearTwo.validationConfusion);
const threeRecalls = classRecallTable(linearThree.validationConfusion);
const heldOut = endToEndData.heldOut;
const wilson = wilsonInterval(heldOut.wilson.successes, heldOut.wilson.trials, heldOut.wilson.z);
/* A Wilson bound is computed from the held-out accuracy, so it is held-out
   evidence and is badged as such. `scoreRecord` refuses any other role. */
const wilsonBound = value => scoreRecord('Wilson bound on accuracy', value, 'held-out');
const looseRule = acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2 });
const strictRule = acceptanceLedger({ thresholdHundredths: 80, wrongCost: 10, deferCost: 2 });
const cheapErrorLoose = acceptanceLedger({ thresholdHundredths: 60, wrongCost: 2, deferCost: 2 });
const cheapErrorStrict = acceptanceLedger({ thresholdHundredths: 80, wrongCost: 2, deferCost: 2 });
/* Derived, not typed: which cultivar went backwards is whatever the recalls say
   went backwards, and asserting it in a literal would make the claim
   unfalsifiable if the data ever changed. */
const worsened = twoRecalls
  .filter((row, index) => threeRecalls[index].recall < row.recall)
  .map(row => row.actual);

const headings = [
  '1. Write the question before choosing the model',
  '2. Keep three different kinds of learning separate',
  '3. Choose a baseline that answers a specific doubt',
  '4. Run a complete offline study',
  '5. Turn errors into testable hypotheses',
  '6. Freeze a decision and write a report someone else can reproduce',
  '7. Close the loop without erasing what you learned',
  '8. Deeper: how strong is the evidence?',
  '9. Practice: make and defend a decision',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

/* `constructedFixture` marks a practice whose numbers come from an invented
   scenario rather than from this study. Badging practice 1's balanced accuracy
   "validation" would claim it was measured on the 36 wine development rows,
   which is the same mislabelling the acceptance queue is declared out of. The
   practices that quote this study's real quantities — 4 and 6 — are deliberately
   NOT exempted, so their badges stay under the sweep. */
function Practice({ title, question, hint, children, constructedFixture }) {
  return <section className="ete-practice" data-constructed-fixture={constructedFixture}>
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>Show the explained solution</summary>{children}</details>
  </section>;
}

const endToEndContent = {
  title: 'End-to-End Supervised Learning & Error Analysis',
  readTime: '~50 min first pass · ~90 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <HeldOutProvider><div className="lesson-pilot endtoend-lesson">
    <LessonIntro
      prerequisites={<>A fitted model, a train/test split and the ordinary classification measures. Accuracy is
        the share of specimens predicted correctly; recall for one class is the share of that class predicted
        correctly. Both are refreshed in section 3 before anything depends on them. The
        preceding <a href="/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml">Time-Series
        Validation &amp; Forecasting Baselines</a> lesson supplies the discipline of choosing a split for a
        reason; here the unit changes from a time origin to a measured specimen, and the reason changes with
        it.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A classifier is useful only as part of a chain: someone asks a question, measurements arrive, a fitted
      transformation turns them into inputs, a model makes a prediction, and someone acts on the result. A good
      score in a notebook does not tell you whether that chain answers the original question. Here we build one
      small, complete study on {endToEndData.provenance.rows} real wine specimens: can we identify a specimen&apos;s
      recorded cultivar from its chemical measurements? The interesting result is that an additional
      measurement helps overall — {change.repaired.length} development specimens repaired
      against {change.broken.length} newly wrong — while making cultivar {worsened.join(' and ')} worse. That
      is the kind of tradeoff an average hides. Every score on this page is labelled with what it is evidence
      about, and the held-out result is not in this document until you have frozen the decision it reports on.
    </LessonIntro>

    <div className="ete-route"><Prose><strong>First pass.</strong> Read sections 1–7, run the offline study in
      section 4, and do practice 1–4. Follow figure A where it appears and work investigation B before reading
      the paragraph after it. Section 8 is a deeper branch on uncertainty, selective prediction and iteration
      design; return to it with practice 5–6. Allow about 45–60 minutes for the core and another 60–90 to
      reproduce and extend the study.</Prose></div>

    <Callout title="Four kinds of number appear on this page, and they are never mixed">
      A <strong>training</strong> score is computed on the {endToEndData.contract.trainRows} rows the model was
      fitted on; it describes the fit. A <strong>validation</strong> score is computed on
      the {endToEndData.contract.validationRows} development rows; it is the evidence this study may inspect
      and choose with. A <strong>selection</strong> quantity is a validation score in its role as the declared
      choice criterion — the same number, a different claim. A <strong>held-out</strong> score is computed once
      on the {endToEndData.contract.testRows} test rows, after the candidate was frozen. Every printed score
      carries its role beside it, so you never have to reconstruct which one you are reading.
    </Callout>

    {/* ============================================================ section 1 */}
    <H2 id={headingId(headings[0])}>{headings[0]}</H2>

    <Prose>The preceding lesson treated a prediction as an origin and a future horizon. Here the unit changes:
      one row is one wine specimen. The dataset&apos;s row order groups cultivars; it is not an observation
      timestamp. Copying a time split merely because a table has an order would change the experiment for the
      wrong reason.</Prose>

    <Prose>Our data are the UCI Wine collection — chemical analyses
      of {endToEndData.provenance.rows} wines from three cultivars grown in the same Italian region, with
      {' '}{endToEndData.provenance.measurementColumns} recorded measurement columns. The lesson supplies all
      rows in <a href={endToEndData.provenance.file} download>wine.csv</a>, with an added specimen ID and
      cultivar codes 0, 1 and 2 holding {endToEndData.provenance.classCounts.join(', ')} specimens. These codes
      are names, not an ordering of wine quality. Its
      identity, licence and every change made to produce this file are
      in <a href={endToEndData.provenance.attribution}>the attribution note</a> served beside
      it; <a href={endToEndData.provenance.doi}>the dataset record</a> is the canonical source.</Prose>

    <Prose>Write an <strong>experiment contract</strong>: the short statement of what information is available,
      what prediction is wanted, and what evidence would justify a choice.</Prose>

    <Table caption="The experiment contract for this study"
      headings={['Contract item', 'This study']}
      rows={[
        ['Prediction unit', endToEndData.contract.predictionUnit],
        ['Target', endToEndData.contract.target],
        ['Initial inputs', endToEndData.contract.initialInputs.join(' and ')],
        ['Candidate added input',
          `${endToEndData.contract.candidateAddedInput}, measured before classification`],
        ['Primary selection metric',
          'Validation balanced accuracy: the mean recall across the three cultivars'],
        ['Supporting diagnostics', 'Overall accuracy, log loss, confusion matrix and development slices'],
        ['Training rows', `${endToEndData.contract.trainRows} specimens, used to fit coefficients and preprocessing`],
        ['Validation rows',
          `${endToEndData.contract.validationRows} specimens, used for the declared comparison and error analysis`],
        ['Test rows', `${endToEndData.contract.testRows} specimens, used after the candidate is frozen`],
        ['Final model protocol', endToEndData.contract.finalProtocol],
      ]}
      footnote={'The first two features are a teaching constraint, chosen to make a measurement-versus-model '
        + 'comparison inspectable. We have no assay-cost data, so we do not claim they are the cheapest '
        + 'measurements. The three-feature candidate asks whether more informative input can help more than '
        + 'making a decision rule more flexible.'} />

    <Prose><strong>What this dataset can support.</strong> This is a retrospective classification exercise
      within a small historical collection. The file does not give enough sampling, winery, repeated-specimen
      or time metadata to establish independent future-vintage performance. A deployment study would need those
      fields and a split matched to its users and future measurements. The random stratified split below
      preserves class representation for this limited exercise; it does not create missing population
      evidence.</Prose>

    {/* ============================================================ section 2 */}
    <H2 id={headingId(headings[1])}>{headings[1]}</H2>

    <Prose>Training changes the model&apos;s fitted state. Validation changes the researcher&apos;s choices.
      Testing estimates the performance of a choice already made. Information can move through a human as
      easily as through a function call.</Prose>

    <Prose>An example makes the distinction concrete. Suppose we calculate the mean alcohol concentration using
      all {endToEndData.provenance.rows} rows, scale everything, and then separate training and test sets. The
      test specimens have already influenced the coordinate system the model sees. The proper sequence is to
      choose the rows first, calculate the training mean and standard deviation, and apply those same values to
      the other rows. <Code>Pipeline</Code> keeps those fitted operations attached to the model.</Prose>

    <Prose>For a feature <Math>{'x'}</Math>, training
      estimates <Math>{'\\mu_{\\mathrm{train}}'}</Math> and <Math>{'s_{\\mathrm{train}}'}</Math>. Every later
      specimen is represented as</Prose>

    <MathBlock>{'z=\\frac{x-\\mu_{\\mathrm{train}}}{s_{\\mathrm{train}}}.'}</MathBlock>

    <Prose>The transformation of a validation specimen uses its own measured <Math>{'x'}</Math>, but not a
      newly fitted validation mean. Learning a coordinate system and applying that coordinate system are
      different operations.</Prose>

    <InformationLaneFigure />

    <Prose>This picture also explains why hiding the test target from <Code>fit</Code> is insufficient. If a
      researcher reads test errors, adds a feature to repair them, and reports the improved score on those same
      rows, the test has become development data. Continuing to work is reasonable; calling that revised score
      an untouched final estimate is the mistake. Use a new holdout or a nested outer evaluation for the next
      confirmatory comparison.</Prose>

    <Prose>Before training, check the table at the level that affects this contract: row identity, label
      meaning, feature units, duplicates or repeated entities, missing values, impossible values, and which
      fields exist when predictions will be made. In our supplied file
      all {endToEndData.provenance.measurementColumns} measurements are finite and the ID is excluded from
      features. The ID preserves traceability; it carries no chemical meaning.</Prose>

    {/* ============================================================ section 3 */}
    <H2 id={headingId(headings[2])}>{headings[2]}</H2>

    <Prose>A baseline is a comparison with a job. Each of our four candidates answers a different doubt, and
      all four are declared together before the comparison.</Prose>

    <Table caption="The four candidates, and the doubt each one is there to answer"
      headings={['Candidate', 'What it is', 'Inputs', 'Declared settings', 'The doubt it answers']}
      rows={[majority, linearTwo, forestTwo, linearThree].map(candidate => [
        candidate.short, candidate.family, candidate.features.join(' + '), candidate.settings,
        candidate.purpose,
      ])}
      footnote={'Only the last three are eligible for selection. The majority baseline is there to make "does '
        + 'measuring anything help" a question with a visible answer, not to win.'} />

    <Prose>Do not change features, data splits and model settings all at once and then attribute a difference
      to one of them. Our two interpretable comparisons hold everything else fixed:</Prose>

    <Table caption="Two comparisons, each changing one thing"
      headings={['Comparison', 'Changed', 'Held fixed']}
      rows={[
        ['Two-feature linear → two-feature forest', 'Function class and declared fitting settings',
          'Specimens, measurements, primary metric'],
        ['Two-feature linear → three-feature linear', 'One measured feature',
          'Specimens, classifier family, C=1, scaling rule'],
      ]} />

    <Prose>For logistic regression, each class has a score <Math>{'a_k=w_k^Tz+b_k'}</Math>. The softmax
      transformation converts these three scores into positive numbers summing to one:</Prose>

    <MathBlock>{'p_k=\\frac{e^{a_k}}{\\sum_{j=0}^{2}e^{a_j}}.'}</MathBlock>

    <Prose>The predicted cultivar is the class with greatest <Math>{'p_k'}</Math>. The fitting objective
      rewards probability assigned to the observed class, with regularization to restrain coefficients. A
      forest instead pools class probabilities from many trees. Its <Code>max_depth=4</Code> and
      {' '}<Code>min_samples_leaf=3</Code> are declared controls on this comparison, not settings discovered by
      searching this test set.</Prose>

    <Callout title="Refresh the measures before anything depends on them">
      If a cultivar has {twoRecalls[0].support} validation specimens and {twoRecalls[0].correct} are correctly
      classified, its recall is {twoRecalls[0].correct}/{twoRecalls[0].support} = {fixed(twoRecalls[0].recall, 2)}.
      <strong> Balanced accuracy</strong> averages the recalls of the three cultivars, so a cultivar with fewer
      rows still contributes one third. <strong>Overall accuracy</strong> gives each specimen equal weight.
      <strong> Log loss</strong> averages <Math>{'-\\log p_{\\text{actual}}'}</Math>, in natural-log units
      called nats: assigning 0.8 to the true class costs about {fixed(natsFor(0.8), 3)} nats; assigning 0.2
      costs about {fixed(natsFor(0.2), 3)}. It distinguishes probabilities even when the winning class is
      unchanged. These quantities answer different questions; choose one primary measure before comparing
      candidates.
    </Callout>

    {/* ============================================================ section 4 */}
    <H2 id={headingId(headings[3])}>{headings[3]}</H2>

    <Prose>The program below exposes the research sequence. <Code>test</Code> is an index list until the final
      block. It selects the winning candidate by validation balanced accuracy; ties use the stated candidate
      order, and that deterministic tie rule is part of the protocol.</Prose>

    <StudyProgram />

    <CandidateComparisonFigure />

    <Prose>Now interpret the result rather than stopping at the printout. Both measured-input models improve
      greatly over the majority predictor, whose balanced accuracy
      of <Score record={majority.validationBalancedAccuracy} /> is what predicting the most frequent training
      cultivar every time is worth. The forest improves the two-feature linear model
      by {forestChange.net} validation {forestChange.net === 1 ? 'specimen' : 'specimens'}. Adding flavanoids
      repairs {change.repaired.length} previously wrong predictions and breaks {change.broken.length} previously
      correct one — a net gain of {change.net}, from <Count part={change.referenceCorrect} whole={change.total}
        role="validation" noun="correct" /> to <Count part={change.candidateCorrect} whole={change.total}
        role="validation" noun="correct" />. The improvement is not just a higher aggregate: we can name the
      changed cases and investigate them.</Prose>

    <PairedChangeFigure />

    <Prose>The two-feature forest moves the same aggregate by a different route: it
      repairs {forestChange.repaired.length} and breaks {forestChange.broken.length}. Equal-looking gains need
      not be made of the same movements, which is exactly why the identifiers are worth keeping aligned.</Prose>

    <FitAndGeneralisationFigure />

    {/* ============================================================ section 5 */}
    <H2 id={headingId(headings[4])}>{headings[4]}</H2>

    <Prose>An <strong>error slice</strong> is a defined subset of examples: one cultivar, a measurement range,
      one acquisition device, or a missingness pattern. Its purpose is to find a coherent failure that suggests
      an action. Always show the denominator. “Three errors” means something different among five cases and
      among five hundred.</Prose>

    <Table caption={`The validation confusion matrix for ${linearTwo.short}, rows actual and columns predicted`}
      headings={['Actual cultivar', 'Predicted 0', 'Predicted 1', 'Predicted 2', 'Support']}
      rows={linearTwo.validationConfusion.map((row, index) => [
        `Actual ${index}`, ...row.map(String), String(twoRecalls[index].support),
      ])}
      footnote={<>Its balanced accuracy is the mean of {twoRecalls.map(row =>
        `${row.correct}/${row.support}`).join(', ')}, which
        is <Score record={linearTwo.validationBalancedAccuracy} />. The confusion matrix and the score
        summarize the SAME predictions through different aggregations; you can reconstruct one from the
        other only if you retain enough counts, and a single average loses the pattern.</>} />

    <Prose>After adding flavanoids, every validation specimen of cultivars 1 and 2 is correct, but cultivar
      {' '}{worsened.join(' and ')} has {threeRecalls[worsened[0]].errors} errors rather
      than {twoRecalls[worsened[0]].errors}. The mean of recalls nevertheless improves
      to <Score record={linearThree.validationBalancedAccuracy} />. A model can improve balanced accuracy and
      still worsen one class.</Prose>

    <RecallComparisonFigure />

    <Prose>Whether that is acceptable depends on the contract, including the consequences for the affected
      cultivar. The colour-intensity split tells a related story, and it points the other way:</Prose>

    <Table caption="Errors inside two fixed colour-intensity slices of the development rows"
      headings={['Validation slice', 'Support', `${linearTwo.short} errors`, `${linearThree.short} errors`]}
      rows={[
        [`Colour intensity < ${endToEndData.colourSliceReference.cutoff}`, String(lowerSlice.slice.n),
          <Count key="l2" part={lowerSlice.referenceErrors} whole={lowerSlice.slice.n} role="validation" />,
          <Count key="l3" part={lowerSlice.candidateErrors} whole={lowerSlice.slice.n} role="validation" />],
        [`Colour intensity ≥ ${endToEndData.colourSliceReference.cutoff}`, String(upperSlice.slice.n),
          <Count key="u2" part={upperSlice.referenceErrors} whole={upperSlice.slice.n} role="validation" />,
          <Count key="u3" part={upperSlice.candidateErrors} whole={upperSlice.slice.n} role="validation" />],
      ]}
      footnote={'This threshold is an explicit exploratory diagnostic, not a known biological discontinuity. We '
        + 'keep it fixed when comparing the candidates. Searching hundreds of cutoffs until one looks dramatic '
        + 'would turn chance variation into a story. Treat this live view as development exploration; any claim selected after looking needs a fresh evaluation.'} />

    <DevelopmentErrorLab />

    <Prose>What could explain a persistent error? There are several distinguishable hypotheses, and they call
      for different next actions:</Prose>

    <Table caption="Reading an error pattern as a hypothesis with a next step"
      headings={['Observed pattern', 'Plausible hypothesis', 'Next discriminating action']}
      rows={[
        ['Both training and validation are poor',
          'Features or function class do not separate the target; optimization or data handling may also be wrong',
          'Inspect a tiny known case, verify transformation and target alignment, then compare a controlled model or feature change'],
        ['Training good, validation much worse', 'Generalization gap, split mismatch or dependence',
          'Check the unit and the split, examine training-size curves and regularization under a fixed validation protocol'],
        ['One acquisition group consistently worse', 'Measurement or population mismatch',
          'Inspect group metadata and raw records; acquire representative validation data'],
        ['Confident mistakes cluster around suspect labels',
          'Label convention or annotation issue, among other possibilities',
          'Review original records without replacing labels merely to agree with the model'],
        ['A new feature fixes many cases but breaks a coherent subgroup',
          'Useful information with a changed decision surface',
          'Quantify both directions and investigate the subgroup\'s decision costs'],
      ]}
      footnote={'These are hypotheses, not diagnoses made from one chart. A useful experiment changes a cause '
        + 'and specifies an expected observation. “Try a bigger model” is a choice; “test whether nonlinear '
        + 'boundaries in the same two measurements reduce cultivar 0 and 1 confusion” is a hypothesis.'} />

    {/* ============================================================ section 6 */}
    <H2 id={headingId(headings[5])}>{headings[5]}</H2>

    <Prose>Under our declared primary metric, the choice is made by reading the validation balanced accuracies
      and taking the best. That reading is what turns a validation measurement into a selection criterion. Make
      the call yourself before the page shows you what it cost.</Prose>

    <FreezeAndReportLab />

    <HeldOutOnly>
     <div className="ete-heldout-report">
      <Prose>The selected model&apos;s fitted scaler and coefficients remain exactly those learned on
        the {endToEndData.contract.trainRows} training rows. Opening
        the {endToEndData.contract.testRows}-row test set once gives the last four lines the study program
        printed:</Prose>
      <StudyHeldOutOutput />

      <Table caption={`The held-out confusion matrix for ${candidateByKey[heldOut.selected].short}, `
        + 'rows actual and columns predicted'}
        headings={['Actual cultivar', 'Predicted 0', 'Predicted 1', 'Predicted 2', 'Support']}
        rows={heldOut.confusion.map((row, index) => [
          `Actual ${index}`, ...row.map(String),
          String(row.reduce((sum, value) => sum + value, 0)),
        ])}
        footnote={'One cultivar-2 specimen is predicted as cultivar 1; everything else is correct.'} />

      <Prose>That is <Count part={heldOut.correct} whole={heldOut.total} role="held-out" noun="correct" />, an
        accuracy of <Score record={heldOut.accuracy} /> and a balanced accuracy
        of <Score record={heldOut.balancedAccuracy} />. The held-out estimate happens to exceed the validation
        estimate of <Score record={linearThree.validationBalancedAccuracy} />. A test set is not required to
        score worse: its cases are different and the estimate varies with sampling.</Prose>
     </div>
    </HeldOutOnly>

    <Prose>Would refitting on all {endToEndData.contract.trainRows + endToEndData.contract.validationRows}
      {' '}development rows be wrong? No, if specified before test evaluation. It would produce a different
      scaler and different coefficients, whose final test predictions need to be evaluated as that different
      model. This study keeps the training-only fitted model so that its development diagnostics and its final
      report refer to the same artifact. Do not silently switch between those protocols.</Prose>

    <HeldOutOnly placeholder={'The compact final report quotes the held-out result, so it opens with the rest '
      + 'of the report above.'}>
      <Callout title="A compact final report someone else could reproduce">
        We evaluated cultivar classification within the supplied {endToEndData.provenance.rows}-specimen Wine
        collection. A fixed stratified {endToEndData.contract.trainRows}/{endToEndData.contract.validationRows}/
        {endToEndData.contract.testRows} train/validation/test protocol compared a majority baseline,
        two-feature logistic regression, a bounded two-feature forest and three-feature logistic regression.
        Validation balanced accuracy selected the three-feature
        model (<Score record={asSelectionCriterion(linearThree.validationBalancedAccuracy)} />). Its added
        flavanoid input fixed {change.repaired.length} and broke {change.broken.length} validation predictions
        relative to the two-feature linear model; the low-colour slice worsened
        from {lowerSlice.referenceErrors}/{lowerSlice.slice.n} to {lowerSlice.candidateErrors}/{lowerSlice.slice.n}
        {' '}errors. The selected training-only fitted pipeline achieved test
        accuracy <Count part={heldOut.correct} whole={heldOut.total} role="held-out" /> and balanced
        accuracy <Score record={heldOut.balancedAccuracy} />. The source file, split IDs, software versions and
        settings accompany the report. Broader winery or future-vintage performance requires representative
        metadata and evaluation.
      </Callout>
    </HeldOutOnly>

    <Prose>This is a worked reporting example, not a suggested claim about wine production. Keep the
      implementation details that affect reproducibility beside the artifact: input schema, units, class
      encoding, fitted transformations, model settings, split IDs, seeds and versions — for this study, NumPy
      {' '}{endToEndData.versions.numpy}, scikit-learn {endToEndData.versions.sklearn} and Python
      {' '}{endToEndData.versions.python}. Model cards provide a useful framework for recording intended use,
      evaluation and limitations, and the source paper also emphasizes evaluation across relevant
      groups.</Prose>

    {/* ============================================================ section 7 */}
    <H2 id={headingId(headings[6])}>{headings[6]}</H2>

    <Prose>Maintain an experiment record with the hypothesis, the changed factor, the evidence already
      consumed, the result, the decision and the next action. You do not need a new framework for every run; a
      small table is enough.</Prose>

    <Table caption="This study's own experiment record"
      headings={['Experiment', 'Information used for the choice', 'Result', 'Decision']}
      rows={[
        ['Linear → forest, same two features', 'Training and validation',
          `${forestChange.repaired.length} validation error repaired, ${forestChange.broken.length} broken`,
          'Useful small gain; compare against the added measurement'],
        ['Add flavanoids, same linear family', 'Training and validation',
          `${change.repaired.length} repaired, ${change.broken.length} broken`,
          'Select under balanced accuracy; retain the subgroup regression in the report'],
        ['Final selected pipeline', 'Test, used for reporting',
          'Reported once, after the candidate was frozen',
          'Record the frozen result; use fresh evaluation for later tuning'],
      ]}
      footnote={'Reproducibility is the ability to regenerate a specified study, not proof that its split '
        + 'answers every future use. A fixed seed stabilizes one realization; it does not estimate how results '
        + 'vary across samples. Those are separate goals.'} />

    <Prose>For a real application, the next step after this study would include an inference contract: which
      named fields are required, how invalid measurements are handled, who receives uncertain results, and how
      inputs and eventual outcomes will be monitored. A dropped column or a unit change can break the chain
      even if the classifier&apos;s coefficients are untouched. Save and apply the whole fitted pipeline, and
      validate the incoming schema before prediction.</Prose>

    <Prose>The next module starts
      with <a href="/learn/path/full-curriculum/perceptrons-neurons-activation-functions?module=deep-learning-fundamentals">Perceptrons,
      Neurons &amp; Activation Functions</a>. It introduces a richer way to compose learned functions. The
      experimental discipline stays: begin with the task and baseline, examine the function&apos;s behavior,
      and demand evidence for an improvement. A neural model is an option to investigate, not the automatic
      winner after a classical model.</Prose>

    {/* ============================================================ section 8 */}
    <H2 id={headingId(headings[7])}>{headings[7]}</H2>

    <Prose><strong>This is a deeper branch.</strong> Return here after the core study. These extensions develop
      the same decision chain without changing the frozen result.</Prose>

    <H3>A small test is uncertain, even when the score looks impressive</H3>

    <Prose>For <Math>{'k'}</Math> successes among <Math>{'n'}</Math> independent Bernoulli trials, a Wilson
      interval for a proportion uses</Prose>

    <MathBlock>{'\\begin{gathered}c=\\frac{\\hat p+z^2/(2n)}{1+z^2/n},\\\\'
      + 'r=\\frac{z\\sqrt{\\hat p(1-\\hat p)/n+z^2/(4n^2)}}{1+z^2/n},\\end{gathered}'}</MathBlock>

    <Prose>and reports <Math>{'c\\pm r'}</Math>.</Prose>

    <HeldOutOnly placeholder={'This scale check is computed from the held-out accuracy, so it opens with the '
      + 'report in section 6.'}>
      {/* Badged, because these bounds are computed from the held-out accuracy
          and section 8 goes on to warn against misattributing the interval.
          A learner reading it most needs to know what kind of evidence it is. */}
      <Prose>With <Math>{'k=35'}</Math>, <Math>{'n=36'}</Math> and <Math>{'z=1.96'}</Math>, it is
        approximately [<Score record={wilsonBound(wilson.lower)} digits={3} />,{' '}
        <Score record={wilsonBound(wilson.upper)} digits={3} />]. That is a useful scale check for
        the accuracy estimate under an independent common-probability model. Our stratified three-class
        sampling fixes class counts, so a formal interval targeted to that design, or to balanced accuracy,
        would have to account for the sampling and the class weights. Do not attach this binomial interval to
        balanced accuracy by renaming its axis.</Prose>
    </HeldOutOnly>

    <Prose>For model comparisons, pairing matters. On the development rows the three-feature and two-feature
      models agree on {change.total - change.repaired.length - change.broken.length} of {change.total}
      {' '}specimens. The informative changed cases are the {change.repaired.length} repairs and
      the {change.broken.length} new error. Treating two accuracy estimates as independent discards that
      pairing. A paired bootstrap resamples the same row indices for both models, then recomputes their score
      difference; dependent specimens would instead require group or block resampling. An interval around a
      validation-selected winner also does not undo the optimism caused by the selection.</Prose>

    <H3>More development experiments consume more development evidence</H3>

    <Prose>Repeatedly inspecting validation errors can overfit your choices to those rows. A practical response
      is to reserve independent outer evaluation, limit comparisons to useful hypotheses, and report the
      selection process. Nested cross-validation repeats selection inside each outer training portion and
      evaluates the selected procedure on that outer holdout. It estimates the <strong>procedure</strong>, not
      just one handpicked fit; its computation and interpretation differ from repeatedly retesting the same
      favorite model.</Prose>

    <H3>A system can choose to defer</H3>

    <Prose>Suppose an invented binary inspection service has {looseRule.cases.length} equally weighted
      validation cases. At one declared confidence rule it answers {looseRule.accepted} and
      gets {looseRule.accepted - looseRule.wrong} of those correct. Coverage
      is {looseRule.accepted}/{looseRule.cases.length} = {fixed(looseRule.coverage, 1)}; conditional error
      among answered cases is {looseRule.wrong}/{looseRule.accepted} = {fixed(looseRule.conditionalError, 3)}.
      At a stricter rule it answers {strictRule.accepted} and all are correct: coverage
      {' '}{fixed(strictRule.coverage, 1)}, observed conditional error {fixed(strictRule.conditionalError, 0)}.
      A useful comparison must include the cost and quality of the {strictRule.deferred} deferred cases&apos;
      alternative handling. Plotting accuracy only on answered cases would hide half the work.</Prose>

    <AcceptanceCostLab />

    <Prose>The threshold is another decision chosen on development evidence. Confidence rankings, calibration
      and acceptance rules are distinct; a high score should not be advertised as a verified probability
      without calibration evidence. This extension connects the model to a service-level decision and to the
      earlier <a href="/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml">Calibration
      &amp; Conformal Prediction</a> lesson.</Prose>

    {/* ============================================================ section 9 */}
    <H2 id={headingId(headings[8])}>{headings[8]}</H2>

    <Practice title="1. Reconstruct a score from changed counts"
      constructedFixture={'an invented confusion matrix from a different study, so none of the four evidence '
        + 'roles of this one applies to the scores derived from it'}
      question={<>A new validation study has this confusion matrix, with rows actual and columns predicted:
        <MathBlock>{'\\begin{pmatrix}8&2&0\\\\1&5&0\\\\0&2&2\\end{pmatrix}.'}</MathBlock>
        Compute accuracy and balanced accuracy. Which class would you inspect first if the three classes have
        equal importance?</>}
      hint="Compute each row's recall before averaging. The total number correct is the diagonal sum.">
      <Prose>There are 20 cases and 15 correct, so accuracy is {fixed(endToEndData.practice.accuracy, 2)}.
        Recalls are 0.8, 5/6 and 0.5, giving balanced
        accuracy {endToEndData.practice.balancedAccuracyNumerator}/{endToEndData.practice.balancedAccuracyDenominator} ≈
        {' '}{fixed(endToEndData.practice.balancedAccuracyNumerator
          / endToEndData.practice.balancedAccuracyDenominator, 6)}. Class 2 has the lowest recall. Its
        four-case support also means one changed outcome would move that recall by 0.25, so inspect records and
        uncertainty before announcing a stable subgroup pattern.</Prose>
    </Practice>

    <Practice title="2. Decide whether a slice is worth prioritizing"
      question="Model A makes 12 errors in 100 cases and model B makes 10. In a five-case subgroup, A makes one error and B makes three. What can you conclude, and what additional question determines your decision?"
      hint="Calculate the errors outside the subgroup and distinguish an empirical comparison from a causal explanation.">
      <Prose>Outside the subgroup, A makes 11 of 95 errors and B makes 7 of 95. B gains four there and loses
        two in the subgroup. It improves aggregate accuracy while worsening the small subgroup. The decision
        depends on the consequences and the subgroup&apos;s relevance, along with uncertainty and whether the
        subgroup was predeclared or discovered during exploration. A testable follow-up is a new representative
        sample of that subgroup with a fixed comparison; “B is better everywhere” contradicts the given
        results.</Prose>
    </Practice>

    <Practice title="3. Repair the experiment, not just the code"
      question="A team opens test predictions, notices that large purchases fail, adds a purchase-size interaction, and reports the new score on the same test rows. No test labels enter fit. Explain what happened and propose a valid continuation."
      hint="Draw the information arrow from the researcher to the feature decision.">
      <Prose>Test outcomes influenced feature selection through the team. Those rows now belong to development
        history. Keep the result as exploratory evidence and freeze the revised procedure before evaluating on
        fresh appropriate holdout data; alternatively use a genuinely independent outer evaluation of the
        selection procedure. Deleting the notebook cell does not erase the information consumed.</Prose>
    </Practice>

    <Practice title="4. Guided diagnosis: inspect a different error slice"
      question="In the supplied validation predictions, compare both linear models on the actual cultivar-1 specimens rather than the displayed colour slices. Reproduce the support and the errors, then explain why the result can coexist with the low-colour regression."
      hint="Use (y[valid] == 1) as the mask. Cultivar and colour range are different, overlapping partitions.">
      <Prose>Cultivar 1 has {classOneSlice.linear_two.class1.n} validation specimens. The two-feature linear
        model makes <Count part={classOneSlice.linear_two.class1.errors} whole={classOneSlice.linear_two.class1.n}
          role="validation" noun="wrong" /> and the three-feature model
        makes <Count part={classOneSlice.linear_three.class1.errors} whole={classOneSlice.linear_three.class1.n}
          role="validation" noun="wrong" />. The low-colour slice groups by a feature rather than by the label
        and contains specimens from more than one cultivar. Overlapping subsets need not move in the same
        direction. Record both definitions so that the figures do not imply one partition is the other — and
        note that investigation B above will reproduce the cultivar comparison only if you group by the label,
        which is a different control from the cutoff it offers.</Prose>
    </Practice>

    <Practice title="5. Deeper: assess an acceptance policy"
      question="An inspection system answers 60 of 80 cases, making 6 errors among them. A stricter rule answers 40 and makes 2 errors. Calculate coverage and conditional error. If every deferred case costs 2 units of human work and every wrong automatic answer costs 10 units, compare the total observed costs, assuming the human path produces correct answers."
      hint="Account for both wrong automatic answers and all deferred cases.">
      <Prose>The first rule has coverage 0.75 and conditional error 0.10; its cost is 6(10) + 20(2) = 100. The
        stricter rule has coverage 0.50 and conditional error 0.05; its cost is 2(10) + 40(2) = 100. Better
        conditional accuracy need not improve total cost. The same reversal is available in investigation C: at
        the ten-case fixture&apos;s default costs, tightening the threshold moves cost
        from {looseRule.cost} to {strictRule.cost}, but with a wrong answer priced at 2 rather than 10 the same
        move goes from {cheapErrorLoose.cost} to {cheapErrorStrict.cost} — the opposite direction. The
        calculation changes again if human errors, delay or capacity are included. Use development data to
        choose the policy before its final evaluation.</Prose>
    </Practice>

    <Practice title="6. Deliver your own reproducible study"
      question="Before opening any new holdout outcomes, declare one additional candidate using the supplied training and development rows — for example a different fixed regularization strength for the three-feature model. State why the change should help, what remains fixed, and how you will choose. Produce a prediction table with IDs, an aggregate comparison, a paired repair and regression count, and one clearly defined development slice. Write a final paragraph separating evidence already consumed from the independent evidence your next conclusion would require."
      hint="A useful study may reject the new candidate. Do not choose a new split seed just because it improves the result.">
      <Prose><strong>Evaluation criteria.</strong> The packet should reproduce the candidate and its
        predictions from named inputs, keep preprocessing fitted on the correct training rows, state the
        selection measure, and report unfavorable results as well as gains. An acceptable conclusion is: “The
        lower-regularization candidate did not improve the declared development score; we retain the original
        choice. Because this lesson already disclosed its test result, this extension does not create a new
        independent test of the modified research process.” A production continuation would reserve new,
        appropriately sampled evaluation data.</Prose>
    </Practice>

    <Callout title="Core readiness">
      You can explain which information each split may change, reproduce the fitted pipeline and its score,
      trace an aggregate improvement to the actual changed cases, and write a conclusion matched to the
      evidence that produced it. The deeper route adds uncertainty, selection-procedure evaluation, and
      decisions that include deferral costs.
    </Callout>

    <Sources alternatives={<>
      <h4>Another way to learn it</h4>
      <ul>
        <li><a href="https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets">Google
          Machine Learning Crash Course: dividing datasets</a> — short worked explanations and
          check-your-understanding exercises about development and test roles and about duplicates. Use it
          before section 2 if the information boundary is unfamiliar. The written page and its exercises were
          reviewed; no video was watched.</li>
        <li><a href="https://scikit-learn.org/stable/common_pitfalls.html">scikit-learn: common pitfalls</a> —
          runnable wrong-and-right examples for preprocessing, selection leakage and randomness. Useful after
          section 4; the {endToEndData.versions.sklearn} documentation was inspected.</li>
        <li><a href="https://www.deeplearningbook.org/contents/guidelines.html">Goodfellow, Bengio and
          Courville, <em>Practical Methodology</em></a> — the canonical methodological chapter behind this
          topic: metrics, baselines, data needs, tuning, debugging and a complete application. Read it after
          the core. It is a 2016 treatment; its model-default recommendations and its use of the word “test”
          during tuning need to be read in their historical context, against this lesson&apos;s explicit
          validation and test distinction.</li>
      </ul>
    </>}>
      <li><a href="https://scikit-learn.org/stable/modules/cross_validation.html">The scikit-learn
        cross-validation guide</a> — a deeper reference for evaluating procedures and choosing splitters;
        bring the preceding module&apos;s resampling foundations.</li>
      <li><a href="https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm">NIST&apos;s Wilson
        interval formulas</a> — the formula used in section 8, with its naming note. The arithmetic here is
        calculated independently.</li>
      <li><a href="https://arxiv.org/abs/1810.03993">Model Cards for Model Reporting</a> — the authors&apos;
        reporting proposal, useful for turning a private experiment into an interpretable handoff. The abstract
        and its stated reporting scope were reviewed; this lesson&apos;s report is its own worked example, not
        a reproduction of the paper&apos;s full template.</li>
      <li><a href={endToEndData.provenance.doi}>UCI Wine</a> — the original dataset identity, contributors and
        licence. The supplied extraction and every change made to it are documented
        in <a href={endToEndData.provenance.attribution}>the attribution note</a> served beside the file.</li>
    </Sources>
  </div></HeldOutProvider>,
};

export default endToEndContent;
