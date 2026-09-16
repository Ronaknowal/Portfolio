import { Callout, H2, H3, Prose, Code } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { SpreadLab, WorldsLab } from '../../components/lesson-labs/BiasVarianceLabs.jsx';
import {
  DeviationFigure, SplitFigure, LearningCurveFigure, AxesFigure,
  SameInputsFigure, LossFigure, DoubleDescentFigure,
} from '../../components/lesson-labs/BiasVarianceFigures.jsx';
import { biasVarianceExamples } from '../bias-variance-examples.js';
import {
  boostingTrajectory, printedRounds, procedures, provenance, trainSizes, validationCurveRecord,
} from '../bias-variance-data.js';
import {
  averageVariance, brierDecomposition, doubleDescentNoise, doubleDescentRisk, finiteExperiment,
  firstCrossing, fixedDesignOptimism, fixtures, hiddenSettingVariance, learningCurveSeries,
  predictionSpread, restrictionEffect, thresholdedError, trajectorySeries, validationCurveSeries,
} from '../bias-variance-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');

const series = Object.fromEntries(procedures.map(record => [record.model, learningCurveSeries(record)]));
const leaf = validationCurveSeries(validationCurveRecord);
const trace = trajectorySeries(boostingTrajectory);
const crossing = firstCrossing(series.tree_leaf1, series.ridge);
const restriction = restrictionEffect(series.tree_leaf20, series.tree_leaf1);
const sensorA = predictionSpread(fixtures.sensorA.predictions, fixtures.sensorA.trueMean, fixtures.sensorA.noiseSpread);
const sensorB = predictionSpread(fixtures.sensorB.predictions, fixtures.sensorB.trueMean, fixtures.sensorB.noiseSpread);
const hidden = hiddenSettingVariance();

const worldsBase = { trainX: fixtures.threeInputs, curvature: 1, sigma: 0.5 };
const atHalf = degree => finiteExperiment({ ...worldsBase, degree, probe: 0.5, grid: [0.5] });
const atZero = degree => finiteExperiment({ ...worldsBase, degree, probe: 0, grid: [0] });
const flatAtHalf = degree => finiteExperiment({ ...worldsBase, curvature: 0, degree, probe: 0.5, grid: [0.5] });
const louderQuadratic = finiteExperiment({ ...worldsBase, sigma: 1, degree: 2, probe: 0.5, grid: [0.5] });
const quadraticHalf = atHalf(2);

const optimism = fixedDesignOptimism(6, 2, 4);
const practiceOptimism = fixedDesignOptimism(10, 3, 2);
const brierA = brierDecomposition(fixtures.brierA.eta, fixtures.brierA.probabilities);
const brierB = brierDecomposition(fixtures.brierB.eta, fixtures.brierB.probabilities);
const zeroOneA = thresholdedError(fixtures.brierA.eta, fixtures.brierA.probabilities).error;
const zeroOneB = thresholdedError(fixtures.brierB.eta, fixtures.brierB.probabilities).error;
const averaging = averageVariance(fixtures.averaging.v, fixtures.averaging.rho, fixtures.averaging.B);
const transferRatios = [0.75, 0.9].map(ratio => ({
  ratio,
  base: 1 - ratio,
  amplified: doubleDescentNoise * ratio / (1 - ratio),
  risk: doubleDescentRisk(ratio).riskApprox,
}));

const headings = [
  '1. Hold the question still; change the training data',
  '2. Why the three terms add',
  '3. Retrain every possible tiny dataset',
  '4. Three horizontal axes, three questions',
  '5. A real question: predicting airfoil sound pressure',
  '6. Turn a curve into the next useful experiment',
  '7. Deeper: why training error is optimistic',
  '8. Deeper: classification and averaging',
  '9. Deeper: a tradeoff need not draw a U',
  '10. Practice: from a calculation to a research decision',
  '11. Readiness and the next question',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}
function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="bv-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const biasVarianceContent = {
  title: 'Bias-Variance Tradeoff & Learning Curves',
  readTime: '~60 min first pass · ~110 min complete read + 60–100 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot bv-lesson">
    <LessonIntro prerequisites={<>A mean, a variance, squared error and the fit/validation distinction from the earlier <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a> lesson. The <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization</a> and <a href="/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml">Feature Selection</a> lessons supply useful connections, refreshed where needed. Matrix notation appears only in the deeper branches, and is introduced there.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A model makes a mistake. Separate three quantities that get confused: one fitted model&rsquo;s error, a learning procedure&rsquo;s sensitivity to the data it was trained on, and the uncertainty left in the target itself. You will calculate a small decomposition by hand, enumerate every possible tiny training set and watch the average fitted curve appear, read three different kinds of curve on {provenance.rows.toLocaleString('en-US')} real wind-tunnel measurements including a crossover that no cartoon predicts, and design a controlled next experiment. Both investigations ask for a recorded prediction before they calculate anything, and retire it the moment an input changes.
    </LessonIntro>
    <div className="bv-route"><Prose><strong>First pass.</strong> Read sections 1–6 and try practices 1–5. You will calculate a small decomposition, read three kinds of curve and propose a controlled next experiment. Sections 7–9 explain training optimism, classification and double descent; their additional mathematics is a deeper route you can return to.</Prose></div>

    <Prose>A model makes an error. Should you collect more examples, change its inputs, simplify it, or let it fit a richer relationship? Those actions solve different problems. This lesson gives you a way to reason about them and then check that reasoning against data.</Prose>
    <Prose>Begin with a distinction: <strong>one fitted model&rsquo;s mistake, a learning procedure&rsquo;s sensitivity to its training data, and the uncertainty remaining in the target are different quantities.</strong> A learning curve helps investigate them; it does not directly display all three.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>Imagine calibrating a sensor. At a fixed input setting, the average correct output is 10 units. You repeatedly collect a small training sample and fit the same procedure. Its predictions at that input might be 8, 10 and 12.</Prose>
    <Prose>The average prediction is 10. There is no average offset here, but individual fitted models differ. Another procedure might always predict 9: it is more stable, yet systematically low.</Prose>
    <Prose>Use these original finite teaching distributions, with each listed prediction equally likely and a fresh observed output of 9 or 11, also equally likely:</Prose>
    <LessonTable caption="Two procedures at one input, with the true target mean at 10" headers={['Procedure', 'Predictions across training draws', 'Mean prediction', 'Squared offset from 10', 'Prediction variance', 'Fresh-target noise variance', 'Expected squared error']} rows={[
      ['A', '8, 10, 12', num(sensorA.averagePrediction), num(sensorA.squaredBias), <><Math>{'8/3'}</Math></>, num(sensorA.noiseVariance), <><Math>{'11/3'}</Math></>],
      ['B', '9, 9, 9', num(sensorB.averagePrediction), num(sensorB.squaredBias), num(sensorB.variance), num(sensorB.noiseVariance), num(sensorB.total)],
    ]} />
    <Prose>For A, average <Math>{'(8-10)^2,(10-10)^2,(12-10)^2'}</Math>: the result is <Math>{'8/3'}</Math>. For B, every fitted prediction has the same offset &minus;1. B has lower expected error in this example even though its average prediction is less accurate.</Prose>
    <Prose>Picture a vertical ruler at the chosen input. Put the true mean at 10, predictions from different training samples as three marked dots, and their average as a separate tick. The distance from the average tick to 10 is the <strong>bias</strong>; the spread of prediction dots around their own average supplies the <strong>variance</strong>. A separate pair of target-outcome dots at 9 and 11 represents noise. Keeping these two kinds of dots separate prevents &ldquo;the model is uncertain&rdquo; from becoming an explanation for every source of error.</Prose>
    <Prose>The learning procedure includes the model family, preprocessing, regularization, optimization and any randomization. Bias is a property of that procedure under a specified training-sampling process, not just a label attached to &ldquo;linear&rdquo; or &ldquo;complex.&rdquo; A richer model can have bias from shrinkage or incomplete optimization. A simpler model can be unbiased at a particular input even if it misses the relationship elsewhere.</Prose>
    <Prose>Try editing the three prediction dots in the investigation below. Before revealing the calculation, record whether total expected error will rise, fall or stay unchanged. Moving one dot toward the true mean can change both the average offset and the spread; the two terms must be recalculated together.</Prose>
    <SpreadLab />
    <Prose>Nor does low variance mean low error. A broken program returning zero for every input can be perfectly stable.</Prose>

    <H2>{headings[1]}</H2>
    <Prose>At a fixed input <Math>{'x'}</Math>, define</Prose>
    <MathBlock>{'\\begin{gathered}f(x)=\\mathbb E[Y\\mid X=x],\\\\[4pt]\\sigma^2(x)=\\operatorname{Var}(Y\\mid X=x).\\end{gathered}'}</MathBlock>
    <Prose>The first is the population&rsquo;s mean target at that input. The second describes how actual targets vary around that mean. Neither is normally known from a single real dataset.</Prose>
    <Prose>Let <Math>{'D'}</Math> denote a random training dataset, including algorithmic randomness if relevant, and let <Math>{'\\hat f_D(x)'}</Math> be the resulting prediction. Its average over repeated training draws is <Math>{'\\bar f(x)=\\mathbb E_D[\\hat f_D(x)]'}</Math>. For a fresh target independent of training given <Math>{'x'}</Math>, with finite second moments,</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E_{D,Y\\mid x}\\!\\left[(Y-\\hat f_D(x))^2\\right]\\\\[6pt]=\\underbrace{(\\bar f(x)-f(x))^2}_{\\text{squared bias}}\\\\[6pt]+\\underbrace{\\mathbb E_D[(\\hat f_D(x)-\\bar f(x))^2]}_{\\text{prediction variance}}\\\\[6pt]+\\underbrace{\\sigma^2(x)}_{\\text{target noise}}.\\end{gathered}'}</MathBlock>
    <Prose>Here is the mechanism, rather than a formula to memorize. Write the error as</Prose>
    <MathBlock>{'\\begin{gathered}Y-\\hat f_D(x)\\\\[6pt]=\\underbrace{Y-f(x)}_{\\text{fresh target deviation}}\\\\[6pt]+\\underbrace{f(x)-\\bar f(x)}_{\\text{fixed offset}}\\\\[6pt]+\\underbrace{\\bar f(x)-\\hat f_D(x)}_{\\text{training-draw deviation}}.\\end{gathered}'}</MathBlock>
    <Prose>Square the sum. Each squared term has the meaning above. The mixed terms have expectation zero: the first and third deviations each have mean zero, and the fresh target deviation is independent of the trained prediction under the stated experiment. Thus the three contributions remain after averaging.</Prose>
    <DeviationFigure />
    <Prose>For all test inputs, average this identity over the intended test-input distribution. Noise may depend on <Math>{'x'}</Math>; a constant noise line is appropriate only if that variance is constant. Equal weighting of a plotted grid measures error on that grid, not automatically error under the population&rsquo;s input frequencies.</Prose>
    <Callout title="What this identity is about">
      It concerns <strong>expected squared error on fresh outcomes</strong>. It does not say that training error equals bias squared plus noise, that every observed test error exceeds the noise floor, or that complexity must move bias and variance in opposite directions. The training&ndash;validation distinction returns in section 7.
    </Callout>

    <H3>&ldquo;Irreducible&rdquo; depends on the information available</H3>
    <Prose>Suppose a hidden setting <Math>{'Z'}</Math> is equally likely to be &minus;1 or +1, and <Math>{'Y=X+Z+\\varepsilon'}</Math>, with independent noise of variance {num(hidden.noiseVariance)}. Given only <Math>{'X'}</Math>, the best mean predictor is <Math>{'X'}</Math>, and remaining variance is <Math>{'1+.25=1.25'}</Math>. If the setting <Math>{'Z'}</Math> is measured before prediction, the best mean becomes <Math>{'X+Z'}</Math>, leaving {num(hidden.givenInputAndSetting)}.</Prose>
    <Prose>The new feature changes the conditioning information. It does not refute the old noise floor; it defines a better-informed problem. Improving measurement may also change the target itself. Conversely, repeatedly fitting a larger model to exactly the same inputs cannot explain an independent future noise draw.</Prose>
    <Prose>For a deeper statement, conditioning on <Math>{'X=x'}</Math>,</Prose>
    <MathBlock>{'\\begin{gathered}\\operatorname{Var}(Y\\mid X)\\\\[4pt]=\\mathbb E[\\operatorname{Var}(Y\\mid X,Z)\\mid X]\\\\[4pt]+\\operatorname{Var}(\\mathbb E[Y\\mid X,Z]\\mid X).\\end{gathered}'}</MathBlock>
    <Prose>The second term is the part that observing <Math>{'Z'}</Math> can explain in this example. It connects useful feature acquisition to the decomposition, and it is drawn as the second panel of the figure above.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>A simulation can reveal bias because we choose the truth. A real learning curve cannot usually do that. Let us first make the controlled case fully inspectable.</Prose>
    <Prose>Use three fixed training inputs, <Math>{'[-1,0,1]'}</Math>, and a true response</Prose>
    <MathBlock>{'f(x)=1+x+c x^2.'}</MathBlock>
    <Prose>At each training input, independently add either <Math>{'-\\sigma'}</Math> or <Math>{'+\\sigma'}</Math>. There are exactly <Math>{'2^3=8'}</Math> equally likely training datasets. Fit a constant, a line or a quadratic by least squares to every dataset. This experiment varies the training outcomes while holding the design points fixed; it is neither a bootstrap nor a simulation of random input locations.</Prose>
    <Prose>For <Math>{'c=1,\\sigma=.5'}</Math>, the noiseless training targets are <Math>{'[1,1,3]'}</Math>. One possible noisy set is <Math>{'[.5,1.5,2.5]'}</Math>. All eight sets matter when defining the average fitted prediction.</Prose>
    <Prose>At the test input <Math>{'x=.5'}</Math>, the true mean is {num(quadraticHalf.probeTruth)}. The three procedures give:</Prose>
    <LessonTable caption="Three fits at one probe input, from the same eight equally likely training datasets" headers={['Fit', 'Mean prediction', 'Squared bias', 'Variance', 'Noise', 'Expected squared error']} rows={[
      ['Constant', <><Math>{'5/3'}</Math></>, <><Math>{'1/144'}</Math></>, <><Math>{'1/12'}</Math></>, <><Math>{'1/4'}</Math></>, <><Math>{'49/144='}</Math>{num(atHalf(0).expectedError)}</>],
      ['Line', <><Math>{'13/6'}</Math></>, <><Math>{'25/144'}</Math></>, <><Math>{'11/96'}</Math></>, <><Math>{'1/4'}</Math></>, <><Math>{'155/288='}</Math>{num(atHalf(1).expectedError)}</>],
      ['Quadratic', <><Math>{'7/4'}</Math></>, '0', <><Math>{'23/128'}</Math></>, <><Math>{'1/4'}</Math></>, <><Math>{'55/128='}</Math>{num(atHalf(2).expectedError)}</>],
    ]} />
    <Prose>The constant is best <strong>at this particular input</strong>. Its small bias is partly a coincidence: its average level lies near the truth there. At <Math>{'x=0'}</Math>, its squared bias is <Math>{'4/9'}</Math>, and the quadratic has lower expected error &mdash; {num(atZero(2).expectedError)} against {num(atZero(0).expectedError)}. Never turn one probe point into a global model ranking.</Prose>
    <Prose>The arithmetic behind the quadratic is particularly revealing. Its prediction at .5 is</Prose>
    <MathBlock>{'-.125\\,y_{-1}+.75\\,y_0+.375\\,y_1.'}</MathBlock>
    <Prose>These interpolation weights sum to one. The mean prediction is {num(quadraticHalf.meanPrediction)}; independence of the three noises gives variance</Prose>
    <MathBlock>{'\\begin{gathered}.25\\{(-.125)^2+.75^2\\\\[4pt]+.375^2\\}=23/128.\\end{gathered}'}</MathBlock>
    <Prose>One negative weight is normal for polynomial interpolation. It also shows why changing an observed target can move another prediction in the opposite direction.</Prose>
    <Prose>Change the true curvature to <Math>{'c=0'}</Math>, keeping the same inputs and noise. A line now has zero bias at .5 and expected error <Math>{'35/96='}</Math>{num(flatAtHalf(1).expectedError)}; the quadratic still has extra prediction variance and error <Math>{'55/128='}</Math>{num(flatAtHalf(2).expectedError)}. Extra flexibility did not buy a better approximation because a line already represents this truth.</Prose>

    <H3>Complete calculation</H3>
    <Prose>Save this as a Python file and run it with NumPy. Arrays use rows for training-dataset realizations and columns for probe inputs. The variance divisor is 8 because we enumerate an entire equally weighted finite distribution; it is not an unbiased-sample-variance estimate.</Prose>
    <Prose>A degree-<Math>{'d'}</Math> fit represents <Math>{'b_0+b_1x+\\cdots+b_dx^d'}</Math>. The design matrix has three rows and <Math>{'d+1'}</Math> columns; each column supplies one power of the input. Least squares chooses coefficients to minimize the sum of squared differences from the three targets. Solving for all eight target columns produces a coefficient array of shape <Math>{'(d+1,8)'}</Math>; evaluating at two probes gives eight predictions at each probe. This is why the program transposes its final product before averaging across training worlds.</Prose>
    <Program example={biasVarianceExamples.finiteWorlds}>
      <Prose>Each printed row is one degree, then four pairs: the mean prediction, the squared bias, the prediction variance and the expected squared error, at the probes 0 and .5 in that order. At the .5 probe this produces the table above; at zero, the line and constant print identical values in all four pairs because every one of the eight datasets gives them the same prediction there, whereas the quadratic&rsquo;s expected error is .5. The larger retained author calculation checks this identity on 61 probe positions and also supplies a five-input version for the investigation below. It uses the same least-squares operation without clipping predictions or changing a hidden regularization parameter.</Prose>
    </Program>
    <Prose>In the investigation below, edit curvature, noise level and the probe location. Predict whether switching from one fit to another will lower, raise or preserve expected squared error, then reveal the actual eight fitted curves, average curve and separate error contributions. A second mode adds training inputs at <Math>{'-.5,.5'}</Math>, enumerating 32 possible datasets. Setting noise to zero is a useful null: a correctly specified, identified polynomial reproduces the truth in every training draw.</Prose>
    <WorldsLab />
    <Checkpoint prompt="Switching the design from three inputs to five, with curvature 1, noise .5 and the probe still at .5, raises the constant fit's expected error from 0.340278 to 0.3625 — even though its prediction variance falls from 0.083333 to 0.05. How can adding two measurement locations make a fit worse at this input?">
      <Prose>Adding the inputs at &minus;.5 and .5 changes the design, and therefore changes what the average fitted constant is. Its level shifts from {num(atHalf(0).meanPrediction)} to 1.5, so its squared bias at this probe rises from {num(atHalf(0).squaredBias)} to 0.0625 &mdash; more than enough to outweigh the fall in variance. Both terms have to be read together. This is a property of this design-specific experiment, not evidence that collecting more data always harms simple models; the two extra inputs are particular chosen locations, not an extra independent sample. The investigation&rsquo;s five-input preset shows both terms side by side.</Prose>
    </Checkpoint>

    <H3>What can a bootstrap establish?</H3>
    <Prose>A bootstrap samples rows with replacement from the one dataset you possess. It can approximate aspects of a fitted procedure&rsquo;s sampling variability when its assumptions are suitable. It does not reveal the unknown <Math>{'f(x)'}</Math> merely by repeating the fit. Replacing <Math>{'f(x)'}</Math> with the original fitted model changes the target of a bias calculation.</Prose>
    <Prose>Likewise, repeating random seeds on unchanged data measures conditional algorithmic variability, not all variability across newly collected training datasets. These are useful experiments when named correctly. They answer different questions from our exact eight-world calculation.</Prose>

    <H2>{headings[3]}</H2>
    <LessonTable caption="Three curves that look alike and answer different questions" headers={['Curve', 'Horizontal axis', 'What stays fixed', 'Question']} rows={[
      ['Learning curve', 'Number of training examples', 'Procedure and evaluation protocol', 'What happens as this procedure receives more data?'],
      ['Validation curve', 'A hyperparameter, such as minimum leaf size', 'Data splits and other settings', 'Which inspected setting predicts better?'],
      ['Training trajectory', 'Iteration, boosting round or epoch', 'Data and training run', 'What happens as this optimization run continues?'],
    ]} />
    <Prose>All may show training and held-out error. They are not interchangeable. A larger minimum leaf size restricts a tree; a larger polynomial degree expands a function family; more trees in an averaging ensemble and more rounds in boosting have different effects.</Prose>
    <Prose>To build a learning curve, choose the deployment-relevant split first. For each training portion, fit anew at each chosen sample size and score both that fitting subset and the associated held-out portion. Preprocessing belongs inside each fitted procedure. Use the same validation examples across sizes within a fold so a size comparison does not silently become a comparison of different test populations.</Prose>
    <SplitFigure />
    <Prose>Sampling smaller training subsets introduces another choice. Randomly ordered prefixes are suitable for an exchangeable-data diagnostic. Chronological prefixes answer a different question because both time and amount of data change; grouped units need group-aware subsets. The real example below states exactly which experiment it runs.</Prose>
    <Prose>One descending curve provides evidence over the inspected sizes. It does not guarantee its future slope. A plateau over a short, noisy interval does not prove you have exhausted all useful information.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>The supplied <a href={provenance.file} download>Airfoil Self-Noise data</a> contains {provenance.rows.toLocaleString('en-US')} wind-tunnel measurements. The inputs are frequency (Hz), angle of attack (degrees), chord length (m), free-stream velocity (m/s), and suction-side displacement thickness (m); the target is scaled sound-pressure level (dB). This is a regression problem about aerodynamic measurements, not a classification problem about the presence of noise. <a href={provenance.page}>UCI dataset and license</a>; this page serves its own unchanged copy, {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a>, beside its <a href={provenance.attribution}>attribution</a>. Nothing is downloaded when the program runs.</Prose>
    <Prose>We ask: <strong>on row-level held-out measurements from this supplied collection, which procedures benefit from increasing training size?</strong> The collection contains related experimental settings; its file does not supply a deployment-ready independent-run identifier. A random row split therefore does not establish performance on a new airfoil family or new experiment. For that goal, define the appropriate physical group and split it deliberately. Our scoped comparison uses a fixed development pool of {provenance.developmentRows.toLocaleString('en-US')} rows; {provenance.reservedRows} rows remain unused by this lesson.</Prose>
    <Prose>Compare four prespecified procedures:</Prose>
    <ul>
      {procedures.map(record => <li key={record.model}><strong>{record.label}.</strong> {record.description}</li>)}
    </ul>
    <Prose>The tree contrast isolates a capacity restriction. The linear model provides another family; its preprocessing is fitted separately within each training subset. We use MSE in squared dB of the supplied target scale. This does not turn a squared error on a logarithmic sound-level scale into acoustic power error.</Prose>
    <Prose>Download the data beside this complete program; install NumPy and scikit-learn if needed. Author calculations used NumPy 2.3.5 and scikit-learn 1.9.1.</Prose>
    <Program example={biasVarianceExamples.airfoilLearningCurve}>
      <Prose>Each of the five folds has {provenance.foldTrainRows} available training rows and {provenance.foldValidationRows} validation rows. Requested size 900 means 900 fitted rows per fold, not 900 total rows split five ways. The returned arrays have shape (5 sizes, 5 folds). The scoring interface maximizes scores, so it returns negative MSE; the program negates them for ordinary positive error reporting. That is also why the leaf-1 tree&rsquo;s training column prints <Code>-0.0</Code>: negating an exact zero keeps its sign bit, and the value is zero. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html">Learning-curve API</a>.</Prose>
    </Program>
    <Prose>The recorded mean validation MSE is:</Prose>
    <LessonTable caption="Mean validation MSE in squared decibels, by fitted rows per fold" headers={['Fitted rows per fold', 'Mean baseline', 'Ridge', 'Tree, leaf 1', 'Tree, leaf 20']} rows={trainSizes.map((size, index) => [
      String(size),
      series.mean.validationMeans[index].toFixed(4),
      series.ridge.validationMeans[index].toFixed(4),
      series.tree_leaf1.validationMeans[index].toFixed(4),
      series.tree_leaf20.validationMeans[index].toFixed(4),
    ])} />
    <Prose>Several things are worth seeing together. Ridge is better with few examples. The unrestricted tree improves enough to pass it as training size grows, first doing so at {crossing} fitted rows per fold. Requiring 20 items per leaf helps the tree at {restriction.helps.join(' and ')} fitted rows but hurts at {restriction.hurts.join(', ')}. &ldquo;Regularize a flexible model&rdquo; is therefore a hypothesis to evaluate, not a guaranteed remedy.</Prose>
    <Prose>The leaf-1 tree&rsquo;s training MSE is zero at every inspected size. Its validation MSE nevertheless decreases from {series.tree_leaf1.validationMeans[0].toFixed(4)} to {series.tree_leaf1.validationMeans.at(-1).toFixed(4)}. Exact training fit does not tell you whether the next larger training set will help, nor does zero training loss automatically imply useless predictions.</Prose>
    <Prose>At size {trainSizes.at(-1)}, Ridge has train/validation MSE {series.ridge.trainingMeans.at(-1).toFixed(4)}/{series.ridge.validationMeans.at(-1).toFixed(4)}; the leaf-20 tree has {series.tree_leaf20.trainingMeans.at(-1).toFixed(4)}/{series.tree_leaf20.validationMeans.at(-1).toFixed(4)}. A small Ridge gap does not certify proximity to the unknown noise floor: another inspected procedure already predicts much better.</Prose>
    <LearningCurveFigure />
    <Prose>The visual compares small multiples with shared axes and the mean baseline visible. Individual fold values stay available, with lines showing the mean. Fold-to-fold spread is a descriptive diagnostic; these folds overlap in their training data, so a standard deviation across five folds is not automatically a confidence interval for a difference. To compare two models, inspect their paired errors on matching observations and use an uncertainty method appropriate to the data dependence.</Prose>

    <H3>Change a setting, then inspect a training trajectory</H3>
    <Prose>Append the following to the same program, saved as <Code>{biasVarianceExamples.settingAndTrajectory.appendedTo}</Code>. The first comparison changes one tree setting at fixed fold-training size {validationCurveRecord.fitRowsPerFold}. The second traces one prespecified boosting fit on a separate split of the development pool.</Prose>
    <Program example={biasVarianceExamples.settingAndTrajectory}>
      <Prose>Validation MSE for minimum leaf sizes {leaf.settings.join(', ')} is respectively {leaf.validationMeans.map(value => value.toFixed(4)).join(', ')}. This inspected restriction does not improve the score. The leaf-1 result differs from {series.tree_leaf1.validationMeans.at(-1).toFixed(4)} above because this run fits {validationCurveRecord.fitRowsPerFold} rather than {trainSizes.at(-1)} rows per fold.</Prose>
    </Program>
    <Prose>The boosting trace is:</Prose>
    <LessonTable caption="Boosting trajectory: the printed rounds, from a run of 120" headers={['Round', 'Training MSE', 'Monitoring MSE']} rows={printedRounds.map(number => [
      String(number), trace.train[number - 1].toFixed(4), trace.monitor[number - 1].toFixed(4),
    ])} />
    <Prose>The best of all {trace.lastRound} inspected rounds is {trace.bestRound}. We have <strong>not observed an optimal stopping point followed by deterioration</strong>. If additional rounds are a worthwhile hypothesis, specify and test them on development data. Do not invent an overfitting turn merely because a standard illustration usually contains one. This staged comparison is also not a ranking against the five-fold experiment: the validation partitions differ.</Prose>
    <AxesFigure />
    <Prose>These are model-development results. The {provenance.reservedRows} reserved rows were not used for any reported selection or score. A final performance report would first freeze a procedure and then evaluate it under an appropriate independent protocol.</Prose>

    <H3>Keep the diagnostic affordable without changing its meaning</H3>
    <Prose>The learning-curve program performs <Math>{'4\\times5\\times5=100'}</Math> fits: four procedures, five sizes and five folds. More candidate settings multiply that work. The staged boosting display reuses one fitted run&rsquo;s intermediate predictions instead of refitting from scratch at every round. A small exact-world example exposes the mathematics cheaply; the real study answers a different empirical question.</Prose>
    <Prose>For a larger problem, inspect a justified smaller grid first, retain split IDs and outcomes, and expand when the unresolved question warrants it. A curve on ten percent of the data cannot guarantee the shape on the other ninety percent. scikit-learn&rsquo;s incremental learning-curve mode requires the estimator&rsquo;s partial-fit interface; a warm-start flag alone does not make it valid. Changing to an incremental training procedure may also change the estimator being studied. Measure actual cost rather than assigning universal sample or runtime cutoffs.</Prose>

    <H2>{headings[5]}</H2>
    <Prose>Before diagnosing a curve, verify that its axes, target, scoring direction and data units are what you think they are. Then separate observation from hypothesis.</Prose>
    <LessonTable caption="From an observation to an experiment that could refute your explanation" headers={['Observation', 'Plausible explanation', 'Useful discriminating experiment']} rows={[
      ['Both errors are poor relative to a meaningful baseline', 'Insufficient features/flexibility, too much shrinkage, poor optimization, or a bug', 'Fit a small known-solvable case; then change one of those constraints'],
      ['Training error is low; held-out error is much larger', 'Sample sensitivity, leakage/mismatch, or an unrepresentative split may matter', 'Audit units/availability; compare paired results under a controlled regularization or data-size change'],
      ['Held-out error improves over inspected sizes', 'More relevant data has helped this procedure over this range', 'Test a larger size with the same protocol and preserve the actual outcome'],
      ['Both curves flatten near each other', 'The current family may be limited; target uncertainty or the sampled range may also dominate', 'Compare a justified alternate representation/model, inspect labels and slices; do not declare the noise floor known'],
      ['Monitoring error worsens while fitting error falls', 'Further fitting is harming this measured validation objective', 'Assess stopping/regularization under a prespecified selection protocol'],
    ]} />
    <Prose>A feature-selection result from the preceding lesson is one candidate input to this process. Importance does not prove that removing a feature will improve the learning curve. Retrain and evaluate the proposed reduced-input pipeline. If collecting a new measurement changes the prediction question or availability, document that change as well.</Prose>
    <Prose>Data acquisition also has a composition question. Ten thousand near-duplicate measurements may add little independent information; a smaller set covering an underserved operating range may answer the actual failure. Plot errors by that range and define the desired deployment population. A global learning curve can hide a subgroup that still lacks coverage.</Prose>
    <Prose>For practical planning, record: the observed result, your proposed mechanism, one changed setting or data collection, held-fixed quantities, evaluation unit, metric and what outcome would count against the hypothesis. This is more useful than assigning a categorical diagnosis from an arbitrary gap percentage.</Prose>

    <H2>{headings[6]}</H2>
    <Prose><strong>This section and the two that follow are the deeper route.</strong> They add matrix notation, loss-specific definitions and an asymptotic calculation. The core route is complete without them.</Prose>
    <Prose>The fitted model used the training outcomes, so they are not fresh observations independent of its predictions. The mixed-term argument in section 2 cannot simply be applied to its residuals.</Prose>
    <Prose>There are also two different risk targets. Conditional risk asks how the particular model trained on the observed <Math>{'D'}</Math> predicts fresh cases. Expected procedure risk averages that risk over new training datasets as well. An independent test set primarily evaluates the fitted model it is given; the bias&ndash;variance identity describes the repeated-training average. Cross-validation estimates depend on its refitting sizes and protocol. Keeping those targets distinct prevents a single test score from being mistaken for a direct population-variance measurement.</Prose>
    <Prose>Consider correctly specified least squares with a fixed full-column-rank design matrix <Math>{'X'}</Math>, <Math>{'n'}</Math> rows and <Math>{'p'}</Math> fitted coefficients, including an intercept if present. Write</Prose>
    <MathBlock>{'\\begin{gathered}y=X\\beta+\\varepsilon,\\quad\\mathbb E\\varepsilon=0,\\\\[4pt]\\operatorname{Cov}(\\varepsilon)=\\sigma^2 I,\\end{gathered}'}</MathBlock>
    <Prose>and <Math>{'H=X(X^\\top X)^{-1}X^\\top'}</Math>. This matrix projects outcomes onto the fitted column space, so <Math>{'\\hat y=Hy'}</Math>.</Prose>
    <Prose>The training residual is <Math>{'(I-H)\\varepsilon'}</Math>. Because <Math>{'I-H'}</Math> is a projection of rank <Math>{'n-p'}</Math>,</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E[\\mathrm{MSE}_{\\mathrm{train}}]\\\\[4pt]=\\frac{\\sigma^2}{n}\\operatorname{tr}(I-H)\\\\[4pt]=\\sigma^2(1-p/n).\\end{gathered}'}</MathBlock>
    <Prose>Now obtain independent new outcomes <Math>{'y\'=X\\beta+\\varepsilon\''}</Math> <strong>at those same input rows</strong>. Their prediction errors are <Math>{'\\varepsilon\'-H\\varepsilon'}</Math>. Independence gives</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E[\\mathrm{MSE}_{\\text{fresh, same }X}]\\\\[4pt]=\\sigma^2(1+p/n).\\end{gathered}'}</MathBlock>
    <Prose>The gap is <Math>{'2p\\sigma^2/n'}</Math>, while average fitted-prediction variance at those inputs is <Math>{'p\\sigma^2/n'}</Math>. Even in this favorable setting the gap is twice that variance, not a direct variance estimate.</Prose>
    <Prose>Take <Math>{'n=6,p=2,\\sigma^2=4'}</Math>. Expected training error is <Math>{'8/3'}</Math>, new-outcome error is <Math>{'16/3'}</Math>, and prediction variance is <Math>{'4/3'}</Math> &mdash; that is {num(optimism.trainingMse)}, {num(optimism.newOutcomeMse)} and {num(optimism.predictionVariance)}. All arise from the same model.</Prose>
    <SameInputsFigure />
    <Prose>For a genuinely new input vector <Math>{'x_*'}</Math>, including any intercept coordinate, conditional on the fixed training design,</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E[(Y_*-\\hat f(x_*))^2\\mid X,x_*]\\\\[4pt]=\\sigma^2+\\sigma^2 x_*^\\top(X^\\top X)^{-1}x_*,\\end{gathered}'}</MathBlock>
    <Prose>under the same correct-model and fresh-noise assumptions. Its leverage depends on location, as the last table of the figure above shows. The same-input average <Math>{'1+p/n'}</Math> is therefore not a universal out-of-distribution or random-design test-error formula.</Prose>
    <Prose>This projection also exposes estimation versus approximation. If the true mean vector is <Math>{'f'}</Math> rather than <Math>{'X\\beta'}</Math>, training error gains <Math>{'\\|(I-H)f\\|^2/n'}</Math>. High training error can contain approximation failure; small training error alone says little about a new input far from the fitted design.</Prose>

    <H3>Effective degrees of freedom and regularization</H3>
    <Prose>For a fixed linear smoother <Math>{'\\hat y=Sy'}</Math>, training error under a mean vector <Math>{'f'}</Math> is</Prose>
    <MathBlock>{'\\begin{gathered}\\frac{\\|(I-S)f\\|^2}{n}\\\\[4pt]+\\frac{\\sigma^2}{n}\\{n-2\\operatorname{tr}(S)\\\\[4pt]+\\operatorname{tr}(S^\\top S)\\}.\\end{gathered}'}</MathBlock>
    <Prose>Fresh outcomes at the same inputs have expected error</Prose>
    <MathBlock>{'\\begin{gathered}\\frac{\\|(I-S)f\\|^2}{n}+\\sigma^2\\\\[4pt]+\\frac{\\sigma^2}{n}\\operatorname{tr}(S^\\top S).\\end{gathered}'}</MathBlock>
    <Prose>Subtracting leaves <Math>{'2\\sigma^2\\operatorname{tr}(S)/n'}</Math>. For ordinary least squares <Math>{'S=H'}</Math>, which is the reduction checked in the figure above; for Ridge, the shrinkage matrix has smaller trace but may introduce bias. This explains the role of an optimism correction without assuming one parameter-count formula fits every learner.</Prose>
    <Prose>Mallows-style <Math>{'C_p'}</Math> corrections estimate this expected optimism using a noise estimate. Likelihood criteria such as AIC and BIC answer related model-selection questions under their own assumptions; they are not measurements of the three decomposition terms. Section 9 of the earlier <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization lesson</a> develops their different predictive and evidence-based aims, alongside minimum description length and an explicit coding example. Generalized cross-validation uses a linear smoother&rsquo;s effective degrees of freedom as an efficient leave-one-out approximation. Hyperparameter selection performed on the same observations adds another adaptive step, so fixed-<Math>{'S'}</Math> algebra is not automatically a complete correction for the selected procedure.</Prose>

    <H2>{headings[7]}</H2>
    <Prose>For a binary target with <Math>{'\\eta(x)=P(Y=1\\mid x)'}</Math>, the squared probability error, or binary Brier loss, has the exact decomposition</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E[(Y-\\hat p_D(x))^2]\\\\[4pt]=(\\eta(x)-\\mathbb E\\hat p_D(x))^2\\\\[4pt]+\\operatorname{Var}(\\hat p_D(x))\\\\[4pt]+\\eta(x)(1-\\eta(x)).\\end{gathered}'}</MathBlock>
    <Prose>This is an exact squared-loss identity, not an approximation to classification accuracy.</Prose>
    <Prose>Thresholded class decisions behave differently. If <Math>{'\\eta=.8'}</Math> and a training procedure predicts class 1 with probability <Math>{'q'}</Math>, its expected zero-one error is</Prose>
    <MathBlock>{'.8(1-q)+.2q=.8-.6q.'}</MathBlock>
    <Prose>Changing <Math>{'q'}</Math> from 0 to .25 reduces error from .8 to .65 while introducing variation in the predicted class. Changing it from 1 to .75 increases error from .2 to .35. Thus increased variation may help or hurt, depending on which decisions it replaces. <a href="https://homes.cs.washington.edu/~pedrod/papers/aaai00.pdf">Domingos&rsquo;s primary paper</a> develops loss-specific definitions; it should not be reduced to &ldquo;the same three positive terms for every metric.&rdquo;</Prose>
    <LossFigure />
    <Prose>A useful separate calculation explains averaging. Suppose <Math>{'B'}</Math> predictors at a fixed input have equal variance <Math>{'v'}</Math> and pairwise correlation <Math>{'\\rho'}</Math>. Their equal-weight average has variance</Prose>
    <MathBlock>{'v\\left(\\rho+\\frac{1-\\rho}{B}\\right).'}</MathBlock>
    <Prose>Expand the variance of the sum: there are <Math>{'B'}</Math> individual variances and <Math>{'B(B-1)'}</Math> pair covariances, then divide by <Math>{'B^2'}</Math>.</Prose>
    <Prose>With <Math>{'v=4,\\rho=.5,B=4'}</Math>, the average variance is {num(averaging.variance)}, not {num(averaging.independentVariance)}. Perfectly correlated predictions gain nothing from averaging; independent ones have variance <Math>{'v/B'}</Math>. If the constituent mean predictions stay the same, averaging does not remove their common bias.</Prose>
    <Prose>This calculation motivates bagging and why diversity matters, but bootstrap models from one dataset are not independent. Changing the training distribution through resampling can also alter their mean predictions. Boosting adds sequential corrective fits; it is not described completely by this fixed-constituent averaging calculation. The earlier ensemble lesson owns those training mechanisms.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>The squared-error identity remains true when expectations exist. It does not prescribe how its terms move with width, depth, training size or optimization time. Double descent concerns those trajectories, not a failure of expanding a square.</Prose>
    <Prose>To see a concrete alternative, consider a specified linear world with <Math>{'p'}</Math> independent standard-Gaussian input coordinates, true coefficient norm <Math>{'\\|\\beta\\|=1'}</Math>, and independent target noise of variance {doubleDescentNoise}. Fit the <strong>minimum Euclidean-norm unregularized least-squares solution</strong>, using the pseudoinverse. Set <Math>{'\\gamma=n/p'}</Math>, the ratio of training rows to coordinates.</Prose>
    <Prose>In the large-<Math>{'n,p'}</Math> approximation with fixed ratio, expected excess squared error is</Prose>
    <MathBlock>{'\\begin{cases}(1-\\gamma)+\\dfrac{.04\\,\\gamma}{1-\\gamma},&\\gamma<1,\\\\[8pt]\\dfrac{.04}{\\gamma-1},&\\gamma>1.\\end{cases}'}</MathBlock>
    <Prose>Add {doubleDescentNoise} for fresh-target noise. These are asymptotic theoretical values, not a timing experiment or finite-sample measured result. <a href="https://arxiv.org/pdf/1912.07242">Nakkiran, Claims 1&ndash;2</a>.</Prose>
    <LessonTable caption="The stated ratios and their approximate fresh-target MSE" headers={['n/p', 'Approximate fresh-target MSE']} rows={fixtures.doubleDescentRatios.map(ratio => [
      ratio.toFixed(2), doubleDescentRisk(ratio).riskApprox.toFixed(6),
    ])} />
    <Prose>Adding data initially helps, then hurts near the interpolation boundary, then helps again. The prediction-variance term below the boundary includes randomness of the observed input subspace as well as amplified target noise. At ratios close to one, small singular values make inversion especially sensitive to noise. More rows beyond that region constrain the fit differently.</Prose>
    <DoubleDescentFigure />
    <Prose>The graph marks the undefined asymptotic boundary at <Math>{'\\gamma=1'}</Math> rather than connecting a smooth finite line through it. There is no simulated hardware or neural architecture in these numbers.</Prose>
    <Prose>For this linear objective, convergent gradient descent from zero finds the minimum-norm solution. That statement does not establish that gradient descent finds an analogous minimum-norm predictor for every nonlinear network. Parameter count alone is also not a reliable interpolation threshold for arbitrary architectures and losses.</Prose>
    <Prose><a href="https://arxiv.org/html/1812.11118v2">Belkin and colleagues</a> document model-wise examples, including tree ensembles; the phenomenon is not exclusive to neural networks. Deep networks add optimization-time and data-size behavior to investigate. The practical consequence is to measure the relevant trajectory and regularization choices, not to replace &ldquo;smaller is always safer&rdquo; with &ldquo;bigger is always safer.&rdquo;</Prose>

    <H2>{headings[9]}</H2>
    <Prose>Try the first five without the deeper branches. Hints and solutions are optional so you can work independently before checking.</Prose>

    <Practice title="1. A new three-world sensor"
      question={<>The true mean at one input is 5. Equally likely fitted predictions are 3, 4 and 8. Fresh target noise has variance 2 and is independent of training. Compute bias, variance and expected squared error. Compare with a procedure always predicting 4.</>}
      hint="Average predictions before measuring their spread. The target mean and average prediction need not be equal.">
      <Prose>The mean is 5, so bias is zero. Prediction variance is <Math>{'(4+1+9)/3=14/3'}</Math>; expected error is <Math>{'20/3'}</Math>. Always predicting 4 gives squared bias 1, variance 0, error 3. The stable biased procedure wins here. Noise has the same value in both comparisons.</Prose>
    </Practice>

    <Practice title="2. Change the noise, keep the procedure"
      question={<>For the quadratic in section 3 at <Math>{'x=.5'}</Math>, keep curvature 1 and change <Math>{'\\sigma'}</Math> to 1. Compute the new variance and total error. Then explain why the bias stays zero.</>}
      hint="The noise variance multiplies the squared interpolation weights; the mean of each training noise remains zero.">
      <Prose>Variance is <Math>{'23/32='}</Math>{num(louderQuadratic.variance)}; fresh-target noise is 1, so expected error is <Math>{'55/32='}</Math>{num(louderQuadratic.expectedError)}. Correctly specified interpolation reproduces the quadratic mean in expectation at these full-rank input points. This conclusion relies on this estimator and experiment, not merely on containing the true function in a broad model family. The investigation&rsquo;s &ldquo;Louder noise&rdquo; setup is exactly this case: apply it and read the candidate row.</Prose>
    </Practice>

    <Practice title="3. Diagnose the diagnosis"
      question="A colleague observes train/validation MSE 12/13 at several inspected sizes and says, “The model has low variance, the noise floor is 13, and collecting more data is pointless.” List what is observed and propose two distinct tests before accepting that conclusion."
      hint="You have no measured population mean or conditional noise variance. Consider another model and another part of the data-generation process.">
      <Prose>The observed curves are close and flat over the inspected sizes. That does not identify the decomposition. Compare a justified richer or less-regularized procedure under the same splits, and inspect target measurement, feature availability, group composition or optimization with known cases. A meaningful new-size experiment can also test the local plateau. Report outcomes that would weaken each hypothesis, rather than calling a gap of 1 a variance estimate.</Prose>
    </Practice>

    <Practice title="4. Investigate a new restriction on real data"
      question={<>Using only the supplied {provenance.developmentRows.toLocaleString('en-US')}-row development pool, compare a tree with maximum depth 4 against the existing leaf-1 tree at the same five sizes and folds. Predict at which size, if any, the restriction will first stop helping. Keep the original feature set, row IDs, metric and shuffle settings fixed.</>}
      hint="Add a separately named model to the program's dictionary, leaving the baseline procedures unchanged. A crossover may not occur in the inspected range."
      revealLabel="Assessment and example conclusion">
      <Prose>A complete answer contains the saved prediction, five paired score comparisons with fold values, and a conclusion tied to the actual result. &ldquo;The restricted model was never better over these sizes&rdquo; is acceptable if supported. Do not tune depth repeatedly and then present the selected score as independent evaluation. The {provenance.reservedRows} reserved rows remain unused while this development exercise continues. For calibration, the leaf-20 restriction in the lesson helps only at {restriction.helps.join(' and ')} fitted rows and hurts at every larger inspected size.</Prose>
    </Practice>

    <Practice title="5. Two sets of evidence"
      question="In an invented study, model A's validation error decreases 20 → 14 → 11 as fitted rows increase 100 → 300 → 900. Model B scores 13 on one different validation split with 900 fitted rows. Explain what may be concluded, and design the missing comparison."
      hint="The data-size trajectory and the model-family comparison are separate.">
      <Prose>A improved over the inspected training sizes under its protocol. B&rsquo;s 13 cannot be cleanly ranked against A&rsquo;s 11 without accounting for the changed evaluation set. Fit the two prespecified procedures on matching training subsets and score the same validation units; inspect paired errors and relevant uncertainty. Neither trajectory certifies a future deployment improvement.</Prose>
    </Practice>

    <Practice title="6. A new optimism calculation — deeper"
      question={<>For correctly specified fixed-design OLS with <Math>{'n=10,p=3,\\sigma^2=2'}</Math>, calculate expected training MSE, fresh-response MSE at the same inputs, their gap, and average fitted-prediction variance. State what changes for a new input location.</>}>
      <Prose>The values are {num(practiceOptimism.trainingMse)}, {num(practiceOptimism.newOutcomeMse)}, {num(practiceOptimism.gap)} and {num(practiceOptimism.predictionVariance)}. For a new <Math>{'x_*'}</Math>, the variance contribution is <Math>{'2x_*^\\top(X^\\top X)^{-1}x_*'}</Math>; the fixed-design average cannot be reused without its location and distribution assumptions. A rank-deficient design would also require the effective rank rather than blindly counting columns.</Prose>
    </Practice>

    <Practice title="7. Is a probability loss the same as accuracy? — deeper"
      question={<>At one input, <Math>{'P(Y=1)=.7'}</Math>. Procedure A always outputs probability .6; B outputs .4 or .8 with equal chance across independent training datasets. Compute expected Brier loss and expected classification error using threshold .5.</>}
      hint="Both procedures average to .6, but only one crosses the class threshold across fits.">
      <Prose>A has Brier loss <Math>{'.01+.21=.22'}</Math>, that is {num(brierA.total)}. B has the same squared bias {num(brierB.squaredBias)} plus prediction variance {num(brierB.variance)} and noise {num(brierB.noise)}, totaling {num(brierB.total)}. A always predicts class 1, giving error {num(zeroOneA)}. B predicts 0 or 1 equally, giving error {num(zeroOneB)}. This example compares two losses explicitly; it does not turn {num(brierB.variance)} into a universal classification-variance term.</Prose>
    </Practice>

    <Practice title="8. Transfer the nonmonotonic example — deeper"
      question={<>Use section 9&rsquo;s stated approximation at <Math>{'\\gamma=.75'}</Math> and .9. Compute risk and explain why adding examples can worsen it without contradicting section 2. Then name two assumptions you would check before applying that conclusion to a neural model.</>}>
      <Prose>At <Math>{'\\gamma=.75'}</Math> the risk is {num(transferRatios[0].base)} + {num(transferRatios[0].amplified)} + {doubleDescentNoise} = {num(transferRatios[0].risk)}; at .9 it is {num(transferRatios[1].base)} + {num(transferRatios[1].amplified)} + {doubleDescentNoise} = {num(transferRatios[1].risk)}. The risk identity allows a changing variance term; it does not assert monotonic learning curves. The example assumes isotropic Gaussian inputs, a correctly specified linear target, independent noise and minimum-norm ridgeless fitting in a large-dimensional limit. A neural model&rsquo;s data geometry, loss, regularization and optimization selection need their own evidence.</Prose>
    </Practice>

    <H2>{headings[10]}</H2>
    <Prose>Core readiness: distinguish bias, prediction variability and target noise; calculate the changed finite example; tell the three plot axes apart; interpret real curves without inventing a diagnosis; design one useful controlled next experiment. Deeper readiness adds the fixed-design optimism derivation, loss-specific distinction and conditional double-descent explanation.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Separate a fitted mistake, sample sensitivity and target noise', 'Section 1, the prediction-ruler investigation, practice 1'],
      ['Say why the three terms add, and where the cross terms went', 'Section 2, figure 1'],
      ['Recompute a finite decomposition after changing curvature, noise or the probe', 'Section 3, the finite-worlds investigation, practice 2'],
      ['Tell a learning curve, a validation curve and a training trajectory apart', 'Section 4, figures 2 and 4'],
      ['Read the recorded airfoil curves, including the crossover and the failed restriction', 'Section 5, figure 3, practice 4'],
      ['Turn an observation into an experiment that could refute it', 'Section 6, practices 3 and 5'],
      ['Explain why training error is optimistic, and why the gap is twice the variance', 'Section 7, figure 5, practice 6'],
      ['Separate a probability loss from a thresholded decision, and explain averaging', 'Section 8, figure 6, practice 7'],
      ['Explain a nonmonotonic risk curve without contradicting the identity', 'Section 9, figure 7, practice 8'],
    ]} />
    <Prose>Next is <a href="/learn/path/full-curriculum/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml">Imbalanced Learning</a>. Carry forward a question that a single average curve can hide: whose mistakes does the metric count, and does the training/evaluation population give the rare cases enough attention?</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://www.youtube.com/watch?v=zrEyxfl2-a8">Caltech &mdash; Learning From Data, Lecture 8</a>, with <a href="https://work.caltech.edu/slides/slides08.pdf">official lecture slides</a>. A visual mathematical alternative on repeated fits and learning curves, best after sections 1&ndash;3. The full slide sequence was read; the video itself was not watched for this checkpoint. The slides use &ldquo;bias&rdquo; for the squared-bias contribution and sometimes compare fresh outcomes at fixed inputs; retain those conventions when following their derivations.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html">scikit-learn &mdash; Single estimator versus bagging</a>. A complete alternative simulation that makes prediction spread visible. Useful after the exact finite calculation. Its constructed function and measured outputs are separate from our airfoil study.</li>
      <li><a href="https://scikit-learn.org/stable/modules/learning_curve.html">scikit-learn &mdash; Learning and validation curves</a>. Practical parameter, shape and score conventions for the programs; the page carries its own title, &ldquo;Validation curves: plotting scores to evaluate models&rdquo;, and contains both subsections. Documentation reviewed at version 1.9.1; a diagnostic curve is not a direct estimate of the unknown population decomposition.</li>
    </ul></>}>
      <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html">The learning_curve API</a> &mdash; fit sizes, shuffle/cv/scoring semantics and the returned shapes used by the programs here. Its incremental mode checks for a partial-fit interface, not a warm-start flag.</li>
      <li><a href="https://homes.cs.washington.edu/~pedrod/papers/aaai00.pdf">Domingos &mdash; A Unified Bias-Variance Decomposition for Zero-One and Squared Loss</a> &mdash; primary loss-specific theory after section 8; its notation differs from this manuscript&rsquo;s signed bias.</li>
      <li><a href="https://arxiv.org/pdf/1912.07242">Nakkiran &mdash; More Data Can Hurt for Linear Regression</a> &mdash; a short deeper exposition behind section 9&rsquo;s explicitly asymptotic formulas. Read the setup and Claims 1&ndash;2 before interpreting its plot. Claim 1 is Nakkiran&rsquo;s; Claim 2 is credited there to Hastie and colleagues, 2019, so follow that attribution if you cite the second branch.</li>
      <li><a href="https://arxiv.org/html/1812.11118v2">Belkin et al. &mdash; Reconciling modern machine learning practice and the bias-variance trade-off</a> &mdash; broader model-wise evidence, including tree ensembles, with assumptions and experimental settings in the paper.</li>
      <li><a href={provenance.page}>UCI &mdash; Airfoil Self-Noise</a>, {provenance.authors}, <a href={provenance.doi}>{provenance.doi}</a>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a> &mdash; data description and license. This page serves <a href={provenance.file} download>the unchanged numeric file</a>, SHA-256 <Code>{provenance.sha256}</Code>, beside its <a href={provenance.attribution}>attribution</a>, with the exact development and fold protocol for offline reproduction.</li>
    </Sources>
    <Prose>The three-world sensor distribution, the &plusmn;noise polynomial worlds, the hidden-setting example, the fixed six-row design of section 7 and the practice matrices are explicitly <strong>constructed calculations</strong>, not measurements. The double-descent numbers are <strong>asymptotic theoretical calculations</strong> from the stated formula, not finite-sample simulations. The airfoil results are calculations on the identified real dataset under one declared row-level protocol, with no reserved row predicted or scored. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>,
};

export default biasVarianceContent;
