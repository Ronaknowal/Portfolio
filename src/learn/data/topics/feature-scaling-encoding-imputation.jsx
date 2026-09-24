import { H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { RulerLab, DonorLab, PipelineLab, TargetEncodingLab } from '../../components/lesson-labs/ScalingLabs.jsx';
import {
  RecordFigure, RulerFigure, CategoryFigure, BoundaryFigure, PipelineFigure, ComparisonFigure, RankFigure,
  EncodingFigure, PoolingFigure,
} from '../../components/lesson-labs/ScalingFigures.jsx';
import { scalingExamples } from '../scaling-examples.js';
import { comparison, fitted, provenance, scaleFixture, split } from '../scaling-data.js';
import { signedHash } from '../scaling-models.js';

const fixture = ['standard', 'minmax', 'robust'];
const four = value => value.toFixed(4);
const counted = Object.fromEntries(comparison.map(row => [row.method, row]));
/** The constructed signed map of section 8, hashed by the model layer rather
 * than written out by hand, so the table cannot drift from the arithmetic. */
const hashMap = { apple: { bucket: 0, sign: 1 }, pear: { bucket: 0, sign: -1 }, banana: { bucket: 1, sign: 1 } };
const hashBags = [{ apple: 3, pear: 1, banana: 2 }, { apple: 2, banana: 2 }];

const headings = [
  '1. Three different questions hiding inside “preprocessing”',
  '2. Scaling changes the meaning of “nearby”',
  '3. Encoding categories means choosing relationships',
  '4. A missing value is a question, not a zero',
  '5. Learn the preparation rule without looking ahead',
  '6. A complete experiment on real penguin measurements',
  '7. Deeper branch: change the shape, not only the ruler',
  '8. Deeper branch: high-cardinality categories and target information',
  '9. Deeper branch: completing data versus representing uncertainty',
  '10. Practice: explain the representation before choosing the function',
  '11. Readiness and the next connection'
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section><Prose><strong>Before running:</strong> {example.question}</Prose><RunnableExample example={example}>{children}</RunnableExample></section>;
}
function Practice({ title, question, hint, children }) {
  return <section className="sc-practice"><H3>{title}</H3><Prose>{question}</Prose>{hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}<details><summary>Show the explained solution</summary>{children}</details></section>;
}

const featureScalingContent = {
  title: 'Feature Scaling, Encoding & Imputation',
  readTime: '~50 min core reading · 50–80 min code and practice · deeper branches a second sitting',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot sc-lesson">
    <LessonIntro prerequisites={<>Python arrays, a table with rows and columns, averages, and the idea of predicting a label from examples. Each new formula is unpacked where it appears. The preceding <a href="/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml">NMF</a> lesson also depended on representation, and <a href="/learn/path/full-curriculum/k-nearest-neighbors-knn?module=classical-ml">k-nearest neighbours</a> supplies the distance rule this lesson keeps changing the ruler for.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A table is not yet a model input. Learn to turn one into a set of coordinates on purpose: choose a ruler for measurements and watch the nearest neighbour change, give categories a geometry instead of an invented order, tell a missing value apart from a measured zero, and keep the fit/transform boundary intact on 344 real penguin observations. Explore each investigation by changing its controls: calculations and diagrams update together, while algorithm traces let you step through the process.
    </LessonIntro>
    <Prose className="sc-route"><strong>First pass.</strong> Read sections 1 to 6, work through the ruler investigation in section 2, the donor investigation in section 4 and the fitted-pipeline investigation in section 6, and run the penguin program. Attempt practice questions 1 to 6. Sections 7, 8 and 9 are deeper branches on nonlinear representations, target encoding and missing-data uncertainty; they extend the core rather than being prerequisites for finishing it, and practices 7 to 9 belong with them. Allow roughly 50 minutes for the core reading and another 50 to 80 for calculation and code.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>A body mass of <Code>4000</Code> might mean grams, a category called <Code>female</Code> is not a smaller number than <Code>male</Code>, and a blank measurement is not a measured zero. Before fitting a model, we need a consistent way to represent what each entry means. This lesson follows one practical question: <strong>can measurements of a penguin help distinguish its species?</strong> The answer will depend partly on the model, and partly on the representation that decides which measurements the model can compare.</Prose>
    <Prose>Suppose one row contains a bill length, body mass, recorded sex, and species:</Prose>
    <LessonTable caption="One row of a table, before any preparation" headers={['Bill length', 'Body mass', 'Recorded sex', 'Species']} rows={[['40 mm', '4,000 g', 'female', 'Adelie']]} />
    <Prose>For our prediction task, species is the <strong>target</strong>: the answer available in the training examples. The other selected columns are <strong>features</strong>: information we intend to have when making a new prediction. A feature is not useful merely because it is present in the file; an identification number or a label recorded only after the answer is known can mislead the experiment.</Prose>
    <Prose>Three preparation operations answer different questions:</Prose>
    <LessonTable caption="Three operations, three questions" headers={['Operation', 'Question', 'Example']} rows={[
      ['Scaling', 'What numerical differences should have comparable influence?', 'Express a difference in body mass relative to its training spread'],
      ['Encoding', 'How should a category become a model-readable representation?', 'Give each recorded category its own indicator coordinate'],
      ['Imputation', 'What input should we supply where a measurement is absent?', 'Insert a training median and optionally retain a missingness indicator']
    ]} />
    <Prose>They can interact, but they are not interchangeable. Converting grams to kilograms changes units. Replacing a missing mass with 4,000 g makes an estimate. Encoding <Code>not_recorded</Code> as its own category records absence without inventing a biological category.</Prose>
    <Prose>The preceding <a href="/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml">NMF lesson</a> also depended on representation: its additive factors required nonnegative input. Subtracting a column mean can create negative numbers, so a preprocessing choice suitable for a distance model may violate an NMF input requirement. The useful question is always <strong>what information and geometry does the next model need?</strong></Prose>
    <RecordFigure />

    <H2>{headings[1]}</H2>
    <H3>A nearest-neighbor decision you can calculate</H3>
    <Prose>Consider a new measurement <Math>{'Q=(40,4000)'}</Math>, where the coordinates are bill length in millimeters and body mass in grams. Two possible neighbors are <Math>{'A=(41,4100)'}</Math> and <Math>{'B=(43,4001)'}</Math>. These are constructed measurements for arithmetic, not rows claimed to come from the real dataset.</Prose>
    <Prose>The usual squared Euclidean distance adds squared coordinate differences:</Prose>
    <MathBlock>{'\\begin{gathered}d^2(Q,A)=(41-40)^2\\\\[4pt] +(4100-4000)^2=10{,}001,\\end{gathered}'}</MathBlock>
    <MathBlock>{'d^2(Q,B)=3^2+1^2=10.'}</MathBlock>
    <Prose>The raw-number rule chooses B. A difference of 100 g overwhelms a difference of a few millimeters. The calculation is well defined, but its implicit relative importance came from the units we happened to write.</Prose>
    <Prose>Now decide, explicitly, that 1 mm and 100 g should each count as one unit of difference. Divide bill differences by 1 and mass differences by 100:</Prose>
    <MathBlock>{'\\begin{gathered}d_s^2(Q,A)=1^2+1^2=2,\\\\[4pt] d_s^2(Q,B)=3^2+0.01^2=9.0001.\\end{gathered}'}</MathBlock>
    <Prose>A is now nearer. Nothing moved in the physical world. We changed the ruler used by the model.</Prose>
    <Prose>For positive divisors <Math>{'s_j'}</Math>, this rule is</Prose>
    <MathBlock>{'d_s^2(x,z)=\\sum_j\\frac{(x_j-z_j)^2}{s_j^2}.'}</MathBlock>
    <Prose>So scaling a feature by <Math>{'1/s_j'}</Math> is equivalent to assigning its squared difference weight <Math>{'1/s_j^2'}</Math>. This is why scaling matters to nearest neighbors, k-means, and distance-based kernels. It also affects the meaning of coefficient penalties and can improve the numerical conditioning of gradient-based fitting. There is no theorem saying equal training variance is the best measure of relevance for every task.</Prose>
    <RulerLab />

    <H3>Learn a ruler from training data</H3>
    <Prose>Often we do not have a justified domain divisor. A common baseline is to fit a separate mean and standard deviation for each feature:</Prose>
    <MathBlock>{'\\begin{gathered}\\mu_j=\\frac1n\\sum_i x_{ij},\\\\[4pt] s_j=\\sqrt{\\frac1n\\sum_i(x_{ij}-\\mu_j)^2},\\\\[4pt] z_{ij}=\\frac{x_{ij}-\\mu_j}{s_j}.\\end{gathered}'}</MathBlock>
    <Prose>Here <Math>{'n'}</Math> counts training rows, <Math>{'i'}</Math> selects a row, and <Math>{'j'}</Math> selects a column. The divisor <Math>{'n'}</Math>, rather than <Math>{'n-1'}</Math>, matches <Code>StandardScaler</Code>&apos;s population-style training variance. The purpose is a transformation, not an unbiased estimate of an unknown population variance.</Prose>
    <Prose>Subtracting the same mean from two rows cancels in their difference. Centering therefore does not change their Euclidean separation; the division changes the relative feature weights. Centering still matters to other operations, including a model&apos;s intercept and ordinary PCA&apos;s variance interpretation.</Prose>
    <Prose>Standardization makes a nonconstant training column have mean zero and variance one. It <strong>does not turn a skewed distribution into a Gaussian distribution</strong>. It also does not establish the assumptions needed for a regression confidence interval: the distribution of a feature and the distribution of a model&apos;s errors are different objects.</Prose>
    <Prose>For a constant training column, the standard deviation is zero. A practical implementation uses a scale of one rather than dividing by zero. Its training values become zero after centering; a different future value need not become zero. Constant columns may be removed, but a value changing after training can also be a useful data-quality signal.</Prose>

    <H3>An outlier makes the choice visible</H3>
    <Prose>Fit three scalers to the five training values <Code>[1, 2, 3, 4, 100]</Code>:</Prose>
    <LessonTable caption="Five training values on three fitted rulers, plus a later value the fit never saw" headers={['Training value', 'Standard scaling', 'Min–max scaling', 'Median/IQR scaling']} rows={[
      ...[1, 2, 3, 4, 100].map((value, index) => [String(value), ...fixture.map(kind => four(scaleFixture[kind].values[index]))]),
      ['New value 150', ...fixture.map(kind => four(scaleFixture[kind].new150))]
    ]} />
    <Prose>Standard scaling uses mean {scaleFixture.statistics.standard.center} and standard deviation about {four(scaleFixture.statistics.standard.scale)}. <strong>Min–max scaling</strong> subtracts the training minimum and divides by the training range: here <Math>{'(x-1)/99'}</Math>. <strong>Robust scaling</strong> here subtracts the training median {scaleFixture.statistics.robust.center} and divides by the interquartile range <Math>{'Q_{75}-Q_{25}=4-2=2'}</Math>, using the stated linear percentile convention.</Prose>
    <Prose>The robust rule preserves visible separation among 1, 2, 3, and 4. It does not remove 100: that observation is still 48.5 transformed units away from the median. A new value 150 is outside the training min–max interval. Forcing it into <Code>[0,1]</Code> would be a separate clipping operation that discards how far outside the range it lies.</Prose>
    <RulerFigure />

    <H3>Column scaling is different from row normalization</H3>
    <Prose>For a row <Math>{'x'}</Math>, L2 normalization divides by its own length, <Math>{'\\|x\\|_2=\\sqrt{\\sum_jx_j^2}'}</Math>. Thus <Code>[3,4]</Code> and <Code>[6,8]</Code> both become <Code>[0.6,0.8]</Code>. Their direction survives; their overall size does not.</Prose>
    <Prose>This can be useful when comparing the composition of documents rather than their lengths, or a spectrum&apos;s shape rather than its overall intensity. It would be a questionable default if total intensity or total body size carries the signal. A zero vector has no mathematical direction; implementations generally leave it zero.</Prose>
    <Prose>Sparse matrices introduce another practical constraint. A table of mostly zero word counts can become dense if we subtract a nonzero column mean. <Code>StandardScaler(with_mean=False)</Code> or <Code>MaxAbsScaler</Code> can preserve zeros when appropriate. Sparse storage and row normalization are tools for a specific representation, not mandatory steps for every dataset.</Prose>
    <Prose>For an ideal threshold decision tree, strictly increasing transformations preserve the order of observed values and therefore the possible training partitions. This explains why unit scaling is usually much less important there. Finite precision, histogram binning, clipping, and transformations that merge values qualify that statement; it is not a promise that every implementation gives identical predictions under every transformation.</Prose>

    <H2>{headings[2]}</H2>
    <H3>One-hot coordinates avoid an invented ordering</H3>
    <Prose>Suppose a feature records <Code>red</Code>, <Code>green</Code>, or <Code>blue</Code>. Assigning numbers 0, 1, and 2 allows a numerical model to treat blue as twice green or to place green between the other two. A color label does not imply those relationships.</Prose>
    <Prose>One-hot encoding assigns one coordinate to each known category. Any two different rows in that table are distance <Math>{'\\sqrt2'}</Math> apart. This is a chosen geometry: all different categories have equal separation in that feature block. With several categorical columns, each block contributes to the total distance, so mixing one-hot blocks and scaled measurements still requires judgment about their relative influence.</Prose>
    <Prose>Dropping one column is sometimes useful for interpreting an unregularized linear model with an intercept. If all three columns are kept, their sum is the intercept column, so its coefficients are not uniquely identified. Predictions can still be fitted with a suitable numerical least-squares solver. Removing red also changes distances: red becomes <Code>[0,0]</Code>, one unit from green, while green and blue remain <Math>{'\\sqrt2'}</Math> apart. Regularization can likewise make the choice of reference coding affect fitted predictions. “Always drop the first category” is not a universal preparation rule.</Prose>
    <CategoryFigure />

    <H3>Missing, unknown, and rare are different states</H3>
    <Prose>A missing category means the value was not recorded. An unknown category means a value is present now but was absent from the fitted vocabulary. A rare category is known but has little training support.</Prose>
    <Prose>For example, a device model called <Code>sensor_C</Code> may be new at prediction time. <Code>OneHotEncoder(handle_unknown=&quot;ignore&quot;)</Code> represents an unknown value with zeros across that categorical block. This avoids an exception, but it does not teach the model how <Code>sensor_C</Code> behaves. With a dropped reference column, all-zero encoding can also coincide with the reference category. Alternative policies include rejecting invalid input or deliberately grouping infrequent or new values into a fitted bucket; the choice belongs to the application.</Prose>
    <Prose>In our penguin program, absent recorded sex becomes <Code>not_recorded</Code>, and that fitted category gets its own coordinate. We keep all known one-hot columns. At deployment, unexpected values should still be monitored even when prediction remains possible.</Prose>

    <H3>When order is real</H3>
    <Prose>For <Code>low</Code>, <Code>medium</Code>, and <Code>high</Code>, an ordinal encoding may express useful order. The values <Code>[0,1,2]</Code> additionally give equal numerical gaps to a linear or distance model. An ordinal scale alone does not justify those gaps. A threshold tree can use the ordering without multiplying by a coefficient, but a single threshold still divides a contiguous portion of the order; it cannot select an arbitrary subset of categories in one split.</Prose>
    <Prose>For very many categories, one-hot width can become expensive or weakly supported. The deeper branch explains target encoding and hashing. Some estimators also provide native categorical treatment. Check the estimator&apos;s actual interface and treatment of categories instead of assuming every model needs the same numeric encoding.</Prose>

    <H2>{headings[3]}</H2>
    <H3>Begin with a transparent estimate</H3>
    <Prose>Consider measured lengths <Code>[10, 20, missing, missing]</Code>. Median imputation fills both missing cells with 15. The filled table has mean 15, but the two missing measurements have not been discovered. Both <Code>[10,20,10,20]</Code>, with mean 15, and <Code>[10,20,50,60]</Code>, with mean 35, are compatible with the observed cells.</Prose>
    <Prose>This distinction matters even when prediction improves. An imputer supplies a usable model input; it does not certify that an estimated value was physically measured.</Prose>
    <Prose>A numeric <strong>missingness indicator</strong> adds a second feature that is 1 when the original value was absent and 0 otherwise. Now an actual 15 and an imputed 15 need not look identical to the model. This can help when absence contains predictive information, such as an optional measurement that technicians order selectively. If collection policy changes, that relationship can change too.</Prose>
    <Prose>Fit the replacement value on training rows, then reuse it for later rows. For entirely missing training columns, specify a stable output policy: dropping the column, keeping an explicit empty feature, or refusing an unusable input. Our program uses <Code>keep_empty_features=True</Code> so its column structure is retained, although its actual numeric training columns are not entirely missing.</Prose>

    <H3>Why the reason for missingness matters</H3>
    <Prose>Let <Math>{'R'}</Math> say whether a measurement was observed. <strong>MCAR</strong> means the missingness process is independent of the data values. <strong>MAR</strong> allows missingness to depend on observed information but, conditional on that information, not additionally on the missing values. <strong>MNAR</strong> allows a remaining dependence on those unseen values. A scale that fails above an unrecorded weight limit illustrates the last case.</Prose>
    <Prose>These describe a data-generating process, not a property a median imputer can establish from a blank cell. Observed data alone generally cannot distinguish MAR from every MNAR alternative. Understanding collection and performing sensitivity analysis matter. Adding an indicator does not, by itself, solve MNAR or recover valid scientific uncertainty. <a href="https://stefvanbuuren.name/fimd/sec-MCAR.html">Van Buuren&apos;s missingness introduction</a> develops these assumptions through concrete measurement examples.</Prose>

    <H3>Borrowing information from other rows</H3>
    <Prose>Nearest-neighbor imputation estimates a missing feature from similar rows that actually contain that feature. With incomplete rows, distance must be calculated using their available overlap.</Prose>
    <Prose>Use these three donor rows, whose columns are <Math>{'a,b,c'}</Math>, and query <Code>[2,12,missing]</Code>:</Prose>
    <LessonTable caption="Three donor rows with different gaps, and a query whose c is absent" headers={['Donor', 'a', 'b', 'c']} rows={[
      ['D1', '1', '10', '100'],
      ['D2', '3', 'missing', '300'],
      ['D3', 'missing', '14', '500']
    ]} />
    <Prose>The nan-aware distance used here takes the squared differences on shared observed coordinates and multiplies by <Math>{'m/q'}</Math>, where <Math>{'m=3'}</Math> is the total number of features and <Math>{'q'}</Math> the number jointly observed. The three squared distances are:</Prose>
    <MathBlock>{'\\begin{gathered}D1:\\tfrac32(1^2+2^2)=7.5,\\\\[4pt] D2:3(1^2)=3,\\\\[4pt] D3:3(2^2)=12.\\end{gathered}'}</MathBlock>
    <Prose>With two neighbors and uniform weights, D2 and D1 supply <Math>{'c=(300+100)/2=200'}</Math>. D2 is a valid donor even though another feature is absent. For a different missing target column, donor eligibility may differ. If no donor has a defined overlap distance, the implementation needs a fallback; scikit-learn uses the relevant training feature&apos;s average when available. Scaling of observed features still affects these distances. <a href="https://scikit-learn.org/stable/modules/impute.html#nearest-neighbors-imputation">The imputation guide</a> describes this feature-by-feature donor behavior.</Prose>
    <DonorLab />
    <Prose>Iterative imputation takes a different approach: initialize missing cells, fit one incomplete column from the others using rows where that column is observed, update its missing entries, then cycle through columns. This models relationships that a separate median ignores. It still depends on the chosen conditional models and on how missingness arose. The deeper uncertainty section explains why one completed table is different from multiple imputation.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>There are two distinct operations:</Prose>
    <Prose><strong>Fit:</strong> learn training medians, means, scales, and categories. <strong>Transform:</strong> apply those already learned values to a table.</Prose>
    <Prose>A new row should not redefine the ruler. If the training minimum and maximum are 1 and 100, the new value 150 maps to <Math>{'149/99'}</Math>; fitting min–max again on the new batch would create a different coordinate system.</Prose>
    <Prose>The same principle applies to evaluation. Split rows before learning preprocessing statistics. Fit preparation and the model using training rows. Apply the fitted preparation to the held-out rows, then count correct predictions. An estimate of future performance is compromised when the fitting procedure gets information it would not have at prediction time. Looking at held-out feature distributions to choose a transformation can also make an experiment adaptive, even without reading labels.</Prose>
    <BoundaryFigure />
    <Prose>A pipeline packages this sequence so the software can repeat it consistently. It does not repair a feature that already leaks the target, a split that puts repeated subjects on both sides, or a manually fitted transformer created before the split. Next, <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a> will repeat this fit/transform boundary inside each training/validation partition.</Prose>

    <H3>Build the saved state, then recognize it in the library</H3>
    <Prose>A fitted preparation rule is a small collection of learned values plus a function that reuses them. The program below builds that collection directly with NumPy, then fits the corresponding scikit-learn objects to the same four constructed training rows. Save it as <Code>fitted_preparation.py</Code> and run it with NumPy and scikit-learn; the pinned environment command in section 6 works for this program too. It needs no data download.</Prose>
    <Program example={scalingExamples.fittedStateParity}>
      <Prose><Code>fit_preparation</Code> computes medians first, fills only the missing training cells, and then learns means and population standard deviations from the filled table. It also saves a sorted category vocabulary. <Code>transform_preparation</Code> reads those values without fitting anything. Its two branches are the numeric and categorical branches that <Code>ColumnTransformer</Code> will join in the real-data program.</Prose>
    </Program>
    <LessonTable caption="The same fitted information in two implementations" headers={['Teaching state', 'Library state', 'How a later row uses it']} rows={[
      ['median', 'SimpleImputer.statistics_', 'Replace a missing numeric measurement'],
      ['mean and scale', 'StandardScaler.mean_ and scale_', 'Subtract the frozen center and divide by the frozen scale'],
      ['categories', 'OneHotEncoder.categories_[0]', 'Keep the same indicator columns and order'],
    ]} />
    <Prose>The first new row becomes <Code>[0.707107, 0, 2, 0, 0, 0]</Code>. Its missing second measurement becomes the training median and therefore has zero centered coordinate here. The third training feature was always 7: its scale is one, so the new value 9 becomes 2. The present but unknown category <Code>new</Code> gets three zeros; the second row&apos;s missing category uses the learned <Code>not_recorded</Code> coordinate instead. Changing the second row&apos;s measurement to 1000 changes its coordinate to about 4.949747 and leaves the first row and every fitted statistic unchanged.</Prose>
    <Prose>This is a dense teaching implementation for moderate numeric values and one string-valued category column. <Code>not_recorded</Code> is a reserved missing marker; passing it as a real category, an entirely missing numeric training column or a changed feature count raises an error. Floating-point overflow also raises an error instead of silently producing unusable fitted state. It does not reproduce sparse storage, weights, rare-category grouping or every near-constant floating-point rule in the library. Numeric transformation takes work proportional to rows × numeric columns. A dense one-hot result has rows × known categories cells: vectorizing that comparison does not remove its memory cost. Use the library&apos;s sparse output when that width is large.</Prose>
    <Prose>To customize the mechanism, replace only the fitted center/scale calculation with a training median and interquartile range, then compare against <Code>RobustScaler</Code> under the same percentile convention; keep imputation and vocabulary fixed. To add a missingness indicator, save the original mask before filling rather than trying to infer absence from the completed numbers. For routine model fitting use the library pipeline below, whose fitted attributes now have a concrete meaning. The <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html">imputer</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">scaler</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">encoder</a> API references describe their fuller contracts.</Prose>
    <Practice title="Implementation practice: retain what was missing" question={<>Extend <Code>transform_preparation</Code> with one missingness indicator for every numeric input column. Append those indicators after the existing output. Transform <Code>[15, 500, NaN]</Code> with category <Code>b</Code>; verify the original six coordinates, the new three indicators and unchanged fitted state.</>} hint={<>Capture <Code>np.isnan(numeric)</Code> before replacing NaNs. Your policy keeps all three indicators, including a column that had no missing training value.</>}>
      <Prose>Immediately after validating the input, save <Code>missing = np.isnan(numeric).astype(float)</Code>. Change the return to <Code>np.column_stack([scaled, indicators, missing])</Code>. The result is approximately <Code>[-0.707107, 1.414214, 0, 0, 1, 0, 0, 0, 1]</Code>. The missing third value becomes its training median 7, so its centered coordinate is zero; its indicator is still one. Assert that the first six coordinates equal the original function&apos;s output and that all saved arrays equal their pre-transform copies. This fixed-width policy differs from <Code>SimpleImputer(add_indicator=True)</Code>, which only includes features missing during fitting; use <Code>MissingIndicator(features=&quot;all&quot;)</Code> if you want the same three-indicator library contract.</Prose>
    </Practice>

    <H2>{headings[5]}</H2>
    <Prose>The supplied <a href="/learn-assets/feature-scaling/penguins.csv" download>penguins.csv</a> contains {provenance.rows} observations in the openly available Palmer Penguins dataset. We use four numeric measurements and recorded sex to predict Adelie, Chinstrap, or Gentoo. Bill length and depth are in millimeters, flipper length in millimeters, and body mass in grams. Two rows lack each of the four measurements; eleven lack recorded sex. The dataset&apos;s original research context is ecological measurement, and this small classification exercise does not establish performance on every future population or collection protocol. Dataset attribution and the original variables are described by the <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html">Palmer Penguins authors</a>.</Prose>
    <Prose>We reserve {split.heldOut} rows and fit on {split.training}, keeping roughly the same species proportions with a fixed stratified split. Here “accuracy” simply means correct species predictions divided by {split.heldOut}. The model takes the majority species among the five nearest training rows.</Prose>
    <Prose>Save the supplied CSV beside this program as <Code>penguins.csv</Code>. A compatible environment is Python 3.12 with NumPy 2.3.5, pandas 3.0.1, and scikit-learn 1.9.1; for a fresh environment install those packages once:</Prose>
    <CodeBlock language="bash">{'python -m pip install "numpy==2.3.5" "pandas==3.0.1" "scikit-learn==1.9.1"'}</CodeBlock>
    <Prose>The block below is executed verbatim against the same served CSV before this page is published, and the output shown underneath it is that run&apos;s own output, not a transcription.</Prose>
    <Program example={scalingExamples.penguinExperiment}>
      <Prose>The four numeric columns go through a median imputer and then a scaler; the single categorical column goes through a constant imputer and then a one-hot encoder. <Code>ColumnTransformer</Code> keeps those two routes apart and concatenates their outputs, and <Code>Pipeline</Code> makes <Code>fit</Code> mean “fit every step on these rows” and <Code>predict</Code> mean “apply every fitted step, then classify”. Only <Code>X.iloc[train]</Code> is ever passed to <Code>fit</Code>.</Prose>
    </Program>
    <Prose>The recorded results are:</Prose>
    <LessonTable caption="One split, one neighbour count, one categorical preparation; only the numeric ruler changes" headers={['Preparation for numeric features', 'Correct / held-out rows', 'Accuracy']} rows={comparison.map(row => [row.label, `${row.correct} / ${split.heldOut}`, four(row.accuracy)])} />
    <ComparisonFigure />
    <Prose>All four models use the same categorical preparation, split, and neighbor count. This controlled comparison makes the influence of a numerical ruler visible. One extra correct row does not establish that min–max or robust scaling is generally superior to standard scaling. We have now inspected this held-out set across several alternatives; selecting a procedure from these results would require a separate evaluation plan. The next lesson builds that plan.</Prose>
    <Prose>The standard model&apos;s fitted medians are <Code>[45.0,17.3,197.0,4000.0]</Code>. After imputation, its training means are approximately <Code>{`[${fitted.standard.center.map(value => four(value)).join(',')}]`}</Code>, and its scales <Code>{`[${fitted.standard.scale.map(value => four(value)).join(',')}]`}</Code>.</Prose>
    <Prose>The first held-out row, zero-based source row 309, is <Code>[51.0,18.8,203.0,4100.0,&quot;male&quot;]</Code>. It becomes:</Prose>
    <MathBlock>{'\\begin{gathered}[1.3091,\\;0.8773,\\;0.1645,\\\\ -0.1128,\\;0,\\;1,\\;0].\\end{gathered}'}</MathBlock>
    <Prose>The final three columns mean <Code>sex_female</Code>, <Code>sex_male</Code>, and <Code>sex_not_recorded</Code>. The negative mass coordinate says this mass is below the fitted mean; it does not mean a negative mass. Species, island, year, and row number were not included as features in this experiment.</Prose>
    <PipelineFigure />
    <PipelineLab />
    <Prose>As a practical extension, compare errors rather than only the score: standard scaling misclassified {split.heldOut - counted.standard.correct} rows here, whereas the raw model misclassified {split.heldOut - counted.raw.correct}. Inspect their actual measured values and nearest-neighbor contributions before inventing a story about why. Do not treat a species label as available input while exploring those errors.</Prose>
    <Checkpoint prompt="Set Investigation 3 to source row 309, clear its body mass, and trace the updated body-mass coordinate back to the frozen median, center and scale.">
      <Prose>The fitted median 4,000 g fills the absent cell, and the frozen centre and scale then give (4000 − 4190.988372093023) ÷ 806.6875829303058 ≈ −0.2368. Every other coordinate is unchanged, because nothing about the fit moved. The value is a numerical estimate supplied to the model, not a recovered observation.</Prose>
    </Checkpoint>

    <H2>{headings[6]}</H2>
    <Prose>Affine scaling maps <Math>{'x'}</Math> to <Math>{'(x-a)/b'}</Math>. It preserves relative gaps within a feature up to a common factor. Sometimes the modeling question calls for a nonlinear relationship instead.</Prose>
    <H3>Logs and power transforms</H3>
    <Prose>If a quantity varies multiplicatively, <Code>log</Code> can make ratios into differences: <Math>{'\\log(100)-\\log(10)=\\log(10)-\\log(1)'}</Math>. This is useful when equal multiplicative changes should have equal influence, as in a model of concentrations or elapsed times spanning several orders of magnitude. It does not justify taking the logarithm of arbitrary signed measurements.</Prose>
    <Prose>For strictly positive <Math>{'x'}</Math>, the Box–Cox family is</Prose>
    <MathBlock>{'g_\\lambda(x)=\\begin{cases}(x^\\lambda-1)/\\lambda,&\\lambda\\ne0,\\\\\\log x,&\\lambda=0.\\end{cases}'}</MathBlock>
    <Prose>Yeo–Johnson extends a related family to zero and negative values. For <Math>{'x\\ge0'}</Math>, naming the shifted value <Math>{'v=x+1'}</Math>:</Prose>
    <MathBlock>{'g_\\lambda(x)=\\begin{cases}\\dfrac{v^\\lambda-1}{\\lambda},&\\lambda\\ne0,\\\\[6pt]\\log v,&\\lambda=0,\\end{cases}'}</MathBlock>
    <Prose>and for <Math>{'x<0'}</Math>, naming the reflected value <Math>{'u=1-x'}</Math>:</Prose>
    <MathBlock>{'g_\\lambda(x)=\\begin{cases}-\\dfrac{u^{2-\\lambda}-1}{2-\\lambda},&\\lambda\\ne2,\\\\[6pt]-\\log u,&\\lambda=2.\\end{cases}'}</MathBlock>
    <Prose>Check the special cases rather than memorizing a name: <Math>{'\\lambda=1'}</Math> gives <Math>{'g(x)=x'}</Math> on both sides. At <Math>{'\\lambda=0'}</Math>, nonnegative inputs use <Code>log1p</Code>, but negative inputs use the negative quadratic branch. At <Math>{'\\lambda=2'}</Math>, negative inputs use a logarithm, not a square root. <Code>PowerTransformer</Code> fits its parameter per training feature by a likelihood criterion and standardizes afterward by default. Better marginal symmetry is a possible useful result; it is not a guarantee of jointly Gaussian features or correctly modeled residuals. The <a href="https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation">preprocessing guide&apos;s nonlinear section</a> gives the API and definitions.</Prose>

    <H3>Quantiles answer a different question</H3>
    <Prose>A quantile transform replaces a value by where it lies in the fitted distribution. For training values <Code>[1,2,3,4,100]</Code>, a simple illustrative rank coordinate <Math>{'(r-1)/(5-1)'}</Math> maps the sorted observations to <Code>[0,.25,.5,.75,1]</Code>. The large final gap becomes the same rank gap as the others. This rank calculation explains the idea; interpolation, ties, and endpoint handling in an actual transformer must be specified separately.</Prose>
    <Prose>Mapping those probabilities through an inverse normal CDF gives normal-quantile coordinates, with finite endpoint handling in software. Rank ordering is generally retained where the map is strictly increasing, but original numerical gaps are not. Ties and saturation outside the fitted range can merge values, making a complete inverse impossible. A “Gaussian-looking” histogram can therefore hide an important loss of magnitude information.</Prose>
    <RankFigure />

    <H3>Features can also express thresholds and interactions</H3>
    <Prose>Discretization assigns a value to a fitted interval: for thresholds 10 and 20, a quantity can be represented as <Code>below 10</Code>, <Code>10 to below 20</Code>, or <Code>20 and above</Code>. One-hot interval features let a linear model fit a stepwise response, at the cost of losing within-bin differences. Bin rules must be fitted on training data when they are data-dependent.</Prose>
    <Prose>A polynomial map can instead add <Math>{'x^2'}</Math> or <Math>{'x_1x_2'}</Math>. The model remains linear in its fitted coefficients while its response varies nonlinearly with the original variables. Spline bases provide smoother local building blocks. These are choices about which relationships a model can express, beyond simply fixing units. A custom deterministic transform, such as converting an angle to sine and cosine, can express that 359° and 1° are nearby. Keep the original unit and period explicit: <Code>sin</Code> and <Code>cos</Code> expect radians in NumPy.</Prose>
    <Prose>This circular representation is particularly useful for direction or time of day. It avoids declaring midnight far from 23:59, while preserving the fact that morning and evening can differ. It would be inappropriate for elapsed time, where completing a 24-hour cycle does not erase duration.</Prose>

    <H2>{headings[7]}</H2>
    <H3>Why a category average can accidentally contain the answer</H3>
    <Prose>Suppose many rows carry a product identifier, and the target is whether a product was returned. A smoothed target encoding represents category <Math>{'c'}</Math> by</Prose>
    <MathBlock>{'t_c=\\frac{\\sum_{i:x_i=c}y_i+\\alpha\\mu}{n_c+\\alpha},'}</MathBlock>
    <Prose>where <Math>{'n_c'}</Math> counts training examples of the category, <Math>{'\\mu'}</Math> is the relevant training target mean, and <Math>{'\\alpha\\ge0'}</Math> controls the pull toward that mean. An unseen category maps to <Math>{'\\mu'}</Math>. A category with one positive example and <Math>{'\\alpha=2,\\mu=.5'}</Math> maps to <Math>{'2/3'}</Math>, rather than an unqualified 1.</Prose>
    <Prose>The danger is easiest to see without smoothing: if a category occurs once, its training encoded value equals that row&apos;s target. The model is being given part of the answer. Smoothing reduces this direct influence but does not replace a separation rule.</Prose>
    <Prose><strong>Cross-fitting</strong> generates each training row&apos;s encoded feature using other training rows. Divide the outer training set into internal folds. For one fold, learn category sums, counts, <strong>and the prior mean</strong> from the other folds, then encode the held-out internal rows. Repeat until each training row has an out-of-fold representation. Once the downstream model is trained, a new external row is encoded from statistics fitted on the whole outer training set. Outer evaluation targets are never used.</Prose>
    <EncodingFigure />
    <Prose>Change only row 0&apos;s target from 1 to 0. Encodings for held-out fold 0 remain unchanged because their donor set did not change. Encodings for fold 1 become <Code>[2/9,2/9,5/9]</Code>. Notice that B changes even though no B target changed: the fold-specific prior changed. Computing one global prior before internal splitting would let a held-out target affect its own encoding through smoothing.</Prose>
    <Prose>The following complete teaching calculation reproduces the table and this contrast using NumPy from the earlier setup:</Prose>
    <Program example={scalingExamples.crossFitEncoding}>
      <Prose>Expected arrays are <Code>[.555556,.777778,.222222,.444444,.222222,.777778]</Code> and <Code>[.555556,.222222,.222222,.222222,.222222,.555556]</Code>. This block is executed and its output pinned by the same verifier as the penguin program, so the arrays above are that run&apos;s own output. The inner loop recomputes <Code>prior</Code> once per held-out fold, from the donors alone; that single line is what keeps a row&apos;s own target out of its own encoding.</Prose>
    </Program>
    <TargetEncodingLab />
    <Prose>For production use, <Code>TargetEncoder.fit_transform</Code> supplies internal cross-fitting, whereas <Code>fit(...).transform(...)</Code> does not produce the same training representation. In scikit-learn 1.9, <Code>cv</Code> can accept a splitter or iterable of splits; older examples using encoder-level <Code>shuffle</Code> and <Code>random_state</Code> are being deprecated. Group or time relationships require appropriate internal and outer splits. The default shuffled split cannot decide that for you. See the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html">current TargetEncoder API</a> and its <a href="https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html">worked cross-fitting example</a>.</Prose>

    <H3>Hashing trades a learned vocabulary for collisions</H3>
    <Prose>Feature hashing assigns each category or token to one of a fixed number of buckets. A signed version also assigns a deterministic sign and adds the signed feature value to that bucket. It is not merely a binary flag.</Prose>
    <Prose>For a constructed map, let <Code>apple</Code> and <Code>pear</Code> both use bucket 0, with signs +1 and −1, and <Code>banana</Code> use bucket 1 with sign +1. Counts <Code>apple:3, pear:1, banana:2</Code> produce <Code>[2,2]</Code>. Counts <Code>apple:2, banana:2</Code> produce the same vector. The collision makes the original dictionary impossible to recover from this vector alone.</Prose>
    <LessonTable caption="Two different bags of counts, one signed hash vector" headers={['Counts', 'bucket 0 = apple − pear', 'bucket 1 = banana', 'Vector']} rows={hashBags.map(bag => {
      const hashed = signedHash(bag, hashMap, 2);
      const part = token => hashed.detail.find(row => row.token === token);
      return [
        Object.entries(bag).map(([token, count]) => `${token}:${count}`).join(', '),
        `${part('apple')?.count ?? 0} − ${part('pear')?.count ?? 0} = ${hashed.vector[0]}`,
        String(hashed.vector[1]),
        `[${hashed.vector.join(',')}]`
      ];
    })} />
    <Prose>Hashing can bound memory and accept previously unseen names without growing a vocabulary. With suitable random-hash assumptions, signed hashing preserves inner products in expectation; that expectation is not a guarantee for every pair under a fixed small hash table. Increasing the number of buckets reduces typical collision pressure while increasing model width. The <a href="https://arxiv.org/pdf/0902.2206">original feature-hashing paper</a> derives this tradeoff.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>Iterative conditional prediction is useful for building model input, but one completed dataset treats its filled cells as if there were no uncertainty about them. A point estimate can look reasonable while standard errors are too small if uncertainty from missing data is ignored.</Prose>
    <Prose>Multiple imputation creates several plausible completed datasets under an explicit imputation model, fits the intended analysis to each, then combines <strong>analysis estimates</strong>, rather than averaging the filled tables first. Properly representing uncertainty requires more than running a deterministic imputer with a different random seed; the imputations must reflect the relevant conditional uncertainty and assumptions.</Prose>
    <Prose>For a scalar estimate, let <Math>{'\\hat\\theta_k'}</Math> be the result from completed dataset <Math>{'k'}</Math>, <Math>{'U_k'}</Math> its estimated variance, and <Math>{'m'}</Math> the number of completed datasets. Define</Prose>
    <MathBlock>{'\\begin{gathered}\\bar\\theta=\\frac1m\\sum_k\\hat\\theta_k,\\\\[4pt] \\bar U=\\frac1m\\sum_kU_k,\\\\[4pt] B=\\frac1{m-1}\\sum_k(\\hat\\theta_k-\\bar\\theta)^2.\\end{gathered}'}</MathBlock>
    <Prose>Rubin&apos;s pooling rule uses total variance <Math>{'T=\\bar U+(1+1/m)B'}</Math>. The first term captures uncertainty within each completed-data analysis; the second captures variation between plausible completions, with a finite-<Math>{'m'}</Math> adjustment. For estimates <Code>[9,10,11]</Code> and within-analysis variances <Code>[4,4,4]</Code>, the pooled estimate is 10, <Math>{'B=1'}</Math>, and <Math>{'T=4+4/3=16/3'}</Math>. Its standard error is about 2.309, larger than 2 from treating the completion as certain. Constructing intervals also requires the appropriate degrees-of-freedom calculation and imputation assumptions; this small arithmetic example is not an automatic validity certificate. The <a href="https://amices.org/mice/reference/pool.html">mice pooling documentation</a> explains the analysis-then-pool workflow and available small-sample treatment.</Prose>
    <PoolingFigure />
    <Prose><Code>IterativeImputer</Code> is an experimental scikit-learn estimator and returns a single completion per transform. Its optional posterior sampling can support repeated completions under the chosen estimator, but a sound multiple-imputation analysis also requires a compatible scientific model, diagnostics, and pooling. A predictive pipeline and a scientific missing-data analysis share tools while answering different questions.</Prose>

    <H2>{headings[9]}</H2>
    <Prose>Try each question before opening its hint or solution. The first six use only the core route.</Prose>
    <Practice title="1. A changed ruler" question="Query [0,0] has candidates A [2,60] and B [5,10]. Which is nearest under raw squared distance? Which is nearest with divisors [1,30]?" hint="Calculate one contribution per feature; the divisor belongs inside the square.">
      <Prose>Raw distances squared are 3,604 and 125, so B wins. Scaled distances squared are <Math>{'4+4=8'}</Math> and <Math>{'25+1/9=25.111\\ldots'}</Math>, so A wins. The input observations are unchanged; relative feature weighting changed. Investigation 1 shows the same mechanism, but its fields are bounded to plausible penguin measurements, so these particular coordinates cannot be typed there; reproduce the effect with its own numbers instead.</Prose>
    </Practice>
    <Practice title="2. Fit once, transform later" question="A training column is [2,4,6]. Find its mean, population-style standard deviation, standardized value for a new 8, and min–max value for that 8. Should adding 8 to the later batch change the saved training statistics?" hint="The training squared deviations are 4, 0, and 4.">
      <Prose>The mean is 4 and standard deviation <Math>{'\\sqrt{8/3}'}</Math>. The standardized new value is <Math>{'4/\\sqrt{8/3}=\\sqrt6\\approx2.4495'}</Math>. Min–max gives <Math>{'(8-2)/(6-2)=1.5'}</Math>. Transformation reuses the fitted statistics; refitting on later inputs would create a different map.</Prose>
    </Practice>
    <Practice title="3. What did normalization discard?" question="A spectrum [2,1,2] and another [6,3,6] are L2-normalized. What are the results? Would this be appropriate if total emitted energy is the prediction signal?">
      <Prose>The lengths are 3 and 9, so both map to <Code>[2/3,1/3,2/3]</Code>. Relative shape remains, but the threefold intensity difference disappears. If total energy matters, preserve a magnitude feature or choose another representation rather than discarding it blindly.</Prose>
    </Practice>
    <Practice title="4. A category that did not exist during fitting" question="A full one-hot vocabulary has small, medium, and large, and an unknown value uses an all-zero block. Compare unknown-to-small distance with small-to-medium distance. What application decision is hidden behind accepting the unknown value?">
      <Prose>The distances are 1 and <Math>{'\\sqrt2'}</Math>. Ignoring unknown categories creates a representation with a specific geometry; it is not neutral. The application must decide whether a new value is valid, should trigger a review or fallback, or belongs in a deliberately learned other-category group.</Prose>
    </Practice>
    <Practice title="5. Changed donors" question="In the imputation table, change D2's c to missing. With two neighbors, what value replaces the query's missing c? What if D1's c changes from 100 to 140 while the other original cells remain unchanged?" hint="First decide who can donate the target feature, then use the distances on the query's observed coordinates.">
      <Prose>With D2 ineligible, D1 and D3 supply <Math>{'(100+500)/2=300'}</Math>. In the separate second change, the original selected donors D2 and D1 remain nearest, so the estimate becomes <Math>{'(300+140)/2=220'}</Math>. Changing a value in the query&apos;s missing target column does not itself enter these overlap distances. Both edits are available in Investigation 2.</Prose>
    </Practice>
    <Practice title="6. Explain a transformed real record" question="For the fitted standard penguin pipeline, keep the first held-out record unchanged except set its body mass to missing. Predict that output coordinate. Does this mean the animal's true mass was 4,000 g?" hint="First apply the saved median, then the saved mean and scale.">
      <Prose>The coordinate becomes <Math>{'(4000-4190.9883721)/806.6875829\\approx-0.2368'}</Math>. This is a numerical estimate passed to the model. It is not a recovered observation. Other coordinate values and the fitted statistics remain unchanged.</Prose>
    </Practice>
    <Practice title="7. Which part of target encoding changed? — deeper" question="In the six-row example, change row 3's target from 0 to 1. Calculate row 0's new encoding. Does row 3's own cross-fitted encoding change?">
      <Prose>Row 0&apos;s donors are rows 1, 3, and 5, now with prior <Math>{'2/3'}</Math>. Its A donor is still positive, so its value becomes <Math>{'7/9'}</Math>. Row 3 belongs to held-out fold 1, whose donor targets are unchanged, so its own value remains <Math>{'4/9'}</Math>. The first change passes through the prior; the null demonstrates excluding one&apos;s own target.</Prose>
    </Practice>
    <Practice title="8. Identity is a useful transform check — deeper" question="Use the Yeo–Johnson formula at λ = 1 on 3 and −3. Why is a generic claim that “negative inputs use a square root” wrong?">
      <Prose>For 3, <Math>{'((3+1)^1-1)/1=3'}</Math>. For −3, <Math>{'-((1+3)^1-1)/1=-3'}</Math>. Both branches depend on <Math>{'\\lambda'}</Math>; the negative exponent is <Math>{'2-\\lambda'}</Math>, with a logarithmic limit at <Math>{'\\lambda=2'}</Math>, not a fixed square root.</Prose>
    </Practice>
    <Practice title="9. Pool estimates, not completed tables — deeper" question="Four completed-data analyses yield estimates [8,10,10,12], each with variance 1. Find the pooled estimate and total variance using the stated rule.">
      <Prose>The average is 10. Squared deviations sum to 8, so <Math>{'B=8/3'}</Math>. Then <Math>{'T=1+(1+1/4)(8/3)=13/3\\approx4.3333'}</Math>. Its standard error is <Math>{'\\sqrt{13/3}\\approx2.0817'}</Math>. Between-completion disagreement contributes substantial uncertainty; validity still depends on how the completions and analyses were constructed.</Prose>
    </Practice>

    <H2>{headings[10]}</H2>
    <Prose>You are ready to continue when you can trace one record through a fitted imputer, scaler, and encoder; explain how scaling changes a distance; distinguish a missing value from an unknown category; and keep the fit/transform boundary intact for a new row. Those are core skills, independent of whether you have finished the deeper branches.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Show that a chosen divisor, not the data, decided which neighbour was nearer', 'Section 2, Investigation 1, practice 1'],
      ['Fit a ruler on a training column and apply it to a value outside the fitted range', 'Section 2, Figure 2, practice 2'],
      ['Give categories a geometry and say what dropping a reference column changes', 'Section 3, Figure 3, practice 4'],
      ['Decide which incomplete rows may donate a missing measurement, and why', 'Section 4, Investigation 2, practice 5'],
      ['Transform an edited held-out record without moving a single fitted statistic', 'Sections 5 and 6, Investigation 3, practice 6'],
      ['Explain how a row’s own target can reach its own encoding, and how cross-fitting stops it', 'Section 8, Investigation 4, practice 7']
    ]} />
    <Prose>The next topic is <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a>. Here, we held a split fixed to inspect a representation. Next, we will decide how to compare multiple procedures without repeatedly treating the same observations as fresh evidence. Later regularization makes the connection between feature units and coefficient penalties explicit; feature selection asks which input information should remain at all.</Prose>
    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html">Compare scalers on data with outliers</a>: a visual alternative with full-range and magnified views of actual housing data. Compare robust scaling with a quantile transform and explain what happens to an identified extreme observation. The example&apos;s code and figure descriptions were inspected; its runtime is not our benchmark.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html">Target Encoder&apos;s Internal Cross Fitting</a>: a worked code-and-results alternative showing why near-unique categories can overfit without cross-fitting. Follow where <Code>fit_transform</Code> occurs inside its pipeline.</li>
      <li><a href="https://stefvanbuuren.name/fimd/sec-MCAR.html">Flexible Imputation of Missing Data: missingness concepts</a> and the <a href="https://amices.org/mice/reference/pool.html">mice pooling reference</a>: a conceptual measurement-based reading and a concrete analysis workflow for the deeper uncertainty branch. Do not substitute a prediction score for evidence that an inferential missingness assumption is valid.</li>
    </ul></>}>
      <li><a href="https://scikit-learn.org/stable/modules/preprocessing.html">Scikit-learn preprocessing guide</a>: the technical reference for current scaler, nonlinear transform, categorical, binning, polynomial and custom-transform behavior. Read the subsection matching a representation you can already explain, then inspect its API rather than treating the whole page as a required first pass.</li>
      <li><a href="https://scikit-learn.org/stable/modules/impute.html">Imputation guide</a>: current simple, iterative, neighbor and indicator behavior, including entirely missing columns. Use it to check a proposed imputer&apos;s exact contract.</li>
      <li><a href="https://allisonhorst.github.io/palmerpenguins/">Palmer Penguins dataset and project</a>: measurements, dataset context, and CC0 availability, by Allison Horst, Alison Hill, and Kristen Gorman, using Palmer Station penguin observations collected by Gorman and colleagues. The <a href="/learn-assets/feature-scaling/penguins.csv" download>CSV this page serves</a> is the unchanged {provenance.bytes.toLocaleString('en-US')}-byte public file, SHA-256 <Code>{provenance.sha256}</Code>, with all {provenance.rows} rows in their original order and missing values left as <Code>NA</Code>.</li>
      <li><a href="https://arxiv.org/pdf/0902.2206">Weinberger and colleagues, Feature Hashing for Large Scale Multitask Learning</a>: primary treatment of signed hashing and its inner-product analysis; useful when a fixed-memory representation matters more than recovering an explicit vocabulary.</li>
    </Sources>
    <Prose>The Q/A/B measurements, the five-value column, the red/green/blue vocabulary, the three donor rows, the six-row encoding table and the three pooled analyses are constructed fixtures with declared values. The penguin results are calculations on the identified real dataset under one fixed stratified split, with every fitted statistic learned from the {split.training} training rows alone. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>
};

export default featureScalingContent;
