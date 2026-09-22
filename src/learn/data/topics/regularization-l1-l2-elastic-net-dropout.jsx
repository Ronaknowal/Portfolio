import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { ThresholdLab, CoordinateLab, AirfoilTraceLab, DropoutLab } from '../../components/lesson-labs/RegularizationLabs.jsx';
import {
  ObjectiveSumFigure, ConstraintGeometryFigure, DuplicateFigure, PathFigure,
  DirectionsFigure, FactorFigure, SmoothnessFigure, ComplexityFigure,
} from '../../components/lesson-labs/RegularizationFigures.jsx';
import { regularizationExamples } from '../regularization-examples.js';
import { candidates, baselineMeanMse, olsMeanMse, provenance, inferenceFixture } from '../regularization-data.js';
import { criteria, scalarSolution, smoothnessComparison, factorOptimum, ridgeFilter } from '../regularization-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');
const scalarRow = (z, strength) => [
  num(z),
  num(scalarSolution(z, strength, 0).coefficient),
  num(scalarSolution(z, strength, 1).coefficient),
  z === 3 ? '5/3' : num(scalarSolution(z, strength, 0.5).coefficient),
];
const mse = (family, strength) => candidates.find(row => row[0] === family && row[1] === strength)[2].toFixed(6);
const smaller = criteria(-150, 3, 100);
const larger = criteria(-146, 5, 100);
const shifted = smoothnessComparison([3, 5, 3], 1);

const headings = [
  '1. What preference are we adding?',
  '2. Why L2 shrinks and L1 can select',
  '3. From one coefficient to a complete fit',
  '4. Correlated features: prediction and attribution are different questions',
  '5. A real comparison: predicting airfoil sound measurements',
  '6. Dropout: change what the learner sees during training',
  '7. Deeper branch: directions, paths and computation',
  '8. Deeper branch: a penalty encodes a representation',
  '9. Deeper branch: AIC, BIC and description length',
  '10. Practice: change the problem, then explain the answer',
  '11. Readiness and the next question',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section><Prose><strong>Before running:</strong> {example.question}</Prose><RunnableExample example={example}>{children}</RunnableExample></section>;
}
function Practice({ title, question, hint, children }) {
  return <section className="rg-practice"><H3>{title}</H3><Prose>{question}</Prose>{hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}<details><summary>Show the explained solution</summary>{children}</details></section>;
}

const regularizationContent = {
  title: 'Regularization (L1, L2, Elastic Net, Dropout)',
  readTime: '~65 min first pass · ~120 min complete read + 70–110 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot rg-lesson">
    <LessonIntro prerequisites={<>A weighted sum, squared error, means, and the fit/validation distinction from the preceding <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a> lesson. <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">Feature Scaling, Encoding &amp; Imputation</a> supplies the fit/transform mechanics used inside every fold here. The extra notation is introduced as it becomes useful.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A model can explain the observations you collected in several different ways. Learn to state the preference you are adding, calculate a soft-threshold and a shrinkage by hand, fit four constructed rows one coordinate at a time, separate prediction from coefficient attribution on duplicate sensors, read a real regularization path on 1,503 aeroacoustic measurements including its unhelpful end, and derive exactly why a mean-preserving dropout mask still raises the expected loss. Explore each investigation by changing its controls: calculations and diagrams update together, while algorithm traces let you step through the process.
    </LessonIntro>
    <Prose className="rg-route"><strong>First pass.</strong> Follow sections 1–6 to understand the objectives, work a tiny fit, compare real observations and explain dropout&rsquo;s train/evaluation distinction. Try practice 1–6. Sections 7–9 deepen the connection to linear algebra, Bayesian priors, parameterization and model-selection criteria; you can return to those branches after the core route.</Prose>

    <Prose>A model can explain the observations you collected in several different ways. Some explanations depend on large, finely balanced coefficients: increase one contribution and almost cancel it with another. A small change in the measurements may then change those coefficients dramatically. Other explanations use many weak contributions that might be real signal, or might just fit the particular sample.</Prose>
    <Prose><strong>Regularization adds a preference to fitting.</strong> It can favor smaller coefficients, fewer nonzero coefficients, smoother neighboring values, or predictions that remain useful when some intermediate inputs are randomly withheld. The preference changes the problem being solved. Whether it improves future predictions is something to assess using the data boundaries from the previous lesson.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>Suppose a prediction is</Prose>
    <MathBlock>{'\\hat y_i=b+x_{i1}w_1+\\cdots+x_{id}w_d.'}</MathBlock>
    <Prose>There are <Math>{'n'}</Math> observed cases and <Math>{'d'}</Math> input features. The coefficient vector <Math>{'w'}</Math> tells us how each feature contributes; the intercept <Math>{'b'}</Math> provides a common offset. Ordinary least squares chooses these values to minimize the sum of squared residuals, where a residual is observed minus predicted value.</Prose>
    <Prose>We will use <strong>half the mean squared error</strong>, plus a penalty:</Prose>
    <MathBlock>{'\\begin{gathered}J(b,w)=\\frac1{2n}\\sum_{i=1}^{n}(y_i-b-x_i^\\top w)^2\\\\[4pt] +\\lambda\\left[\\rho\\sum_j|w_j|+\\frac{1-\\rho}{2}\\sum_jw_j^2\\right].\\end{gathered}'}</MathBlock>
    <Prose>The factor one-half simplifies derivatives; averaging by <Math>{'n'}</Math> keeps the data term on a per-case scale. <Math>{'\\lambda'}</Math> is the nonnegative penalty strength. The mixing fraction <Math>{'\\rho'}</Math> lies between zero and one. The intercept is excluded from this penalty. This convention governs the main regression calculations; a deeper denoising example will explicitly declare its unaveraged objective:</Prose>
    <LessonTable caption="Three names, one objective" headers={['Choice', 'Penalty in this convention', 'What it encourages']} rows={[
      ['Ridge, or L2', <><Math>{'\\lambda\\sum_j w_j^2/2'}</Math>, using ρ=0</>, 'Smaller coefficient norm; stable treatment of weakly determined directions'],
      ['Lasso, or L1', <><Math>{'\\lambda\\sum_j|w_j|'}</Math>, using ρ=1</>, 'Shrinkage with the possibility of exact zero coefficients'],
      ['Elastic net', 'Both terms, using 0<ρ<1', 'Sparse fits with an additional strictly convex preference'],
    ]} />
    <Prose>The vertical bars mean absolute value: both +3 and −3 contribute 3 to an L1 penalty and 9 to a squared L2 penalty. “L1” and “L2” name norms, or ways to measure a vector&rsquo;s size. Neither name identifies which observations are allowed to influence fitting; preprocessing and penalty selection still belong inside the training/validation protocol.</Prose>
    <Prose>A numerical comparison makes the tradeoff visible. In a one-coefficient problem, suppose the data term is <Math>{'\\frac12(w-3)^2'}</Math>. With ridge strength λ=1, w=3 gives perfect data fit but total cost 4.5. At w=1.5, data cost is 1.125 and penalty is 1.125, totaling 2.25. A worse fit to these observed data is preferable under the new objective. That does <strong>not</strong> by itself establish that w=1.5 predicts future cases better.</Prose>
    <ObjectiveSumFigure />

    <H3>Units change the meaning of a penalty</H3>
    <Prose>If a feature measured in meters is replaced by the same values in centimeters, its numeric values multiply by 100. Dividing its coefficient by 100 preserves every prediction, but its L1 cost divides by 100 and its squared L2 cost divides by 10,000. A raw coefficient penalty therefore favors that larger numerical feature scale for the same predictive contribution.</Prose>
    <Prose>Standardizing a numeric column using its training mean and standard deviation is one useful way to make the preference refer to a one-standard-deviation change. It is not an instruction to erase meaningful physical units in every problem. A domain-specific penalty can deliberately assign different costs to different coefficients. Sparse indicator columns also need a considered convention: scaling a rare binary feature to unit variance changes the cost of its effect.</Prose>
    <Prose>The preceding <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">Feature Scaling, Encoding &amp; Imputation</a> supplies the fit/transform mechanics. Here the important question is: <strong>what change in the original input does one unit of this coefficient represent?</strong> Changing target units also changes the numerical balance between squared error, L1 and L2; a λ value is meaningful only with its objective and scaling convention.</Prose>
    <Prose>The unpenalized intercept has a useful consequence. If every training target increases by seven, the fitted intercept can increase by seven while slopes stay the same. There is no reason to shrink that common offset toward zero merely because the measurement origin changed. Penalizing an intercept can be a deliberate modeling choice, but it should not happen accidentally because a column of ones was included in <Math>{'w'}</Math>.</Prose>

    <H2>{headings[1]}</H2>
    <H3>One coefficient, with the arithmetic exposed</H3>
    <Prose>Start with</Prose>
    <MathBlock>{'\\frac12(w-z)^2+\\frac\\lambda2w^2.'}</MathBlock>
    <Prose>Here <Math>{'z'}</Math> is the coefficient preferred by the data alone in this simple normalized problem. The derivative is <Math>{'(w-z)+\\lambda w'}</Math>. Setting it to zero gives</Prose>
    <MathBlock>{'w_{\\text{ridge}}=\\frac{z}{1+\\lambda}.'}</MathBlock>
    <Prose>At z=3 and λ=1, the answer is 1.5. At z=0.4, it is 0.2. Both are pulled toward zero. In this scalar example a nonzero <Math>{'z'}</Math> does not become exactly zero at finite λ. In a multi-feature problem, an individual ridge coefficient can equal zero because of the data geometry; ridge simply has no threshold region that systematically creates sparsity.</Prose>
    <Prose>Replace the squared penalty by <Math>{'\\lambda|w|'}</Math>. For positive <Math>{'w'}</Math>, the derivative of the objective is <Math>{'w-z+\\lambda'}</Math>, giving w=z−λ if that answer is positive. For negative <Math>{'w'}</Math>, the derivative is <Math>{'w-z-\\lambda'}</Math>, giving w=z+λ if negative. If neither case is valid, the minimum is at zero. Combining the cases:</Prose>
    <MathBlock>{'S(z,\\lambda)=\\operatorname{sign}(z)\\max(|z|-\\lambda,0).'}</MathBlock>
    <Prose>This is <strong>soft-thresholding</strong>. It removes the central interval [−λ, λ] and shrinks surviving values toward zero. It differs from hard thresholding, which would keep a surviving <Math>{'z'}</Math> unchanged.</Prose>
    <LessonTable caption="Three data preferences at λ = 1, under three penalties" headers={['Data preference z, with λ=1', 'Ridge', 'Lasso', 'Elastic net, ρ=0.5']} rows={[
      scalarRow(3, 1), scalarRow(0.4, 1), scalarRow(-2, 1),
    ]} />
    <Prose>For elastic net the same calculation gives <Math>{'S(z,\\lambda\\rho)/[1+\\lambda(1-\\rho)]'}</Math>. Its L1 part sets the threshold and its L2 part changes the denominator. Elastic net is a family of preferences, not a guarantee that its answer or prediction error lies between those of separately tuned ridge and lasso.</Prose>
    <ThresholdLab />

    <H3>The two-dimensional geometry, without an exaggerated claim</H3>
    <Prose>A constrained version asks for the smallest data loss among coefficients inside a fixed penalty budget. In two dimensions, an L2-norm budget forms a disk and an L1-norm budget forms a diamond. The first data-loss contour that meets the allowed region identifies a constrained optimum. Diamond corners and faces make exact zero coordinates possible over a range of data preferences.</Prose>
    <Prose>They do not force every optimum to a corner. With independent normalized coordinates, z=(3,0.4) and lasso λ=0.1, the answer is (2.9,0.3): both coordinates survive. At λ=1 the answer becomes (2,0). The elastic-net budget still has nonsmooth behavior where a coordinate crosses zero; its curved edges do not remove that threshold.</Prose>
    <ConstraintGeometryFigure />

    <H2>{headings[2]}</H2>
    <H3>Separate the offset, then write the matrix equation</H3>
    <Prose>Let <Math>{'X'}</Math> contain the <Math>{'n'}</Math> rows of features, and let <Math>{'y'}</Math> contain the <Math>{'n'}</Math> targets. Subtract each training feature mean and the training target mean. Write the centered arrays as <Math>{'Z'}</Math> and <Math>{'t'}</Math>. After fitting slopes <Math>{'w'}</Math>, recover the intercept as <Math>{'b=\\bar y-\\bar x^\\top w'}</Math>.</Prose>
    <Prose>For ridge, differentiation gives</Prose>
    <MathBlock>{'(Z^\\top Z+n\\lambda I)w=Z^\\top t.'}</MathBlock>
    <Prose><Math>{'I'}</Math> is an identity matrix. The <Math>{'n\\lambda'}</Math> appears because our data term is averaged by <Math>{'n'}</Math>. For λ&gt;0, any nonzero vector <Math>{'v'}</Math> satisfies</Prose>
    <MathBlock>{'\\begin{gathered}v^\\top(Z^\\top Z+n\\lambda I)v\\\\[4pt] =\\|Zv\\|^2+n\\lambda\\|v\\|^2>0.\\end{gathered}'}</MathBlock>
    <Prose>Thus the matrix is positive definite and the centered ridge slopes are unique, even if columns repeat or d&gt;n. In code, solve the linear system instead of explicitly forming its inverse. For poorly conditioned problems, an SVD-based solver avoids forming the squared condition number of the normal equations; positive λ helps mathematically but does not excuse careless numerics.</Prose>

    <H3>Coordinate descent: let each feature explain the remaining residual</H3>
    <Prose>Lasso and elastic net can update one coefficient at a time. Temporarily remove feature <Math>{'j'}</Math>&rsquo;s current contribution from the prediction. The partial residual is</Prose>
    <MathBlock>{'r_j=t-Zw+Z_{:,j}w_j.'}</MathBlock>
    <Prose>Define a data curvature and a residual association:</Prose>
    <MathBlock>{'\\begin{gathered}a_j=\\frac{Z_{:,j}^\\top Z_{:,j}}n,\\\\[4pt] c_j=\\frac{Z_{:,j}^\\top r_j}n.\\end{gathered}'}</MathBlock>
    <Prose>The exact coordinate minimizer is</Prose>
    <MathBlock>{'w_j\\leftarrow\\frac{S(c_j,\\lambda\\rho)}{a_j+\\lambda(1-\\rho)}.'}</MathBlock>
    <Prose>Each update uses the current values of the other coefficients. A full pass through all coordinates is a <strong>sweep</strong>. With correlated columns, changing one coefficient changes the residual available to the next, so several sweeps can be needed. A coefficient that is zero during one sweep can become nonzero later; the partial residual can change.</Prose>
    <Prose>For pure lasso, a zero coefficient at an optimum permits absolute residual association <Math>{'|Z_{:,j}^\\top(t-Zw)/n|'}</Math> at most λ. A nonzero coefficient requires equality to λ, with the association&rsquo;s sign matching the coefficient. Strict inequality therefore forces zero; equality alone can occur with a zero or nonzero coefficient. These are optimality conditions involving the <strong>current full residual</strong>, not a one-time test of the ordinary-least-squares coefficient or proof that a feature is irrelevant to the world.</Prose>

    <H3>Four rows we can calculate by hand</H3>
    <Prose>Use the following constructed input:</Prose>
    <LessonTable caption="Four constructed rows with orthogonal centered columns" headers={['Row', 'First feature', 'Second feature', 'Target']} rows={[
      ['0', '1', '1', '3.4'], ['1', '1', '−1', '2.6'], ['2', '−1', '1', '−2.6'], ['3', '−1', '−1', '−3.4'],
    ]} />
    <Prose>The means are zero, <Math>{'Z^\\top Z/n=I'}</Math> and <Math>{'Z^\\top t/n=(3,0.4)'}</Math>. These columns are orthogonal: after accounting for one, the residual association of the other stays the same. At λ=1, one coordinate sweep therefore gives ridge (1.5,0.2), lasso (2,0), or elastic net with ρ=0.5 equal to (5/3,0). Their total objective values are 2.29, 2.58 and approximately 2.496667, respectively. These costs come from <strong>different penalty functions</strong>, so the smallest of those numbers is not a valid way to choose which family predicts best.</Prose>
    <CoordinateLab />
    <Checkpoint prompt="Changing row 0's target from 3.4 to 7.4 gives lasso slopes (3, 0.4) and intercept 1 at λ = 1. Adding seven to every original target instead changes only the intercept, to seven. Why do two edits of the same size behave so differently?">
      <Prose>The first edit is not a common shift: it raises one row only, which changes the centered target and therefore the residual association of both columns. The second edit adds the same constant to every row, so every centered target is unchanged and only the unpenalized intercept absorbs it. Both are reproducible in the investigation above through its two presets.</Prose>
    </Checkpoint>

    <H3>A complete small implementation</H3>
    <Prose>The following NumPy program implements the common objective, including an unpenalized intercept and residual updates. It checks the optimality conditions rather than stopping just because the last coefficient movement looks small. For nonzero <Math>{'w_j'}</Math>, the smooth gradient plus <Math>{'\\lambda\\rho\\operatorname{sign}(w_j)'}</Math> should be zero. For a zero coordinate, the smooth gradient may lie anywhere within [−λρ, λρ]. The maximum violation is reported as a residual, not as a test-set error.</Prose>
    <Prose>Use Python with NumPy 2.3.5 (<Code>python -m pip install numpy==2.3.5</Code> in a new environment). Save as <Code>coordinate_regularization.py</Code> and run it.</Prose>
    <Program example={regularizationExamples.coordinateFit}>
      <Prose>A constant centered feature has a=0 and no residual association. Setting its coefficient to zero is appropriate; when its objective is completely flat it is a declared representative solution. This instructional solver does not implement sparse storage, screening or optimized paths. Current library implementations use additional numerical machinery and diagnostics; the <a href="https://scikit-learn.org/stable/modules/linear_model.html#lasso">scikit-learn linear-model guide</a> documents coordinate descent and its optimality-gap approach.</Prose>
      <Prose>The third line is padded by NumPy&rsquo;s own array formatting, which aligns the two entries; the values are 1.666667 and exactly 0.</Prose>
    </Program>

    <H2>{headings[3]}</H2>
    <Prose>Imagine two sensors report exactly the same centered value <Math>{'x'}</Math>. The model&rsquo;s prediction depends only on the sum <Math>{'s=w_1+w_2'}</Math>, because <Math>{'xw_1+xw_2=xs'}</Math>. If the target is 2x, no amount of fitting those duplicate measurements can reveal which sensor “caused” the signal.</Prose>
    <Prose>Use two rows x=−1 and x=1, with targets −2 and 2. Our data loss is <Math>{'\\frac12(s-2)^2'}</Math>. With lasso λ=1, the optimal sum is s=1. Every nonnegative pair with that sum has the same data cost and the same L1 penalty: (1,0), (0.5,0.5), and (0,1) all minimize the objective. A coordinate solver starting from zero may return (1,0) because it visits the first column first. A different order can return the other endpoint. That algorithmic choice is not evidence about the sensors&rsquo; scientific importance.</Prose>
    <Prose>The squared L2 penalty prefers balanced coefficients because, for fixed <Math>{'s'}</Math>,</Prose>
    <MathBlock>{'w_1^2+w_2^2=\\frac{s^2}{2}+\\frac{(w_1-w_2)^2}{2}.'}</MathBlock>
    <Prose>The difference term is smallest when both weights equal s/2. With ridge λ=1, the optimum is (2/3,2/3). With elastic net λ=1 and ρ=0.5, it is (0.6,0.6). These methods also change the best sum; they do not merely redistribute the lasso answer.</Prose>
    <DuplicateFigure />
    <Prose>For nearly identical columns the conclusion becomes a tendency, with conditions, rather than exact equality. The elastic-net L2 term supplies strict convexity and encourages similar coefficients for similarly scaled, strongly positively correlated columns. It does not promise that every correlated group will always be selected together or that feature selection will be perfectly stable. Full-column-rank lasso is unique even when its columns are correlated; correlation alone is not a proof of nonuniqueness. <a href="https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf">Zou and Hastie&rsquo;s primary analysis</a> develops grouping, and <a href="https://arxiv.org/pdf/1206.0313">Tibshirani&rsquo;s uniqueness paper</a> states the more precise solution conditions.</Prose>
    <Prose>This distinction matters in applications such as correlated chemical measurements or groups of gene-expression features. A sparse predictor may be cheaper to measure and easier to inspect. Its nonzero list is still a property of this fitted model, its feature representation and its penalty. Prediction, stable selection and causal explanation require different evidence. The next lesson on feature importance will make those questions explicit.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>The Airfoil Self-Noise collection contains 1,503 observations from aeroacoustic experiments. Each row gives frequency in hertz, angle of attack in degrees, chord length in meters, free-stream speed in meters per second and suction-side displacement thickness in meters. The target is scaled sound-pressure level in decibels. These are physical observations, not points drawn to guarantee that one regularizer wins. <a href="https://archive.ics.uci.edu/dataset/291/airfoil+self+noise">UCI&rsquo;s dataset description</a> supplies the measurement context and <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> license; this page serves <a href="/learn-assets/regularization/airfoil-self-noise.dat" download>the unchanged data file</a> ({provenance.bytes.toLocaleString('en-US')} bytes, tab separated with six columns) and its attribution beside it. Nothing is downloaded when the program runs.</Prose>
    <Prose>The task is numeric prediction for held-out rows under the declared row-level experiment. Related experimental settings occur in the collection, and independent run IDs are not available here. This assessment does not establish performance on an entirely new airfoil or experimental run.</Prose>
    <Prose>We use a predeclared development set of {provenance.developmentRows.toLocaleString('en-US')} rows and reserve {provenance.reservedRows} rows, split with seed {provenance.splitSeed}. The reserved rows receive no prediction or score in this lesson. Within development, three shuffled folds with seed {provenance.foldSeed} compare fitting choices. This is a <strong>development comparison and selection record</strong>. Its selected scores are not newly independent performance claims. The later <a href="/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml">learning-curves lesson</a> reuses the input with a different protocol; its numbers should not be read as a direct contest against these fits.</Prose>

    <H3>Let a linear model express curved relationships</H3>
    <Prose>For this comparison, transform the five raw inputs into their five original terms, five squared terms and ten pairwise products: twenty features total. A term such as frequency × chord can represent an interaction between measurements. The model remains linear in the twenty fitted coefficients, although its prediction is nonlinear in the original inputs.</Prose>
    <Prose>Fit a scaler to those twenty columns inside each training fold, then fit the model. The polynomial recipe itself is fixed, but scaling learns from data and must stay within the fold. All three families use λ values <Code>[0.001,0.01,0.1,1,10,100]</Code>; elastic net fixes ρ=0.5 for this comparison. We also compute a training-mean baseline and unpenalized least squares with the same twenty-feature representation. Mean squared error has units of squared decibels.</Prose>

    <H3>Library names do not define the objective</H3>
    <Prose>For scikit-learn&rsquo;s <Code>Ridge</Code>, the documented objective is summed squared error plus <Code>alpha</Code> times squared coefficient norm. For <Code>Lasso</Code> and <Code>ElasticNet</Code>, the data term is half the <strong>mean</strong> squared error. Therefore, to fit our common convention on <Code>n_fit</Code> rows:</Prose>
    <LessonTable caption="Matching each estimator to this manuscript's objective" headers={['Estimator', 'Settings matching this manuscript']} rows={[
      [<Code>Ridge</Code>, <Code>alpha = n_fit * strength</Code>],
      [<Code>Lasso</Code>, <Code>alpha = strength</Code>],
      [<Code>ElasticNet</Code>, <><Code>alpha = strength</Code>, <Code>l1_ratio = ratio</Code></>],
    ]} />
    <Prose>At the same numeric <Code>alpha</Code>, ridge and lasso do not have the same normalized penalty strength. For logistic regression the inverse-strength parameter <Code>C</Code> also needs its estimator&rsquo;s documented loss normalization; “C=1/λ” without specifying that loss can miss a sample-count factor. The <a href="https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification">current objectives</a> are the reference, rather than a shared parameter name.</Prose>

    <H3>Complete offline program</H3>
    <Prose>Save as <Code>airfoil_regularization.py</Code> beside <Code>airfoil-self-noise.dat</Code>. In a new environment use <Code>python -m pip install numpy==2.3.5 scikit-learn==1.9.1</Code>. The program performs 54 candidate fits, three OLS fits and three selected refits; baseline means require no estimator fit. It does not train a neural network or download data, and it ran with no warnings.</Prose>
    <Program example={regularizationExamples.airfoilComparison}>
      <Prose>The file is tab separated with CRLF line endings, which is the whitespace layout <Code>np.loadtxt</Code> reads by default, so the six columns arrive in source order with no parsing options.</Prose>
    </Program>
    <LessonTable caption="Recorded development results, rounded to six decimals" headers={['λ', 'Ridge MSE', 'Lasso MSE', 'Elastic-net MSE']} rows={[0.001, 0.01, 0.1, 1, 10, 100].map(strength => [
      String(strength), mse('ridge', strength), mse('lasso', strength), mse('elastic_net', strength),
    ])} />
    <Prose>Every number in this section, and in the figure and investigation below it, is bit-exact in the pinned environment named above; another NumPy or scikit-learn version can move the last digits. The mean baseline is {baselineMeanMse.toFixed(6)} and unpenalized OLS is {olsMeanMse.toFixed(6)}. All three searches select the smallest λ in this predeclared grid, 0.001. Their selected differences are small; this experiment does not justify a strong ranking of families or claim a universal interior “sweet spot.” In a real development project, a boundary selection can motivate another declared search, with the assessment boundary still protected.</Prose>
    <Prose>Lasso at λ=0.1 leaves nine nonzero coefficients in each of the three folds. At λ=10 it leaves none and predicts the training mean, matching the baseline exactly. The λ=0.001 final lasso refit on all development rows keeps all twenty terms, even though two individual folds kept nineteen. L1 can create sparsity; the selection objective and sample do not guarantee that the chosen fit will be sparse.</Prose>
    <PathFigure />
    <AirfoilTraceLab />
    <Prose>The starting measurements are 1,250 Hz, 17.4 degrees, chord 0.0254 m, speed 31.7 m/s and displacement thickness 0.0176631 m, and that row&rsquo;s recorded prediction is approximately {inferenceFixture.basePrediction.toFixed(6)} dB. Keeping the other four values fixed and changing frequency to 1,750 Hz gives approximately {inferenceFixture.changedPrediction.toFixed(6)} dB, a decrease of {(inferenceFixture.basePrediction - inferenceFixture.changedPrediction).toFixed(6)} dB. This is a deterministic scenario under the fitted model, not evidence that physically changing frequency causes that exact change in a new experiment.</Prose>

    <H3>Selecting λ without leaking the scaler</H3>
    <Prose>An efficient regularization-path estimator such as <Code>LassoCV</Code> can reuse nearby solutions. But <Code>Pipeline(StandardScaler(), LassoCV(...))</Code> fits that outer scaler on all data supplied to the pipeline <strong>before</strong> the estimator runs its internal folds. Those inner validation rows then influenced scaling. The same issue arises if you compute <Code>X_scaled</Code> once and pass it to an internal CV estimator.</Prose>
    <Prose>Our explicit loop fits the whole pipeline within each fold. A <Code>GridSearchCV</Code> wrapped around a complete pipeline is another clear option when its parameter convention matches the intended objective. Path efficiency is useful, but it does not move preprocessing inside folds automatically. After selection, refitting preprocessing and the chosen model on all development rows is appropriate; those final fitted transformations must travel with the model for inference.</Prose>

    <H2>{headings[5]}</H2>
    <Prose>The first three methods add an explicit parameter cost. <strong>Dropout randomly withholds selected input or intermediate values during fitting.</strong> A later neural-network lesson will build layers in detail. For now, a hidden unit is simply a learned weighted sum followed by an activation function, and an activation is the value it passes onward. A dropout mask multiplies selected values by zero while leaving other paths available.</Prose>
    <Prose>Use <Math>{'q'}</Math> for the <strong>keep probability</strong>, so the drop probability is 1−q. With inverted dropout, an activation <Math>{'a'}</Math> becomes</Prose>
    <MathBlock>{'\\begin{gathered}\\tilde a=\\frac{m}{q}a,\\\\[4pt] m\\sim\\operatorname{Bernoulli}(q),\\quad0<q\\le1.\\end{gathered}'}</MathBlock>
    <Prose>A Bernoulli variable equals one with probability <Math>{'q'}</Math> and zero otherwise. Dividing retained values by <Math>{'q'}</Math> gives <Math>{'\\mathbb E[\\tilde a]=a'}</Math>. For q=0.5, a value 2 becomes either zero or four, each with probability one-half. At ordinary deterministic evaluation, the dropout operation is the identity: it passes the original activation through.</Prose>

    <H3>An exact connection to a penalty</H3>
    <Prose>Consider a linear prediction <Math>{'\\tilde y=\\sum_j w_jx_jm_j/q'}</Math> with independent masks and a fixed target <Math>{'y'}</Math>. The mean prediction is <Math>{'w^\\top x'}</Math>, but squared loss also responds to variation around that mean. Expanding the square gives</Prose>
    <MathBlock>{'\\begin{gathered}\\mathbb E_m\\!\\left[\\frac12(y-\\tilde y)^2\\right]\\\\[4pt] =\\frac12(y-w^\\top x)^2\\\\[4pt] +\\frac{1-q}{2q}\\sum_j w_j^2x_j^2.\\end{gathered}'}</MathBlock>
    <Prose>The cross terms from independent centered mask noise vanish. Each retained/rescaled input has variance <Math>{'x_j^2(1-q)/q'}</Math>. Averaging over training rows therefore produces a data-dependent diagonal quadratic penalty. This is exact for this linear, squared-loss, independent-mask setup. Other losses and nonlinear networks require different analysis or approximations; dropout is not universally the same as adding a fixed L2 penalty. <a href="https://nlp.stanford.edu/pubs/wager2013dropout.pdf">Wager, Wang and Liang</a> develop the more general feature-noising connection.</Prose>
    <Prose>For x=(2,1), w=(1,−1), y=1 and q=0.5, the deterministic prediction is one and its half-squared loss is zero. Enumerate all four masks:</Prose>
    <LessonTable caption="Every mask, with its actual probability" headers={['Mask', 'Noisy prediction', 'Half-squared loss', 'Probability']} rows={[
      ['(0,0)', '0', '0.5', '1/4'], ['(0,1)', '−2', '4.5', '1/4'], ['(1,0)', '4', '4.5', '1/4'], ['(1,1)', '2', '0.5', '1/4'],
    ]} />
    <Prose>The mean prediction is one but expected loss is 2.5. The penalty formula gives <Math>{'(1-q)/(2q)\\,(4+1)=2.5'}</Math>, exactly matching the enumeration. This is why matching an average activation does not make a noisy training objective identical to a clean evaluation loss.</Prose>
    <DropoutLab />
    <Prose>A complete exact enumeration uses only Python:</Prose>
    <Program example={regularizationExamples.dropoutMasks}>
      <Prose>Keep must be positive; at q=1 the zero-probability branches simply contribute nothing. Complete dropout-network training is owned by the later <a href="/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals">Dropout, DropPath &amp; Stochastic Depth</a> lesson rather than a second unexplained network here.</Prose>
    </Program>

    <H3>What changes when the rest of the network is nonlinear?</H3>
    <Prose>An average input passed through a nonlinear operation need not equal the average of that operation&rsquo;s outputs. For a small example, let a noisy value be zero or two equally often, and pass it through <Math>{'f(u)=\\max(0,u-1)'}</Math>. The mean of <Math>{'f'}</Math> is 0.5. Passing the mean input one through <Math>{'f'}</Math> gives zero. Consequently, ordinary dropout-off inference is not generally an exact average of every masked nonlinear network. The <a href="https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">original dropout paper</a> motivates and investigates the approximation; its benchmark outcomes are not universal guarantees.</Prose>
    <Prose>In PyTorch, <Code>nn.Dropout(p=...)</Code> uses <strong>drop probability</strong>, and ordinary evaluation uses <Code>model.eval()</Code>. Disabling gradient recording with <Code>no_grad()</Code> or <Code>inference_mode()</Code> is a separate operation; it does not itself switch dropout into evaluation behavior. Intentional Monte Carlo dropout is another inference procedure, not an automatic uncertainty guarantee. The <a href="https://docs.pytorch.org/docs/main/generated/torch.nn.Dropout.html">API contract</a> specifies element masking and inverted scaling. The later deep lesson owns mask placement, residual paths and normalization interactions.</Prose>
    <Callout title="Comparing models under a declared evaluation mode">
      Compare models on the same held-out cases using a clearly declared evaluation mode. A lower dropout-off loss on the <strong>training rows</strong> is still a training-data result; a gap between noisy training loss and clean loss does not prove generalization. Dropout rates, placement and combination with weight penalties require validation. There is no universal best rate, fixed extra-epoch multiplier or rule that adding more regularizers must improve a model.
    </Callout>

    <H2>{headings[6]}</H2>
    <H3>Ridge shrinks directions of information</H3>
    <Prose>Write a singular value decomposition of the centered design as <Math>{'Z=U\\Sigma V^\\top'}</Math>. The columns of <Math>{'V'}</Math> describe orthogonal directions in coefficient space; a singular value <Math>{'\\sigma_j'}</Math> tells us how strongly changing that direction changes the fitted observations. Along a positive-singular-value direction, ridge uses</Prose>
    <MathBlock>{'w_\\lambda=\\sum_j\\frac{\\sigma_j}{\\sigma_j^2+n\\lambda}(u_j^\\top t)v_j.'}</MathBlock>
    <Prose>The corresponding fitted-data component is multiplied by <Math>{'\\sigma_j^2/(\\sigma_j^2+n\\lambda)'}</Math>. A large singular value retains more of its unpenalized fit. A small singular value is attenuated more, preventing division by a tiny number from producing a huge coefficient response. Components in the null space are set to zero by positive ridge regularization.</Prose>
    <Prose>For example, with nλ=1, singular values 4 and 0.5 give fitted-component multipliers 16/17≈{ridgeFilter(4, 1).dataMultiplier.toFixed(4)} and 0.25/1.25={ridgeFilter(0.5, 1).dataMultiplier}. With nλ=4 they become {ridgeFilter(4, 4).dataMultiplier} and approximately {ridgeFilter(0.5, 4).dataMultiplier.toFixed(5)}. The weakly identified direction receives much stronger relative shrinkage. This is a more accurate explanation of stabilization than saying every original feature coefficient is multiplied by the same number.</Prose>
    <Prose>As λ decreases to zero, ridge approaches the minimum-Euclidean-norm least-squares solution. When the design is full column rank, that is the usual unique OLS solution. As λ grows, the slopes approach zero and the unpenalized intercept leaves the training mean prediction. Individual coefficients can move non-monotonically or cross zero in correlated designs; the scalar shrinkage formula does not describe each original coordinate independently.</Prose>
    <DirectionsFigure />

    <H3>What an L1 path tells you</H3>
    <Prose>For centered data and pure lasso, the all-zero slope vector satisfies the optimality conditions when</Prose>
    <MathBlock>{'\\begin{gathered}\\lambda\\ge\\lambda_{\\max}\\\\[4pt] =\\max_j|Z_{:,j}^\\top t|/n.\\end{gathered}'}</MathBlock>
    <Prose>For the four-row example, λ<sub>max</sub>=3. This gives a principled starting point for a path from a zero-slope fit toward less penalization. Nearby λ values can use the previous solution as a warm start. In general correlated designs, coordinates may enter, leave or change sign along the path; the number of selected features is not guaranteed to move monotonically at every path point.</Prose>
    <Prose>When a lasso solution is nonunique, every minimizer has the same fitted values on the training design, although coefficient allocations can differ. There exists a sparse representative with a limited active set; under common general-position uniqueness conditions the number of nonzero coefficients is at most the design rank. This is more precise than asserting that <strong>every</strong> solution always has at most <Math>{'n'}</Math> nonzeros. In the duplicate-column example, one can distribute an optimal positive sum over many identical columns without changing fit or L1 cost. Prediction away from the observed design can differ if those columns no longer remain identical. The <a href="https://arxiv.org/pdf/1206.0313">uniqueness analysis</a> is the reference for these distinctions.</Prose>

    <H3>Choose a solver for the matrix you actually have</H3>
    <Prose>For dense n×d data with n≥d, forming the normal-equation matrix costs order nd² and a dense solve order d³; storing that matrix costs order d². If d is much larger than n, the identity</Prose>
    <MathBlock>{'w=Z^\\top(ZZ^\\top+n\\lambda I)^{-1}t'}</MathBlock>
    <Prose>offers an n×n system instead. Use a solve here too. These dimensions suggest alternatives; they do not establish an exact practical crossover at n=d. Conditioning, sparsity, factorization reuse and available memory matter. Iterative least-squares or matrix-vector methods can avoid either dense Gram matrix.</Prose>
    <Prose>The residual-maintaining coordinate implementation costs order nd per dense sweep. Recomputing the entire prediction <Math>{'Zw'}</Math> separately for every coordinate would introduce unnecessary extra work. Sparse column storage can make an update depend on its nonzero entries; specialized solvers add screening, active sets and warm starts. The number of sweeps depends on tolerance and conditioning, so a universal “10–200 sweeps” promise is inappropriate. Check convergence warnings and optimality diagnostics before interpreting a fit.</Prose>
    <Prose>For distributed ridge with manageable d, sums of local <Math>{'Z_i^\\top Z_i'}</Math> and <Math>{'Z_i^\\top t_i'}</Math> can recover the corresponding global sufficient statistics, provided centering, weights and normalization are handled consistently. The d² communication/storage requirement remains. Consensus optimization can distribute lasso-type objectives, but it needs its own convergence and communication design. A local solver does not become a distributed algorithm merely by putting its call in a task scheduler.</Prose>

    <H3>Early stopping and weight decay are related, but have precise contracts</H3>
    <Prose>Under ordinary gradient descent on the unpenalized centered mean-square objective, initialized at zero, a direction whose Gram eigenvalue is a&gt;0 has fitted-component factor <Math>{'1-(1-\\eta a)^t'}</Math> after <Math>{'t'}</Math> steps with step size <Math>{'\\eta'}</Math>. Ridge&rsquo;s factor is <Math>{'a/(a+\\lambda)'}</Math>. Both can suppress weakly learned directions, but they are different filters. For one step, η=0.1 and a values 1 and 4 give factors 0.1 and 0.4. Matching those with ridge would require λ values 9 and 6 respectively; one common λ does not reproduce both. Appropriate step-size conditions are also needed for the iteration to remain stable.</Prose>
    <Prose>Likewise, a gradient step on a loss plus <Math>{'\\lambda\\|w\\|^2/2'}</Math> is</Prose>
    <MathBlock>{'w^+=(1-\\eta\\lambda)w-\\eta\\nabla L(w).'}</MathBlock>
    <Prose>For this ordinary update, a multiplicative decay with factor <Math>{'1-\\eta\\lambda'}</Math> is equivalent. In an adaptive optimizer, adding λw to a gradient sends it through the optimizer&rsquo;s gradient transformation; decaying weights separately generally does not. That is the distinction behind AdamW. Inspect the optimizer&rsquo;s actual convention and parameter groups, including whether biases and normalization parameters are included. The <a href="https://arxiv.org/pdf/1711.05101">decoupled-weight-decay paper</a> derives the difference. Detailed optimizer dynamics belong to the optimization lessons; a <Code>weight_decay</Code> argument is not a universal mathematical identity.</Prose>

    <H2>{headings[7]}</H2>
    <H3>Bayesian priors: track the noise scale</H3>
    <Prose>Assume <Math>{'y\\mid b,w,X'}</Math> has independent Gaussian errors with known variance <Math>{'\\sigma^2'}</Math>. Ignoring constants, the negative log likelihood is <Math>{'\\|y-b\\mathbf1-Xw\\|^2/(2\\sigma^2)'}</Math>. A Gaussian prior <Math>{'w_j\\sim N(0,\\tau^2)'}</Math> contributes <Math>{'\\|w\\|^2/(2\\tau^2)'}</Math>. Multiplying the combined negative log posterior by <Math>{'\\sigma^2/n'}</Math> gives our ridge objective with</Prose>
    <MathBlock>{'\\lambda=\\frac{\\sigma^2}{n\\tau^2}.'}</MathBlock>
    <Prose>An independent Laplace prior with density proportional to <Math>{'\\exp(-|w_j|/s)'}</Math> instead gives lasso strength <Math>{'\\lambda=\\sigma^2/(ns)'}</Math>. The intercept can receive a separate prior or be treated as unpenalized. These are <strong>maximum a posteriori</strong>, or MAP, fits: the most favored parameter value under the stated likelihood/prior combination.</Prose>
    <Prose>The factors matter. Writing “Gaussian variance 1/λ” without the likelihood and normalization can be wrong for the objective being used. If a prior and noise scale are held fixed while <Math>{'n'}</Math> changes, our normalized λ changes inversely with <Math>{'n'}</Math>. Holding λ fixed over different training sizes is a different convention, useful for a controlled regularization comparison but not the same fixed-prior experiment.</Prose>
    <Prose>A Laplace prior is continuous; it assigns probability zero to any exact singleton w<sub>j</sub>=0, as do other continuous densities. Its posterior mode can be exactly zero because of the density&rsquo;s kink. That does not give a posterior probability that a feature is absent. A full Bayesian analysis includes uncertainty and integrates predictions over parameter values; replacing it by one penalized fit discards that information. The <a href="https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf">elastic-net paper&rsquo;s Bayesian section</a> connects the priors, while this local derivation specifies our factors explicitly.</Prose>

    <H3>Same predictor, different parameter penalty</H3>
    <Prose>Suppose a one-dimensional model is written with two factors, predicting abx, and the data cost is <Math>{'\\frac12(ab-1)^2'}</Math>. Every pair with ab=1 has zero data cost. But adding <Math>{'\\lambda(a^2+b^2)'}</Math> gives different costs along that same-prediction curve: (1,1) costs 2λ, while (2,0.5) costs 4.25λ.</Prose>
    <Prose>Balancing the factors minimizes the penalty <strong>among zero-data-loss pairs</strong>, but the full regularized optimum can prefer nonzero data loss. Let p=ab. Since <Math>{'a^2+b^2\\ge2|ab|=2|p|'}</Math>, with equality attainable by equal-magnitude factors, the full problem reduces to</Prose>
    <MathBlock>{'\\min_p\\frac12(p-1)^2+2\\lambda|p|.'}</MathBlock>
    <Prose>Soft-thresholding gives <Math>{'p^*=\\max(1-2\\lambda,0)'}</Math>. At λ=0.25, the optimal product is {factorOptimum(0.25).product}, achieved by a=b=√0.5 or both negative. The data cost is {factorOptimum(0.25).data} and penalty {factorOptimum(0.25).penalty}, totaling {factorOptimum(0.25).total}; balanced zero-data-loss factors would total {factorOptimum(0.25).balancedZeroLoss.total}. At λ≥0.5, both optimal factors are zero. At λ=0, every ab=1 pair minimizes the unpenalized problem.</Prose>
    <Prose>This is a concrete example of a parameterization changing what a familiar L2 penalty means for the represented function. It is not evidence that balancing arbitrary neural layers universally improves prediction. The learning point is to inspect the <strong>whole objective</strong>, not only a symmetry of its data-loss term.</Prose>
    <FactorFigure />

    <H3>Sometimes smoothness is a better preference than small values</H3>
    <Prose>If <Math>{'w'}</Math> represents values at neighboring positions, penalizing differences can be more meaningful than pulling every value toward zero. Let <Math>{'L'}</Math> compute neighboring differences and minimize</Prose>
    <MathBlock>{'\\frac12\\|y-w\\|^2+\\frac\\lambda2\\|Lw\\|^2.'}</MathBlock>
    <Prose>For three positions, take <Math>{'Lw=(w_2-w_1,w_3-w_2)'}</Math>. With y=(0,2,0) and λ=1, solve <Math>{'(I+L^\\top L)w=y'}</Math> to obtain (0.5,1,0.5). Ordinary identity-based ridge with the same unaveraged convention gives (0,1,0). Both reduce the middle spike, but the difference penalty spreads it across neighboring positions. Adding a constant to every input shifts the difference-penalty solution by the same constant because <Math>{'L'}</Math> annihilates a constant vector.</Prose>
    <Prose>This is a small instance of <strong>generalized Tikhonov regularization</strong>, where <Math>{'\\|Lw\\|^2'}</Math> expresses which patterns are expensive. A difference operator encourages smoothness; another <Math>{'L'}</Math> can encode a different scientifically justified relation. It is useful in inverse problems, where measurements are indirect and many latent signals could explain them. The condition for a unique generalized quadratic fit is that no nonzero direction lies in both the data operator&rsquo;s null space and <Math>{'L'}</Math>&rsquo;s null space. A difference penalty alone does not necessarily remove every ambiguity.</Prose>
    <SmoothnessFigure />
    <Prose>Other useful penalties encode other structures. An L1 penalty on differences, often called total-variation or fused regularization in appropriate settings, can favor piecewise-constant regions rather than smooth variation. A group-lasso penalty sums Euclidean norms of predeclared coefficient groups, allowing an entire group to become zero. A multi-task penalty can select the same input across several prediction outputs. These are different assumptions about where sparsity belongs: individual coefficients, neighboring changes, predefined groups or shared tasks. They are not interchangeable names for elastic net&rsquo;s tendency to balance correlated individual coefficients.</Prose>
    <Prose>An engaging further example is reconstructing an image from a few line projections. The unknown pixel values form <Math>{'w'}</Math>, and a known projection operator maps them to measurements. If the image is sparse in the chosen representation, L1 regularization can express that prior structure. Most natural images are not sparse as raw pixels, so the representation is part of the scientific claim. The inspected <a href="https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html">tomography reconstruction example</a> shows the actual operator, synthetic image and comparison; its particularly favorable sparse image is not a guarantee of exact recovery for arbitrary scans.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>Coefficient penalties are not the only way to control fitting flexibility. Suppose we compare candidate probability models, each fitted by maximum likelihood on the same observations. A more flexible model often has a larger training likelihood simply because it had more freedom. AIC, BIC and minimum description length account for complexity for different reasons. They do not turn a development-selected score into a fresh test result.</Prose>

    <H3>AIC: correct optimism when estimating predictive fit</H3>
    <Prose>Let <Math>{'\\ell(\\hat\\theta)'}</Math> be the maximized natural-log likelihood and <Math>{'k'}</Math> the number of freely fitted parameters in a regular parametric model. The familiar formula is</Prose>
    <MathBlock>{'\\operatorname{AIC}=-2\\ell(\\hat\\theta)+2k.'}</MathBlock>
    <Prose>The negative likelihood term measures fit; smaller is better. The correction reflects that parameters were chosen on the data being scored. Under the usual regular, correctly specified parametric assumptions, it estimates expected predictive log-loss, up to a constant shared by candidates. The 2k correction is an asymptotic result, not a universal fee for every parameter in every algorithm; model misspecification can require a different optimism correction.</Prose>
    <Prose>For a Gaussian regression with unknown noise variance estimated by maximum likelihood, substitution gives a data-dependent term <Math>{'n\\log(\\mathrm{RSS}/n)'}</Math> plus constants, followed by 2k. Count an estimated variance parameter and intercept consistently. Known-variance formulas differ. Small-sample corrections such as AICc have model-specific assumptions; they are not a universal replacement for checking sample size, dependence or misspecification.</Prose>
    <Prose>For shrinkage estimators, the effective flexibility can differ from the raw number of stored coefficients. The later <a href="/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml">Bias–Variance &amp; Learning Curves</a> derives the fixed-linear-smoother optimism correction and its trace-based degrees of freedom. Counting twenty stored ridge coefficients as twenty freely fitted OLS coefficients would miss that shrinkage. Information-criterion implementations for lasso use additional model and variance-estimation assumptions; inspect those rather than attaching 2k to an arbitrary penalized training objective. The <a href="https://scikit-learn.org/stable/modules/linear_model.html#aic-and-bic-criteria">scikit-learn criterion derivation</a> specifies its actual Gaussian convention.</Prose>

    <H3>BIC: a large-sample evidence approximation</H3>
    <Prose>For the same kind of regular, fixed-dimensional model,</Prose>
    <MathBlock>{'\\operatorname{BIC}=-2\\ell(\\hat\\theta)+k\\log n.'}</MathBlock>
    <Prose>The evidence for a model integrates likelihood over its parameter prior, rather than evaluating only the best point. A local quadratic approximation around the maximum makes each well-identified parameter direction contribute a width of order <Math>{'n^{-1/2}'}</Math>. Multiplying <Math>{'k'}</Math> such widths contributes <Math>{'n^{-k/2}'}</Math>; taking −2 times the log gives the <Math>{'k\\log n'}</Math> term. Prior densities and local curvature contribute terms that the basic large-n expression suppresses.</Prose>
    <Prose>This argument needs an identifiable, regular interior solution and suitable priors, with dimension held fixed as <Math>{'n'}</Math> grows. In a correctly specified collection satisfying the needed conditions, BIC can consistently favor the correct model dimension. That is a different goal from minimizing predictive loss at a finite sample size. Neural networks, mixture singularities, growing dimensions and boundary parameters do not automatically satisfy this derivation. BIC values are not exact posterior probabilities. <a href="https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf">Grünwald&rsquo;s technical discussion</a>, sections 2.6.3 and 2.9.2, develops the evidence approximation and its limits.</Prose>
    <Prose>Take two constructed fitted-model records on n=100 observations:</Prose>
    <LessonTable caption="Two constructed records, and two criteria that disagree" headers={['Model', 'Maximized log likelihood', 'k', 'AIC', 'BIC']} rows={[
      ['Smaller', '−150', '3', String(smaller.aic), smaller.bic.toFixed(4)],
      ['Larger', '−146', '5', String(larger.aic), larger.bic.toFixed(4)],
    ]} />
    <Prose>The larger model improves −2 log likelihood by eight. AIC charges four for its two added parameters, while BIC charges about {(larger.bicPenalty - smaller.bicPenalty).toFixed(4)}. They choose differently because they use different justified approximations and goals. These are hand records for arithmetic, not empirical evidence that one criterion is better. Compare only compatible likelihoods on the same observations, with constants and target transformations treated consistently.</Prose>

    <H3>MDL: pay to describe the explanation as well as its errors</H3>
    <Prose>Minimum description length asks how compactly a declared coding scheme can describe the observed data. In a simple two-part version, pay for a model/parameter description and then for the data given that description:</Prose>
    <MathBlock>{'\\begin{gathered}L(\\text{explanation})\\\\[4pt] +\\,L(\\text{data}\\mid\\text{explanation}).\\end{gathered}'}</MathBlock>
    <Prose>Here <Math>{'L'}</Math> denotes code length, not the regression loss notation used earlier. A pattern that perfectly fits the observations is not free: the receiver must be told which pattern or parameter values were selected. For a discrete probability model, ideal data-code length is <Math>{'-\\log_2 P(\\text{data}\\mid\\text{model})'}</Math>. Continuous measurements additionally need a declared precision or corresponding density-based construction.</Prose>
    <Prose>A tiny fully specified code makes this concrete. Both parties know the message contains sixteen bits. Two modes are allowed:</Prose>
    <ul>
      <li>Mode 0: send a zero flag followed by all sixteen literal bits, for seventeen bits total.</li>
      <li>Mode 1: send a one flag followed by a four-bit pattern; the receiver repeats that pattern four times, for five bits total.</li>
    </ul>
    <Prose>For <Code>0101010101010101</Code>, mode 1 sends the flag and <Code>0101</Code>: five bits. The chosen pattern was learned from the data, and its four bits were paid for. For <Code>0101010001010101</Code>, no four-bit pattern repeated four times reproduces the message, so this scheme uses the seventeen-bit literal mode. These codes are unambiguous because the first flag identifies the remaining length. The scheme is intentionally limited; another declared code might exploit a different pattern. We are not claiming to compute the shortest possible program for every message.</Prose>
    <ComplexityFigure />
    <Prose>Modern MDL includes refined universal codes, not only a hand-selected parameter code. For a finite discrete model class with a finite normalizer, normalized maximum likelihood assigns</Prose>
    <MathBlock>{'P_{\\mathrm{NML}}(D)=\\frac{P(D\\mid\\hat\\theta_D)}{\\sum_{D\'}P(D\'\\mid\\hat\\theta_{D\'})}.'}</MathBlock>
    <Prose>The denominator accounts for all datasets of the stated size that the model family can fit well. Its logarithm supplies a complexity cost and makes the expression a probability distribution. Some model classes have an infinite normalizer and require another construction. Under specific fixed-dimensional regular asymptotics, an MDL expression can share BIC&rsquo;s leading complexity term; <strong>MDL and BIC are not identical in general</strong>. The inspected <a href="https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf">MDL tutorial</a> provides both the basic coding view and the refined distinction.</Prose>
    <Prose>These criteria extend the same habit as regularization: state the preference, its units and assumptions, then distinguish the quantity optimized from the outcome ultimately needed. Cross-validation remains useful when it matches the intended future use and encompasses the whole selection recipe; analytic or coding criteria are useful when their assumptions and purpose fit the problem.</Prose>

    <H2>{headings[9]}</H2>
    <Prose>Try the first six without the deeper branches. Hints and solutions are optional so you can work independently before checking.</Prose>

    <Practice title="1. Shrinkage with a different sign"
      question={<>For <Math>{'\\frac12(w+2.4)^2'}</Math> and λ=0.6, find ridge, lasso and elastic-net ρ=0.5 coefficients. Explain why a zero answer is not appropriate here.</>}
      hint="The data preference z is −2.4. Apply the threshold before the elastic-net denominator.">
      <Prose>Ridge is −2.4/1.6={num(scalarSolution(-2.4, 0.6, 0).coefficient)}. Lasso is {num(scalarSolution(-2.4, 0.6, 1).coefficient)}. Elastic net is (−2.4+0.3)/1.3=−21/13≈{num(Number(scalarSolution(-2.4, 0.6, 0.5).coefficient.toFixed(6)))}. The absolute data preference exceeds each relevant threshold.</Prose>
    </Practice>

    <Practice title="2. What changes under a different measurement origin?"
      question="In the four-row example, fit lasso with λ=0.5, then increase every target by three. Give both sets of slopes and intercepts. Would penalizing the intercept necessarily preserve this result?">
      <Prose>The slopes are (2.5,0) in both fits. The original intercept is zero and the shifted intercept is three. Excluding the intercept allows an exact translation without changing the slope objective. An intercept penalty introduces an extra cost for that translation and can change the result.</Prose>
    </Practice>

    <Practice title="3. A missing sample-count factor"
      question={<>You want our ridge objective with λ=0.2 on eighty training rows. Which <Code>Ridge(alpha=...)</Code> matches it? Which <Code>Lasso(alpha=...)</Code> matches pure L1 at the same λ convention? What happens to ridge&rsquo;s native alpha on a sixty-row fold?</>}>
      <Prose>Ridge needs alpha=16 on eighty rows and alpha=12 on sixty rows. Lasso uses alpha=0.2 in either case. This follows from multiplying our ridge objective by 2n, not from treating the two parameter names as equivalent. It does not say that matching numerical λ gives the two penalty shapes identical effects.</Prose>
    </Practice>

    <Practice title="4. Duplicate sensors"
      question="The duplicate-feature example now has target 3x and lasso λ=0.5. Give the optimal coefficient sum and two different minimizers. What extra fact would you need before calling one sensor causally important?">
      <Prose>The optimal sum is 2.5. Pairs (2.5,0) and (1.25,1.25) both minimize the objective, as do other nonnegative allocations of that sum. The observational duplicate design does not identify which sensor is causally relevant; that requires an appropriate causal question, assumptions and evidence beyond this fit.</Prose>
    </Practice>

    <Practice title="5. Exact dropout without uniform mask probabilities"
      question="Let x=(1,2), w=(2,0), y=1 and keep probability q=0.75. Compute the clean prediction, expected noisy prediction, clean half-squared loss and expected noisy half-squared loss. Why does the second mask not affect the answer?"
      hint="Only the first coordinate contributes. Its noisy prediction is zero with probability 1/4 and 8/3 with probability 3/4.">
      <Prose>Both clean and expected predictions are two. Clean half-squared loss is 1/2. Expected noisy loss is <Math>{'(1/4)(1/2)+(3/4)(25/18)=7/6'}</Math>. The difference is 2/3, matching <Math>{'(1-q)/(2q)\\,4'}</Math>. The second coefficient is zero, so changing that mask cannot change the weighted sum. The dropout investigation loads this exact fixture from its practice preset.</Prose>
    </Practice>

    <Practice title="6. Read the actual experiment"
      question="A learner sees the λ=0.001 lasso result and says: “Lasso always selects fewer inputs than ridge, and the smallest score proves this family will win on a new airfoil.” Identify two separate errors. What does the λ=10 result legitimately demonstrate?">
      <Prose>The final selected lasso fit keeps all twenty terms, so L1 does not guarantee sparsity at the selected setting. These are development selection scores in a row-level design, not independent evidence about a new airfoil/run or a decisive family ranking. At λ=10 all lasso slopes are zero in the three folds, and the unpenalized intercept reproduces the corresponding mean baseline.</Prose>
    </Practice>

    <Practice title="7. A changed factor penalty — deeper"
      question={<>Minimize <Math>{'\\frac12(ab-1)^2+0.1(a^2+b^2)'}</Math>. Give the optimal product and balanced factors. Compare its total cost with the balanced zero-data-loss pair (1,1).</>}>
      <Prose>The product is {factorOptimum(0.1).product} and equal-sign factors have magnitude √0.8≈{factorOptimum(0.1).magnitude.toFixed(6)}. Data cost is {factorOptimum(0.1).data.toFixed(2)}, penalty is {factorOptimum(0.1).penalty.toFixed(2)} and total is {factorOptimum(0.1).total.toFixed(2)}, below the {factorOptimum(0.1).balancedZeroLoss.total.toFixed(1)} of (1,1). Minimizing the penalty while insisting on zero data loss misses the actual full optimum.</Prose>
    </Practice>

    <Practice title="8. A code has to pay for its chosen pattern — deeper"
      question={<>Using the declared sixteen-bit code, encode <Code>1110111011101110</Code>. Give the mode, payload and total length. If someone chooses a different four-bit pattern after seeing the data but charges only the flag, what is missing?</>}>
      <Prose>Mode 1, payload 1110, total five bits. The receiver must learn which of sixteen possible patterns was selected, so the four-bit pattern description cannot be omitted. The receiver already knows the total message length and the repeat rule under this declared code.</Prose>
    </Practice>

    <Practice title="9. Criteria can disagree — deeper"
      question="On n=50 observations, a smaller model has log likelihood −80 and k=2. A larger model has log likelihood −77 and k=4. Compute AIC and BIC for both. Does disagreement imply a calculation error?">
      <Prose>AIC values are {criteria(-80, 2, 50).aic} and {criteria(-77, 4, 50).aic}, favoring the larger model. BIC values are <Math>{'160+2\\log50\\approx'}</Math>{criteria(-80, 2, 50).bic.toFixed(4)} and <Math>{'154+4\\log50\\approx'}</Math>{criteria(-77, 4, 50).bic.toFixed(4)}, favoring the smaller. The different penalties reflect different goals and assumptions; disagreement alone is not an error. These formulas still require compatible regular likelihood models and correctly counted parameters.</Prose>
    </Practice>

    <Practice title="10. A smoothness-preserving shift — deeper"
      question="Change the difference-penalty input from (0,2,0) to (3,5,3), keeping λ=1 and the unaveraged objective. Predict the solution without solving another matrix system. Would identity-based ridge make the same shift?">
      <Prose>The difference-penalty solution becomes ({shifted.difference.map(num).join(', ')}), because adding a constant lies in L&rsquo;s null space. Identity-based ridge gives ({shifted.identity.map(num).join(', ')}), so its output shift is only 1.5. The penalties express different preferences.</Prose>
    </Practice>

    <H2>{headings[10]}</H2>
    <Prose>You are ready to move on when you can state the fitted objective and its normalization, calculate a shrinkage/threshold update, distinguish prediction from coefficient attribution, fit preprocessing within each validation fold, and explain why mean-preserving dropout still changes expected loss. You should also be able to read the real comparison without forcing a U shape or treating a development-selected score as independent evidence.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['State the objective, its 1/2n normalization and the unpenalized intercept', 'Section 1, figure 1'],
      ['Calculate a soft-threshold and a shrinkage, and say when the answer is exactly zero', 'Section 2, the threshold investigation, practice 1'],
      ['Run one coordinate sweep from partial residuals and recover the intercept', 'Section 3, the coordinate investigation, practices 2 and 3'],
      ['Give two coefficient vectors with the same fitted prediction', 'Section 4, figure 3, practice 4'],
      ['Read the recorded airfoil path, including λ=10 and the non-sparse selection', 'Section 5, figure 4, practice 6'],
      ['Explain why a mean-preserving mask still raises expected loss', 'Section 6, the mask investigation, practice 5'],
      ['Separate ridge directions, factor penalties and complexity criteria', 'Sections 7 to 9, practices 7 to 10'],
    ]} />
    <Prose>Next is <a href="/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml">Feature Selection &amp; Importance: SHAP, Permutation &amp; Mutual Information</a>. A zero or large coefficient is only one kind of statement. We will ask which features are useful to a fitted predictor, how removing or perturbing a feature changes its performance, what information exists before fitting, and why none of those questions automatically identifies causes.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html">Compressive sensing: tomography reconstruction with an L1 prior</a>: a visual and code learning route from projections to an image. Read the synthetic image&rsquo;s sparsity assumption and operator construction before interpreting the favorable comparison.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html">Lasso model selection: AIC, BIC and cross-validation</a>: an inspected criterion/path visualization and executable example. It illustrates estimator choices; when adapting its internally cross-validated estimator, keep learned preprocessing inside the folds as explained here.</li>
      <li><a href="https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf">Grünwald, A Tutorial Introduction to the Minimum Description Length Principle</a>: start with chapter 1&rsquo;s coding explanation, then use sections 2.5.3, 2.6.3 and 2.9.2 for normalized maximum likelihood, Bayesian evidence and the distinction from BIC. The extra branches are optional further study rather than prerequisites for the core regularization workflow.</li>
    </ul></>}>
      <li><a href="https://scikit-learn.org/stable/modules/linear_model.html">Scikit-learn linear models</a> — inspect the actual ridge/lasso/elastic-net objectives, coordinate updates, path diagnostics, multi-task penalties and information-criterion assumptions. Compare each equation with its parameter names before copying a strength value between estimators.</li>
      <li><a href="https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf">Zou and Hastie, Regularization and Variable Selection via the Elastic Net</a> — sections 2–3 explain grouping and the original paper&rsquo;s distinction between a mixed-penalty estimate and its historical rescaled variant. Current <Code>ElasticNet</Code> follows its documented mixed objective; do not add the paper&rsquo;s extra rescaling to a library prediction automatically.</li>
      <li><a href="https://arxiv.org/pdf/1206.0313">Tibshirani, The Lasso Problem and Uniqueness</a> — the KKT conditions, equal fitted-value property and uniqueness assumptions correct the oversimplified claim that correlation always makes lasso nonunique.</li>
      <li><a href="https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">Srivastava and colleagues, Dropout</a> — the mask/network diagrams, training procedure and empirical model-averaging discussion. Its particular architecture/rate findings are observations from its experiments, not universal placement rules.</li>
      <li><a href="https://nlp.stanford.edu/pubs/wager2013dropout.pdf">Wager, Wang and Liang, Dropout Training as Adaptive Regularization</a> — deeper analysis of noising under generalized linear losses. Our exact two-input square-loss derivation is the preparation for its data-dependent penalties.</li>
      <li><a href="https://arxiv.org/pdf/1711.05101">Loshchilov and Hutter, Decoupled Weight Decay Regularization</a> — the update equations explain why adaptive-optimizer weight decay needs a precise convention.</li>
      <li><a href="https://archive.ics.uci.edu/dataset/291/airfoil+self+noise">Airfoil Self-Noise at UCI</a>, {provenance.authors}, <a href={provenance.doi}>{provenance.doi}</a>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a> — measurement definitions, attribution and data license. This page serves the unchanged numeric file, SHA-256 <Code>{provenance.sha256}</Code>, with the exact development/fold protocol for offline reproduction.</li>
    </Sources>
    <Prose>The scalar shrinkage examples, the four-row and duplicate designs, the dropout mask enumeration, the singular-value filters, the two-factor optimum, the three-position smoothness system, the fitted-likelihood records and the bit-code examples are explicitly <strong>constructed calculations</strong>, not observed measurements. The airfoil results are calculations on the identified real dataset under one declared row-level split and one predeclared grid, with no reserved row predicted or scored. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>,
};

export default regularizationContent;
