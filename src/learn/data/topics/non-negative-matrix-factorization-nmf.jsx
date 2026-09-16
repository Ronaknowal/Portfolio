import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { MixtureLab, UpdateLab, ContributionLab } from '../../components/lesson-labs/NmfLabs.jsx';
import {
  AmbiguityFigure, BuildRowFigure, CandidateFigure, DictionaryFigure, FitTransformFigure, ResidualFigure,
  SupportFigure, UpdatePhaseFigure, ZeroLockFigure,
} from '../../components/lesson-labs/NmfFigures.jsx';
import { nmfExamples } from '../nmf-examples.js';
import { NMF_DIGITS } from '../nmf-data.js';
import { bilinearMidpoint, fixtures, sweep, sweepCost } from '../nmf-models.js';

const firstSweep = sweep(fixtures.X, fixtures.startW, fixtures.startH);
const nonconvex = bilinearMidpoint(1, [1, 1], [2, 0.5]);
const cost = sweepCost(180, 64, 8);
const { runs, baselines, splitSizes, fit, versions } = NMF_DIGITS;
const bestRun = runs.reduce((low, run) => (run.validationMse < low.validationMse ? run : low));
const six = value => value.toFixed(6);

const headings = [
  '1. A component is a pattern, and an activation is an amount',
  '2. What gets optimized?',
  '3. Learn one factor while holding the other still',
  '4. The same observations can have different explanations',
  '5. Fit real digits and inspect the result',
  '6. Practical choices that change the model',
  '7. Deeper: geometry, optimization and the limits of factorization',
  '8. Deeper applications: words, spectra and streams',
  '9. Practice: explain, change, diagnose',
  '10. Readiness and the next step',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section><Prose><strong>Before running:</strong> {example.question}</Prose><RunnableExample example={example}>{children}</RunnableExample></section>;
}
function Practice({ title, question, hint, children }) {
  return <section className="nm-practice"><H3>{title}</H3><Prose>{question}</Prose>{hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}<details><summary>Show the explained solution</summary>{children}</details></section>;
}

const nmfContent = {
  title: 'Non-Negative Matrix Factorization (NMF)',
  readTime: '~50 min core reading · ~60 min code and practice · deeper branches a separate sitting',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot nm-lesson">
    <LessonIntro prerequisites={<>Matrix multiplication, nonnegative weighted sums, squared error and the idea of a gradient, all refreshed locally where they enter. <a href="/learn/path/full-curriculum/vectors-matrices-tensor-operations?module=math-foundations">Vectors, Matrices &amp; Tensor Operations</a> and <a href="/learn/path/full-curriculum/pca-dimensionality-reduction?module=classical-ml">PCA</a> are the places to revisit if matrix shapes or reconstruction error feel unfamiliar. The preceding <a href="/learn/path/full-curriculum/independent-component-analysis-ica?module=classical-ml">Independent Component Analysis</a> lesson looked for independent hidden sources; this one asks for a different property entirely.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      Suppose you have hundreds of small pictures of handwritten digits. You could store every picture separately. Could you instead learn a small collection of reusable ink patterns and describe each picture by how much of each pattern to add? Learn to trace one reconstructed cell by hand, read a signed residual before it is squared, step both phases of a multiplicative update from your own start, see two different nonnegative dictionaries explain the same measurements exactly, and then fit {splitSizes.train} real digit images and pull an actual held-out reconstruction apart component by component. Every investigation asks for a prediction before it shows an answer, and retires that prediction the moment an input changes.
    </LessonIntro>
    <Prose className="nm-route"><strong>First pass.</strong> Read sections 1 to 6, working through the additive reconstruction and update investigations as they appear, then attempt exercises 1 to 5 in section 9. This route takes you from matrix entries to a complete offline fit and an interpretation of its errors. Return to section 7 for optimization, nonnegative rank and separability, and section 8 for topic models, spectra and streaming; exercises 6 to 8 assess those deeper branches. Allow roughly 50 minutes for the core reading and another hour for its code and practice. The deeper branches are a separate sitting.</Prose>

    <Prose>That is the central question of <strong>non-negative matrix factorization</strong>. It learns nonnegative patterns and nonnegative amounts whose sums approximate your observations. The same arithmetic can describe a document as a combination of word patterns or a measured spectrum as a combination of spectral patterns. The useful result is a compact, inspectable representation of the measurements.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>The preceding <a href="/learn/path/full-curriculum/independent-component-analysis-ica?module=classical-ml">ICA lesson</a> looked for independent hidden sources in mixtures of signals. NMF asks for a different property: all the numbers used to build an observation must be nonnegative. Independence is not part of its basic objective.</Prose>
    <Prose>You need to multiply a row of numbers by a matrix and add squared errors. We will refresh both. A gradient means the local direction in which an error changes; the first fit can be followed before reading its derivation.</Prose>
    <Prose>Our convention is <strong>observations in rows, features in columns</strong>:</Prose>
    <LessonTable caption="Every matrix in this lesson, with its shape and meaning" headers={['Matrix', 'Shape', 'Meaning']} rows={[
      [<Math key="x">{'X'}</Math>, <Math key="xs">{'n\\times d'}</Math>, 'The observed nonnegative measurements'],
      [<Math key="w">{'W'}</Math>, <Math key="ws">{'n\\times k'}</Math>, 'How much each observation uses each component'],
      [<Math key="h">{'H'}</Math>, <Math key="hs">{'k\\times d'}</Math>, 'The component patterns in the original features'],
      [<Math key="r">{'\\widehat X=WH'}</Math>, <Math key="rs">{'n\\times d'}</Math>, 'Reconstructed measurements'],
    ]} />
    <Prose>The letter <Math>{'k'}</Math> is the <strong>number of components</strong>. People sometimes call it the factorization rank, although the product can have matrix rank smaller than <Math>{'k'}</Math>. A column of <Math>{'W'}</Math> follows one component across observations; a row of <Math>{'H'}</Math> follows that component across features. Some references transpose the whole convention. Check shapes before translating a formula.</Prose>
    <Prose>Consider three constructed observations with three features:</Prose>
    <MathBlock>{'\\begin{gathered}X=\\begin{bmatrix}2&1&3\\\\1&2&3\\\\3&3&6\\end{bmatrix},\\\\[6pt] W=\\begin{bmatrix}2&1\\\\1&2\\\\3&3\\end{bmatrix},\\\\[6pt] H=\\begin{bmatrix}1&0&1\\\\0&1&1\\end{bmatrix}.\\end{gathered}'}</MathBlock>
    <Prose>The first component contributes equally to features 1 and 3. The second contributes equally to features 2 and 3. Observation 1 is</Prose>
    <MathBlock>{'2[1,0,1]+1[0,1,1]=[2,1,3].'}</MathBlock>
    <Prose>To find one reconstructed cell, multiply corresponding entries and add:</Prose>
    <MathBlock>{'\\widehat X_{ij}=\\sum_{r=1}^{k}W_{ir}H_{rj}.'}</MathBlock>
    <Prose>For observation 1, feature 3, this is <Math>{'2\\cdot1+1\\cdot1=3'}</Math>. Every contribution is zero or positive. There is no cancellation between components.</Prose>
    <BuildRowFigure />
    <MixtureLab />

    <H3>The interpretation contract</H3>
    <Callout title="What nonnegativity does and does not buy you, once for the whole lesson">
      Nonnegativity guarantees additive reconstruction. Recognizable physical parts, sparse factors, independence and a unique explanation require additional assumptions or evidence. A component can be broad, overlap another component or combine unrelated physical processes. Calling it “eye,” “topic” or “material” is an interpretation to evaluate against its features and domain evidence. NMF weights are not probabilities unless a specified normalization gives them that meaning.
    </Callout>
    <Prose>This distinction also separates the recent lessons: a GMM responsibility is a normalized conditional probability of a mixture assignment; a t-SNE coordinate locates a point in a neighborhood map; an ICA coordinate estimates a source under independence assumptions; an NMF activation contributes to an additive reconstruction. None is a generic unit of “hidden meaning.”</Prose>

    <H2>{headings[1]}</H2>
    <Prose>Usually the observations do not fit a small number of patterns exactly. We therefore choose <Math>{'W,H\\geq0'}</Math> to minimize a reconstruction loss. For the <strong>squared Frobenius loss</strong>,</Prose>
    <MathBlock>{'\\begin{gathered}F(W,H)=\\tfrac12\\|X-WH\\|_F^2\\\\[4pt] =\\tfrac12\\sum_{i=1}^{n}\\sum_{j=1}^{d}(X_{ij}-\\widehat X_{ij})^2.\\end{gathered}'}</MathBlock>
    <Prose>The subscript <Math>{'F'}</Math> means: square every matrix entry, sum, then take the square root for the norm. We square that norm in the objective. The one-half cancels a factor of two when differentiating; it does not change which factors minimize the loss.</Prose>
    <Prose>If an observation is <Math>{'[2,1,3]'}</Math> and its reconstruction is <Math>{'[1.5,1,2.5]'}</Math>, the residual is <Math>{'[.5,0,.5]'}</Math>. Its contribution to <Math>{'F'}</Math> is <Math>{'\\tfrac12(.25+0+.25)=.25'}</Math>. Keep the residual signed when displaying it: positive means missing reconstructed mass; negative means excess.</Prose>
    <Prose>This objective values an absolute error of 2 equally at a feature value of 2 and a feature value of 20. Rescaling one feature by ten can make its squared error a hundred times more influential. Thus preprocessing is a modeling decision. Standard centering would introduce negative entries and change the additive interpretation. Dividing all image entries by the known maximum 16, as we do later, preserves zero and relative weights. Arbitrarily shifting a signed dataset until it is nonnegative introduces a baseline pattern the model must explain.</Prose>
    <ResidualFigure />

    <H3>A loss encodes which discrepancies matter</H3>
    <Prose>For a nonnegative observation <Math>{'x'}</Math> and positive reconstruction <Math>{'y'}</Math>, two other common entrywise losses are</Prose>
    <MathBlock>{'\\begin{gathered}d_{\\mathrm{KL}}(x,y)=x\\log(x/y)-x+y,\\\\[4pt] d_{\\mathrm{IS}}(x,y)=x/y-\\log(x/y)-1.\\end{gathered}'}</MathBlock>
    <Prose>The first is <strong>generalized Kullback–Leibler divergence</strong>, summed over entries. With the convention <Math>{'0\\log(0/y)=0'}</Math>, a zero observation contributes <Math>{'y'}</Math>. If <Math>{'x>0,y=0'}</Math>, the divergence is infinite. It becomes the usual KL divergence when the arrays are normalized probability distributions. The second is <strong>Itakura–Saito divergence</strong>, used for strictly positive entries here.</Prose>
    <Prose>Compare the same absolute overestimate:</Prose>
    <LessonTable caption="One absolute overestimate of 2, valued by three entrywise losses" headers={[<span key="p">Observed <Math>{'x'}</Math>, reconstructed <Math>{'y'}</Math></span>, <Math key="f">{'\\tfrac12(x-y)^2'}</Math>, 'Generalized KL', 'Itakura–Saito']} rows={[
      ['2, 4', '2', '.613706', '.193147'],
      ['20, 22', '2', '.093796', '.004401'],
    ]} />
    <Prose>KL and IS distinguish these two contexts. They still distinguish themselves: scaling both <Math>{'x'}</Math> and <Math>{'y'}</Math> by <Math>{'c>0'}</Math> scales squared loss by <Math>{'c^2'}</Math>, KL by <Math>{'c'}</Math>, and leaves IS unchanged. These follow by substituting into the formulas; “relative error” is too vague to describe all three.</Prose>
    <Prose>Independent Gaussian errors with common variance yield squared-error fitting of the means. Independent Poisson counts with means <Math>{'\\widehat X_{ij}'}</Math> yield generalized-KL fitting after terms independent of the factors are removed. Choosing the latter is an assumption about count variability, not a rule that every count dataset follows a Poisson model. TF-IDF values, for example, are weighted text features rather than integer counts.</Prose>
    <Prose>These losses belong to the beta-divergence family, with <Math>{'\\beta=2,1,0'}</Math> respectively. The library supports them, but its coordinate-descent solver uses Frobenius loss; the multiplicative solver supports the other beta losses. Strictly positive input is required for its <Math>{'\\beta\\leq0'}</Math> cases. <a href="https://scikit-learn.org/stable/modules/decomposition.html#nmf-with-a-beta-divergence">Scikit-learn’s NMF guide</a> documents those conventions.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>The difficult part is that both the component patterns and their amounts are unknown. Changing both at once creates a jointly nonconvex problem. A natural strategy is to alternate:</Prose>
    <Prose>1. Hold the activations <Math>{'W'}</Math> fixed and improve the patterns <Math>{'H'}</Math>. 2. Hold the newly updated patterns fixed and improve <Math>{'W'}</Math>. 3. Repeat while monitoring the objective and a stopping criterion.</Prose>
    <Prose>One especially transparent method uses <strong>multiplicative updates</strong>. For the squared loss above,</Prose>
    <MathBlock>{'\\begin{gathered}H\\leftarrow H\\odot\\frac{W^\\top X}{(W^\\top W)H},\\\\[6pt] W\\leftarrow W\\odot\\frac{XH^\\top}{W(HH^\\top)}.\\end{gathered}'}</MathBlock>
    <Prose>The symbol <Math>{'\\odot'}</Math> and the fraction mean entrywise multiplication and division. Ordinary adjacent matrix products still mean matrix multiplication. The second update uses the new <Math>{'H'}</Math>.</Prose>
    <Prose>Here is the reason for the ratio. The gradient with respect to <Math>{'H'}</Math> is</Prose>
    <MathBlock>{'\\nabla_HF=(W^\\top W)H-W^\\top X.'}</MathBlock>
    <Prose>The first term reflects the reconstruction currently produced; the second reflects the observed data. If the second is larger at a cell, the gradient is negative there, so increasing that cell can reduce loss. Multiplying by their ratio increases it. If the first term is larger, the ratio decreases it. A positive value multiplied by a nonnegative ratio stays nonnegative.</Prose>

    <H3>A full numerical step</H3>
    <Prose>Use the <Math>{'X'}</Math> from section 1, but initialize the unknown factors as</Prose>
    <MathBlock>{'\\begin{gathered}W^{(0)}=\\begin{bmatrix}1&.5\\\\.5&1\\\\1&1\\end{bmatrix},\\\\[6pt] H^{(0)}=\\begin{bmatrix}1&.2&.8\\\\.2&1&.8\\end{bmatrix}.\\end{gathered}'}</MathBlock>
    <Prose>The initial loss is {firstSweep.lossBefore.toFixed(2)}. To update <Math>{'H_{11}'}</Math>, the observed-data numerator is <Math>{'1\\cdot2+.5\\cdot1+1\\cdot3=5.5'}</Math>. The first row of <Math>{'W^\\top W'}</Math> is <Math>{'[2.25,2]'}</Math>, so the denominator is <Math>{'2.25\\cdot1+2\\cdot.2=2.65'}</Math>. Therefore <Math>{'H_{11}'}</Math> becomes <Math>{'1\\cdot5.5/2.65=2.075472'}</Math>.</Prose>
    <Prose>Updating all entries gives</Prose>
    <MathBlock>{'\\begin{gathered}H^{(1)}\\approx\\\\[4pt] {\\small\\begin{bmatrix}2.075472&.408163&2.470588\\\\.408163&2.075472&2.470588\\end{bmatrix}.}\\end{gathered}'}</MathBlock>
    <Prose>Using this new pattern matrix in the <Math>{'W'}</Math> update gives</Prose>
    <MathBlock>{'W^{(1)}\\approx\\begin{bmatrix}.826888&.393655\\\\.393655&.826888\\\\1.212145&1.212145\\end{bmatrix}.'}</MathBlock>
    <Prose>The loss after both updates is {firstSweep.loss.toFixed(7)}. Notice that some activations decreased even though the corresponding pattern entries increased. What matters is their product.</Prose>
    <UpdatePhaseFigure />
    <UpdateLab />

    <H3>Complete NumPy program</H3>
    <Prose>Create a Python environment with <Code>python -m pip install numpy</Code>. Save this as <Code>nmf_step.py</Code> and run <Code>python nmf_step.py</Code>. The example has strictly positive initial factors and no entirely zero data row or column, so its displayed update needs no added denominator constant. The supported numerical example is small and bounded.</Prose>
    <Program example={nmfExamples.multiplicativeStep}>
      <Prose>The bounded author calculation executed these updates with NumPy {versions.numpy}. The fit approaches the exact product from section 1, while its factor values need not match the chosen factors there. The next section explains why.</Prose>
    </Program>

    <H3>Why a flat error trace is not enough</H3>
    <Prose>A zero multiplied by a ratio remains zero. An entry initialized at exactly zero can become <strong>zero locked</strong> even when increasing it would improve the fit. For example, hold <Math>{'W=[1]'}</Math>, set <Math>{'X=[2,1]'}</Math> and <Math>{'H=[0,1]'}</Math>. The gradient for the first <Math>{'H'}</Math> entry is <Math>{'-2'}</Math>, so a positive move helps. A guarded multiplicative implementation that leaves zero entries at zero cannot make that move. By contrast, the nonnegative least-squares solution with this fixed <Math>{'W'}</Math> is exactly <Math>{'[2,1]'}</Math>.</Prose>
    <ZeroLockFigure />
    <Prose>The original update’s upper-bound argument establishes non-increasing objective values under its mathematical conditions. Objective decrease, stationarity, a local minimum and a global minimum are different statements. At a nonnegative boundary, stationarity permits a positive gradient at a zero variable: movement toward negative values is forbidden. A zero gradient everywhere is neither the correct boundary test nor a proof of a local minimum. <a href="https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf">Lin’s analysis</a>, sections II to IV, separates these issues and supplies modified updates with a convergence argument. We derive the relevant conditions in section 7.</Prose>
    <Prose>Adding an epsilon to every denominator or clipping factors after every step changes the algorithm. It can be useful numerical engineering, but the unmodified proof cannot simply be copied onto that changed procedure. For ordinary work, use a maintained solver with documented behavior and inspect its convergence information.</Prose>
    <Checkpoint prompt="Start H₁₁ at 3 instead of 1, keeping everything else the same. Does the first H phase grow it or shrink it, and why is that a better prediction task than watching H₁₁ from 1?">
      <Prose>The numerator does not involve <Math>{'H'}</Math> at all, so it stays 5.5. The denominator is <Math>{'2.25\\cdot3+2\\cdot.2=7.15'}</Math>, giving <Math>{'3\\times5.5/7.15=30/13\\approx2.307692'}</Math>: it shrinks. From 1 the same cell grows. A prediction task whose every supported answer is “increase” teaches nothing; the investigation above carries both cases as presets.</Prose>
    </Checkpoint>

    <H2>{headings[3]}</H2>
    <Prose>Even before considering local optimization, the data may admit several exact nonnegative factorizations. The factors from section 1 yield <Math>{'X'}</Math> exactly. So do</Prose>
    <MathBlock>{'\\begin{gathered}W_2=\\begin{bmatrix}1.5&.5\\\\.5&1.5\\\\2&2\\end{bmatrix},\\\\[6pt] H_2=\\begin{bmatrix}1.25&.25&1.5\\\\.25&1.25&1.5\\end{bmatrix}.\\end{gathered}'}</MathBlock>
    <Prose>Check the first row: <Math>{'1.5[1.25,.25,1.5]+.5[.25,1.25,1.5]=[2,1,3]'}</Math>. The second explanation has components that each contribute to every feature. They are not simply the first components with their order or scale changed. The observations alone have not identified which set of components is physically real.</Prose>
    <AmbiguityFigure />
    <Prose>Two simpler ambiguities always deserve attention. <strong>Permutation:</strong> exchange two rows of <Math>{'H'}</Math> and the corresponding columns of <Math>{'W'}</Math>; the product stays the same. <strong>Scale:</strong> multiply row <Math>{'r'}</Math> of <Math>{'H'}</Math> by any <Math>{'c>0'}</Math> and divide column <Math>{'r'}</Math> of <Math>{'W'}</Math> by <Math>{'c'}</Math>; every contribution stays the same.</Prose>
    <Prose>Consequently, a larger raw activation in one fit does not establish a stronger physical component than in another fit. First align component identities and choose a stated scale convention.</Prose>

    <H3>Normalize while preserving the product</H3>
    <Prose>Let <Math>{'s_r=\\sum_jH_{rj}>0'}</Math>. Define <Math>{'\\widetilde H_{rj}=H_{rj}/s_r'}</Math> and <Math>{'\\widetilde W_{ir}=W_{ir}s_r'}</Math>. Then <Math>{'\\widetilde W\\widetilde H=WH'}</Math>, and every pattern sums to one. A zero pattern contributes nothing and is handled separately rather than divided by zero.</Prose>
    <Prose>For the first observation of section 1, both pattern sums are 2. The normalized patterns are <Math>{'[.5,0,.5]'}</Math> and <Math>{'[0,.5,.5]'}</Math>; the activations become <Math>{'[4,2]'}</Math>. Their sum, 6, is the reconstructed total mass. Dividing these new activations by 6 gives mixture proportions <Math>{'[2/3,1/3]'}</Math> <strong>for the normalized reconstruction</strong>. The original activation vector <Math>{'[2,1]'}</Math> was not itself a probability distribution.</Prose>
    <Prose>This normalization preserves reconstruction, but generally changes a penalty on factor magnitudes. Apply it for inspection after an unpenalized fit, or explicitly account for it when the optimization includes regularization.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>Our <a href="/learn-assets/nmf/digits-300.csv" download>offline CSV</a> contains 300 real digit images: 30 examples of each label from the scikit-learn optical-digits collection. E. Alpaydin and C. Kaynak collected the underlying <a href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits">UCI Optical Recognition of Handwritten Digits dataset</a> for recognition research. Each image has 64 block counts arranged as 8×8; each count is an integer from 0 to 16. These are reduced handwritten bitmaps, not MNIST images. Attribution, selection and the CC BY 4.0 license are in <a href="/learn-assets/nmf/data-provenance.md">the data provenance</a>.</Prose>
    <Prose>The scientific question here is narrower than recognition: <strong>can a small additive dictionary reconstruct previously withheld images, and what do its patterns actually look like?</strong> Labels construct a balanced teaching collection and stratify the split; they are never factorization features. Writer identities are unavailable in this extract, so this split concerns withheld images within the collection, not a claim about new writers.</Prose>
    <Prose>We divide every block count by 16. Use {splitSizes.train} images to learn patterns, {splitSizes.validation} for validation comparisons, and reserve {splitSizes.test} for a final diagnostic. For a new row, <Code>transform</Code> finds its nonnegative activation amounts while keeping the fitted dictionary fixed. It is a small optimization problem, not multiplication by the dictionary transpose as in an orthogonal PCA projection.</Prose>
    <FitTransformFigure />
    <Prose>Install <Code>numpy</Code> and <Code>scikit-learn</Code> in your own environment, save the CSV beside this program as <Code>digits-300.csv</Code>, save the code as <Code>nmf_digits.py</Code>, and run <Code>python nmf_digits.py</Code>.</Prose>
    <CodeBlock language="bash">{'python -m pip install "numpy==2.3.5" "scikit-learn==1.9.1"'}</CodeBlock>
    <Program example={nmfExamples.digitDictionary}>
      <Prose>The author calculation executed the same data, split, fits and quantities under scikit-learn {versions.sklearn}. MSE means the average squared residual across all selected images and all 64 scaled features. Its units are squared fractions of the maximum block count. Here additional components improve validation reconstruction over the inspected range. The higher-rank runs also show an initialization effect. At one component, both seeds nearly agree: a useful null rather than a reason to invent variability.</Prose>
    </Program>
    <CandidateFigure />
    <Prose>The separately specified eight-component comparison gives test MSE {six(baselines.meanTestMse)} for the training-mean image, {six(baselines.pcaTestMse)} for PCA and {six(baselines.nmfTestMse)} for NMF. PCA wins this reconstruction comparison. It is allowed signed, centered patterns and uses an additional mean image; NMF supplies the additive constraint we wanted to inspect. This is a comparison of those representation choices, not equal storage bits or a digit-classification benchmark. There is no reason to tune the example until NMF wins.</Prose>
    <DictionaryFigure />
    <Prose>For the selected image, rank components by the sum of their actual contributions, not by raw <Math>{'W_{ir}'}</Math>. Removing a component changes the reconstruction by exactly its contribution image. You can now say whether a pattern is concentrated around a stroke, spreads over several regions, or overlaps another pattern. That is stronger evidence than naming every component a digit part in advance.</Prose>
    <ContributionLab />

    <H3>Choosing a useful component count</H3>
    <Prose>If your actual objective is validation MSE among these candidates, {bestRun.k} components with seed {bestRun.seed} is the best inspected candidate. The fixed eight-component panel was chosen for a readable demonstration, not mislabeled as the selected optimum. A genuine subsequent test evaluation would fit the chosen procedure according to its declared train/validation policy and evaluate once on the untouched test set. Repeatedly trying choices after viewing test results turns the test set into further validation.</Prose>
    <Prose>The best attainable unregularized training error cannot increase when another component is allowed: an old fit can be embedded by adding a zero component. A particular local solver run can break that visual trend by finding a worse solution. Neither an elbow nor a stable cluster assignment identifies a universal “true number of topics.” Choose a count using the task: reconstruction on withheld data, stable interpretable patterns, downstream usefulness and the cost of a larger dictionary. When comparing patterns across seeds, normalize and match them one-to-one before measuring similarity.</Prose>

    <H2>{headings[5]}</H2>
    <H3>Solvers and initialization</H3>
    <Prose>With <Math>{'H'}</Math> fixed, fitting each row of <Math>{'W'}</Math> is a nonnegative least-squares problem with <Math>{'k'}</Math> unknowns. There are <Math>{'n'}</Math> such row problems. With <Math>{'W'}</Math> fixed, fitting each column of <Math>{'H'}</Math> gives <Math>{'d'}</Math> problems. These are convex subproblems; alternating between them does not make the joint problem convex.</Prose>
    <Prose><strong>Coordinate descent</strong> updates one coefficient or block using the other current values. It can activate a zero entry when the feasible descent direction points into the positive region. A full alternating NNLS method solves each subproblem to an appropriate accuracy; one sweep of coordinate updates should not be described as an exact solve of every subproblem. Multiplicative updates offer an especially readable mechanism and support different losses. There is no solver ranking independent of matrix shape, sparsity, stopping criteria and requested accuracy.</Prose>
    <Prose>NNDSVD initializes nonnegative factors from singular-vector information. <Code>nndsvd</Code> retains zeros, <Code>nndsvda</Code> fills those zeros with the data mean, and <Code>nndsvdar</Code> uses small random fills. In the inspected scikit-learn version, <Code>init=None</Code> chooses <Code>nndsvda</Code> when the component count fits within the matrix dimensions, otherwise random initialization. Set parameters explicitly in a reproducible lesson. For multiplicative fitting, initial zeros deserve special attention because of zero locking. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html">NMF API</a>.</Prose>

    <H3>Sparsity is an additional preference</H3>
    <Prose>To prefer fewer active contributions or narrower patterns, add a penalty such as</Prose>
    <MathBlock>{'\\begin{gathered}F(W,H)+\\lambda_W\\sum_{ir}W_{ir}\\\\[4pt] +\\lambda_H\\sum_{rj}H_{rj}.\\end{gathered}'}</MathBlock>
    <Prose>For nonnegative factors these sums are their entrywise L1 norms. Penalizing <Math>{'W'}</Math> discourages an observation from spreading mass across many components; penalizing <Math>{'H'}</Math> discourages a component from spreading mass across many features. The effect depends on scale and the complete objective. A sparsity penalty is not a constraint that all component word lists become distinct.</Prose>
    <Prose>Scikit-learn exposes <Code>alpha_W</Code>, <Code>alpha_H</Code> and <Code>l1_ratio</Code>, including mixed L1/L2 penalties. Its objective scales the <Math>{'W'}</Math> penalty by the number of features and the <Math>{'H'}</Math> penalty by the number of samples, so a hand-written <Math>{'\\lambda'}</Math> is not automatically the same numerical parameter. The <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">regularization lesson</a> develops this statistical tradeoff after preprocessing and validation.</Prose>

    <H3>A diagnosis table</H3>
    <LessonTable caption="What to inspect when a factorization misbehaves" headers={['Observation', 'Inspect next', 'A justified response']} rows={[
      ['Input contains negative values', 'Units, centering and the intended meaning of zero', 'Choose a meaningful nonnegative representation or a method supporting signed values'],
      ['Objective barely changes', 'Initialization, stopping tolerance, gradients at active/boundary variables, iteration limit', 'Distinguish a good fit from boundary stalling; compare a documented alternate solver'],
      ['Similar top words or patterns', 'Full normalized factors and activations across observations', 'Check redundancy, a shared background and genuine overlap; do not infer an exact topic count from top words alone'],
      ['Different component numbers between runs', 'Permutation and scale matching', 'Compare contributions and aligned shapes'],
      ['Low reconstruction error but unhelpful factors', 'The task’s semantic or scientific diagnostics', 'Revisit representation, loss, constraints and count rather than rewarding error alone'],
      ['Sparse matrix becomes huge', 'Intermediate products and storage format', <span key="s">Keep data sparse and use small Gram products; avoid constructing <Math>{'WH'}</Math> merely to update factors</span>],
    ]} />

    <H2>{headings[6]}</H2>
    <Prose>This branch explains why alternating updates work, why general NMF remains hard, and when an additional geometric assumption helps. The core fit and interpretation do not depend on completing its proofs.</Prose>

    <H3>Separate convexity and joint nonconvexity</H3>
    <Prose>For fixed <Math>{'W'}</Math>, the Hessian of a column’s least-squares objective is <Math>{'W^\\top W'}</Math>, which is positive semidefinite because <Math>{'v^\\top W^\\top Wv=\\|Wv\\|^2\\geq0'}</Math>. The nonnegative feasible region is convex. The corresponding statement holds for rows of <Math>{'W'}</Math> with <Math>{'H'}</Math> fixed.</Prose>
    <Prose>For joint nonconvexity, take the scalar problem <Math>{'f(w,h)=\\tfrac12(1-wh)^2'}</Math>. Both <Math>{'(w,h)=(1,1)'}</Math> and <Math>{'(2,.5)'}</Math> have zero loss. Their midpoint <Math>{'(1.5,.75)'}</Math> has product {nonconvex.midpoint.product} and loss {nonconvex.midpoint.loss}. Convexity would require the midpoint loss to be at most zero. This directly tests the objective, rather than incorrectly concluding that any loss of a bilinear product must be nonconvex.</Prose>

    <H3>The majorization argument</H3>
    <Prose>For a column <Math>{'h'}</Math> of <Math>{'H'}</Math>, let <Math>{'A=W^\\top W'}</Math> and <Math>{'b=W^\\top x'}</Math>. Its objective has gradient <Math>{'Ah-b'}</Math> and Hessian <Math>{'A'}</Math>. Assume the fixed <Math>{'W'}</Math> has no all-zero component column and the current point <Math>{'h\''}</Math> is positive; then every <Math>{'(Ah\')_r>0'}</Math>. An all-zero component column is unused and must be removed or handled separately before this inverse-based derivation. Choose the diagonal matrix <Math>{'D_{rr}=(Ah\')_r/h\'_r'}</Math>. The quadratic function</Prose>
    <MathBlock>{'\\begin{gathered}G(h,h\')=F(h\')\\\\[4pt] +(h-h\')^\\top\\nabla F(h\')\\\\[4pt] +\\tfrac12(h-h\')^\\top D(h-h\')\\end{gathered}'}</MathBlock>
    <Prose>touches the objective at <Math>{'h\''}</Math> and lies above it. To see the key inequality, write <Math>{'z_r=h\'_ru_r'}</Math>. Since <Math>{'A'}</Math> is symmetric with nonnegative entries,</Prose>
    <MathBlock>{'\\begin{gathered}z^\\top(D-A)z\\\\[4pt] =\\tfrac12\\sum_{rs}A_{rs}h\'_rh\'_s(u_r-u_s)^2\\geq0.\\end{gathered}'}</MathBlock>
    <Prose>Minimizing this separable quadratic gives <Math>{'h=h\'-D^{-1}(Ah\'-b)=h\'\\odot b/(Ah\')'}</Math>, the multiplicative rule. Therefore <Math>{'F(h_{\\mathrm{new}})\\leq G(h_{\\mathrm{new}},h\')\\leq G(h\',h\')=F(h\')'}</Math>. This is an upper-bound minimization argument, and it is derived here for the squared-loss column problem only: the other beta divergences have their own multiplicative updates and their own auxiliary functions, so this derivation does not by itself carry over to them. EM in the earlier <a href="/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml">GMM lesson</a> used a lower bound while maximizing a log likelihood; the inequality direction changes with the optimization problem. <a href="https://papers.nips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf">Lee and Seung’s original algorithm paper</a>.</Prose>
    <Prose>For the nonnegative constraints, the first-order conditions are</Prose>
    <MathBlock>{'\\begin{gathered}W,H\\geq0,\\\\[4pt] \\nabla_WF\\geq0,\\quad \\nabla_HF\\geq0,\\\\[4pt] W\\odot\\nabla_WF=0,\\\\[4pt] H\\odot\\nabla_HF=0.\\end{gathered}'}</MathBlock>
    <Prose>A positive variable must have zero gradient; a zero variable must not have a negative gradient pointing into a feasible decrease. The zero-locked example violates the latter condition. Factor rescaling can also change the size of a gradient-based diagnostic without changing reconstruction, so a stopping metric needs a stated scale convention.</Prose>

    <H3>Nonnegative rank and an informative surprise</H3>
    <Prose>The ordinary rank of a matrix measures how many signed linear basis directions suffice for exact reconstruction. <strong>Nonnegative rank</strong> is the smallest <Math>{'k'}</Math> permitting an exact nonnegative factorization. It is at least ordinary rank and can be larger.</Prose>
    <Prose>Consider</Prose>
    <MathBlock>{'S=\\begin{bmatrix}0&0&1&1\\\\1&0&0&1\\\\1&1&0&0\\\\0&1&1&0\\end{bmatrix}.'}</MathBlock>
    <Prose>Its ordinary rank is 3: the sum of rows 1 and 3 equals the sum of rows 2 and 4, and the first three rows are independent. Its nonnegative rank is 4. Each nonnegative rank-one contribution has rectangular positive support and cannot place positive mass in one of the zero cells, since another contribution cannot cancel it. The positive positions <Math>{'(1,3),(2,4),(3,1),(4,2)'}</Math> cannot share a single such rectangle pairwise: for any pair, at least one crossed cell is zero. At least four rank-one contributions are necessary; taking <Math>{'W=I_4,H=S'}</Math> shows four suffice.</Prose>
    <SupportFigure />
    <Prose>General exact NMF includes NP-hard instances, as the canonical survey discusses. Nonconvexity alone would not prove NP-hardness. Structured cases can be much easier. Under a <strong>separability</strong> assumption, every needed component direction appears among the observed rows (after our row-oriented convention). With nonzero rows normalized to sum to one, all other rows lie in the convex hull of these anchor rows. Finding its extreme points can identify candidate patterns instead of searching for arbitrary hidden directions. Noise, redundant anchors and rank/conditioning assumptions matter to algorithmic guarantees.</Prose>
    <Prose>For example, observations <Math>{'[1,0],[0,1],[.25,.75],[.6,.4]'}</Math> visibly include the two endpoints. Removing both endpoints leaves many wider enclosing segments consistent with the remaining mixtures, just as in section 4. A general unconstrained NMF fit is not automatically a separable model. The survey’s geometric algorithms and their explicit assumptions are a useful next theoretical reading. <a href="https://arxiv.org/pdf/1401.5226">Gillis, <em>The Why and How of NMF</em></a>: section 3.2 for near-separable geometry, and section 4 for the connections that place nonnegative rank beside problems in mathematics and computer science.</Prose>

    <H2>{headings[7]}</H2>
    <H3>Documents as nonnegative word patterns</H3>
    <Prose>Suppose the vocabulary is <Code>[orbit, rocket, goal, team]</Code> and two component rows are <Math>{'[3,2,0,0]'}</Math> and <Math>{'[0,0,1,4]'}</Math>. A document activation <Math>{'[2,1]'}</Math> reconstructs <Math>{'[6,4,1,4]'}</Math>. The first component supplies most of the space-related words; the second supplies the sports-related words. A mixed document can use both without forcing a hard class assignment.</Prose>
    <Prose>Real text needs a vocabulary and weighting policy. <Code>CountVectorizer</Code> produces counts; <Code>TfidfVectorizer</Code> downweights words common across the fitted corpus. Vocabulary filtering and IDF estimation belong inside the training split when evaluating new-document behavior. Check actual text examples and full weight patterns, not just attractive top-word lists. Removing a domain word as a “stop word” can remove the signal you wanted to discover.</Prose>
    <Prose>Here is a complete constructed transfer example. Save it as <Code>nmf_words.py</Code>; it needs the same NumPy/scikit-learn installation as section 5.</Prose>
    <Program example={nmfExamples.wordPatterns}>
      <Prose>The exact vocabulary order is <Code>['goal', 'orbit', 'rocket', 'team']</Code>, so the four <em>columns</em> are alphabetical, while the two component <em>rows</em> come out in whichever order the fit produced. This program was declared unexecuted in the content phase; it has since been run under the same NumPy {versions.numpy} and scikit-learn {versions.sklearn} as the digit experiment, and the output above is that run rather than an expectation. Each normalized nonzero pattern sums to 1. Inspect whether the two patterns divide the vocabulary as expected and how the mixed document uses them, and notice which component the fit labels first: its component 1 is the sports pattern, the mirror image of the constructed component 1 above, which is permutation ambiguity from section 4 visible immediately. The mixed fifth document uses both components equally. The last line is the most instructive: the unseen pair <Code>rocket team</Code> reconstructs as <Code>[0.5, 0.5, 0.5, 0.5]</Code>, putting exactly as much mass on <Code>goal</Code> and <Code>orbit</Code> — two words that document never contained — as on the two it did. An additive dictionary rebuilds a new document out of whole components, so a word arrives with every other word its component carries. This tiny constructed corpus exposes the operation; the real-data evidence for this lesson remains the digit experiment.</Prose>
    </Program>
    <Prose>Probabilistic latent semantic analysis can express a normalized count table using a latent-topic mixture. The KL objective has a corresponding likelihood interpretation when totals and factors are normalized appropriately. Latent Dirichlet Allocation adds a hierarchical generative model with Dirichlet priors; normalizing arbitrary NMF factors after fitting does not supply that prior model or its posterior uncertainty. Nor does choosing LDA establish calibration of a scientific claim. The comparison is about modeling assumptions and the desired output, not a universal speed ranking.</Prose>

    <H3>A spectrum can combine materials, with explicit assumptions</H3>
    <Prose>In a simple linear mixing model, a measurement across three wavelength bands might be <Math>{'.3[.2,.6,.4]+.7[.8,.3,.1]=[.62,.39,.19]'}</Math>. The two vectors describe material spectra, and the coefficients are nonnegative amounts. If physics and calibration justify abundance fractions, impose a sum-to-one constraint as well; unconstrained NMF does not supply it.</Prose>
    <Prose>This connection explains both the appeal of NMF and the need for separability or other information: if a pure material is observed, its spectrum can anchor the mixture geometry. If every observation is mixed, several dictionaries may explain it. Nonlinear light interactions, an unknown background and wavelength-dependent measurement uncertainty may require a richer model. Weighted least squares uses each measurement’s uncertainty to set its influence; equal Frobenius weights silently assume equal precision. The hyperspectral treatment and environmental-factorization references in the <a href="https://arxiv.org/pdf/1401.5226">canonical survey</a> provide the documented application context; the three-band numbers here are a constructed illustration.</Prose>
    <Prose>For audio, the signed waveform first becomes a nonnegative magnitude or power spectrogram. Factor rows can represent frequency patterns and activations can vary across time, depending on orientation. Reconstructing a usable waveform additionally requires a treatment of phase and source assignment; magnitude addition is a modeling approximation, not the same exact linear model used for instantaneous ICA. These domain details belong in the source-separation lesson rather than being hidden inside the word “isolate.”</Prose>

    <H3>What larger matrices cost</H3>
    <Prose>With dense <Math>{'X'}</Math>, efficient Frobenius updates cost on the order of <Math>{'ndk+(n+d)k^2'}</Math> per alternating sweep. For the digit fit above, with <Math>{'n=180'}</Math>, <Math>{'d=64'}</Math> and <Math>{'k=8'}</Math>, that is about {cost.dense.toLocaleString('en-US')} operations per sweep, and the two factors together hold {cost.factorStorage.toLocaleString('en-US')} numbers. Compute denominators as <Math>{'(W^\\top W)H'}</Math> and <Math>{'W(HH^\\top)'}</Math>, rather than first forming the full <Math>{'n\\times d'}</Math> reconstruction. If <Math>{'X'}</Math> has <Math>{'s'}</Math> stored nonzeros, the data products can use roughly <Math>{'sk'}</Math> work, while the factor and Gram terms remain. This advantage is available to properly organized multiplicative updates as well as coordinate methods.</Prose>
    <Prose>Sparse input does not make the dense factors free. Storing <Math>{'W,H'}</Math> costs <Math>{'k(n+d)'}</Math> numbers; computing every residual still touches many reconstructed entries unless a suitable algebraic objective computation is used. Do not infer elapsed seconds from these operation counts.</Prose>
    <Prose>In an online dictionary method, encode a batch using the current dictionary, then update the dictionary using information retained from past batches. For squared loss, sums of activation outer products and data–activation products form sufficient quadratic statistics for the fixed past encodings. In our orientation those statistics have shapes <Math>{'k\\times k'}</Math> and <Math>{'d\\times k'}</Math>. They compress old data contributions for this update, while changing encodings or the data distribution introduces further choices. Forgetting factors reduce the weight of older batches.</Prose>
    <Prose><Code>MiniBatchNMF</Code> exposes a maintained incremental route; its batch size, loss and convergence behavior should be selected from its current documentation and measured for the real task. General online dictionary-learning results have assumptions and do not automatically apply to every beta divergence or streaming implementation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html">MiniBatchNMF documentation</a>.</Prose>
    <Prose>Tensor factorization extends this idea to data with three or more axes, such as time × frequency × sensor. Keeping those axes can preserve relationships lost by flattening them. CP and Tucker impose different factor structures; they require their own tensor-shaped derivations. They are optional specialist extensions, not prerequisites for the matrix workflow here.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>Attempt the task before opening a hint or solution. The numeric cases differ from the running fit.</Prose>
    <Practice title="1. Reconstruct and preserve scale"
      question={<>With <Math>{'H=[[2,0,1],[0,1,2]]'}</Math> and <Math>{'w=[1,3]'}</Math>, reconstruct the observation. Normalize each pattern to sum to one while preserving the product. What are the normalized mixture proportions and reconstructed total?</>}
      hint="Multiply each activation by its pattern’s old row sum when dividing that pattern by the same sum.">
      <Prose>The reconstruction is <Math>{'[2,3,7]'}</Math>. Both patterns sum to 3, so normalized patterns are <Math>{'[2/3,0,1/3]'}</Math> and <Math>{'[0,1/3,2/3]'}</Math>, with new activations <Math>{'[3,9]'}</Math>. Their total is 12 and their normalized proportions are <Math>{'[1/4,3/4]'}</Math>. The third feature is large because both patterns contribute to it, not because the observation belongs to a third component.</Prose>
    </Practice>
    <Practice title="2. One coordinate’s update"
      question={<>Hold <Math>{'W=[[1],[2]]'}</Math>, use <Math>{'X=[[2,1],[4,3]]'}</Math>, and initialize <Math>{'H=[[1,1]]'}</Math>. Compute one Frobenius multiplicative update of <Math>{'H'}</Math>. Does it solve the fixed-<Math>{'W'}</Math> least-squares problem in this special case?</>}
      hint={<>Here <Math>{'W^\\top W'}</Math> is the scalar 5; calculate both entries of <Math>{'W^\\top X'}</Math>.</>}>
      <Prose>The numerator is <Math>{'[10,7]'}</Math> and the denominator is <Math>{'[5,5]'}</Math>, giving <Math>{'H=[2,1.4]'}</Math>. Reconstruction becomes <Math>{'[[2,1.4],[4,2.8]]'}</Math>, with half-squared Frobenius loss <Math>{'.1'}</Math>. Each feature has one positive coefficient with unconstrained optimum <Math>{'b/5'}</Math>, so this update reaches the NNLS optimum for fixed <Math>{'W'}</Math>. That one-dimensional coincidence does not make a general simultaneous update an exact solution of a multivariable NNLS problem.</Prose>
    </Practice>
    <Practice title="3. A misleading component claim"
      question="A colleague shows two nonnegative factors with low residual error and says, “We have proved there are exactly six biological sources; the largest activation identifies the strongest source.” Identify three missing pieces of reasoning and propose evidence to collect."
      hint="Separate component count, scientific interpretation, and the scale ambiguity.">
      <Prose>Six was a chosen factor count and needs a task-based comparison with other counts. Nonnegative components need biological validation, such as agreement with independent markers or controlled mixtures, before being called physical sources. Raw activations depend on dictionary scale; inspect normalized contributions using meaningful measurement units. A defensible study would record those choices, stability across fits and samples, withheld reconstruction or downstream outcomes, and external biological evidence. Merely reducing residual error addresses only the reconstruction question.</Prose>
    </Practice>
    <Practice title="4. Investigate an actual held-out image"
      question="In the digit fit, choose a test image other than the first. Compute each component’s total contribution, remove the largest contributor, and report the before/after MSE for that image. Predict which pixels will lose reconstructed intensity before calculating the removal."
      hint={<>For selected row <Math>{'i'}</Math>, the contribution totals are <Code>W_test[i] * H.sum(axis=1)</Code>. Remove the chosen rank-one contribution from the reconstructed row; leave the remaining amounts fixed. The investigation in section 5 does exactly this for any of the 60 reserved images.</>}>
      <Prose>The removed image is exactly <Code>W_test[i, r] * H[r]</Code>. Every removed pixel contribution is nonnegative. Some overpredicted pixels can move closer to their observations, but at an exact rowwise NNLS optimum the total row MSE cannot improve by removing a component: setting its coefficient to zero was already a feasible choice. Numerical fits meet that expectation only to their optimization accuracy. Report the source-row ID, selected component, actual contribution vector and both MSE values. Check that the difference between before and after equals the contribution elementwise. Explain which individual pixel errors improve and which worsen. A component with zero activation supplies an exact unchanged case. This task holds the other contributions fixed; reoptimizing them would be a separate comparison.</Prose>
    </Practice>
    <Practice title="5. Repair an evaluation pipeline"
      question="A document study fits TF-IDF and NMF on all documents, divides the activations into train/test, and reports a classifier score on the latter as evidence for new-document performance. Repair the order. What should happen to a completely unseen word at inference?"
      hint="Identify every object whose learned state used information from the test documents.">
      <Prose>Split documents first under the intended deployment unit. Fit vocabulary, IDF and NMF dictionary on training documents only, then train the classifier using those activations. Apply the frozen vectorizer and dictionary to validation/test documents. A word absent from the fitted vocabulary contributes no feature in this fixed representation; inspect how often that occurs and whether it makes a document’s representation uninformative. If model selection is repeated, fit the entire pipeline separately within each training fold. The next two lessons develop preprocessing and cross-validation in detail.</Prose>
    </Practice>
    <Practice title="6. Deeper: test joint convexity with different points"
      question={<>For <Math>{'f(w,h)=\\tfrac12(2-wh)^2'}</Math>, compare <Math>{'(1,2)'}</Math>, <Math>{'(4,.5)'}</Math> and their midpoint. Use the result to test convexity.</>}>
      <Prose>Both endpoints have zero loss. The midpoint is <Math>{'(2.5,1.25)'}</Math>, whose product is 3.125 and loss is <Math>{'\\tfrac12(1.125)^2=.6328125'}</Math>. This exceeds the average endpoint loss, contradicting convexity. The test concerns the actual squared-loss function.</Prose>
    </Practice>
    <Practice title="7. Deeper: choose a loss by its scaling behavior"
      question="A positive spectral observation and its prediction are both multiplied by 3 because of a common gain change. Derive how Frobenius, KL and IS losses change. Which comparison is gain-invariant?">
      <Prose>The residual triples, so squared loss increases by 9. In KL, the log ratio is unchanged and both outside linear terms triple, so loss increases by 3. In IS only the ratio appears, so the loss is unchanged. IS is gain-invariant for this joint rescaling. Whether that is desirable depends on whether absolute signal strength carries information for the task.</Prose>
    </Practice>
    <Practice title="8. Deeper: boundary stationarity"
      question={<>For fixed <Math>{'W=[1]'}</Math>, <Math>{'X=[3,1]'}</Math> and <Math>{'H=[0,2]'}</Math>, compute <Math>{'\\nabla_HF'}</Math>. Which coordinates violate the nonnegative first-order conditions, and how would a feasible improving move behave?</>}>
      <Prose>The gradient is <Math>{'[-3,1]'}</Math>. The zero first coordinate has negative gradient: increasing it is a feasible descent move, so it violates stationarity. The positive second coordinate has nonzero gradient: decreasing it slightly is also feasible and improves loss. The fixed-<Math>{'W'}</Math> optimum is <Math>{'[3,1]'}</Math>. A multiplicative zero-locked first coordinate could remain wrong even while the second coordinate improves.</Prose>
    </Practice>

    <H2>{headings[9]}</H2>
    <Prose>For the core route, you should be able to trace a single reconstructed cell, explain one ratio update, preserve a product under factor normalization, fit a frozen dictionary to new observations, and interpret an actual residual without assuming that a component is a physical source. The deeper route adds the nonnegative first-order conditions, the difference between ordinary and nonnegative rank, and the extra assumption behind anchor-based recovery.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Trace one reconstructed cell back to its two products', 'Section 1, the additive figure, the mixture investigation'],
      ['Read a signed residual and say why KL and IS value it differently', 'Section 2, the residual figure, practice 7'],
      ['Substitute one multiplicative ratio and say which way the cell moves', 'Section 3, the update worksheet and investigation, practice 2'],
      ['Preserve a product while normalizing patterns to sum to one', 'Section 4, practice 1'],
      ['Reproduce the held-out digit comparison and report it as it came out', 'Section 5, the candidate figure, practice 4'],
      ['Separate objective decrease from stationarity and from a local minimum', 'Sections 3 and 7, the zero-lock contrast, practice 8'],
      ['Explain why nonnegative rank can exceed ordinary rank', 'Section 7, the support-rectangle figure'],
    ]} />
    <Prose>Next is <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">Feature Scaling, Encoding &amp; Imputation</a>. NMF made preprocessing consequences visible: centering changes signs, feature scaling changes squared-error influence, and a learned transformation must be fitted within the intended information boundary. The next lesson builds those choices into a complete representation pipeline.</Prose>
    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://arxiv.org/pdf/1401.5226">Nicolas Gillis, <em>The Why and How of Nonnegative Matrix Factorization</em></a> — freely available survey/chapter. Read the image/text examples for another visual explanation, then section 3 for algorithms and section 3.2 for separable geometry. The survey uses observations in columns, so transpose the matrix convention when comparing it with this lesson. Its numerical benchmark is historical, not a prediction of current library timings.</li>
      <li><a href="https://papers.nips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf">Lee and Seung, <em>Algorithms for Non-negative Matrix Factorization</em></a> — original seven-page algorithm paper. Best after the ratio walkthrough; it develops both Euclidean and generalized-KL updates and the auxiliary-function argument. Read it together with the next reference for precise convergence distinctions.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html">Scikit-learn’s example comparing decomposition patterns</a> — an alternate visual activity: compare what different constraints make visible in the same image collection. Its example requires its own dataset retrieval; this lesson supplies a separate offline digit dataset.</li>
    </ul></>}>
      <li><a href="https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf">Chih-Jen Lin, <em>On the Convergence of Multiplicative Update Algorithms for Non-negative Matrix Factorization</em></a> — deeper mathematical reading on boundary behavior, first-order conditions and modified updates. Requires comfort with gradients and limit points; section II explains the central issue before the proof.</li>
      <li><a href="https://scikit-learn.org/stable/modules/decomposition.html#nmf">Scikit-learn decomposition guide</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html">NMF API</a> — current parameter semantics, losses, initialization and transformation behavior, inspected as version {versions.sklearn}. Use these when reproducing the programs rather than copying defaults from an older tutorial.</li>
      <li><a href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits">Optical Recognition of Handwritten Digits, UCI</a> — data collection, feature construction and attribution, licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. See the <a href="/learn-assets/nmf/data-provenance.md">local provenance</a> before reusing or exporting the provided subset.</li>
    </Sources>
    <Prose>The linked papers and substantive documentation were read for this lesson. No video is required to complete its explanations or exercises; the inspected visual example and canonical chapter provide alternate learning routes. The three-by-three matrices, both exact factorizations, the loss comparison, the zero-lock case, the nonnegative-rank matrix, the three-band spectrum and the five-document corpus are constructed fixtures with declared inputs. The digit results are calculations on the identified real dataset under one fixed split, with <Code>{`k = ${fit.k}, seed = ${fit.seed}`}</Code> declared in advance for readable image panels. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>,
};

export default nmfContent;
