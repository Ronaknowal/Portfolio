import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  AcquisitionLab, DeploymentLab, HalvingLab, MixtureLab, SearchReplayLab, SearchSpaceLab,
} from '../../components/lesson-labs/AutomlLabs.jsx';
import {
  ArchitectureFigure, BilevelFigure, EvidenceLoopFigure, ObservedResultsFigure, WeightProvenanceFigure,
} from '../../components/lesson-labs/AutomlFigures.jsx';
import { automlExamples } from '../automl-examples.js';
import {
  candidates, estimatorFits, foldRows, inspection, majorityBaseline, provenance, roles, sourceRows,
  versions,
} from '../automl-data.js';
import {
  activationKernel, bilevelDerivatives, countConfigurations, declaredCurves, declaredMixtureInputs,
  declaredResource, declaredSpace, expectedImprovement, halvingSchedule, hyperbandBrackets,
  mixtureState, networkBlocks, paretoAnalysis, portfolioAnalysis, replayPrefix, samplingMeasure,
} from '../automl-models.js';

/** Print a computed number with a typographic minus sign and no float dust.
 *
 * Note that this module imports the page's `Math` rendering component, which
 * shadows the global `Math` object. Nothing here may call `Math.log` and the
 * like; every such constant is computed in automl-models.js instead.
 */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');

const development = roles.find(role => role.role === 'development');
const inspectionRole = roles.find(role => role.role === 'inspection');
const reservedRole = roles.find(role => role.role === 'reserved');
const selected = inspection.find(entry => entry.role === 'selected');
const baseline = inspection.find(entry => entry.role === 'declared_baseline');
const space = countConfigurations(declaredSpace);
const halving = halvingSchedule();
const brackets = hyperbandBrackets(9, 3);
const scalar = bilevelDerivatives({ w: 0, alpha: 0.2, xi: 0.1 });
const stationaryScalar = bilevelDerivatives({ w: 0.2, alpha: 0.2, xi: 0.1 });
const practiceScalar = bilevelDerivatives({ w: 0, alpha: 0.5, xi: 0.2, valTarget: 2 });
const mixture = mixtureState(declaredMixtureInputs);
const practiceMixture = mixtureState({
  operations: [
    { id: 'up', label: 'first operation', formula: 'o(x) = 3x', evaluate: x => 3 * x },
    { id: 'down', label: 'second operation', formula: 'o(x) = −x', evaluate: x => -x },
  ],
  active: ['up', 'down'], logits: [0, 0], x: 1, target: 0, step: 0,
});
const kernel = activationKernel(['110', '101']);
const singularKernel = activationKernel(['110', '110']);
const portfolio = portfolioAnalysis();
const pareto = paretoAnalysis();
const replayTwo = replayPrefix({ budget: 2 });
const replayThree = replayPrefix({ budget: 3 });
const replayFour = replayPrefix({ budget: 4 });
const familyMeasure = samplingMeasure(declaredSpace, 'family-uniform');
const configurationMeasure = samplingMeasure(declaredSpace, 'configuration-uniform');

const headings = [
  '1. Begin with the experiment, before the optimizer',
  '2. A search space is a small language of valid experiments',
  '3. Decide how much evidence to buy for each candidate',
  '4. Neural architecture search, with the neural part explained locally',
  '5. A real search you can inspect end to end',
  '6. Turn search results into a useful learning system',
  '7. Deeper: differentiable search and cheaper architecture evidence',
  '8. Optional: translate the contract into maintained tools',
  '9. Practice: design, calculate, and diagnose',
  '10. Connections and other ways to learn',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <Prose>Install what it imports, then run it beside the data file:</Prose>
    <CodeBlock language="bash">{example.setup}</CodeBlock>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}
function OptionalProgram({ example, children }) {
  return <section className="am-optional-program">
    <H3>{example.title}</H3>
    <p className="lesson-note">Save as <Code>{example.file}</Code>, beside the previous files.</p>
    <CodeBlock language="bash">{example.setup}</CodeBlock>
    <CodeBlock language="python">{example.code}</CodeBlock>
    {children}
  </section>;
}
function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="am-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const automlContent = {
  title: 'AutoML & Neural Architecture Search (NAS)',
  readTime: '~55 min first pass · ~105 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot am-lesson">
    <LessonIntro prerequisites={<>A fitted pipeline, a cross-validation split and a held-out comparison, from <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a>. Scaling from <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">Feature Scaling, Encoding &amp; Imputation</a>, penalty strength from <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization</a>, and the error-cost distinction from <a href="/learn/path/full-curriculum/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml">Imbalanced Learning</a>. <strong>No prior neural-network course is required:</strong> section 4 defines a unit, an activation, a layer and a parameter count before anything depends on them.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      You specify what counts as a valid candidate, how candidates will be judged, and how much work is allowed; a search procedure does the rest. You will build a conditional search space and count it correctly, decide what a cheap evaluation is worth, meet the neural machinery locally, then read a real {estimatorFits}-fit study on {provenance.rows.toLocaleString('en-US')} banknote measurements in which the winning network is perfect on its folds and three other candidates tie by making the same single mistake. The investigations update their calculations and visual explanations as you change valid inputs.
    </LessonIntro>
    <div className="am-route"><Prose><strong>First pass.</strong> Read sections 1–5 in order, including the small neural bridge in section 4, and try practices 1–6. You will build a valid search space, understand two ways to spend its budget, and interpret a real experiment. Section 6 turns the result into a workflow. <strong>Section 7 is a deeper branch</strong> on differentiable search and cheap architecture proxies; section 8 is optional library translation. You can stop after section 6 and be ready for the next topic.</Prose></div>

    <Prose>You have a dataset, several reasonable models, and an afternoon. A logistic model might work. A tree might need less preprocessing. A small neural network might capture a useful interaction. Each choice brings more choices: which columns to transform, how much to regularize, how large a model to fit, and when to stop trying alternatives.</Prose>
    <Prose><strong>Automated machine learning, or AutoML, organizes and carries out some of these experiments.</strong> You specify what counts as a valid candidate, how candidates will be judged, and how much work is allowed. A search procedure proposes candidates; an evaluator fits and scores them; a record of the results guides the next decision. Neural architecture search, or NAS, applies this idea to the structure of a neural network.</Prose>
    <Prose>The most useful question is not &ldquo;Can software find the best model?&rdquo; It is <strong>&ldquo;What decisions have I permitted it to make, and what evidence will justify its recommendation?&rdquo;</strong> That question connects everything you have just learned about preprocessing, validation, regularization, feature selection, and imbalanced learning.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>Imagine an inspection system that assigns class 0 or class 1 to a measured object. An AutoML system can optimize classification accuracy if that is the criterion you give it. It cannot infer that missing one kind of object costs twelve times as much as reviewing another. The <a href="/learn/path/full-curriculum/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml">previous lesson on imbalanced learning</a> explained why that distinction can change both the model and the decision threshold.</Prose>
    <Prose>Write the following contract before examining search results:</Prose>
    <LessonTable caption="The experiment contract for this lesson's study, fixed before any candidate is fitted" headers={['Decision', "A concrete answer for this lesson's study"]} rows={[
      ['Prediction task', 'Predict the original numeric class of a banknote feature vector.'],
      ['Evidence available at prediction time', 'Four supplied numeric image descriptors.'],
      ['Candidate choices', `${space.total} declared preprocessing/model configurations.`],
      ['Selection criterion', `Highest arithmetic mean of ${foldRows.length} validation-fold accuracies.`],
      ['Separation rule', 'Identical feature vectors stay in the same data role and CV fold.'],
      ['Work allowance', `${foldRows.length} fits per candidate, then two declared refits.`],
      ['Final comparison', 'Selected candidate versus a predeclared logistic baseline on a separate inspection partition.'],
      ['Protected evidence', 'A further reserved partition receives no predictions.'],
    ]} />
    <Callout title="What the objective is, and is not">
      Accuracy is a declared educational objective here, not a claim that all real authentication mistakes have equal
      consequences. A deployment contract would also specify the consequences of errors, acceptable latency, expected
      data sources, and what happens when the input is unsuitable.
    </Callout>
    <EvidenceLoopFigure />

    <H3>What is being optimized?</H3>
    <Prose>A <em>configuration</em> describes choices made outside ordinary model fitting: a model family, preprocessing steps, regularization strength, or hidden-layer widths. Model fitting then estimates the numerical parameters for that configuration. For logistic regression, the regularization strength belongs to the configuration; the fitted coefficients are model parameters.</Prose>
    <Prose>Let <Math>{'\\lambda'}</Math> describe a valid configuration and <Math>{'\\mathcal A_\\lambda'}</Math> its complete fitting procedure. Write the loss of fold <Math>{'k'}</Math> as <Math>{'\\ell_k'}</Math>. A cross-validation objective to minimize is</Prose>
    <MathBlock>{'\\begin{gathered}\\ell_k=L\\!\\left(\\mathcal A_\\lambda(D_{-k}),D_k\\right),\\\\[6pt]\\hat f(\\lambda)=\\frac1K\\sum_{k=1}^{K}\\ell_k.\\end{gathered}'}</MathBlock>
    <Prose>Here <Math>{'D_{-k}'}</Math> contains the fitting rows for fold <Math>{'k'}</Math>, <Math>{'D_k'}</Math> contains that fold&rsquo;s validation rows, and <Math>{'L'}</Math> measures prediction error. For accuracy, use <Math>{'L=1-\\text{accuracy}'}</Math>. The selected configuration minimizes this estimated error within the permitted space. This problem is often called <em>combined algorithm selection and hyperparameter optimization</em>, abbreviated <strong>CASH</strong>. It does not require any particular optimizer.</Prose>
    <Prose>The hat on <Math>{'\\hat f'}</Math> matters. It is an estimate influenced by the dataset, split, random seed, and fitting procedure. Searching more configurations can improve the best observed validation score while exploiting more of that estimate&rsquo;s noise. A separate assessment evaluates the selected procedure; it is not another search surface. The <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">cross-validation lesson</a> develops nested evaluation when you need to assess the whole selection procedure across outer folds.</Prose>
    <Prose>For unequal fold sizes, an unweighted mean of fold accuracies and accuracy pooled over all out-of-fold predictions differ slightly. Either may be a deliberate objective. State which one determines selection, and use the same rule for every candidate. Our experiment records both and selects by the mean of fold accuracies.</Prose>

    <H3>The boundary contains preprocessing too</H3>
    <Prose>&ldquo;I only used the validation labels at the end&rdquo; is insufficient. Fitting a scaler, imputer, feature selector, or learned encoder on all rows can let validation information enter earlier. The fitting procedure <Math>{'\\mathcal A_\\lambda'}</Math> includes these operations. In each fold, fit the complete pipeline on that fold&rsquo;s fitting rows, then transform its validation rows using the fitted objects.</Prose>
    <Prose>If observations share an object, patient, account, experimental run, or exact duplicated feature record, an ordinary row split may put closely related evidence on both sides. Group according to the deployment question. Chronological forecasting needs a time-respecting design. AutoML can execute such a design; selecting the right design remains part of the scientific problem.</Prose>

    <H2>{headings[1]}</H2>
    <Prose>Suppose a form asks for a model family, tree depth, neighbor count, and neural-network width. Most combinations make no sense. A logistic model has no tree depth. A one-layer network has no second-layer width. A good search space encodes those dependencies rather than evaluating an enormous table of meaningless combinations.</Prose>
    <Prose>For our study, the grammar is:</Prose>
    <CodeBlock language="text">{automlExamples.searchSpaceGrammar.code}</CodeBlock>
    <Prose>There are <Math>{'2\\times2+2+2+3=11'}</Math> configurations. Multiplying every option count together would treat inactive choices as real and count configurations that do not exist.</Prose>
    <SearchSpaceLab />
    <Prose>This grammar also documents a limitation. Our search cannot discover an SVM, a new feature extraction method, or a network with three hidden layers. Even exhaustive search is exhaustive only within its specified language.</Prose>

    <H3>The sampling rule expresses a preference</H3>
    <Prose>For {space.total} inexpensive configurations, enumeration is transparent. Larger spaces often use random sampling. That still requires a distribution.</Prose>
    <Prose>Suppose a positive regularization parameter spans <Math>{'10^{-4}'}</Math> to <Math>{'10^{2}'}</Math>. Sampling its numeric value uniformly allocates almost all probability to the largest decades. Sampling</Prose>
    <MathBlock>{'\\begin{gathered}u\\sim\\operatorname{Uniform}(-4,2),\\\\[4pt]\\lambda=10^u\\end{gathered}'}</MathBlock>
    <Prose>gives each decade equal probability. This is useful when ratios, rather than equal absolute increments, express comparable changes. It is a modeling choice, not a rule for every parameter: integer depths and probabilities often need other distributions.</Prose>
    <Prose>The family selection rule matters too. Uniformly choosing one of {familyMeasure.bands.length} families and then a valid setting gives each family {familyMeasure.bands[0].fraction} of the trials. Uniformly choosing one of the {space.total} configurations gives logistic regression {configurationMeasure.bands[0].fraction} of the trials and a tree {configurationMeasure.bands[1].fraction}. Neither is &ldquo;unbiased&rdquo; without specifying the intended reference measure. The investigation above draws both measures as proportional bands once you have applied an edit.</Prose>

    <H3>Learning where to try next</H3>
    <Prose>Random search ignores observed scores when proposing the next configuration. <strong>Bayesian optimization</strong> builds a predictive model, called a <em>surrogate</em>, of the objective and uses that model to choose an evaluation. Its uncertainty describes uncertainty about a candidate&rsquo;s objective under its assumptions; it is not the candidate classifier&rsquo;s probability for an individual example.</Prose>
    <Prose>A common acquisition rule is <strong>expected improvement</strong>. If the best observed loss is <Math>{'b'}</Math> and a surrogate treats a candidate&rsquo;s unknown loss <Math>{'F'}</Math> as random, improvement is <Math>{'\\max(b-F,0)'}</Math>. The acquisition value is</Prose>
    <MathBlock>{'\\operatorname{EI}=\\mathbb E[\\max(b-F,0)].'}</MathBlock>
    <Prose>Consider a constructed surrogate with <Math>{'b=0.4'}</Math>:</Prose>
    <LessonTable caption="A constructed surrogate comparison. These are surrogate predictions about an unknown objective, not observed losses." headers={['Candidate', 'Predicted mean loss', 'Predicted standard deviation', 'Expected improvement']} rows={[
      ['A', '0.35', '0.02', num(expectedImprovement(0.4, 0.35, 0.02))],
      ['B', '0.40', '0.20', num(expectedImprovement(0.4, 0.40, 0.20))],
      ['C', '0.50', '0', num(expectedImprovement(0.4, 0.50, 0))],
    ]} />
    <Prose>B has a worse predicted mean than A but a larger expected improvement: its uncertain lower-loss possibilities compensate for its other possibilities under this acquisition rule. This is a specific exploration decision, not a promise that B will actually perform better.</Prose>
    <AcquisitionLab />
    <Prose>For a Gaussian surrogate prediction <Math>{'F\\sim\\mathcal N(\\mu,\\sigma^2)'}</Math>, write <Math>{'z=(b-\\mu)/\\sigma'}</Math>. Let <Math>{'\\Phi(z)'}</Math> be the probability that a standard normal variable is at most <Math>{'z'}</Math>, and <Math>{'\\phi(z)=e^{-z^2/2}/\\sqrt{2\\pi}'}</Math> its density. Integrating <Math>{'(b-f)'}</Math> over the part of the density below <Math>{'b'}</Math> gives</Prose>
    <MathBlock>{'\\begin{gathered}\\operatorname{EI}=(b-\\mu)\\Phi(z)+\\sigma\\phi(z),\\\\[4pt]\\sigma>0.\\end{gathered}'}</MathBlock>
    <Prose>The first term measures mean advantage weighted by the probability of improvement; the second accounts for uncertainty in the lower tail. When <Math>{'\\sigma=0'}</Math>, use <Math>{'\\max(b-\\mu,0)'}</Math>. Real classification losses are bounded, so a Gaussian predictive approximation can place some probability outside their physical range. The example exposes the decision calculation without asserting that this approximation is always appropriate.</Prose>
    <Prose>Gaussian processes are one possible surrogate, developed <a href="/learn/path/full-curriculum/gaussian-processes-gp?module=classical-ml">later in this module</a>. Tree-based surrogates and density-estimation approaches also support search. Conditional spaces require an appropriate representation of inactive parameters; Gaussian processes are not inherently forbidden from such spaces. Random search, evolutionary methods, and sequential surrogates should be compared using the same space, resources, and evaluation protocol. An optimizer cannot rescue a search space that excludes useful solutions.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>Evaluating a configuration is often more expensive than proposing it. For <Math>{'n'}</Math> candidates and <Math>{'K'}</Math> folds, ordinary CV requires <Math>{'nK'}</Math> estimator fits, before final refits. Fits can have very different costs. A count of candidates is not a wall-clock budget; concurrent workers also change wall time without eliminating computational work.</Prose>
    <Prose>An evaluator can use a cheaper approximation, or <strong>lower fidelity</strong>, such as fewer training epochs, fewer fitting examples, or a smaller input resolution. Its usefulness depends on whether it preserves enough information about the expensive target evaluation.</Prose>

    <H3>Successive halving: spend more on survivors</H3>
    <Prose>Here is a fully specified constructed example. Nine candidates begin with one resource unit each. Keep the best third, increase their resource to three units, keep the best third again, and increase the survivor to nine units. Lower loss is better.</Prose>
    <LessonTable caption="Nine constructed candidates on a three-rung resource ladder. Losses are constructed; the units are constructed resource units, not seconds." headers={['Candidate', ...declaredResource.map(unit => `Loss at ${unit} unit${unit === 1 ? '' : 's'}`)]} rows={declaredCurves.map(curve => [
      curve.id, ...curve.losses.map(loss => num(loss)),
    ])} />
    <Prose>At the first rung, {halving.rungs[0].survivors.join(', ')} survive. At the second, {halving.rungs[1].survivors.join(', ')} survives. The selected candidate finishes at {num(halving.selectedFinalLoss)}. {halving.counterfactualId} would have reached {num(halving.counterfactualFinalLoss)}, but its slow start eliminated it before that evidence was purchased.</Prose>
    <Prose>If each rung restarts training, work is <Math>{'9(1)+3(3)+1(9)=27'}</Math> units. If training can genuinely resume from the retained state, incremental work is <Math>{'9(1)+3(3-1)+1(9-3)=21'}</Math>. Fitting all nine at full resource would cost {halving.work.allFull} units under this equal-unit-cost construction. Resume is a property of the actual training procedure and state, not something any estimator&rsquo;s similarly named option guarantees.</Prose>
    <HalvingLab />
    <Prose>Hyperband repeats this resource-allocation idea across several <em>brackets</em>. Some brackets start many candidates cheaply; others start fewer candidates with more evidence before elimination. This hedges the breadth-versus-depth choice. It does not make every early ranking reliable. The <a href="https://www.jmlr.org/papers/volume18/16-558/16-558.pdf">original algorithm and analysis</a> and <a href="https://homes.cs.washington.edu/~jamieson/hyperband.html">Kevin Jamieson&rsquo;s worked bracket table</a> make those two loops explicit; the investigation above reproduces all {brackets.brackets.length} brackets for <Math>{'R=9,\\eta=3'}</Math> in a panel of its own.</Prose>

    <H3>A cheaper measurement can change the question</H3>
    <Prose>A model that is best after one epoch need not be best after fifty. A small-data winner may not be best with all fitting data. A reduced image resolution can remove the very feature that distinguishes two architectures. Weight sharing, introduced in section 7, changes how candidate weights are obtained rather than merely shortening an otherwise identical fit.</Prose>
    <Prose>Before adopting a proxy, compare it with the intended evaluation on a declared set of candidates. Inspect ranking changes and the actual promising region, not just one overall correlation. Save sufficient final evaluations to detect slow starters or proxy-specific advantages. There is no universal correlation threshold or fixed fraction of budget that makes every proxy safe.</Prose>
    <Callout title="Budget is an evidence policy">
      It should state candidate counts or stopping rules, resource per evaluation, repeated seeds where justified,
      concurrency, and what work is reserved for evaluating selected candidates. Record failed and interrupted trials as
      well as successful ones. A failed fit is useful diagnostic evidence; it is not a secretly excellent score or a
      reason to hide that part of the search space.
    </Callout>

    <H2>{headings[3]}</H2>
    <Prose>A neural network composes parameterized transformations. Start with a single unit. It multiplies input values by learned weights, adds a learned bias, and applies an activation function:</Prose>
    <MathBlock>{'h=\\tanh(w_1x_1+\\cdots+w_dx_d+b).'}</MathBlock>
    <Prose>The function <Math>{'\\tanh'}</Math> bends and bounds the weighted sum. A layer contains several such units. The next layer receives their outputs. In binary classification, a final sigmoid transforms a final weighted sum into a number between zero and one; a classification threshold converts that number into a class decision. The trained number is not automatically calibrated just because it lies in that interval.</Prose>
    <Prose>Without nonlinear activations, composing affine layers would still give an affine transformation. A nonlinear hidden layer allows the model to represent interactions and curved boundaries that a single linear score cannot. The later deep-learning module develops training and representation in depth; this is enough machinery to understand what our small search changes.</Prose>

    <H3>Weights and architecture are different decisions</H3>
    <Prose>An architecture says which layers and connections exist and how large their intermediate representations are. Training estimates the weights within that structure. Choosing eight hidden units instead of sixteen changes the structure; changing one fitted connection weight does not.</Prose>
    <Prose>For four input features, a hidden layer of width <Math>{'h'}</Math>, and one output:</Prose>
    <MathBlock>{'\\begin{gathered}\\text{parameter count}\\\\[4pt]=(4+1)h+(h+1)=6h+1.\\end{gathered}'}</MathBlock>
    <Prose>The added ones account for bias parameters. Width eight gives {networkBlocks(4, [8]).total} parameters; width sixteen gives {networkBlocks(4, [16]).total}. Two hidden layers of widths eight and eight give</Prose>
    <MathBlock>{'\\begin{gathered}(4+1)8+(8+1)8\\\\[4pt]+(8+1)1=121.\\end{gathered}'}</MathBlock>
    <ArchitectureFigure />
    <Prose>Searching these three structures is a small, legitimate NAS experiment. It is deliberately restricted: the activation, fitting algorithm, and regularization are fixed. A larger NAS space might include convolutional operations, skip connections, or repeated cells, but each additional choice requires a valid shape rule and an evaluation budget.</Prose>

    <H3>A graph must also be executable</H3>
    <Prose>Think of a more general architecture as a directed acyclic computation graph. A node stores an intermediate tensor; an edge applies an operation. Two paths can be added only if their output shapes agree, or if an explicit projection makes them agree. Concatenation joins selected dimensions and changes the downstream shape. An identity edge preserves its input; a zero edge contributes a zero tensor of the required shape.</Prose>
    <Prose>NAS therefore has three separable components:</Prose>
    <LessonTable caption="The three separable components of any NAS method" headers={['Component', 'The question it answers', 'Our small study']} rows={[
      ['Search space', 'Which executable architectures are allowed?', 'Three fixed hidden-width patterns.'],
      ['Search strategy', 'Which candidate is evaluated next?', `All three are included in a declared ${space.total}-candidate study.`],
      ['Performance estimation', 'How is a candidate judged?', `${foldRows.length} group-respecting folds, independently fitted weights.`],
    ]} />
    <Prose>Swapping an evolutionary proposer for a Bayesian proposer changes the second component. Using a shared-weight supernetwork changes the third. Claims about &ldquo;a better NAS method&rdquo; need to say which components changed.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>The <a href={provenance.page}>UCI Banknote Authentication dataset</a>, attributed to {provenance.author}, provides four numerical image descriptors and a binary class for {provenance.rows.toLocaleString('en-US')} records. The descriptors are variance, skewness, kurtosis (spelled &ldquo;curtosis&rdquo; in the source), and entropy of the supplied image-derived measurements. We retain the original class labels 0 and 1; the inspected documentation does not establish which numeric label means genuine or forged. This page serves its own <a href={provenance.file} download>unchanged copy</a>, {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a>, beside its <a href={provenance.attribution}>attribution</a>.</Prose>
    <Prose>The source contains <strong>{provenance.uniqueGroups.toLocaleString('en-US')} unique feature vectors</strong>. {provenance.repeatedGroups} vectors occur repeatedly, accounting for {provenance.repeatedExtraRows} additional rows. All identical vectors have matching class labels. We retain every row and keep identical vectors together during splitting. That prevents exact-feature copies from appearing on opposite sides of a boundary. It does not establish physical banknote identity: the source provides no specimen or capture-session identifiers with which to evaluate independence at that level.</Prose>
    <Prose>The fixed group splits produce:</Prose>
    <LessonTable caption="The three fixed data roles. Groups, not rows, are partitioned, so row counts are not round numbers." headers={['Role', 'Unique feature groups', 'Rows', 'Class-1 rows', 'Use']} rows={roles.map(role => [
      role.role.charAt(0).toUpperCase() + role.role.slice(1),
      role.groups.toLocaleString('en-US'), String(role.rows), String(role.positives), role.use,
    ])} />
    <Prose>These are group-stratified partitions: stratification uses one class label per unique feature group. Because group sizes vary, row-level class proportions and fold sizes need not be identical. Keeping duplicate groups intact takes priority over forcing exact row counts.</Prose>

    <H3>Run the complete bounded study</H3>
    <Prose>Save the openly licensed source file as <Code>banknote-data.csv</Code> beside the program below. It contains five comma-separated values per row and no header. The <a href={provenance.download}>dataset download</a> contains <Code>{provenance.member}</Code>; renaming that file changes no data.</Prose>
    <Prose>The author calculation used Python {versions.python}, NumPy 2.3.5, SciPy 1.18.1, and scikit-learn 1.9.1. Version changes can alter optimizer stopping behavior or floating-point details.</Prose>
    <Program example={automlExamples.banknoteSearch}>
      <Prose>Every candidate sees the same {foldRows.length} validation groups, of {foldRows.join(', ')} rows. <Code>clone</Code> creates a fresh estimator for each fit. A pipeline refits its scaler inside the fold. No scaling operation runs on all development rows before CV. The selected model is then refitted on all development rows, which is appropriate after the configuration is fixed.</Prose>
    </Program>

    <H3>What actually happened</H3>
    <Prose>There were {candidates.length * foldRows.length} fold fits and two final refits: <strong>{estimatorFits} estimator fits</strong>. The retained run produced no fitting warnings.</Prose>
    <ObservedResultsFigure />
    <Prose>The width-16 network wins the declared criterion. On the separate {inspectionRole.rows}-row inspection partition, it classifies {selected.correct} rows correctly. The predeclared standardized logistic baseline with <Math>{'C=1'}</Math> classifies {baseline.correct} correctly. Its confusion matrix, with true classes as rows and predicted classes as columns, is <Math>{`\\begin{bmatrix}${baseline.confusion[0].join('&')}\\\\${baseline.confusion[1].join('&')}\\end{bmatrix}`}</Math>; the selected network&rsquo;s is <Math>{`\\begin{bmatrix}${selected.confusion[0].join('&')}\\\\${selected.confusion[1].join('&')}\\end{bmatrix}`}</Math>. Always predicting the development majority class {majorityBaseline.predictedClass} would classify {majorityBaseline.correct} of {majorityBaseline.rows} inspection rows correctly.</Prose>
    <Prose>Three observations deserve explanation:</Prose>
    <ul>
      <li><strong>More layers did not win.</strong> The two-hidden-layer model has more parameters than the width-16 model but makes one out-of-fold error. This small result does not establish a universal advantage for shallow networks; it shows why architecture size is a choice to evaluate.</li>
      <li><strong>Scaling did not improve every fixed logistic configuration.</strong> At a fixed <Math>{'C'}</Math>, changing feature scales changes the relationship between a coefficient penalty and effects in original feature units. It also affects numerical conditioning. The search is comparing complete procedures, not testing a theorem that preprocessing always helps. The <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">regularization lesson</a> explains this geometry.</li>
      <li><strong>A simple competing family was already strong.</strong> Three-neighbor classification and the smaller networks each make one out-of-fold error. Searching only neural networks would conceal that context.</li>
    </ul>
    <Callout title="What a perfect finite score does and does not establish">
      The {inspectionRole.rows} records do not cover new currencies, capture devices, adversarial counterfeits, or future
      distribution shifts. Exact-feature grouping removes one identifiable leakage route; it does not prove independence
      of every physical specimen. No confidence interval calculated under independent Bernoulli trials can repair missing
      specimen identities. The experiment establishes a reproducible comparison on this declared source and partition,
      with a further {reservedRole.rows} reserved rows still unscored.
    </Callout>

    <H3>Replay a budget without pretending to run a new optimizer</H3>
    <Prose>The recorded candidate table can be revealed in a fixed random order, generated with seed 75. Advance the budget and inspect whether the newly observed candidate changes the recommended configuration or just adds evidence. With two candidates revealed, {replayTwo.recommendedId} is recommended. With three, {replayThree.recommendedId} ties its score and wins the predeclared registry-order tie rule: the best score stays at {num(replayThree.best)} even though the recommended model changes. With four, {replayFour.recommendedId} becomes the winner at {num(replayFour.best)}.</Prose>
    <SearchReplayLab />
    <Prose>This is a replay of a fully evaluated finite table. It is not a measured comparison between random search and Bayesian optimization, and it does not erase the {estimatorFits} fits used to construct the evidence. The inspection outcomes remain separate from the replay: you cannot select a different candidate by browsing its inspection accuracy because those additional predictions were never made.</Prose>
    <Checkpoint prompt={`At budget 3 the replay recommends ${replayThree.recommendedId}. Can you attach the width-16 network's ${selected.correct}-of-${selected.rows} inspection result to that recommendation?`}>
      <Prose>No. That result belongs to the procedure the <em>complete</em> search selected and then refitted on all {development.rows} development rows. At budget 3 the width-16 network has not been revealed at all, and the recommended configuration is a different one, whose inspection predictions were never computed. Borrowing the number would report a development selection score and an independent assessment as if they came from the same experiment — which is the specific confusion this whole section exists to prevent.</Prose>
    </Checkpoint>

    <H2>{headings[5]}</H2>
    <H3>Search, combine, and transfer are different operations</H3>
    <Prose>Selecting the best observed single candidate is only one use of a search history. An ensemble can combine predictions from several fitted models. Diversity matters: averaging two models with identical errors adds little; complementary errors can help. The combination itself must be learned and assessed with appropriate data separation. Our own study supplies a caution here: three of the strongest candidates make <em>the same</em> single out-of-fold mistake, at file line {sourceRows['349'].line}.</Prose>
    <Prose>In a weighted average of class-1 probabilities,</Prose>
    <MathBlock>{'\\begin{gathered}\\hat p(x)=\\sum_m a_m\\hat p_m(x),\\\\[4pt]a_m\\ge0,\\quad\\sum_m a_m=1,\\end{gathered}'}</MathBlock>
    <Prose>the weights are another learned choice. Greedy ensemble selection can repeatedly add the candidate that improves the current combination; selection with replacement gives a model extra weight through repeated inclusion. Stacking instead trains a second-level predictor on candidate outputs, normally using out-of-fold predictions for its training inputs. These are distinct procedures. The auto-sklearn chapter of the <a href="https://automl.org/book/">AutoML book</a> describes greedy ensemble selection, so &ldquo;take the top few and stack them&rdquo; is not an accurate description of that original method.</Prose>
    <Prose>Out-of-fold prediction prevents each row&rsquo;s base prediction from coming from a base model fitted on that same row. It does not magically protect every subsequent search decision. If you repeatedly choose ensembles using the same out-of-fold record, that record has become selection evidence. Preserve an assessment boundary for the complete system.</Prose>
    <Prose><strong>Meta-learning</strong> uses experience from previous tasks to guide a new task. It may suggest configurations, learn which task characteristics predict useful choices, or transfer fitted representations. A <em>portfolio</em> is a small collection of configurations selected to cover different tasks well.</Prose>
    <Prose>Consider constructed losses for three configurations on two old tasks:</Prose>
    <LessonTable caption="A constructed portfolio matrix. These losses are invented to expose complementarity; they are not measurements." headers={['Configuration', 'Old task 1', 'Old task 2', 'Mean']} rows={portfolio.ids.map((id, index) => [
      id, num(portfolio.oldTaskLosses[index][0]), num(portfolio.oldTaskLosses[index][1]), num(portfolio.means[index]),
    ])} />
    <Prose>{portfolio.singleId} is the best single default at {num(portfolio.singleMean)}. But trying {portfolio.bestPair.members.join(' and ')} and selecting between them on each task gives a mean best loss of {num(portfolio.bestPair.mean)}. A useful portfolio covers complementary strengths, rather than simply containing the best average performer twice. On a new task with losses A = {num(portfolio.newTaskLosses[0])}, B = {num(portfolio.newTaskLosses[1])}, C = {num(portfolio.newTaskLosses[2])}, that old portfolio reaches only {num(portfolio.newTaskPortfolioBest)} while the excluded {portfolio.newTaskOverallBestId} would have reached {num(portfolio.newTaskOverallBest)}. Transfer requires validation on new tasks; similar dataset summaries are evidence to investigate, not a guarantee of similar model rankings.</Prose>
    <Prose>Current AutoML systems can also use pretrained tabular models or learned portfolios. Their pretraining data, licensing, resource needs, and task overlap become part of the evaluation. The dedicated <a href="/learn/topic/automl-as-meta-learning">AutoML as meta-learning lesson</a> develops those cross-task decisions.</Prose>

    <H3>Sometimes the search language is an explanation</H3>
    <Prose>An interesting application is the Automatic Statistician: its modeling grammar can combine components representing smooth change, periodicity, or noise in a time series. A search can then return a structured model that supports a verbal explanation, such as a seasonal pattern whose amplitude changes over time. This illustrates why the search space determines what kinds of explanations can be produced.</Prose>
    <Prose>The interpretation is conditional on the grammar, data, and fitted model. A periodic component is not proof of a physical cause. The <a href="https://automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf">book&rsquo;s Automatic Statistician chapter</a> describes kernel composition, structure search, and generated descriptions. The later Gaussian-process lesson supplies the probability model underlying those kernels; the model-selection criteria introduced in <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">regularization</a> explain why raw fitting likelihood alone favors unnecessary complexity.</Prose>

    <H3>Optimize for the device that will run the result</H3>
    <Prose>A model with fewer parameters need not be faster on a specific device. Memory access, tensor shapes, operator implementations, parallelism, and data movement all matter. Multiplication counts and parameter counts are useful descriptors; deployment latency is a measurement with conditions: hardware, software, input shape, batch size, precision, warm-up, and the timing boundary.</Prose>
    <DeploymentLab />
    <Prose>A point is <em>Pareto dominated</em> if another point is no worse in every objective and strictly better in at least one. The frontier contains choices that require a trade-off. It does not choose the trade-off for you. A soft score that penalizes latency can still prefer an over-budget model. If {pareto.cap} ms is a hard requirement, filter out infeasible candidates using a defined measurement protocol before choosing among the remainder. The <a href="https://arxiv.org/pdf/1807.11626">MnasNet paper</a> is a primary example of architecture search that includes target-device latency; its results should not be transplanted as timing predictions for another device.</Prose>

    <H3>A practical decision sequence</H3>
    <ol>
      <li>Establish a useful baseline and a valid deployment-related split. Include error costs or group-specific requirements in the objective when the task demands them.</li>
      <li>Define a compact space with justified ranges and valid conditional settings. Include a strong simple competitor, not only an expensive family you hope will win.</li>
      <li>Measure a few representative fits to understand cost and failures. Decide whether enumeration, random proposals, a surrogate, or a fidelity scheduler addresses the actual bottleneck.</li>
      <li>Record preprocessing, folds, seeds, resources, scores, warnings, and active settings for each trial. Keep the selection rule fixed, including ties.</li>
      <li>Evaluate the selected procedure with evidence outside its selection loop. Assess deployment latency, memory, and relevant error patterns under their own declared protocols.</li>
      <li>Preserve the fitted preprocessing and model together, along with input schema and environment information. In production, monitor task-relevant changes and outcomes; a statistical change in input distribution alone does not quantify prediction harm.</li>
    </ol>
    <Prose>No permanent league table can rank AutoML libraries for every dataset and resource limit. As inspected in September 2026, FLAML exposes task, metric, budget, estimator, and resampling controls; AutoGluon&rsquo;s current tabular presets include different ensembles and learned model portfolios; KerasTuner exposes neural hyperparameter spaces and tuners. Microsoft NNI is archived and read-only, so treat it as a historical resource rather than a default maintained choice. The linked documentation in section 10 records the current interfaces instead of promising fixed installation times or universal winners.</Prose>

    <H2>{headings[6]}</H2>
    <Prose><strong>This section is the deeper branch.</strong> It connects the graph view in section 4 to gradients, and it is useful when you want to understand what a differentiable NAS method optimizes, and what changes when a searched network becomes a deployed network. The core route is complete without it.</Prose>

    <H3>Replace a discrete choice with a mixture</H3>
    <Prose>Suppose an edge could apply one of several shape-compatible operations <Math>{'o_1,\\ldots,o_m'}</Math>. Introduce architecture logits <Math>{'\\alpha_1,\\ldots,\\alpha_m'}</Math>, turn them into softmax weights,</Prose>
    <MathBlock>{'\\begin{gathered}p_i=\\frac{e^{\\alpha_i}}{\\sum_j e^{\\alpha_j}},\\\\[4pt]\\overline o(x)=\\sum_i p_i\\,o_i(x),\\end{gathered}'}</MathBlock>
    <Prose>and evaluate the mixture while searching. The logits are not class probabilities. They control how candidate operations contribute to an intermediate computation.</Prose>
    <Prose>For a constructed scalar edge at input <Math>{'x=2'}</Math>, let the operations be zero, identity, and negation. Their outputs are <Math>{'[0,2,-2]'}</Math>. With logits <Math>{'[\\log2,0,0]'}</Math>, the probabilities are <Math>{'[0.5,0.25,0.25]'}</Math>, so the mixed output is {num(mixture.mixed)}. If the target is 1 and loss is <Math>{'\\tfrac12(\\overline o-1)^2'}</Math>, the loss is {num(mixture.loss)}.</Prose>
    <Prose>The softmax derivative gives</Prose>
    <MathBlock>{'\\begin{gathered}\\frac{\\partial\\overline o}{\\partial\\alpha_i}=p_i\\big(o_i-\\overline o\\big),\\\\[6pt]\\frac{\\partial L}{\\partial\\alpha_i}=(\\overline o-1)\\,p_i\\big(o_i-\\overline o\\big).\\end{gathered}'}</MathBlock>
    <Prose>The gradient is <Math>{'[0,-0.5,0.5]'}</Math>. One gradient step of size {mixture.step} changes the logits to <Math>{'[\\log2,0.2,-0.2]'}</Math>, yielding output about {num(mixture.updatedOutput)}. The identity operation&rsquo;s contribution increases, which moves the mixture toward the target.</Prose>
    <MixtureLab />
    <Prose>The <a href="https://arxiv.org/pdf/1806.09055">DARTS paper</a> uses continuous mixtures to search cell structures. Its convolutional-cell discretization retains two strong nonzero operations from distinct incoming nodes for each intermediate node; its recurrent-cell construction uses one. This is more specific than independently keeping the largest logit on every possible edge. Shape-compatible search and the final graph-construction rule both belong in a reproducible method.</Prose>

    <H3>The architecture should anticipate trained weights</H3>
    <Prose>For a network, operations can have trainable weights <Math>{'w'}</Math> in addition to architecture variables <Math>{'\\alpha'}</Math>. The intended nested problem is</Prose>
    <MathBlock>{'\\begin{gathered}\\min_\\alpha L_{\\mathrm{val}}(w^*(\\alpha),\\alpha),\\\\[4pt]w^*(\\alpha)\\in\\arg\\min_w L_{\\mathrm{train}}(w,\\alpha).\\end{gathered}'}</MathBlock>
    <Prose>The inner problem asks which weights fit the training data for an architecture. The outer problem asks how that fitted architecture performs on validation data. This is <strong>bilevel optimization</strong>. Validation observations used to optimize architecture are selection data; they cannot simultaneously serve as an untouched final test.</Prose>
    <Prose>Solving the inner training problem from scratch after every architecture change is expensive. A one-step approximation uses</Prose>
    <MathBlock>{'w\'=w-\\xi\\nabla_wL_{\\mathrm{train}}(w,\\alpha)'}</MathBlock>
    <Prose>and differentiates <Math>{'L_{\\mathrm{val}}(w\',\\alpha)'}</Math>. Treat the current <Math>{'w'}</Math> as fixed for this approximation. Writing <Math>{'v=\\nabla_{w\'}L_{\\mathrm{val}}(w\',\\alpha)'}</Math>, the chain rule gives</Prose>
    <MathBlock>{'\\begin{gathered}\\nabla_\\alpha L_{\\mathrm{val}}(w\',\\alpha)\\\\[4pt]-\\;\\xi\\,\\nabla^2_{\\alpha,w}L_{\\mathrm{train}}(w,\\alpha)\\,v.\\end{gathered}'}</MathBlock>
    <Prose>The first term is the direct effect of architecture on validation loss at the updated weights. The second is the effect of architecture on the training step, which changes those weights. The mixed Hessian multiplies a vector; an implementation need not store a full matrix. A central finite difference of training architecture-gradients at <Math>{'w\\pm\\epsilon v'}</Math>, divided by <Math>{'2\\epsilon'}</Math>, approximates this product, subject to the usual step-size and numerical-error trade-offs.</Prose>
    <Prose>In the DARTS terminology, setting <Math>{'\\xi=0'}</Math> gives the <strong>first-order approximation</strong>. Keeping the nonzero one-step dependency includes the mixed derivative and is called the second-order approximation. A one-step unroll is therefore not automatically &ldquo;first order&rdquo; merely because it contains one training step.</Prose>
    <Prose>Make that concrete with a scalar problem you can differentiate by hand. Use <Math>{'L_{\\mathrm{train}}=\\tfrac12(w-\\alpha)^2'}</Math> and <Math>{'L_{\\mathrm{val}}=\\tfrac12(w-1)^2'}</Math>, at <Math>{'w=0'}</Math>, <Math>{'\\alpha=0.2'}</Math> and <Math>{'\\xi=0.1'}</Math>. The one-step weight is <Math>{'w\'=0.02'}</Math>. The direct architecture derivative at fixed <Math>{'w'}</Math> is {num(scalar.lanes[0].outer)}: <Math>{'\\alpha'}</Math> does not appear explicitly in the validation formula. The one-step derivative is <Math>{'(0.02-1)(0.1)='}</Math>{num(scalar.lanes[1].outer)}. Solving the inner problem exactly gives <Math>{'w^*=\\alpha'}</Math>, so the true outer derivative is <Math>{'\\alpha-1='}</Math>{num(scalar.lanes[2].outer)}.</Prose>
    <BilevelFigure />
    <Prose>These three numbers answer different questions. Even if the current training gradient is zero at a particular point, its derivative with respect to architecture need not be zero — the figure&rsquo;s second setting shows exactly that, with an outer derivative of {num(stationaryScalar.lanes[1].outer)} where the current and one-step weights are identical. Equality of two values at a point does not justify dropping the chain-rule term. This distinction prevents a common confusion between evaluating an expression and differentiating the function that produced it.</Prose>
    <Prose>Nonconvex training can have multiple local solutions, and an approximation may poorly track their response to architecture changes. A large identity-operation weight can also reflect search dynamics, optimization ease, or the relaxation, rather than a universally best final graph. Adding an architecture penalty changes the objective; it does not guarantee that every such failure disappears.</Prose>

    <H3>Independent fits, shared weights, and proxies</H3>
    <Prose>In our banknote study, each architecture receives fresh fitted weights in each fold. A <em>supernetwork</em> can instead contain many candidate subgraphs and train shared weights. Evaluating a subgraph then reuses part of that state. This saves work but couples candidates: an operation may benefit from how frequently it was sampled, which other paths trained it, and which weights it shares. The shared-weight ranking can differ from the ranking after independent full training.</Prose>
    <WeightProvenanceFigure />
    <Prose>Reinforcement-learning search can view architecture choices as a sequence of actions with a validation-based reward. Evolutionary search can mutate architectures and select promising descendants. Bayesian optimization can model the architecture-to-score relation. Network morphisms can expand a network while preserving its current function under specific constructions. These are different ways to propose candidates or reuse training; none removes the need to define the final evaluator and compare against a competent simple search under matching conditions.</Prose>
    <Prose>One unusual proxy, <strong>NASWOT</strong>, examines activation patterns at random initialization without performing ordinary network training. A ReLU activation outputs <Math>{'\\max(0,z)'}</Math> for its incoming weighted sum <Math>{'z'}</Math>; call the unit active when that sum is positive. For a small batch passing through these units, record a binary code indicating which units are active. If there are <Math>{'N_A'}</Math> recorded units, one kernel counts shared activation decisions:</Prose>
    <MathBlock>{'K_{ij}=N_A-d_H(c_i,c_j),'}</MathBlock>
    <Prose>where <Math>{'d_H'}</Math> is Hamming distance: the number of positions at which two codes differ. The proposed score uses <Math>{'\\log\\det K'}</Math>. With codes 110 and 101, <Math>{`K=\\begin{bmatrix}${kernel.matrix[0].join('&')}\\\\${kernel.matrix[1].join('&')}\\end{bmatrix}`}</Math>, so the determinant is {kernel.determinant}. Identical codes give a singular matrix and determinant {singularKernel.determinant}. This helps visualize the score&rsquo;s preference for differentiated activation patterns; it does not prove that such differentiation will generalize after training.</Prose>
    <Prose>The <a href="https://proceedings.mlr.press/v139/mellor21a/mellor21a.pdf">NASWOT paper</a> evaluates this signal on architecture benchmarks and studies sensitivity to initialization and batches. Its main construction uses actual input mini-batches; Gaussian random inputs are an ablation, not the definition of the method. &ldquo;Without training&rdquo; still involves computation, including a forward pass and a matrix calculation. A practical implementation must specify handling of singular kernels and distinguish any numerical regularization from the original exact formula.</Prose>

    <H3>Keep the final comparison honest</H3>
    <Prose>Use the same task data, candidate space, fitting budget, and deployment protocol when comparing search methods, unless a changed component is the explicit subject of the experiment. Count search work, proxy evaluation, architecture selection, and final retraining. Separate variability from architecture-training seeds and variability from the search process itself. Repeatedly consulting a public benchmark&rsquo;s test outcomes can turn that benchmark into selection evidence, even when no single training script reads those labels.</Prose>
    <Prose>The later <a href="/learn/topic/neural-architecture-search-nas">dedicated NAS lesson</a> develops convolutional cells, search benchmarks, shared-weight implementation, and hardware evaluation. The local mechanism here is complete enough to explain what those methods optimize without pretending that a tiny multilayer perceptron validates a full image-model search system.</Prose>

    <H2>{headings[7]}</H2>
    <Prose>These examples show how to preserve the contract when adopting a library. They are optional runnable extensions, not sources of the observed scores in section 5. Both use only the first development fold from <Code>banknote_search.py</Code>; neither accesses inspection or reserved rows, and that boundary is checked by this lesson&rsquo;s own verifier rather than asserted. Neither library is installed in this site&rsquo;s lesson runtime, so <strong>no output is recorded for them here</strong>: run them yourself and record the versions you used.</Prose>

    <H3>FLAML: make the validation method explicit</H3>
    <OptionalProgram example={automlExamples.banknoteFlaml}>
      <Prose>The 30-second setting is a requested search budget, and eight iterations is an additional bound; actual duration includes library and estimator behavior. <Code>best_loss</Code> is a minimized objective, so for the requested accuracy metric its ideal interpretation is one minus accuracy. The printed accuracy reuses selection data and is labeled accordingly. It is not the {foldRows.length}-fold score from section 5 or an independent final assessment. FLAML&rsquo;s <Code>auto</Code> evaluation mode can choose holdout or CV, which is why the example specifies its intended method. See the <a href="https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/">official task-oriented AutoML documentation</a> for resampling and estimator controls.</Prose>
    </OptionalProgram>

    <H3>KerasTuner: a conditional neural search space</H3>
    <Prose>This extension searches one or two tanh hidden layers. The second width exists only when depth is two. It uses a separate fitting procedure from section 5: minibatch Adam rather than L-BFGS. An epoch is one pass through the fitting data; Adam updates weights using batches. The deep-learning module explains that optimizer. Here its settings are fixed except for a declared learning-rate choice. The tuner stores its trial state in <Code>banknote-nas-study/conditional-mlp</Code>; keeping the same directory resumes compatible state. Use a new descriptive project name for a different declared experiment.</Prose>
    <OptionalProgram example={automlExamples.banknoteKerasSearch}>
      <Prose>The <Code>conditional_scope</Code> registers when a setting is active; it does not skip execution of its Python body. An inactive second width can be <Code>None</Code>, which is why the model adds that layer only inside <Code>if depth == 2</Code>. Four trials explore part of a twelve-configuration space: four one-layer combinations and eight two-layer combinations. They do not exhaust it.</Prose>
      <Prose>The tuner selects using validation performance, including its checkpoint behavior across epochs. That checkpoint choice is part of the selection procedure. A high printed score still needs assessment outside this holdout before supporting a generalization claim. The <a href="https://keras.io/keras_tuner/api/hyperparameters/">conditional-hyperparameter reference</a> and <a href="https://keras.io/keras_tuner/getting_started/">complete getting-started guide</a> explain the interface and retrieval of selected settings.</Prose>
    </OptionalProgram>

    <H2>{headings[8]}</H2>
    <Prose>Try the core questions 1&ndash;6 before the deeper questions. Open a hint or solution when you need it. Changing an answer after a reveal is useful reflection, but it is different from an unaided first attempt.</Prose>

    <Practice title="1. Count the experiments"
      question={<>A new space contains logistic regression with three <Math>{'C'}</Math> values and two preprocessing options; trees with four depths; and neural networks with either one hidden layer of width 6 or 12, or two hidden layers with each width independently 6 or 12. How many valid configurations exist? How many fits does four-fold CV require before refits?</>}
      hint="Add independent family branches. Multiply settings that are simultaneously active within a branch.">
      <Prose>Logistic contributes 6, trees 4, one-layer networks 2, and two-layer networks 4. Total 16 configurations; four-fold CV requires 64 fits. Counting an inactive second width for a one-layer network would double-count its functionally identical configurations. This exercise uses a different registry from the first investigation, which also includes nearest neighbors; apply the same branch-by-branch counting rule to the families declared here.</Prose>
    </Practice>

    <Practice title="2. Read the evidence boundary"
      question="A colleague standardizes the full development matrix, runs fold-based AutoML on it, then tries five cost thresholds on the final inspection set and reports the best inspection cost. Identify two distinct boundary violations and repair them."
      hint="Ask which rows fitted the preprocessing and which rows selected the threshold.">
      <Prose>The scaler learned validation-fold information before CV; place it inside each candidate pipeline and fit it on each fold&rsquo;s fitting rows. The inspection set selected the threshold; choose thresholds using selection data under the task&rsquo;s cost policy, then assess the fixed model-plus-threshold procedure with separate evidence. Calling the second step &ldquo;just postprocessing&rdquo; does not restore independence.</Prose>
    </Practice>

    <Practice title="3. Spend a small fidelity budget"
      question={<>Four candidates have losses at resources 1, 2, and 4: A = (0.10, 0.09, 0.08), B = (0.11, 0.08, 0.07), C = (0.12, 0.07, 0.02), D = (0.20, 0.18, 0.15). Keep half at each cut. Which candidate wins? What is the work with restart and with genuine continuation? Which full-resource winner is missed?</>}>
      <Prose>A and B survive resource 1; B survives resource 2 and finishes at 0.07. Restart work is <Math>{'4+2(2)+1(4)=12'}</Math>; continuation work is <Math>{'4+2(2-1)+1(4-2)=8'}</Math>. C would reach 0.02 but is removed at the first cut. Its good later value cannot inform a real early decision unless that evidence is actually purchased.</Prose>
    </Practice>

    <Practice title="4. Count network parameters, including biases"
      question="For five inputs, hidden widths 6 and 3, and one binary output, compute every parameter block. Would replacing the two layers with one width-12 layer necessarily improve accuracy?">
      <Prose>The blocks contain {networkBlocks(5, [6, 3]).blocks.map(block => block.equation.split(' = ')[0]).join(', ')}, that is {networkBlocks(5, [6, 3]).blocks.map(block => block.total).join(', ')} parameters, totaling {networkBlocks(5, [6, 3]).total}. A one-layer width-12 model has <Math>{'(5+1)12+(12+1)='}</Math>{networkBlocks(5, [12]).total}. Neither parameter count determines accuracy: representation, optimization, regularization, and data all matter. Compare them under a declared evaluator.</Prose>
    </Practice>

    <Practice title="5. Interpret the real search replay"
      question="Reveal only the first three candidates in the replay investigation. Explain why the recommended candidate can change while the best-so-far score stays flat. Can the selected width-16 model's eventual inspection result be attached to the budget-three recommendation?">
      <Prose>{replayTwo.recommendedId} and {replayThree.recommendedId} tie at {num(replayThree.best)} mean fold accuracy. The registry-order tie rule favors {replayThree.recommendedId} when it becomes available. A flat maximum does not imply the selected model is unchanged. The width-16 model has not been revealed at budget three and its inspection result belongs to a different selected procedure. The replay cannot borrow that outcome.</Prose>
    </Practice>

    <Practice title="6. Apply a hard deployment constraint"
      question="Three hypothetical models have (latency, accuracy) pairs P = (3 ms, 0.92), Q = (6 ms, 0.96), and R = (5 ms, 0.91). Identify the frontier and choose under a 5 ms cap. Explain why adding a finite latency penalty to accuracy need not enforce the cap.">
      <Prose>P dominates R, while P and Q trade speed for accuracy. The frontier is P and Q; the cap leaves P as the best feasible candidate. A finite penalty allows accuracy gains to compensate for exceeding the cap. Feasibility filtering encodes a hard bound directly, provided the latency measurement itself matches the requirement. You can enter these three pairs directly in the deployment investigation.</Prose>
    </Practice>

    <Practice title="7. Calculate expected improvement — deeper"
      question="The incumbent loss is 0.3. Candidate U has a deterministic predicted loss of 0.25. Candidate V has Gaussian predicted mean 0.3 and standard deviation 0.1. Which has larger EI? What additional fact would you need before claiming that it will actually improve validation performance?">
      <Prose>U has EI {num(expectedImprovement(0.3, 0.25, 0))}. V has <Math>{'0.1\\phi(0)=0.1/\\sqrt{2\\pi}\\approx'}</Math>{num(expectedImprovement(0.3, 0.3, 0.1))}, so U wins this acquisition comparison. Actual performance requires an evaluation; the surrogate&rsquo;s probability model and acquisition ranking are not observed objective values.</Prose>
    </Practice>

    <Practice title="8. Differentiate an operation mixture — deeper"
      question="At one input, two operations output 3 and −1. Their logits are equal, the target is zero, and loss is half squared error. Find the mixture and both architecture gradients. Then add 7 to both logits.">
      <Prose>Probabilities are one half, output is {num(practiceMixture.mixed)}, and loss is {num(practiceMixture.loss)}. Gradients are <Math>{'1(0.5)(3-1)=1'}</Math> and <Math>{'1(0.5)(-1-1)=-1'}</Math>. Adding a common constant leaves probabilities, output, loss, and gradients unchanged. It changes a redundant coordinate representation, not the mixture. The mixture investigation&rsquo;s common-logit setup applies exactly this null.</Prose>
    </Practice>

    <Practice title="9. Separate three architecture derivatives — deeper"
      question={<>Use <Math>{'L_{train}=\\tfrac12(w-\\alpha)^2'}</Math>, <Math>{'L_{val}=\\tfrac12(w-2)^2'}</Math>, current <Math>{'w=0'}</Math>, <Math>{'\\alpha=0.5'}</Math>, and <Math>{'\\xi=0.2'}</Math>. Compute the first-order direct derivative, the one-step derivative, and the exact-inner outer derivative.</>}>
      <Prose>The direct derivative is {num(practiceScalar.lanes[0].outer)}. The one-step weight is {num(practiceScalar.stepped)} and its derivative with respect to architecture is {num(practiceScalar.lanes[1].dependency)}, so the one-step outer derivative is <Math>{'(0.1-2)(0.2)='}</Math>{num(practiceScalar.lanes[1].outer)}. The exact inner optimum is <Math>{'w^*=\\alpha'}</Math>, yielding derivative <Math>{'\\alpha-2='}</Math>{num(practiceScalar.lanes[2].outer)}. These are three distinct functions being differentiated, not rounding differences.</Prose>
    </Practice>

    <Practice title="10. Design an experiment that could disappoint you"
      question="Choose a small classification task with a documented prediction-time feature set. Declare a simple baseline, two justified model families, a conditional space, grouping or temporal boundaries, selection metric, budget, tie rule, and final assessment. Predict one result that would make you simplify the system. Explain which observation would invalidate your original split rather than merely favor a different optimizer."
      revealLabel="Example response and assessment criteria">
      <Prose>A valid response could compare a standardized regularized linear model with bounded-depth trees for repeated measurements from devices, keeping each device in one fold when deployment concerns new devices. It would reserve devices for final assessment and declare latency conditions. A near-tie favoring the simpler baseline could justify choosing it. Discovering that device identifiers were duplicated across roles would require repairing the evaluation boundary and reassessing affected conclusions. A large search score alone is not evidence that the split is valid. Other tasks can satisfy the same criteria with different models and boundaries.</Prose>
    </Practice>

    <Prose>You are ready to continue when you can distinguish a configuration from fitted weights, count a conditional space, protect fitting and selection boundaries, explain a fidelity failure, and interpret the real result without treating the finite search winner as a universal best model. The deeper questions prepare you to inspect differentiable NAS implementations and proxy claims.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Say which decisions and evidence a search is permitted to use', 'Section 1, the evidence-loop figure, practice 2'],
      ['Count a conditional space correctly, and say what a sampling rule assumes', 'Section 2, the grammar investigation, practice 1'],
      ['Separate a surrogate mean, its uncertainty, an acquisition value and an observed loss', 'Section 2, the improvement investigation, practice 7'],
      ['Trace successive halving and account for its work two ways', 'Section 3, the halving investigation, practice 3'],
      ['Count a network\u2019s parameters, biases included, and distinguish weights from architecture', 'Section 4, the architecture figure, practice 4'],
      ['Read the real study without borrowing an inspection result the replay never bought', 'Section 5, the results figure, the replay investigation, practice 5'],
      ['Distinguish a best default from a complementary portfolio, and a frontier from a hard cap', 'Section 6, the deployment investigation, practice 6'],
      ['Compute an operation mixture\u2019s gradient and explain the discretization gap', 'Section 7, the mixture investigation, practice 8'],
      ['Tell the direct, one-step and exact-inner architecture derivatives apart', 'Section 7, the bilevel figure, practice 9'],
      ['Say what a shared-weight score or an untrained proxy actually measured', 'Section 7, the provenance figure'],
    ]} />

    <H2>{headings[9]}</H2>
    <Prose>The next topic in this module is <a href="/learn/path/full-curriculum/hidden-markov-models-hmm?module=classical-ml">Hidden Markov Models</a>. AutoML chooses among learning procedures; an HMM introduces a particular probabilistic structure for observations that arrive in sequence and depend on unobserved states. It will distinguish summing over possible hidden paths from finding one best path. This is a change in modeling assumptions, not simply another knob for the current independent-row classifier.</Prose>
    <Prose>For focused review, revisit <a href="/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml">feature scaling and encoding</a>, <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">cross-validation</a>, <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">regularization</a>, and <a href="/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml">feature selection</a>. For deeper branches, use <a href="/learn/path/full-curriculum/gaussian-processes-gp?module=classical-ml">Gaussian processes</a>, <a href="/learn/topic/neural-architecture-search-nas">dedicated NAS</a>, and <a href="/learn/topic/automl-as-meta-learning">AutoML as meta-learning</a>.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://www.automl.org/book/">Hutter, Kotthoff and Vanschoren &mdash; Automated Machine Learning: Methods, Systems, Challenges</a>, openly licensed. Chapters 1&ndash;3 separate hyperparameter optimization, meta-learning, and architecture search; the auto-sklearn and Automatic Statistician chapters show different uses of a search history and search language. Read the relevant section after its local example rather than treating the whole book as a prerequisite.</li>
      <li><a href="https://homes.cs.washington.edu/~jamieson/hyperband.html">Kevin Jamieson &mdash; Hyperband</a>. A compact scheduling explanation with the algorithm, bracket table, and experimental protocol. Its breadth/depth table is a useful second representation of the halving investigation. The accompanying <a href="https://www.jmlr.org/papers/volume18/16-558/16-558.pdf">JMLR paper</a> gives assumptions and analysis.</li>
      <li><a href="https://arxiv.org/pdf/1806.09055">Liu, Simonyan and Yang &mdash; DARTS</a>, sections 2.1&ndash;2.4. Read alongside the locally derived mixture and scalar bilevel example; distinguish its relaxation, approximation, and discretization stages.</li>
      <li><a href="https://proceedings.mlr.press/v139/mellor21a/mellor21a.pdf">Mellor and colleagues &mdash; Neural Architecture Search without Training</a>. A surprising proxy to examine critically, especially its activation-pattern construction and ablations. The two-code calculation here supplies a concrete entry point.</li>
    </ul></>}>
      <li><a href="https://arxiv.org/pdf/1807.11626">Tan and colleagues &mdash; MnasNet</a> &mdash; deployment-aware search. Focus on the distinction between measured latency and operation counts, and between a soft objective and an actual feasibility requirement.</li>
      <li><a href={provenance.page}>UCI &mdash; Banknote Authentication</a>, {provenance.author}, <a href={provenance.doi}>{provenance.doi}</a>, licensed <a href={provenance.licenseUrl}>{provenance.license}</a> &mdash; the observed input and its licensing. This page serves <a href={provenance.file} download>the unchanged file</a>, SHA-256 <Code>{provenance.sha256}</Code>, beside its <a href={provenance.attribution}>attribution</a>, with the exact grouping and split protocol for offline reproduction.</li>
      <li><a href="https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/">FLAML &mdash; task-oriented AutoML</a>, <a href="https://keras.io/keras_tuner/getting_started/">KerasTuner getting started</a> and <a href="https://keras.io/keras_tuner/api/hyperparameters/">its conditional hyperparameters</a> &mdash; complete API context for section 8. Their tutorial scores are not this lesson&rsquo;s measurements.</li>
      <li><a href="https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html">AutoGluon &mdash; tabular essentials</a> &mdash; current broader tabular workflows. Compare its presets and resource implications against your contract. For historical code, the <a href="https://github.com/microsoft/nni">NNI repository</a> explicitly records its archived status.</li>
    </Sources>
    <Prose>The conditional grammar&rsquo;s counts, the expected-improvement table, the nine fidelity curves, the network parameter blocks, the operation mixture and its gradients, the scalar bilevel derivatives, the portfolio matrix and the activation-code kernel are explicitly <strong>constructed calculations</strong>, not measurements. The five latency/accuracy pairs are explicitly <strong>hypothetical deployment numbers</strong> with no device, product or benchmark behind them. The banknote outcomes are calculations on the identified real dataset under one declared grouping and split protocol, with {reservedRole.rows} reserved rows neither predicted nor scored. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>,
};

export default automlContent;
