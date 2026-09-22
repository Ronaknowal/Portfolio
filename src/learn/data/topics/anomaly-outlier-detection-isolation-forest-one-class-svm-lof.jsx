import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { IsolationLab, LofNeighbourhoodLab, LofModeLab, KernelBoundaryLab, AlertPopulationLab } from '../../components/lesson-labs/AnomalyDetectionLabs.jsx';
import { TemperatureThresholdLab } from '../../components/lesson-labs/AnomalyTemperatureLab.jsx';
import { ProvenanceFigure, FirstCutFigure, ReachFloorFigure, ThresholdRulerFigure } from '../../components/lesson-labs/AnomalyDetectionFigures.jsx';
import { anomalyExamples } from '../anomaly-detection-examples.js';

const headings = [
  '1. Name the observation before choosing an algorithm',
  '2. Separate fitting, scoring and taking action',
  '3. Isolation Forest: an empty gap can make a point easy to separate',
  '4. LOF: compare local spacing with nearby local spacing',
  '5. The fitting mode is part of the LOF mathematics',
  '6. One-Class SVM: a compatibility boundary built from similarities',
  '7. Choose a comparison that matches the task',
  '8. A good ranking still needs a decision policy',
  '9. Deeper branch: where the One-Class SVM constraints come from',
  '10. Real monitoring: temperature, change and alert workload',
  '11. Deeper branch: local bounds, resource costs and changing references',
  '12. Practice: explain the mechanism before choosing a label',
  '13. What to remember, and another way to learn it'
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section><Prose><strong>Before running:</strong> {example.question}</Prose><RunnableExample example={example}>{children}</RunnableExample></section>;
}
function Practice({ title, question, hint, children }) {
  return <section className="ad-practice"><H3>{title}</H3><Prose>{question}</Prose>{hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}<details><summary>Show the explained solution</summary>{children}</details></section>;
}

const anomalyDetectionContent = {
  title: 'Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)',
  readTime: '~60 min first pass · ~105 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot anomaly-lesson">
    <LessonIntro prerequisites={<>Distances, averages and the idea of a training set are enough to start; the logarithm, the exponential and kernel notation are explained where they enter. The preceding <a href="/learn/path/full-curriculum/dbscan-density-based-clustering?module=classical-ml">DBSCAN</a> lesson supplies the density vocabulary reused in sections 4 and 5, and its noise label is exactly the question this lesson takes over.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      Learn to find observations worth investigating and to explain the comparison that found them: integrate the cut intervals of an isolation tree by hand, watch a reachability floor change which query looks isolated, put a kernel boundary’s midpoint outside a region whose two anchors are on it, turn two rates into a review queue, and then choose a threshold on 22,671 real machine-temperature rows and meet the workload it creates. Change the controls to see the calculation, diagram and measurements update together.
    </LessonIntro>
    <Prose className="ad-route"><strong>First pass.</strong> Read sections 1 to 8, work the temperature investigation in section 10, and attempt practices A to F. That route teaches you to explain each mechanism, to keep fitting separate from thresholding and to report a real result honestly. Sections 9 and 11 develop the optimization, the resource costs and the monitoring details, and practices G to J belong with them; return to those when their questions arise. Allow roughly 60 minutes for the core reading and another 60 to 90 minutes for the code and practice.</Prose>

    <Prose>A machine usually runs near a familiar temperature. Today it is much hotter. That reading might indicate a fault, a planned operating change, a different load or a failing sensor. A detector can tell you that the reading is unusual. Deciding what happened needs more evidence than the detector has.</Prose>
    <Prose>So the useful starting question is not “which algorithm is best?” It is <strong>unusual compared with what, and what will we do if we find it?</strong> You will learn three comparisons. Isolation Forest asks how easily random cuts separate an observation. Local Outlier Factor compares an observation’s neighbourhood with its neighbours’ neighbourhoods. One-Class SVM learns a boundary around a reference population. You will then turn their scores into decisions and compare all three with a plain baseline on a real machine-temperature series.</Prose>

    <Callout title="What these words mean here, once for the whole lesson">
      A <strong>score</strong> is a number used to order observations. It is not a probability, not a distance in the original units and not a verdict. <strong>Anomalous</strong>, <strong>outlier</strong> and <strong>alert</strong> name roles in a declared procedure: an observation the chosen comparison ranks high, a row the chosen threshold selects, an item sent to a reviewer. None of them means “faulty”, “wrong” or “bad data”. Every score below is oriented so that <strong>larger means more unusual</strong>. We use these words as procedural roles from here on and will not repeat this caution after every result.
    </Callout>

    <H2>{headings[0]}</H2>
    <Prose>Suppose our reference measurements are 0, 1, 2, 3 and 12. The gap between 3 and 12 is large compared with the others, which makes 12 a reasonable candidate for investigation. It does not tell us whether 12 is a mistake. If the values record operating modes, 12 may be entirely legitimate. If someone entered the wrong units, the same geometry exposes a data-quality problem instead. The arithmetic is identical; only the investigation separates the two.</Prose>
    <Prose>An observation can also be a transaction, a session, a patient measurement, an image or a time window. How you represent it decides which differences are visible at all. A detector given only temperature cannot recognise “ordinary under heavy load but excessive while idle” unless load, operating mode or a suitable residual enters the representation.</Prose>
    <LessonTable caption="Three kinds of unusual, and what each needs from the representation" headers={['case', 'what is unusual', 'example and needed representation']} rows={[
      ['Point', 'One observation differs from the chosen reference', 'A sensor reports 900 while comparable readings are near 90'],
      ['Contextual', 'The observation is unusual under its context', '90 is ordinary under load but unusual while idle; include or condition on load'],
      ['Collective', 'A group or sequence is unusual although each value looks ordinary', 'An unusually long flat trace; represent duration, variability or the sequence']
    ]} />
    <Prose>These cases overlap. Putting a one-hour change beside the current temperature, as section 10 does, introduces a little temporal context. It does not turn an ordinary tabular detector into a sequence model.</Prose>
    <Prose>The previous lesson supplies a useful boundary. A point DBSCAN labelled <em>noise</em> simply failed to attach to a density-connected component under one metric and one parameter pair. That is a geometric result about a chosen rule. Here we add two things it never had: a declared reference population and a decision rule. The difference matters in both directions. A dense collection of repeated bad measurements forms a perfectly good cluster, and a rare legitimate operating mode can lie outside every cluster.</Prose>
    <Checkpoint prompt="A detector flags a rare but scheduled shutdown. Has the algorithm necessarily failed?">
      <Prose>No. It may have correctly found an unusual operating state, which is exactly what it was asked to do. It fails the intended <em>alerting</em> task only if the system was supposed to suppress scheduled shutdowns and had the information needed to recognise one. The task and the available context decide, not the score.</Prose>
    </Checkpoint>

    <H2>{headings[1]}</H2>
    <Prose><strong>Outlier detection</strong> means you have one collection that may already contain unusual observations. You fit a comparison to that collection and inspect its own members. Reviewing a batch of sensor records for data-quality problems works this way.</Prose>
    <Prose><strong>Novelty detection</strong> means you fit on a reference collection intended to represent acceptable behaviour, then score later observations against it. Learning from a reviewed operating period and monitoring the next period works this way. “Reference” is a declared assumption. It is not a guarantee that the historical data were clean.</Prose>
    <Prose>In scikit-learn this distinction decides which LOF methods you may call at all, which is why section 5 treats it as mathematics rather than as configuration. Other literature sometimes uses <em>novelty</em> more broadly, for newly encountered concepts; here we follow the library’s fitting-and-query convention.</Prose>
    <Prose>Keep three objects separate:</Prose>
    <MathBlock>{'\\begin{gathered}\\text{reference data}\\longrightarrow A,\\\\ x\\longrightarrow A(x),\\\\ \\text{alert if }A(x)>\\tau.\\end{gathered}'}</MathBlock>
    <Prose>Fitting decides the score. Thresholding decides who gets attention. Two thresholds give different alert counts from exactly the same model and the same ranking, so a disagreement about alert volume is often not a disagreement about the model at all.</Prose>
    <ProvenanceFigure />
    <Prose>Do not fit a scaler on the complete series before splitting: that lets later information influence earlier distances. Where rows repeat by customer, device or patient, keep whole groups on one side of the split when the intended test is performance on new groups. For future monitoring, preserve chronological order.</Prose>

    <H3>Use one score orientation at the decision boundary</H3>
    <Prose>The contracts below refer to the APIs inspected for scikit-learn 1.9.1. Keep the model’s own offset distinct from a threshold your application calibrates.</Prose>
    <LessonTable caption="Library outputs and the anomaly-oriented quantity used in this lesson" headers={['output', 'direction and meaning', 'our quantity']} rows={[
      ['Isolation Forest score_samples(X)', 'Smaller is more unusual; a negative isolation-based score', '−score_samples(X)'],
      ['One-Class SVM score_samples(X)', 'Unshifted kernel score; smaller is less compatible with the region', '−score_samples(X)'],
      ['LOF negative_outlier_factor_', 'Negative training-row LOF values', '−negative_outlier_factor_'],
      ['LOF novelty=True score_samples(X_new)', 'Negative LOF-style scores for new queries against the frozen reference', '−score_samples(X_new)']
    ]} />
    <Prose>The decision function subtracts an offset from the normality-oriented score, so a negative decision value means outside the estimator’s own selected boundary. That is not a calibrated probability, and the kernel decision value is not a Euclidean distance in the original space.</Prose>
    <Prose>The rest of the lesson uses <Math>{'A>\\tau'}</Math> with a <strong>strict</strong> inequality. Scores tied exactly at the threshold stay unflagged. A percentile setting therefore need not flag the fraction it names, which section 8 makes concrete.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>Choose a cut uniformly along the interval from 0 to 12. Any cut between 3 and 12 separates the last point from the other four, and that interval occupies 9 of the 12 units, so the first-cut isolation probability is</Prose>
    <MathBlock>{'\\frac{12-3}{12-0}=\\frac34.'}</MathBlock>
    <Prose>To isolate 0 on the first cut, the cut must land between 0 and 1: probability <Math>{'1/12'}</Math>. No classifier has learned the label “bad”. The empty gap alone gives 12 many opportunities for early separation.</Prose>
    <FirstCutFigure />
    <Prose>In several dimensions an isolation tree chooses a feature and a cut between that feature’s current minimum and maximum, then repeats inside each child. A path records how many cuts a query follows before reaching a terminal node. Many random trees reduce the dependence on any one lucky cut.</Prose>
    <IsolationLab />

    <H3>Why a terminal node can hold several observations</H3>
    <Prose>A practical tree stops at a depth limit, commonly <Math>{'\\lceil\\log_2\\psi\\rceil'}</Math> for subsample size <Math>{'\\psi'}</Math>, or when it can no longer separate the remaining values. If the terminal node still holds <Math>{'m'}</Math> reference rows, stopping does not mean all <Math>{'m'}</Math> were individually isolated, so we add an average remaining-path correction:</Prose>
    <MathBlock>{'\\begin{gathered}c(m)=2H_{m-1}-\\frac{2(m-1)}m,\\\\ H_r=1+\\tfrac12+\\cdots+\\tfrac1r,\\end{gathered}'}</MathBlock>
    <Prose>with <Math>{'c(0)=c(1)=0'}</Math> and <Math>{'c(2)=1'}</Math>. This normaliser comes from an average search-path calculation, not from a model of anomaly probability. Implementations often approximate the harmonic term for larger <Math>{'m'}</Math>; our tiny calculations use the exact finite sum.</Prose>
    <Prose>If a query reaches depth <Math>{'d'}</Math> at a leaf of size <Math>{'m'}</Math>, its corrected path is <Math>{'h=d+c(m)'}</Math>. Average across trees, then normalise:</Prose>
    <MathBlock>{'s(x)=2^{-\\mathbb E[h(x)]/c(\\psi)}.'}</MathBlock>
    <Prose>If the average path equals the normaliser, the exponent is −1 and the score is exactly <Math>{'1/2'}</Math>. Shorter paths give larger scores. The exponential rescales a path statistic, so 0.8 does <strong>not</strong> mean an 80% chance of failure.</Prose>
    <Prose>For the five positions, <Math>{'c(5)=77/30\\approx2.5667'}</Math>. Integrating exactly over all one-dimensional cut sequences, with depth cap 3 and the leaf correction above, gives:</Prose>
    <LessonTable caption="Exact expectations over the one-dimensional cut construction, depth cap 3" headers={['position', 'expected corrected path', 'score']} rows={[
      ['0', '31/12 ≈ 2.5833', '0.497755'],
      ['1', '73/22 ≈ 3.3182', '0.408159'],
      ['2', '17/5 = 3.4', '0.399239'],
      ['3', '17/6 ≈ 2.8333', '0.465258'],
      ['12', '841/660 ≈ 1.2742', '0.708845']
    ]} />
    <Prose>These are expectations over that construction, <strong>not</strong> outputs promised by a finite random forest or by the library. Replace 12 with 4 and the endpoints tie at about 0.569715. Make all five values identical and every corrected path equals <Math>{'c(5)'}</Math>, so every score is 0.5 and the ranking carries no information at all. The lab above reproduces each of these cases; change the positions and read the intervals.</Prose>

    <H3>Run the examples</H3>
    <Prose>Use Python 3.12 in a virtual environment. From a terminal in your example folder, install the versions the calculations used:</Prose>
    <CodeBlock language="bash">{'python -m venv .venv\n# Windows PowerShell:\n.\\.venv\\Scripts\\Activate.ps1\n# macOS or Linux:\n# source .venv/bin/activate\npython -m pip install numpy==2.3.5 pandas==3.0.1 scikit-learn==1.9.1'}</CodeBlock>
    <Prose>Save each Python block in its own file, named <Code>isolation_example.py</Code>, <Code>lof_modes.py</Code>, <Code>lof_arithmetic.py</Code>, <Code>kernel_boundary.py</Code> and <Code>temperature_monitor.py</Code>, and run one with <Code>python isolation_example.py</Code>. The last program also needs the linked CSV and JSON beside it. These are small teaching programs that expose the calculation; they are not hardened monitoring services.</Prose>
    <Prose>The program below uses one-dimensional trees, the exact harmonic correction, a local random generator and subsampling. It accepts ordinary finite numeric input with at least two rows. The library’s construction and its harmonic approximation need not match these scores.</Prose>
    <Program example={anomalyExamples.isolation}>
      <Prose>The seeded run identifies 12, and the duplicate-only construction gives 0.5 and 0.5. Look at the fitted sample size: asking for 256-row subsamples does not put 256 observations in a five-row tree, and normalising as though it did would change what the score means.</Prose>
    </Program>

    <H3>Subsampling and representation</H3>
    <Prose>If unusual observations arrive as a group, its members can shield one another from rapid isolation, one form of <strong>masking</strong>. A smaller subsample often leaves fewer of them together and exposes the separation, but it can equally omit a legitimate rare mode. There is no universally best subsample size.</Prose>
    <Prose>The original method uses axis-aligned cuts. Translating a coordinate, or rescaling it by a positive factor, preserves the corresponding uniform-cut construction in exact arithmetic; arbitrary rotations generally do not. Irrelevant features waste cuts. Random-projection variants change the mechanism and should not be substituted silently for the method derived here.</Prose>

    <H2>{headings[3]}</H2>
    <Prose>Consider two legitimate groups on a line: 0, 1, 2 and 20, 24, 28. The second is more spread out. A global nearest-distance rule can penalise it even though its spacing is internally consistent. LOF asks a local question instead: <strong>is this point less locally supported than its own neighbours are?</strong></Prose>
    <Prose>We use exactly <Math>{'k=2'}</Math> other reference rows throughout, with stable row-order tie breaking, and distance is the absolute difference. <em>Self</em> means the same observation identity, not every row that happens to share a coordinate.</Prose>
    <H3>Step 1: measure each neighbour’s usual radius</H3>
    <Prose>For a reference point <Math>{'o'}</Math>, let <Math>{'r_k(o)'}</Math> be the distance to its kth selected neighbour.</Prose>
    <H3>Step 2: put a floor under each distance</H3>
    <MathBlock>{'\\begin{gathered}\\operatorname{reach}_k(p,o)\\\\ =\\max\\{d(p,o),\\,r_k(o)\\}.\\end{gathered}'}</MathBlock>
    <Prose>The radius belongs to <strong>neighbour <Math>{'o'}</Math></strong>. The floor stops an exceptionally short distance to <Math>{'o'}</Math> from manufacturing an arbitrarily large local density just because <Math>{'p'}</Math> nearly coincides with it. Swapping <Math>{'p'}</Math> and <Math>{'o'}</Math> changes which radius applies, so reachability need not be symmetric.</Prose>
    <ReachFloorFigure />
    <Prose>If you read the optional density-hierarchy branch of the previous lesson, keep the formulas apart. OPTICS reachability uses the <em>source</em> point’s core radius; HDBSCAN mutual reachability uses <em>both</em> endpoints’ core radii; LOF uses the <em>neighbour’s</em> radius in this directed comparison. Our k counts other rows and excludes the training row itself, whereas DBSCAN’s min_samples counts the point itself. None of those earlier formulas is needed to take the maxima above.</Prose>
    <H3>Step 3: take the reciprocal of average reach</H3>
    <MathBlock>{'\\operatorname{lrd}_k(p)^{-1}=\\frac1k\\sum_{o\\in N_k(p)}\\operatorname{reach}_k(p,o)'}</MathBlock>
    <Prose>The reciprocal is large when average reach is small. Its units are inverse distance, and it is <strong>not</strong> a probability density normalised to integrate to one. At reference point 1 both distances are 1, but the radii of neighbours 0 and 2 are both 2, so both reaches are 2 and <Math>{'\\operatorname{lrd}(1)=1/((2+2)/2)=1/2'}</Math>. At point 0 the reaches are 1 and 2, giving <Math>{'2/3'}</Math>. All six are <Math>{'2/3,\\;1/2,\\;2/3,\\;1/6,\\;1/8,\\;1/6'}</Math>.</Prose>
    <H3>Step 4: take a dimensionless comparison</H3>
    <MathBlock>{'\\operatorname{LOF}_k(p)=\\frac1k\\sum_{o\\in N_k(p)}\\frac{\\operatorname{lrd}_k(o)}{\\operatorname{lrd}_k(p)}.'}</MathBlock>
    <Prose>Near 1 means comparable local support. Larger means the selected neighbours have greater density proxies than the query. Values below 1 are ordinary, and no theorem says an acceptable observation scores exactly 1. For point 1 both neighbour densities are <Math>{'2/3'}</Math>, so the factor is <Math>{'(2/3)/(1/2)=4/3'}</Math>. For point 0 the mean neighbour density is <Math>{'7/12'}</Math>, so the factor is <Math>{'(7/12)/(2/3)=7/8'}</Math>. The wider group repeats the pattern exactly: <Math>{'7/8,\\;4/3,\\;7/8,\\;7/8,\\;4/3,\\;7/8'}</Math>. Even this regular example has non-unit values, so “anything above 1 is bad” confuses a finite-neighbourhood effect with an application decision.</Prose>

    <H3>Farther from a reference point, yet the lower factor</H3>
    <Prose>Freeze those six rows as the reference. Query 4 has neighbours 2 and 1 at distances 2 and 3, with radius floors 2 and 1, so its reaches are 2 and 3 and its density is <Math>{'2/5'}</Math>. The mean reference density is <Math>{'7/12'}</Math>:</Prose>
    <MathBlock>{'\\operatorname{LOF}(4)=\\frac{7/12}{2/5}=\\frac{35}{24}\\approx1.4583.'}</MathBlock>
    <Prose>Query 17 has neighbours 20 and 24, with reaches <Math>{'\\max(3,8)=8'}</Math> and <Math>{'\\max(7,4)=7'}</Math>, so its density is <Math>{'2/15'}</Math> and its mean neighbour density is <Math>{'7/48'}</Math>:</Prose>
    <MathBlock>{'\\operatorname{LOF}(17)=\\frac{7/48}{2/15}=\\frac{35}{32}.'}</MathBlock>
    <Prose>That is about 1.09375, against 1.4583 for query 4. Query 17 is <em>farther</em> from its nearest reference row, 3 units against 2, and yet takes the lower factor. Its nearby group normally has wider spacing, which is precisely the comparison LOF is built to make.</Prose>
    <LofNeighbourhoodLab />

    <H3>Ties and duplicates change the contract</H3>
    <Prose>The original paper includes every point within the kth-neighbour distance, so ties can give more than <Math>{'k'}</Math> neighbours. Our calculation selects exactly <Math>{'k'}</Math>, as the library’s fixed-neighbour calculation does, and tied selections can then depend on implementation ordering. State the convention before comparing anyone’s numbers with anyone else’s.</Prose>
    <Prose>Repeated coordinates need care. Exclude a training row by identity, not by deleting every zero distance or by dropping the first sorted entry. If all the relevant reaches vanish, unstabilised reciprocals are infinite and the ratios are undefined. Library stabilisation makes the arithmetic computable; it does not create separation between identical records. The hand calculations here keep distinct coordinates and positive reaches for exactly this reason.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>A training row leaves its own identity out of its neighbourhood. A <strong>new query</strong> has no identity here, so if it lands on a training coordinate, that reference row is a perfectly valid zero-distance neighbour. The neighbour sets differ, so the scores can differ. They can also agree numerically: at coordinate 0 with k = 2, both give 7/8 despite using different neighbours.</Prose>
    <Prose>The consequence is worth stating plainly: scoring the training array as queries does not recover the training rows’ factors. In our example, training rows 1 and 24 have factor <Math>{'4/3'}</Math>; treated as new queries, the same coordinates score <Math>{'7/8'}</Math>.</Prose>
    <LofModeLab />
    <Program example={anomalyExamples.lofModes}>
      <Prose>Use the default <Code>novelty=False</Code> with <Code>fit_predict</Code> when you are reviewing the fitted collection. Use <Code>novelty=True</Code> when later queries are the task, and score those queries with <Code>score_samples</Code>, <Code>decision_function</Code> or <Code>predict</Code>. New queries never become one another’s neighbours, so scoring a batch does not adapt the frozen reference.</Prose>
    </Program>
    <Prose>The next program exposes the floors and the ratios directly. It uses exactly k other rows, stable ties and distinct one-dimensional reference values, and it deliberately omits a production search index and duplicate stabilisation.</Prose>
    <Program example={anomalyExamples.lofArithmetic}>
      <Prose>The full pairwise matrix makes the calculation inspectable but uses quadratic storage. Use a suitable neighbour search instead of this matrix on a large dataset.</Prose>
    </Program>

    <H2>{headings[5]}</H2>
    <Prose>Suppose we have reviewed reference observations but few failures. We can still ask for a region that represents the reference reasonably well while allowing some training observations to fall outside it.</Prose>
    <Prose>One-Class SVM constructs a separating hyperplane in a feature space. A <strong>kernel</strong> computes similarities in that space without ever building every feature. The RBF kernel is</Prose>
    <MathBlock>{'\\begin{gathered}K(x,z)=\\exp(-\\gamma\\|x-z\\|^2),\\\\ \\gamma>0.\\end{gathered}'}</MathBlock>
    <Prose>When <Math>{'x=z'}</Math> the squared distance is zero and the similarity is 1; greater distance reduces it. Increasing gamma makes the similarity decay over a shorter input distance, so features and units must already be meaningful before you choose gamma. A fitted decision has the form</Prose>
    <MathBlock>{'g(x)=\\sum_i\\alpha_iK(x_i,x)-\\rho.'}</MathBlock>
    <Prose>The non-negative weights pick out the reference observations that shape the boundary; those with non-zero weights are the support vectors. The offset rho fixes the zero contour: positive <Math>{'g'}</Math> is inside the selected region and negative is outside. None of this implies a convex or even connected region in the input space, because a hyperplane in a nonlinear feature space can describe separated pieces in the original coordinates.</Prose>
    <H3>Two reference observations expose the mechanism</H3>
    <Prose>Take <Math>{'x_1=-1'}</Math>, <Math>{'x_2=1'}</Math> and <Math>{'\\nu=1/2'}</Math>, using the normalised optimization derived in section 9. Symmetry gives <Math>{'\\alpha_1=\\alpha_2=1/2'}</Math> and puts both references on the boundary:</Prose>
    <MathBlock>{'\\begin{gathered}\\rho=\\frac{1+e^{-4\\gamma}}2,\\\\ g(x)=\\frac{e^{-\\gamma(x+1)^2}+e^{-\\gamma(x-1)^2}}2-\\rho.\\end{gathered}'}</MathBlock>
    <Prose>At the midpoint the two similarities agree, so <Math>{'g(0)=e^{-\\gamma}-(1+e^{-4\\gamma})/2'}</Math>. For gamma 0.1 that is about +0.069677 and the midpoint belongs to the learned region. For gamma 1 it is about −0.141278: the midpoint is <em>outside</em> while both reference points remain exactly on the boundary. At gamma 1, moving a little inward from either reference gives a positive score, so the non-negative region has separated pieces. Describing every One-Class SVM region as “one connected blob of normal data” would miss this entirely.</Prose>
    <KernelBoundaryLab />
    <Program example={anomalyExamples.kernel} />

    <H3>Nu is not tomorrow’s fault prevalence</H3>
    <Prose>In exact optimization, under the conditions in section 9, nu bounds a fraction of training margin violations and a fraction of support vectors. It is a constraint-and-regularisation parameter. It is not known anomaly prevalence, not a promised future false-positive rate, and not a guarantee that exactly nu times n training predictions come out negative. Solver tolerances and the treatment of boundary points matter whenever you compare software counts with the theorem, so use a separate calibration set when the application needs an explicit alert budget.</Prose>
    <Prose>Two API details cause avoidable confusion. <Code>gamma='scale'</Code> sets gamma from the fitted data’s variance; it does not standardise each feature. And gamma is not the RBF standard deviation: in the alternative form <Math>{'\\exp(-\\|x-z\\|^2/(2\\sigma^2))'}</Math>, gamma equals <Math>{'1/(2\\sigma^2)'}</Math>.</Prose>

    <H2>{headings[6]}</H2>
    <LessonTable caption="Match the mechanism to a question about the representation" headers={['question about the representation', 'mechanism', 'what to inspect']} rows={[
      ['Do unusual observations separate after few random cuts?', 'Isolation Forest', 'Feature relevance, axis orientation, subsample variability, rare legitimate groups'],
      ['Do acceptable regions have different local spacings?', 'LOF', 'Neighbour count, metric, duplicates, whether the reference represents those regions'],
      ['Can reviewed observations define a useful nonlinear support region?', 'One-Class SVM', 'Scaling, gamma, nu, kernel cost, later-data stability'],
      ['Is a simple baseline sufficient?', 'Domain rule, residual or robust deviation', 'Its assumptions and calibration under the same evaluation protocol']
    ]} />
    <Prose>There is no universal ranking in that table. A comparison is only meaningful with the available training information and the evaluation protocol held fixed. Give one method reviewed normal data and another a contaminated batch and you have changed the task, not only the algorithm.</Prose>
    <Prose>Four applications make the representation issue concrete:</Prose>
    <Prose><strong>Calibration drift in an instrument.</strong> Monitor residuals against a stable reference standard rather than raw readings that legitimately change with the specimen. Monitoring within one instrument and transferring to unseen instruments need different splits.</Prose>
    <Prose><strong>A stuck sensor.</strong> Each repeated value may be entirely common. Near-zero trailing variability and unusual duration expose the collective event that a level-only detector misses.</Prose>
    <Prose><strong>Scientific sample or manufacturing-batch review.</strong> A rare specimen may be the most interesting legitimate observation in the batch. Rank it for inspection and preserve its identity and raw measurements instead of deleting it automatically.</Prose>
    <Prose><strong>Unexpected access patterns.</strong> Session-level features can expose a new combination of ordinary actions. Repeated activity by one account calls for account-aware evaluation when the test is transfer to new accounts.</Prose>
    <Prose>Each of those names a unit, a context, a reference and an intended response. “Remove every outlier before training” is not an adequate policy, because the unusual measurements are often the investigation’s target.</Prose>

    <H2>{headings[7]}</H2>
    <Prose>A hypothetical detector catches 80% of faults and flags 1% of non-fault observations. Faults occur in 0.1% of 100,000 observations. The expected counts are 100 faults, of which 80 are flagged; 99,900 non-fault observations, of which 999 are flagged; 1,079 alerts in total, of which only about 7.4% correspond to faults. These are specified hypothetical rates, not a claim about any real application. A low false-positive rate still overwhelms reviewers when the non-fault population is far larger.</Prose>
    <MathBlock>{'P(\\text{fault}\\mid\\text{alert})=\\frac{pt}{pt+(1-p)f}'}</MathBlock>
    <Prose>for prevalence <Math>{'p'}</Math>, sensitivity <Math>{'t'}</Math> and false-positive rate <Math>{'f'}</Math>, provided the alert probability is positive. The numerator counts the fraction that are faults and flagged; the denominator counts everything flagged. If nobody is flagged, precision is undefined rather than automatically 0 or 1.</Prose>
    <AlertPopulationLab />

    <H3>Calibrate the action separately</H3>
    <Prose>With representative reviewed calibration data you can choose a threshold from workload, costs or a labelled operating point. An unlabelled empirical score quantile controls an <strong>observed calibration alert fraction</strong>, which is not a known false-positive fraction.</Prose>
    <ThresholdRulerFigure />
    <Prose>Numeric contamination in Isolation Forest and LOF selects their fitted score offset. It does not discover the true fraction of faults: with every other setting fixed, changing that offset changes labels without changing the ranking at all. One-Class SVM has nu instead, and its effect is not merely a post-fit percentile.</Prose>
    <Prose>If reliable labels exist, inspect precision–recall behaviour, recall at an affordable budget and actual counts. ROC summaries do not display alert workload by themselves, and changing only the threshold cannot improve ranking AUC because the score order is unchanged. Labels obtained only for reviewed top-ranked items tell you nothing about recall among everything never reviewed.</Prose>
    <Prose>For incidents, say whether the evaluation counts <strong>rows or events</strong>. Consecutive positives from one physical incident are not independent successful detections. State the alert grouping, the event matching and the latency definition before reporting a number.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>Let phi be the feature map and n the number of reference observations. The normalised primal problem is</Prose>
    <MathBlock>{'\\min_{w,\\rho,\\xi}\\;\\frac12\\|w\\|^2+\\frac1{\\nu n}\\sum_{i=1}^n\\xi_i-\\rho'}</MathBlock>
    <Prose>subject to <Math>{'\\langle w,\\phi(x_i)\\rangle\\ge\\rho-\\xi_i'}</Math>, <Math>{'\\xi_i\\ge0'}</Math> and <Math>{'0<\\nu\\le1'}</Math>. The term <Math>{'-\\rho'}</Math> rewards moving the separating level away from the origin, the norm restrains <Math>{'w'}</Math>, and slack lets a reference observation sit below the level at a cost. This is the origin-separating formulation; a support-vector enclosing-sphere formulation is related but should not be conflated with it without stating the equivalence conditions.</Prose>
    <Prose>Introduce non-negative multipliers alpha for the first constraints and beta for the non-negative slack. Stationarity of the Lagrangian gives <Math>{'w=\\sum_i\\alpha_i\\phi(x_i)'}</Math>, <Math>{'\\sum_i\\alpha_i=1'}</Math> and <Math>{'\\alpha_i+\\beta_i=1/(\\nu n)'}</Math>. Substituting yields</Prose>
    <MathBlock>{'\\begin{gathered}\\min_\\alpha\\;\\frac12\\sum_{i,j}\\alpha_i\\alpha_jK(x_i,x_j),\\\\ 0\\le\\alpha_i\\le\\frac1{\\nu n},\\qquad\\sum_i\\alpha_i=1.\\end{gathered}'}</MathBlock>
    <Prose>A positive-semidefinite kernel makes this a convex quadratic problem. In the two-reference example, symmetry and minimising this quadratic give equal weights, and the cap <Math>{'1/(\\nu n)=1'}</Math> allows them.</Prose>
    <Prose>The standard nu property is stated for an exact solution with non-zero rho. A <strong>strict margin violator</strong> has positive slack, which forces its multiplier to the cap. With <Math>{'m'}</Math> such observations, <Math>{'m/(\\nu n)\\le\\sum_i\\alpha_i=1'}</Math>, so <Math>{'m\\le\\nu n'}</Math>. And if <Math>{'s'}</Math> support vectors each contribute at most <Math>{'1/(\\nu n)'}</Math> to a sum of 1, then <Math>{'1\\le s/(\\nu n)'}</Math>, so <Math>{'s\\ge\\nu n'}</Math>. Both statements concern strict training violations and non-zero multipliers, never unseen labels. A point exactly on the boundary is not a strict violator, and approximate software results with tiny signed decisions near zero are not exact evidence about the theorem.</Prose>
    <Prose>A free support vector, with <Math>{'0<\\alpha_i<1/(\\nu n)'}</Math>, gives <Math>{'\\rho=\\sum_j\\alpha_jK(x_j,x_i)'}</Math> by complementary slackness. With no free support vector, use appropriate offset bounds instead of that convenient equality. Library coefficient normalisations can differ from the paper’s formulation, so account for them before comparing multiplier values.</Prose>

    <H2>{headings[9]}</H2>
    <Prose>We now use the supplied <a href="/learn-assets/anomaly-detection/machine_temperature_system_failure.csv" download>machine_temperature_system_failure.csv</a> from the Numenta Anomaly Benchmark. It records the temperature of an industrial machine’s internal component. The source describes a planned shutdown and later failure-related behaviour, and the supplied <a href="/learn-assets/anomaly-detection/nab-event-windows.json" download>annotations</a> contain <strong>four time windows</strong>. They are not verified fault labels for every row, and there is no documented one-to-one mapping between the windows and that prose description.</Prose>
    <Prose>Both files are pinned to an upstream commit and supplied offline under the benchmark’s <a href="/learn-assets/anomaly-detection/NAB-LICENSE.txt" download>MIT license</a>; the references give the exact commit. The source does not state temperature units or a timestamp timezone, so we use the recorded units and the timestamps as supplied.</Prose>

    <H3>Inspect the stream before fitting</H3>
    <Prose>There are 22,695 raw rows and 22,683 unique timestamps: twelve extra rows share a timestamp with another row. We explicitly average measurements at the same timestamp, which is a chosen measurement policy rather than a silent deletion. For a level <Math>{'v_t'}</Math> we then form <Math>{'x_t=[v_t,\\;v_t-v_{t-1\\mathrm{h}}]'}</Math>.</Prose>
    <Prose>The second feature looks up the exact timestamp one hour earlier. It does not assume that twelve rows earlier always means one hour. Rows without that earlier measurement are excluded, twelve of them here, and nothing is interpolated or filled from a future value. The result is causal once the current timestamp’s measurements are available. If duplicate records can arrive late, a deployment needs a closing delay or a revision policy; retrospective aggregation does not establish zero-latency operation.</Prose>
    <LessonTable caption="Each period supplies exactly one thing, and 22,671 feature rows remain" headers={['role', 'time interval', 'feature rows']} rows={[
      ['Fit the reference and the scaler', 'Before 6 December 2013', '885'],
      ['Calibrate the threshold', '6 December through before 10 December', '1,152'],
      ['Inspect later performance', 'From 10 December onward', '20,634']
    ]} />
    <Prose>The early reference is an operating assumption. Having no published annotation window in that period does not certify it as fault-free. Fit the scaler there alone and freeze its means and scales for every later period.</Prose>

    <H3>Give a simple baseline the same opportunity</H3>
    <Prose>The baseline scores absolute level deviation from the reference median, divided by the reference median absolute deviation:</Prose>
    <MathBlock>{'\\begin{gathered}A_{\\mathrm{base}}(v)=\\frac{|v-m|}{s},\\\\ m=\\operatorname{median}(v_{\\mathrm{fit}}),\\\\ s=\\operatorname{median}(|v_{\\mathrm{fit}}-m|).\\end{gathered}'}</MathBlock>
    <Prose>Here the reference median <Math>{'m'}</Math> is about 81.8620 and the reference deviation <Math>{'s'}</Math> about 4.42426, in recorded temperature units. We use the raw deviation without a normal-consistency multiplier. A zero denominator would need another declared scale or a constant-reference policy; this dataset’s is positive.</Prose>
    <Prose>Keep one difference visible: the baseline sees the level only, while the learned detectors see level and one-hour change. This is therefore not an algorithm comparison on identical features. All three learned detectors share the reference rows and the scaler, and their settings are fixed in advance: 100 isolation trees with subsample 256 and seed 17; an RBF One-Class SVM with gamma 0.5 and nu 0.05; novelty LOF with 20 neighbours. We compare two predeclared calibration quantiles, 0.95 and 0.99, and do not pick a winner using the test annotations.</Prose>
    <Checkpoint prompt="The 0.99 calibration quantile leaves 11 alerts among the 1,152 calibration rows. Does about 1% of the 20,634 later rows follow?">
      <Prose>No. The quantile fixes a <em>score</em>, and the later rows are a different population whose score distribution can move. Below, that same 0.99 threshold produces 1,548 later alerts for Isolation Forest, about 7.5% of the period, and 9,232 for One-Class SVM, about 45%. An empirical calibration percentile controls the calibration alert fraction and nothing else.</Prose>
    </Checkpoint>

    <H3>The complete offline analysis</H3>
    <Prose>Save this program beside the supplied CSV and JSON. It needs NumPy, pandas and scikit-learn, and it downloads nothing. The author calculations used Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 and scikit-learn 1.9.1; later releases may need an API and output check.</Prose>
    <Program example={anomalyExamples.temperature}>
      <Prose>The test period holds 2,268 rows inside windows and 18,366 outside. Every method alerts in all four windows, and yet the workload differs by an order of magnitude.</Prose>
    </Program>
    <Prose><strong>Outside-window alerts are unmatched workload, not verified false positives.</strong> Inside-window rows are not individually confirmed faults either. We have not run the benchmark’s official scoring, identified precise fault onsets or carried out a prospective early-warning study. An alert at a window’s beginning is early relative to an annotation boundary; it is not proof of prediction before failure.</Prose>
    <TemperatureThresholdLab />
    <Prose>Investigate the representation and the stability of the reference rather than tuning until the complex detector beats the baseline. Documented load or operating mode would help if it existed; we cannot invent variables the dataset never recorded. A sound next study compares the level-only and level-plus-change representations on a separate validation period, locks the choice, and only then evaluates a final future period.</Prose>

    <H2>{headings[10]}</H2>
    <H3>When does local regularity keep LOF near 1?</H3>
    <Prose>Suppose every reachability distance needed for a point <strong>and for its neighbours’ density calculations</strong> lies between positive <Math>{'a'}</Math> and <Math>{'b'}</Math>. Then each average reach lies in <Math>{'[a,b]'}</Math>, each local density in <Math>{'[1/b,1/a]'}</Math> and each ratio in <Math>{'[a/b,b/a]'}</Math>. Averaging preserves the bounds, so <Math>{'a/b\\le\\operatorname{LOF}\\le b/a'}</Math>. When <Math>{'a'}</Math> and <Math>{'b'}</Math> are close, the interval stays near 1. The assumption covers the neighbours’ own neighbourhoods too, so checking only the query’s distances is not enough. It explains local regularity; it does not supply a universal alert threshold.</Prose>
    <H3>What grows with reference size?</H3>
    <Prose>For T isolation trees with capped subsample size psi, the original average-case account is roughly <Math>{'T\\psi\\log\\psi'}</Math> for construction and <Math>{'nT\\log\\psi'}</Math> for scoring n rows, with feature processing and tree shape contributing as well. Reading a massive dataset still costs work: a fixed subsample does not make the whole application independent of its input size.</Prose>
    <Prose>LOF needs neighbourhoods and local density statistics, and a brute-force distance matrix uses quadratic storage. Search indexes help in favourable dimensions, while high-dimensional distances can make both the search and the interpretation difficult. Approximate neighbours change the scores and require a declared tradeoff.</Prose>
    <Prose>Kernel One-Class SVM can require substantial pairwise-kernel work and storage. Its practical limit depends on the data, the solver and the kernel, not on a universal row cutoff. Kernel approximations combined with a suitable linear one-class learner are a different computation and should be evaluated rather than advertised as identical.</Prose>
    <H3>Updating a reference changes the question</H3>
    <Prose>Frozen novelty LOF does not learn from each new batch, and neither does a fitted One-Class SVM or an ordinary Isolation Forest. Changed equipment behaviour can make a former reference unsuitable, so state when a new reference is allowed, which reviewed observations may enter it, and how old incidents are protected from being normalised away. Compare distributions and workload over time, but remember that stable scores alone do not prove stable fault detection. Keep model, scaler and threshold versions together, and let rolling updates use only information available by that time: hindsight selection of a clean-looking reference leaks future knowledge.</Prose>

    <H2>{headings[11]}</H2>
    <Prose>Each task changes the worked example. Try the question before opening its hint or its solution, and round only at the final step.</Prose>
    <Practice title="A. A different isolation gap" question="Reference values are 0, 2, 3, 4, 10, and a first cut is uniform between the minimum and the maximum. What is the probability of immediately isolating 10? What is it for 0? Does the larger probability establish that 10 is faulty?" hint="Find the intervals that produce a singleton, then divide their lengths by the full span.">
      <Prose>Cuts in (4, 10) isolate 10, giving 6/10 = 0.6. Cuts in (0, 2) isolate 0, giving 2/10 = 0.2. Landing exactly on a value has probability zero in this continuous construction. The result is a geometric separation advantage, not a fault label. Set these five positions in the isolation lab to see both intervals.</Prose>
    </Practice>
    <Practice title="B. Finish a truncated path" question="A depth-capped tree was fitted on four rows. A query reaches depth 2 in a leaf holding two reference rows. Compute its corrected path and normalised score. What goes wrong if the denominator uses a requested max_samples of 256?" hint="c(2) = 1 and c(4) = 13/6. The denominator describes the sample actually fitted. The state is reachable: successive splits can leave 3 and then 2 rows on the query’s path.">
      <Prose>The path is 2 + 1 = 3, so the single-tree normalised score is <Math>{'2^{-3/(13/6)}=2^{-18/13}\\approx0.3830'}</Math>. Using c(256) compares this four-row tree with a far larger reference construction and inflates the score. This is a supplied terminal state, not a claim that one tree gives a reliable forest estimate. The isolation lab always holds five positions, so it cannot build this four-row tree; the arithmetic here is the whole exercise.</Prose>
    </Practice>
    <Practice title="C. Change the local reach calculation" question="A query’s three neighbours are at distances 0.4, 0.6 and 1.0. Their kth-neighbour radii are 0.3, 0.8 and 0.5, and their reference local densities are 1, 2 and 1.5. Find the query’s density and its factor. Then multiply every distance and radius by 10, preserving the geometry. What happens to the densities and to the factor?" hint="Apply each maximum first. Take the reciprocal of the average reach, not the average of the reciprocals.">
      <Prose>The reaches are 0.4, 0.8 and 1.0, with mean 11/15, so the query density is 15/11. The mean neighbour density is 1.5, so the factor is 1.5/(15/11) = 1.1. A common positive scale factor of 10 divides every density, the reference densities included, by 10, so the ratio stays 1.1. Rescaling only one coordinate of multidimensional data need not preserve the geometry, and then nothing is guaranteed.</Prose>
    </Practice>
    <Practice title="D. Diagnose the invalid comparison" question="A novelty LOF model is fitted on reviewed references. You compare the negated score_samples on those references with the negated negative_outlier_factor_, expecting equality, and they differ. Is that necessarily a bug? How should you compare training and later observations?" hint="Draw the neighbours of a new query that sits exactly on a training coordinate.">
      <Prose>It is not a bug. Query scoring may include the coordinate-matching reference row, whereas training LOF excludes its own identity: two different neighbourhood contracts. Use the training factors as an in-sample diagnostic and query scores on a separate later calibration or test set. Do not pool them as though they were the same held-out measurement. The fitting-mode lab reproduces both numbers side by side.</Prose>
    </Practice>
    <Practice title="E. Widen the reference anchors" question="Move the One-Class SVM anchors to −2 and +2, keep nu = 1/2 and set gamma = 0.25. Derive the midpoint decision and compare it with anchors −1 and +1 at gamma = 1. Why do they agree?" hint="The squared separation is now 16 and the squared midpoint distance is 4.">
      <Prose>Equal weights still apply, so rho is <Math>{'(1+e^{-16\\gamma})/2=(1+e^{-4})/2'}</Math>. The midpoint sum is <Math>{'e^{-4\\gamma}=e^{-1}'}</Math>, so <Math>{'g(0)\\approx-0.141278'}</Math>, the same value as before. Doubling the distances and dividing gamma by four preserves every RBF exponent. Units and gamma have to be considered together, which is why gamma copied from another dataset means nothing on its own.</Prose>
    </Practice>
    <Practice title="F. A team with 200 review slots" question="There are 50,000 observations, fault prevalence 0.2%, sensitivity 90% and false-positive rate 0.5%. Compute the expected alerts and the precision. Is a 200-review budget sufficient? Does reviewing only the top 200 preserve 90% sensitivity?" hint="Count the fault and non-fault populations separately.">
      <Prose>There are 100 faults, giving 90 true alerts, and of 49,900 non-fault observations 249.5 are expected to be flagged. The expected total is 339.5 alerts with precision 90/339.5 ≈ 26.51%. A fractional expected count describes an average, not a fractional record. The budget is insufficient, and raising the threshold or taking the top 200 changes the operating point, so the old sensitivity cannot be carried over. Enter these four numbers in the review-queue lab to see the population diagram.</Prose>
    </Practice>
    <Practice title="G. Training bound or test promise?" question="An exact One-Class SVM solution has n = 80, nu = 0.15 and non-zero rho. Give the strict training violation bound and the support-vector bound. Does it guarantee at most 12 false alerts in the next 80 observations?" hint="Each multiplier is at most 1/12 and they sum to 1.">
      <Prose>At most 12 strict training violators and at least 12 support vectors. No future false-alert bound follows: the future population can differ, boundary points are a separate case, and the theorem contains no future fault labels at all.</Prose>
    </Practice>
    <Practice title="H. Write an honest temperature recommendation" question="Use the 0.99 rows of the real-data table. Which method has the smallest unmatched row workload, and how much larger is the One-Class SVM workload? Why is that insufficient to establish the best fault detector? Write five sentences including a next experiment." hint="Use the same 18,366 outside-window rows, preserve the annotation limits and remember the different feature sets.">
      <Prose>The baseline has 445 unmatched alerts and One-Class SVM has 7,913, about 17.78 times as many. A suitable report reads: “Under the fixed reference period and the 0.99 calibration quantile, all methods alerted in all four published windows. The level-only baseline had 445 outside-window row alerts, against 577 for Isolation Forest, 7,913 for One-Class SVM and 6,699 for novelty LOF. These are not verified false positives, and four-window coverage does not measure onset or alert usefulness. The baseline uses level while the learned detectors use level plus one-hour change. I would compare matched representations on an additional validation period, investigate operating-regime change, and reserve a later period for a locked final comparison.”</Prose>
    </Practice>
    <Practice title="I. An event metric can hide repeated work" question="Two non-overlapping event windows cover rows 3–5 and 9–11. Detector A alerts at 3, 4, 5, 9, 10 and 11; detector B alerts at 3, 9 and 12. Find the window hits, the row alerts and the outside-window alerts. Which has the higher event recall under this definition? Can you infer pointwise fault precision?" hint="Extra alerts inside one window do not create new events.">
      <Prose>Both hit 2 of 2 windows. A has six row alerts and none outside; B has three row alerts and one outside. Window-hit recall ties while the workload differs by a factor of two. Window annotations do not identify which individual rows were faulty, so pointwise fault precision is undetermined for either detector.</Prose>
    </Practice>
    <Practice title="J. Make a stuck sensor visible" question="A healthy sensor often reports near 40; a stuck sensor reports exactly 40 for three hours. A level-only detector gives ordinary scores throughout. Propose a causal feature and a comparison. Name a legitimate state that could resemble the episode." hint="The unusual property belongs to the sequence, not to any single value.">
      <Prose>Use trailing variability, or elapsed duration since the last change, computed from the current and earlier readings only. Compare it with reviewed durations under the same operating mode, taking measurement resolution into account. A controlled constant process, quantisation or a scheduled idle state can be equally flat. The feature makes the pattern visible; it does not establish its cause.</Prose>
    </Practice>

    <H2>{headings[12]}</H2>
    <Prose>You can now trace a score back to paths through random cuts, to ratios of local reachability densities, or to a kernel compatibility boundary. More importantly, you can keep those scores separate from thresholds, review workload and annotated events, and say out loud which period supplied which decision.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Integrate first-cut intervals and correct a truncated path', 'Section 3, isolation lab, practices A and B'],
      ['Say whose radius sets each floor and why a farther query can score lower', 'Section 4, neighbourhood lab, practice C'],
      ['Explain why a training factor and a new-query score differ at one coordinate', 'Section 5, fitting-mode lab, practice D'],
      ['Put a midpoint outside a region whose anchors are on its boundary', 'Sections 6 and 9, kernel lab, practices E and G'],
      ['Turn prevalence and two rates into a review queue and an honest precision', 'Section 8, review-queue lab, practice F'],
      ['Choose a threshold on real data and report its unmatched workload', 'Section 10, temperature lab, practices H and I']
    ]} />
    <Prose>The next topic, <a href="/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml">Gaussian Mixture Models &amp; the EM Algorithm</a>, introduces a learned probability density and soft component responsibilities. They answer a different question: a point far from every component can still prefer one component strongly relative to the others while receiving very little total density. Turning even a valid density into an alert still needs a reference population, a representation and a decision policy.</Prose>
    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://webcast.in2p3.fr/video/anomaly_detection_algorithms_in_scikitlearn">Nicolas Goix, “Anomaly detection algorithms in scikit-learn”, recorded talk</a> with its <a href="https://ngoix.github.io/nicolas_goix_osi_presentation.pdf">companion slides</a> — a short visual alternative for the fitting modes and random isolation, best after sections 2 to 5. The 15-slide companion was read and the recording page checked; the recording was not watched in full. It is a 2015 resource for intuition, so use current APIs rather than copying its historical code.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html">The scikit-learn LOF novelty example</a> — another worked frozen-reference and later-query view, useful immediately after section 5.</li>
    </ul></>}>
      <li><a href="https://arindam.cs.illinois.edu/papers/09/anomaly.pdf">Chandola, Banerjee and Kumar, “Anomaly Detection: A Survey”</a> — the canonical map. Begin with section 2 for data, anomaly types, labels and outputs; its classification, nearest-neighbour and clustering sections place these three mechanisms among the other families. The authoring review inspected the taxonomy and the relevant passages, not every derivation.</li>
      <li><a href="https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf">Liu, Ting and Zhou, “Isolation Forest”</a> — read tree construction and path normalisation in section 2, the swamping and masking discussion in section 3, then subsampling and the height limit in section 4.1. Its experiments motivate the choices rather than guaranteeing a universally optimal setting.</li>
      <li><a href="https://sigmodrecord.org/publications/sigmodRecord/0006/pdfs/LOF_%20Identifying%20Density-Based%20Local%20Outliers.pdf">Breunig, Kriegel, Ng and Sander, “LOF: Identifying Density-Based Local Outliers”</a> — sections 4 and 5 define reachability, tied neighbourhoods and the local bounds used in section 11. Compare their possibly larger tied neighbourhoods with our explicitly fixed-k calculation.</li>
      <li><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf">Schölkopf, Platt, Shawe-Taylor, Smola and Williamson, “Estimating the Support of a High-Dimensional Distribution”</a> — the primal and dual development and the nu-property proof explain the training-bound conditions. Best read after section 9 rather than as a first introduction to kernels.</li>
      <li><a href="https://scikit-learn.org/stable/modules/outlier_detection.html">The scikit-learn outlier and novelty guide</a> with the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html">IsolationForest</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html">OneClassSVM</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html">LocalOutlierFactor</a> API pages — method availability, score orientation and parameter behaviour, reviewed for release 1.9.1. Recheck the version before reproducing an API-dependent detail.</li>
      <li><a href="https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md">The NAB data descriptions</a>, the <a href="https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/labels/combined_windows.json">pinned annotation windows</a> at commit ea702d7 and the <a href="https://github.com/numenta/NAB/issues/376">duplicate-timestamp report</a> — the provenance and preprocessing behind section 10. The CSV, the window JSON and the MIT notice are supplied offline with this page.</li>
    </Sources>
    <Prose>The five positions, the two spaced groups, the two kernel anchors and the review-queue rates are constructed fixtures with declared values. The temperature results are calculations on the identified real dataset under the stated split, settings and quantiles. None of them is a benchmark, a fault label or a claim about any future dataset.</Prose>
  </div>
};

export default anomalyDetectionContent;
