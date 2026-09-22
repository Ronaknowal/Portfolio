import { Callout, H2, H3, Prose, Code } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  CountInformationLab, SubsetSearchLab, DonorPermutationLab, CoalitionReferenceLab, WineInferenceLab,
} from '../../components/lesson-labs/SelectionLabs.jsx';
import {
  QuestionRoutesFigure, XorSquareFigure, ArrivalOrderFigure, SelectionProcedureFigure, TreeExplanationFigure,
} from '../../components/lesson-labs/SelectionFigures.jsx';
import { selectionExamples } from '../selection-examples.js';
import {
  alternativeReference, candidates, explainedCases, fourFieldModel, inspectionPredictions,
  majorityBaseline, permutationRecords, provenance, selectedModel, split,
} from '../selection-data.js';
import {
  familywiseProbability, groupedUnanimity, informationFromCounts, searchFitCounts, sigmoid,
  subsetWorld, unanimityGame, xorWorld,
} from '../selection-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = (value, digits = 9) => String(Number(value.toFixed(digits))).replace('-', '−');
const bits = value => num(value, 6);

const noisyCopy = informationFromCounts([[3, 1], [1, 3]]);
const practiceTable = informationFromCounts([[2, 0], [0, 6]]);
const xor = xorWorld();
const xorSubsets = subsetWorld([0, 1, 1, 0]);
const forwardFive = searchFitCounts({ features: 5, keep: 2, folds: 3, method: 'forward' });
const rfeFive = searchFitCounts({ features: 5, keep: 2, method: 'rfe' });
const forwardSix = searchFitCounts({ features: 6, keep: 3, folds: 4, method: 'forward' });
const rfeSix = searchFitCounts({ features: 6, keep: 3, method: 'rfe' });
const threePlayer = groupedUnanimity([2, 1]);
const fourPlayer = groupedUnanimity([3, 1]);
const row104 = explainedCases[0];
const shapVersion = '0.52.0';

const headings = [
  '1. Ask which question you need to answer',
  '2. Information before fitting: what does one variable reveal?',
  '3. Select a subset with the learning procedure inside the boundary',
  '4. Permutation: disturb an input, keep the model fixed',
  '5. SHAP: allocate one prediction relative to a declared reference',
  '6. A complete observed-data workflow',
  '7. Deeper branches: choose the right information and explanation',
  '8. Practice: change the question and calculate its answer',
  '9. Readiness and what comes next',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section><Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample></section>;
}
function Practice({ title, question, hint, children }) {
  return <section className="fs-practice"><H3>{title}</H3><Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>Show the explained solution</summary>{children}</details></section>;
}

const featureSelectionContent = {
  title: 'Feature Selection & Importance (SHAP, Permutation, Mutual Info)',
  readTime: '~60 min first pass · ~115 min complete read + 60–100 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot fs-lesson">
    <LessonIntro prerequisites={<>Training versus validation, a prediction function and an average, from the preceding <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization</a> lesson and the earlier <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a>. Entropy and coalition notation are introduced here before they are used.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A laboratory measures thirteen properties of each sample and wants to stop paying for some of them. Learn to tell four different questions apart — what a measurement reveals in the data, what a fitted model relies on, what survives removal and refitting, and how one prediction is allocated against a reference — then calculate each of them by hand on tables you edit yourself, and finally run all four on 178 real chemical analyses with a protected reserve. Every investigation updates its topic-specific results from valid control changes, with no expected-answer input.
    </LessonIntro>
    <Prose className="fs-route"><strong>First pass.</strong> Read sections 1–6, work through the small count, permutation and coalition examples, then try practice 1–6. Section 6 connects them in a reproducible observed-data workflow. Section 7 explores deeper questions about dependence, search and stability; it is optional on a first visit.</Prose>

    <Prose>A laboratory measures thirteen properties of each sample. A classifier makes useful predictions, but running every assay takes time. Which measurements could the laboratory stop collecting? Now imagine a second request: explain why the classifier gave one particular sample a high score. These sound similar, yet they need different experiments.</Prose>
    <Prose>The preceding <a href="/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml">Regularization</a> lesson showed why different coefficient vectors can represent the same predictor. Here we ask more precisely what an input contributes: to the information in the data, to a fitted model&rsquo;s performance, or to one prediction relative to a reference. That precision makes an explanation useful rather than merely persuasive.</Prose>

    <H2>{headings[0]}</H2>
    <Prose><strong>Feature selection</strong> chooses which inputs a learning procedure will receive. Its goal might be comparable accuracy with fewer measurements, less memory, a simpler model, or a particular scientific investigation. It can help or hurt prediction; fewer columns is not automatically better. Nor does every model use every supplied column. A tree may never split on one of them.</Prose>
    <Prose><strong>Feature importance</strong> assigns a quantity to an input under a stated question. There is no single intrinsic importance number waiting inside a dataset.</Prose>
    <LessonTable caption="Five questions, and what each one holds fixed" headers={['Question', 'What stays fixed?', 'Suitable starting point']} rows={[
      ['Does this measurement tell us anything about the target on its own?', 'Observed distribution and variable definitions', 'Mutual information or an appropriate univariate statistic'],
      ['How much does this fitted model rely on the correctly paired measurement?', 'Model, assessment rows and performance metric', 'Permutation importance'],
      ['Could we train a useful model without this measurement?', 'Training/assessment procedure; refit the model', 'Subset comparison or a removal-and-refit experiment'],
      ['How is this prediction allocated relative to this reference?', 'Model, instance, output scale and missing-feature rule', 'Shapley/SHAP attribution'],
      ['Would changing this real-world quantity change the outcome?', 'A stated causal problem', 'Causal assumptions and evidence beyond these diagnostics'],
    ]} />
    <Prose>A large association may come from a measurement taken after the outcome becomes known. That input can score beautifully while being unavailable at prediction time. Establish availability and the unit of prediction before ranking anything. An identifier is not automatically forbidden: a known group identity can be useful in a suitable task. But memorizing unique row IDs will not teach a classifier how to handle unseen rows.</Prose>
    <Prose>One helpful distinction is <strong>selecting a useful subset</strong> versus <strong>discovering every relevant variable</strong>. Two instruments may measure the same quantity. A low-cost predictor may need only one; a scientific inventory might care about both. State which goal is intended before interpreting an excluded column as irrelevant. The <a href="https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf">Guyon–Elisseeff introduction</a> develops this distinction and the interaction examples behind it.</Prose>
    <QuestionRoutesFigure />

    <H2>{headings[1]}</H2>
    <H3>Start with counts rather than a formula</H3>
    <Prose>Suppose eight observations have binary measurement X and binary target Y:</Prose>
    <LessonTable caption="Eight observations, as a two-by-two table of counts" headers={['', 'Y=0', 'Y=1', 'Row total']} rows={[
      ['X=0', '3', '1', '4'],
      ['X=1', '1', '3', '4'],
      ['Column total', '4', '4', '8'],
    ]} />
    <Prose>Before observing X, the target is evenly split. After seeing X=0, Y=0 occurs three quarters of the time; after X=1, Y=1 does. The measurement has reduced our uncertainty without perfectly predicting the answer.</Prose>
    <Prose>For a discrete variable, <strong>entropy</strong> measures the average information required to identify its outcome:</Prose>
    <MathBlock>{'H(Y)=-\\sum_y p(y)\\log_2 p(y).'}</MathBlock>
    <Prose>A fair binary target has entropy one bit. A certain target has zero. An outcome that is rarer carries more information when it occurs, because <Math>{'-\\log_2 p'}</Math> is larger. We define a zero-probability contribution as zero by its limiting value; we do not take a literal logarithm of zero in code.</Prose>
    <Prose>Conditional entropy averages the remaining uncertainty after X is known. In the table, each row has probabilities 3/4 and 1/4, so</Prose>
    <MathBlock>{'\\begin{gathered}H(Y\\mid X)=-\\tfrac34\\log_2\\tfrac34-\\tfrac14\\log_2\\tfrac14\\\\[4pt]\\approx0.811278\\text{ bits}.\\end{gathered}'}</MathBlock>
    <Prose><strong>Mutual information</strong>, or MI, is the reduction:</Prose>
    <MathBlock>{'\\begin{gathered}I(X;Y)=H(Y)-H(Y\\mid X)\\\\[4pt]\\approx0.188722\\text{ bits}.\\end{gathered}'}</MathBlock>
    <Prose>An equivalent form compares the joint distribution with what independence would predict:</Prose>
    <MathBlock>{'\\begin{gathered}I(X;Y)=\\\\[4pt]\\sum_{x,y:\\,p(x,y)>0}p(x,y)\\log_2\\frac{p(x,y)}{p(x)p(y)}.\\end{gathered}'}</MathBlock>
    <Prose>For independent variables the numerator equals the denominator in every occupied cell, making the log ratio zero. More generally the weighted sum is a divergence and is nonnegative, even though individual occupied cells can contribute negative terms. Zero population MI characterizes independence. MI is symmetric in X and Y, has no positive/negative direction like correlation, and is not an accuracy percentage.</Prose>
    <CountInformationLab />
    <Prose>Here is a complete small calculation. Save as <Code>information_from_counts.py</Code>; it requires NumPy.</Prose>
    <Program example={selectionExamples.informationFromCounts}>
      <Prose>The corresponding totals are {bits(noisyCopy.mutualInformation)}, 1 and 0 bits. The zero cells are excluded from the logarithm rather than evaluated at it, and the guard clauses refuse a negative count or a zero total instead of returning a number that looks like an answer.</Prose>
    </Program>

    <H3>A pair can matter even when neither member matters alone</H3>
    <Prose>Consider four equally likely states:</Prose>
    <LessonTable caption="The XOR world: four equally likely states" headers={['A', 'B', 'Y: are the bits different?']} rows={[
      ['0', '0', '0'], ['0', '1', '1'], ['1', '0', '1'], ['1', '1', '0'],
    ]} />
    <Prose>This is XOR. Knowing A alone leaves the two target values equally likely. So does B. Therefore I(A;Y)=I(B;Y)={num(xor.projections[0].information, 6)}. Knowing the pair determines Y, so I((A,B);Y)={num(xor.jointInformation, 6)} bit. A ranking that discards every zero-MI individual input would discard both essential parts of this particular mechanism.</Prose>
    <Prose>This does not make all filters inherently univariate. A filter is independent of the final predictor; it can evaluate joint or conditional information. Univariate filters are a common, economical subclass. Their limitation is the question they ask, not a proof that all non-model criteria ignore interactions.</Prose>
    <Prose>The opposite issue is redundancy: two exact copies can each reveal the same one bit, while together still reveal only one. Summing their individual MI values double-counts that information. Slightly correlated measurements are subtler: they can still supply complementary signal or independent measurement noise. A correlation threshold alone does not prove one is safe to discard.</Prose>
    <XorSquareFigure />

    <H3>Estimation is not population knowledge</H3>
    <Prose>The count-table value is an empirical estimate when counts come from a sample. With eight unique ID categories and a balanced binary target, each observed category has one known label. The empirical MI is one bit even if new IDs carry no target information. That is memorization in the contingency table, not established predictive signal.</Prose>
    <Prose>Continuous variables need an estimator rather than a literal finite category per distinct value. Binning introduces a resolution choice: very fine bins can memorize, while coarse bins can hide relationships. Nearest-neighbor estimators use local distances instead; choosing neighborhood size trades resolution against estimator variability. Numeric storage does not decide the statistical type: category codes remain discrete, while a rounded measurement may be modeled as continuous if that matches the question and measurement process.</Prose>
    <Prose>The current <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html"><Code>mutual_info_classif</Code> contract</a> distinguishes discrete and continuous inputs, uses a seed for tiny tie-breaking perturbations, returns <strong>nats</strong>, and clips negative estimates to zero. A nat uses the natural logarithm; divide by ln 2 to express the same quantity in bits. A returned zero is not a certificate of population independence. Its mixed continuous/discrete estimation is not simply “run the same continuous KSG formula on integer class labels.”</Prose>
    <Prose>Other useful univariate statistics answer narrower questions. ANOVA&rsquo;s F statistic compares between-class mean variation with within-class variation, so equal class means can hide distributional differences. A contingency-table <Math>{'\\chi^2'}</Math> statistic compares observed categorical counts with independent expected counts. Scikit-learn&rsquo;s <Code>chi2</Code> selector is intended for nonnegative count/frequency-like feature inputs; merely rescaling arbitrary continuous measurements to [0,1] does not establish the inferential assumptions of a count test. A score used for ranking and a calibrated significance test are distinct uses.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>Three practical families organize the search:</Prose>
    <LessonTable caption="Filters, wrappers and embedded selection" headers={['Family', 'How it chooses', 'What to inspect']} rows={[
      ['Filter', 'Statistics of the training data, independently of the final fitted predictor', 'Joint versus univariate criterion, estimator resolution, redundancy'],
      ['Wrapper', 'Fit and assess candidate input subsets with a chosen learner', 'Search strategy, complete validation boundary, computation'],
      ['Embedded', 'Selection is part of fitting, such as an L1-penalized objective or tree split choices', 'Model assumptions, tuning and selection stability'],
    ]} />
    <Prose><strong>Forward selection</strong> starts from an empty subset and adds the candidate that produces the best chosen validation result. <strong>Backward selection</strong> starts with all available inputs and removes a candidate. These greedy procedures do not reconsider every past decision, so neither guarantees the globally best subset.</Prose>
    <Prose><strong>Recursive feature elimination</strong>, or RFE, fits a model, ranks its available inputs by a specified model quantity, removes the weakest and refits. Removing a column can change the ranking of the survivors. RFE&rsquo;s elimination criterion is not itself necessarily a validation score. RFECV adds a cross-validation procedure to select the retained size; its selected CV score still participated in selection. An outer assessment is needed to assess the whole recipe independently. These distinctions and the current selector APIs are documented in the <a href="https://scikit-learn.org/stable/modules/feature_selection.html">feature-selection guide</a>.</Prose>
    <Prose>The previous regularization lesson already derived why L1 can produce zeros. It does not identify only truly useless inputs, and support can change nonmonotonically along a correlated-data path. A selected subset reflects the objective, feature scale, sample and penalty. A tree&rsquo;s internal importance can also feed a selector, but the importance itself is a score; a threshold or size rule is what turns it into a retained subset.</Prose>

    <H3>See the search fail on a complete tiny world</H3>
    <Prose>Use the four equally likely XOR states. For each subset, let a lookup predictor output the most common target among states with those observed values, breaking ties toward zero. Empty, A-only and B-only subsets each achieve {num(xorSubsets.subsets[0].accuracy, 4)} accuracy; the pair achieves {num(xorSubsets.subsets[3].accuracy, 4)}. These are exact scores over a declared finite world, not held-out experimental estimates.</Prose>
    <Prose>A forward rule that stops unless accuracy strictly improves never leaves the empty set. A rule forced to retain two inputs reaches the pair, while backward removal from the perfect pair would reject either one-column reduction under the same strict-improvement requirement. The final size and stopping rule are part of the algorithm, not administrative details.</Prose>
    <SubsetSearchLab />
    <Prose>For a real dataset, the lookup table is replaced by a specified fitted learner and the score by suitable validation. A linear learner without interaction features cannot solve the XOR task just because a wrapper presents both columns. “Wrapper” is not a guarantee of interaction sensitivity.</Prose>

    <H3>Protect the selection boundary</H3>
    <Prose>If a training fold selects features by their relation to Y, its selector must see only that fold&rsquo;s training rows and labels. Selecting on the full dataset first allows validation labels to influence the representation. A pipeline inside the outer CV loop enforces the usual fit/transform boundary. However, wrapping an <strong>internally</strong> cross-validated selector after a globally fitted transform can still expose its inner validation rows to that transform. Put every learned operation inside the actual boundary being claimed.</Prose>
    <Prose>Selection size, thresholds, encodings, feature groups inferred from correlations, and any score-driven domain revision are all choices. If an inspection result causes another choice, those inspection rows become development information. The <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">previous CV lesson</a> explains nested assessment of the complete procedure. Leakage is a fact about information flow; it is not disproved by one run in which the corrected score happens to rise.</Prose>
    <Prose>Computation depends on the actual search. Forward selection from d=5 to k=2 assesses 5+4={forwardFive.subsets} subsets. With three folds, that is {forwardFive.candidateFits} fits before a final refit. RFE removing one at a time from five to two fits at sizes five, four and three to choose removals, then fits the retained two-feature model: {rfeFive.total} fits in that simple protocol. Neither count is “one fit per original feature” universally. Batched removal changes the candidates and cost; a percentage-step implementation&rsquo;s exact convention must be checked.</Prose>

    <H2>{headings[3]}</H2>
    <Prose>Suppose a fitted predictor has mean squared error 2 on the assessment rows. Copy those rows and shuffle one input column among them, leaving the target and other columns in their original rows. Its error is now 5. Under this shuffle, the error increase is three in squared-target units.</Prose>
    <Prose>Repeated random donor permutations estimate</Prose>
    <MathBlock>{'\\operatorname{PI}_j=\\frac1R\\sum_{r=1}^R(A_r-B).'}</MathBlock>
    <Prose>This is an estimate, averaged over <Math>{'R'}</Math> donor repeats, not a population quantity. Read the three symbols in order. <Math>{'\\pi_r'}</Math> is the donor row ordering for repeat <Math>{'r'}</Math>, and the hybrid row <Math>{'\\tilde x_i'}</Math> keeps every coordinate of row <Math>{'i'}</Math> except coordinate <Math>{'j'}</Math>, which arrives from donor row <Math>{'\\pi_r(i)'}</Math>. <Math>{'A_r'}</Math> is the mean loss over those hybrid rows, <Math>{'\\tfrac1n\\sum_i L(y_i,f(\\tilde x_i))'}</Math>, and <Math>{'B'}</Math> is the mean loss before any shuffle, <Math>{'\\tfrac1n\\sum_i L(y_i,f(x_i))'}</Math>. The fitted function <Math>{'f'}</Math> is unchanged. For a higher-is-better score such as accuracy, use original score minus shuffled score instead. A positive value then consistently means performance deteriorated after disturbing the input.</Prose>
    <Prose>Each shuffle preserves that column&rsquo;s empirical values but disturbs its pairing with targets <strong>and other inputs</strong>. It can leave some rows unchanged and does not make a finite sample perfectly independent. It may also create unusual combinations outside the observed joint distribution. This is an explicit perturbation experiment on a particular model, population sample and metric. It is not an automatically unbiased measure of a feature&rsquo;s causal effect or universal usefulness. The <a href="https://scikit-learn.org/stable/modules/permutation_importance.html">permutation guide</a> describes the fixed-model procedure and metric-dependent interpretation.</Prose>

    <H3>Redundancy does not make the fitted model refit itself</H3>
    <Prose>Let four rows have two identical sensor columns: both equal (−1,−1,1,1). The target is that same vector. Three predictors fit perfectly: <Math>{'f_1(x)=x_1'}</Math>, <Math>{'f_2(x)=x_2'}</Math> and <Math>{'f_3(x)=(x_1+x_2)/2'}</Math>. Use the declared donor ordering (2,3,0,1), which exchanges the signs.</Prose>
    <LessonTable caption="MSE increase under three fixed predictors and three perturbations" headers={['Fixed predictor', 'Shuffle sensor 1', 'Shuffle sensor 2', 'Shuffle both together']} rows={[
      ['First sensor only', '4', '0', '4'],
      ['Second sensor only', '0', '4', '4'],
      ['Average of sensors', '1', '1', '4'],
    ]} />
    <Prose>The first predictor cannot suddenly use sensor 2 after sensor 1 is corrupted; its equation never reads sensor 2. The average predictor already uses both, so corrupting one produces a different result. Removing sensor 1 <strong>and refitting</strong> would be a fourth experiment: a newly trained predictor could use sensor 2 alone. These are not conflicting answers to one question.</Prose>
    <Prose>For a logical feature group, apply the <strong>same donor ordering to all its columns</strong>. This preserves their within-group pairing while disturbing their relation to the target and remaining groups. Independent permutations inside the group would answer another question and could turn a valid one-hot category into an impossible vector. For a preprocessing pipeline, permuting a raw categorical field before encoding often makes the intended group explicit.</Prose>
    <DonorPermutationLab />
    <Prose>This table also illustrates model uncertainty. Several models can perform equally well and rely on different inputs. <a href="https://www.jmlr.org/papers/volume20/18-760/18-760.pdf">Fisher, Rudin and Dominici</a> formalize reliance across a specified set of well-performing models. Their model-class result is more than averaging one model&rsquo;s permutation repeats; the candidate class and allowed loss tolerance matter.</Prose>

    <H3>What the uncertainty bars do and do not show</H3>
    <Prose>Repeat-to-repeat variation reflects random donor choices with the <strong>same fitted model and assessment sample</strong>. It does not include uncertainty from new training samples, a different model, different assessment cases or a changed deployment population. More repeats reduce Monte Carlo error in that fixed experiment; they cannot repair the wrong perturbation or a weak assessment design.</Prose>
    <Prose>A negative measured importance means the shuffled version scored better on this assessment. Sampling variability, harmful fitted reliance or a metric-insensitive effect are possible explanations. Zero importance can arise because a feature is unused, because the metric did not change, or because the chosen perturbations did not alter relevant decisions. It does not alone prove independence from the target. Importance on training rows is permitted but describes training behavior; use suitable unseen assessment rows to investigate predictive behavior beyond fitting.</Prose>
    <Prose>Tree impurity importance asks something else. For a split node t, weighted impurity reduction is its sample fraction times the parent impurity minus the child-weighted impurities. Sum the reductions at splits using a feature, then apply the estimator&rsquo;s normalization. It records how that particular fitted tree partitioned its training criterion. Many candidate splits can favor chance reductions, so a high-cardinality measurement can receive excessive training importance. Held-out perturbation tests a different property; neither score should be disguised as the other by normalizing all bars to a common-looking scale.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>Permutation importance begins with performance across labeled cases. <strong>Shapley attribution</strong> begins with one output and asks how to allocate its difference from a reference among input features. SHAP applies this idea to model explanations. The allocation can be useful, but it is incomplete until we define what a prediction means when only some inputs are supplied.</Prose>

    <H3>First define the game</H3>
    <Prose>Suppose our model is <Math>{'f(a,b)=a+b+ab'}</Math> and the instance is (2,3), with prediction eleven. Use (0,0) as the declared reference. A coalition is simply a subset of features whose instance values are retained; replace the other values by the reference. Write <Math>{'v(S)'}</Math> for that coalition&rsquo;s output:</Prose>
    <LessonTable caption="Four coalitions, and the input each one actually evaluates" headers={['Retained instance features S', 'Input actually evaluated', 'v(S)']} rows={[
      ['None', '(0,0)', '0'],
      ['A', '(2,0)', '2'],
      ['B', '(0,3)', '3'],
      ['A and B', '(2,3)', '11'],
    ]} />
    <Prose>If A arrives first, it adds two; B then adds nine. If B arrives first, it adds three; A then adds eight. Average over the two possible arrival orders:</Prose>
    <MathBlock>{'\\begin{gathered}\\phi_A=(2+8)/2=5,\\\\[4pt]\\phi_B=(3+9)/2=6.\\end{gathered}'}</MathBlock>
    <Prose>They add to eleven, the difference from the reference. Each receives its standalone contribution plus half of the interaction six. The model itself remains nonlinear; the additive accounting is for this particular instance and reference.</Prose>
    <ArrivalOrderFigure />
    <Prose>For <Math>{'d'}</Math> features, every ordering is equally weighted. If a subset <Math>{'S'}</Math> arrives before feature <Math>{'j'}</Math>, there are <Math>{'|S|!'}</Math> ways to order its members and <Math>{'(d-|S|-1)!'}</Math> ways to order those after <Math>{'j'}</Math>. Dividing by <Math>{'d!'}</Math> gives</Prose>
    <MathBlock>{'\\begin{gathered}\\Delta_j(S)=v(S\\cup\\{j\\})-v(S),\\\\[4pt]w_s=\\frac{s!\\,(d-s-1)!}{d!},\\\\[4pt]\\phi_j=\\sum_{S\\subseteq F\\setminus\\{j\\}}w_{|S|}\\,\\Delta_j(S).\\end{gathered}'}</MathBlock>
    <Prose>Here <Math>{'\\Delta_j(S)'}</Math> is what feature <Math>{'j'}</Math> adds to the coalition <Math>{'S'}</Math>, and <Math>{'w_s'}</Math> is the share of orderings in which a subset of size <Math>{'s'}</Math> arrives first. The notation <Math>{'v'}</Math> matters: it is the <strong>specified coalition game</strong>, not an ordinary model <Math>{'f'}</Math> mysteriously accepting missing columns. A telescoping sum along each ordering gives <Math>{'v(F)-v(\\varnothing)'}</Math>; averaging preserves that total. Thus <Math>{'v(\\varnothing)+\\sum_j\\phi_j=v(F)'}</Math>, commonly called efficiency or local accuracy when <Math>{'v(F)=f(x)'}</Math>.</Prose>
    <Prose>The Shapley rule also treats players symmetrically when all their marginal contributions match, gives zero to a player that changes no coalition value, and is linear when two games are added. These properties characterize the allocation <strong>for a fixed game</strong>. They do not uniquely choose a background population, certify causal truth, or prove that this is the best explanation for every user. The <a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf">original SHAP paper</a> explicitly specifies the simplified-input mapping behind its uniqueness result.</Prose>

    <H3>Replace missing coordinates using actual reference rows</H3>
    <Prose>One zero reference is often not meaningful. Instead, take a declared background collection B. For every background row, retain the instance values in S and fill the remaining coordinates from that row. Average the model outputs:</Prose>
    <MathBlock>{'v_{x,B}(S)=\\frac1{|B|}\\sum_{b\\in B}f(x_S,b_{\\bar S}).'}</MathBlock>
    <Prose>Missing coordinates from the <strong>same donor row stay together</strong>. This preserves dependence among those missing coordinates while breaking dependence between them and the retained values. It is often called a background-replacement or interventional game. It need not be a real-world intervention on the target-generating system.</Prose>
    <Prose>For <Math>{'f(a,b)=a+b+ab'}</Math>, x=(2,3), and background rows (0,0),(1,1), the four coalition values become 1.5, 3.5, 5 and 11. The Shapley values are now four and 5.5, summing to 9.5 above the new baseline 1.5. <strong>The model and instance prediction did not change.</strong> The reference question changed. Nor is the average prediction necessarily the prediction at the average input: f(.5,.5)=1.25 differs from the average 1.5.</Prose>
    <CoalitionReferenceLab />
    <Prose>Save the complete following program as <Code>coalition_attribution.py</Code>. It computes every coalition once, caches its value, then allocates the differences. It is intentionally bounded to small <Math>{'d'}</Math>; it is the mechanism, not a replacement for scalable explainers.</Prose>
    <Program example={selectionExamples.coalitionAttribution}>
      <Prose>The empty subset is mask zero; A is bit zero and B is bit one. Neither a zero-valued feature nor a missing table entry automatically means an absent player. The guards refuse a ragged background, a mismatched width, a non-finite input and a predictor that does not return one finite scalar per row, rather than producing an allocation that looks complete.</Prose>
    </Program>

    <H3>Dependence changes which question the game answers</H3>
    <Prose>Another game is observational conditioning:</Prose>
    <MathBlock>{'v_x^{\\mathrm{cond}}(S)=\\mathbb{E}\\bigl[f(X)\\mid X_S=x_S\\bigr].'}</MathBlock>
    <Prose>It asks what output to expect after learning those observed values under a specified joint distribution. Compare this with taking missing values from unconditional reference rows. Conditional expectations can respect dependence, but estimating them accurately is a separate statistical problem; an exact Shapley sum cannot repair inaccurate conditional estimates.</Prose>
    <Prose>Let <Math>{'X_1=X_2'}</Math> be a fair binary variable and let <Math>{'f(x)=x_1'}</Math>. Explain x=(1,1). The baseline is 1/2. Under conditioning, knowing either coordinate reveals both, so v(&#123;1&#125;)=v(&#123;2&#125;)=1 and each feature receives 1/4. Under background replacement, learning <Math>{'X_2'}</Math> while replacing <Math>{'X_1'}</Math> still averages to 1/2; the attributions are 1/2 for <Math>{'X_1'}</Math> and zero for <Math>{'X_2'}</Math>. Both games are enumerated inside the investigation above.</Prose>
    <Prose>There is no violation of the dummy-player rule: <Math>{'X_2'}</Math> changes coalition values in the conditional game by revealing <Math>{'X_1'}</Math>. It does not change them in the replacement game. State whether the explanation concerns information revealed by observations or the model&rsquo;s response to replaced coordinates. <a href="https://martinjullum.com/publication/aas-2021-explaining/aas-2021-explaining.pdf">Aas, Jullum and Løland</a> develop dependent-feature conditional estimation and distinguish these targets.</Prose>
    <Prose><a href="https://proceedings.mlr.press/v108/janzing20a/janzing20a.pdf">Janzing, Minorics and Blöbaum</a>, section 3, analyze this same duplicate-input example from the model-input intervention perspective. Their distinction between the algorithm&rsquo;s output and the real-world outcome is crucial: intervention on the inputs of a known program does not establish the effect of changing the corresponding physical quantities in the world.</Prose>
    <Prose>Tree-path-dependent SHAP uses a tree&rsquo;s recorded path counts. It is <strong>not generally the true conditional expectation under the data&rsquo;s joint distribution</strong>. Interventional TreeSHAP uses an explicit background and a different game. Current <Code>TreeExplainer</Code> has an <Code>auto</Code> mode that chooses based on whether background data are supplied; specify the intended mode explicitly rather than relying on defaults. “Exact TreeSHAP” means exact for its defined algorithm/game assumptions, not exact causal discovery.</Prose>
    <Callout title="Name the class and the output units">
      For classifier explanations, name the class and output units. Raw outputs can be margins or log odds for some models; probabilities require a probability-scale game. If baseline plus attributions reconstructs a logit, applying the sigmoid to the <strong>total</strong> yields the probability. Applying it separately to each contribution does not produce additive probability effects. A mean-absolute attribution summarizes magnitude across selected explained rows; it has no label-performance term and therefore cannot tell you what fraction of accuracy survives feature removal.
    </Callout>

    <H2>{headings[5]}</H2>
    <H3>Question, data and boundaries</H3>
    <Prose>The <a href={provenance.doi}>UCI Wine dataset</a> contains {provenance.rows} chemical analyses from three cultivars grown in the same Italian region. It provides thirteen numeric measurements and a cultivar label. This is cultivar classification, not prediction of wine quality. The preserved source file has no missing entries. Its supplied names/table do not establish measurement units for all columns, so the visual labels retain source-scale values without inventing physical units. This page serves <a href={provenance.file} download>the unchanged data file</a> ({provenance.bytes.toLocaleString('en-US')} bytes, comma separated, SHA-256 <Code>{provenance.sha256}</Code>) and <a href={provenance.attribution}>its attribution</a> beside it. Nothing is downloaded when the page renders.</Prose>
    <Prose>The thirteen fields are alcohol, malic acid, ash, alkalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 and proline. The class column is the target and is excluded from the inputs. We retain the provider&rsquo;s source order and all rows.</Prose>
    <Prose>We ask whether a shallow classifier can use fewer measurements and then inspect a separately declared small model. The boundaries are:</Prose>
    <ul>
      <li>{split.development} development rows and {split.reserved} reserved rows, stratified split seed {split.splitSeed}. The {split.reserved} reserved rows receive no prediction or score here.</li>
      <li>Within development, {split.fitting} fitting rows and {split.inspection} inspection rows, stratified seed {split.innerSeed}.</li>
      <li>Within the {split.fitting} fitting rows, three stratified folds, seed {split.foldSeed}, compare MI-selected sizes {split.sizes.join(', ')}. The selector and depth-{split.treeDepth} tree fit separately in each fold. The tree has a minimum of {split.minSamplesLeaf} fitting cases per leaf.</li>
    </ul>
    <Prose>The inspection rows do not choose the subset size. Once inspected, they are part of what the author knows; use the still-reserved rows or appropriate new data if further changes are made. This small historical collection does not establish performance on a new region, vineyard, laboratory or measurement process. No corresponding grouping metadata supports such an assessment.</Prose>
    <Prose>For a transparent local explanation, a second model uses four raw fields declared before fitting: {fourFieldModel.labels.join(', ')}. Four inputs allow all sixteen coalitions to be inspected. It is not the MI-selected model, and its smaller input set is not chosen by the inspection score. Both models use the same tree settings and fitting rows.</Prose>

    <H3>Run the selection, then inspect the fixed model</H3>
    <Prose>Save this as <Code>wine_feature_study.py</Code> beside <Code>coalition_attribution.py</Code> and the provided <Code>wine.data</Code>. Setup: Python with NumPy and scikit-learn. The record below ran with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1.</Prose>
    <Program example={selectionExamples.wineStudy}>
      <Prose>All Wine measurements are treated as continuous inputs to this MI estimator, following the source&rsquo;s measurement description; the target is discrete. Tree splits do not require standardization in this example. This is a declared teaching estimator choice, not a universal rule for integer-valued measurements. The size tie rule prefers the first, smaller listed size. The record contains {split.fits.selectionCv} selection-CV fits, {split.fits.selectedRefit} selected refit and {split.fits.fourFieldRefit} four-feature refit: {split.fits.total} fits total.</Prose>
    </Program>

    <H3>Read the recorded outcomes without inventing a winner</H3>
    <LessonTable caption="The three evaluated retained sizes" headers={['MI-selected size', 'Correct counts in the three validation folds', 'Mean fold accuracy']} rows={candidates.map(entry => [
      String(entry.k),
      entry.folds.map(fold => `${fold.correct}/${fold.total}`).join(', '),
      entry.meanAccuracy.toFixed(6),
    ])} />
    <Prose>The rule selects {selectedModel.k}. Its final training-selected fields are {selectedModel.labels.join(', ')}. Across the three inner folds, the six retained identities are not identical. The column count is one hyperparameter; each fitted selector still learns a specific subset from its own rows.</Prose>
    <Prose>The selected model classifies {selectedModel.correct}/{selectedModel.total} inspection rows correctly, as does the independently declared four-field model. The training-majority baseline, always predicting class {majorityBaseline.class}, is correct on {majorityBaseline.correct}/{majorityBaseline.total}. These small-sample results demonstrate the protocol and a possible measurement reduction. They do not prove that these are the best six chemical assays, that the two models are equivalent, or that the tiny CV difference is statistically decisive.</Prose>
    <SelectionProcedureFigure />
    <Prose>For the fixed four-field model, the recorded inspection permutation results are:</Prose>
    <LessonTable caption="Two different quantities for the same four fields, in two different units" headers={['Field', 'Mean accuracy decrease', 'SD across twenty donor permutations', 'Tree impurity importance']} rows={permutationRecords.map(entry => [
      entry.label, entry.mean.toFixed(6), entry.sd.toFixed(6), entry.mdi.toFixed(6),
    ])} />
    <Prose>Accuracy decrease is a proportion: {permutationRecords[0].mean.toFixed(6)} corresponds to about {(100 * permutationRecords[0].mean).toFixed(1)} percentage points in this averaging scheme. The impurity column is normalized training-criterion reduction. These are different units and mechanisms; no common heatmap should pretend the numbers measure the same thing. The tree never splits on malic acid, so changing that coordinate leaves its prediction function unchanged. That is a verified property of this fitted tree, not a statement that malic acid has no association with cultivar.</Prose>

    <H3>Explain one actual prediction from actual coalition evaluations</H3>
    <Prose>Source row {row104.sourceId} (zero-based file index), the first inspection row, has ({fourFieldModel.labels.join(', ')})=({row104.input.map(value => num(value, 4)).join(',')}), and its actual cultivar is {row104.actualClass}. The four-feature tree predicts class {inspectionPredictions[0]}; its class-1 probability is {num(row104.prediction, 4)}. We explain the class-1 probability, using all {split.fitting} fitting rows as the background. Their average class-1 prediction is {num(row104.baseline, 4)}.</Prose>
    <LessonTable caption="The allocation, in probability units" headers={['Contribution', 'Probability units']} rows={[
      ['Baseline', `+${num(row104.baseline, 4)}`],
      ...fourFieldModel.labels.map((label, index) => [
        `${label.charAt(0).toUpperCase()}${label.slice(1)} attribution`,
        row104.phi[index] === 0 ? '0' : (row104.phi[index] > 0 ? `+${num(row104.phi[index], 4)}` : num(row104.phi[index], 4)),
      ]),
      ['Reconstructed class-1 probability', num(row104.prediction, 4)],
    ]} />
    <Prose>The negative attribution can have magnitude greater than the baseline because positive contributions offset part of it. An individual attribution is not a probability and need not lie between zero and one. The final probability does.</Prose>
    <Prose>The alcohol-only coalition has value {num(row104.coalitions[1], 4)}; the flavanoids-only coalition has value {num(row104.coalitions[4], 4)}, proline-only {num(row104.coalitions[8], 4)}, and flavanoids-plus-proline {num(row104.coalitions[12], 4)}. Every coalition retaining this instance&rsquo;s alcohol has value {num(row104.coalitions[1], 4)} in the saved tree. Averaging the marginal differences across all orderings yields the table, with floating-point reconstruction error about 1.1×10⁻¹⁶.</Prose>
    <TreeExplanationFigure />
    <WineInferenceLab />
    <Checkpoint prompt="Changing alcohol from 12.51 to 13.5 makes the saved tree output a class-1 probability of one, while changing malic acid from 1.73 to 4.1 changes nothing at all. Both are single-field edits of similar size. Why do they behave so differently?">
      <Prose>Alcohol is the field the root node splits on, and 13.5 falls on the other side of its threshold {String(fourFieldModel.tree.threshold[0])}, so the sample leaves the entire left subtree and reaches a different leaf. Malic acid is never used by any split in this tree, so no comparison anywhere reads it and no hybrid row&rsquo;s leaf can change. Both are reproducible in the investigation above through its two presets, and neither involves retraining.</Prose>
    </Checkpoint>
    <Prose>For a deliberate reference contrast, use the twelve fitting rows with the largest source indices. The original file is arranged by class, so this is a class-3 cohort, <strong>not a representative population sample</strong>. Its baseline class-1 prediction is {num(alternativeReference.baseline, 4)}. For the same row {row104.sourceId}, the contributions become approximately ({alternativeReference.phi.map(value => num(value, 6)).join(', ')}), still reconstructing {num(alternativeReference.prediction, 4)}. The statement has changed from comparison with the whole fitting reference to comparison with that cohort. Selecting reference rows is a substantive explanation decision, not an invisible speed trick. The investigation above loads that cohort from its reference-contrast preset.</Prose>
    <Prose>For a production tree explainer, the optional following program makes the same mode, background and output choice explicit. Save as <Code>check_tree_explanation.py</Code> beside the two prior programs. It additionally requires a compatible current <Code>shap</Code> installation.</Prose>
    <Program example={selectionExamples.treeExplainerCheck}>
      <Prose>The printed lines above come from the imported study, which this program re-runs; the program&rsquo;s own result is that its three assertions pass silently and a waterfall is drawn. Executed here with <Code>shap {shapVersion}</Code>, <Code>TreeExplainer</Code> in explicit interventional mode on the probability scale agreed with the exhaustive sixteen-coalition oracle for source row {row104.sourceId} to within 1×10⁻⁶ in both the baseline and all four attributions, and each of the twelve explained rows reconstructed its own predicted class-1 probability to within the same tolerance.</Prose>
      <Prose>Current <a href="https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html"><Code>TreeExplainer</Code> documentation</a> describes its dependence modes, model-dependent raw outputs and multi-output shapes. For a decision-tree classifier the raw output already <em>is</em> the class probability, so <Code>model_output="probability"</Code> changes nothing here; for a gradient-boosted or margin-scale model it changes the game being solved. Naming it is what makes the two cases distinguishable. Explicitly selecting the class avoids confusing a samples×features×classes result with a two-dimensional attribution matrix. A reconstruction check is necessary numerical evidence; by itself it cannot certify that the background or explanatory question is appropriate.</Prose>
    </Program>

    <H2>{headings[6]}</H2>
    <H3>Conditional information and measurement budgets</H3>
    <Prose>Once a subset S is available, the additional information in candidate <Math>{'X_j'}</Math> is</Prose>
    <MathBlock>{'\\begin{gathered}I(X_j;Y\\mid X_S)=\\\\[4pt]H(Y\\mid X_S)-H(Y\\mid X_S,X_j).\\end{gathered}'}</MathBlock>
    <Prose>In XOR, I(B;Y)={num(xor.projections[1].information, 4)} but I(B;Y|A)=1 bit. For an exact copy B=A, conditioning on A makes B contribute no additional information. This gives a principled target for complementarity, but estimating high-dimensional conditional information from few rows is difficult. A formula does not eliminate sample requirements or measurement error.</Prose>
    <Prose>A Markov blanket B for target Y is a set that makes Y conditionally independent of the remaining inputs given B, under the stated distribution. That is stronger than selecting the largest marginal MI values or dropping highly correlated pairs. The later <a href="/learn/path/full-curriculum/bayesian-networks-causal-graphical-models?module=classical-ml">Bayesian Networks &amp; Causal Graphical Models</a> gives graph semantics and the assumptions connecting graph structure to such conditional relationships.</Prose>
    <Prose>For a measurement budget, a subset can be assessed by prediction loss and acquisition cost together. A hypothetical pair of sensors might cost 2 and 8 units; an equally accurate substitute costing 3 could be useful even if a global importance bar is smaller. Cost may attach to a group: once an assay panel is run, several outputs may have almost no extra acquisition cost. Keeping one feature from every panel may therefore save fewer resources than keeping a larger number from one cheap panel. These are declared engineering costs, not costs inferred from attribution magnitudes.</Prose>
    <Prose>Unsupervised selection has a different target again. Removing a constant training column is often useful; removing every low-variance column can discard a rare alarm that matters greatly for a high-cost event. Without target labels, use a declared objective such as reconstruction, reliable measurement or clustering quality. Earlier PCA and NMF construct new coordinates from inputs rather than simply selecting old columns. They may reduce dimension without reducing the number of raw sensors that must be collected.</Prose>

    <H3>Stability, significance and repeated selection</H3>
    <Prose>A selector can be unstable while prediction is stable. The duplicate-sensor example makes that possible without any estimation noise. Across resampled training sets, record which inputs are selected and how their predictions compare on appropriate assessment cases. A selection frequency is a descriptive property of that resampling-and-fitting procedure, not automatically the probability that the feature is truly relevant. Formal stability-selection error bounds need additional assumptions and a specified algorithm.</Prose>
    <Prose>Significance asks how unusual a statistic would be under a specified null model. An importance score alone is not a p-value. If one runs one hundred independent valid null tests at level 0.05, the probability of at least one false rejection is 1−0.95¹⁰⁰≈{num(familywiseProbability(100, 0.05), 4)}. Dependence changes that arithmetic; selection after seeing the results introduces further issues. False-discovery or family-wise procedures can be appropriate for valid input p-values, but they do not turn an arbitrary MI estimate into a significance test.</Prose>
    <Prose>Random probe features can reveal suspicious overfitting behavior; consistently outranking a few such probes does not by itself prove relevance or guarantee an error rate. Do not remove inconvenient observations solely because they weaken the desired importance story. Investigate measurement and label quality using a documented rule. Selecting rows as well as columns is another part of the learning procedure and its assessment boundary.</Prose>
    <Prose>A beeswarm spread over many explained cases measures variation of attribution <strong>across those cases</strong>. It is not a confidence interval over new training fits. A plot of importance across CV fits, a plot across permutation repeats and a plot across individual predictions visualize different distributions. Name which one is shown. The next Bias–Variance lesson develops why changing training data can change a fitted function even when its average performance is similar.</Prose>

    <H3>Grouping players changes the attribution game</H3>
    <Prose>Grouping columns for a valid perturbation is often useful, but “add their individual Shapley values” and “treat the group as one player” can differ. Consider three players A,B,C with value one only when all three are present, and zero for every other coalition. Each individual Shapley value is {num(threePlayer.individual[0], 6)}, so A+B sum to {num(threePlayer.summedIndividual[0], 6)}. Now form two players: group G=&#123;A,B&#125;, and C. Both are needed for the value one, so each grouped player receives {num(threePlayer.grouped[0], 6)}. The possible arrival orders changed.</Prose>
    <Prose>Use summed individual values when that is the declared aggregation, and grouped-game values when groups are the explanatory players. A valid one-hot feature can be treated as one original variable, avoiding impossible partial-category coalitions. Feature engineering can create similar choices: explaining alcohol and alcohol² as separate players is different from explaining the single raw alcohol measurement that generates both. The decision should follow what the learner or application regards as a meaningful change.</Prose>

    <H3>Scale computation to the mechanism</H3>
    <Prose>For <Math>{'d'}</Math> raw inputs, exhaustive coalition enumeration requires <Math>{'2^d'}</Math> coalition values per explained instance; background replacement additionally evaluates <Math>{'|B|'}</Math> hybrid rows per coalition. Cached coalition values avoid recomputing the same subset for every feature. The local Wine calculation uses sixteen coalitions and {split.fitting} background rows per instance, not an exponential native search over all thirteen source columns.</Prose>
    <Prose>Sampling arrival orders estimates Shapley values using their average marginal increments; sampling coalitions with the Shapley kernel instead yields a weighted regression problem. These are related approximations, not identical algorithms. KernelSHAP uses special coalition-size weights and constraints at empty/full coalitions. Its cost depends on the evaluated coalitions, background size, model prediction cost and regression solve; there is no universal O(d²) end-to-end guarantee or fixed number of seconds.</Prose>
    <Prose>Tree-specific algorithms exploit shared paths. The <a href="https://arxiv.org/pdf/1802.03888">original TreeSHAP algorithm</a> gives an O(TLD²) bound per explained instance for T trees, at most L leaves and depth D in that algorithm; background-dependent variants and implementations add their own costs. That expression is not a benchmark for a million rows or a guarantee for every dependence model. Inspect the actual supported algorithm, approximation setting and output scale, then measure its cost on the intended workload. Never run large background explanations on every page render.</Prose>
    <Prose>For differentiable models, gradients measure local sensitivity, while integrated gradients accumulates gradients along a specified path from a reference. Deep SHAP and gradient-based approximations make additional choices. They are useful extensions, but sharing an additive-looking plot does not make all methods exact Shapley estimators. Nor are attention weights automatically causal or faithful explanations. A later interpretability investigation should define its intervention, reference and evaluation just as carefully as this tabular lesson.</Prose>
    <Prose>Finally, a global sum of mean absolute SHAP values is an attribution-magnitude total over a declared set of predictions. If you plot a cumulative <strong>fraction</strong> of that total, it must finish at one when the total is positive. If every attribution is zero, the fraction is undefined and should be labeled accordingly. Even a correct 95% cumulative fraction does not imply 95% retained accuracy, explained variance or information. For a measurement-removal decision, refit and assess the reduced-input recipe.</Prose>

    <H2>{headings[7]}</H2>
    <Prose>Attempt the first six before opening solutions. The remaining questions use the deeper branches.</Prose>

    <Practice title="1. A different count table"
      question="The counts are [[2,0],[0,6]]. What are H(Y), H(Y|X) and I(X;Y) in bits? Why is perfect prediction now less than one bit of information?"
      hint="The target probabilities are 1/4 and 3/4. Conditional on either occupied row, the answer is certain.">
      <Prose>H(Y) = −(1/4) log₂(1/4) − (3/4) log₂(3/4) ≈ {bits(practiceTable.targetEntropy)} bits. Conditional entropy is {num(practiceTable.conditionalEntropy, 6)}, so MI is {bits(practiceTable.mutualInformation)} bits. Perfectly revealing a target cannot reveal more uncertainty than it originally had; this target is not balanced. The first investigation loads this exact table from its practice preset.</Prose>
    </Practice>

    <Practice title="2. A feature that becomes useful later"
      question="For fair independent A and B with Y=A XOR B, give I(A;Y) and I(B;Y). After A is known, how many additional bits about Y does B reveal? Would adding a perfect duplicate C=A create a second independent bit about Y?">
      <Prose>The first two values are {num(xor.projections[0].information, 4)} and {num(xor.projections[1].information, 4)}, and the conditional value is one bit. C supplies no information beyond A because it is determined by A. A and B together determine Y; copying A does not create new target information.</Prose>
    </Practice>

    <Practice title="3. A partial permutation"
      question={<>Use the duplicate rows (−1,−1),(−1,−1),(1,1),(1,1), target (−1,−1,1,1), and predictor <Math>{'(x_1+x_2)/2'}</Math>. Shuffle only the first column using donors (0,2,1,3). What is the MSE increase? What happens if both columns use that same permutation?</>}>
      <Prose>The first-column perturbation gives predictions (−1,0,0,1), with squared errors (0,1,1,0), so MSE rises by 0.5. Perturbing both gives (−1,1,−1,1), squared errors (0,4,4,0), and increase two. Some donor rows preserve values; a shuffle need not alter every case. The donor investigation loads this exact case from its practice preset.</Prose>
    </Practice>

    <Practice title="4. Change the interaction strength"
      question={<>For <Math>{'f(a,b)=a+b+2ab'}</Math>, instance (1,2), and reference (0,0), calculate all four coalition values and the Shapley values. Then explain why the A-first increment differs from the A-second increment.</>}>
      <Prose>Coalition values are 0,1,2,7. A receives (1+5)/2=3 and B receives (2+6)/2=4. The interaction term contributes four only after both variables are present; averaging arrival orders allocates two of that interaction to each. The coalition investigation loads this case from its practice preset.</Prose>
    </Practice>

    <Practice title="5. Information flow rather than a score test"
      question="A colleague selects the six highest-MI inputs using every development label, then cross-validates a model on those columns. After moving the selector inside the folds, the score unexpectedly increases slightly. Was the original procedure free of leakage?">
      <Prose>No. Validation labels influenced the original selected representation regardless of the observed score direction. The corrected experiment changes the information boundary; one realized difference is not a universal test for whether leakage occurred. Preserve an independent assessment of the full selection procedure.</Prose>
    </Practice>

    <Practice title="6. Read the actual Wine result"
      question="Malic acid has zero permutation importance and zero replacement SHAP in the four-field tree. What precisely has been established? A proposed reduced model is judged only by retaining 95% of mean absolute SHAP. What calculation is missing?">
      <Prose>The saved tree never uses malic acid, so replacing that coordinate cannot change its output. This does not establish population independence or that no other model can use it. To judge a reduced measurement set, fit the reduced-input learning procedure and assess it on appropriate protected data; attribution magnitude is not a retained-performance guarantee.</Prose>
    </Practice>

    <Practice title="7. Explain a different output — deeper"
      question="A binary model's raw margin explanation has baseline −0.2 and contributions +0.8 and −0.1. What is the reconstructed margin and its probability under the sigmoid? Can the two contributions be separately passed through the sigmoid and added?">
      <Prose>The margin is 0.5 and sigmoid(0.5)=1/(1+exp(−0.5))≈{num(sigmoid(0.5), 6)}. The sigmoid is nonlinear, so transforming and adding the individual terms does not yield an additive probability explanation. A probability-space coalition game must explain that output directly.</Prose>
    </Practice>

    <Practice title="8. Count the search fits — deeper"
      question="Forward selection chooses three of six inputs using fourfold validation. Count candidate fits and one final refit. Compare simple RFE removing one at a time from six to three, including its final retained-model fit and no CV.">
      <Prose>Forward selection assesses 6+5+4={forwardSix.subsets} candidate subsets, with four fits each: {forwardSix.candidateFits} plus one final refit, {forwardSix.total} in all. RFE fits six-, five- and four-feature models to select removals, then the final three-feature model: {rfeSix.total} fits. These methods use different selection quantities and do not have identical statistical guarantees.</Prose>
    </Practice>

    <Practice title="9. Change the players — deeper"
      question="Four players A,B,C,D receive value one only when all four are present. Compare the sum of individual A+B+C Shapley values with the value assigned when G={A,B,C} is one player and D the other.">
      <Prose>The individual game is symmetric, so each receives {num(unanimityGame(4).phi[0], 4)} and the sum is {num(fourPlayer.summedIndividual[0], 4)}. The two-player game is symmetric, so G receives {num(fourPlayer.grouped[0], 4)}. Grouping changes possible arrival orders and the allocation target; it is not generally the same as summing afterward.</Prose>
    </Practice>

    <Practice title="10. Design a useful deployment investigation — deeper"
      question="A factory wants to replace an expensive sensor with two cheaper correlated sensors. A fitted model gives the expensive sensor the largest permutation score. Propose a comparison that answers the factory's question, including the unit of assessment and actual costs.">
      <Prose>Define the future task, such as predicting faults on new machines or later operating periods, and split by that unit and information availability. Compare predeclared full and cheaper-sensor learning procedures, fitting every transform/selection stage within their development folds. Assess the locked procedures on protected machines or later periods, using relevant error costs and actual acquisition/maintenance costs. A fixed-model permutation tests present reliance; it does not evaluate the retrained substitute. Preserve calibration/threshold assessment if decisions depend on predicted probabilities.</Prose>
    </Practice>

    <H2>{headings[8]}</H2>
    <Prose>You are ready to continue when you can distinguish data information, fixed-model reliance, removal-and-refit performance and local attribution; calculate MI from a small table; trace a permutation donor; derive a two-feature Shapley allocation; and state the reference, output units and protected data boundary in an actual analysis. You should be able to explain why an excluded feature can still be informative and why an exact explanation can still answer the wrong question.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Say which of the five questions a request is actually asking', 'Section 1, figure 1'],
      ['Compute entropy, conditional entropy and MI from a count table, in bits', 'Section 2, the count investigation, practice 1'],
      ['Explain why two zero-MI inputs can jointly determine the target', 'Section 2, figure 2, practice 2'],
      ['Run a greedy search over a finite world and say why its stopping rule matters', 'Section 3, the lattice investigation'],
      ['Trace a donor ordering through a fixed predictor and read a zero correctly', 'Section 4, the donor investigation, practice 3'],
      ['Build coalition values from actual hybrid rows and allocate them exactly', 'Section 5, figure 3, the coalition investigation, practice 4'],
      ['State the reference, the class and the output units of an explanation', 'Section 5 and section 6, figure 5, the wine investigation'],
      ['Read the recorded Wine study without inventing a winner', 'Section 6, figure 4, practices 5 and 6'],
      ['Separate conditional information, stability, grouping and computation cost', 'Section 7, practices 7 to 10'],
    ]} />
    <Prose>Next is <a href="/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml">Bias–Variance Tradeoff &amp; Learning Curves</a>. It develops what changes when training samples change, why prediction and selection stability can differ, and how to read learning curves. After that, Imbalanced Learning connects the metric and measurement choices here to rare events and unequal error costs.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html">Permutation with correlated features: a visual and code example</a>: inspected tree/permutation plots, a correlation dendrogram and a removal-and-refit comparison. Its correlation grouping is exploratory and uses the full X; for a protected assessment, fit grouping within the development boundary and choose a linkage appropriate to the distance. Its particular result is not a universal fallback rule for every fixed model.</li>
      <li><a href="https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html">SHAP waterfall notebook</a>: a visual and code route for reading a single prediction decomposition, its background and log-odds units, then comparing individual cases. The inspected notebook also discusses why a striking pattern needs further investigation rather than instant causal interpretation.</li>
    </ul></>}>
      <li><a href="https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf">Guyon and Elisseeff, An Introduction to Variable and Feature Selection</a> — section 2 covers univariate ranking, section 3 gives the small geometrical examples of individual versus joint usefulness — 3.3 is the XOR case — and section 4 covers subset search. Sections 5–7 connect construction, validation, stability and scientific discovery. Read the geometric examples alongside our exact XOR world; historical implementation recommendations need their assumptions checked.</li>
      <li><a href="https://scikit-learn.org/stable/modules/feature_selection.html">Scikit-learn feature-selection guide</a> — the current map of variance filters, univariate selectors, RFE/RFECV, embedded selectors and sequential search. Useful when translating a declared selection question into an API.</li>
      <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html">Mutual information for classification</a> — estimator inputs, discrete/continuous flags, reproducibility and nat units. It is an estimator contract, not a promise to reveal every interaction.</li>
      <li><a href="https://scikit-learn.org/stable/modules/permutation_importance.html">Permutation feature importance</a> — the fixed-model algorithm and the role of the metric and assessment data.</li>
      <li><a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf">Lundberg and Lee, A Unified Approach to Interpreting Model Predictions</a> — sections 2–4 define the additive explanation, assumptions and coalition weighting. Our two-order calculation is preparation for its kernel-regression formulation.</li>
      <li><a href="https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html">TreeExplainer API</a> — explicitly choose dependence mode, background and output scale; inspect multi-output shapes before plotting.</li>
      <li><a href="https://arxiv.org/pdf/1802.03888">Lundberg, Erion and Lee, Consistent Individualized Feature Attribution for Tree Ensembles</a> — section 3 explains shared-path computation and the stated TreeSHAP complexity; section 4 extends the game to interaction allocations. These algorithmic bounds are more informative than an unsupported timing comparison.</li>
      <li><a href="https://martinjullum.com/publication/aas-2021-explaining/aas-2021-explaining.pdf">Aas, Jullum and Løland, Explaining Individual Predictions When Features Are Dependent</a> — the conditional-game construction and the separate problem of estimating missing-feature distributions.</li>
      <li><a href="https://proceedings.mlr.press/v108/janzing20a/janzing20a.pdf">Janzing, Minorics and Blöbaum, Feature Relevance Quantification in Explainable AI</a> — section 3 distinguishes an intervention on a program&rsquo;s input from a causal claim about the world and explains the duplicate-input example.</li>
      <li><a href="https://www.jmlr.org/papers/volume20/18-760/18-760.pdf">Fisher, Rudin and Dominici, All Models are Wrong, but Many are Useful</a> — sections 2–4 expand the question from one fitted model to a declared collection of well-performing models. Its section 4 defines model class reliance, the concept the section above borrows. Their formal reliance ratios and bounds have assumptions beyond our exact zero-loss demonstration.</li>
      <li><a href={provenance.doi}>UCI Wine dataset</a>, {provenance.creator}, licensed <a href={provenance.licenseUrl}>{provenance.license}</a> — original description, attribution and license. This page serves the unchanged numeric member, SHA-256 <Code>{provenance.sha256}</Code>, with the exact split, models and attribution records for offline reproduction.</li>
    </Sources>
    <Prose>The count tables, the XOR world, the duplicate-sensor rows and the polynomial games are explicitly <strong>constructed calculations</strong>, not observed measurements. The Wine results are calculations on the identified real dataset under one declared stratified split, {split.fits.total} fits and one predeclared four-field model, with no reserved row predicted or scored. None of them is a benchmark or a claim about any future dataset, and none of them identifies a cause.</Prose>
  </div>,
};

export default featureSelectionContent;
