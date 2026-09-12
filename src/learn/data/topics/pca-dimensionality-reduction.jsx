import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { PcaProjectionLab, PcaMetricLab, PcaBudgetLab, PcaTaskInformationLab } from '../../components/lesson-labs/PcaLabs.jsx';
import { ProjectionShadowFigure, ConservationFigure, TransformShapesFigure, WineOverviewFigure, BiplotReadingFigure, GaussianSpectrumFigure, ResidualAlarmFigure } from '../../components/lesson-labs/PcaFigures.jsx';
import { pcaExamples } from '../pca-examples.js';

const headings = [
  '1. Can we keep fewer numbers without losing the pattern?',
  '2. Rotate the ruler, then keep its readings',
  '3. Why maximum spread and minimum loss give the same answer',
  '4. Fit once, project, and reconstruct in Python',
  '5. Scaling changes which differences matter',
  '6. Choose the number of components for a stated purpose',
  '7. Read components, scores and plots without mixing them up',
  '8. Connect PCA to clustering and prediction',
  '9. Practice: calculate, diagnose and transfer',
  '10. Deeper branches',
  '11. What you should now be able to do'
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section><Prose><strong>Before running:</strong> {example.question}</Prose><RunnableExample example={example}>{children}</RunnableExample></section>;
}
function Practice({ title, question, hint, children }) {
  return <section className="pca-practice"><H3>{title}</H3><Prose>{question}</Prose>{hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}<details><summary>Show the explained solution</summary>{children}</details></section>;
}

const pcaContent = {
  title: 'PCA & Dimensionality Reduction',
  readTime: '~50 min first pass · ~95 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot pca-lesson">
    <LessonIntro prerequisites={<>You need averages, squared distances and basic array operations. New notation is introduced as it is used. Review <a href="/learn/path/full-curriculum/k-means-hierarchical-clustering?module=classical-ml">K-Means &amp; Hierarchical Clustering</a> for the idea that a representation is part of the question, and <a href="/learn/path/full-curriculum/numpy-arrays-broadcasting-vectorization?module=programming-scientific-computing">NumPy arrays and broadcasting</a> if <Code>@</Code> or array shapes are unfamiliar.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      Learn to replace many measurements with a few well-chosen coordinates, recover approximate measurements from them, and decide whether what was lost matters for your purpose. You will rotate a ruler through four points by hand, then fit 178 real wines, choose a component count against a budget you set, and see exactly when a high explained-variance percentage still throws away the thing you cared about.
    </LessonIntro>
    <Prose className="pca-route"><strong>First-pass route.</strong> Read sections 1 through 6, run the four labs and the first four programs as you meet them, then try practice 1 to 4 in section 9 and the readiness check in section 11. That takes you from a picture to a working analysis in about 50 minutes plus code time. Section 7 develops interpretation, section 8 connects PCA to clustering and prediction, and section 10 holds the deeper mathematics, computation and applications. Return to those when their questions become relevant.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>Imagine receiving a spreadsheet with 178 wines and 13 chemical measurements for each wine. You want to compare the samples. Thirteen separate columns are easy to store but hard to see together: a scatterplot has only two axes. Which two numbers should represent each wine?</Prose>
    <Prose>One option is to pick two existing measurements. Another is to make two new measurements by combining the original thirteen. If several chemical quantities rise and fall together, a shared combination could describe their variation more efficiently than separate columns.</Prose>
    <Prose><strong>Principal component analysis, or PCA, finds such combinations.</strong> It orders them by how much variation they capture in the data you give it. You can keep all the combinations as a new coordinate system, or keep only the first few to obtain a smaller representation. Keeping fewer coordinates is the dimensionality reduction step.</Prose>
    <Prose>The aim here is concrete: make a compact view, recover approximate measurements from it, and decide whether the information lost is acceptable for the intended use.</Prose>
    <Prose>Our real example is the <a href="https://archive.ics.uci.edu/dataset/109/wine">UCI Wine dataset</a>: chemical analyses from three cultivars grown in the same Italian region, supplied for studying wine origin. A row is one wine; the 13 numeric inputs include alcohol, malic acid, color intensity and proline. The cultivar label is separate. It will help us inspect the result, but PCA will receive only the measurements. This page offers <a href="/learn-assets/pca/wine.csv" download>the same 178 rows as a CSV</a>; attribution and the file's construction are in the references.</Prose>
    <Prose>First we will use four invented points whose arithmetic fits on a page. Then we will return to the wines and answer two different questions: what makes a useful two-dimensional picture, and what makes an acceptable compressed measurement record?</Prose>

    <H2>{headings[1]}</H2>
    <Prose>Suppose two sensors describe the position of a moving marker. Their readings have the same unit. We observe:</Prose>
    <LessonTable caption="Four observations with two readings each" headers={['observation', 'first reading', 'second reading']} rows={[['A', 1, 1], ['B', 2, 0], ['C', 4, 4], ['D', 5, 3]]} />
    <Prose>The points form a narrow diagonal cloud. If we could record only one number per observation, a ruler laid along that diagonal would distinguish the lower-left observations from the upper-right observations. A ruler laid across the cloud would mostly measure its small thickness.</Prose>
    <ProjectionShadowFigure />
    <H3>Put the origin in the middle of the cloud</H3>
    <Prose>The average reading is (3, 2). Subtract it from every point:</Prose>
    <LessonTable caption="The same observations after centering" headers={['observation', 'centered first reading', 'centered second reading']} rows={[['A', '−2', '−1'], ['B', '−1', '−2'], ['C', '1', '2'], ['D', '2', '1']]} />
    <Prose>This operation is <strong>centering</strong>. It changes where zero is, while preserving all distances between points. We now describe differences from the average observation.</Prose>
    <Prose>The fitted line passes through the mean in the original picture and through zero in the centered picture. Without centering, an SVD of the raw readings solves a different problem: fitting directions through the original zero. A large offset can then influence the direction. It need not dominate every dataset, but it is no longer the same centered PCA calculation.</Prose>
    <H3>A direction tells us where the ruler points</H3>
    <Prose>A direction is a vector. Use</Prose>
    <MathBlock>{'v_1=\\frac{1}{\\sqrt 2}(1,1).'}</MathBlock>
    <Prose>Both entries are approximately 0.7071. Dividing by √2 makes its length one. This <strong>unit-length</strong> convention matters: otherwise doubling the numbers in the direction would double all ruler readings without improving the direction.</Prose>
    <Prose>For a centered point a = (a₁, a₂), its reading along the ruler is the <strong>dot product</strong>:</Prose>
    <MathBlock>{'z=a\\cdot v_1=a_1v_{11}+a_2v_{12}.'}</MathBlock>
    <Prose>This number is its <strong>score</strong> on the first principal component. A direction belongs to the whole fitted model; a score belongs to one observation.</Prose>
    <Prose>For A, the score is −2/√2 − 1/√2 = −3/√2 ≈ −2.1213. For D, it is 3/√2 ≈ 2.1213. A negative score means “on the other side of the mean along the chosen direction,” not a negative concentration or an invalid measurement.</Prose>
    <H3>Reconstruct an observation from its score</H3>
    <Prose>Multiply the score by the direction to return to a point on the line. Then add the mean to return to the original coordinates:</Prose>
    <MathBlock>{'\\widehat{x}=\\mu+zv_1.'}</MathBlock>
    <Prose>For A:</Prose>
    <MathBlock>{'\\begin{gathered}(3,2)+\\frac{-3}{\\sqrt2}\\,\\frac{(1,1)}{\\sqrt2}\\\\ =(3,2)+(-1.5,-1.5)\\\\ =(1.5,0.5).\\end{gathered}'}</MathBlock>
    <Prose>The reconstructed A is close to (1, 1), but it is not identical. The residual, the original minus its reconstruction, is (−0.5, 0.5).</Prose>
    <LessonTable caption="One score per observation, its reconstruction and the squared distance lost" headers={['observation', 'one score', 'reconstruction', 'squared distance lost']} rows={[['A', '−2.1213', '(1.5, 0.5)', '0.5'], ['B', '−2.1213', '(1.5, 0.5)', '0.5'], ['C', '2.1213', '(4.5, 3.5)', '0.5'], ['D', '2.1213', '(4.5, 3.5)', '0.5']]} />
    <Prose>A and B now have the same representation. Their difference lay across the ruler, in the direction we discarded. Compression has a visible meaning: two distinct observations can become indistinguishable.</Prose>
    <Checkpoint prompt="If you reverse the ruler, making the direction −v₁, what happens to A's score and reconstruction?">
      <Prose>The score changes sign. The direction also changes sign, so their product does not: (−z)(−v₁) = zv₁. The reconstruction is still (1.5, 0.5). This is why a sign-flipped component describes exactly the same model.</Prose>
    </Checkpoint>
    <H3>Investigation: find the most useful ruler</H3>
    <Prose>Start with the four observations. Record whether you expect turning the horizontal ruler toward the diagonal to decrease, increase or preserve the total squared reconstruction error. Rotate it, compare the result with your prediction, and inspect the perpendicular residual segments.</Prose>
    <PcaProjectionLab />
    <Prose>Then change one observation yourself. Does the best direction move toward it? Finally, translate all four observations by the same amount and refit. The mean moves; the centered geometry, the variances and every loss stay exactly the same. Reset restores the four original observations so you can check the calculation above.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>For each centered point, the projection and its residual form a right triangle. The squared length of the original vector is the sum of the two squared lengths:</Prose>
    <MathBlock>{'\\|a\\|^2=\\|\\widehat a\\|^2+\\|a-\\widehat a\\|^2.'}</MathBlock>
    <Prose>The symbol ‖a‖² means the sum of the squared coordinates of a. Add this identity over all observations:</Prose>
    <MathBlock>{'\\begin{gathered}\\text{total centered squared length}\\\\ =\\text{retained squared length}\\\\ \\quad+\\text{residual squared length}.\\end{gathered}'}</MathBlock>
    <Prose>The total on the left does not change when we turn the ruler. Therefore the direction that retains the most squared length also loses the least. These are two views of the same optimization, not competing definitions of PCA.</Prose>
    <Prose>For the four-point example, total centered squared length is 5 + 5 + 5 + 5 = 20. The diagonal retains 18 and loses 2. The horizontal ruler retains 10 and loses 10. The perpendicular diagonal retains 2 and loses 18.</Prose>
    <ConservationFigure />
    <H3>From spread to variance</H3>
    <Prose>The sample variance of a list of centered readings is their squared sum divided by n − 1, where n is the number of observations. For our four scores:</Prose>
    <MathBlock>{'\\lambda_1=\\frac{18}{4-1}=6.'}</MathBlock>
    <Prose>The perpendicular direction v₂ = (1, −1)/√2 has score variance 2/3. It is the second principal direction. Its scores complete the coordinate system:</Prose>
    <LessonTable caption="Both scores for each observation" headers={['observation', 'first score', 'second score']} rows={[['A', '−3/√2', '−1/√2'], ['B', '−3/√2', '1/√2'], ['C', '3/√2', '−1/√2'], ['D', '3/√2', '1/√2']]} />
    <Prose>Keeping both scores permits exact reconstruction. Keeping the first score retains</Prose>
    <MathBlock>{'\\frac{6}{6+2/3}=0.9=90\\%'}</MathBlock>
    <Prose>of the total sample variance. This is the <strong>explained variance ratio</strong>.</Prose>
    <Prose>Its precise meaning is valuable: on the data used to fit this centered, unwhitened PCA, the retained coordinates contain 90% of the total squared deviation from the mean. The other 10% is reconstruction loss in the same geometry.</Prose>
    <Callout title="What explained variance certifies, and what it does not">
      Explained variance measures this squared-error objective and nothing else. It is not a percentage of facts, class information or scientific meaning preserved. A low-variance direction can contain the distinction a task needs; a large-variance direction can reflect an unwanted artifact. Section 8 constructs that case with a component that retains 99% of the variance and none of the label, and section 10.3 separates finite-sample variation from population structure. Every percentage below should be read with this distinction; we will not repeat it after each one.
    </Callout>
    <H3>Error units: three different summaries of the same residuals</H3>
    <Prose>Our example has total squared error, or <strong>SSE</strong>, equal to 2. The average squared distance per observation is 2/4 = 0.5. The average squared error per numeric entry is 2/(4×2) = 0.25.</Prose>
    <Prose>They are all correct, but answer different questions. A statement such as “MSE is 0.25” needs its denominator. Hereafter, a reconstruction MSE means an average over all entries; the projection workbench shows SSE so its pieces add directly.</Prose>

    <H2>{headings[3]}</H2>
    <Prose>Let X have shape (n, d): n observations and d input features. Fit a mean and k directions from permitted fitting data. Then transform any observation by subtracting that same mean and taking its k dot products.</Prose>
    <TransformShapesFigure />
    <Prose>Use Python with NumPy and scikit-learn. If needed, install them in a project environment:</Prose>
    <CodeBlock language="bash">{'python -m pip install numpy scikit-learn'}</CodeBlock>
    <Prose>The examples were checked with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1. They use the CPU and require no network once the packages are installed; the Wine data ship with scikit-learn. Save each standalone program as its own file and run it with that environment’s Python. Programs marked as continuations extend the variables of the program named just before them, so paste them below it in the same file. Printed floating-point values are rounded; tiny final-bit differences are normal.</Prose>
    <H3>A compact implementation using SVD</H3>
    <Prose>SVD is a matrix decomposition that supplies the directions and their strengths. You can use the program now; section 10.1 explains why it computes PCA.</Prose>
    <Program example={pcaExamples.svd}><Prose><Code>directions</Code> stores principal directions as <strong>rows</strong>. That is why the forward multiplication uses <Code>.T</Code> and the reconstruction does not. This convention agrees with scikit-learn’s <Code>components_</Code>. The program assumes a finite numeric matrix with at least two rows and nonzero total variation; missing values and constant data are discussed in section 10.2. It is a teaching implementation with no automatic choice of scaling or component count.</Prose></Program>
    <H3>The library version exposes the same operations</H3>
    <Program example={pcaExamples.library}><Prose><Code>fit</Code> learns the mean and directions; <Code>transform</Code> uses them; <Code>inverse_transform</Code> reconstructs. For the new point, (6, 4) − (3, 2) = (3, 2). Its retained component is (2.5, 2.5), giving (5.5, 4.5) after restoring the mean. We did not move the fitted ruler to accommodate the new observation. PCA centers its inputs; it does <strong>not</strong> standardize their scales automatically.</Prose></Program>

    <H2>{headings[4]}</H2>
    <Prose>A squared distance adds differences from all features. If one feature is in thousands and another is in tenths, the first can dominate that sum. A change of measurement unit can therefore change the PCA answer without changing the objects being measured.</Prose>
    <H3>An exact unit-change experiment</H3>
    <Prose>Take four centered points: (−2, −1), (−2, 1), (2, −1), (2, 1). Variation along the horizontal coordinate is four times variation along the vertical coordinate. PC1 is horizontal and retains 80% of the variance.</Prose>
    <Prose>Now multiply the second coordinate by 10, for example by expressing the same lengths in a unit ten times smaller. Horizontal variance stays 16/3; vertical variance becomes 400/3. PC1 becomes vertical and retains 400/416 ≈ 96.15%.</Prose>
    <Prose>PCA has answered the new numerical question correctly. We changed the relative penalty assigned to errors on the two axes.</Prose>
    <Prose><strong>Standardization</strong> divides each centered feature by its own standard deviation:</Prose>
    <MathBlock>{'y_{ij}=\\frac{x_{ij}-\\mu_j}{s_j}.'}</MathBlock>
    <Prose>A difference of one then means one fitted standard deviation for that feature. On the rectangle, standardization produces a square with equal variance along the two coordinates. No unique first direction is preferred. It does not uncover a secret diagonal; it makes the symmetry explicit.</Prose>
    <PcaMetricLab />
    <Prose>Try a multiplier you choose, then repeat with standardization. Edit the rectangle’s width or height and predict the multiplier at which the two axes tie: it is the ratio of width to height.</Prose>
    <H3>Make a scaling decision for the wines</H3>
    <Prose>For this exploratory view, we want each chemical feature to contribute on a comparable relative scale. We will standardize. That is a modeling choice, not a universal preprocessing law. If a later task supplies measurement-error variances or physical costs, those may define a more appropriate weighting. Standardizing a nearly constant noisy feature can give its noise disproportionate influence.</Prose>
    <Prose>The following analysis describes <strong>all 178 supplied observations</strong>. Its purpose is to compare representations of this fixed collection. The next section starts a separate train/validation analysis for choosing a representation for other observations.</Prose>
    <Program example={pcaExamples.wineScaling}><Prose>The raw first component is almost entirely the proline direction. Its roughly 99.8% variance share reflects the original column magnitudes. After standardization, PC1 retains about 36.2% and PC2 about 19.2%. Their combined 55.4% is a substantially less complete reconstruction of the standardized measurements than a casual two-dimensional picture might suggest.</Prose></Program>
    <Prose>That lower percentage does not make standardization a failure. We changed what counts as a large error. Comparing the raw and standardized percentages as though they measured the same objective would be like comparing a distance in centimeters with an unrelated distance in seconds.</Prose>
    <WineOverviewFigure />
    <Prose>Plotting the cultivar labels over the scores lets us ask whether this unsupervised summary aligns with a known grouping. Do not include the label as a fourteenth numeric feature; the plot would then partly encode the answer we hoped to inspect. Class numbers 0, 1, 2 in scikit-learn are codes, not quantities with meaningful distances; the CSV uses the original 1, 2, 3 labels.</Prose>

    <H2>{headings[5]}</H2>
    <Prose>Before choosing k, finish this sentence: “I need the representation to…”</Prose>
    <LessonTable caption="The purpose decides the rule" headers={['purpose', 'a useful decision rule', 'what to examine']} rows={[
      ['Show the observations', 'Start with two components; inspect further pairs if helpful', 'Labeled score plots, retained variation and original features'],
      ['Store approximate measurements', 'Smallest k meeting an explicit error budget', 'Reconstruction error, feature-level residuals and total storage'],
      ['Help a prediction model', 'Compare the complete pipeline with a no-PCA baseline', 'Validation performance on the actual task'],
      ['Reduce measurement noise', 'Evaluate against a justified signal/noise model or independent target', 'Error relative to the target, not merely the noisy input']
    ]} />
    <Prose>A <strong>scree plot</strong> shows the variance contributed by each component. A cumulative plot adds those contributions. An elbow can suggest where gains become smaller, but a clear elbow need not exist. For the standardized full Wine collection, the first two components retain 55.41%, eight retain 92.02%, and ten retain 96.17%. Those are useful accounting facts, not universal cutoff recommendations.</Prose>
    <H3>Keep fitting information separate from evaluation information</H3>
    <Prose>To choose a transform for future observations, split the data first. Learn imputation, means, scales and PCA directions on the training portion. Apply those fitted operations unchanged to validation observations. If the observations are grouped by person, session, source or time, make the split reflect the intended use rather than mixing related rows arbitrarily.</Prose>
    <Prose>This is the lesson’s home for <strong>data leakage</strong>: even a transform that never reads labels can leak evaluation information through its fitted means or directions. A pipeline passed into cross-validation refits its preprocessing within each training fold. Putting previously transformed full data into cross-validation does not. <a href="https://scikit-learn.org/stable/common_pitfalls.html#data-leakage">Scikit-learn’s leakage guide</a> develops the point.</Prose>
    <H3>A worked compression budget</H3>
    <Prose>For an exercise in measurement reconstruction, set this budget before examining the results: retain the fewest components whose validation squared error is at most <strong>10% of the error from always returning the training mean</strong>.</Prose>
    <Prose>Our baseline stores no per-observation coordinates. Every reconstructed validation row is the training mean. We measure all errors in the same training-standardized coordinates. A ratio of 0.10 means 90% less squared error than this baseline on the validation rows; it is not a classification score.</Prose>
    <Program example={pcaExamples.budget}><Prose>The label is used only to keep the three cultivar proportions represented in both portions, a <strong>stratified split</strong>. PCA still receives no labels. The baseline’s validation MSE is 1.0963; training standardization does not force validation variance or mean to equal the training values. Seven components leave 12.65% of baseline error. Eight leave 9.60%, so <strong>eight is the smallest count satisfying this particular budget</strong>. Two components are useful for a picture but miss the budget substantially. If the goal had instead been “retain at least 95% of training variance,” this split would select ten.</Prose></Program>
    <PcaBudgetLab />
    <Prose><strong>Why not just minimize validation reconstruction error?</strong> With one fixed orthonormal basis, each added component removes a nonnegative squared residual for every observation. The error therefore cannot increase as k grows, even on validation data. Minimizing that error alone selects all dimensions. Compression needs a budget, penalty or constraint in addition to error.</Prose>
    <Prose>We have used validation results to choose a representation. They are development evidence. A final claim about performance on future data needs a further untouched evaluation set or an appropriate resampling protocol. Section 10.3 explains how this differs from denoising against a clean target.</Prose>
    <H3>Can the measurements actually be recovered in their original units?</H3>
    <Prose>For a retained k, first reconstruct in the standardized coordinate system, then undo standardization:</Prose>
    <Program example={pcaExamples.originalUnits}><Prose>The thirteen output columns are approximate recovered measurements, not thirteen independently retained numbers. Keep the training scales, means, component directions and feature order with the scores; they are the decoder.</Prose></Program>

    <H2>{headings[6]}</H2>
    <H3>Coefficients describe a direction; scores describe observations</H3>
    <Prose>For the standardized Wine collection in section 5, PC1 gives approximately these weights to selected features:</Prose>
    <LessonTable caption="Selected PC1 direction coefficients, standardized Wine, all 178 rows" headers={['feature', 'PC1 direction coefficient']} rows={[['Total phenols', '0.3947'], ['Flavanoids', '0.4229'], ['Nonflavanoid phenols', '−0.2985'], ['Proline', '0.2868']]} />
    <Prose>The complete score uses all thirteen coefficients. Holding the other standardized coordinates fixed, increasing flavanoids by one fitted standard deviation increases the score by about 0.4229. This tells you how the score is calculated. It does not establish that changing a compound causes a change in cultivar or quality.</Prose>
    <Prose>Call a component something descriptive only after inspecting its coefficients, the observations and the scientific context. “A contrast involving several phenolic measurements” is supported more directly than a claim that PC1 is an intrinsic quality axis. The sign convention can be reversed with no change in reconstructions, as section 2 showed.</Prose>
    <Prose>The word <strong>loading</strong> has more than one convention. Here a loading means a unit direction coefficient, as in <Code>components_</Code>. Some sources use feature–score correlations or direction coefficients multiplied by the square root of an eigenvalue. Label the quantity being shown rather than assuming the numbers should match across software.</Prose>
    <Prose>For a centered feature with sample standard deviation sⱼ and a nonzero-variance score, the feature–score correlation is</Prose>
    <MathBlock>{'\\operatorname{corr}(X_j,Z_\\ell)=\\frac{v_{j\\ell}\\sqrt{\\lambda_\\ell}}{s_j}.'}</MathBlock>
    <Prose>You can check the distinction in the four-point example: the first coefficient is 1/√2 ≈ 0.7071, but its correlation with the first score is √0.9 ≈ 0.9487. A coefficient and a correlation answer different questions.</Prose>
    <H3>A biplot overlays two kinds of objects</H3>
    <Prose>A <strong>score plot</strong> puts observations at their component coordinates. A <strong>biplot</strong> also draws arrows for features. To read one, ask what scales the arrows and observations use.</Prose>
    <Prose>Our biplot uses observation scores zᵢ and feature arrows aⱼ = (vⱼ₁, vⱼ₂). Their dot product gives the rank-two approximation to the centered feature value:</Prose>
    <MathBlock>{'\\widehat{x}_{ij}-\\mu_j=z_i\\cdot a_j.'}</MathBlock>
    <Prose>For standardized PCA, that value is in standardized units. If arrows are enlarged for visibility, their display multiplier must be stated and removed before this calculation. With this particular scaling, arrow angles alone are not a general formula for original feature correlations. Other biplot scalings support different interpretations; <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/">Jolliffe and Cadima</a> survey the conventions.</Prose>
    <BiplotReadingFigure />
    <H3>Close in the picture can mean far in the omitted coordinates</H3>
    <Prose>If two observations overlap in PC1/PC2, inspect their remaining scores or original features before concluding they are duplicates. The first picture deliberately discards information. A point with an ordinary-looking score can also have a large residual away from the retained subspace. Section 10.5 uses that residual as a diagnostic.</Prose>

    <H2>{headings[7]}</H2>
    <Prose>The previous module topic, <a href="/learn/path/full-curriculum/k-means-hierarchical-clustering?module=classical-ml">K-Means &amp; Hierarchical Clustering</a>, made the representation part of the clustering question. PCA lets us make that connection exact.</Prose>
    <H3>Rotating all coordinates preserves Euclidean distances</H3>
    <Prose>An orthonormal coordinate change preserves the length of every difference vector. For a complete matrix of orthonormal directions V:</Prose>
    <MathBlock>{'\\|(x-y)V\\|^2=\\|x-y\\|^2.'}</MathBlock>
    <Prose>Centering cancels from x − y. Keeping <strong>all</strong> coordinates therefore changes neither Euclidean pair distances nor the K-Means squared-distance objective for a corresponding partition. A numerical algorithm can still make different choices at ties; the mathematical objective has not changed.</Prose>
    <Prose>Keeping only k directions gives</Prose>
    <MathBlock>{'\\begin{gathered}\\|x-y\\|^2=\\|(x-y)V_k\\|^2\\\\ \\quad+\\|(x-y)V_{\\mathrm{discarded}}\\|^2.\\end{gathered}'}</MathBlock>
    <Prose>This is the same right-triangle accounting used for reconstruction in section 3, now applied to a difference between two observations. Distances can only shrink under orthogonal projection; how much they shrink depends on the pair.</Prose>
    <Prose>For A and B, the original squared distance is 2. Their one-component scores coincide, so the retained squared distance is 0; all 2 lies in the discarded direction. A 90% variance summary has erased 100% of this pair’s distance. That is why a global percentage cannot guarantee each neighborhood is preserved.</Prose>
    <H3>A 99% variance component can lose the entire label</H3>
    <Prose>Construct four observations (−10, −1), (−10, 1), (10, −1), (10, 1). Define a class by the sign of the second coordinate. PC1 is the horizontal axis and retains 100/101 ≈ 99.01% of the variance. After keeping it, each horizontal location contains both classes at the same score. No deterministic classifier using that score alone can distinguish the two observations at either location.</Prose>
    <Prose>Using the second coordinate alone separates the classes perfectly on this constructed dataset. We are comparing two objectives: explain feature variation and retain information about this label.</Prose>
    <PcaTaskInformationLab />
    <Prose>For a prediction task, compare a no-PCA pipeline with PCA pipelines using validation on the actual prediction metric. Section 10.4 gives a complete small example. If the goal is clustering, examine stability and relevant external evidence, not just the attractiveness of a PCA scatterplot.</Prose>
    <Prose>Reducing d to k can reduce work in computing each pairwise distance. It does not reduce the number of pairs among n observations. An explicitly stored dense distance matrix is still n × n.</Prose>

    <H2>{headings[8]}</H2>
    <Prose>Try each question before opening its hint or solution. The first four are the core checkpoint. Questions 5 to 8 connect to the later branches.</Prose>
    <Practice title="1. Change the data, keep the method" question="Use (1,0), (3,2), (5,4). Calculate the mean, a first principal direction, all three scores, the sample variance of those scores and the one-component reconstruction SSE. Then add (8,4) as a new observation, using the original fitted transform. What is its reconstruction and squared error?" hint="The original three points are exactly on a line of slope one. The new point is not on that line. Keep the original mean when transforming it.">
      <Prose>The mean is (3, 2). Choose v₁ = (1, 1)/√2. Centered rows are (−2, −2), (0, 0), (2, 2) and scores are −2√2, 0, 2√2. Sample variance is (8 + 0 + 8)/2 = 8. Their one-component reconstruction SSE is zero.</Prose>
      <Prose>The new centered point is (5, 2). Its score is 7/√2; reconstructing gives (3, 2) + (3.5, 3.5) = (6.5, 5.5). Residual (1.5, −1.5) has squared length 4.5. Exact training reconstruction did not imply exact reconstruction of every possible new point.</Prose>
    </Practice>
    <Practice title="2. Change the direction instead of memorizing a percentage" question="In the four-point example from section 2, keep the second direction (1, −1)/√2 instead of the first. Find A's score and reconstruction. What are the total SSE, per-entry MSE and retained variance fraction?" hint="The two components divide total centered squared length 20 into 18 and 2. Swapping which is retained swaps the loss.">
      <Prose>A’s score is −1/√2. Its reconstruction is (3, 2) + (−0.5, 0.5) = (2.5, 2.5). The total SSE is 18; per-entry MSE is 18/8 = 2.25; retained variance is 10%. A’s individual squared error is 4.5. This checks whether you can trace the inverse mapping, not just recite “keep the largest component.”</Prose>
    </Practice>
    <Practice title="3. Repair a misleading scaling conclusion" question="A report says: “The raw Wine first component retains 99.8%; the standardized first component retains only 36.2%. Standardization destroyed 63.6% of the information.” Explain the error and write a better one-sentence conclusion." hint="Ask what squared error each percentage uses as its denominator.">
      <Prose>The representations use different feature weightings, so their percentages do not measure the same error objective. Neither percentage measures all useful information. One acceptable conclusion is: “Raw PCA is dominated by large-magnitude measurements, whereas standardized PCA spreads retained relative variation across more components; choose the weighting from the intended task.” A strong answer also identifies which original feature dominates, proline, rather than merely saying “scaling matters.”</Prose>
    </Practice>
    <Practice title="4. Run a new compression budget" question="Run the complete program in section 6. Change the allowed validation error fraction from 0.10 to 0.06. Predict the smallest acceptable count before reading the result. Report the count, the two errors bracketing the budget, the coordinate system and whether you have measured final test performance." hint="You need both the last count that fails and the first count that passes. Reuse the training transform; do not fit a fresh validation PCA.">
      <Prose>Nine components leave about 0.072354 of the training-mean baseline error; ten leave about 0.052066. The smallest acceptable count is <strong>10</strong>. Errors use training-standardized measurements on the 45 validation rows. These are model-selection results, not an untouched final test result. A different split can give a different count. The budget lab above reproduces these two ratios exactly when you set 0.06.</Prose>
    </Practice>
    <Practice title="5. Diagnose a pipeline leak" question="A colleague computes StandardScaler().fit_transform(X) on the entire dataset, applies PCA once, and then uses cross-validation to select a classifier. Explain what must move inside the folds. Separately, why is the descriptive full-collection analysis in section 5 not claiming that kind of generalization evidence?">
      <Prose>Both the fitted scaler and fitted PCA must be learned using each fold’s training portion, along with the classifier. Pass the complete unfitted pipeline into cross-validation. Section 5 describes the observations supplied to the transform; it does not estimate predictive performance on unseen observations. The use of the data and the claim made about the output determine the evaluation obligation.</Prose>
    </Practice>
    <Practice title="6. Audit a storage promise" question="You have n = 100 rows and d = 20 features. You keep k = 10 PCA scores per row. Count stored scalar values if you retain the score matrix, component directions and mean, with no feature scaling and the same scalar precision throughout. Does this halve storage? At what number of rows does this choice first save storage?" hint="The decoder also takes space. Compare nk + dk + d with nd.">
      <Prose>Original: 100 × 20 = 2000 scalars. Compressed representation plus decoder: 100 × 10 + 20 × 10 + 20 = 1220, or 61% of the original, so 39% saved, not 50%. Savings require 10n + 220 {"<"} 20n, hence n {">"} 22; the first integer is 23. For 22 rows the counts tie. Metadata, storage format and quantization are outside this scalar-count calculation.</Prose>
    </Practice>
    <Practice title="7. Separate uncorrelated from independent" question="Give U the values −1, 0, 1 with equal probability and define W = U². Show that U and W have zero covariance but are not independent. Why does this matter when describing PCA scores?">
      <Prose>E[U] = 0 and E[UW] = E[U³] = 0, so their covariance is zero. But W = 0 tells us U = 0, while W = 1 tells us |U| = 1; they are dependent. PCA diagonalizes the fitted sample covariance of its scores. It does not generally produce statistically independent variables. The later ICA topic studies a different objective.</Prose>
    </Practice>
    <Practice title="8. Why does a noise-only scree plot have a leading component?" question="Run the short Gaussian example in section 10.3. Before executing it, decide whether the first two sample components must retain exactly 10% of variation. Explain the result without inventing a hidden two-dimensional cause. Change the seed and repeat; state what you would need before giving an empirical component a scientific name.">
      <Prose>The population covariance is the 20-dimensional identity, so every fixed two-dimensional orthonormal projection captures 10% of population variance. PCA chooses its directions after inspecting a finite sample. The chosen directions capitalize on that sample’s uneven spread. For seed 23 the first two retain about 24.59% of the sample variation. Another seed changes the number.</Prose>
      <Prose>A good interpretation distinguishes the fitted sample result from population structure, and proposes evidence tied to the intended claim: repeated data, stability of the retained subspace, a suitable null or noise model, known measurement factors or held-out task performance. Naming the tallest component is not that evidence.</Prose>
    </Practice>
    <H3>Independent mini-project</H3>
    <Prose>Using the supplied Wine data, write a short analysis that answers <strong>one</strong> purpose: exploratory visualization, measurement compression, or prediction. State the unit of observation, which rows fit the transform, the scale choice, the baseline and the success criterion before reporting results. Include one changed setting and an observation-level diagnosis, not only a global score.</Prose>
    <Prose>For compression, a complete submission contains the reproducible split, selected count, error against the mean baseline, a per-feature or per-row residual inspection, retained decoder information and a statement about what the validation result supports. The 6% variation above provides an exact numerical checkpoint; explaining why it leads to ten components is part of the task.</Prose>
    <Prose>For prediction, use the pipeline in section 10.4 as a starting point, then change one justified modeling choice. Report the fold results for both PCA and the no-PCA baseline, not just whichever mean is higher. The point is to make and evaluate a decision, not to force PCA to win.</Prose>

    <H2>{headings[9]}</H2>
    <Prose>Each branch below answers a question that becomes relevant after the core route. Read them in any order.</Prose>
    <H3>10.1 Why eigenvectors and SVD compute the same PCA</H3>
    <Prose>Read this branch when you want to connect the geometric calculation to the matrix algorithm. <a href="/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu?module=math-foundations">Matrix Decompositions</a> and <a href="/learn/path/full-curriculum/eigenvalues-eigenvectors?module=math-foundations">Eigenvalues &amp; Eigenvectors</a> provide extended background.</Prose>
    <Prose>Write the centered data matrix as A. Its sample covariance matrix is</Prose>
    <MathBlock>{'C=\\frac{A^\\top A}{n-1}.'}</MathBlock>
    <Prose>The diagonal entries are feature variances. An off-diagonal entry measures how two features vary together. The matrix has shape (d, d), is symmetric, and is positive semidefinite: for any direction v, vᵀCv is a squared length divided by n − 1, so it cannot be negative.</Prose>
    <Prose>For the four-point example:</Prose>
    <MathBlock>{'C=\\frac13\\begin{bmatrix}10&8\\\\8&10\\end{bmatrix}.'}</MathBlock>
    <Prose>Multiplying by (1, 1) gives 6(1, 1); multiplying by (1, −1) gives (2/3)(1, −1). These directions are <strong>eigenvectors</strong>: the covariance matrix stretches each without changing its direction. The stretch factors 6 and 2/3 are its eigenvalues.</Prose>
    <Prose>Why should this solve the optimization? The score vector for a unit direction v is Av. Its sample variance is</Prose>
    <MathBlock>{'\\frac{(Av)^\\top(Av)}{n-1}=v^\\top Cv.'}</MathBlock>
    <Prose>A symmetric covariance matrix admits an orthonormal eigenbasis. Express a unit direction as v = Σⱼaⱼvⱼ, where Σⱼaⱼ² = 1. Its score variance is</Prose>
    <MathBlock>{'v^\\top Cv=\\sum_j a_j^2\\lambda_j\\leq\\lambda_1.'}</MathBlock>
    <Prose>This is a weighted average of the eigenvalues, with nonnegative weights adding to one. It cannot exceed the largest eigenvalue. Choosing its eigenvector achieves the maximum. Restricting to directions perpendicular to that vector gives the next largest eigenvalue, and so on. Eigenvalues are ordered <strong>nonincreasingly</strong>, allowing ties.</Prose>
    <Prose>You can also use a Lagrange multiplier for the constraint vᵀv = 1. Differentiating vᵀCv − λ(vᵀv − 1) gives 2Cv − 2λv = 0. The stationary directions are eigenvectors; the weighted-average argument identifies the maximum rather than merely finding a stationary point.</Prose>
    <Prose>Now take the reduced SVD:</Prose>
    <MathBlock>{'\\begin{gathered}A=USV^\\top,\\\\ C=V\\frac{S^2}{n-1}V^\\top.\\end{gathered}'}</MathBlock>
    <Prose>Here r = min(n, d), U has shape (n, r), S is the diagonal (r, r) matrix of singular values, and Vᵀ has shape (r, d). Zero singular values are allowed. The nonzero covariance eigenvalues are sⱼ²/(n − 1); any omitted covariance eigenvalues are zero. The rows of Vᵀ are the principal directions returned by the program in section 4.</Prose>
    <Prose>The first k score columns can be calculated two ways:</Prose>
    <MathBlock>{'Z=AV_k=U_kS_k.'}</MathBlock>
    <Prose>This is the same score matrix: dot products with directions on the left, SVD factors on the right. Its sample covariance is diagonal, with the retained eigenvalues on the diagonal. The scores are uncorrelated <strong>on the fitting sample</strong>. Uncorrelated does not mean independent; practice 7 gives a counterexample.</Prose>
    <Prose>For orthonormal retained directions, VₖVₖᵀ is the projection matrix in feature space. The reconstructed centered matrix is AVₖVₖᵀ. The singular values give its error directly:</Prose>
    <MathBlock>{'\\begin{gathered}\\|A-AV_kV_k^\\top\\|_F^2\\\\ =\\sum_{j>k}s_j^2=(n-1)\\sum_{j>k}\\lambda_j.\\end{gathered}'}</MathBlock>
    <Prose>The Frobenius norm squared, ‖·‖²_F, adds the squares of all matrix entries. This identity explains the 90% variance and 10% loss result without a separate error model. The truncated SVD is optimal among rank-at-most-k matrix approximations for this squared-error objective, the <strong>Eckart–Young</strong> result. It does not assert optimality for label preservation or nonlinear compression.</Prose>
    <Prose>If you remember the covariance example from Eigenvalues &amp; Eigenvectors, these are the same centered observations and the same eigenvalues. Here we have connected them to a fitted mean, reconstruction and a practical compression decision.</Prose>
    <Program example={pcaExamples.eigen}><Prose>The comparison uses reconstructions, so a harmless sign difference cannot make it fail. The geometry and covariance viewpoints have a long history: Pearson’s closest-fit formulation dates to 1901 and Hotelling’s statistical treatment to 1933. Their connection is useful because the two viewpoints lead to the same computation, not because the names must be memorized.</Prose></Program>

    <H3>10.2 What happens at zero variance, ties and whitening?</H3>
    <Prose><strong>Rank.</strong> Centering makes the rows sum to zero, so the centered matrix has rank at most min(n − 1, d). With four observations in twenty features, at most three principal components have nonzero sample variance. That is a finite-data constraint, not a discovery that the population has only three degrees of freedom. Retaining all nonzero training directions exactly reconstructs training data; a future point can still lie outside their span.</Prose>
    <Prose><strong>Tied directions.</strong> If two eigenvalues are equal, any orthonormal basis within their shared eigenspace is valid. If k cuts through that tie, the selected k-dimensional subspace can also be nonunique. If eigenvalues are merely close, small data changes can rotate individual directions substantially. Compare reconstructions or projection matrices, and compare whole tied subspaces where appropriate, rather than treating every changed vector as an error.</Prose>
    <Prose><strong>Constant data.</strong> If every observation is identical, total centered variance is zero. The mean reconstructs everything; a “fraction explained” divides zero by zero and has no informative value. If only one column is constant, it has no centered variation. <Code>StandardScaler</Code> leaves such a column with scale factor one rather than dividing by zero. Missing values require an explicit strategy before ordinary PCA; fitting an imputer follows the same information boundary as the other fitted operations in section 6.</Prose>
    <Prose><strong>Whitening changes the metric again.</strong> PCA alone rotates and optionally truncates. Whitening additionally divides each nonzero-variance score by the square root of its fitted variance:</Prose>
    <MathBlock>{'w_{ij}=z_{ij}/\\sqrt{\\lambda_j}.'}</MathBlock>
    <Prose>The retained score columns then have unit sample variance on the fitting data. Directions with small original variance get magnified relative to large-variance directions. Euclidean distances in whitened coordinates therefore differ from distances in ordinary PCA coordinates. This can suit a downstream model, but should be evaluated for that model’s purpose.</Prose>
    <Prose>For the four-point example, fit <Code>PCA(n_components=2, whiten=True, svd_solver="full")</Code>. In the checked scikit-learn version, the transformed columns have variance 1 using <Code>ddof=1</Code>, and 0.75 using <Code>ddof=0</Code>. These denominators differ: dividing by four rather than three changes the variance by 3/4. Whitening is not a way to make a zero-variance component informative.</Prose>
    <Prose><Code>StandardScaler</Code> uses the <Code>ddof=0</Code> convention for its fitted scales, while PCA reports sample variances with n − 1. For fully observed nonconstant columns standardized on n fitting observations, PCA’s sample feature variances are consequently n/(n − 1). The common factor does not change directions or variance ratios, but it matters when checking absolute values.</Prose>

    <H3>10.3 Sampling variation is not hidden structure</H3>
    <Prose>Suppose twenty independent Gaussian measurements each have population variance one. The population covariance is the identity. It has no preferred direction. A finite sample will not have exactly that covariance; the entries fluctuate.</Prose>
    <Program example={pcaExamples.gaussian}><Prose>These leading components explain 24.59% of this sample’s variation, despite the population having no special two-dimensional subspace. PCA selected the strongest directions in the very sample being summarized. The <a href="/learn/path/full-curriculum/random-matrix-theory?module=math-foundations">Random Matrix Theory</a> lesson develops how sample spectra behave when the number of features is large relative to the number of observations.</Prose></Program>
    <GaussianSpectrumFigure />
    <Prose>A useful stability check refits on repeated samples or resampled rows and compares retained subspaces. It should respect the data’s grouping or dependence. Stable directions can still describe a stable artifact, so stability and scientific interpretation answer different questions.</Prose>
    <Prose><strong>Denoising changes the target of error.</strong> Suppose a true two-sensor signal is (s, s), and a measurement adds error (e, −e). Projection onto the diagonal removes that error exactly: the signal and error occupy perpendicular directions. If the signal variation dominates, PCA can identify that diagonal from suitable data.</Prose>
    <Prose>If the error instead is (e, e), it lies along the same direction as the signal. The diagonal projection preserves it. PCA cannot separate two contributions merely because we call one “noise.”</Prose>
    <Prose>When evaluating denoising, compare a reconstruction to a justified clean target, independent measurement or explicit noise model. Reconstructing the noisy input in all dimensions gives zero input reconstruction error while preserving every noise realization. This is why the monotone input-error curve in section 6 is not, by itself, a denoising validation curve.</Prose>

    <H3>10.4 Evaluate PCA inside a prediction pipeline</H3>
    <Prose>Here is a complete development comparison using the same Wine data. It asks whether reducing the features helps a logistic-regression classifier predict the cultivar. Each candidate is evaluated on the same five stratified folds. The scaler and PCA are refitted inside each training fold.</Prose>
    <Program example={pcaExamples.pipeline}><Prose><Code>None</Code> labels the no-PCA baseline. On these folds, neither compressed candidate improves its mean accuracy. Eight components retain more predictive utility than two, but the original standardized features work well for this classifier and dataset size. The fold values show variation hidden by a single mean; they are not independent replications or a confidence interval. Selecting a pipeline after inspecting them uses development evidence, as discussed in section 6. <a href="/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml">Cross-Validation &amp; Hyperparameter Tuning</a> develops the evaluation protocol further.</Prose></Program>

    <H3>10.5 Two useful applications beyond a scatterplot</H3>
    <Prose><strong>Compression requires a decoder.</strong> For n observations with d features, the original numeric matrix stores nd scalars. Keeping k scores per observation, k feature directions and a mean stores nk + dk + d scalars before any scaling metadata. For n = 1000, d = 100, k = 10, that is 11,100 rather than 100,000 scalars at equal precision. Whether a particular file becomes that much smaller also depends on its format and encoding. Practice 6 shows why small datasets can have much less impressive savings.</Prose>
    <Prose>For images, each column can be a pixel and each row an image. Reshape a principal direction into the original image layout: it becomes a <strong>pattern of positive and negative pixel weights</strong>. Reconstruction adds weighted patterns to the mean image. A direction can contain negative weights even when every observed pixel is nonnegative; the direction describes changes around the mean. Keeping two such patterns is a two-coordinate image representation, not an image with two pixels.</Prose>
    <Prose><strong>A discarded direction can be a useful alarm.</strong> Consider a constructed two-sensor system whose normal readings vary close to (s, s). Fit its usual diagonal direction. A later centered observation (3, 3) has a large score along the usual mode but zero residual. The observation (3, −3) has zero diagonal score but residual squared length 18. A score-only display calls the second point ordinary-looking; its residual shows a strong disagreement between the sensors.</Prose>
    <ResidualAlarmFigure />
    <Prose>This suggests inspecting two diagnostics: how far an observation travels within the usual subspace, and how far it falls outside it. Turning those diagnostics into an operational alarm requires data about normal variation, relevant faults and false-alarm costs. The geometry alone does not supply an alarm threshold.</Prose>
    <Prose>A related question appears with neural population recordings: a row might represent one time bin and columns different neurons. Scores summarize population variation; reconstructing reveals which activity patterns are discarded. Shuffling row order does not change the fitted covariance, so ordinary PCA by itself does not learn temporal dynamics. Connecting score points in time order adds a trajectory display, not a dynamical model. The planned <a href="/learn/path/full-curriculum/dimensionality-reduction-manifold-analysis-for-neural-data?module=computational-neuroscience">Dimensionality Reduction &amp; Manifold Analysis for Neural Data</a> develops the measurement, time-axis and validation issues needed for that use.</Prose>

    <H3>10.6 Choose a computation that fits the data</H3>
    <Prose>The mathematical objective does not prescribe forming a huge covariance matrix. For dense real data, an economy-size direct SVD costs on the order of nd·min(n, d) arithmetic operations; forming and diagonalizing AᵀA instead costs approximately nd² + d³. These are scaling models, not measured running times.</Prose>
    <Prose>On the nonzero singular spectrum, forming AᵀA squares the condition number. That can obscure small components in floating-point arithmetic. Direct SVD avoids that particular loss of conditioning, while covariance eigendecomposition can still be an efficient choice for many rows and relatively few, well-conditioned features.</Prose>
    <LessonTable caption="Computational approaches and when to consider them" headers={['approach', 'what it computes or stores', 'when to consider it']} rows={[
      ['Direct full SVD', 'Complete reduced factorization, then truncation', 'Small or moderate dense problems; reliable reference for retained components'],
      ['Covariance eigendecomposition', 'A d×d covariance matrix and its eigenvectors', 'Many more rows than features, with manageable d² storage'],
      ['Randomized SVD', 'A sketch targeting roughly k directions', 'k much smaller than the smaller matrix dimension'],
      ['Incremental PCA', 'Updates a retained approximation from batches', 'Rows do not all fit in memory'],
      ['Truncated SVD without centering', 'Low-rank factors of the supplied matrix', 'Sparse count or TF–IDF matrices when uncentered approximation is the intended objective']
    ]} />
    <Prose>Randomized SVD uses a random test matrix Ω with about k + p columns, where p is oversampling. Compute Y = AΩ, find an orthonormal basis Q for those columns, and decompose the smaller matrix QᵀA. Power iterations can improve separation of leading directions at the cost of extra passes. Approximation quality depends on the spectrum and settings. For a fixed number of passes and small k, the leading dense multiplication work is roughly nd(k + p); reduced factorizations add work. <a href="https://arxiv.org/abs/0909.4061">Halko, Martinsson and Tropp</a> give the algorithm and its bounds.</Prose>
    <Prose>The sketch workspace is not the entire memory cost. A million-by-five-thousand float64 input alone contains 5×10⁹ numbers, or about <strong>40 GB</strong> in decimal units. Selecting 50 components does not make that input disappear. Batch processing, memory mapping or another data-access plan may still be necessary.</Prose>
    <Prose>Incremental PCA keeps a compressed approximation between batches. Its answer can depend on batch size and order; it is not merely a different rounding of an exact full solution. If preprocessing scales are also learned from a stream, plan how they are fixed or updated consistently. Do not standardize each batch independently and assume the resulting coordinates have a common meaning.</Prose>
    <Prose>For the small dense examples in this lesson, <Code>svd_solver="full"</Code> makes the choice explicit. Current scikit-learn also supports <Code>"covariance_eigh"</Code>, <Code>"randomized"</Code> and <Code>"arpack"</Code>; the default <Code>"auto"</Code> selects according to shape and requested dimension. It is not a universal randomized-SVD default. For randomized comparisons, set <Code>random_state</Code>; record the solver and package version when reproducibility matters.</Prose>

    <H3>10.7 When the question calls for another method</H3>
    <Prose>PCA’s best reconstruction is linear: the decoded points lie on a flat affine subspace. If observations follow a curved shape, a flat projection can overlap different parts of that shape. A two-dimensional surface rolled through three-dimensional space is a standard example: its intrinsic dimension is two, but flattening it is not the same operation as projecting onto a plane. A neighborhood method may help, provided its assumptions, sampling and parameters support the geometry.</Prose>
    <LessonTable caption="Choose what to investigate next" headers={['need', 'candidate and essential distinction']} rows={[
      ['Keep a subset of actual measurements', 'Feature selection retains original columns; PCA usually mixes many columns in every score.'],
      ['Use a nonlinear similarity with a PCA-like spectral approach', 'Kernel PCA centers a kernel matrix and finds components in its feature space. It supports a forward transform for new points; an approximate inverse is a separate fitted problem, not a prerequisite. Dense kernel storage grows as n².'],
      ['Preserve distances approximately without learning covariance directions', 'Random projections choose a suitable random map. Johnson–Lindenstrauss guarantees concern finite sets, distortion and probability, with target dimension scaling like log(n)/ε²; they do not order coordinates by explained variance.'],
      ['Explore nonlinear neighborhoods', 't-SNE compares neighborhood probability distributions rather than minimizing PCA reconstruction loss. UMAP can transform new observations; it is not limited to static pictures.'],
      ['Seek components with stronger statistical separation', 'ICA targets independence under additional assumptions. Decorrelation alone is weaker.'],
      ['Describe nonnegative data through additive parts', 'NMF imposes nonnegativity rather than PCA’s orthogonality.'],
      ['Learn an encoder and decoder with nonlinear functions', 'An autoencoder trains a reconstruction model. Labels and a GPU are not inherently required; architecture, capacity and evaluation still matter. Its code coordinates need not be ordered principal components.']
    ]} />
    <Prose>The nonlinear methods have their own lesson, <a href="/learn/path/full-curriculum/t-sne-umap-manifold-learning?module=classical-ml">t-SNE, UMAP &amp; Manifold Learning</a>; so do <a href="/learn/path/full-curriculum/independent-component-analysis-ica?module=classical-ml">ICA</a> and <a href="/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml">NMF</a>. There are also variants that change PCA itself: sparse directions can simplify interpretation; robust formulations change how outliers influence the fit; functional PCA treats whole curves as observations; probabilistic PCA introduces a latent-variable and noise model. Each adds a modeling choice. “Robust PCA” in particular can refer to different formulations, including low-rank-plus-sparse decomposition. Use the reference’s actual objective rather than assuming a variant is ordinary PCA with better guarantees.</Prose>
    <Prose><strong>PCA versus regression.</strong> Ordinary least squares predicts a designated response by minimizing vertical response errors. PCA treats the selected feature coordinates symmetrically under the chosen metric and minimizes perpendicular reconstruction errors. A best prediction line and a first principal axis therefore need not coincide. In the four-point example, regressing the second reading on the first gives slope (8/3)/(10/3) = 0.8; PCA’s first axis has slope 1. Their objectives explain the difference.</Prose>

    <H2>{headings[10]}</H2>
    <Prose>PCA learns a coordinate system from centered variation. Scores tell you where observations lie in that system. Keeping fewer scores loses the variation in discarded directions, and reconstruction makes that loss inspectable. Scaling sets the geometry; the purpose sets the acceptable loss.</Prose>
    <LessonTable caption="Readiness check" headers={['you should be able to', 'where it was taught']} rows={[
      ['Explain a direction, a score and a reconstruction using the same point', 'Sections 2 and 4, projection workbench'],
      ['Predict the effect of changing a unit, reversing a direction and discarding a coordinate', 'Sections 2, 5 and 8, metric and label labs'],
      ['Reproduce the four-point result and the changed Wine error budget', 'Sections 4 and 6, practice 4'],
      ['State which data fitted the transform and what evidence your evaluation supplies', 'Section 6 and 10.4'],
      ['Give a case where high explained variance does not preserve a task-relevant distinction', 'Section 8']
    ]} />
    <Prose>The next module topic is <a href="/learn/path/full-curriculum/clustering-evaluation-validation-silhouette-ari-nmi?module=classical-ml">Clustering Evaluation &amp; Validation (Silhouette, ARI, NMI)</a>. After constructing clusters and changing representations, the next question is how to evaluate the grouping. Follow that module sequence; the optional branches above are connections, not substitutions for the next lesson.</Prose>
    <Sources alternatives={<><Prose>Use these after attempting the local examples. The core lesson is self-contained; these offer a second route through the same ideas.</Prose><ul>
      <li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/">Jolliffe &amp; Cadima, “Principal component analysis: a review and recent developments” (2016)</a> — the open canonical survey. Section 2 connects definitions and graphical interpretation; section 3 introduces extensions. Useful when different loading or biplot conventions seem inconsistent. It is a conceptual reference, not current Python API documentation.</li>
      <li><a href="https://www.statlearning.com/resources-python">James, Witten, Hastie, Tibshirani &amp; Taylor, <em>An Introduction to Statistical Learning with Applications in Python</em>, Chapter 12 resources</a> and <a href="https://islp.readthedocs.io/en/stable/labs/Ch12-unsup-lab.html#principal-components-analysis">the PCA notebook</a> — another hands-on route through scaled data, scores, biplots and variance plots. The notebook uses additional packages and datasets; consult its version instructions rather than replacing your environment. The book itself is free from <a href="https://www.statlearning.com/home">the authors’ site</a>.</li>
      <li><a href="https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-19-video/">Philippe Rigollet, MIT 18.650, Lecture 19: Principal Component Analysis</a> and <a href="https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/d85e1a9d113142ade8ce5e4f5ef0b4e8_MIT18_650F16_PCA.pdf">companion slides</a> — a mathematical alternative after section 10.1, connecting projected variance and eigenvectors. The slides use a 1/n empirical covariance convention; this lesson uses 1/(n − 1), so directions and ratios agree while absolute variances differ by that factor. The slides were reviewed for this recommendation; the full video was not watched.</li>
      <li><a href="https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html">Scikit-learn, “Importance of Feature Scaling”</a> — a worked Wine example of how scale affects PCA and downstream estimators. Read after sections 5 and 6 and note that its protocol differs from the split used here.</li>
    </ul></>}>
      <li><a href="https://archive.ics.uci.edu/dataset/109/wine">Aeberhard &amp; Forina, Wine dataset, UCI record 109</a>, DOI <a href="https://doi.org/10.24432/C5PC7J">10.24432/C5PC7J</a>, licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>, record and license inspected on 12 September 2026. The values used everywhere here are the mirror bundled with scikit-learn 1.9.1 through <Code>load_wine</Code>; a direct UCI download was not used. The downloadable CSV puts the cultivar first as 1, 2, 3, adds a header from the loader’s feature names, keeps all 178 rows in bundled order and applies no filtering, imputation or rescaling. Units are as supplied; do not infer vintage, quality or clinical meaning from column names.</li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html">NumPy SVD</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">PCA</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a> and the <a href="https://scikit-learn.org/stable/modules/decomposition.html">decomposition guide</a> — factor orientation, fit/transform semantics, solver names and the <Code>ddof</Code> conventions checked above; documentation may describe a newer release than the tested 1.9.1.</li>
      <li><a href="https://scikit-learn.org/stable/common_pitfalls.html">Scikit-learn’s common pitfalls</a> — fitted preprocessing and fold boundaries, the reference for section 6’s leakage rule.</li>
      <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html">KernelPCA</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html">IncrementalPCA</a>, <a href="https://scikit-learn.org/stable/modules/random_projection.html">random projection guide</a> and <a href="https://umap-learn.readthedocs.io/en/latest/transform.html">UMAP’s transform tutorial</a> — the API facts behind section 10.7’s comparison table.</li>
      <li><a href="https://arxiv.org/abs/0909.4061">Halko, Martinsson &amp; Tropp (2011), “Finding Structure with Randomness”</a> — randomized low-rank factorization and the assumptions behind its approximation bounds. Read the algorithm overview before the proofs; it supplies no machine-independent timing benchmark.</li>
    </Sources>
    <Prose>The small point clouds, projection calculations, sensor scenarios and practice problems in this lesson are constructed teaching examples. Wine results are calculations on the identified real dataset. The Gaussian spectrum is a seeded simulation. None is a hardware benchmark or a claim about every future dataset.</Prose>
  </div>
};

export default pcaContent;
