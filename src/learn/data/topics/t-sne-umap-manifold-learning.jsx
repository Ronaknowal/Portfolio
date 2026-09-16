import { H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { ManifoldRouteFigure, ManifoldNeighborFigure, ManifoldProbabilityFigure, ManifoldForceFigure, ManifoldFuzzyGraphFigure, ManifoldDigitsFigure, ManifoldMdsFigure, ManifoldLleFigure, ManifoldTopologyFigure } from '../../components/lesson-labs/ManifoldFigures.jsx';
import { ManifoldGraphLab, ManifoldProbabilityLab, ManifoldFuzzyLab, ManifoldDigitLab } from '../../components/lesson-labs/ManifoldLabs.jsx';
import { manifoldExamples } from '../manifold-examples.js';
import '../../components/lesson-labs/manifold-lesson.css';

const headings = [
  "1. Can a picture help us inspect handwritten digits?",
  "2. A short route can leave the surface",
  "3. Decide what a map should preserve",
  "4. t-SNE: turn a neighborhood into a probability row",
  "5. t-SNE: move the map to match those preferences",
  "6. UMAP: construct a graph, then optimize a layout",
  "7. Inspect real digits and measure what survives",
  "8. Use the representation for its intended task",
  "9. Deeper branch: distances, reconstruction weights and graph eigenvectors",
  "10. Deeper branch: derive and run a tiny exact t-SNE optimizer",
  "11. Practice: make a changed choice and explain the result",
  "12. References & another way to learn it",
  "13. Ready to move on?"
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example} />
  </section>;
}

const manifoldContent = {
  title: 't-SNE, UMAP & Manifold Learning',
  readTime: '~60 min first pass · deeper branches + 60–90 min investigations and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot manifold-lesson">
    <LessonIntro prerequisites="Vector distances, weighted averages, basic probabilities and the idea of following a gradient. PCA supplies the baseline; neighbor graphs and entropy are introduced here." sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      Build a route, redistribute neighbor preferences, inspect a fuzzy connection and audit real handwritten images. Measure what survives when many measurements become a two-dimensional map, then return for the mathematical derivations.
    </LessonIntro>
    <H2>{headings[0]}</H2>

    <Prose>{"Imagine opening a spreadsheet with 300 handwritten digits. Each row contains 64 measurements: the darkness in an 8 × 8 grid. You want to inspect which writing styles resemble one another, locate unusual examples, and notice where different digits share similar strokes. A table of 19,200 numbers makes that difficult. A two-dimensional map could make it manageable—if you can find out which relationships the map kept."}</Prose>

    <Prose>{"PCA gives a useful starting map by retaining two directions of large variation. This lesson explores methods that start from relationships between observations instead: distances along a surface, local reconstruction weights, or weighted neighbor connections. "}<strong>{"An embedding"}</strong>{" is the resulting collection of lower-dimensional coordinates. There are still 300 observations; each now has two coordinates for display."}</Prose>

    <Prose><strong>{"First-pass route:"}</strong>{" read sections 1–8, trying the neighborhood, affinity and map-audit investigations as you reach them. Run the digits program in section 7 and attempt practice 1–4 in section 11. That route takes you from a geometric idea to a measured answer about a real map. Return to section 9 for Isomap, MDS and LLE calculations, and section 10 for the t-SNE derivation and exact optimizer. Those deeper branches preserve the broader manifold-learning scope. Allow roughly 50–65 minutes for the core reading and 60–90 minutes for investigations and practice; the deeper branches add another sitting."}</Prose>

    <Prose>{"You need distances between numeric vectors, a weighted average, basic probabilities, and the idea of following a loss gradient downhill. "}<a href={"/learn/path/full-curriculum/pca-dimensionality-reduction?module=classical-ml"}>{"PCA & Dimensionality Reduction"}</a>{" supplies the baseline and scaling discussion. We introduce neighbor graphs and entropy here; no topology course is required. Eigenvalues enter only in the deeper branch, with a short bridge there."}</Prose>

    <Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml"}>{"Gaussian Mixture Models (GMM) & EM Algorithm"}</a>{" fits a probability density to observations. A t-SNE picture answers a different question: which relationships can we expose in a small coordinate system? Its neighbor probabilities are weights used to construct a map, not mixture responsibilities or a probability density over future digit images."}</Prose>

    <H2>{headings[1]}</H2>

    <Prose>{"A piece of paper is locally two-dimensional even after you curl it in three-dimensional space. Moving along the paper needs two coordinates; describing a point's position in the room needs three. The first count is its "}<strong>{"intrinsic dimension"}</strong>{", the second its "}<strong>{"ambient dimension"}</strong>{". A smooth manifold behaves like a flat space within sufficiently small neighborhoods. Real measurements may only approximately follow such a structure: noise, discrete classes and unobserved factors need not form one clean surface."}</Prose>

    <Prose>{"The "}<strong>{"manifold hypothesis"}</strong>{" is a modeling idea that many measured variables can vary through fewer underlying degrees of freedom. For a camera observing one rigid object turning on a fixed axis, angle is a possible latent coordinate. If lighting and camera position also change, they introduce more variation. Merely declaring images “a manifold” does not tell us its dimension or supply a good image distance."}</Prose>

    <Prose>{"Here is the distance problem in a form we can calculate. Put seven points on a U-shaped path:"}</Prose>

    <LessonTable caption={"Seven observations on a U-shaped path"} headers={["Point", "A", "B", "C", "D", "E", "F", "G"]} rows={[
      [<>{"Coordinate"}</>, <>{"(0,0)"}</>, <>{"(0,1)"}</>, <>{"(0,2)"}</>, <>{"(1,2)"}</>, <>{"(2,2)"}</>, <>{"(2,1)"}</>, <>{"(2,0)"}</>]
    ]} />

    <Prose>{"A and G are 2 units apart across the opening. Following the six unit segments of the U takes 6 units. The second quantity is the distance along this particular path. For a smooth surface, the shortest allowed route along it is called a "}<strong>{"geodesic"}</strong>{"."}</Prose>

    <ManifoldRouteFigure />

    <Prose>{"A finite dataset does not give us every point on the surface. We construct a "}<strong>{"neighbor graph"}</strong>{": observations are vertices, chosen neighbor pairs are edges, and each edge carries its input distance. Adding edge lengths along a shortest graph path approximates a geodesic when the sampling and chosen connections support that interpretation."}</Prose>

    <Prose>{"In the U example, connect distinct points whose distance is at most a radius ε. At ε = 1, only the six consecutive unit edges appear, so the A–G graph distance is 6. At ε = 2, the graph contains the direct A–G edge and the distance becomes 2. At ε = 0.75, no distinct points connect; no finite A–G graph distance exists. These are three different graphs built from exactly the same observations."}</Prose>

    <Prose><strong>{"Investigation A — Build a route before flattening it."}</strong>{" Choose endpoints, edit one point's coordinates, and commit your prediction about whether a proposed radius leaves a route and how its length changes. Apply the radius, trace the resulting shortest path, then explain which edge caused your result. Try an unfamiliar coordinate edit, not only the supplied U."}</Prose>

    <ManifoldGraphLab />

    <Prose>{"This is the central choice behind Isomap: use estimated surface distances before finding coordinates. It also explains why “more neighbors” is not an automatic improvement. Additional edges can repair a disconnected graph or introduce shortcuts. A Swiss roll—the familiar sheet curled into a spiral—has the same problem across adjacent layers. PCA fits one linear projection; Isomap attempts to reconstruct distance along the sampled sheet. PCA still has a well-defined reconstruction objective on curved data, even when two components are a poor map of that surface."}</Prose>

    <H2>{headings[2]}</H2>

    <Prose>{"Methods called dimensionality reduction can optimize quite different quantities."}</Prose>

    <LessonTable caption={"What each dimensionality-reduction method tries to preserve"} headers={["Method", "What the computation tries to retain", "What you can inspect"]} rows={[
      [<>{"PCA"}</>, <>{"A low-rank linear reconstruction with small squared error"}</>, <>{"Variance retained, residuals, loadings"}</>],
      [<>{"Classical MDS"}</>, <>{"Inner products obtained from a dissimilarity matrix"}</>, <>{"Distance reconstruction and negative eigenvalues"}</>],
      [<>{"Isomap"}</>, <>{"Graph shortest-path distances, followed by classical MDS"}</>, <>{"Connectivity, shortcuts, geodesic distortion"}</>],
      [<>{"LLE"}</>, <>{"Each point's weights for reconstruction from nearby points"}</>, <>{"Neighbor choice and reconstructed positions"}</>],
      [<>{"t-SNE"}</>, <>{"A normalized distribution of pairwise neighbor affinities"}</>, <>{"Affinity mismatch and neighborhood preservation"}</>],
      [<>{"UMAP"}</>, <>{"A weighted neighbor graph followed by a sampled attractive/repulsive layout procedure"}</>, <>{"Graph construction, layout changes and neighborhood preservation"}</>]
    ]} />

    <Prose>{"You will learn the two common visualization methods first, then work the distance and reconstruction methods in section 9. This table is a comparison of objectives, not a numerical ranking."}</Prose>

    <Prose><strong>{"The map-reading contract."}</strong>{" This paragraph is the lesson's home for interpreting a learned layout. A near pair on the map can be a false neighbor; an input neighbor can be torn apart. Axis directions, area, inter-island spacing and visual density have no general measurement-unit interpretation in t-SNE or ordinary UMAP. Their locally adaptive input affinities help explain why original densities are changed. Count selected observations to measure how many examples they contain; the member count is retained even when the group's area changes. To assess distance, variability, density or clusters in the original representation, return to that representation and a defined diagnostic. A colored island is a finding to inspect, not a class or density model supplied by the embedding."}</Prose>

    <Prose>{"Two checks will make that contract operational. Fix a neighbor count k, exclude each point itself, and choose a rule for equal distances. Let N_X(i) be point i's k nearest input neighbors and N_Y(i) its k nearest map neighbors. The "}<strong>{"neighbor retention"}</strong>{" is"}</Prose>

    <MathBlock>{"R_k=\\frac{1}{nk}\\sum_{i=1}^n |N_X(i)\\cap N_Y(i)|."}</MathBlock>

    <Prose>{"It is the fraction of directed neighbor selections that survive. If 7 of a query's 10 original neighbors remain among its 10 plotted neighbors, its retention is 0.7. The three new neighbors are false neighbors; the three displaced ones are missing neighbors. Averaging can hide a particularly poor query, so inspect both local lists and the collection average."}</Prose>

    <ManifoldNeighborFigure />

    <Prose>{"For an exact example use input positions "}<Code>{"[0,1,3,7]"}</Code>{" for A,B,C,D and proposed map positions "}<Code>{"[0,5,1,11]"}</Code>{". With k = 1, the input selections are A→B, B→A, C→B, D→C. The map selects A→C, B→C, C→A, D→B. None survives: R₁ = 0. The picture still consists of four perfectly ordinary points."}</Prose>

    <Prose><strong>{"Trustworthiness"}</strong>{" gives a larger penalty when a false map neighbor was far down the input ranking. If r_X(i,j) is j's input rank from i, and U_k(i) = N_Y(i) \\ N_X(i), then the usual normalization is"}</Prose>

    <MathBlock>{"T_k=1-\\frac{2}{nk(2n-3k-1)}\\sum_i\\sum_{j\\in U_k(i)}(r_X(i,j)-k)."}</MathBlock>

    <Prose>{"Here use 1 ≤ k < n/2, as required by the scikit-learn function. In the four-point example every false neighbor has rank 2. The penalty sum is 4, the multiplier is 1/8, and T₁ = 0.5. "}<strong>{"Continuity"}</strong>{" reverses the roles of X and Y to penalize missing input neighbors. Both equal 1 on a distance-preserving, tie-free map. Rigidly rotating a plot does not change any pairwise distance, so it should not be interpreted as instability. Rankings with exact distance ties require a stated tie rule; our visible neighbor lists use source-row order, while the library score uses its own distance sorting. Small tie-boundary differences can therefore occur."}</Prose>

    <Prose>{"These diagnostics assess the chosen metric and k. The formulas, original-space records and query images turn an impression about a map into an inspectable claim. "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html"}>{"Trustworthiness API and reference"}</a>{"."}</Prose>

    <H2>{headings[3]}</H2>

    <Prose>{"For each input xᵢ, t-SNE asks: if i chooses another observation as its neighbor, how should its preference decline with distance? It uses a Gaussian-shaped weight, then normalizes all candidates in that row:"}</Prose>

    <MathBlock>{"p_{j|i}=\\frac{\\exp(-\\|x_i-x_j\\|^2/(2\\sigma_i^2))}\n{\\sum_{l\\ne i}\\exp(-\\|x_i-x_l\\|^2/(2\\sigma_i^2))},\\quad p_{i|i}=0."}</MathBlock>

    <Prose>{"The bandwidth σᵢ controls how quickly preference falls. Consider three other points at distances 1,2,3 from i, and σᵢ = 1. Their unnormalized weights are e⁻⁰·⁵, e⁻², e⁻⁴·⁵. Dividing by their sum gives approximately "}<Code>{"[0.805512, 0.179734, 0.014753]"}</Code>{". The closest candidate receives about 80.6% of this row's preference. The row is a distribution over neighbor choices, not the probability that the measurement was generated by a component."}</Prose>

    <ManifoldProbabilityFigure />

    <Prose>{"Instead of using one σ for every point, t-SNE chooses bandwidths to reach a target "}<strong>{"perplexity"}</strong>{". For a probability row p, entropy measures its spread:"}</Prose>

    <MathBlock>{"H(p)=-\\sum_j p_j\\log_2 p_j,\\qquad \\operatorname{Perp}(p)=2^{H(p)}."}</MathBlock>

    <Prose>{"Equal preference over m candidates gives entropy log₂m and perplexity m. The uneven row above has perplexity about 1.7244. It has three positive probabilities but concentrates preference enough to behave, in this entropy sense, like fewer than two equally likely choices. Perplexity is an effective count, not a cutoff after an exact number of neighbors."}</Prose>

    <Prose>{"For this row, σ = 0.5 gives perplexity 1.0175; σ = 2 gives 2.7864. A binary search can adjust σ until the desired entropy is reached. In a sparse region, the needed σ can be larger than in a dense region. This is one mechanism behind the map-reading contract's density issue."}</Prose>

    <Prose><strong>{"Investigation B — Change who receives the probability."}</strong>{" Edit the three candidate distances and commit a prediction about the closest candidate's probability or the change in perplexity before applying a bandwidth. The equal-distance fixture is a useful test: all three candidates get 1/3 for every positive bandwidth, so no bandwidth can make its perplexity 2. A tie among m equally closest candidates likewise places a lower limit m on achievable perplexity as bandwidth approaches zero. A search tolerance cannot create information that the distances do not contain."}</Prose>

    <ManifoldProbabilityLab />

    <Prose>{"Conditional rows need not agree: i may strongly prefer j while j has several even closer candidates. Standard symmetric t-SNE combines them as"}</Prose>

    <MathBlock>{"p_{ij}=\\frac{p_{j|i}+p_{i|j}}{2n}."}</MathBlock>

    <Prose>{"P is symmetric, its diagonal is zero, and its entries sum to 1 over "}<strong>{"ordered"}</strong>{" distinct pairs. For example, with n = 4, p_{B|A}=0.8 and p_{A|B}=0.6 contribute p_AB = p_BA = 1.4/8 = 0.175. The division by 2n averages the two directions and all rows. These conventions explain the factor 4 in the gradient later. "}<a href={"https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf"}>{"Original t-SNE paper, sections 2–3"}</a>{"."}</Prose>

    <H2>{headings[4]}</H2>

    <Prose>{"Each input observation i has a map position yᵢ. In a two-dimensional map, yᵢ has two coordinates. t-SNE gives each pair a heavy-tailed weight"}</Prose>

    <MathBlock>{"t_{ij}=\\frac{1}{1+\\|y_i-y_j\\|^2},\\qquad\nq_{ij}=t_{ij}/Z,\\qquad Z=\\sum_{k\\ne l}t_{kl},\\quad t_{ii}=q_{ii}=0."}</MathBlock>

    <Prose>{"The Student-t kernel with one degree of freedom falls much more slowly than a Gaussian. At distance 3 its weight is 1/10; the Gaussian weight with σ = 1 is about 0.0111. This heavier tail gives the layout more room to represent moderately related points at separated positions. It addresses the "}<strong>{"crowding problem"}</strong>{": many neighbors at moderate input distances cannot all occupy corresponding nearby locations in a low-dimensional space. It does not remove the need to compromise."}</Prose>

    <Prose>{"t-SNE minimizes"}</Prose>

    <MathBlock>{"C=\\mathrm{KL}(P\\|Q)=\\sum_{i\\ne j}p_{ij}\\log(p_{ij}/q_{ij})."}</MathBlock>

    <Prose>{"P is fixed after input affinities are built; moving Y changes Q. A large p paired with a tiny q is expensive: the map gives too little preference to an important input pair. KL is asymmetric. That emphasis does not eliminate repulsion—every q shares the denominator Z. Moving a low-p pair close also takes probability mass away from other pairs."}</Prose>

    <ManifoldForceFigure />

    <Prose>{"The gradient for standard symmetric t-SNE is"}</Prose>

    <MathBlock>{"\\nabla_{y_i}C=4\\sum_{j\\ne i}(p_{ij}-q_{ij})t_{ij}(y_i-y_j)."}</MathBlock>

    <Prose>{"Gradient descent subtracts a step-size times this vector. If pᵢⱼ > qᵢⱼ, that pair's contribution pulls i toward j. If pᵢⱼ < qᵢⱼ, it pushes i away from j. Many pair contributions add, so the total motion need not point at the pair you selected. Section 10 derives the expression and runs a complete tiny optimizer."}</Prose>

    <Prose><strong>{"Early exaggeration"}</strong>{" temporarily multiplies the attractive p term by a factor α > 1. Repulsion still uses Q. This helps build local groups before the ordinary refinement phase. The exaggerated matrix has sum α, so it is not a probability distribution; do not treat an exaggerated training diagnostic and the final normalized KL as points on one unchanged objective curve. A large step size or exaggeration can also produce poor optimization. Plain gradient descent need not decrease the loss at every step."}</Prose>

    <Prose>{"The objective is nonconvex: starting positions and optimization settings can lead to different answers. PCA initialization is a useful starting choice; a seed and recorded environment support repeatable experiments. Reproducibility means repeating the same computation, while robustness means checking whether meaningful neighborhoods survive deliberate changes. Neither requires two plots to have the same orientation. These are separate questions."}</Prose>

    <Prose>{"The following settings are operational choices, not rules for discovering the “correct” number of clusters:"}</Prose>

    <LessonTable caption={"t-SNE parameter choices and practical checks"} headers={["Setting", "Role and practical check"]} rows={[
      [<><Code>{"perplexity"}</Code></>, <>{"Input affinity scale. Must be positive and less than n in the API; exact row support/ties further affect what entropy is achievable. Compare several valid values."}</>],
      [<><Code>{"init"}</Code></>, <>{"Starting map; "}<Code>{"'pca'"}</Code>{", "}<Code>{"'random'"}</Code>{", or supplied coordinates. Changing a seed may have no visible effect if the selected initialization uses no effective randomness."}</>],
      [<><Code>{"learning_rate"}</Code>{", "}<Code>{"early_exaggeration"}</Code></>, <>{"Step scale and temporary attraction. Inspect optimization rather than choosing by visual separation."}</>],
      [<><Code>{"max_iter"}</Code></>, <>{"Iteration budget, including the early phase; complete the refinement phase before interpreting a map."}</>],
      [<><Code>{"method"}</Code>{", "}<Code>{"angle"}</Code></>, <>{"Exact pair sums versus a tree approximation; angle controls approximation, not a display rotation."}</>]
    ]} />

    <Prose>{"In scikit-learn 1.9.1, "}<Code>{"learning_rate='auto'"}</Code>{" is "}<Code>{"max(n / early_exaggeration / 4, 50)"}</Code>{". For n = 300 and α = 12, it is 50, not 25 or 6.25. Its learning-rate convention differs by a factor of four from several other t-SNE implementations; copy the convention with the value. "}<Code>{"max_iter"}</Code>{" replaced the old name "}<Code>{"n_iter"}</Code>{" in version 1.5. "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"}>{"TSNE API"}</a>{"."}</Prose>

    <H2>{headings[5]}</H2>

    <Prose>{"UMAP starts from a local neighbor graph. Around each observation, it rescales distance relative to local spacing; nearby connections become strong and more distant connections become weak. It then combines the directed connections into one weighted graph and finds map coordinates through attractive and sampled repulsive updates. The graph and the coordinates are different objects."}</Prose>

    <Prose>{"For the ordinary local-connectivity setting, let ρᵢ be the distance to i's nearest positive-distance neighbor. A directed membership strength has the form"}</Prose>

    <MathBlock>{"v_{ij}=\\exp\\{-\\max(0,d(x_i,x_j)-\\rho_i)/\\sigma_i\\}"}</MathBlock>

    <Prose>{"for retained neighbors, and zero otherwise. Thus a retained neighbor at or inside ρᵢ gets strength 1. σᵢ is fitted to a local neighbor-mass target. For the implementation's usual training neighbor arrays, which include self at index zero, "}<Code>{"smooth_knn_dist"}</Code>{" sums the nonself memberships and targets log₂("}<Code>{"n_neighbors"}</Code>{"). Keep this self-count convention explicit when translating a diagram into code. Nondefault "}<Code>{"local_connectivity"}</Code>{", duplicate points, limited support and the library's numerical floors require the actual implementation rather than blindly applying “ρ is the smallest array entry.” "}<a href={"https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html"}>{"UMAP graph-construction source"}</a>{"."}</Prose>

    <Prose>{"A small calibration example makes that target concrete. Take a sorted training neighbor-distance row "}<Code>{"[0, 1, 1+ln 2, 1+ln 2]"}</Code>{", including self. With "}<Code>{"n_neighbors=4"}</Code>{", the target is log₂4=2. Here ρ=1 and σ=1 give the three nonself strengths "}<Code>{"1, 1/2, 1/2"}</Code>{", whose sum is exactly 2. Self is not one of those three graph edges. This constructed row demonstrates local calibration; it is not a claim about the distances of the digits."}</Prose>

    <Prose>{"UMAP's fuzzy union combines both directions:"}</Prose>

    <MathBlock>{"w_{ij}=v_{ij}+v_{ji}-v_{ij}v_{ji}."}</MathBlock>

    <Prose>{"With vᵢⱼ = 1/2 and vⱼᵢ = 1/4, the combined strength is 1/2 + 1/4 − 1/8 = 5/8. Taking the arithmetic average would instead give 3/8. If only one directed edge exists with strength 1/2, the union remains 1/2. If either direction has strength 1, the union has strength 1. These are membership strengths defined by this graph construction; a value of 0.625 is not an empirically calibrated 62.5% chance of a “true” topological connection."}</Prose>

    <ManifoldFuzzyGraphFigure />

    <Prose><strong>{"Investigation C — Build and inspect one fuzzy connection."}</strong>{" Edit distances and local scales for two neighborhoods. Predict the merged edge strength before applying the edit. Then inspect an idealized single-pair attraction/repulsion cost as you choose a candidate map separation. One control changes graph input; another changes the map. They must not silently overwrite each other."}</Prose>

    <ManifoldFuzzyLab />

    <Prose>{"In the map, UMAP uses a smooth similarity"}</Prose>

    <MathBlock>{"\\nu_{ij}=(1+a\\|y_i-y_j\\|^{2b})^{-1}."}</MathBlock>

    <Prose>{"The positive parameters a,b are fitted from "}<Code>{"min_dist"}</Code>{" and "}<Code>{"spread"}</Code>{". "}<Code>{"min_dist"}</Code>{" changes the preferred small-distance shape; it is not a hard exclusion radius that guarantees all output distances exceed it. "}<Code>{"spread"}</Code>{" sets a companion scale. "}<Code>{"n_neighbors"}</Code>{" changes the graph's input neighborhood scale, whereas "}<Code>{"min_dist"}</Code>{" changes layout behavior after the graph is constructed. UMAP can target more than two or three coordinates. For the ordinary API, choose integer "}<Code>{"n_neighbors"}</Code>{" at least 2 and smaller than the available training collection, positive integer "}<Code>{"n_components"}</Code>{", positive "}<Code>{"spread"}</Code>{", and "}<Code>{"0 ≤ min_dist ≤ spread"}</Code>{". Too-large neighborhood requests may be truncated by the implementation; choose the intended size explicitly, and leave enough samples for spectral initialization in the requested output dimension. "}<a href={"https://umap-learn.readthedocs.io/en/latest/parameters.html"}>{"Parameter tutorial"}</a>{"."}</Prose>

    <Prose>{"The all-pairs formulation motivates a binary cross-entropy cost, omitting terms constant in Y:"}</Prose>

    <MathBlock>{"L_{\\rm full}=-\\sum_{i<j}[w_{ij}\\log\\nu_{ij}+(1-w_{ij})\\log(1-\\nu_{ij})]."}</MathBlock>

    <Prose>{"The sum includes absent graph edges, whose w is zero: omitting them also omits repulsion. This expression is cross-entropy; it is not a symmetric divergence. Subtracting each pair's fixed Bernoulli entropy term gives a sum of Bernoulli KL divergences, again directed from input weights to output similarities. That subtracted term depends only on the input graph, so it does not change the minimizing layout."}</Prose>

    <Prose>{"For a transparent single-pair example, set a = b = 1 and w = 5/8. At separation r = 1, ν = 1/2 and the cost is log 2 ≈ 0.693147. At r = 2, ν = 1/5 and the cost is about 1.089578. The ideal single-pair minimum occurs at ν = w, hence r = √(1/w − 1) = √(3/5) ≈ 0.774597. This follows by differentiating "}<Code>{"−w log ν − (1−w) log(1−ν)"}</Code>{". It explains the opposing terms; an entire graph cannot generally give every pair its individual optimum."}</Prose>

    <Prose><strong>{"Actual optimization is sampled."}</strong>{" The ordinary implementation schedules positive edges according to their weights, applies attraction, and samples other vertex indices for repulsion. Sampled vertices are not guaranteed to be non-neighbors. Positive edge scheduling already incorporates the weight; multiplying it by w again in every scheduled attraction step would change the procedure. Negative sampling also changes effective attraction/repulsion weighting: it is not merely an unbiased, faster evaluation of the displayed full-pair cost. This distinction is analyzed by "}<a href={"https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf"}>{"Damrich & Hamprecht, sections 4–6"}</a>{". Our ideal pair calculation is deliberately labeled as that calculation, not as the trajectory of the library's optimizer."}</Prose>

    <Prose>{"Here is the algorithm's causal order, without pretending it is runnable Python:"}</Prose>

    <CodeBlock language={"text"}>{"Choose an input metric and neighborhood size.\nFind neighbors; fit local rho and sigma; form directed memberships.\nMerge reciprocal memberships into the weighted graph.\nInitialize map coordinates, commonly from a spectral graph embedding.\nSchedule positive edges using their membership strengths.\nFor each scheduled edge: attract its endpoint coordinates.\nFor sampled other vertices: apply repulsive updates.\nDecrease the step size over the epoch schedule; return the coordinates."}</CodeBlock>

    <Prose>{"UMAP's topological construction motivates this procedure. Moving a finite graph into two dimensions adds graph estimation and optimization choices, so the map-reading contract still applies. The original mathematical framework is a deeper reading, not a guarantee about every practical plot. "}<a href={"https://arxiv.org/html/1802.03426v3"}>{"UMAP paper, algorithm and theoretical framework"}</a>{"."}</Prose>

    <H2>{headings[6]}</H2>

    <Prose>{"The supplied "}<a href={"/learn-assets/manifold-learning/digits-300.csv"}>{"digits-300.csv"}</a>{" contains 64 block-darkness measurements and a digit label per observation. E. Alpaydin and C. Kaynak's "}<strong>{"Optical Recognition of Handwritten Digits"}</strong>{" collection was created to study handwritten digit recognition. The measurements are integers from 0 to 16. This is the UCI optical-digits dataset distributed through scikit-learn, not MNIST. "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI dataset, provenance and CC BY 4.0 license"}</a>{"."}</Prose>

    <Prose>{"We select the first 30 available rows of each label from "}<Code>{"load_digits"}</Code>{", merge them in original row order, and retain that source-row identifier. The labels therefore influence "}<strong>{"selection"}</strong>{". They are never passed to the embedding fit. The balanced subset is convenient for inspection; its proportions describe this selected collection rather than handwriting in a population. The full provenance and exact selection recipe are in "}<a href={"/learn-assets/manifold-learning/data-provenance.md"}>{"data-provenance.md"}</a>{"."}</Prose>

    <Prose>{"Our question is specific: "}<strong>{"does a nonlinear map preserve more of these 300 images' ten nearest pixel-distance neighbors than a two-component PCA map?"}</strong>{" All pixel features share a 0–16 scale. Dividing every feature by 16 preserves Euclidean rankings and puts values in 0–1 units; it does not learn scales from the data. Separately standardizing every pixel would change the metric by giving a quiet corner pixel the same variance scale as an active center pixel. We keep the common pixel scale for this question."}</Prose>

    <Prose>{"This is an exploratory map of one fixed collection. All 300 observations can participate in its fit. Section 8 changes the protocol when the purpose is predicting future observations."}</Prose>

    <Prose>{"Save this complete program as "}<Code>{"inspect_digits.py"}</Code>{" beside the supplied CSV. Install NumPy and scikit-learn into your own Python environment if needed, then run "}<Code>{"python inspect_digits.py"}</Code>{". The content probes used Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1. Matplotlib is not required for the numeric result. To reproduce that numerical environment in your own environment:"}</Prose>

    <CodeBlock language={"sh"}>{"python -m pip install numpy==2.3.5 scikit-learn==1.9.1\npython inspect_digits.py"}</CodeBlock>

    <Program example={manifoldExamples.digitsAudit} />

    <Prose>{"The complete program was executed with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1, reproducing the saved coordinates. The recorded run used 12 BLAS/OpenMP threads: "}<Code>{"n_jobs=1"}</Code>{" in TSNE does not constrain every numerical thread pool. Library builds and thread settings can change a nonlinear fit, so keep the saved coordinates for an exact map comparison. Recorded rounded results:"}</Prose>

    <ManifoldDigitsFigure />

    <Prose>{"For this collection and metric, all three t-SNE settings retain more of the original ten-neighbor selections than PCA in two dimensions. Perplexity 30 retains about 77.0%, so roughly 23.0% of directed ten-neighbor selections change. T₁₀ ≈ 0.9901 is high despite that difference: many replacements were not extremely remote in input rank. This is why the rank-weighted score and the direct retention fraction answer complementary questions."}</Prose>

    <Prose><strong>{"Investigation D — Audit an image's neighbors."}</strong>{" Choose an image before choosing its map. Predict how many of its k input neighbors a candidate map will retain, then reveal the image tiles and identity-matched neighbor edges. Change k to 5 or 20 and explain whether your previous conclusion still applies. A good investigation can find an image for which a globally stronger map has lower local retention. That query is an opportunity to inspect the data rather than a reason to hide it."}</Prose>

    <ManifoldDigitLab />

    <Prose>{"The saved comparisons also include seeds 7 and 19. With the specified PCA initialization these two seeds produced identical coordinates in the recorded environment: a useful null result, not a promise that every seed always changes a plot. A separate random-initialization pair at perplexity 30 supplies an actual alternative-initialization comparison. The plots use the saved coordinates and their measured results. Relocating or jittering individual observations would change their distances; a common rigid rotation preserves distances and neighbor selections."}</Prose>

    <H3>{"Try UMAP on the same offline collection"}</H3>

    <Prose>{"Save the following complete program as "}<Code>{"inspect_umap.py"}</Code>{" beside the CSV. It uses the same X and the same diagnostic. Install the package named "}<strong><Code>{"umap-learn"}</Code></strong>{", which provides the import "}<Code>{"umap"}</Code>{":"}</Prose>

    <CodeBlock language={"sh"}>{"python -m pip install umap-learn==0.5.12\npython inspect_umap.py"}</CodeBlock>

    <Prose>{"This complete program was executed with UMAP 0.5.12, Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1, using one numerical thread. It produced 300 two-dimensional coordinates, R₁₀ = 0.7143 and T₁₀ = 0.9890. These are measurements for this selection, metric and configuration, not a universal ranking of UMAP against t-SNE. Record your installed versions and thread settings when comparing a new fit."}</Prose>

    <Program example={manifoldExamples.umapFit} />

    <Prose>{"The output shape is "}<Code>{"(300, 2)"}</Code>{"; compare your measured scores with the recorded run. After the first fit, change just one parameter: try "}<Code>{"n_neighbors"}</Code>{" 5 or 50 while keeping the rest fixed, or "}<Code>{"min_dist"}</Code>{" 0.0 or 0.5 while keeping "}<Code>{"n_neighbors=15"}</Code>{". Compare the same neighbor audit. The graph can change in the first experiment; it stays fixed in the second if the neighbor search and other inputs are unchanged. Recording a seed can reduce parallel execution in UMAP; its reproducibility documentation explains the tradeoff. "}<a href={"https://umap-learn.readthedocs.io/en/latest/reproducibility.html"}>{"UMAP reproducibility"}</a>{"."}</Prose>

    <H2>{headings[7]}</H2>

    <H3>{"A map of this collection or a transform for future images?"}</H3>

    <Prose>{"A fixed-collection map can be fitted to every displayed row. A supervised evaluation asks a different question: how well will a model handle observations unavailable during fitting? Split first, fit preprocessing and the reducer only on the training portion, apply those fitted operations to validation/test rows, and evaluate the whole pipeline. Fit a separate scaler on new rows and you change the meaning of their coordinates."}</Prose>

    <Prose>{"Ordinary "}<Code>{"umap-learn"}</Code>{" supports "}<Code>{"transform(X_new)"}</Code>{". It finds relationships from new points to the fitted training data, initializes their coordinates using that reference, and refines them with the fitted training map held fixed. It is an out-of-sample procedure, not a learned neural encoder. A far-out new image can be poorly represented if the fitted reference supplies no relevant neighbors. Parametric UMAP instead learns an encoder; it still requires evaluation on the intended future distribution. "}<a href={"https://umap-learn.readthedocs.io/en/latest/transform.html"}>{"UMAP transform tutorial"}</a>{", "}<a href={"https://arxiv.org/abs/2009.12981"}>{"Parametric UMAP paper"}</a>{"."}</Prose>

    <Prose>{"Scikit-learn's "}<Code>{"TSNE"}</Code>{" has no "}<Code>{"transform"}</Code>{" method. Other t-SNE implementations offer reference-map extensions, and parametric variants learn mappings, so keep that limitation tied to the implementation rather than the entire family. For example, "}<a href={"https://opentsne.readthedocs.io/en/stable/examples/01_simple_usage/01_simple_usage.html"}>{"openTSNE's new-point tutorial"}</a>{" explains its reference embedding. Independently fitting t-SNE on test rows does not put those rows into the training map's coordinates."}</Prose>

    <Prose>{"Here is a complete UMAP train/validation example using the same CSV. The label is passed only to the classifier. Calling "}<Code>{"UMAP.fit_transform(X_train, y_train)"}</Code>{" would invoke label-informed behavior; this example deliberately does not do so. All three methods receive the same stratified 225/75 split; labels in this split and in the collection's original selection are disclosed. Save the program as "}<Code>{"compare_transforms.py"}</Code>{" beside the CSV and run "}<Code>{"python compare_transforms.py"}</Code>{" after the UMAP setup above."}</Prose>

    <Program example={manifoldExamples.heldoutTransform} />

    <Prose>{"In the recorded run, pixels and PCA-10 each classified 73 of 75 validation images correctly (0.9733), while UMAP-10 classified 71 (0.9467). The nonlinear representation did not improve this particular classifier and split. Two observations are a small difference on a small validation set; this is a baseline comparison, not evidence of a universal ordering."}</Prose>

    <Prose>{"Ten output dimensions are a candidate for a prediction task, not a visually chosen optimum. Tune the dimension and reducer settings inside training-only resampling, then use untouched evaluation data for a final performance estimate. A coordinate system can differ between cross-validation folds without invalidating cross-validation: each fold trains its classifier and evaluates its validation rows in "}<strong>{"that fold's own fitted transform"}</strong>{". The broken workflow is mixing coordinates from unrelated fits within one model, or allowing evaluation data to influence its fitting."}</Prose>

    <H3>{"What should be clustered?"}</H3>

    <Prose>{"If your question concerns clusters in the pixel metric, cluster and assess that representation or a justified preprocessing of it. You may color a map with that result to inspect it. Clustering after UMAP is also a possible modeling choice, but it changes the geometry and density on which the clustering operates. Validate that complete choice against the intended task and a baseline. A density-based clusterer does not automatically repair the density changes in an embedding. "}<a href={"/learn/path/full-curriculum/clustering-evaluation-validation-silhouette-ari-nmi?module=classical-ml"}>{"Clustering Evaluation & Validation"}</a>{" supplies the evaluation framework; "}<a href={"/learn/path/full-curriculum/dbscan-density-based-clustering?module=classical-ml"}>{"DBSCAN & Density-Based Clustering"}</a>{" explains its particular density model."}</Prose>

    <H3>{"What will become expensive?"}</H3>

    <Prose>{"One dense n × n float64 matrix occupies 8n² bytes before overhead or copies: 10,000 rows require 800 MB; 100,000 require 80 GB; 500,000 require 2 TB, in decimal units. Isomap's all-pairs shortest-path distances can require such storage. LLE can use a sparse weight matrix, so “an n × n matrix” alone does not prove it needs dense n² storage; the solver and sparsity matter."}</Prose>

    <Prose>{"Exact t-SNE uses dense pair relationships. Barnes–Hut uses sparse input affinities and a tree approximation for repulsion, commonly described as O(n log n) per gradient evaluation in low output dimension. In two dimensions the tree is a quadtree; in three it is an octree. An angle of zero does not recreate every aspect of the separately implemented exact method because affinity sparsification is a separate choice. Interpolation methods such as FIt-SNE use grid-based approximations to repulsive sums. "}<a href={"https://jmlr.org/papers/v15/vandermaaten14a.html"}>{"Barnes–Hut t-SNE paper"}</a>{", "}<a href={"https://www.nature.com/articles/s41592-018-0308-4"}>{"FIt-SNE paper"}</a>{"."}</Prose>

    <Prose>{"For UMAP, the sparse graph has roughly O(nk) stored neighbor relationships; positive/negative updates depend on retained edges, epoch count and negative-sample rate. Approximate neighbor search has its own data- and implementation-dependent cost. There is no universal n^1.14 runtime law or “faster at every scale” conclusion. Benchmark the actual data shape, metric, precision, implementation, initialization/JIT policy, hardware and thread count. First compare a representative subset; preserve sample-selection information when interpreting its geometry."}</Prose>

    <H2>{headings[8]}</H2>

    <Prose>{"This branch develops the classical manifold methods rather than treating them as names in a menu. A "}<strong>{"Gram matrix"}</strong>{" stores pairwise dot products. An eigenvector of a symmetric matrix is a direction scaled by the matrix; its eigenvalue is that scale. The algorithms below use these directions to assemble coordinates. "}<a href={"/learn/path/full-curriculum/eigenvalues-eigenvectors?module=math-foundations"}>{"Eigenvalues & Eigenvectors"}</a>{" supplies further background."}</Prose>

    <H3>{"9.1 Classical MDS: recover coordinates from distances"}</H3>

    <Prose>{"Suppose we only know three pairwise distances: d_AB=2, d_BC=3, d_AC=5. A suitable one-dimensional configuration is "}<Code>{"[0,2,5]"}</Code>{". After subtracting its mean 7/3, the positions are "}<Code>{"[-7/3,-1/3,8/3]"}</Code>{"."}</Prose>

    <Prose>{"Classical multidimensional scaling recovers this kind of coordinate information without already knowing the positions. Square the distance matrix D elementwise, center both its rows and columns with J = I − 11ᵀ/n, and form"}</Prose>

    <MathBlock>{"B=-\\tfrac12 J D^{\\circ2}J."}</MathBlock>

    <Prose>{"Why does this work? Squared distance is "}<Code>{"||x_i||² + ||x_j||² − 2 x_i·x_j"}</Code>{". Double centering removes the first two separate row/column terms, leaving the centered dot products. Here"}</Prose>

    <MathBlock>{"B=\\frac19\\begin{bmatrix}49&7&-56\\\\7&1&-8\\\\-56&-8&64\\end{bmatrix}."}</MathBlock>

    <Prose>{"It equals yyᵀ for y = "}<Code>{"[-7/3,-1/3,8/3]"}</Code>{", so it has one positive eigenvalue 38/3 and two zero eigenvalues. If B = VΛVᵀ, use columns "}<Code>{"v_l sqrt(lambda_l)"}</Code>{" for the retained positive eigenvalues. The resulting sign can flip while all reconstructed distances stay the same."}</Prose>

    <ManifoldMdsFigure />

    <Prose>{"For general dissimilarities, B can have negative eigenvalues. That indicates these dissimilarities are not exactly the squared-distance geometry of a Euclidean point set; keeping positive components is then an approximation. "}<strong>{"Metric MDS"}</strong>{" is a different optimization: minimize distance stress "}<Code>{"Σ_{i<j}(δ_ij − ||y_i−y_j||)²"}</Code>{". "}<strong>{"Nonmetric MDS"}</strong>{" fits a monotone relation to the dissimilarities and prioritizes their ordering. Do not compare a reported strain, raw stress and normalized stress as the same quantity. Classical MDS applied to Euclidean distances recovers the PCA score geometry of the centered input when the same dimension is retained: both are extracting the same centered dot-product structure by different routes. "}<a href={"https://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds"}>{"MDS guide"}</a>{"."}</Prose>

    <H3>{"9.2 Isomap: change the distances before MDS"}</H3>

    <Prose>{"Isomap builds a neighbor graph, computes shortest paths, and passes those graph distances to classical MDS. In our U with radius 1, graph distances are "}<Code>{"|i−j|"}</Code>{" for indices 0…6. They are exactly the distances of the straight coordinates "}<Code>{"[0,1,2,3,4,5,6]"}</Code>{"; after centering, classical MDS gives "}<Code>{"[-3,-2,-1,0,1,2,3]"}</Code>{", up to sign. The A–G distance remains 6. These are the same route lengths you counted in section 2, now used to recover all coordinates at once."}</Prose>

    <Prose>{"Increasing the radius to 1.5 creates diagonal corner shortcuts, giving A–G length "}<Code>{"2 + 2sqrt(2)"}</Code>{" ≈ 4.828427. At radius 2 it admits the direct two-unit crossing. Isomap cannot tell that an edge traverses empty space when the input metric calls it a neighbor. Conversely, disconnected components have infinite intercomponent graph distances; forcing them into one finite-distance calculation requires an explicit decision, not an unexplained layout."}</Prose>

    <Prose>{"Recovering a sampled manifold's intrinsic geometry needs conditions: sufficient sampling, useful local distances, an appropriate connected graph, and geometry representable in the requested Euclidean coordinates. A curved sphere cannot be flattened into a plane preserving every geodesic distance. A hole in a sheet can make shortest routes bend around it, so a method based on global distance preservation can behave differently from one preserving local reconstruction. The canonical Swiss-roll and Swiss-hole comparison is valuable for precisely that reason. "}<a href={"https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html"}>{"Scikit-learn Swiss roll and Swiss hole example"}</a>{"."}</Prose>

    <H3>{"9.3 LLE: keep the recipe for making each point from neighbors"}</H3>

    <Prose>{"Locally Linear Embedding chooses nearby observations and expresses each point as a weighted combination of them. For point xᵢ with neighbor indices Nᵢ, solve"}</Prose>

    <MathBlock>{"\\min_{w_{ij}}\\left\\|x_i-\\sum_{j\\in N_i}w_{ij}x_j\\right\\|^2,\n\\qquad \\sum_{j\\in N_i}w_{ij}=1."}</MathBlock>

    <Prose>{"The weights may be negative. The sum-to-one constraint makes the recipe invariant under translation: adding c to every point adds c to both reconstructed and original positions. Rotating the whole neighborhood or multiplying all coordinates by one common nonzero scale also preserves the unregularized optimal weights. An arbitrary anisotropic feature rescaling generally does not."}</Prose>

    <Prose>{"Take xᵢ = (1,0), with neighbors (0,0) and (3,0). Writing "}<Code>{"1 = 0*w_left + 3*w_right"}</Code>{" and "}<Code>{"w_left+w_right=1"}</Code>{" gives weights "}<Code>{"(2/3,1/3)"}</Code>{". If their output positions are 0 and 6, the recipe puts the target at 2. It preserves relative placement in that local patch even though the patch has doubled in size."}</Prose>

    <ManifoldLleFigure />

    <Prose>{"Once every row of W is fitted, LLE chooses all output positions together by minimizing"}</Prose>

    <MathBlock>{"\\sum_i\\left\\|y_i-\\sum_jw_{ij}y_j\\right\\|^2\n=\\operatorname{tr}(Y^T(I-W)^T(I-W)Y)."}</MathBlock>

    <Prose>{"Without constraints, Y = 0 would give zero error. Requiring centered coordinates and a fixed coordinate covariance prevents that collapse. The solution uses the bottom nonconstant eigenvectors of "}<Code>{"(I−W)ᵀ(I−W)"}</Code>{". A very small reconstruction residual is therefore not directly comparable with Isomap distance error or a t-SNE KL score."}</Prose>

    <Prose>{"For one neighborhood set zⱼ = xⱼ − xᵢ and Cⱼₗ = zⱼ·zₗ. If C is invertible, the constrained least-squares solution is "}<Code>{"w=C⁻¹1 / (1ᵀC⁻¹1)"}</Code>{"; implement it with a linear solve. Singular or nearly singular neighborhoods need regularization or another formulation. Standard LLE usually adds a small multiple of the trace to the diagonal. Neighbor count, local rank and regularization affect the result. Modified LLE uses multiple reconstruction vectors to address this sensitivity; it is a different estimator, not a tolerance setting with guaranteed identical output."}</Prose>

    <H3>{"9.4 What the neighboring spectral methods change"}</H3>

    <Prose><strong>{"Laplacian Eigenmaps"}</strong>{" starts with a weighted graph A and minimizes "}<Code>{"Σ_{i,j}A_ij ||y_i−y_j||²"}</Code>{", with centering/scale constraints. Let D contain vertex weighted degrees and L=D−A; the corresponding coordinate problem uses a Laplacian eigenproblem. This penalizes separated connected vertices directly, whereas LLE penalizes failure of a multi-neighbor reconstruction recipe. UMAP often uses a spectral graph embedding to initialize its later optimization. Initialization supplies starting coordinates; it is not the final UMAP objective."}</Prose>

    <Prose><strong>{"Hessian LLE"}</strong>{" uses local second-order structure, and "}<strong>{"Local Tangent Space Alignment"}</strong>{" estimates local tangent coordinates then aligns overlapping patches. The latter's “tangent” means the best local flat approximation, connecting back to the paper surface in section 2. Both need enough well-conditioned local observations to estimate the chosen structure. These are specialist alternatives when the geometry and sampling support their local assumptions; their derivations are further reading. This lesson owns their manifold-method comparison; "}<a href={"/learn/path/full-curriculum/spectral-graph-theory?module=math-foundations"}>{"Spectral Graph Theory"}</a>{" owns graph spectra and "}<a href={"/learn/path/full-curriculum/differential-geometry-riemannian-manifolds?module=math-foundations"}>{"Differential Geometry & Riemannian Manifolds"}</a>{" owns the continuous geometry."}</Prose>

    <H3>{"9.5 A hole in the picture versus a hole in a distance construction"}</H3>

    <Prose>{"Take a unit square A=(0,0), B=(1,0), C=(1,1), D=(0,1). In a Vietoris–Rips construction, connect pairs at distance at most ε and include a filled triangle whenever its three edges exist. For 1 ≤ ε < √2, only the four sides connect, leaving one loop. At ε = √2 the diagonals and their filled triangles enter, and that loop disappears."}</Prose>

    <Prose>{"Now project to the first coordinate: A,D both map to 0 and B,C both map to 1. At ε = 0, each coincident pair connects; at ε = 1 all pairs connect and the filled simplices appear together. There is no corresponding interval containing the square's unfilled loop. The distance structure changed before the topology calculation began. This exact projection counterexample is not a simulated t-SNE/UMAP output; it isolates why computing a topological summary before and after reduction answers different questions. "}<a href={"/learn/path/full-curriculum/topology-topological-data-analysis-tda?module=math-foundations"}>{"Topology & Topological Data Analysis"}</a>{" develops filtrations and persistence."}</Prose>

    <ManifoldTopologyFigure />

    <H2>{headings[9]}</H2>

    <Prose>{"This branch exposes the update rather than trying to replace a numerical library. Recall tᵢⱼ = "}<Code>{"(1+||y_i−y_j||²)⁻¹"}</Code>{", qᵢⱼ=tᵢⱼ/Z, and Σpᵢⱼ=1 over ordered distinct pairs. Ignoring the fixed "}<Code>{"Σ p log p"}</Code>{" term gives"}</Prose>

    <MathBlock>{"C=-\\sum_{i\\ne j}p_{ij}\\log t_{ij}+\\log Z+\\text{constant}."}</MathBlock>

    <Prose>{"For a pair involving i, "}<Code>{"∂log t_ij/∂y_i = −2t_ij(y_i−y_j)"}</Code>{". Symmetry means both ordered pairs "}<Code>{"(i,j)"}</Code>{" and "}<Code>{"(j,i)"}</Code>{" contribute. The first term's derivative is therefore "}<Code>{"4Σ_j p_ij t_ij(y_i−y_j)"}</Code>{". Since "}<Code>{"∂t_ij/∂y_i = −2t_ij²(y_i−y_j)"}</Code>{", differentiating log Z gives "}<Code>{"−4Σ_j (t_ij/Z)t_ij(y_i−y_j)"}</Code>{". Recognize tᵢⱼ/Z as qᵢⱼ and combine the terms to recover the gradient in section 5. In particular, ignoring the derivative of Z loses the repulsion."}</Prose>

    <Prose>{"The following NumPy-only program fits bandwidths and optimizes a one-dimensional layout for four scalar observations. It deliberately uses no momentum or early exaggeration; the purpose is to expose the normalized objective and its gradient. Unique relevant distances make perplexity 2 achievable for each row. Save this program as "}<Code>{"tiny_tsne.py"}</Code>{" and run "}<Code>{"python tiny_tsne.py"}</Code>{" with NumPy installed. This small program supports its stated fixture; the interactive controls have separate bounds and validation."}</Prose>

    <Program example={manifoldExamples.tinyTsne} />

    <Prose>{"Calculated and probed results for this fixture:"}</Prose>

    <CodeBlock language={"text"}>{"initial KL=0.053522\nfinal KL=0.018227\n[-1.775811 -0.759392  0.422370  2.112833]"}</CodeBlock>

    <Prose>{"The first gradient is approximately "}<Code>{"[-0.077785,0.077611,0.112603,-0.112429]"}</Code>{". Subtracting half of it moves A right, B left, C left and D right. Its components sum to zero: shifting all coordinates together cannot change a distance-based objective. Centering Y after an update chooses a convenient origin without changing the loss. A finite-difference derivative independently matched the analytic initial gradient within 1.3 × 10⁻¹⁰ in the authoring probe. The lower final KL is evidence about the fixed P, not a universal claim that this optimizer found the global minimum."}</Prose>

    <H2>{headings[10]}</H2>

    <Prose>{"Try each task before opening its solution. Numerical exercises change the fixture; the final investigation changes the question."}</Prose>

    <H3>{"1. A sensor corridor"}</H3>

    <Prose>{"Five sensors lie at (0,0), (0,2), (1,2), (2,2), (2,0). Edges connect pairs at Euclidean distance ≤ ε. Find the endpoint graph distance for ε = 1, 2 and √5. Explain why the middle setting already fails to recover the six-unit corridor route."}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"list the endpoint's incident edges before looking for a long route."}</Prose>

    </details>

    <details>

    <summary>{"Solution and reasoning"}</summary>

    <Prose>{"At ε=1 the endpoint sensors are isolated, so no route joins them. At ε=2 there is already a direct endpoint-to-endpoint edge of length 2, hence the shortest distance is 2; at √5 that edge remains and the shortest distance is still 2. Euclidean distance is a lower bound on the length of any polygonal path between fixed endpoints. The corridor interpretation is external knowledge that this sampling/threshold graph fails to encode. Choosing the largest radius did not solve the modeling problem."}</Prose>

    </details>

    <H3>{"2. Perplexity without a Gaussian"}</H3>

    <Prose>{"A four-candidate row has probabilities "}<Code>{"(1/2,1/4,1/8,1/8)"}</Code>{". Calculate its entropy in bits and perplexity. If two candidates are merged into one event, can you keep calling the original perplexity a neighbor count for the new event space?"}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"use "}<Code>{"−p log₂p"}</Code>{" for each event; first identify how many events the new distribution has."}</Prose>

    </details>

    <details>

    <summary>{"Solution and reasoning"}</summary>

    <Prose>{"H = 1/2 + 1/2 + 3/8 + 3/8 = 1.75 bits, giving perplexity 2^1.75 ≈ 3.363586. Merging the last two events makes "}<Code>{"(1/2,1/4,1/4)"}</Code>{", with H=1.5 and perplexity √8 ≈ 2.828427. Perplexity is defined for the actual probability distribution over events; it is neither the count of positive entries nor a property independent of representation."}</Prose>

    </details>

    <H3>{"3. Audit a misleading score"}</H3>

    <Prose>{"Your map preserves 6 of each query's 8 nearest neighbors on average. A report says “trustworthiness is 0.99, so 99% of neighbors are correct.” Repair the sentence and supply the directly relevant percentage. Explain what data would let you calculate trustworthiness itself."}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"the two metrics count different things."}</Prose>

    </details>

    <details>

    <summary>{"Solution and reasoning"}</summary>

    <Prose>{"Neighbor retention is 6/8=75%, so 25% of directed eight-neighbor selections change. Trustworthiness 0.99 means a small normalized rank penalty for false map neighbors, under its chosen k and metric. It does not mean a 99% retention rate. Calculate it from the input ranks of all map neighbors that were not input top-eight neighbors, together with n and k. Inspect poor individual queries even when the global score is high."}</Prose>

    </details>

    <H3>{"4. A fuzzy edge after one observation moves"}</H3>

    <Prose>{"Before editing, the directed strengths are 0.2 and 0.6. Compute the fuzzy union. After an observation moves, only the second strength changes to 0.9. Compute the new union. For the ideal pair model a=b=1, find the separation giving ν equal to each union strength. Explain why these two optima do not predict a library UMAP trajectory."}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"solve "}<Code>{"1/(1+r²)=w"}</Code>{" after calculating w."}</Prose>

    </details>

    <details>

    <summary>{"Solution and reasoning"}</summary>

    <Prose>{"Initially w=0.2+0.6−0.12=0.68, giving r=√(8/17)≈0.685994. Afterwards w=0.2+0.9−0.18=0.92, giving r=√(2/23)≈0.294884. The stronger ideal connection prefers a closer pair. A full graph has competing pairs, and the sampled UMAP procedure has the weighting described in section 6. These are exactly reproducible isolated-pair calculations, not output coordinates."}</Prose>

    </details>

    <H3>{"5. Recover a position by two routes"}</H3>

    <Prose>{"Three collinear points have pairwise distances 3,4,7, with the seven-unit distance between the endpoints. Give centered one-dimensional MDS coordinates. Independently find the LLE weights of the middle point using the endpoints. If the mapped endpoints are −2 and 12, where does that recipe put the middle point?"}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"place the original points at 0,3,7 first; the sum-to-one weights preserve their relative position."}</Prose>

    </details>

    <details>

    <summary>{"Solution and reasoning"}</summary>

    <Prose>{"The mean is 10/3, so centered MDS coordinates are "}<Code>{"[-10/3,-1/3,11/3]"}</Code>{", or their simultaneous negatives. The middle position is 3/7 of the way from 0 to 7, giving weights "}<Code>{"(4/7,3/7)"}</Code>{". The mapped middle is "}<Code>{"(4/7)(−2)+(3/7)(12)=4"}</Code>{". Both routes express the same relative position for this line, but in general MDS uses all dissimilarities while LLE uses local recipes."}</Prose>

    </details>

    <H3>{"6. Diagnose the deployment plan"}</H3>

    <Prose>{"An engineer fits one StandardScaler on training images, another on test images, fits UMAP separately to both scaled sets, and sends the test coordinates to a classifier trained on the training map. They suggest switching to PCA initialization to fix the inconsistency. Give the actual repair and two baselines."}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"keep a single fitted sequence of operations for both portions."}</Prose>

    </details>

    <details>

    <summary>{"Solution and reasoning"}</summary>

    <Prose>{"Fit the scaler and UMAP on training data only. Apply "}<Code>{"scaler.transform"}</Code>{" followed by the fitted reducer's "}<Code>{"transform"}</Code>{" to test rows. Train and evaluate the classifier within that coordinate system. Initialization does not align independently fitted maps or repair different feature scales. Compare the complete pipeline with a classifier on appropriately scaled input features and with a training-fitted PCA pipeline. Choose hyperparameters using training-only validation; retain an untouched final test set."}</Prose>

    </details>

    <H3>{"7. Independent digits audit"}</H3>

    <Prose>{"Use the supplied collection, choose k from {5,10,20}, and pick one image by its source-row identifier before inspecting its maps. Compare PCA with two t-SNE perplexities. Record a prediction, its actual retained neighbor identities, and one visually tempting inference you can test in pixel space. Then choose a second image with a different writing style and repeat without changing your metric."}</Prose>

    <details>

    <summary>{"Hint"}</summary>

    <Prose>{"start by sorting both distance rows and displaying the corresponding 8 × 8 pixel grids. Similar labels and similar pixel measurements need not be the same criterion."}</Prose>

    </details>

    <details>

    <summary>{"Success criteria and example response"}</summary>

    <Prose>{"your answer names the data selection, pixel metric, source rows, k, map settings and the preserved/missing/false neighbors; the retained fraction agrees with those lists; you use an actual image or original-space measurement to investigate your inference. Explain why one query need not follow the aggregate ranking. An acceptable conclusion is: “At k=10, this image loses three original neighbors in map A and four in map B. I prefer A for inspecting this particular local similarity, even though B scores better over all rows.” Populate the counts from your chosen input rather than copying that illustrative sentence. "}<a href={"/learn-assets/manifold-learning/calculated-inputs.json"}>{"calculated-inputs.json"}</a>{" supplies the saved coordinate variants for exact independent checking."}</Prose>

    </details>

    <H2>{headings[11]}</H2>

    <Prose>{"Alternate explanations and practice:"}</Prose>

    <ul>
      <li><strong>{"Wattenberg, Viégas & Johnson — "}<a href={"https://distill.pub/2016/misread-tsne/"}>{"How to Use t-SNE Effectively"}</a></strong>{". Interactive article, useful after sections 3–5 for comparing varied density, cluster arrangements, stopping times and perplexities on controlled examples. It is a 2016 explanation, so use current library documentation for API defaults."}</li>
      <li><strong>{"Leland McInnes — "}<a href={"https://speakerdeck.com/lmcinnes/a-guide-to-dimension-reduction"}>{"A Bluffer's Guide to Dimension Reduction"}</a></strong>{". Creator's slide deck from the "}<a href={"https://pydata.org/nyc2018/schedule/presentation/1/"}>{"PyData NYC 2018 talk"}</a>{", useful after this lesson to compare neighbor graphs with matrix factorizations. It is an intuition-oriented survey with compressed assumptions; use the papers for derivations and current documentation for API details."}</li>
      <li><strong>{"Scikit-learn — "}<a href={"https://scikit-learn.org/stable/modules/manifold.html"}>{"Manifold learning chapter"}</a></strong>{". Freely accessible implementation-oriented chapter, especially useful on the return route through section 9. It covers Isomap, standard/modified/Hessian LLE, Laplacian Eigenmaps, tangent-space alignment, MDS and t-SNE. Library details cited here use the 1.9.1 snapshot."}</li>
      <li><strong>{"UMAP authors — "}<a href={"https://umap-learn.readthedocs.io/en/latest/how_umap_works.html"}>{"How UMAP Works"}</a></strong>{". Illustrated prose from simplices and local distance to weighted graphs; read after section 6 if the topology motivation interests you. Keep the graph/theory/layout distinction and the sampled-optimization discussion from this lesson alongside its intuitive account."}</li>
    </ul>

    <Prose>{"Technical sources and precise follow-up:"}</Prose>

    <ul>
      <li><strong>{"van der Maaten & Hinton (2008), "}<a href={"https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf"}>{"Visualizing Data Using t-SNE"}</a></strong>{". Original open paper. Sections 2–4 and Appendix A are the return route for SNE, crowding, symmetrization, gradient and optimization; the paper's experimental datasets are not the source of this lesson's numerical results."}</li>
      <li><strong>{"McInnes, Healy & Melville, "}<a href={"https://arxiv.org/html/1802.03426v3"}>{"UMAP"}</a></strong>{". Original paper with mathematical and algorithmic treatments. Return to its graph construction and algorithm after section 6; the categorical/topological proofs are specialist reading requiring additional mathematics."}</li>
      <li><strong>{"Damrich & Hamprecht (2021), "}<a href={"https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf"}>{"On UMAP's True Loss Function"}</a></strong>{". Research paper for the gap between full-pair formulas and sampling. Sections 3–6 develop the effective-loss argument and a controlled ring example; useful deeper reading after deriving the ideal pair cost."}</li>
      <li><strong>{"UMAP authors — "}<a href={"https://umap-learn.readthedocs.io/en/latest/parameters.html"}>{"parameters"}</a>{", "}<a href={"https://umap-learn.readthedocs.io/en/latest/transform.html"}>{"new-data transform"}</a>{", "}<a href={"https://umap-learn.readthedocs.io/en/latest/reproducibility.html"}>{"reproducibility"}</a></strong>{". Worked official documentation for parameter roles, fitting boundaries and seed/thread behavior. The examples on this page were executed with UMAP 0.5.12; consult the matching installed version when reproducing results."}</li>
      <li><strong>{"Alpaydin & Kaynak — "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"Optical Recognition of Handwritten Digits"}</a></strong>{". Dataset source, feature definitions and CC BY 4.0 license. Attribution and subset details travel with the supplied data."}</li>
    </ul>

    <H2>{headings[12]}</H2>

    <Prose>{"Without looking back, explain why increasing a graph neighborhood can shorten an estimated geodesic; how t-SNE turns distances into a row and then moves its coordinates; how UMAP's graph differs from its layout; and why 0.99 trustworthiness does not mean 99% neighbor retention. Reproduce one changed numerical exercise and audit one actual image. Those tasks are a stronger readiness check than recognizing a visually separated map."}</Prose>

    <Prose>{"Next is "}<a href={"/learn/path/full-curriculum/independent-component-analysis-ica?module=classical-ml"}>{"Independent Component Analysis (ICA)"}</a>{". It asks whether measured mixtures can be expressed using statistically independent sources. Return to the measured feature or signal matrix for that question: the next lesson does not require passing a distorted two-dimensional visualization into ICA. PCA finds variance directions, neighbor embeddings organize relationships, and ICA introduces a different criterion for a different representation problem."}</Prose>
  </div>,
};

export default manifoldContent;
