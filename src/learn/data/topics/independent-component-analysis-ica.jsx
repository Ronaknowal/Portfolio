import { H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  IcaMixingFigure, IcaDependenceFigure, IcaWhiteningFigure, IcaFixedPointFigure, IcaSplitFigure, IcaOutcomeFigure,
} from '../../components/lesson-labs/IcaFigures.jsx';
import { IcaRotationLab, IcaContributionLab } from '../../components/lesson-labs/IcaLabs.jsx';
import { icaExamples } from '../ica-examples.js';
import '../../components/lesson-labs/ica-labs.css';

const headings = [
  '1. Follow one sample through a mixture',
  '2. Zero correlation can hide complete dependence',
  '3. Whitening removes a stretch, leaving a separation question',
  '4. Why non-Gaussianity gives a direction',
  '5. Follow a FastICA update',
  '6. A real recording: fit blindly, choose with development data, evaluate later',
  '7. Deeper branch: what is a component, and what happens if we remove it?',
  '8. Deeper branch: objectives, extensions and computation',
  '9. Practice with changed inputs',
  '10. Readiness and the next question',
  '11. References & another way to learn it',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example }) {
  return <section>
    <Prose><strong>{'Before running: '}</strong>{example.question}</Prose>
    <RunnableExample example={example} />
  </section>;
}

function Practice({ title, children, hint, solution }) {
  return <>
    <H3>{title}</H3>
    {children}
    <details><summary>Hint</summary><div>{hint}</div></details>
    <details><summary>Solution</summary><div>{solution}</div></details>
  </>;
}

const icaContent = {
  title: 'Independent Component Analysis (ICA)',
  readTime: '~45–55 min first pass · 45–75 min code and practice · deeper branches add ~25 min',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot ica-lesson">
    <LessonIntro
      prerequisites="Dot products, matrix multiplication, an average and a variance. PCA supplies the geometry of projections and reconstruction; the probability and optimization ideas are introduced here. The Python examples need NumPy and scikit-learn, and the supplied CSV lets the real one run offline."
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      {'Separate an exact four-state mixture by hand, watch a distribution change while its covariance does not, trace one fixed-point update, then ask a real electrical recording a narrow, measurable question — and keep the answer it gives.'}
    </LessonIntro>

    <Prose>{'Two sensors can both contain the same two signals in different proportions. A loud pulse in one recording might come from the event you care about, an interfering source, or both. Instead of keeping the direction with the largest variation, can we find combinations that separate the contributions?'}</Prose>

    <Prose><strong>{'Independent component analysis'}</strong>{' estimates a linear representation whose component signals are as statistically independent as its model and estimation method can make them. We will first separate an exact four-state mixture. Then we will ask a narrower, measurable question of a real electrical recording: does an ICA component track a simultaneously recorded reference more closely than an original channel or a principal component?'}</Prose>

    <Prose><strong>{'First-pass route.'}</strong>{' Read sections 1–5, using the mixing figure and the rotation investigation as you go. Run the short NumPy example in section 5, then read and run the real-data comparison in section 6. Try practice 1, 2 and 4 in section 9 before the readiness check. Sections 7 and 8 are deeper branches for component removal, objectives, computation and extensions. Expect about 45–55 minutes of reading on the first route, plus 45–75 minutes for code and practice; the deeper branches add about 25 minutes.'}</Prose>

    <Prose>{'You need dot products, matrix multiplication, an average and variance. '}<a href="/learn/path/full-curriculum/pca-dimensionality-reduction?module=classical-ml">{'PCA & Dimensionality Reduction'}</a>{' supplies the geometry of projections, eigenvectors and reconstruction. '}<a href="/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations">{'Probability Distributions & Bayes’ Theorem'}</a>{' reviews independence and moments. We will introduce the specific probability and optimization ideas locally. Python examples need NumPy and scikit-learn; the supplied CSV lets the real example run offline.'}</Prose>

    <Prose>{'The previous topic, '}<a href="/learn/path/full-curriculum/t-sne-umap-manifold-learning?module=classical-ml">{'t-SNE, UMAP & Manifold Learning'}</a>{', asks how to display relationships among observations in a small number of coordinates. ICA asks a different question about the observations’ '}<strong>{'generating mixture'}</strong>{'. Neither a separated-looking embedding nor uncorrelated PCA coordinates supplies that generating model.'}</Prose>

    <H2>{headings[0]}</H2>

    <Prose>{'Suppose source values at one instant are s₁ = 1 and s₂ = −1. Sensor 1 receives twice the first source plus the second; sensor 2 receives the first plus twice the second:'}</Prose>

    <MathBlock>{'\\begin{gathered} x_1 = 2s_1 + s_2 = 1, \\\\ x_2 = s_1 + 2s_2 = -1. \\end{gathered}'}</MathBlock>

    <Prose>{'The compact notation is'}</Prose>

    <MathBlock>{'x = As, \\qquad A = \\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \\end{bmatrix}.'}</MathBlock>

    <Prose>{'Here s is the column of two source amplitudes, x is the column of two observed amplitudes, and A is the '}<strong>{'mixing matrix'}</strong>{'. Its column aⱼ tells how source j contributes across sensors. Its row i gives sensor i’s recipe. The coefficients have units of observed amplitude per source amplitude; our hand example uses arbitrary units.'}</Prose>

    <IcaMixingFigure />

    <Prose>{'If we knew A, ordinary algebra would solve the problem:'}</Prose>

    <MathBlock>{'\\begin{gathered} A^{-1} = \\tfrac13 \\begin{bmatrix} 2 & -1 \\\\ -1 & 2 \\end{bmatrix}, \\\\ s_1 = (2x_1 - x_2)/3, \\\\ s_2 = (-x_1 + 2x_2)/3. \\end{gathered}'}</MathBlock>

    <Prose>{'Substituting x = (1, −1)ᵀ recovers (1, −1)ᵀ. The difficult part of '}<strong>{'blind source separation'}</strong>{' is estimating the recipes when only many observed x’s are available. “Blind” describes that missing mixing information; assumptions still do substantial work.'}</Prose>

    <Prose>{'For a dataset, each row is one simultaneous observation and each column is a sensor. Thus X has shape n × d, S has shape n × k, and A has shape d × k. With row storage, the same relation is X = SAᵀ. An unmixing operator B with shape k × d produces Ŝ = (X − 1μᵀ)Bᵀ, where μ is the fitted vector of sensor means. We reserve W below for the rotation '}<strong>{'after whitening'}</strong>{', so B = WK when K is the whitener.'}</Prose>

    <H3>The model’s conditions belong here</H3>

    <Prose>{'Our exact model is a '}<strong>{'noiseless, instantaneous, constant linear mixture'}</strong>{': the sensor reading now depends on source values now through one fixed matrix. The basic identifiable case has mutually independent, nondegenerate sources, at most one Gaussian source, and a square invertible mixing matrix. With more sensors than sources, a full-column-rank mixing model can first be represented in its source-dimensional signal subspace. More sources than sensors requires additional structure and a different separation method. The moment calculations in this lesson assume finite variances, and kurtosis additionally needs finite fourth moments.'}</Prose>

    <Prose>{'Independence here is between the source variables at the same observation. A signal may still resemble its own recent past. Ordinary FastICA uses the distribution of simultaneous samples rather than an explicit temporal model. Autocorrelation changes how much independent information a recording supplies, so 20,000 time samples need not provide the information of 20,000 independent draws.'}</Prose>

    <Prose>{'A real room introduces propagation delays and echoes, so the cocktail-party story is a useful motivation for this simplified model. Biological recordings also contain measurement noise and sources that may share activity. We will use the model as a tool and evaluate its result, with the model conditions available here whenever needed. The foundational tutorial introduces this same distinction between the ideal mixture and its applications. '}<a href="https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf">{'Hyvärinen & Oja, 2000, sections 1–2'}</a>{'.'}</Prose>

    <H2>{headings[1]}</H2>

    <Prose>{'The expectation E[u] is a probability-weighted average. Covariance measures whether two centered variables tend to have products of the same sign:'}</Prose>

    <MathBlock>{'\\operatorname{Cov}(u,v) = E[(u - Eu)(v - Ev)].'}</MathBlock>

    <Prose>{'Zero covariance is one equality about an average. '}<strong>{'Independence'}</strong>{' is stronger: for every pair of events about the variables, the probability of both equals the product of the individual probabilities. For densities, this becomes p(u,v) = p(u)p(v). Independence implies zero covariance when the required moments exist.'}</Prose>

    <Prose>{'For an exact counterexample, let u take −1, 0, 1, each with probability 1/3, and let v = u². Then Eu = 0, Ev = 2/3, and E[uv] = E[u³] = 0, so covariance is zero. Nevertheless, observing v = 0 tells us u = 0 exactly. The joint probability of u = 0, v = 0 is 1/3, whereas the product of the marginals is 1/9.'}</Prose>

    <IcaDependenceFigure />

    <Prose>{'PCA finds orthogonal directions that diagonalize covariance. Whitening also rescales them to unit variance. Neither operation tests all these joint probabilities. For a '}<strong>{'jointly Gaussian'}</strong>{' vector, zero cross-covariances do imply independence. Having separately Gaussian-looking histograms is a weaker observation than establishing a joint Gaussian model.'}</Prose>

    <H3>An exact source distribution we can carry through every step</H3>

    <Prose>{'Let s₁ and s₂ be independent fair choices from {−1, 1}. There are four equally likely source states:'}</Prose>

    <LessonTable caption="Four equiprobable source states and the observations they produce"
      headers={['Source state (s₁, s₂)', 'Mixed observation (x₁, x₂)']}
      rows={[
        ['(−1, −1)', '(−3, −3)'],
        ['(−1, 1)', '(−1, 1)'],
        ['(1, −1)', '(1, −1)'],
        ['(1, 1)', '(3, 3)'],
      ]} />

    <Prose>{'Every marginal sign has probability 1/2; every pair has probability 1/4. These four rows enumerate a designed probability distribution. They are not a claim that four arbitrary measurements suffice to fit useful real-world ICA.'}</Prose>

    <Prose>{'The source means are zero and E[ssᵀ] = I, the identity matrix. The mixed covariance is'}</Prose>

    <MathBlock>{'\\Sigma_x = A I A^{\\mathsf T} = \\begin{bmatrix} 5 & 4 \\\\ 4 & 5 \\end{bmatrix}.'}</MathBlock>

    <Prose>{'For example, sensor 1 has average squared value (9 + 1 + 1 + 9)/4 = 5, and the average product of sensor readings is (9 − 1 − 1 + 9)/4 = 4.'}</Prose>

    <H2>{headings[2]}</H2>

    <Prose>{'The covariance has eigenvectors v₊ = (1, 1)ᵀ/√2 and v₋ = (1, −1)ᵀ/√2, with eigenvalues 9 and 1. Multiplying Σₓ v₊ = 9 v₊ verifies the first pair. PCA projects onto these directions. It obtains scores'}</Prose>

    <MathBlock>{'\\begin{gathered} p_+ = (x_1 + x_2)/\\sqrt2, \\\\ p_- = (x_1 - x_2)/\\sqrt2. \\end{gathered}'}</MathBlock>

    <Prose>{'Their variances are 9 and 1. '}<strong>{'Whitening'}</strong>{' divides each score by its standard deviation:'}</Prose>

    <MathBlock>{'\\begin{gathered} z_1 = \\frac{x_1 + x_2}{3\\sqrt2} = \\frac{s_1 + s_2}{\\sqrt2}, \\\\[4pt] z_2 = \\frac{x_1 - x_2}{\\sqrt2} = \\frac{s_1 - s_2}{\\sqrt2}. \\end{gathered}'}</MathBlock>

    <Prose>{'Now E[zzᵀ] = I. Yet z₁ = 0 forces z₂ to be ±√2, never zero. Both individual zero events have probability 1/2, but their intersection has probability zero. The whitened coordinates are still dependent mixtures.'}</Prose>

    <IcaWhiteningFigure />

    <Prose>{'A final linear combination recovers the sources:'}</Prose>

    <MathBlock>{'\\begin{bmatrix} s_1 \\\\ s_2 \\end{bmatrix} = \\frac1{\\sqrt2} \\begin{bmatrix} 1 & 1 \\\\ 1 & -1 \\end{bmatrix} \\begin{bmatrix} z_1 \\\\ z_2 \\end{bmatrix}.'}</MathBlock>

    <Prose>{'This matrix is orthogonal: its rows have length one and dot product zero. It includes a reflection; when we speak informally of the remaining “rotation,” the allowed orthogonal transformations include reflections and sign changes.'}</Prose>

    <Prose>{'In general, if Σₓ = VDVᵀ with positive eigenvalues, the whitener is K = D'}<sup>{'−1/2'}</sup>{'Vᵀ. Then z = K(x − μ) has identity covariance. With unit-variance independent sources in the square noiseless model, Q = KA obeys QQᵀ = I. That calculation explains why searching over orthogonal W’s after whitening is sufficient. It reduces the unknown scaling and shearing before the independence search.'}</Prose>

    <Prose>{'If an eigenvalue is zero, dividing by its square root is impossible; a constant or redundant channel provides no new direction. Very small eigenvalues can amplify noise. Estimate effective rank and choose a defensible subspace. Reducing to k < d principal coordinates before ICA chooses a '}<strong>{'variance-based subspace'}</strong>{'; it does not select the k most independent or most non-Gaussian physical sources.'}</Prose>

    <H2>{headings[3]}</H2>

    <Prose>{'For a centered variable with nonzero variance, its '}<strong>{'excess kurtosis'}</strong>{' is'}</Prose>

    <MathBlock>{'\\kappa(y) = \\frac{E[y^4]}{E[y^2]^2} - 3.'}</MathBlock>

    <Prose>{'The subtraction gives a Gaussian value of zero. Our unit-variance binary source has E[s⁴] = 1, hence κ = −2. A unit-variance Laplace source has excess kurtosis 3. Positive and negative departures can both supply useful information.'}</Prose>

    <Prose>{'Take independent, centered unit-variance sources and a unit-length combination y = a s₁ + b s₂, with a² + b² = 1. Expanding the fourth power gives'}</Prose>

    <MathBlock>{'E[y^4] = a^4 E[s_1^4] + 6a^2b^2 + b^4 E[s_2^4].'}</MathBlock>

    <Prose>{'The odd cross-terms vanish because each contains a zero source mean. Subtract 3(a² + b²)², and the result is'}</Prose>

    <MathBlock>{'\\kappa(y) = a^4 \\kappa(s_1) + b^4 \\kappa(s_2).'}</MathBlock>

    <Prose>{'For two binary sources with a = b = 1/√2, this is −1, compared with −2 for either unmixed source. For two Laplace sources it is 1.5, compared with 3. '}<strong>{'Equal source kurtoses do not destroy separation:'}</strong>{' the fourth powers change with the direction. At angle θ, two equal Laplace sources give 3(cos⁴θ + sin⁴θ), with maxima at source axes rather than a ring of equal maxima.'}</Prose>

    <Prose>{'The central limit theorem offers the intuition that many independent contributions can make a standardized sum more Gaussian. The fourth-moment identity is the exact reason in our example. The general theorem is a limiting statement under conditions; it does not say every mixture of two arbitrary distributions improves every measure of Gaussianity.'}</Prose>

    <H3>The Gaussian ambiguity is a population fact</H3>

    <Prose>{'If s₁, s₂ are independent standard Gaussians, their joint density is proportional to exp[−(s₁² + s₂²)/2]. An orthogonal transformation preserves that sum of squares. The transformed pair has the same joint Gaussian distribution and independent coordinates. Observations alone cannot tell which of these bases was the original source basis.'}</Prose>

    <Prose>{'With two or more Gaussian sources, their Gaussian subspace has this unresolved rotation. At most one Gaussian source is allowed in the usual identifiable ICA model; other non-Gaussian sources can then determine the remaining direction. This is about what the probability model identifies, not whether a numerical solver happens to return an array. Finite Gaussian samples can have accidental fourth-moment structure, and a solver can follow it.'}</Prose>

    <Prose>{'Kurtosis is only one diagnostic. A non-Gaussian variable taking 0 with probability 2/3, and ±√3 with probability 1/6 each, has variance 1 and fourth moment 3: its excess kurtosis is also zero. Its sixth moment is 9, whereas a standard Gaussian’s is 15. Therefore a threshold such as “all kurtoses close to zero” cannot establish Gaussianity or decide ICA suitability by itself.'}</Prose>

    <H3>Investigation: rotate a distribution, not just its covariance</H3>

    <Prose>{'In the rotation investigation, record whether your proposed new projection will have greater, equal or smaller absolute excess kurtosis than the active one. Enter an angle of your own before revealing the result. The covariance remains I under every orthogonal rotation, while the joint support and fourth moment can change.'}</Prose>

    <Prose>{'Start with the binary source distribution above. Try a projection halfway between its source directions, then a direction near one source. Return with the Gaussian population selected. Explain what the Gaussian null case removes from the search.'}</Prose>

    <IcaRotationLab />

    <H3>A broader objective</H3>

    <Prose>{'For a continuous variable, differential entropy is H(y) = −∫ p(y) log p(y) dy: an average log-density measure, not a histogram bar’s height. Among distributions with a fixed variance, the Gaussian has maximum entropy. '}<strong>{'Negentropy'}</strong>{' measures the gap J(y) = H(y'}<sub>{'G'}</sub>{') − H(y), where the Gaussian has the same variance.'}</Prose>

    <Prose>{'Estimating a full density can be difficult. FastICA commonly uses a nonquadratic contrast related to an approximation J(y) ∝ [E G(y) − E G(ν)]², with ν ~ N(0, 1). This is a surrogate for searching, not an exact measured mutual information. Choices include G(u) = log cosh(u), G(u) = −e'}<sup>{'−u²/2'}</sup>{', and G(u) = u⁴/4. The cube choice makes a particularly transparent calculation; fourth powers are also sensitive to unusually large observations. The log-cosh derivative grows more gently. No one nonlinearity is best for every source distribution.'}</Prose>

    <Prose>{'The entropy argument in this paragraph concerns continuous densities. The binary hand example uses exact probabilities and kurtosis; it is not being assigned a differential entropy. Section 8 connects the continuous objective to independence and likelihood.'}</Prose>

    <H2>{headings[4]}</H2>

    <Prose>{'For whitened observations zᵢ ∈ ℝᵏ, a unit vector w produces a component yᵢ = wᵀzᵢ. The unit-length constraint keeps its variance at one, so a larger objective must come from changing the distribution rather than simply magnifying every amplitude.'}</Prose>

    <Prose>{'Let g = G′. A one-component FastICA iteration computes'}</Prose>

    <MathBlock>{'\\begin{gathered} r = \\frac1n \\sum_i z_i\\, g(w^{\\mathsf T} z_i) - \\bar g\'\\, w, \\\\[4pt] \\bar g\' = \\frac1n \\sum_i g\'(w^{\\mathsf T} z_i), \\\\[4pt] w_{\\mathrm{new}} = r / \\lVert r \\rVert. \\end{gathered}'}</MathBlock>

    <Prose>{'The first average weights each observation by a nonlinear function of its current projection. The second term corrects the current direction; normalization restores the constraint. In the log-cosh version, g(u) = tanh u and g′(u) = 1 − tanh²u.'}</Prose>

    <Prose>{'For a hand trace, use the four whitened diamond points from section 3 and w = (0.8, 0.6). With g(u) = u³, the projections are the signed values 0.8√2 and 0.6√2. The first average is (2(0.8)³, 2(0.6)³) = (1.024, 0.432). The average derivative is E[3y²] = 3. Thus'}</Prose>

    <MathBlock>{'\\begin{gathered} r = (1.024,\\, 0.432) - 3(0.8,\\, 0.6) \\\\ = (-1.376,\\, -1.368). \\end{gathered}'}</MathBlock>

    <Prose>{'After normalization and an optional sign flip for easier comparison, the next vector is approximately (0.7091653, 0.7050422). It moved toward (1, 1)/√2, which extracts s₁. This is the same final combination we found by algebra in section 3, now approached by a distribution-based update.'}</Prose>

    <IcaFixedPointFigure />

    <H3>Why this update has that form</H3>

    <Prose>{'At a stationary point of E[G(wᵀz)] constrained by ‖w‖² = 1, the objective gradient must align with w: E[z g(wᵀz)] − βw = 0. The multiplier β accounts for the constraint; multiplying by wᵀ gives β = E[y g(y)] at a stationary point.'}</Prose>

    <Prose>{'Newton’s method solves an equation using its derivative. The derivative matrix here contains E[zzᵀ g′(wᵀz)] − βI. FastICA makes the approximation E[zzᵀ g′] ≈ E[g′]I in whitened coordinates. Simplifying the resulting Newton-like step and discarding a scalar that normalization will remove gives the update above. Whitening makes E[zzᵀ] = I; it does '}<strong>{'not'}</strong>{' by itself make that factorization exact. The method searches for a fixed direction, with local convergence depending on the distribution, contrast and initialization. '}<a href="https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf">{'Hyvärinen & Oja, section 6, equations 41–43'}</a>{'.'}</Prose>

    <Prose>{'For several components, unconstrained repetitions could rediscover the same direction. '}<strong>{'Deflation'}</strong>{' estimates them one at a time. After '}<strong>{'each update'}</strong>{', subtract its projection on every previously found unit direction, then normalize. If a previous direction is q, subtract (qᵀr)q. Orthogonality enforces distinct uncorrelated coordinates in whitened space; the nonlinear objective still supplies the separation criterion.'}</Prose>

    <Prose><strong>{'Parallel or symmetric FastICA'}</strong>{' updates all rows of W and orthogonalizes them together, using W ← (WWᵀ)'}<sup>{'−1/2'}</sup>{'W when that inverse square root exists. A sign-aware stopping test is 1 − |w'}<sub>{'new'}</sub>{'ᵀw| < tolerance. Parallel and antiparallel vectors describe the same source direction. A small directional change records numerical convergence, which is distinct from a successful application diagnostic.'}</Prose>

    <H3>Complete NumPy example</H3>

    <Prose>{'Use a Python environment with NumPy and scikit-learn. The author’s calculation snapshot used Python 3.12.14 and these package versions; install them once if they are not already available:'}</Prose>

    <CodeBlock language="sh">{'python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1'}</CodeBlock>

    <Prose>{'Save the following program as '}<Code>{'ica_by_hand.py'}</Code>{' and run '}<Code>{'python ica_by_hand.py'}</Code>{'. This program uses the full four-state distribution, not sampled sine waves claimed to be independent. It implements deflation with the orthogonalization inside the iteration. The finite, full-rank fixture is supplied; a zero update or a rank-deficient input needs diagnosis before normalization in a general-purpose implementation.'}</Prose>

    <Prose>{'The covariance average divides by four because the rows enumerate four equiprobable states. NumPy’s '}<Code>{'eigh'}</Code>{' returns eigenvalues in ascending order; section 3 deliberately listed the larger one first. The code’s whitened axes can therefore differ in order and sign from the figure while representing the same information.'}</Prose>

    <Program example={icaExamples.handSeparation} />

    <Prose>{'The output checks whitening, source agreement and reconstruction separately. Here the two high correlations must match different recovered columns; inspect C to confirm the one-to-one correspondence. In a general k-source benchmark, match components by a one-to-one assignment maximizing total absolute correlation, then resolve signs and scales. Taking each row’s best match independently can reuse one recovered component and exaggerate recovery.'}</Prose>

    <Prose>{'The exact hand calculation and a small author probe support these expected results. The program above was executed again for this page, in the recorded environment, and its printed output is the output shown.'}</Prose>

    <H2>{headings[5]}</H2>

    <Prose>{'An abdominal electrode records overlapping electrical activity. In the '}<strong>{'Abdominal and Direct Fetal ECG Database'}</strong>{', researchers recorded four abdominal channels and a simultaneous direct fetal ECG reference to study fetal-heartbeat measurement from abdominal signals. The supplied extract is the first 20 seconds of record '}<Code>{'r01'}</Code>{', version 1.0.0, sampled at 1,000 Hz. A row is one instant, not one person. The four abdominal values are inputs; the direct channel is an external comparison signal. '}<a href="https://physionet.org/content/adfecgdb/1.0.0/">{'Dataset description and acquisition details'}</a>{'.'}</Prose>

    <Prose>{'The immediate question is deliberately measurable without specialist physiology: '}<strong>{'which representation contains a single coordinate with stronger absolute linear correlation to that reference over a later four-second interval?'}</strong>{' Correlation measures aligned waveform variation. It is not a fetal-beat detector, a clinical accuracy measure, or a count of recovered physiological sources. A reference measured at a different location can differ in waveform and polarity. This is a short within-recording investigation; evaluating a clinical or cross-person claim would require a different task, endpoints and independent participants.'}</Prose>

    <Prose><a href="/learn-assets/ica/r01-first20s.csv">{'Download the provided CSV'}</a>{' and keep it beside '}<Code>{'ica_recording.py'}</Code>{'. Save the program below under that name, then run '}<Code>{'python ica_recording.py'}</Code>{' from their directory. After the one-time package installation, the program needs no network access. '}<a href="/learn-assets/ica/data-provenance.md">{'Data provenance'}</a>{' gives the original authors, ODC-By 1.0 license, source and extract hashes, exact columns and conversion. The stored integers are ADC counts. The program converts them to microvolts using the EDF header’s calibration. The provider already performed acquisition/filtering steps; the lesson adds no filter or resampling.'}</Prose>

    <IcaSplitFigure />

    <Prose>{'Before running, predict which of raw channels, PCA coordinates or ICA coordinates will give the largest held-out absolute correlation. The comparison uses all four coordinates for both decompositions. It chooses the coordinate within each method only on development data, then freezes the choice. The record, interval, split and ICA settings were fixed before observing the comparison; we keep an inconvenient outcome rather than search for a better seed or time window.'}</Prose>

    <Program example={icaExamples.realRecording} />

    <Prose>{'Columns are representation, chosen one-based coordinate, development |r|, test |r|. PCA coordinate 4 has the largest test value in this fixed comparison. ICA coordinate 2 improves on the selected raw channel, but its independence-oriented objective does not optimize this reference-correlation diagnostic. The change from development to test also makes the short interval’s variability visible. The practical conclusion is to retain the comparison and investigate the signal/task relationship before preferring an ICA pipeline. The lower-variance PCA coordinate mattered here; automatically discarding it would have removed that candidate.'}</Prose>

    <IcaOutcomeFigure />

    <Prose>{'The library handles fitted centering and whitening. '}<Code>{'ica.components_'}</Code>{' is B = WK, shape 4 × 4; '}<Code>{'mixing_'}</Code>{' is its pseudoinverse, shape 4 × 4; '}<Code>{'mean_'}</Code>{' has four sensor means. '}<Code>{'transform(X_new)'}</Code>{' uses those fitted quantities. Calling '}<Code>{'fit_transform'}</Code>{' on a new interval would estimate a new decomposition and invalidate a comparison that assumes fixed component identities. '}<Code>{"whiten='unit-variance'"}</Code>{' normalizes fitted source variances; '}<Code>{'whiten=False'}</Code>{' expects an already whitened input. Explicit settings avoid relying on an older default. '}<a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html">{'FastICA API'}</a>{'.'}</Prose>

    <Prose>{'A full-rank '}<Code>{'inverse_transform(transform(X))'}</Code>{' reconstructs sensor values to numerical precision even when the components are unhelpful. Any invertible change of coordinates can do that. Reconstruction checks algebra and information retention; the reference comparison asks about this application. If you reduce the component count, reconstruction instead returns the retained subspace contribution plus the mean.'}</Prose>

    <Prose>{'To study stability, compare several predeclared seeds and non-overlapping recording blocks, align component sign/permutation before comparing them, and report all runs. Use development data for any method or component-count choice. A final test interval stays untouched until those choices are frozen. Randomly mixing neighboring samples across folds would make the rows seem more independent than the recording permits. The provider’s offline filtering also means this extract cannot establish a real-time pipeline’s performance.'}</Prose>

    <H2>{headings[6]}</H2>

    <Prose><strong>{'Return here after the first-pass example.'}</strong>{' An ICA representation has useful ambiguities even in its ideal identifiable setting. Since'}</Prose>

    <MathBlock>{'x = \\sum_j a_j s_j,'}</MathBlock>

    <Prose>{'multiplying sⱼ by any nonzero c, and dividing aⱼ by c, leaves every observation unchanged. Negative c includes a sign flip. Reordering sources and the corresponding columns also preserves the sum. ICA therefore identifies sources up to scale, sign and permutation. A unit-variance convention fixes scale in a useful way, but component index and polarity are still conventions. The model supplies no default ranking by explained variance.'}</Prose>

    <Prose>{'For sensor reconstruction, the '}<strong>{'mixing column'}</strong>{' aⱼ tells where a component contributes. The '}<strong>{'unmixing row'}</strong>{' bⱼᵀ tells which sensor combination estimates it. These are dual operations, not equal vectors: for our A, mixing column 1 is (2, 1)ᵀ, while inverse row 1 is (2, −1)/3. Plotting an inverse row as if it were a spatial contribution pattern reverses the meaning.'}</Prose>

    <Prose>{'In the four-state example, each unit-variance source contributes ‖aⱼ‖² = 5 to the sum of sensor variances. More generally the contribution is Var(sⱼ)‖aⱼ‖², provided the sources are uncorrelated and amplitudes use compatible sensor units. Doubling a source and halving its column leaves that product unchanged. The column norm alone is not a universal energy measure across normalization conventions.'}</Prose>

    <Prose>{'Suppose a domain investigation identifies component 2 as an unwanted contribution. Removing it means setting its scores to zero and reconstructing:'}</Prose>

    <MathBlock>{'x_{\\rm kept} = x - a_2 s_2.'}</MathBlock>

    <Prose>{'At the worked instant s = (1, −1)ᵀ, observed x = (1, −1)ᵀ. Component 2 contributes (−1, −2)ᵀ; subtracting that contribution leaves (2, 1)ᵀ, the first source’s contribution. The altered result is not expected to equal the original sensors.'}</Prose>

    <Prose><strong>{'Investigation C1 — edit a contribution.'}</strong>{' Enter a new two-source amplitude pair and choose which component to keep. Record a predicted sensor amplitude before reconstructing. Observe the contribution vectors, their sum, and the removed difference. Then rescale a source and inversely rescale its mixing column: the reconstructed observations should stay fixed.'}</Prose>

    <IcaContributionLab />

    <Prose>{'In EEG/MEG practice, a component’s time course, spatial pattern and relationship to an auxiliary eye or cardiac channel can help identify an artifact candidate. A statistical component label such as “blink-like” is an interpretation based on this evidence. It is not an anatomical source location or guaranteed neurophysiological cause. Removing a component also removes any wanted activity it contains, which is why before/after task-signal checks and sensitivity to exclusion choices matter. '}<a href="https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html">{'MNE’s artifact tutorial'}</a>{' shows this inspect–exclude–reconstruct workflow. Detailed filtering, referencing, rank changes and experimental leakage belong to the planned '}<a href="/learn/path/full-curriculum/neural-preprocessing-artifact-rejection-and-leakage-safe-pipelines?module=computational-neuroscience">{'Neural Preprocessing, Artifact Rejection and Leakage-Safe Pipelines'}</a>{' lesson.'}</Prose>

    <H2>{headings[7]}</H2>

    <Prose><strong>{'This branch connects the mechanism to the wider ICA literature.'}</strong>{' Read it when you want to distinguish a model, an estimation objective and an algorithm for that objective.'}</Prose>

    <H3>Independence, entropy and likelihood</H3>

    <Prose>{'For continuous components with suitable finite entropies, total dependence can be expressed as'}</Prose>

    <MathBlock>{'I(y_1,\\ldots,y_k) = \\sum_j H(y_j) - H(y).'}</MathBlock>

    <Prose>{'This is the KL divergence between the joint density and the product of its marginals, so it is nonnegative and zero exactly at mutual independence. It is commonly called total correlation or multi-information when there are more than two components. ICA seeks to '}<strong>{'minimize'}</strong>{' it.'}</Prose>

    <Prose>{'For a whitened vector z and square orthogonal W, y = Wz has unit marginal variances, and H(y) = H(z) + log|det W| = H(z). The Gaussian entropy reference is fixed too. Therefore'}</Prose>

    <MathBlock>{'\\sum_j J(y_j) = \\text{constant} - \\sum_j H(y_j),'}</MathBlock>

    <Prose>{'so maximizing the sum of marginal negentropies is equivalent to minimizing total dependence under these conditions. This is the formal connection behind section 4. Maximizing the sum of marginal entropies would point in the opposite direction when joint entropy is fixed.'}</Prose>

    <Prose>{'A likelihood formulation starts by choosing source densities pⱼ. For square invertible B, a change of variables gives'}</Prose>

    <MathBlock>{'p_x(x) = \\lvert \\det B \\rvert \\prod_j p_j\\big(b_j^{\\mathsf T}(x - \\mu)\\big).'}</MathBlock>

    <Prose>{'For independent observation vectors, the dataset log likelihood is n log|det B| + Σᵢⱼ log pⱼ(bⱼᵀ(xᵢ − μ)). For temporally dependent recordings, that sum is a marginal fitting contrast rather than the full time-series joint likelihood. The determinant accounts for how a linear transformation changes volume. Without it, changing scale could appear beneficial for the wrong reason. Incorrect source-density choices can also change the estimator.'}</Prose>

    <Prose>{'Infomax connects an appropriately chosen nonlinear output transformation to entropy maximization and this likelihood perspective. It is an alternative estimation route, not a claim that deterministic input–output mutual information equals dependence among recovered coordinates. FastICA is a fixed-point algorithm tied to specified contrasts. Picard is another optimizer using preconditioning and an approximate Hessian; its published comparisons concern stated objectives and datasets, not a universal speed or stability ranking. The lineage from early adaptive separation, Comon’s ICA formulation, Infomax and fixed-point methods helps explain why several algorithms share the ICA name. '}<a href="https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf">{'Canonical book, chapters 7–14'}</a>{', '}<a href="https://arxiv.org/abs/1706.08171">{'Picard paper'}</a>{'.'}</Prose>

    <H3>When another model is needed</H3>

    <LessonTable caption="Situations that break a condition of the instantaneous noiseless model"
      headers={['Situation', 'What changes in the reasoning?']}
      rows={[
        ['Additive sensor noise x = As + ε', 'Whitened covariance includes noise; sample directions can be biased or noise-amplified. A noisy latent-variable model can make that assumption explicit. Discarding low-variance coordinates may also discard a weak wanted source.'],
        ['Several Gaussian sources with different temporal structure', 'Marginal non-Gaussianity cannot identify their Gaussian subspace. Methods based on several lagged covariance matrices, such as SOBI, use information ordinary FastICA ignores. Distinct lag profiles and their assumptions must be established.'],
        ['Delays or reverberation', 'The model becomes xₜ = Σ_ℓ A_ℓ s₍ₜ₋ℓ₎. Multiplying by one instantaneous inverse generally leaves delayed terms. A convolutive source-separation method addresses this different model.'],
        ['More sources than sensors', 'A rectangular underdetermined mixture cannot be inverted to recover arbitrary source values. Sparsity or other additional structure can support specialized methods. Disabling an orthogonality constraint does not solve that counting problem.'],
        ['Nonlinear mixing', 'An expressive encoder can reconstruct data without identifying independent generating causes. Nonlinear ICA needs additional identifiable structure; an ordinary autoencoder or VAE is not automatically a source-separation solution.'],
        ['Nonnegative additive data', 'NMF constrains factors to be nonnegative. That is a different structural assumption from ICA independence, with its own nonuniqueness and interpretation questions.'],
      ]} />

    <Prose>{'For the delay row, a two-tap example makes the issue concrete: xₜ = Asₜ + Cs₍ₜ₋₁₎. Even if A is known, A⁻¹xₜ = sₜ + A⁻¹Cs₍ₜ₋₁₎. The extra term survives. The planned '}<a href="/learn/path/full-curriculum/source-separation-audio-denoising-demucs-band-split-rnn?module=nlp-cv-multimodal">{'Source Separation & Audio Denoising'}</a>{' topic will provide the application route beyond the instantaneous model.'}</Prose>

    <H3>Cost and reproducibility</H3>

    <Prose>{'For n observations and d sensors, forming a dense covariance and diagonalizing it costs approximately O(nd² + d³), with the exact method and shape affecting the practical choice. SVD can avoid explicitly forming that covariance. After retaining k dimensions, each parallel FastICA iteration costs O(nk² + k³): projected/nonlinear averages plus symmetric orthogonalization. Deflation has repeated data passes and projections on earlier directions. More iterations, components and samples all add work.'}</Prose>

    <Prose>{'Storing X requires O(nd) values; a dense d × d float64 covariance requires 8d² bytes—800,000,000 bytes at d = 10,000. This is a storage calculation, not a timing benchmark. There is no universal “five times components squared” sample threshold that ensures reliable recovery. Distribution shape, dependence, noise, conditioning and the intended error criterion affect the data requirement.'}</Prose>

    <Prose>{'The real example here uses only four sensor channels and 12,000 training instants. It is a small CPU exercise. Set an iteration cap, retain convergence warnings, inspect rank, and record versions and seeds. If convergence is poor, diagnose scaling, rank, outliers, contrast and model mismatch before merely raising the cap. A fixed seed makes the computational starting point reproducible; it does not make the estimate insensitive to changed data.'}</Prose>

    <H2>{headings[8]}</H2>

    <Prose>{'Attempt each task before opening its hint and solution. The calculations here change the demonstrated numbers or the decision being made.'}</Prose>

    <Practice title="1. A new sensor recipe"
      hint={<Prose>{'Subtract the second sensor equation from the first. To preserve observations after rescaling, alter the corresponding column, not the corresponding row.'}</Prose>}
      solution={<Prose>{'The equations are 3s₁ + s₂ = 5 and s₁ + s₂ = −1. Subtracting gives 2s₁ = 6, hence s₁ = 3, s₂ = −4. Replace s₁ by 6 and column 1 by (1.5, 0.5)ᵀ, leaving column 2 at (1, 1)ᵀ. The reconstructed readings are 9 − 4 = 5 and 3 − 4 = −1. Changing a row instead would alter a sensor recipe and would not implement this ambiguity.'}</Prose>}>
      <Prose>{'You observe x = (5, −1)ᵀ under A = [[3, 1], [1, 1]]. Recover the two source values. Then give one different source/mixing pair that generates exactly the same observations, using scale ambiguity.'}</Prose>
    </Practice>

    <Practice title="2. Equal kurtosis, changed mixing weights"
      hint={<Prose>{'Use a² + b² = 1 for variance and fourth powers for the excess kurtosis.'}</Prose>}
      solution={<Prose>{'Variance is 1. Kurtosis is 3(9/16 + 1/16) = 30/16 = 1.875. Equal weighting gives 1.5 and a pure source gives 3. The equal marginal kurtoses are compatible with a directional contrast; there is no flat ring. The value 1.875 is the exact independent variation to reproduce in the rotation investigation at 30° from a source axis.'}</Prose>}>
      <Prose>{'Two independent standardized Laplace sources have excess kurtosis 3. A unit projection uses weights a = √3/2, b = 1/2. Find its variance and excess kurtosis. Compare it with equal weighting and a pure source. Explain whether the equal source kurtoses make separation impossible.'}</Prose>
    </Practice>

    <Practice title="3. Repair two plausible implementations"
      hint={<Prose>{'For A, replace E[z(wᵀz)] by E[zzᵀ]w. For B, consider two initializations entering the same attraction region.'}</Prose>}
      solution={<Prose>{'A gives r = Iw − w = 0, so normalization is undefined. Variance has no preferred direction after whitening; use a suitable nonquadratic contrast. B can converge repeatedly to the same direction. Subtracting the already found direction only at the end can leave a near-zero residual, and the intermediate search never respected the constraint. Orthogonalize inside every iteration before normalizing, or use a symmetric multi-component algorithm.'}</Prose>}>
      <Prose>{'Program A uses g(u) = u on whitened data. Program B estimates every row independently and subtracts previously found directions only after each row has converged. Explain the failure mechanism in each and state a repair.'}</Prose>
    </Practice>

    <Practice title="4. Choose without looking at the answer interval"
      hint={<Prose>{'The test column is for the already fixed choice, even if a different coordinate looks more attractive there.'}</Prose>}
      solution={<Prose>{'Select component 1 using |−0.60| = 0.60 and report test |0.15| = 0.15. Reporting 0.80 would evaluate a selection made with test information. A revised selection rule becomes a new method to develop and evaluate on fresh held-out data. An actual polarity reversal between intervals is also a useful stability finding; taking absolute values was a declared diagnostic choice, not a way to erase it from investigation.'}</Prose>}>
      <Prose>{'A new recording gives these '}<strong>{'signed'}</strong>{' correlations with an external reference:'}</Prose>
      <LessonTable caption="Signed correlations with an external reference, for a new recording"
        headers={['Component', 'Development', 'Test']}
        rows={[['1', '−0.60', '0.15'], ['2', '0.45', '−0.80'], ['3', '0.20', '0.30']]} />
      <Prose>{'The protocol is “choose largest development absolute correlation, then report the test absolute correlation.” Which coordinate and result belong in the report? A colleague wants to change the selection after seeing the test column. What should happen next?'}</Prose>
    </Practice>

    <Practice title="5. A nuisance component contains wanted activity"
      hint={<Prose>{'Multiply the whole component by its mixing column before separating wanted and nuisance terms.'}</Prose>}
      solution={<Prose>{'Exclusion removes a bₜ + 0.2a qₜ, including wanted signal (0.4qₜ, −0.2qₜ)ᵀ. In this simulation, compare reconstructed task amplitudes or task-event recovery with the known qₜ before and after exclusion. On measured data, use a justified task endpoint, auxiliary information and sensitivity to plausible exclusion sets. Reduced visible artifact amplitude alone does not answer the task-preservation question.'}</Prose>}>
      <Prose>{'In a simulation, the true task signal is qₜ, but an ICA candidate is uₜ = bₜ + 0.2qₜ, where bₜ is nuisance activity. Its mixing column is a = (2, −1)ᵀ. What task contribution is removed when the entire candidate is excluded? Propose a check before deciding whether that removal is acceptable.'}</Prose>
    </Practice>

    <Practice title="6. Design a modest follow-up"
      hint={<Prose>{'Changing only the seed measures one kind of variability. A later block and another participant ask different questions.'}</Prose>}
      solution={<>
        <Prose><strong>{'Example solution and success criteria. '}</strong>{'Predeclare several later non-overlapping blocks and the same fitting/development/test durations; carry all four-channel inputs and the same fixed settings into each. Report each method’s selected-coordinate diagnostic for every block, including failures to converge. Keep participant identity separate, and reserve different participants for a future cross-person claim.'}</Prose>
        <Prose>{'If PCA’s advantage reverses across blocks or all correlations collapse, revise the original finding to describe its dependence on that short interval. A good answer states the question, respects information availability, keeps the baseline, records unsuccessful runs, and distinguishes within-recording robustness from population generalization. A new complete clinical study is outside this small exercise.'}</Prose>
      </>}>
      <Prose>{'Keep the real-data program’s fitting and selection boundary. Propose a follow-up that asks whether its finding persists, without using the existing test result to choose a favorable replacement. State the unit of evaluation and one outcome that would make you revise the conclusion.'}</Prose>
    </Practice>

    <H2>{headings[9]}</H2>

    <Prose>{'You are ready to move on from the first-pass route when you can explain why the whitened diamond is dependent, calculate a fourth-moment contrast on changed weights, trace a normalized FastICA update, and keep fitting, coordinate selection and evaluation distinct in the recording example. After the deeper component-removal branch, also distinguish a mixing column from an unmixing row and predict what component exclusion subtracts from the sensors.'}</Prose>

    <Prose>{'Try these from memory: What assumption makes an orthogonal search sufficient after whitening? Why can two Gaussian sources rotate without changing the observed model? Why does exact reconstruction say little about source usefulness? Why do component numbers need matching across fits?'}</Prose>

    <Prose>{'Next, '}<a href="/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml">{'Non-Negative Matrix Factorization (NMF)'}</a>{' asks what changes when both factors must be nonnegative and combine additively. That constraint can suit counts or magnitudes. It supplies a different factorization goal; physical meaning and uniqueness still need evidence.'}</Prose>

    <H2>{headings[10]}</H2>

    <ul>
      <li><strong>{'Hyvärinen & Oja — '}<a href="https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf">{'Independent Component Analysis: Algorithms and Applications'}</a></strong>{'. Free author-hosted tutorial, useful after sections 3–5. Sections 2–6 connect identifiability, non-Gaussianity, whitening and fixed points. The relevant model and algorithm passages were read; examples use older notation/software context.'}</li>
      <li><strong>{'Hyvärinen, Karhunen & Oja — '}<a href="https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf">{'Independent Component Analysis'}</a></strong>{'. Author-hosted book manuscript. Chapters 6–10 deepen whitening and estimation objectives; chapters 13, 15–19 and 22 organize practical issues, noise, temporal structure, convolutive mixing and brain-imaging applications. The contents and selected relevant passages were inspected, not the entire book. Matrix calculus and probability are useful for its proofs.'}</li>
      <li><strong>{'Andrew Ng / Stanford — '}<a href="https://see.stanford.edu/Course/CS229/45">{'CS229 Lecture 15'}</a>{', '}<a href="https://www.youtube.com/watch?v=QGd06MTRMHs">{'YouTube recording'}</a>{', and '}<a href="https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture15.html">{'substantive transcript'}</a></strong>{'. Another route through the mixture model, Gaussian symmetry and likelihood/CDF reasoning. The official bookmarks locate ICA at 39:49 and the algorithm at 47:41. The ICA transcript and companion notes were reviewed; the video/audio was not watched or evaluated. Use the current lesson/API reference for software details, and expect transcription errors in formulas.'}</li>
      <li><strong>{'Andrew Ng — '}<a href="https://cs229.stanford.edu/notes2021fall/cs229-notes11.pdf">{'CS229 ICA notes'}</a></strong>{'. A short mathematical alternative for section 8’s likelihood route. The model, ambiguities, Gaussian example and change-of-variables/likelihood derivation were inspected. This is not a FastICA implementation tutorial.'}</li>
      <li><strong>{'scikit-learn — '}<a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html">{'FastICA reference'}</a></strong>{'. Consult after running the real example for '}<Code>{'components_'}</Code>{', '}<Code>{'mixing_'}</Code>{', whitening and iteration semantics. Parameter and attribute sections were checked against the installed 1.9.1 snapshot. Moving stable documentation can change.'}</li>
      <li><strong>{'MNE — '}<a href="https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html">{'Repairing artifacts with ICA'}</a></strong>{'. An application tutorial showing fitted decompositions, component inspection, auxiliary-channel evidence and exclusion/reconstruction. Filtering, fitting and component-identification passages were reviewed, not executed here. It assumes knowledge of EEG/MEG recordings and uses a separate MNE API.'}</li>
      <li><strong>{'Jezewski and colleagues / PhysioNet — '}<a href="https://physionet.org/content/adfecgdb/1.0.0/">{'Abdominal and Direct Fetal ECG Database, v1.0.0'}</a></strong>{'. The source of the actual simultaneous measurements. Read the acquisition description before interpreting the example. The local extract is distributed with '}<a href="/learn-assets/ica/data-provenance.md">{'its attribution and calibration'}</a>{' under '}<a href="https://opendatacommons.org/licenses/by/1-0/">{'ODC-By 1.0'}</a>{'.'}</li>
      <li><strong>{'Ablin, Cardoso & Gramfort — '}<a href="https://arxiv.org/abs/1706.08171">{'Faster Independent Component Analysis by Preconditioning with Hessian Approximations'}</a></strong>{'. Advanced alternative-optimizer reading after section 8. The abstract and author description of its objective/preconditioner were checked; this lesson did not reproduce its benchmarks. Treat speed comparisons as specific experimental results.'}</li>
    </ul>
  </div>,
};

export default icaContent;
