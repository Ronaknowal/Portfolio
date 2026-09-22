import { Prose, H2, H3, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock as SharedMathBlock } from '../../components/content/Math.jsx';
import { LessonTable } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { VectorFunctionFigure, ConditioningSliceFigure, ObservationMatrixFigure, KernelGeometryFigure, HistoricalForecastFigure, ProbeChoiceFigure } from '../../components/lesson-labs/GaussianProcessFigures.jsx';
import { GaussianConditioningLab, GaussianForecastLab, GaussianProbeLab } from '../../components/lesson-labs/GaussianProcessLabs.jsx';
import { gaussianProcessExamples } from '../gaussian-process-examples.js';

function MathBlock({ children }) {
  return <div className="gp-equation" role="region" tabIndex={0} aria-label="Equation; scroll horizontally if needed"><SharedMathBlock>{children}</SharedMathBlock></div>;
}

export default {
  title: 'Gaussian Processes (GP)',
  readTime: '~60 min read + 90 min practice; optional deeper branches ~45 min',
  hasIntegratedGuide: true,
  content: () => <div className="gp-lesson">
<Prose>{"A temperature probe gives you two readings along a pipe. You want the temperature between them, but you also need to decide where another measurement would be useful. A curve alone leaves out the second question. A Gaussian process lets you express which curves are plausible, update those beliefs with observations, and inspect uncertainty at places you have not measured."}</Prose>

<Prose>{"The key move is to describe how "}<strong>{"function values vary together"}</strong>{". If nearby temperatures usually move together, one reading tells us something about its neighbors. How far that information travels is a modeling decision, encoded in a covariance function."}</Prose>

<Prose><strong>{"First pass:"}</strong>{" follow sections 1–7, including the two-observation calculation and the CO₂ experiment. You should finish able to explain a GP prediction, distinguish its two uncertainty bands, run a small regression, and recognize a misleading forecast. Section 8 explores measurement selection; section 9 develops classification, scalable inference, and the connection to kernel ridge regression. Those branches need the core equations but are optional on a first reading."}</Prose>

<Prose>{"You need vectors, matrix multiplication, an average, and the idea of a normal distribution. A normal variable has a center called its mean and spread described by its variance; standard deviation is the square root of variance. We introduce the required Gaussian conditioning operation here. The previous CRF lesson modeled dependent discrete labels. Here the dependent quantities are numerical function values, and Gaussian algebra makes the basic regression calculation exact."}</Prose>

<H2>{"1. A distribution over function values"}</H2>

<Prose>{"Imagine recording a possible temperature at each of three positions. One possible state is the vector "}<Code>{"[1.0, 0.6, −0.2]"}</Code>{". Another is "}<Code>{"[−0.5, 0.1, 0.4]"}</Code>{". Drawing many such vectors and joining values at their positions gives many possible curves. The lines joining the points are a drawing convention; a finite drawing is not the entire continuous function."}</Prose>

<VectorFunctionFigure />

<Prose>{"A "}<strong>{"Gaussian process"}</strong>{" is a collection of random variables, one for each input, such that every finite collection has a joint multivariate normal distribution. We write"}</Prose>

<MathBlock>{"f\\sim\\operatorname{GP}(m,k),\\qquad\n(f(x_1),\\ldots,f(x_n))^T\\sim\\mathcal N(m_X,K_{XX})."}</MathBlock>

<Prose>{"Here "}<Math>{"m(x)"}</Math>{" is the prior mean at input "}<Math>{"x"}</Math>{", and "}<Math>{"K_{ij}=k(x_i,x_j)"}</Math>{" is the covariance between two function values. The word “process” does not require time: inputs could be positions, material compositions, or settings of an expensive simulator. “Gaussian” describes distributions of function values, not a requirement that the inputs form a bell curve."}</Prose>

<Prose>{"Covariance records whether deviations from the mean tend to move together. Its diagonal entries are variances. If outputs are degrees Celsius, covariance has units °C². Correlation divides covariance by the two standard deviations and is dimensionless. A covariance need not lie between zero and one, and negative covariance can be valid."}</Prose>

<Prose>{"There is a constraint: any finite kernel matrix must be symmetric and positive semidefinite. In plain terms, every weighted combination of its random variables must have nonnegative variance:"}</Prose>

<MathBlock>{"\\operatorname{Var}(a^Tf_X)=a^TK_{XX}a\\geq0."}</MathBlock>

<Prose>{"An arbitrary “similarity” score does not necessarily satisfy this requirement. For example, the symmetric matrix  "}<Math>{"\\begin{bmatrix}1&2\\\\2&1\\end{bmatrix}"}</Math>{" would give the difference of its variables variance "}<Math>{"1+1-2(2)=-2"}</Math>{". It cannot be a covariance matrix."}</Prose>

<Prose>{"There is a familiar finite-dimensional example. Let "}<Math>{"f(x)=a+bx"}</Math>{", with independent "}<Math>{"a,b\\sim\\mathcal N(0,1)"}</Math>{". Every vector of function values is a linear transformation of Gaussian weights, so this is a GP with "}<Math>{"m(x)=0"}</Math>{" and "}<Math>{"k(x,z)=1+xz"}</Math>{". Bayesian linear regression already supplies uncertainty over functions. More flexible kernels let us work without explicitly constructing a large feature vector. This connection is developed from both weight and function viewpoints in "}<a href={"https://gaussianprocess.org/gpml/chapters/RW2.pdf"}>{"GPML, chapter 2"}</a>{"."}</Prose>

<Prose><strong>{"Quick prediction."}</strong>{" For this random-line prior, can a sampled curve have a sudden bend? No: its possible curves are straight lines. Gaussian marginals alone do not mean “anything can happen”; the covariance restricts how values relate."}</Prose>

<H2>{"2. One measurement: the entire mechanism in two numbers"}</H2>

<Prose>{"Separate the underlying quantity "}<Math>{"f(x)"}</Math>{" from a measurement:"}</Prose>

<MathBlock>{"y=f(x)+\\varepsilon,\\qquad \\varepsilon\\sim\\mathcal N(0,\\sigma_n^2)."}</MathBlock>

<Prose>{"The noise has mean zero and is independent of the function. A measurement can lie above or below the underlying curve. With noise, a sensible fitted curve need not pass through every observation."}</Prose>

<Prose>{"Consider two locations, an observed location "}<Math>{"x_0"}</Math>{" and a target "}<Math>{"x_*"}</Math>{". Give both latent values prior mean zero and variance one. Their covariance is ρ. Suppose the measurement noise variance is 0.25 and we observe "}<Math>{"y_0=2"}</Math>{". The measurement has variance "}<Math>{"1+0.25=1.25"}</Math>{". Conditioning gives"}</Prose>

<MathBlock>{"\\underbrace{\\mathbb E[f_*\\mid y_0]}_{\\text{updated center}}\n=\\frac{\\rho}{1.25}\\,2,\n\\qquad\n\\underbrace{\\operatorname{Var}(f_*\\mid y_0)}_{\\text{remaining uncertainty}}\n=1-\\frac{\\rho^2}{1.25}."}</MathBlock>

<Prose>{"For ρ = 0.5, the mean becomes "}<strong>{"0.8"}</strong>{" and the variance becomes "}<strong>{"0.8"}</strong>{". A positively related location moves upward when the observed location is high. The reduction in variance is "}<Math>{"0.5^2/1.25=0.2"}</Math>{": some uncertainty was shared with the measurement and has now been resolved."}</Prose>

<Prose>{"At ρ = 0, the reading changes neither the target mean nor its variance. With no covariance, these jointly Gaussian quantities are independent. Returning to ρ = 0.5, if the reading changes from 2 to −2 while the covariance and noise stay fixed, the target mean changes sign, but its variance stays 0.8. The observed value tells us "}<strong>{"where"}</strong>{" to move; the covariance and observation precision tell us "}<strong>{"how much information"}</strong>{" the measurement contains."}</Prose>

<ConditioningSliceFigure />

<Prose>{"There are two different future questions:"}</Prose>

<LessonTable caption={"Two prediction questions"} headers={["Question","Distribution in this example","What remains uncertain?"]} rows={[[<>{"What is the underlying value "}<Math>{"f_*"}</Math>{"?"}</>,<><Math>{"\\mathcal N(0.8,0.8)"}</Math></>,<>{"The latent function"}</>],[<>{"What would a new measurement "}<Math>{"y_*"}</Math>{" report?"}</>,<><Math>{"\\mathcal N(0.8,1.05)"}</Math></>,<>{"The function plus new independent noise"}</>]]} />

<Prose>{"The second variance is "}<Math>{"0.8+0.25"}</Math>{". A pointwise 95% Bayesian credible interval for the latent value is "}<Math>{"0.8\\pm1.96\\sqrt{0.8}"}</Math>{", approximately [−0.953, 2.553]. The new-observation predictive interval is approximately [−1.208, 2.808]. These statements are conditional on the specified model and hyperparameters. They are not a promise that 95% of an entire curve lies inside a pointwise band, nor automatic frequentist coverage on a different data-generating process."}</Prose>

<GaussianConditioningLab direct />

<H2>{"3. Many measurements: condition one larger Gaussian"}</H2>

<Prose>{"Let "}<Math>{"X"}</Math>{" contain "}<Math>{"n"}</Math>{" observed inputs and "}<Math>{"X_*"}</Math>{" contain "}<Math>{"q"}</Math>{" target inputs. Define "}<Math>{"r=y-m_X"}</Math>{", the observed residual from the prior mean. Let "}<Math>{"R"}</Math>{" be the "}<Math>{"n\\times n"}</Math>{" observation-noise covariance; independent equal-variance noise gives "}<Math>{"R=\\sigma_n^2I"}</Math>{"."}</Prose>

<Prose>{"The joint model and conditional result are"}</Prose>

<MathBlock>{"\\begin{bmatrix}y\\\\f_*\\end{bmatrix}\n\\sim\\mathcal N\\left(\n\\begin{bmatrix}m_X\\\\m_*\\end{bmatrix},\n\\begin{bmatrix}K_{XX}+R&K_{X*}\\\\K_{*X}&K_{**}\\end{bmatrix}\\right),"}</MathBlock>

<MathBlock>{"C=K_{XX}+R,\\qquad\n\\mu_*=m_*+K_{*X}C^{-1}r,\\qquad\n\\Sigma_*=K_{**}-K_{*X}C^{-1}K_{X*}."}</MathBlock>

<Prose>{"Track the shapes: "}<Math>{"C"}</Math>{" is "}<Math>{"n\\times n"}</Math>{"; "}<Math>{"K_{X*}"}</Math>{" is "}<Math>{"n\\times q"}</Math>{"; μ is a "}<Math>{"q"}</Math>{"-vector; Σ is "}<Math>{"q\\times q"}</Math>{". Its diagonal gives marginal variances. Off-diagonal entries tell us how prediction errors at different targets remain related. Drawing an independent error bar at each target does not display that relationship."}</Prose>

<Prose>{"The mean formula starts from the prior and adds an observation-based correction. The covariance formula starts from prior uncertainty and subtracts the part explained by observations. With fixed kernel and noise, adding an independent noisy observation cannot increase the conditional variance. Refitting hyperparameters changes the model itself, so comparisons across refits do not inherit that guarantee."}</Prose>

<Prose>{"Known unequal Gaussian noise is still exact: use "}<Math>{"R=\\operatorname{diag}(\\sigma_1^2,\\ldots,\\sigma_n^2)"}</Math>{". Known correlated Gaussian noise can also be handled with a full "}<Math>{"R"}</Math>{", with appropriate cross-covariance terms if future noise is correlated with past noise. Unknown input-dependent noise requires an additional estimation model; it is not the same problem as simply supplying known variances."}</Prose>

<H3>{"A two-observation calculation"}</H3>

<Prose>{"Use inputs "}<Math>{"X=[0,2]"}</Math>{", values "}<Math>{"y=[1,-1]"}</Math>{", zero mean, noise variance 0.25, and the radial basis function (RBF) kernel"}</Prose>

<MathBlock>{"k(x,z)=\\exp\\left[-\\frac{(x-z)^2}{2\\ell^2}\\right],\\qquad \\ell=1."}</MathBlock>

<Prose>{"The off-diagonal covariance is "}<Math>{"e^{-2}\\approx0.135335"}</Math>{", so"}</Prose>

<MathBlock>{"C=\\begin{bmatrix}1.25&0.135335\\\\0.135335&1.25\\end{bmatrix}."}</MathBlock>

<Prose>{"At the midpoint "}<Math>{"x_*=1"}</Math>{", both cross-covariances are "}<Math>{"e^{-1/2}\\approx0.606531"}</Math>{". The opposite observed values cancel in the mean, giving zero. But their information does not cancel: latent variance falls from one to "}<strong>{"0.468895"}</strong>{". A zero prediction can be an informed estimate rather than an absence of evidence."}</Prose>

<LessonTable caption={"Computed two-observation predictions"} headers={["Target x_*","Posterior mean","Latent variance","New-observation variance"]} rows={[[<>{"0"}</>,<>{"0.775717"}</>,<>{"0.199407"}</>,<>{"0.449407"}</>],[<>{"1"}</>,<>{"0"}</>,<>{"0.468895"}</>,<>{"0.718895"}</>],[<>{"2"}</>,<>{"−0.775717"}</>,<>{"0.199407"}</>,<>{"0.449407"}</>],[<>{"4"}</>,<>{"−0.121112"}</>,<>{"0.985182"}</>,<>{"1.235182"}</>]]} />

<Prose>{"At 4 the data have little influence under this kernel. Farther away, the RBF cross-covariances approach zero: the mean returns to the prior mean and latent variance returns to the prior variance, one. The uncertainty does not disappear because the mean returns to zero."}</Prose>

<ObservationMatrixFigure />
<GaussianConditioningLab />

<H3>{"Compute by solving, not by explicitly inverting"}</H3>

<Prose>{"For a positive-definite "}<Math>{"C"}</Math>{", Cholesky factorization gives "}<Math>{"C=LL^T"}</Math>{". Solve "}<Math>{"Lz=r"}</Math>{" and "}<Math>{"L^T\\alpha=z"}</Math>{". For all targets, solve "}<Math>{"LV=K_{X*}"}</Math>{". Then"}</Prose>

<MathBlock>{"\\mu_*=m_*+K_{X*}^T\\alpha,\\qquad\n\\Sigma_*=K_{**}-V^TV."}</MathBlock>

<Prose>{"This avoids explicitly forming a matrix inverse and reuses the same factor for many predictions. Here is the complete small example. Save it as "}<Code>{"gp_conditioning.py"}</Code>{". In an isolated Python environment install "}<Code>{"numpy==2.3.5 scipy==1.18.1"}</Code>{", then run "}<Code>{"python gp_conditioning.py"}</Code>{"."}</Prose>

<RunnableExample example={gaussianProcessExamples[0]} />
<H3>Match this exact posterior to the library</H3>
<Prose>Append the following to the program above. The length scale is one, the prior mean is zero, hyperparameter optimization is disabled, and <Code>alpha=0.25</Code> adds measurement variance to the training diagonal. With those contracts matched, the scratch Cholesky solve and <Code>GaussianProcessRegressor</Code> agree on means, the full latent covariance and log marginal likelihood.</Prose>
<CodeBlock language="python">{'from sklearn.gaussian_process import GaussianProcessRegressor\nfrom sklearn.gaussian_process.kernels import RBF\n\nlibrary = GaussianProcessRegressor(\n    kernel=RBF(1.0), alpha=0.25, optimizer=None, normalize_y=False,\n).fit(x[:, None], y)\nlibrary_mean, library_covariance = library.predict(\n    targets[:, None], return_cov=True,\n)\nnp.testing.assert_allclose(mean, library_mean, atol=1e-12)\nnp.testing.assert_allclose(covariance, library_covariance, atol=1e-12)\nnp.testing.assert_allclose(log_marginal, library.log_marginal_likelihood(), atol=1e-12)'}</CodeBlock>
<Prose>The distinction between <Code>alpha</Code> and <Code>WhiteKernel</Code> matters: alpha conditions the training solve but is not automatically added to the returned test covariance. For a future noisy measurement, add the measurement variance to its diagonal. A fitted kernel containing a WhiteKernel already includes that white-noise term in the test diagonal; do not add it twice. Numerical jitter may share the training diagonal with measurement noise but is a computational aid, not automatically a future measurement variance. These statements refer to the fixed-kernel, unnormalized setup above; changing normalization changes the units of the covariance. See the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html">estimator's alpha and prediction contracts</a>.</Prose>
<details className="lesson-solution"><summary>Implementation practice: separate latent and observation uncertainty</summary><Prose>Repeat the comparison at variances 0.01 and 1.0. Then replace the library kernel by <Code>RBF(1.0) + WhiteKernel(variance)</Code>, with alpha zero and optimization still disabled. State which returned quantity changes.</Prose><details><summary>Hint and reasoned solution</summary><Prose>Pass the same variance to the scratch <Code>predict</Code> function. Both fits use the same training covariance, so the posterior means remain equal. WhiteKernel adds variance to each test diagonal; cross-covariance to training does not receive that independent noise term. The full test covariance therefore equals the scratch latent covariance plus variance times the identity. Keep the latent result if your target is the underlying function; use the observation result for a new noisy reading.</Prose></details></details>

<Prose>{"The last line checks a conceptual claim: changing only measured values preserves covariance. A materially negative computed variance signals a problem. Tiny negative roundoff may be clipped only within a documented numerical tolerance. If Cholesky fails, investigate an invalid kernel, duplicate noise-free observations, numerical scale, or nearly dependent rows. Small diagonal jitter can stabilize a valid near-singular system, but it changes that system; choose and report it relative to the covariance scale. Do not disguise substantial extra modeled noise as a numerical detail."}</Prose>

<H2>{"4. Kernels express the kinds of change you expect"}</H2>

<Prose>{"For the RBF kernel "}<Math>{"k=\\sigma_f^2\\exp[-r^2/(2\\ell^2)]"}</Math>{", "}<Math>{"r=|x-z|"}</Math>{", "}<Math>{"\\sigma_f^2"}</Math>{" is latent variance and ℓ is a length scale in input units. It is a correlation range, not a period. Smaller ℓ means observations have more local influence. It does not give a periodic prior or force the curve to interpolate noisy measurements."}</Prose>

<Prose>{"In the same two-observation problem, midpoint mean is zero at every listed scale, but uncertainty differs:"}</Prose>

<LessonTable caption={"Length-scale comparison"} headers={["RBF length ℓ","Midpoint latent variance","Mean at target 4","Why it changes"]} rows={[[<>{"0.3"}</>,<>{"0.999976"}</>,<>{"approximately 0"}</>,<>{"Neither observation strongly informs the gap"}</>],[<>{"1"}</>,<>{"0.468895"}</>,<>{"−0.121112"}</>,<>{"Information connects nearby locations"}</>],[<>{"3"}</>,<>{"0.127300"}</>,<>{"−0.867255"}</>,<>{"Long-range dependence strongly constrains the gap"}</>]]} />

<KernelGeometryFigure />

<Prose>{"The following choices answer different modeling questions. The normalized stationary forms below are multiplied by an output variance amplitude when needed."}</Prose>

<LessonTable caption={"Kernel assumptions"} headers={["Kernel","Form or construction","Modeling question"]} rows={[[<>{"RBF"}</>,<><Math>{"e^{-r^2/(2\\ell^2)}"}</Math></>,<>{"Is an extremely smooth latent function plausible?"}</>],[<>{"Matérn 3/2"}</>,<><Math>{"(1+\\sqrt3r/\\ell)e^{-\\sqrt3r/\\ell}"}</Math></>,<>{"Should changes be less smooth than an RBF permits?"}</>],[<>{"Matérn 5/2"}</>,<><Math>{"(1+\\sqrt5r/\\ell+5r^2/(3\\ell^2))e^{-\\sqrt5r/\\ell}"}</Math></>,<>{"Is a smoother, but still finite-smoothness, model appropriate?"}</>],[<>{"Periodic"}</>,<><Math>{"e^{-2\\sin^2(\\pi r/p)/\\ell^2}"}</Math></>,<>{"Should positions one period "}<Math>{"p"}</Math>{" apart share the same latent value?"}</>],[<>{"Linear"}</>,<><Math>{"\\sigma_b^2+\\sigma_w^2xz"}</Math></>,<>{"Could a random intercept and slope explain the function?"}</>],[<>{"Rational quadratic"}</>,<><Math>{"(1+r^2/(2a\\ell^2))^{-a},\\ a>0"}</Math></>,<>{"Would a mixture of RBF length scales help?"}</>]]} />

<Prose>{"Matérn parameter ν controls mean-square differentiability: an integer-order mean-square derivative of order "}<Math>{"j"}</Math>{" exists when ν > "}<Math>{"j"}</Math>{". Thus 3/2 and 5/2 permit one and two such derivatives. RBF permits every order. This is a property of the stochastic model, not something proved by a smooth-looking finite plot. "}<a href={"https://gaussianprocess.org/gpml/chapters/RW4.pdf"}>{"GPML, chapter 4, §§4.1–4.2"}</a>{" develops these distinctions."}</Prose>

<Prose>{"Adding valid kernels gives another valid kernel. If "}<Math>{"f=f_1+f_2"}</Math>{" and the two zero-mean GP components are independent, their covariances add. A trend plus a seasonal effect therefore has a direct probabilistic interpretation."}</Prose>

<Prose>{"Multiplying valid kernels also gives a valid kernel. For example, periodic × RBF preserves seasonal resemblance while making it fade across distant years. This is often called locally periodic covariance. The resulting model is a GP with that product covariance; multiplying two GP sample functions does "}<strong>{"not"}</strong>{" generally produce Gaussian function values."}</Prose>

<Prose>{"For multiple input features, an RBF can use"}</Prose>

<MathBlock>{"k(x,z)=\\sigma_f^2\\exp\\left[-\\frac12\\sum_j\\frac{(x_j-z_j)^2}{\\ell_j^2}\\right]."}</MathBlock>

<Prose>{"A large fitted "}<Math>{"\\ell_j"}</Math>{" means the model changes little across that feature's observed range, holding others fixed. This automatic relevance determination (ARD) is sensitive to units, correlated inputs, and fitting assumptions; it does not establish causal importance. Fit any scaling on training inputs and preserve it for later inputs."}</Prose>

<Prose><strong>{"Try a counterexample."}</strong>{" Under a perfectly periodic kernel, a point ten complete periods away can be strongly related to an observation. “Farther away always means more uncertainty” is therefore an RBF-style intuition, not a universal GP rule."}</Prose>

<H2>{"5. Learn hyperparameters while checking the task you actually care about"}</H2>

<Prose>{"Kernel parameters and noise assumptions affect both the fit and the uncertainty. One way to choose them is the "}<strong>{"log marginal likelihood"}</strong>{". It is the log density of all training observations after integrating out the latent function values:"}</Prose>

<MathBlock>{"\\log p(y\\mid X,\\theta)\n=-\\tfrac12r^TC^{-1}r-\\tfrac12\\log|C|-\\tfrac n2\\log(2\\pi)."}</MathBlock>

<Prose>{"The first term penalizes residuals in directions the covariance considers unlikely. The determinant accounts for the volume over which probability density is distributed. The last term normalizes the Gaussian. A model cannot freely broaden every direction to fit anything without changing how much density it gives the actual observations. Using "}<Math>{"C=LL^T"}</Math>{", compute half the log determinant as "}<Math>{"\\sum_i\\log L_{ii}"}</Math>{", as the program did."}</Prose>

<Prose>{"For our deliberately conflicting observations "}<Code>{"[1, −1]"}</Code>{", log marginal likelihood is −2.861021, −2.952256, and −4.022773 for lengths 0.3, 1, and 3. Among these three fixed candidates, the short scale gives the observations higher density. This is a small comparison, not proof that the short scale is globally optimal or that shorter scales always win."}</Prose>

<Prose>{"Optimizing θ integrates out "}<Math>{"f"}</Math>{" but "}<strong>{"does not integrate out θ"}</strong>{". A single optimized setting is an empirical-Bayes, plug-in choice. Full hyperparameter inference averages predictions over a posterior on θ and can reflect additional uncertainty. Optimization can have local optima; multiple initialized fits and sensible bounds help diagnose this, but do not guarantee a global optimum. "}<a href={"https://gaussianprocess.org/gpml/chapters/RW5.pdf"}>{"GPML, chapter 5, §§5.3–5.4"}</a>{" treats marginal likelihood alongside cross-validation."}</Prose>

<Prose>{"Training marginal likelihood and held-out forecasting answer different questions. A high training density does not show that the next two years will be forecast well. Preserve a development protocol that matches the future use, compare a simple baseline, and keep final test outcomes out of model selection."}</Prose>

<H2>{"6. A real experiment: predict future monthly CO₂"}</H2>

<Prose>{"The NOAA Global Monitoring Laboratory measures atmospheric CO₂ at Mauna Loa. Our offline file contains monthly means for "}<strong>{"1990–1999"}</strong>{", 120 rows. Outputs are parts per million (ppm). These are historical observations, not simulated points. The source marks interpolated months with negative spread/uncertainty fields; none of the selected months has that flag. The retained provenance records the download and public-domain attribution. "}<a href={"https://gml.noaa.gov/ccgg/trends/data.html"}>{"NOAA data and explanation"}</a>{"."}</Prose>

<Prose>{"Question: after observing through December 1997, how well can a small GP forecast the next 24 monthly means? To choose its kernel family, first simulate an earlier decision:"}</Prose>

<ol><li>{"Fit on 1990–1995, 72 months."}</li>
<li>{"Predict 1996–1997, 24 development months; choose the lower mean absolute error (MAE)."}</li>
<li>{"Refit the chosen family on 1990–1997, then evaluate 1998–1999 once."}</li></ol>

<Prose>{"MAE is Σ|actual − predicted| divided by the number of predictions. It has the same units as the observations. We also show root mean squared error (RMSE), which emphasizes larger errors, and count actual observations inside nominal pointwise 95% predictive intervals. The later evaluation lesson develops these metrics more broadly."}</Prose>

<Prose>{"We compare an RBF-only model with a sum of linear, periodic, and RBF components. The latter can express continuing trend, an annual pattern, and smooth departures. Annual period is fixed at one year. Input is "}<Code>{"(year − 1990) + (month − 0.5)/12"}</Code>{", an evenly spaced monthly coordinate. We subtract the training mean and add it back at prediction; future observations do not determine that mean."}</Prose>

<Prose>{"The observation noise variance "}<strong>{"0.09 ppm²"}</strong>{" is a fixed instructional modeling assumption, corresponding to standard deviation 0.3 ppm. It is not NOAA's reported measurement uncertainty or a fitted scientific conclusion. The final test will help expose the limitations of this simple noise model and covariance family."}</Prose>

<Prose>{"Save "}<Code>{"mauna-loa-monthly.csv"}</Code>{" beside the following "}<Code>{"co2_gp.py"}</Code>{" program. The lesson download contains these historical rows and provenance, so running it requires no data fetch. In the environment used for the small example, install "}<Code>{"scikit-learn==1.9.1"}</Code>{" and run "}<Code>{"python co2_gp.py"}</Code>{"."}</Prose>

<RunnableExample example={gaussianProcessExamples[1]} />

<Prose>{"The kernel's amplitude and permitted length scales are optimized during fitting; "}<Code>{"alpha"}</Code>{" adds the fixed training-noise variance. Because these kernels contain no WhiteKernel, "}<Code>{"return_std"}</Code>{" describes the latent process in this setup. The explicit addition of 0.09 constructs the new-observation variance. A model with WhiteKernel has different prediction semantics, so adding its noise again would double count it. "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html"}>{"GaussianProcessRegressor API"}</a>{"."}</Prose>

<Prose>{"The retained author run used Python 3.12.14 and the package versions above:"}</Prose>

<LessonTable caption={"Fixed historical evaluations"} headers={["Evaluation","MAE (ppm)","RMSE (ppm)","Observations inside 95% predictive intervals"]} rows={[[<>{"RBF, development"}</>,<>{"5.432943"}</>,<>{"5.884270"}</>,<>{"14/24"}</>],[<>{"Trend + periodic + RBF, development"}</>,<>{"0.320601"}</>,<>{"0.416399"}</>,<>{"24/24"}</>],[<>{"Selected family refitted, final test"}</>,<>{"1.233775"}</>,<>{"1.312967"}</>,<>{"13/24"}</>]]} />

<Prose>{"The seasonal-naive baseline repeats each month of 1997 for both forecast years. It uses no 1998 observations to predict 1999 and gets test MAE "}<strong>{"3.813333 ppm"}</strong>{". The selected GP predicts levels better in this experiment, but its test uncertainty is much less reliable than the development result suggested."}</Prose>

<HistoricalForecastFigure />

<Prose>{"The count 13/24 is an observed coverage diagnostic for this one correlated time period; it is not an independent-binomial estimate from 24 unrelated cases. Optimized hyperparameters, covariance misspecification, changing growth, and the simplified noise assumptions can all affect coverage. This experiment does not identify a unique cause. A useful next development study would test additional earlier forecast origins and inspect residual structure before choosing richer trend or noise assumptions. Do not tune on these test outcomes and continue calling the same period untouched test data."}</Prose>

<GaussianForecastLab />

<Prose>{"This example connects to "}<a href={"https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html"}>{"scikit-learn's longer CO₂ kernel-design walkthrough"}</a>{". Its locally periodic construction provides a useful extension; the experiment here uses a smaller, independently specified period and a held-out comparison."}</Prose>

<H2>{"7. Practice: calculate, diagnose, and transfer"}</H2>

<H3>{"A. A weaker connection"}</H3>

<Prose>{"Both latent variances are one, observation noise variance is one, the cross-covariance is 0.4, and the observed value is −3. Calculate the posterior mean, latent variance, and a future observation variance with the same noise. What changes if the cross-covariance is zero?"}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"The observed variable's variance includes noise. Use that total in both denominators; add future noise only after calculating latent variance."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"The observed variance is 2. Mean "}<Math>{"0.4(-3)/2=-0.6"}</Math>{"; latent variance "}<Math>{"1-0.16/2=0.92"}</Math>{"; observation variance "}<Math>{"0.92+1=1.92"}</Math>{". Zero cross-covariance gives mean 0, latent variance 1, and observation variance 2. An observation can be extreme while teaching nothing about an independent target."}</Prose></details>

<H3>{"B. A constant prior is a strong claim"}</H3>

<Prose>{"Let "}<Math>{"k(x,z)=1"}</Math>{" everywhere, with zero observation noise. Can this model accommodate two distinct values, 1 and −1, at different inputs? Would a small jitter fix the modeling issue?"}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"Calculate the prior variance of "}<Math>{"f(x)-f(z)"}</Math>{"."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"It is "}<Math>{"1+1-2=0"}</Math>{", so the values must be equal almost surely. The conflicting observations have no support under the model, and the covariance matrix is singular. Added diagonal variance permits observation disagreement only by changing the noise assumptions. Numerical stabilization does not make a constant latent function capable of varying."}</Prose></details>

<H3>{"C. Diagnose a suspicious improvement"}</H3>

<Prose>{"A colleague changes observation values, leaves locations and all hyperparameters fixed, and reports much narrower latent bands. Name a precise check. Then explain when a changed width could be legitimate."}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"Find where "}<Math>{"y"}</Math>{" appears in the conditional covariance formula."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"There is no "}<Math>{"y"}</Math>{" in that formula. Compare the two covariance arrays while holding kernel, noise, input preprocessing, and target grid fixed. In the supplied implementation they must match to numerical tolerance. If the fit also re-estimated kernel/noise parameters or target normalization, the model changed, and widths may legitimately differ. Log those settings before diagnosing the solver."}</Prose></details>

<H3>{"D. A forecast review"}</H3>

<Prose>{"An engineer says, “Our test MAE beats seasonal-naive, so the GP's 95% band is validated.” Write a short correction and one next study using the supplied experiment."}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"Point accuracy and interval performance measure different things. Preserve the used test's status."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"“The GP improves test MAE from 3.813 to 1.234 ppm, but only 13 of 24 test observations are inside its nominal 95% intervals. That period does not support the interval claim.” An appropriate next study uses multiple earlier training cutoffs, forecasts a fixed horizon, and compares residual patterns and interval behavior for predeclared kernels/noise assumptions. Reserve a later, genuinely unused period for a subsequent final evaluation. Do not merely enlarge bands until this test count looks satisfactory."}</Prose></details>

<H3>{"E. Your own kernel proposal"}</H3>

<Prose>{"A sensor signal has a drifting baseline and a daily cycle whose shape slowly changes. Propose a covariance composition; explain how two distant readings at the same hour should relate."}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"The seasonal effect needs both recurrence and decay across days."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"One defensible model is a long-scale RBF or explicit trend component plus periodic("}<Math>{"p=1"}</Math>{" day) × RBF with a longer day-to-day decay scale, plus separately modeled observation noise. Same-hour readings remain strongly related nearby in time, but the periodic component's covariance fades over many days. Validate the decay scale using held-out future periods. Different compositions can be justified if their assumptions and evaluation protocol are explicit."}</Prose></details>

<H2>{"8. Deeper application: where should we measure next?"}</H2>

<Prose>{"Read this branch after you can interpret the conditional covariance. Suppose the pipe experiment's main goal is reducing uncertainty at target "}<Math>{"x_t=1"}</Math>{", and a new noisy reading costs the same at either candidate location. Use the "}<strong>{"current posterior covariance"}</strong>{" "}<Math>{"c_D"}</Math>{", after accounting for existing data. A measurement at candidate "}<Math>{"z"}</Math>{", with independent noise variance "}<Math>{"\\sigma_n^2"}</Math>{", reduces target variance by"}</Prose>

<MathBlock>{"\\Delta(z)=\\frac{c_D(x_t,z)^2}{c_D(z,z)+\\sigma_n^2}."}</MathBlock>

<Prose>{"This is the one-observation update again, now starting from the current posterior rather than the original prior. The candidate's own uncertainty is only part of the decision: it must also inform the target."}</Prose>

<Prose>{"For the two-observation RBF example, candidates 1 and 4 reduce target variance by "}<strong>{"0.305834"}</strong>{" and "}<strong>{"0.001888"}</strong>{", respectively. Location 4 is quite uncertain but weakly related to the target after existing observations. Measuring near the gap is much more useful for this particular goal. With zero target-candidate covariance, the reduction is exactly zero. With very noisy new observations, the reduction tends toward zero even if the locations are related."}</Prose>

<ProbeChoiceFigure />
<GaussianProbeLab />

<Prose>{"An optimization goal is different. If you seek a small objective value, a common acquisition is expected improvement. For a noiseless incumbent "}<Math>{"b"}</Math>{", predictive mean μ and standard deviation "}<Math>{"s>0"}</Math>{", let "}<Math>{"z=(b-\\mu)/s"}</Math>{". Then"}</Prose>

<MathBlock>{"\\operatorname{EI}=(b-\\mu)\\Phi(z)+s\\phi(z),"}</MathBlock>

<Prose>{"where Φ and φ are the standard normal cumulative distribution and density. For "}<Math>{"b=1"}</Math>{", candidate A with μ=0.8, "}<Math>{"s=0.1"}</Math>{" has EI 0.200849; B with μ=1, "}<Math>{"s=0.5"}</Math>{" has EI 0.199471. A slightly wins despite B's greater uncertainty. At "}<Math>{"s=0"}</Math>{", use the continuous limit "}<Math>{"\\max(b-\\mu,0)"}</Math>{". With noisy observations, the best observed value need not be a known latent incumbent; noisy acquisitions must account for that distinction. Expected improvement chooses experiments that might improve an objective, while target-variance reduction chooses experiments that clarify a target. Neither is a universal “pick the most uncertain” rule."}</Prose>

<Prose><strong>{"Transfer check."}</strong>{" Suppose A instead has μ=1.2, "}<Math>{"s=0.1"}</Math>{", while B has μ=1, "}<Math>{"s=0.2"}</Math>{", with incumbent 1. Which has greater expected improvement?"}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"A can improve only through the low tail of a distribution mostly above the incumbent. B is centered at the incumbent, so its first EI term is zero."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"A has "}<Math>{"z=-2"}</Math>{", EI approximately 0.000849. B has "}<Math>{"z=0"}</Math>{", EI "}<Math>{"0.2/\\sqrt{2\\pi}\\approx0.079788"}</Math>{", so B wins. Uncertainty is useful when it creates a meaningful chance of improvement; its effect depends on the mean and objective too."}</Prose></details>

<H2>{"9. Deeper connections and extensions"}</H2>

<H3>{"Classification changes the likelihood"}</H3>

<Prose>{"A Gaussian latent value can drive a class probability through a sigmoid: "}<Math>{"p(y_i=1\\mid f_i)=1/(1+e^{-f_i})"}</Math>{". The likelihood is Bernoulli, so multiplying it by the Gaussian prior no longer produces an exact Gaussian posterior. A Laplace approximation finds the posterior mode "}<Math>{"\\hat f"}</Math>{" and uses local curvature to approximate its shape:"}</Prose>

<MathBlock>{"q(f)=\\mathcal N(\\hat f,(K^{-1}+W)^{-1}),\n\\quad W_{ii}=\\pi_i(1-\\pi_i),\\quad\\pi_i=\\operatorname{sigmoid}(\\hat f_i)."}</MathBlock>

<Prose>{"The prediction integrates the sigmoid over uncertain latent values. Generally "}<Math>{"\\mathbb E[\\operatorname{sigmoid}(f_*)]\\ne\\operatorname{sigmoid}(\\mathbb E[f_*])"}</Math>{". The latter discards uncertainty before converting to probability. A probability close to 0.5 can reflect ambiguous outcomes or uncertain latent values; one probability alone does not separate those causes. Expectation propagation and variational inference provide other approximations. The "}<a href={"https://scikit-learn.org/stable/modules/gaussian_process.html"}>{"scikit-learn GP guide"}</a>{" explains its Laplace classifier and contrasts its multiclass strategies with a direct joint multiclass likelihood."}</Prose>

<H3>{"What makes exact regression expensive?"}</H3>

<Prose>{"For dense "}<Math>{"n\\times n"}</Math>{" covariance, storage is "}<Math>{"O(n^2)"}</Math>{", and one Cholesky factorization is "}<Math>{"O(n^3)"}</Math>{". Hyperparameter fitting repeats expensive evaluations. After fitting, a single mean prediction takes "}<Math>{"O(n)"}</Math>{" algebra beyond kernel evaluation; its variance requires a triangular solve taking "}<Math>{"O(n^2)"}</Math>{". For many targets, batch solves reuse the factor. A full covariance among "}<Math>{"q"}</Math>{" targets additionally needs "}<Math>{"O(q^2)"}</Math>{" output storage and cross-target work. Prediction is not uniformly linear just because the mean is."}</Prose>

<Prose>{"At "}<Math>{"n=50{,}000"}</Math>{", one dense float64 matrix alone occupies "}<Math>{"8n^2=20\\times10^9"}</Math>{" bytes, about 20 GB decimal. Factorization workspaces and copies require more. There is no universal row count at which a GP becomes unusable: kernel structure, precision, hardware, repeated fits, and latency requirements matter."}</Prose>

<Prose>{"Inducing-variable methods summarize the function through "}<Math>{"m\\ll n"}</Math>{" latent values "}<Math>{"u=f(Z)"}</Math>{". The locations "}<Math>{"Z"}</Math>{" need not be a subset of observed inputs. Under a Gaussian model, define "}<Math>{"Q=K_{XZ}K_{ZZ}^{-1}K_{ZX}"}</Math>{". Titsias's variational regression bound is"}</Prose>

<MathBlock>{"\\log\\mathcal N(y;0,Q+\\sigma_n^2I)\n-\\frac{\\operatorname{tr}(K_{XX}-Q)}{2\\sigma_n^2}."}</MathBlock>

<Prose>{"The trace term penalizes latent variance the inducing representation leaves unexplained. It is an approximation objective with a reason for its correction, not a claim that selected points exactly replace all data. Dense inducing calculations commonly involve "}<Math>{"O(nm^2+m^3)"}</Math>{" work. "}<a href={"https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf"}>{"Titsias, 2009, equation 9"}</a>{"."}</Prose>

<Prose>{"Stochastic variational methods keep a distribution "}<Math>{"q(u)"}</Math>{" and optimize an evidence lower bound whose likelihood contribution is a sum over observations. Minibatches estimate that sum, while a KL term compares "}<Math>{"q(u)"}</Math>{" with its prior. This gives a route to large datasets and non-Gaussian likelihoods; approximation quality still depends on the representation and optimization. "}<a href={"https://arxiv.org/abs/1309.6835"}>{"Hensman, Fusi and Lawrence, 2013"}</a>{"."}</Prose>

<Prose>{"Structured kernel interpolation instead approximates covariance using interpolation onto inducing locations with useful grid structure. Fast matrix-vector products can support iterative linear solves, with costs depending on that structure and convergence. "}<a href={"https://proceedings.mlr.press/v37/wilson15.html"}>{"Wilson and Nickisch, 2015"}</a>{". Neither technique makes every arbitrary kernel calculation exact and linear-time."}</Prose>

<Prose>{"Random features approximate a kernel by a fixed feature inner product. Use the "}<strong>{"same sampled feature map"}</strong>{" for training and prediction. If a feature matrix has "}<Math>{"n\\times D"}</Math>{" entries, its storage is "}<Math>{"O(nD)"}</Math>{"; that does not make a dense ridge solve "}<Math>{"O(nD)"}</Math>{". Forming its normal matrix costs "}<Math>{"O(nD^2)"}</Math>{", with "}<Math>{"O(D^3)"}</Math>{" factorization, before choices such as iterative optimization. A Gaussian prior on finite feature weights defines an approximate GP and can retain Bayesian uncertainty; using only point-estimated linear weights does not automatically do so."}</Prose>

<H3>{"The GP–kernel ridge connection, with its boundary"}</H3>

<Prose>{"Kernel ridge regression minimizes"}</Prose>

<MathBlock>{"\\frac1n\\sum_i(y_i-f(x_i))^2+\\lambda\\|f\\|_{\\mathcal H_k}^2."}</MathBlock>

<Prose>{"Its fitted function is "}<Math>{"k(x,X)(K+n\\lambda I)^{-1}y"}</Math>{". For a zero-mean GP with the same fixed kernel and independent Gaussian noise, the posterior mean is identical when "}<strong><Math>{"\\sigma_n^2"}</Math>{" = nλ"}</strong>{". If the loss were a sum instead of an average, the matching factor would change. The deterministic regularization objective alone does not supply the GP's posterior intervals. "}<a href={"https://arxiv.org/pdf/1807.02582"}>{"Kanagawa et al., 2018, Proposition 3.6"}</a>{"."}</Prose>

<Prose>{"There is a subtle difference between a posterior mean and a sampled path. Brownian covariance "}<Math>{"k(s,t)=\\min(s,t)"}</Math>{" on [0,1] has an RKHS of absolutely continuous functions anchored at zero with square-integrable derivative. Brownian sample paths almost surely do not belong to that RKHS, even though its geometry defines the kernel. A fitted mean can be much more regular than a typical random path. This infinite-dimensional example should not be generalized to every finite-rank GP: the random-line GP has paths in its finite-dimensional span. The same "}<a href={"https://arxiv.org/pdf/1807.02582"}>{"survey's §4.1"}</a>{" explains why sample-path membership needs care."}</Prose>

<H3>{"Functions can connect different kinds of observations"}</H3>

<Prose>{"For a sufficiently differentiable kernel, differentiating it gives covariance between a function and a derivative:"}</Prose>

<MathBlock>{"\\operatorname{Cov}(f(x),f'(z))=\\partial_zk(x,z),\\quad\n\\operatorname{Cov}(f'(x),f'(z))=\\partial_x\\partial_zk(x,z)."}</MathBlock>

<Prose>{"For RBF amplitude one, "}<Math>{"\\operatorname{Var}(f'(x))=1/\\ell^2"}</Math>{", with units of output²/input². Slope measurements from a simulator can therefore enter the same joint conditioning system as value measurements. The kernel must support the derivatives being observed; a rough prior cannot be differentiated merely because the program can differentiate its formula away from the diagonal."}</Prose>

<Prose>{"A related extension gives a kernel two output indices, "}<Math>{"k((x,a),(z,b))"}</Math>{", to express covariance between different sensors or tasks. Separately fitting one GP per output assumes away those cross-output connections. These extensions are valuable when there is a defensible relationship between measurements, and require checking the resulting joint covariance rather than treating every input column as interchangeable."}</Prose>

<Prose><strong>{"Connection check."}</strong>{" An average-loss KRR fit has 40 examples and λ=0.025. What noise variance matches its mean under the fixed-kernel, zero-mean GP assumptions? What additional claim would be unjustified from the KRR objective alone?"}</Prose>

<details><summary>{"Hint"}</summary><Prose>{"Keep the factor of "}<Math>{"n"}</Math>{" from the average loss."}</Prose></details>

<details><summary>{"Solution"}</summary><Prose>{"The matching variance is "}<Math>{"40(0.025)=1"}</Math>{", not 0.025. Claiming posterior credible intervals from the deterministic KRR objective alone is unjustified: those intervals need probabilistic assumptions, including the GP prior and observation model."}</Prose></details>

<H2>{"10. References & another way to learn it"}</H2>

<ul><li><a href={"https://gaussianprocess.org/gpml/chapters/RW.pdf"}>{"Rasmussen and Williams, "}<em>{"Gaussian Processes for Machine Learning"}</em></a>{": the free canonical textbook. Chapter 2 develops regression from weights and functions; chapter 4 explains covariance choices; chapter 5 treats model selection. Chapters 3 and 8 extend the core to classification and approximations. Use these after the numerical conditioning example."}</li>
<li><a href={"https://distill.pub/2019/visual-exploration-gaussian-processes/"}>{"Görtler, Kehlbeck and Deussen, "}<em>{"A Visual Exploration of Gaussian Processes"}</em></a>{": an interactive article for seeing joint Gaussians, conditioning, and function samples. Its geometric view is especially useful if matrix notation feels disconnected from the picture. Keep variance and standard deviation distinct when translating a covariance diagonal into a plotted width."}</li>
<li><a href={"https://scikit-learn.org/stable/modules/gaussian_process.html"}>{"scikit-learn, Gaussian processes guide"}</a>{": practical reference for regression, classification, kernels, and API assumptions. Consult it when choosing how to represent known noise or reading prediction output."}</li>
<li><a href={"https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html"}>{"scikit-learn, CO₂ forecasting example"}</a>{": a longer worked kernel-composition example. Compare its locally periodic structure with the simpler fixed experiment here; its reported result is a different experiment."}</li>
<li><a href={"https://mlg.eng.cam.ac.uk/teaching/4f13/1213/lect0304.pdf"}>{"Rasmussen and Ghahramani, Cambridge lectures 3–4"}</a>{": compact lecture notes on the move from Bayesian linear models to GP regression. Suitable after section 3 as another mathematical route."}</li>
<li><a href={"https://arxiv.org/pdf/1807.02582"}>{"Kanagawa et al., "}<em>{"Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences"}</em></a>{": advanced reading for the exact KRR correspondence and distinctions between RKHS functions and sample paths."}</li>
<li><a href={"https://proceedings.mlr.press/v5/titsias09a.html"}>{"Titsias, variational inducing variables"}</a>{", "}<a href={"https://arxiv.org/abs/1309.6835"}>{"Hensman et al., stochastic variational GPs"}</a>{", and "}<a href={"https://proceedings.mlr.press/v37/wilson15.html"}>{"Wilson and Nickisch, structured kernel interpolation"}</a>{": three different mechanisms for scaling inference. Read the mechanism you need rather than treating their complexity statements as interchangeable."}</li></ul>

<Prose>{"The next topic in this module is "}<strong>{"Semi-Supervised Learning"}</strong>{". Here unlabeled locations acquired predictions through a covariance model and observed numerical values. Next, unlabeled examples help classification through assumptions about input geometry, class structure, or agreeing views. A large unlabeled collection is useful only when those assumptions connect its structure to the labels we need. "}</Prose>
<aside className="lesson-intro"><p><strong>Run the two examples offline.</strong> Download <a href="/learn/examples/gaussian-processes-gp/gp_conditioning.py" download>gp_conditioning.py</a>, <a href="/learn/examples/gaussian-processes-gp/co2_gp.py" download>co2_gp.py</a> and <a href="/learn/examples/gaussian-processes-gp/mauna-loa-monthly.csv" download>the NOAA monthly CSV</a>. Keep the CSV beside co2_gp.py. <a href="/learn/examples/gaussian-processes-gp/README.md" download>Data provenance and attribution</a>.</p></aside>
</div>,
};
