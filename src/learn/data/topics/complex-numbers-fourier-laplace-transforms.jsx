import { H2, H3, Prose, Callout } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { ComplexArithmeticLab, RootsUnityFigure, PhasorSynthesisLab, HarmonicProjectionLab, FourierConvergenceLab, PulseTransformFigure, FiniteFourierLab, FourierButterflyFigure, SamplingAliasLab, WindowSpectrumLab, ConvolutionBoundaryLab, FilterResponseLab, LaplaceRegionLab } from '../../components/lesson-labs/ComplexTransformsLabs.jsx';
import { complexTransformExamples } from '../complex-transforms-examples.js';
function Example({
  name
}) {
  const example = complexTransformExamples[name];
  return <section><p className="lesson-note">{example.environment}</p><Prose><strong>Before running.</strong> {example.question}</Prose><RunnableExample example={example} /><Prose>{example.interpretation}</Prose></section>;
}
function Practice({
  title,
  prompt,
  hint,
  children
}) {
  return <section className="lesson-check"><H3>{title}</H3><Prose>{prompt}</Prose><details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Reasoned solution</summary>{children}</details></section>;
}
export default {
  title: 'Complex Numbers, Fourier & Laplace Transforms',
  readTime: '~100 min read + 4 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="complex-transforms-lesson">
    <LessonIntro prerequisites="Use elementary algebra, trigonometry and the preceding calculus and ordinary differential equations lessons. We introduce complex arithmetic and every transform convention locally. The finite DFT and sample experiments require only sums; the continuous derivations use integration." sections={[['1-make-rotation-an-arithmetic-operation', 'Complex numbers and roots'], ['2-describe-an-oscillation-with-a-rotating-arrow', 'Phasors and synthesis'], ['3-recover-the-ingredients-by-projection', 'Fourier coefficients'], ['4-ask-what-a-fourier-series-converges-to', 'Convergence and Gibbs'], ['5-give-a-nonperiodic-signal-a-spectrum', 'Continuous transforms'], ['6-transform-a-finite-list-exactly', 'DFT and FFT'], ['7-distinguish-a-signal-from-its-samples', 'Sampling and aliasing'], ['8-understand-the-window-before-reading-a-spectrum', 'Leakage and normalization'], ['9-predict-the-output-of-a-filter', 'Convolution and response'], ['10-add-exponential-weighting-and-a-region', 'Laplace and convergence'], ['11-preserve-initial-conditions-and-input-timing', 'Solving with Laplace'], ['12-transfer-the-method-and-check-its-limits', 'Practice and next steps']]}>A sensor records a complicated-looking wave. Which oscillations produced it? What survives sampling? What will a filter do immediately after it is switched on? Complex numbers make rotations calculable; Fourier methods recover oscillatory coordinates; Laplace methods also track growth, decay and initial conditions. We will connect those jobs without treating the three tools as interchangeable formulas.</LessonIntro>

    <H2>1. Make rotation an arithmetic operation</H2>
    <Prose>A real number gives a position on a line. A <strong>complex number</strong> z=a+bi gives two coordinates: real part a horizontally and imaginary part b vertically. The symbol i is defined by i²=−1. It is not a measurement error or an instruction to discard a value. It lets ordinary addition and multiplication describe operations on this plane.</Prose>
    <Prose>Add coordinates separately. Multiply by expanding brackets and replacing i² with −1. For z=1+2i and w=2−i, the product is 2−i+4i−2i²=4+3i. Multiplying any a+bi by i gives −b+ai: the point (a,b) turns a quarter-turn counterclockwise about zero. This already makes a sequence of rotations easier to calculate.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &(a+bi)+(c+di)\\
      &\quad=(a+c)+(b+d)i,\\
      &(a+bi)(c+di)\\
      &\quad=(ac-bd)+(ad+bc)i.
    \end{aligned}`}</MathBlock>
    <Prose>The <strong>conjugate</strong> z̄=a−bi reflects the point across the real axis. Multiplying z by z̄ gives a²+b², the squared length. That length is the <strong>modulus</strong> |z|=√(a²+b²). To divide by a nonzero w, multiply numerator and denominator by w̄: z/w=zw̄/|w|². Division by zero remains undefined.</Prose>
    <H3>Length and angle explain the multiplication</H3>
    <Prose>For z≠0, let r=|z| and choose an angle θ from the positive real axis. Then z=r(cos θ+i sin θ). Angles differing by 2π describe the same point. The <strong>principal argument</strong> chooses one representative; this lesson uses −π&lt;θ≤π. Use the signs of both coordinates to select the quadrant, as <code>atan2(b,a)</code> does. The ratio b/a alone cannot distinguish opposite directions, and is undefined when a=0. Zero has a modulus but no direction.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      e^{i\theta}&=\cos\theta+i\sin\theta,\\
      (r e^{i\theta})(\rho e^{i\phi})
        &=r\rho e^{i(\theta+\phi)}.
    \end{aligned}`}</MathBlock>
    <Prose><strong>Euler's formula</strong> is the first line. One justification is to separate the even and odd powers in the absolutely convergent power series for exp(iθ): they are exactly the cosine and sine series. Equivalently, cos θ+i sin θ starts at 1 and has derivative i times itself, the rotation equation. The angle-addition identities then explain the second line: multiply lengths and add angles. Division divides lengths and subtracts angles. A real exponential factor gives eᵘ⁺ⁱᵛ=eᵘ(cos v+i sin v), combining growth or decay with rotation.</Prose>
    <ComplexArithmeticLab />
    <H3>Roots and logarithms need their branches stated</H3>
    <Prose>Solving zⁿ=w is different from asking software for one chosen nth root. If w=ρeⁱᵠ≠0 and n is a positive integer, all n distinct roots have length ρ¹⁄ⁿ and angles (φ+2πk)/n for k=0,…,n−1. For w=0 the only root is 0. In particular, z²=−1 has solutions i and −i, while a principal square-root function selects one of them.</Prose>
    <RootsUnityFigure />
    <Prose>The same periodicity makes the complex logarithm multivalued: all logarithms of z≠0 are ln|z|+i(θ+2πk). A principal value selects an angle; a continuous analytic branch needs a domain that avoids a cut and zero. For example, principal values of log(−1)+log(−1) sum to 2πi, while log(1)=0. Thus a real-valued logarithm identity cannot be applied globally without checking the branch. We use ordinary integer powers and clearly specified phases below, so no hidden root selection is needed.</Prose>
    <Prose><strong>Run the examples.</strong> Save any complete block as <code>transforms.py</code> and run <code>python transforms.py</code>. The arithmetic and direct-sum examples need Python 3.12 or a compatible Python 3. Blocks that use scientific packages name them; install with <code>python -m pip install numpy scipy sympy</code> when needed. Every block includes its own inputs and imports, and its displayed output was obtained from execution. Browser diagrams use bounded JavaScript calculations; the Python programs run in your own environment.</Prose>
    <Example name="complex" />

    <H2>2. Describe an oscillation with a rotating arrow</H2>
    <Prose>An arrow of length A rotating at f turns per second has complex position Aeⁱ⁽²πᶠᵗ⁺ᵠ⁾. Its horizontal coordinate is A cos(2πft+φ). The length is the <strong>amplitude</strong>, f is the <strong>frequency</strong> in hertz, and φ is the <strong>phase</strong> in radians at t=0. Angular frequency ω=2πf measures radians per second. An arrow rotating once per second has f=1 Hz and ω=2π rad/s; confusing those units changes the signal.</Prose>
    <Prose>The fixed complex amplitude Aeⁱᵠ is often called a <strong>phasor</strong>. It stores size and starting angle while the common eⁱωᵗ supplies the time dependence. If several terms have the same frequency, add their phasors before taking the real part. Two equal terms separated by phase π cancel. Terms at different frequencies cannot be replaced by one fixed phasor rotating at a single frequency.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &A\cos(2\pi ft+\phi)\\
        &=\frac{A e^{i\phi}}{2}e^{i2\pi ft}\\
        &\quad+\frac{A e^{-i\phi}}{2}e^{-i2\pi ft}.
    \end{aligned}`}</MathBlock>
    <Prose>A real cosine is the sum of two opposite rotations. Their coefficients are conjugates, so their imaginary parts cancel. Each coefficient has magnitude A/2, not A. That factor will reappear when we read a two-sided Fourier spectrum.</Prose>
    <Prose>Our running signal is <strong>x(t)=2 cos(2πt)+cos(6πt+π/2)</strong>, measured in volts. It combines a 1 Hz tone of amplitude 2 V and a 3 Hz tone of amplitude 1 V. It repeats every second. Its positive-frequency coefficients are c₁=1 and c₃=i/2; the negative-frequency coefficients are c₋₁=1 and c₋₃=−i/2. Those four rotating terms synthesize the entire waveform.</Prose>
    <PhasorSynthesisLab />
    <Prose>A time delay τ replaces t by t−τ. Each frequency coefficient is multiplied by exp(−i2πfτ), meaning the phase change is −2πfτ. The change is proportional to frequency. Changing only the 3 Hz phase changes the wave's shape; it generally does not delay the whole wave. This distinction matters in audio alignment, communication timing and interpreting filter phase.</Prose>
    <Checkpoint prompt="A real cosine has amplitude 6 and phase π/3. What are its two complex coefficients?"><Prose>At +f the coefficient is 3eⁱπ⁄³=1.5+(3√3/2)i. At −f it is the conjugate 1.5−(3√3/2)i. Adding the two rotating terms makes a real signal of amplitude 6. Assigning magnitude 6 to both would double the amplitude.</Prose></Checkpoint>

    <H2>3. Recover the ingredients by projection</H2>
    <Prose>Suppose the ingredients are unknown and only the waveform is given. Try one candidate rotation, multiply by the opposite rotation, and average over a whole period. A matching component stops rotating and accumulates. Other integer harmonics complete whole turns and cancel. This is a <strong>projection</strong>: measure how much a signal points in one basis direction.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      c_k&=\frac1T\int_0^T x(t)e^{-i2\pi kt/T}\,dt,\\
      x(t)&=\sum_k c_k e^{i2\pi kt/T}.
    \end{aligned}`}</MathBlock>
    <Prose>The first line defines a coefficient for any integrable periodic signal. For a finite combination the second line is exact with a finite sum. An infinite series needs the convergence conditions in the next section. A full-period average makes the coefficients carry the same units as x. The k=0 coefficient is the ordinary mean because its exponential factor is 1.</Prose>
    <H3>Why the projection isolates exactly one coefficient</H3>
    <Prose>The average of eⁱ²π⁽ʲ⁻ᵏ⁾ᵗ⁄ᵀ is 1 when j=k. Otherwise its antiderivative has the same value at the two endpoints, because eⁱ²π⁽ʲ⁻ᵏ⁾=1, so the average is 0. These basis functions are <strong>orthonormal</strong> under the inner product ⟨x,y⟩=(1/T)∫x(t)ȳ(t)dt: different directions have zero inner product and each direction has squared length 1. Substituting a finite synthesis sum into the coefficient integral therefore retains just cₖ.</Prose>
    <HarmonicProjectionLab />
    <Prose>At k=3, the running signal leaves i/2. The real projection integrates to zero and the imaginary projection to +1/2. At k=2, both integrate to zero despite nonzero contributions along the way. The program uses a 128-point equally spaced average; it agrees here because this finite trigonometric signal has no frequencies that alias into the selected bins. Such agreement is a property of this fixture, not a promise that any integral can be replaced exactly by 128 samples.</Prose>
    <Example name="projection" />
    <H3>Projection is also the best finite least-squares fit</H3>
    <Prose>For this least-squares argument, assume x is square-integrable over the period, so its average squared magnitude is finite. Choose a finite set K of harmonics and let P_K x denote the sum of their measured coefficients. The residual x−P_K x is orthogonal to every retained harmonic. Any other approximation q using those harmonics differs from P_K x within their span. Expanding the squared norm gives a cross term of zero:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &\|x-q\|^2\\
        &=\|x-P_Kx\|^2+\|P_Kx-q\|^2,\\
      &\|P_Kx\|^2=\sum_{k\in K}|c_k|^2.
    \end{aligned}`}</MathBlock>
    <Prose>Both squared terms are nonnegative, so the projection minimizes average squared error among those finite candidates. For our exact four-component signal the mean square is 1+1+1/4+1/4=2.5 V². Keeping only the ±1 pair leaves error 0.5 V². This is the mechanism behind frequency-selective compression: discarded coefficients quantify lost squared energy under the stated basis and norm, although small squared error need not preserve every perceptually or scientifically important feature.</Prose>

    <H2>4. Ask what a Fourier series converges to</H2>
    <Prose>A finite sum of smooth waves is smooth. Can increasingly many such waves represent a jump? Yes in useful senses, but not uniformly across the jump. Consider a period-one square wave that is +1 for 0&lt;t&lt;1/2 and −1 for −1/2&lt;t&lt;0. Define its value at the jumps as 0 for this example. Odd symmetry removes cosine coefficients. Integrating the sine coefficients over the two half-periods gives 4/(πk) for odd k and 0 for even k.</Prose>
    <MathBlock>{String.raw`S_m(t)=\frac4\pi\sum_{j=0}^{m-1}
      \frac{\sin(2\pi(2j+1)t)}{2j+1}.`}</MathBlock>
    <FourierConvergenceLab />
    <Prose>For a periodic function that is continuously differentiable on finitely many pieces, with finite one-sided function and derivative limits, its symmetric Fourier partial sums converge at each point to the average of the left and right limits. At a continuity point that is the function value. At a jump it is the midpoint, regardless of the value assigned at that single point. Changing one point changes neither the coefficient integrals nor the limiting midpoint.</Prose>
    <H3>A local proof route for the convergence statement</H3>
    <Prose>Use period 1 to keep notation short. Summing the finite geometric series of basis functions gives the <strong>Dirichlet kernel</strong> Dₙ(u)=sin((2n+1)πu)/sin(πu), with its removable value 2n+1 at zero. The nth symmetric partial sum is ∫₋₁⁄₂¹⁄₂x(t−u)Dₙ(u)du. This is an exact finite identity obtained by interchanging a finite sum and an integral.</Prose>
    <Prose>Dₙ is even and its integral over a period is 1. Pair u and −u, and subtract the midpoint of the two limits at t. The remaining integral is ∫₀¹⁄₂q(u) sin((2n+1)πu)du, where q(u) is [x(t+u)+x(t−u)−x(t+)−x(t−)]/sin(πu). Under the stated piecewise differentiability assumptions its numerator is O(u) near zero, canceling the denominator's linear zero; away from zero q is integrable on finitely many pieces.</Prose>
    <Prose>Why does that oscillatory integral tend to zero? Approximate this integrable q in integral absolute error by a finite step function. Each step's integral against sin(λu) is bounded by a constant divided by λ. The approximation error contributes at most its integral absolute error, since |sin|≤1. First make that error small, then let λ grow. This proves the needed vanishing integral and hence the midpoint limit. A plotted partial sum illustrates the conclusion; it does not establish these hypotheses for a different function.</Prose>
    <H3>Gibbs: the peak moves while its height persists</H3>
    <Prose>Differentiating the finite square-wave sum gives 8∑cos(2π(2j+1)t). The geometric-sum identity shows that its first positive zero is t=1/(4m), the first peak. As m grows, that location approaches the jump. Substituting that location into the sum produces a Riemann sum tending to (2/π)∫₀π(sin u)/u du≈1.178980. The excess above +1 is about 0.178980, or <strong>8.949% of the jump height 2</strong>. It is not 8.949% of the upper level.</Prose>
    <Prose>Meanwhile the average squared error is 1−(8/π²)∑ⱼ₌₀ᵐ⁻¹1/(2j+1)² and tends to zero. At each fixed non-jump point the values converge; at the moving peak the overshoot persists. Continuous partial sums cannot converge uniformly to this discontinuous square wave, because a uniform limit of continuous functions is continuous.</Prose>
    <Example name="series" />
    <Prose>For general square-integrable periodic signals, the trigonometric basis is complete and the partial sums converge in the mean-square norm, with total mean square equal to the sum of all squared coefficients. This broader statement is about equivalence up to sets of measure zero, not necessarily pointwise convergence everywhere. A proof extends finite orthogonality using density of trigonometric polynomials in the square-integrable space; one route first approximates continuous periodic functions with positive averaging kernels, then approximates square-integrable functions by continuous ones. The functional-analysis and real-analysis lessons develop those approximation steps.</Prose>
    <Prose>Do not differentiate an infinite series merely because each term is differentiable. If a periodic function is continuous and piecewise continuously differentiable, integration by parts gives the derivative coefficient i2πk cₖ (period 1), with no boundary jump term. To infer pointwise convergence of the differentiated series, sufficient extra regularity is that the derivative itself is piecewise continuously differentiable; at derivative jumps the midpoint qualification applies. The square wave has jumps in the original function, so its formal derivative requires generalized impulses rather than an ordinary pointwise function at those jumps.</Prose>

    <H2>5. Give a nonperiodic signal a spectrum</H2>
    <Prose>A single pulse need not repeat. Instead of one set of integer harmonics, use a continuous frequency variable. With frequency f in hertz, this lesson defines the <strong>continuous Fourier transform</strong> and inverse by:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      X(f)&=\int_{-\infty}^{\infty}x(t)e^{-i2\pi ft}\,dt,\\
      x(t)&=\int_{-\infty}^{\infty}X(f)e^{i2\pi ft}\,df.
    \end{aligned}`}</MathBlock>
    <Prose>The forward integral exists absolutely if ∫|x(t)|dt is finite, written x∈L¹. For the ordinary inverse above, a convenient sufficient condition is that both x and X are integrable; it recovers x at continuity points. Other important signals need weaker limiting or generalized interpretations. A sinusoid continuing forever is not integrable over the whole real line. Treat it with a Fourier series or generalized delta lines, rather than pretending its ordinary transform integral converges.</Prose>
    <LessonTable caption="Choose the domain before choosing a transform" headers={['Given object', 'Frequency coordinates', 'Normalization here']} rows={[['Periodic continuous signal', 'Integer k, physical frequency k/T', 'Coefficient averages divide by T'], ['Integrable nonperiodic signal', 'Continuous f in Hz', 'Forward time integral; inverse frequency integral'], ['Finite list of N values', 'N bins k, frequencies modulo sample rate', 'Unscaled forward sum; inverse divides by N'], ['Exponentially weighted signal', 'Complex s plus a region of convergence', 'Bilateral or explicitly declared unilateral integral']]} />
    <Prose>If x is in volts and t in seconds, X has units V·s. Multiplying X by df, measured in Hz=1/s, restores volts in the inverse. Some books use angular frequency ω instead: the forward exponent is −iωt and the inverse has factor 1/(2π). Neither convention is intrinsically better, but every coefficient, axis and normalization must use the chosen one consistently.</Prose>
    <PulseTransformFigure />
    <Prose>For a centered unit-height pulse of width τ, integrate only from −τ/2 to τ/2. At f≠0 the exponential antiderivative gives sin(πfτ)/(πf)=τ sinc(fτ), where sinc(u)=sin(πu)/(πu) and sinc(0)=1 by continuity. Thus X(0)=τ is the pulse area. Its first zeros are at ±1/τ. Stretching a signal in time compresses its frequency shape and changes its integral scale.</Prose>
    <Example name="pulse" />
    <Prose>The sinc transform of a rectangle is not absolutely integrable, so that rectangle does not satisfy the convenient two-L¹ inversion hypothesis. It can still be inverted as a symmetric improper limit, returning half the jump at each edge. Convolving the rectangle with itself gives a continuous triangle whose transform is the squared sinc: both sides are now integrable, and ordinary inversion applies.</Prose>
    <H3>Why a transform followed by an inverse can recover the signal</H3>
    <Prose>There is a useful proof route that avoids treating an infinite oscillatory integral as an ordinary delta function. Multiply X(f) by e⁻πεᶠ² for ε&gt;0 before inverting. Absolute integrability permits exchanging the integrals. The result is x convolved with gε(t)=ε⁻¹⁄²e⁻πᵗ²⁄ε. This Gaussian has integral 1; as ε→0 its mass concentrates near zero. Splitting the convolution into a small neighborhood and its tail shows it tends to x(t) at a continuity point. If X∈L¹, dominated convergence removes the multiplier on the frequency side, giving the displayed inverse.</Prose>
    <Prose>The Gaussian transform used in that argument follows by differentiating its integral with respect to t, then integrating by parts: the integral solves I′(t)=−(2πt/ε)I(t), and the Gaussian integral gives I(0)=ε⁻¹⁄². Its solution is gε(t). Thus the proof relies on an explicitly normalized concentrating kernel, not on an unexplained picture of an infinitely narrow spike.</Prose>
    <LessonTable caption="Three useful transform rules in the Hertz convention" headers={['Time operation', 'Frequency result', 'Reason / condition']} rows={[['x(t−τ)', 'exp(−i2πfτ) X(f)', 'Substitute u=t−τ; magnitudes stay the same'], ['x(at), a≠0', 'X(f/a)/|a|', 'Substitution changes both frequency and integration scale'], ['(x*h)(t)', 'X(f)H(f)', 'For integrable x,h, exchange integrals and substitute t−u']]} />
    <Prose>For appropriate square-integrable signals, the transform extends to preserve energy: ∫|x(t)|²dt=∫|X(f)|²df. This is the continuous Plancherel identity, understood in L² when an ordinary integral formula does not apply. Our finite coefficient and DFT proofs are exact elementary counterparts; they do not alone prove this infinite-dimensional extension. For complex x, energy uses x times its conjugate, not x².</Prose>

    <H2>6. Transform a finite list exactly</H2>
    <Prose>A computer starts with N samples x[0],…,x[N−1]. The <strong>discrete Fourier transform</strong>, or DFT, is a change of coordinates for that finite vector. It does not require an infinite signal or a convergence theorem. We use an unscaled forward transform and a 1/N-scaled inverse:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      X[k]&=\sum_{n=0}^{N-1}x[n]e^{-i2\pi kn/N},\\
      x[n]&=\frac1N\sum_{k=0}^{N-1}X[k]e^{i2\pi kn/N}.
    \end{aligned}`}</MathBlock>
    <Prose>To prove the inverse, substitute the forward expression. For each input index j, the inner sum is ∑ₖeⁱ²πᵏ⁽ⁿ⁻ʲ⁾⁄ᴺ. It equals N when n=j and 0 otherwise by the finite geometric-series formula. The factor 1/N therefore returns exactly x[n]. Expanding the squared frequency norm and using the same orthogonality gives ∑|x[n]|²=(1/N)∑|X[k]|². No limit or sampling theorem is needed for either identity.</Prose>
    <FiniteFourierLab />
    <Prose>For x=[1,2,0,−1] and k=1, the rotation factors are 1,−i,−1,i. The four weighted contributions are 1,−2i,0,−i, totaling 1−3i. All bins together are [2,1−3i,0,1+3i]. Time-domain squared energy is 1+4+0+1=6. Frequency-domain energy is (4+10+0+10)/4=6.</Prose>
    <Example name="dft" />
    <H3>Bins, physical frequencies and real-signal symmetry</H3>
    <Prose>For samples n/fₛ taken at fₛ samples per second, bin k corresponds to kfₛ/N, modulo fₛ. The upper indices usually represent negative frequencies: for N=8, the conventional order is 0,1,2,3,−4,−3,−2,−1 times fₛ/8. The Nyquist bin at an even N is self-conjugate, and +fₛ/2 and −fₛ/2 coincide on the sample grid. For odd N there is no bin exactly at Nyquist.</Prose>
    <Prose>Real samples satisfy X[N−k]=X[k]̄ because conjugating the forward sum reverses its rotation. DC, and the Nyquist bin when it exists, are real up to numerical roundoff. A bin-centered isolated real cosine of amplitude A has |X[k]|=NA/2 at each of its two interior bins. One-sided amplitude displays multiply by 2/N there, but use 1/N for DC and Nyquist. Off-bin tones, overlapping tones and windows need more care than this isolated-bin rule.</Prose>
    <H3>The FFT computes the same transform with reused work</H3>
    <Prose>A direct DFT has N outputs and N contributions per output: O(N²) arithmetic. A radix-2 <strong>fast Fourier transform</strong> separates even and odd indices for an even N. Each half is a DFT of length N/2. If E[k] and O[k] are those results and W=e⁻ⁱ²π⁄ᴺ, the two outputs use the same rotated odd contribution:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      X[k]&=E[k]+W^kO[k],\\
      X[k+N/2]&=E[k]-W^kO[k].
    \end{aligned}`}</MathBlock>
    <FourierButterflyFigure />
    <Prose>For N a power of two, recurse until length one. Each level does O(N) combines and there are log₂N levels, so the arithmetic cost is O(N log N). The example validates that specific radix-2 requirement. A DFT itself exists for every positive N; production FFT libraries use other decompositions and algorithms for other sizes. FFT is an algorithm for the DFT, not a different spectrum or a guarantee of a particular machine's runtime.</Prose>
    <Example name="fft" />

    <H2>7. Distinguish a signal from its samples</H2>
    <Prose>Exact recovery of a sample vector does not imply exact recovery of the continuous signal that produced it. At sample times n/fₛ, adding an integer multiple of fₛ to a frequency changes the exponential phase by 2π times an integer. Its samples are unchanged. This ambiguity is <strong>aliasing</strong>.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &e^{i2\pi(f+mf_s)n/f_s}\\
      &\quad=e^{i2\pi fn/f_s},\qquad m\in\mathbb Z.
    \end{aligned}`}</MathBlock>
    <SamplingAliasLab />
    <Prose>At fₛ=16, a 13 Hz complex rotation has the same samples as −3 Hz. Consequently cos(2π·13t+π/3) and cos(2π·3t−π/3) agree on that grid. The positive-frequency fold reverses the phase. They are different between sample times. At 32 samples/s this particular pair becomes distinguishable, but some still-higher-frequency candidates can again share those samples.</Prose>
    <Example name="alias" />
    <Prose>A sampling theorem removes ambiguity by restricting the candidate class. For a square-integrable continuous-time signal whose spectrum is zero outside |f|≤B, uniform infinite samples at fₛ&gt;2B determine the signal. With the usual bandlimited representative it can be reconstructed by sinc interpolation; convergence has the appropriate bandlimited L² interpretation. Informally, sampling replicates the spectrum at multiples of fₛ. The strict separation keeps the copies from overlapping, so an ideal low-pass reconstruction can isolate the original.</Prose>
    <Prose>The hypotheses are doing work: the signal must be bandlimited before sampling, the samples must be on the stated uniform grid, and the ideal formula uses all integer sample indices. A finite, noisy record and a truncated interpolation kernel do not satisfy the exact ideal experiment. At f=fₛ/2, sin(2πft) gives zero at every sample, so merely writing “twice the frequency is always sufficient” loses a boundary case.</Prose>
    <Prose>An <strong>anti-alias filter</strong> attenuates unwanted frequencies before they enter the sampler. Once two inputs have produced identical stored samples, a digital filter cannot determine which one was present without extra information. The same issue appears in video wheels seeming to rotate backward, periodic patterns in images and undersampled instrument signals. The mechanism is identical: distinct continuous or spatial patterns coincide on the observation grid.</Prose>

    <H2>8. Understand the window before reading a spectrum</H2>
    <Prose>A finite observation is the continuing signal multiplied by a window that is zero outside the recorded interval. Multiplication in time spreads a tone by the window's frequency transform. A sinusoid that completes a noninteger number of cycles in the record distributes its energy among many DFT bins. This is <strong>spectral leakage</strong>; it is not evidence that the physical source suddenly contains all of those separate tones.</Prose>
    <Prose>Distinguish three choices. The observation count N at fixed sampling rate sets the duration T=N/fₛ. The last recorded time is (N−1)/fₛ; the DFT continuation period is N/fₛ. The window weights w[n] select how strongly each recorded value contributes. The transform length M≥N may include added zeros. Its frequency grid has spacing fₛ/M, but those zeros are not new measurements.</Prose>
    <WindowSpectrumLab />
    <Prose>The default 5.5 Hz tone at 64 samples/s is observed for 1 second with N=64, so it lies halfway between the 5 and 6 Hz bins. Padding to M=256 makes the spacing 0.25 Hz and reveals more points on the same finite-record transform. Recording N=128 observes 2 seconds and actually changes that transform; this tone then completes 11 cycles. A denser grid can help interpolate a peak, but it does not by itself narrow the window's main lobe or separate arbitrary close sources.</Prose>
    <Prose>The periodic Hann window here is w[n]=(1−cos(2πn/N))/2 for n=0,…,N−1. It tapers the boundaries, reducing distant sidelobes relative to a rectangular window while broadening the main lobe. A symmetric Hann uses a different denominator; naming only “Hann” is insufficient to reproduce every numerical detail. A chosen taper trades one kind of spectral ambiguity for another.</Prose>
    <H3>Amplitude and density require different normalization</H3>
    <Prose>For an isolated interior tone under conditions where the opposite-frequency contribution does not contaminate its bin, divide magnitude by the coherent gain Σw and double to combine the real signal's two sides. For squared-density scaling, divide squared magnitude by fₛΣw². These are not the same denominator.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      X_w[k]&=\sum_{n=0}^{N-1}w[n]x[n]e^{-i2\pi kn/M},\\
      P[k]&=\frac{|X_w[k]|^2}{f_s\sum_n w[n]^2},\\
      \Delta f&=\frac{f_s}{M}.
    \end{aligned}`}</MathBlock>
    <Prose>Apply the finite Parseval identity to the M-point zero-padded vector wx. Then (1/M)∑|Xw|²=∑w²|x|², so summing the <em>two-sided</em> density times Δf gives exactly ∑w²|x|²/∑w². For real data, a one-sided version merges the equal positive and negative powers and keeps DC and the even-length Nyquist bin undoubled. Its integral is the same. The units are V²/Hz, and integration gives V².</Prose>
    <Prose>This identity assumes the same declared samples, weights, FFT normalization and <strong>no detrending</strong>. If a routine subtracts a mean or fits a trend first, the identity concerns that altered signal. It is an exact finite energy accounting identity, not a guarantee that one periodogram is an unbiased or consistent estimate of a population power spectral density. Repeated records, uncertainty, nonstationarity and time-frequency analysis belong to the dedicated spectral-analysis material.</Prose>
    <Example name="window" />

    <H2>9. Predict the output of a filter</H2>
    <Prose>A linear time-invariant system responds to a shifted, scaled input by shifting and scaling its response. Think of a discrete input as a sum of weighted unit impulses at different indices. If h[n] is the response to one unit impulse at zero, superposition gives y[n]=∑ⱼx[j]h[n−j]. This weighted overlap is <strong>convolution</strong>. In continuous time the corresponding integral is y(t)=∫x(u)h(t−u)du when the integral and the system model are well defined.</Prose>
    <ConvolutionBoundaryLab />
    <Prose>Substitute the convolution into the Fourier transform. Changing variables v=t−u separates the exponential as e⁻ⁱ²πᶠᵘe⁻ⁱ²πᶠᵛ; the two integrals become X(f)H(f). Integrability conditions justify this exchange. Thus convolution becomes multiplication in frequency. The DFT version uses periodic indexing, so multiplying two same-length DFTs gives <strong>circular</strong> convolution. To obtain finite linear convolution, pad to at least N+L−1 before transforming, where L is the kernel length.</Prose>
    <Example name="convolution" />
    <H3>An exponential mode keeps its shape through a linear filter</H3>
    <Prose>Consider y′+ay=ax with a&gt;0. For a complex input eⁱωᵗ, try a particular solution Heⁱωᵗ. Substitution gives (iω+a)H=a, so H(iω)=a/(a+iω). The output at that frequency is scaled by |H|=a/√(a²+ω²) and shifted by arg H=−atan(ω/a). The frequency stays the same. Superposition handles our two-tone input.</Prose>
    <Prose>With a=2π, the corner frequency a/(2π) is 1 Hz: that tone has gain 1/√2 and lag −π/4. The 3 Hz tone has gain 1/√10 and lag −atan 3. The word “corner” here means the gain is 1/√2 of its zero-frequency value, or half its squared gain. It does not mean frequencies above it are removed completely.</Prose>
    <FilterResponseLab />
    <Prose>That calculation finds the steady periodic <em>particular</em> solution yss. The homogeneous solution Ce⁻ᵃᵗ must still supply the initial condition. Therefore y(t)=yss(t)+[y₀−yss(0)]e⁻ᵃᵗ for t≥0. Our default yss(0)=1.3 V, so starting at y₀=0 adds −1.3e⁻²πᵗ. At t=0.25 s the steady value is 1.1 V while the actual output is about 0.829757 V. A spectrum of the eventual periodic response does not choose what happened at switch-on.</Prose>
    <Example name="filter" />
    <Prose>The same solution follows from an integrating factor: y(t)=y₀e⁻ᵃᵗ+∫₀ᵗae⁻ᵃ⁽ᵗ⁻ᵘ⁾x(u)du. The causal impulse response is h(t)=ae⁻ᵃᵗ for t≥0 and zero before. Its convolution describes the zero-state response; the separate initial-state term is essential when y₀≠0. This agreement between time-domain and frequency-domain routes is a useful correctness check.</Prose>

    <H2>10. Add exponential weighting and a region</H2>
    <Prose>Oscillatory weighting alone may not make an infinite integral converge, especially for a growing or persistent causal signal. The <strong>Laplace transform</strong> adds a real exponential weight. Write s=σ+iω. Then e⁻ˢᵗ=e⁻σᵗe⁻ⁱωᵗ: σ changes the growth/decay balance, while ω supplies rotation. In this section ω is angular frequency, so a Hertz frequency f corresponds to s=i2πf.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      &X(s)=\int_{-\infty}^{\infty}x(t)e^{-st}\,dt,\\
      &s\in\mathrm{ROC}\quad\Longleftrightarrow\\
      &\quad\int_{-\infty}^{\infty}|x(t)e^{-st}|\,dt<\infty.
    \end{aligned}`}</MathBlock>
    <Prose>This is the <strong>bilateral</strong> transform, with its region of absolute convergence, or ROC. For right-sided x(t)=e⁻ᵃᵗ at t≥0, direct integration gives 1/(s+a) only when Re(s+a)&gt;0. For left-sided x(t)=−e⁻ᵃᵗ at t≤0, integration gives the same rational expression only when Re(s+a)&lt;0. Formula plus ROC identifies the intended transform; the formula alone loses the support information.</Prose>
    <LaplaceRegionLab />
    <Prose>At the boundary σ=−a, a nonzero ω leaves a pure oscillation whose finite integral does not converge as its horizon grows. At ω=0 the integral grows linearly. Outside the appropriate half-plane, an exponential factor grows in the direction of integration. A finite-horizon number, or a finite algebraic denominator, does not establish existence of the infinite transform.</Prose>
    <Example name="laplace" />
    <H3>When evaluating on the imaginary axis is legitimate</H3>
    <Prose>If the imaginary axis lies in the bilateral ROC, setting s=iω gives the ordinary Fourier transform in angular-frequency notation. For a causal stable exponential, the ROC includes that axis. For a causal growing exponential it does not, even though the rational formula can be evaluated at many imaginary-axis points. Analytic continuation of a formula beyond its ROC is not the original absolutely convergent transform integral there.</Prose>
    <Prose>For causal initial-value problems we usually use a <strong>unilateral</strong> transform from t=0 onward. A sufficient ordinary-function existence condition is local integrability and an exponential bound |x(t)|≤Meᵅᵗ for large t; Re(s)&gt;α then controls the tail. These are sufficient conditions, not a classification of every transformable object. Ordinary right-sided signals make the unilateral integral coincide with the bilateral transform of their zero extension, but the initial-condition convention must still be declared.</Prose>

    <H2>11. Preserve initial conditions and input timing</H2>
    <Prose>First use ordinary post-initial data at <strong>0+</strong>. Assume y is locally absolutely continuous for t≥0 and y and its derivative have suitable exponential bounds. Integration by parts gives ∫₀∞y′e⁻ˢᵗdt=[ye⁻ˢᵗ]₀∞+s∫₀∞ye⁻ˢᵗdt. In the right half-plane where the boundary at infinity vanishes, the lower boundary contributes −y(0+).</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      \mathcal L\{y'\}&=sY-y(0^+),\\
      \mathcal L\{y''\}&=s^2Y-sy(0^+)-y'(0^+).
    \end{aligned}`}</MathBlock>
    <Prose>The second formula needs the corresponding derivative regularity. These boundary terms are how initial information survives turning derivatives into algebra. For y′+2y=2, y(0+)=3, the equation becomes (s+2)Y=2/s+3, so Y=1/s+2/(s+2). Inverting the familiar pairs gives y(t)=1+2e⁻²ᵗ. Substitution gives y′+2y=2 and y(0+)=3, two independent checks.</Prose>
    <Example name="initial" />
    <H3>A delayed input must keep its clock aligned</H3>
    <Prose>Let u(t−a) be the step that is zero before a and one after a; its value at a single boundary does not affect ordinary integrals. For a≥0, substitute v=t−a to obtain L{'{'}u(t−a)f(t−a){'}'}=e⁻ᵃˢF(s). Both the gate and the function's argument are shifted. The rule does not replace f(t−a) with f(t).</Prose>
    <Prose>For example, u(t−1)t=u(t−1)[(t−1)+1]. Its transform is e⁻ˢ(1/s²+1/s), not e⁻ˢ/s². The latter belongs to a ramp that starts at zero at time 1, while the former turns on at height 1. Drawing the input or rewriting it in the shifted clock prevents this common error.</Prose>
    <Example name="shift" />
    <Prose>Now solve y′+2y=2[u(t−1)−u(t−2)] with y(0+)=1. The initial response is e⁻²ᵗ. A unit-amplitude equilibrium step response is g(t)=(1−e⁻²ᵗ)u(t). The positive input step at 1 contributes g(t−1); switching it off at 2 contributes −g(t−2). Thus y=e⁻²ᵗ+g(t−1)−g(t−2). The state stays continuous at both switches, while its derivative changes to reflect the input jump.</Prose>
    <Example name="delayed" />
    <H3>Repeated poles and impulses encode different mechanisms</H3>
    <Prose>L{'{'}e⁻ᵃᵗ{'}'}=1/(s+a) follows directly from integration. Differentiating that integral with respect to a in a convergent right half-plane gives L{'{'}te⁻ᵃᵗ{'}'}=1/(s+a)². More generally the causal inverse of 1/(s+a)ᵐ is tᵐ⁻¹e⁻ᵃᵗ/(m−1)!. A repeated pole therefore brings a polynomial factor, not merely a second identical exponential term.</Prose>
    <Prose>An ideal <strong>impulse</strong> δ(t) is a generalized object with unit area and zero duration, defined by how it integrates against test functions; it is not an ordinary finite-height input value. For the causal impulse response of y″+2y′+y=δ(t), the pre-impulse state is at rest. Integrating through zero gives a unit jump in y′ while y stays continuous, so h(0+)=0 and h′(0+)=1. After zero, the ordinary homogeneous equation gives h(t)=te⁻ᵗ. Its transform is 1/(s+1)².</Prose>
    <Example name="repeated" />
    <Callout type="warning"><strong>Choose one origin convention.</strong> Our ordinary IVP formulas use 0+ and the state after any impulse at the origin. A generalized unilateral convention beginning at 0− includes the origin impulse, assigns L{'{'}δ(t){'}'}=1 and uses pre-impulse initial values in derivative formulas. Either account for the jump first and solve the post-impulse IVP, or include the impulse with pre-impulse data. Combining its transform with already-jumped initial data counts the same event twice.</Callout>
    <H3>Poles, final values and hidden states need qualifications</H3>
    <Prose>A <strong>pole</strong> is a singularity of the transform, such as s=−a in 1/(s+a). For a causal, proper, rational input-output transfer function after exact cancellations, all finite poles strictly in the left half-plane give a decaying, integrable ordinary impulse-response part. A finite direct feedthrough term may add a delta at zero without destroying bounded-input bounded-output stability. In this rational setting, a pole on or to the right of the imaginary axis prevents that input-output stability. The ROC and causality assumption matter; poles alone do not specify a general bilateral signal.</Prose>
    <Prose>Input-output stability does not necessarily mean every internal state is stable. Consider A=diag(1,−1), input vector B=(0,1), and output C=(0,1). The input excites and the output observes only the decaying second coordinate, giving H(s)=1/(s+1). A nonzero initial first coordinate grows like eᵗ while remaining hidden from that input-output map. State-space controllability, observability and internal stability are distinct questions.</Prose>
    <Example name="hidden" />
    <Prose>The final-value shortcut y(∞)=limₛ→₀sY(s) also needs conditions. For an ordinary causal signal with a strictly proper rational Y(s), a useful sufficient condition requires every pole of sY(s) to lie strictly in the open left half-plane after cancellations; then the remaining transient terms decay. For y(t)=sin t, Y=1/(s²+1), and the algebraic limit sY→0 exists, but poles at ±i violate the condition and sin t has no final value. The shortcut is a theorem with hypotheses, not a way to invent a nonexistent limit.</Prose>

    <H2>12. Transfer the method and check its limits</H2>
    <H3>A short bridge to discrete-time dynamics</H3>
    <Prose>For a sequence, the <strong>Z-transform</strong> replaces a continuous integral by ∑ₙx[n]z⁻ⁿ together with a region where it converges absolutely. The causal sequence aⁿ for n≥0 is a geometric series, giving 1/(1−az⁻¹) when |z|&gt;|a|. On z=eⁱω the unit-circle values give the discrete-time Fourier transform only if that circle is in the ROC. For |a|≥1 it is not, even though a rational expression can still be written.</Prose>
    <Prose>Sampling an exponential eˢᵗ at t=nΔt produces (eˢΔᵗ)ⁿ, so its discrete mode is z=eˢΔᵗ. Negative real parts map inside the unit circle and positive real parts outside. This maps individual modes; it does not make arbitrary continuous and discrete transfer functions interchangeable without a sampling, hold or discretization model. The earlier ODE lesson and subsequent numerical-analysis lesson explain why a solver's stability can differ from the continuous system's.</Prose>
    <H3>Complete the two-tone investigation with changed inputs</H3>
    <Prose>Take x(t)=1.5 cos(4πt+π/4)+0.75 cos(10πt−π/3). Record 64 samples at 32 samples/s, and drive y′+4πy=4πx with y(0)=−0.5. The tones are at 2 and 5 Hz, so the two-second record places them exactly at bins 4 and 10. Their positive-frequency coefficients X[k]/N are 0.75eⁱπ⁄⁴ and 0.375e⁻ⁱπ⁄³. Compute their gains and phase lags, add the transient, then check against a time-domain integral.</Prose>
    <Example name="capstone" />
    <Prose>The recovered coefficients are about 0.530330+0.530330i and 0.187500−0.324760i. The total output at 0.25 s is about −0.901760 V. Those results depend on the declared phases, sampling rate, transform normalization and initial condition. Matching only a spectrum's magnitudes would leave several of those choices unidentified.</Prose>

    <H3>Independent practice</H3>
    <Prose>Attempt these changed cases before opening a hint. The finish line is to choose the representation, state its conditions, calculate the result and check it by another route. A plot or a library call alone is not an explanation.</Prose>
    <Practice title="1. A quarter-turn and a logarithm branch" prompt="Let z=2−3i. Compute iz, |iz| and z/iz. Then explain why principal log(z²) need not equal 2 log(z) for every nonzero z." hint="Multiplication by i turns (a,b) into (−b,a). For the log counterexample choose z=−1.">
      <Prose>iz=3+2i, with modulus √13. Since z≠0, z/(iz)=1/i=−i. For z=−1, principal log(z²)=log 1=0 but 2 log(−1)=2πi under our argument convention. The exponential agrees, but the chosen logarithm values differ by a full angular turn.</Prose>
    </Practice>
    <Practice title="2. Find all roots, not one chosen value" prompt="Find all cube roots of −8i. Check their lengths and cubes." hint="Write −8i as 8 exp(−iπ/2), then divide the angle plus 2πk by three.">
      <Prose>The three roots have length 2 and angles −π/6, π/2 and 7π/6. In coordinates they are √3−i, 2i and −√3−i. Cubing each multiplies its length to 8 and its angle to −π/2 modulo 2π, returning −8i. A software principal cube root gives just the first under this argument convention.</Prose>
    </Practice>
    <Practice title="3. Recover a sine, a cosine and an offset" prompt="Find the nonzero period-one coefficients of x(t)=3+4 cos(4πt)−2 sin(2πt). What is its mean square?" hint="Use sin θ=(eⁱθ−e⁻ⁱθ)/(2i), and add squared coefficient magnitudes.">
      <Prose>c₀=3, c₂=c₋₂=2, c₁=i and c₋₁=−i. The mean square is 9+4+4+1+1=19. Directly, the constant contributes 9, the cosine 4²/2=8 and the sine 2²/2=2; full-period cross terms vanish. Missing either conjugate coefficient loses half of a real tone's energy.</Prose>
    </Practice>
    <Practice title="4. Delay every component consistently" prompt="Delay the running 1 Hz and 3 Hz signal by 1/4 second. Give c₁ and c₃ after the delay. Would adding the same phase to both frequencies generally do this?" hint="Multiply the original coefficients by exp(−i2πfτ) independently.">
      <Prose>c₁ becomes e⁻ⁱπ⁄²=−i. c₃=(i/2)e⁻ⁱ³π⁄²=(i/2)i=−1/2. The negative-frequency partners remain conjugates. The phase changes are −π/2 and −3π/2, so a single common phase increment generally does not encode a common time delay.</Prose>
    </Practice>
    <Practice title="5. Change the square wave's assigned jump value" prompt="Assign the square wave value 7 at t=0 without changing either open interval. What do the coefficient integrals and symmetric partial sums do there? Does falling mean squared error force the maximum error to vanish?" hint="One point has zero integral weight, and every sine term is zero at the origin.">
      <Prose>The coefficients remain unchanged and every partial sum at zero remains 0, the midpoint of the side limits −1 and +1. The error at that assigned point is 7. Even with midpoint assignment, the moving peak retains nonzero overshoot, so mean-square convergence does not force uniform convergence. State which error and which point are being discussed.</Prose>
    </Practice>
    <Practice title="6. Scale and shift a pulse" prompt="A unit-height pulse has width 3 seconds and center 2 seconds. Find X(0), the first positive zero of its magnitude, and the full complex transform." hint="Start with the centered width-3 pulse, then apply the time-shift factor.">
      <Prose>X(f)=3 sinc(3f)e⁻ⁱ⁴πᶠ. Its zero-frequency value is area 3 V·s, and its first positive zero is 1/3 Hz. Shifting the pulse changes phase but not magnitude. At the pulse edges, symmetric inverse limits return half the height; the convenient both-L¹ inversion condition is not satisfied by this sinc transform.</Prose>
    </Practice>
    <Practice title="7. A DFT with only two nonzero bins" prompt="Compute the DFT and squared energy of [2,0,−2,0]. Remove only bin 1, rather than its conjugate pair. Why is the inverse no longer real?" hint="The input is a bin-centered cosine. Check k=1 and k=3 directly.">
      <Prose>The DFT is [0,4,0,4], with time energy 8 and frequency energy (16+16)/4=8. Retaining only bin 3 gives the inverse eⁱ³πⁿ⁄², namely [1,−i,−1,i]. Its imaginary part does not cancel because the conjugate-symmetry condition was broken. Removing both bins gives the zero vector.</Prose>
    </Practice>
    <Practice title="8. Aliasing with a nonzero phase" prompt="At 20 samples/s, give a positive-frequency cosine below 10 Hz with the same samples as cos(2π·17t+π/4). Can a later digital low-pass filter recover which original tone was present?" hint="17−20=−3. Re-express a negative-frequency real cosine with a positive frequency.">
      <Prose>The alias is cos(2π·3t−π/4). On t=n/20 their phases differ only by full turns after using cosine's evenness. A filter applied to the identical sample vectors must give the same output for both. Distinguishing them requires prior information or a different acquisition procedure, such as an appropriate analog anti-alias filter and sampling rate.</Prose>
    </Practice>
    <Practice title="9. Padding is not another observation" prompt="You record N=64 samples at 64 samples/s and compute a length-512 FFT. Give the observation period and bin spacing. Under a periodic Hann window with no detrending, what quantity does the density integrate to?" hint="Keep N, M, Σw and Σw² separate.">
      <Prose>The observation period is 1 s and the spacing is 64/512=0.125 Hz. The last observation is at 63/64 s. The density integrates to Σw²x²/Σw², with the stated window and one-sided merging; it need not equal the unweighted sample mean square. For the lesson's specified 5.5 Hz periodic-Hann fixture it is 0.5 V². Padding refines the frequency grid, not the one-second observation's information.</Prose>
    </Practice>
    <Practice title="10. Catch a wrapped tail" prompt="Find linear and length-3 circular convolution for x=[2,−1,1], h=[1,2]. What FFT length is sufficient for the linear result?" hint="Compute the four linear values first and fold the fourth into the first for the circular result.">
      <Prose>The linear result is [2,3,−1,2]. Folding modulo 3 gives [4,3,−1]. Padding both inputs to at least 3+2−1=4 prevents this overlap and allows inverse-FFT multiplication to recover the linear result. The extra length represents the possible output support, not an arbitrary power-of-two requirement.</Prose>
    </Practice>
    <Practice title="11. Supply the missing transient" prompt="Solve y′+2y=2 cos(2t), y(0)=1. State the steady response and the full response separately." hint="H(i2)=2/(2+2i)=(1−i)/2. Evaluate the particular solution at zero.">
      <Prose>The steady response is (cos 2t+sin 2t)/2, with amplitude 1/√2 and lag −π/4. Its initial value is 1/2, so the full solution adds (1/2)e⁻²ᵗ. Substitution verifies the equation and y(0)=1. The derivative at zero is 2x(0)−2y(0)=0, a further local check.</Prose>
    </Practice>
    <Practice title="12. Same expression, incompatible regions" prompt="Interpret 1/(s−2) as a right-sided and a left-sided transform. Does either interpretation have an ordinary Fourier transform?" hint="Use a=−2 in the two exponential integrations and check whether σ=0 lies in the chosen ROC.">
      <Prose>Right-sided e²ᵗu(t) has ROC σ&gt;2 and no ordinary Fourier transform. Left-sided −e²ᵗu(−t) has ROC σ&lt;2, includes σ=0, and is integrable, so its ordinary Fourier transform is 1/(iω−2). A pole at +2 does not by itself settle this bilateral question without support and ROC.</Prose>
    </Practice>
    <Practice title="13. A delayed ramp with a nonzero initial state" prompt="Solve y′+y=u(t−2)(t−2), y(0+)=2. Give Y(s), the solution before and after t=2, and the value at the switching time." hint="The shifted ramp has transform e⁻²ˢ/s². Decompose 1/[s²(s+1)].">
      <Prose>Y(s)=2/(s+1)+e⁻²ˢ/[s²(s+1)]. Since 1/[s²(s+1)]=1/s²−1/s+1/(s+1), the added response is u(t−2)[(t−2)−1+e⁻⁽ᵗ⁻²⁾]. Before 2 the solution is 2e⁻ᵗ; after 2 add that bracket. At t=2 the bracket is zero, so y(2)=2e⁻² and the state is continuous. Differentiating the after-switch expression gives y′+y=t−2.</Prose>
    </Practice>
    <Practice title="14. Reject two tempting limit shortcuts" prompt="A causal sequence is (1.2)ⁿ and a causal continuous signal is sin t. Explain why plugging z=eⁱω into the first rational formula, or computing lim sY(s) at zero for the second, does not establish a Fourier transform or a final value." hint="Check the Z-transform ROC and the poles of sY(s), respectively.">
      <Prose>The sequence's ROC is |z|&gt;1.2, so the unit circle is excluded; the geometric series defining its ordinary frequency response does not converge there. For sin t, sY=s/(s²+1) has imaginary-axis poles ±i, violating the stated final-value condition even though its algebraic limit at zero is 0. The signal keeps oscillating. In both cases a computable formula is being used outside the theorem that would give it the desired interpretation.</Prose>
    </Practice>

    <Prose>You are ready to continue when you can track units and normalization, explain a coefficient by cancellation, distinguish finite reconstruction from continuous identifiability, select a boundary for convolution and retain the initial-state response. The next topic in this module is <a href="/learn/path/full-curriculum/conditioning-stability-numerical-analysis?module=math-foundations">Conditioning, Stability &amp; Numerical Analysis</a>: it asks how these and other computations respond to perturbed inputs and finite arithmetic. Specialist spectral, control and signal-processing lessons develop acquisition and system-design decisions beyond this foundation.</Prose>
    <Sources alternatives={<div><p><a href="https://www.3blue1brown.com/lessons/fourier-transforms/">3Blue1Brown's Fourier-transform visual explanation</a> offers a winding-and-averaging perspective; its companion text and original <a href="https://www.youtube.com/watch?v=spUNpyF58BY">video</a> are useful after section 3. Keep track of its finite averaging interval when comparing it with our normalized Fourier coefficients and unnormalized continuous transform.</p><p><a href="https://www.youtube.com/watch?v=sn3orkHWqUQ">MIT's complex-number lesson</a> gives a worked rectangular/polar treatment, with an <a href="https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/82481983927c276e34903cf879525acc_sn3orkHWqUQ.pdf">official transcript</a> for reading alongside it. The transcript and video identities were checked; the full videos were not watched for this authoring review.</p><p><a href="https://www.dsprelated.com/freebooks/mdft/Spectrum_Analysis_Sinusoid_Windowing.html">Julius O. Smith's sinusoid-windowing chapter</a> is a deeper written route into finite-record spectra. Its older MATLAB examples use their declared windows and normalizations; reproduce those choices before comparing outputs.</p></div>}>
      <li><a href="https://ocw.mit.edu/courses/es-1803-differential-equations-spring-2024/mites_1803_s24_topic4.pdf">MIT complex-arithmetic notes</a>: rectangular/polar operations and Euler notation; selected definitions and calculations reviewed.</li>
      <li><a href="https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/72c3d01402fa6b7c109b7b88d399115e_MIT6_003F11_lec15.pdf">MIT Fourier-series lecture</a>: orthogonality, coefficients and periodic system response.</li>
      <li><a href="https://www.jirka.org/diffyqs/html/moreonfourier_section.html">Jiří Lebl, Fourier convergence</a>: piecewise-smooth midpoint convergence and regularity; <a href="https://dlmf.nist.gov/6.16">NIST DLMF's Gibbs analysis</a> supplies the sine-integral connection and the overshoot convention. Our finite-square-wave formulas are derived and independently checked here.</li>
      <li><a href="https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/d74bab2dfa6e465d61fed45763d50528_MIT6_003F11_lec16.pdf">MIT continuous Fourier transforms</a> and <a href="https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/12e6e5d7567fca2e993ef8563fef5a60_MIT6_003F11_lec21.pdf">sampling lecture</a>: pulse/scaling, inverse conventions and the strict bandlimit condition.</li>
      <li><a href="https://math.stanford.edu/~andras/172-4.pdf">András Vasy's Fourier-transform notes</a>: Gaussian calculation, inversion and the integrability conditions behind exchanging limits. The notes use an angular-frequency convention; our local derivation converts the constants to hertz.</li>
      <li><a href="https://numpy.org/doc/stable/reference/routines.fft.html">NumPy FFT documentation</a>: forward/inverse conventions, bin ordering and normalization. The executed examples used NumPy 2.3.5; later documentation may describe a newer release.</li>
      <li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html">SciPy periodogram documentation</a>: density versus spectrum scaling, detrending defaults and one-sided merging. This lesson explicitly disables detrending and supplies the periodic window array.</li>
      <li><a href="https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/84b35cc755ccd722ddbdbba00fd41e8f_MIT6_003F11_lec06.pdf">MIT Laplace-transform lecture</a>: transform plus ROC and sidedness; <a href="https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/22aba4126352ce0f76930d858d8dffa5_MIT18_03SCF11_s29_1text.pdf">MIT derivative rules</a> makes the generalized pre-initial convention explicit. Our ordinary 0+ calculations and generalized 0− discussion are deliberately distinguished.</li>
      <li><a href="https://courses.grainger.illinois.edu/ece486/sp2026/documentation/handbook/lec04.html">University of Illinois control-systems handbook</a>: the pole condition for using a rational final-value theorem, and examples of steady-state reasoning.</li>
    </Sources>
  </div>
};
