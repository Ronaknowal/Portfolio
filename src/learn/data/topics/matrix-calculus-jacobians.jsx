import { Callout, Code, H2, H3, Prose } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample';
import { AffineGradientLab, BranchSumFigure, ChainRuleLab, FiniteDifferenceLab, JacobianMeaningFigure, LocalJacobianLab, MatrixSquareFigure, SharedBiasFigure } from '../../components/lesson-labs/MatrixCalculusLabs';
import { matrixCalculusExamples } from '../matrix-calculus-examples.js';
import MechanismProgram from '../../components/lesson-labs/MechanismProgram.jsx';
import { mechanismProgram } from '../matrix-calculus-mechanism-program.js';
function Practice({
  title,
  children,
  hint,
  solution
}) {
  return <section className="calculus-practice">
      <H3>{title}</H3>
      {children}
      <details><summary>Optional hint</summary><Prose>{hint}</Prose></details>
      <details><summary>Worked solution and reasoning</summary>{solution}</details>
    </section>;
}
export default {
  title: 'Matrix Calculus & Jacobians',
  readTime: '~55 min read + 90 min practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot matrix-calculus-lesson">
      <LessonIntro exampleKind="Optional NumPy" prerequisites="Coordinate vectors, matrix multiplication, transposes and simple algebra. We refresh scalar slopes, partial derivatives and the product rule before using them; no autodiff framework is required." sections={[['1-how-does-a-small-change-travel', 'A slope as a map'], ['2-one-slope-for-each-input-output-pair', 'Read a Jacobian'], ['3-local-prediction-is-not-exact-evaluation', 'Inspect approximation'], ['4-compose-local-maps-with-the-chain-rule', 'Chain rule'], ['5-bring-a-scalar-gradient-back', 'Reverse gradients'], ['6-differentiate-a-whole-matrix-program', 'Matrix-shaped gradients'], ['7-check-a-complete-batch-backward-pass', 'Batch & gradient checks'], ['8-compute-products-without-storing-every-slope', 'Automatic differentiation'], ['9-deeper-operators-and-useful-connections', 'Matrix and curvature depth'], ['10-practise-on-changed-problems', 'Independent practice']]}>
        A model transforms several inputs into several outputs. If one input changes, which outputs move—and by how much? Build a local map of those changes, compose it through a calculation, and use it to explain where a training gradient comes from.
      </LessonIntro>

      <H2>1. How does a small change travel?</H2>
      <Prose>Suppose a square has side length x metres and area s=x² square metres. Increase its side by h metres. The exact area change is (x+h)²−x²=2xh+h². At x=2 and h=0.01, the change is 0.0401 m². The simple prediction 2xh gives 0.04 m²; the leftover is only 0.0001 m².</Prose>
      <MathBlock>{String.raw`\frac{(x+h)^2-x^2}{h}=2x+h\quad(h\ne0)\\[6pt]\lim_{h\to0}(2x+h)=2x`}</MathBlock>
      <Prose>The <strong>derivative</strong> is the limiting change per unit input: here 2x. At x=2, multiply a small side change by 4 metres to estimate the area change. Derivative units are output units divided by input units. It is neither the area nor the finite area change.</Prose>
      <Prose>With vectors, we cannot divide by an input vector. Instead, ask for a <strong>linear map</strong> that turns an input change into an output change. A Jacobian is that derivative map written as a matrix in chosen coordinates. This reconnects the earlier vector-and-matrix lesson's linear maps to a function that may itself be nonlinear.</Prose>
      <Callout accent="green" label="A compact calculus refresher">Hold constants fixed. Sums differentiate term by term. For a product a(t)b(t), the rate is a′b+ab′: both factors can move. If you need more practice with limits, partial derivatives and gradients, <a href="/learn/path/full-curriculum/multivariate-calculus-gradients">Multivariate Calculus & Gradients</a> supplies a fuller route later in this module. The steps needed here are developed in place.</Callout>

      <H2>2. One slope for each input–output pair</H2>
      <Prose>Consider the two-input function f(x₁,x₂)=(x₁²+x₂,x₁x₂). The two outputs are named f₁ and f₂. A <strong>partial derivative</strong> changes one input while holding the other fixed. For f₁, changing x₁ gives slope 2x₁; changing x₂ gives slope 1. For f₂, the respective slopes are x₂ and x₁.</Prose>
      <MathBlock>{String.raw`\begin{gathered}J_f(x)=\begin{bmatrix}2x_1&1\\x_2&x_1\end{bmatrix}\\[5pt]
J_f(2,3)=\begin{bmatrix}4&1\\3&2\end{bmatrix}\end{gathered}`}</MathBlock>
      <Prose>For f:ℝⁿ→ℝᵐ, we use <strong>output rows, input columns</strong>: J has shape m×n and Jᵢⱼ=∂fᵢ/∂xⱼ. A row describes one output's sensitivity to every input; a column describes every output's response to one input. Indices in these mathematical formulas begin at 1.</Prose>
      <JacobianMeaningFigure />
      <Prose>If f₁ were measured in volts and x₂ in seconds, J₁₂ would have units volts/second. Comparing raw entries with different units does not establish which input is “most important.” Rescaling an input changes its derivative's numerical value. Fix coordinate conventions and meaningful perturbation sizes before comparing sensitivities.</Prose>
      <Checkpoint prompt="Is the Jacobian always square, or equal to the matrix of a linear layer?"><Prose>No: n inputs and m outputs give m×n, including rectangular shapes. For f(x)=Ax+b, the Jacobian with respect to x is A at every point. For a nonlinear f it generally depends on x; its output value and its Jacobian remain different objects.</Prose></Checkpoint>
      <Practice title="Build a Jacobian at a new point" hint="Differentiate each output with respect to each input; substitute the point last." solution={<Prose>For h=(x₁+3x₂,x₁x₂²), J=[[1,3],[x₂²,2x₁x₂]]. At (−1,2), J=[[1,3],[4,−4]]. Direction v=(2,1) gives Jv=(5,4). The second row means a positive x₂ change decreases the second output locally at this point.</Prose>}>
        <Prose>For h(x₁,x₂)=(x₁+3x₂,x₁x₂²), find J at (−1,2), compute J(2,1), and interpret one negative entry. Show each partial derivative rather than guessing a shape.</Prose>
      </Practice>

      <H2>3. Local prediction is not exact evaluation</H2>
      <Prose>At x=(2,3), choose Δx=(0.01,−0.02). Multiplying the Jacobian gives (4×0.01−0.02,3×0.01+2×(−0.02))=(0.02,−0.01). This predicts the <em>change</em>; to predict the new output, add it to f(x)=(7,6).</Prose>
      <MathBlock>{String.raw`f(x+\Delta x)=f(x)+J_f(x)\Delta x+r(\Delta x)\\[5pt]
\frac{\|r(\Delta x)\|}{\|\Delta x\|}\longrightarrow0\quad\text{as }\Delta x\to0`}</MathBlock>
      <Prose>The remainder r becomes small compared with the input change in every direction. That is the meaning of <strong>differentiability</strong> here. For our polynomial, r=(Δx₁²,Δx₁Δx₂) exactly. The actual change is (0.0201,−0.0102), not exactly the prediction. Halving Δx quarters this particular remainder. Merely being differentiable does not guarantee a quadratic remainder for every function.</Prose>
      <LocalJacobianLab />
      <Prose>Write a direction as v and a scalar step as t, so Δx=tv. The <strong>directional derivative</strong> is Jv: change per unit t at zero. Some texts require a unit-length direction; we allow any v and state it. Doubling v doubles Jv. The map can predict no first-order change in a direction while higher-order change remains.</Prose>
      <details className="calculus-practice"><summary>Deeper: having partial derivatives is not enough</summary><Prose>Define g(a,b)=ab/√(a²+b²) away from (0,0), and g(0,0)=0. Along either coordinate axis g=0, so both partial derivatives at the origin are zero. Along (a,b)=(t,t), however, |g|=|t|/√2 and the input length is √2|t|. Their ratio stays 1/2. The zero linear map does not leave a remainder negligible compared with the input. Continuous partial derivatives in a neighborhood are a useful sufficient condition for differentiability; mere existence at one point is not.</Prose></details>
      <Prose><strong>Optional code:</strong> each program below includes its imports and inputs. With NumPy available, save a block as <Code>calculus_example.py</Code> and run <Code>python calculus_example.py</Code>, or use a fresh notebook cell. The dual-number and Decimal examples use only the standard library. These are numerical checks of the mathematics, not prerequisites for reading it.</Prose>
      <RunnableExample example={matrixCalculusExamples.local}><Prose>The remainder signs say which prediction was too high or too low. A gradient check should inspect numerical agreement and scale, not only the resulting array shape.</Prose></RunnableExample>

      <H2>4. Compose local maps with the chain rule</H2>
      <Prose>Suppose y=g(x) and z=f(y). A small input change first produces dy=Jg(x)dx; that intermediate change then produces dz=Jf(y)dy. Substitution gives the chain rule. Here d denotes the first-order differential, while Δ denotes a finite change that may include a remainder.</Prose>
      <MathBlock>{String.raw`J_{f\circ g}(x)=J_f(g(x))J_g(x)\\[5pt]
(p\times m)(m\times n)=p\times n`}</MathBlock>
      <Prose>The rightmost matrix acts first, exactly as in ordinary matrix composition. Evaluate the later derivative at the actual intermediate value g(x), not at x. The chain rule requires differentiability at the points being composed.</Prose>
      <Prose>Take A=[[1,2],[−1,1]], y=Ax, and z=(y₁²,y₂²). At x=(1,2), y=(5,1). The square operation's Jacobian is diag(2y)=diag(10,2): a diagonal matrix with those entries and zeros elsewhere. Multiplying diag(10,2)A gives [[10,20],[−2,2]]. For input direction v=(1,0), y changes at rate (1,−1), then z at rate (10,−2).</Prose>
      <Checkpoint prompt="Can I reverse the matrix order if both products have a valid shape?"><Prose>No. Here A·diag(10,2)=[[10,4],[−10,2]], which is also 2×2 but describes the wrong order. Shape checks reject some errors; they cannot prove the chain rule or the evaluation points are correct.</Prose></Checkpoint>

      <H2>5. Bring a scalar gradient back</H2>
      <Prose>Now turn z into one scalar L=q·z with fixed q=(1,−1). At the same point L=25−1=24. Its gradient with respect to z is q. Under our convention a <strong>gradient</strong> is a column: gₓ=∇ₓL. The scalar function's Jacobian is the row gₓᵀ, and dL=gₓᵀdx.</Prose>
      <MathBlock>{String.raw`dL=g_y^\top dy=g_y^\top J_g\,dx\\[5pt]
g_x=J_g^\top g_y`}</MathBlock>
      <Prose>Why a transpose? Both expressions must give the same scalar change for every dx. Collecting the coefficient of each input change gives Jgᵀgᵧ. This is a <strong>pullback of the output sensitivity</strong>, not an inverse transformation of the output value. A rectangular or singular Jacobian can still propagate a gradient.</Prose>
      <Prose>For the chain above, g_z=(1,−1), g_y=(10,−2), and g_x=Aᵀg_y=(12,18). Forward propagation along v=(1,0) gives dL/dt=1×10+(−1)×(−2)=12. Reverse propagation gives g_x·v=12. These agree because they apply the same derivative in different orders.</Prose>
      <ChainRuleLab />
      <Prose>A <strong>Jacobian–vector product (JVP)</strong> sends one input direction forward: Jv. A <strong>vector–Jacobian product (VJP)</strong> is often written as the row gᵀJ; with our column gradients, the returned input-shaped values are Jᵀg. Read a library's layout convention before transposing code.</Prose>
      <BranchSumFigure />
      <RunnableExample example={matrixCalculusExamples.chain}><Prose>The program deliberately keeps tangent variables separate from gradient variables. A tangent is a chosen input movement propagated forward; a gradient describes the scalar objective's sensitivity propagated backward.</Prose></RunnableExample>
      <details className="calculus-practice"><summary>Deeper: why a negative gradient is a local descent direction</summary><Prose>For nonzero Euclidean gradient g, choose Δx=−ηg with η&gt;0. The first-order loss change is −η‖g‖², which is negative. Differentiability therefore gives decrease for sufficiently small η, but this calculation supplies no universal step size. A large step, a kink, constraints or numerical error can invalidate the prediction. The gradient depends on the inner product used to turn the derivative into a vector; this lesson uses the ordinary real Euclidean/Frobenius conventions.</Prose></details>

      <H2>6. Differentiate a whole matrix program</H2>
      <Prose>Training also changes a matrix of weights. For a scalar L(W), define the <strong>matrix gradient</strong> G_W entry by entry: (G_W)ᵢⱼ=∂L/∂Wᵢⱼ. Its shape matches W. Pair it with a matrix perturbation dW by multiplying matching entries and summing.</Prose>
      <MathBlock>{String.raw`dL=\sum_{i,j}(G_W)_{ij}\,dW_{ij}
=\operatorname{tr}(G_W^\top dW)`}</MathBlock>
      <Prose>This is the <strong>Frobenius inner product</strong>. The trace tr sums diagonal entries; expanding that trace produces the same entrywise sum. We do not need to flatten W or materialize one huge Jacobian simply to express dL.</Prose>
      <Prose>Let Y=XW+b. Rows of X are observations, columns are features. W turns d input features into m outputs; b has m shared output biases. Given an incoming scalar-loss gradient G=∂L/∂Y, how does one parameter contribute? Since Yᵢⱼ=ΣₖXᵢₖWₖⱼ+bⱼ, changing Wₖⱼ changes Yᵢⱼ with slope Xᵢₖ. Each observation contributes GᵢⱼXᵢₖ; add all observations.</Prose>
      <Prose>This storage convention puts each observation in a <em>row</em>. Written as column vectors instead, the same observation obeys y=Wᵀx+b, so its input Jacobian is Wᵀ. We have not changed the output-row/input-column Jacobian convention; we changed how many observations are arranged in the batch arrays.</Prose>
      <MathBlock>{String.raw`\begin{aligned}
(G_W)_{kj}&=\sum_i X_{ik}G_{ij}\\
(G_b)_j&=\sum_i G_{ij}\\
(G_X)_{ik}&=\sum_j G_{ij}W_{kj}
\end{aligned}`}</MathBlock>
      <Prose>In matrix form, G_W=XᵀG and G_X=GWᵀ; G_b sums G over the observation rows, leaving one entry per output column. The input gradient sums every output affected by that input. “dW,” “db” and “dX” are common code names for these loss gradients, but in a mathematical differential dW means a perturbation. We use <Code>grad_W</Code> in code to keep that distinction visible.</Prose>
      <LessonTable caption="An affine batch's backward shape contract" headers={['Quantity', 'Shape', 'Meaning']} rows={[['X / W / b', '(N,d) / (d,m) / (m,)', 'Observations, shared weights and bias'], ['Y and G', '(N,m)', 'Outputs and incoming loss sensitivities'], ['XᵀG', '(d,N)@(N,m) → (d,m)', 'One gradient per weight'], ['sum(G, axis=0)', '(m,)', 'Collect every use of each bias'], ['GWᵀ', '(N,m)@(m,d) → (N,d)', 'One sensitivity per input entry']]} />
      <SharedBiasFigure />
      <Prose>More generally, a forward broadcast repeats one source value; its reverse rule sums the contributions of those repeats. A forward sum repeats the incoming scalar sensitivity to its source entries; a mean also divides by their count. A transpose permutes axes back. These are derivative rules for the computation performed, not guesses based on compatible shapes.</Prose>

      <H2>7. Check a complete batch backward pass</H2>
      <Prose>Choose a concrete training objective: L=(1/(2N))Σᵢⱼ(Yᵢⱼ−Tᵢⱼ)², where T is the target and N counts observations. The square's derivative cancels the 1/2, so G=(Y−T)/N. We average over observations and <em>sum</em> output coordinates. Taking the mean over all Nm entries would produce an additional factor 1/m.</Prose>
      <AffineGradientLab />
      <RunnableExample example={matrixCalculusExamples.affine}><Prose>With X=(4,3), W=(3,2) and G=(4,2), check that the three gradient shapes match their inputs and parameters. A positive bias gradient means increasing that bias locally increases this loss; it does not mean the bias's current value is positive.</Prose></RunnableExample>
      <Prose>To check one weight without trusting our formula, perturb only that entry by +h and −h, run the complete loss twice, subtract and divide by 2h. Repeat for every coordinate on this small fixture. In a large model, checking several independent directions can be cheaper: compare [L(θ+hv)−L(θ−hv)]/(2h) with ∇L·v.</Prose>
      <MathBlock>{String.raw`\begin{aligned}D_{\mathrm{central}}(h)&=\frac{L(\theta+hv)-L(\theta-hv)}{2h}\\
&\approx\nabla L(\theta)^\top v\end{aligned}`}</MathBlock>
      <RunnableExample example={matrixCalculusExamples.gradientCheck}><Prose>The incorrect bias mean has the right shape but is half the correct gradient. The independent loss perturbations expose it. At a point with zero gradients, that particular bug could escape detection; use multiple inputs and directions.</Prose></RunnableExample>
      <FiniteDifferenceLab />
      <Prose>For a sufficiently smooth scalar function, forward differences have error of order h and central differences order h² in exact arithmetic. Subtracting nearby floating-point values and dividing by a tiny h can amplify roundoff. Test several h values; no single tolerance or step works for all scales. Freeze random draws, batches and mutable state so both evaluations represent the same function.</Prose>
      <RunnableExample example={matrixCalculusExamples.differences}><Prose>For x³ the exact central error is h². Decimal arithmetic exposes that truncation term without the early binary-double cancellation seen in the lab. At |x|=0, left and right slopes remain different however small h becomes.</Prose></RunnableExample>
      <Checkpoint prompt="If an autodiff library returns zero at a kink, have we proved that the derivative is zero?"><Prose>No. A framework can define a useful branch or subgradient convention where the ordinary derivative does not exist. Read the operation's rule and distinguish it from differentiability. Agreement between two checks at only one symmetric point is not a proof.</Prose></Checkpoint>

      <H2>8. Compute products without storing every slope</H2>
      <Prose><strong>Automatic differentiation (AD)</strong> applies known derivative rules to the operations in an executed program. It does not estimate slopes by subtracting nearby outputs, nor need to print one giant symbolic expression. It still uses finite-precision arithmetic and differentiates the implemented calculation, which can differ from the model you intended.</Prose>
      <Prose>A tiny forward-mode engine can carry two numbers through every operation: its ordinary value a and its tangent ȧ. Addition adds tangents; multiplication uses ȧb+aḃ. Start each input with the chosen component of v. The final tangents are Jv. There is no small finite step h.</Prose>
      <RunnableExample example={matrixCalculusExamples.dual}><Prose>This miniature engine supports only addition and multiplication. Seeding (1,0) or (0,1) extracts one Jacobian column; seeding (1,−2) computes a combination directly. Production systems support many more operations and handle storage, tracing and custom derivatives.</Prose></RunnableExample>
      <LessonTable caption="Choose the derivative information actually needed" headers={['Question', 'Operation', 'Reason']} rows={[['Response along one input direction', 'Jv / forward mode', 'Propagate one tangent; no full J needed'], ['Sensitivity of one scalar objective', 'Jᵀg / reverse mode', 'One output seed reaches all input sensitivities'], ['Every output–input slope', 'Full m×n Jacobian', 'n forward basis seeds or m reverse basis seeds; batching/sparsity can help'], ['How a gradient changes along v', 'Hessian–vector product Hv', 'Differentiate the gradient in one direction']]} />
      <Prose>For many inputs and one scalar loss, reverse mode can compute the gradient with work proportional to evaluating the differentiable program, with operation-dependent overhead. It often saves intermediate values for the reverse pass, trading memory against recomputation. Forward mode is often preferable for few input directions. Dimensions suggest a choice; they do not establish an absolute runtime winner. A dense m×n Jacobian has mn entries before considering any framework overhead.</Prose>
      <section id="matrix-calculus-code-route" aria-label="Manual derivatives and maintained automatic differentiation">
        <H3>Run the same derivative map through PyTorch</H3>
        <Prose>The NumPy pullback above owns the mechanism. The following complete program maps it to <Code>torch.func.jvp</Code>, <Code>vjp</Code> and <Code>jacrev</Code>, then checks a rectangular affine batch against ordinary autograd. Install NumPy 2.3.5 and a suitable PyTorch 2.14.0 distribution; run <Code>python derivative-library-bridge.py</Code>. The recorded run uses CPU float64.</Prose>
        <Prose>For f(x)=(x₁²+x₂,x₁x₂), x=(2,3), direction (.01,−.02) gives Jv=(.02,−.01); output cotangent (1,−1) gives Jᵀq=(1,−1). Their scalar pairing agrees at .03. The affine routine keeps W in [input,output] order; <Code>F.linear</Code> receives its transpose. Both objectives sum squared residuals and divide by twice the batch size. Taking an additional mean over outputs would halve this two-output gradient.</Prose>
        <MechanismProgram {...mechanismProgram} title="Complete derivative maps and PyTorch comparison" />
        <Prose>A dense m-by-n Jacobian needs mn stored entries. JVP and VJP apply a derivative operator without requiring that matrix; their work follows the underlying program, while reverse mode retains intermediates. For an N-by-d input and d-by-k weight, the affine forward and pullbacks take O(Ndk) arithmetic and store the inputs, parameters, outputs and gradients. This is an explicit differentiable affine primitive, not a new general AD engine: <a href="/learn/topic/backpropagation-automatic-differentiation">Backpropagation &amp; Automatic Differentiation</a> owns graph construction and reverse accumulation. Library derivative conventions at kinks still require inspection.</Prose>
        <details><summary>Implementation practice: change the batch and direction</summary><Prose>Duplicate every affine observation and target, then change the vector-function direction to (−2,1). Check parameter gradients, each copy's input gradient, and the adjoint pairing before comparing frameworks.</Prose><details><summary>Solution and acceptance checks</summary><Prose>Mean-over-batch parameter gradients stay unchanged; each copied input gets half its former gradient. The changed JVP is (−7,−4), so q·Jv=−3 equals (Jᵀq)·v. Test these properties with a nonsquare batch and keep the exact target shape: accepting a broadcast target changes the objective.</Prose></details></details>
        <Prose><a href="https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html" target="_blank" rel="noreferrer">PyTorch's JVP contract</a> describes matching primal/tangent structures; the <a href="https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html" target="_blank" rel="noreferrer">function-transform tutorial</a> explains compositions and the dense-Jacobian alternatives.</Prose>
      </section>

      <H2>9. Deeper operators and useful connections</H2>
      <Prose>The core backward pass is now complete. These branches connect the same derivative map to matrix algorithms, normalized outputs and second-order behavior. Choose the question you need; each develops its conventions before using them.</Prose>
      <details className="calculus-practice"><summary>Matrix squaring: a derivative that acts on a matrix</summary>
        <Prose>For F(A)=A², perturb A by tE. Expanding gives (A+tE)²=A²+t(AE+EA)+t²E². Thus DF(A)[E]=AE+EA. The bracket means “apply the derivative operator to E.” For a p×q matrix input and an r×s matrix output, the entrywise derivative has four indices; flattening would give an (rs)×(pq) matrix. Its action can be simpler than materializing that array.</Prose>
        <MatrixSquareFigure />
        <RunnableExample example={matrixCalculusExamples.square} />
        <Prose>For a scalar objective L(A)=½‖A‖²_F, dL=ΣᵢⱼAᵢⱼdAᵢⱼ, so its matrix gradient is A. For L(x)=xᵀBx with fixed real B, the product rule gives gradient (B+Bᵀ)x; simplifying to 2Bx requires B to be symmetric.</Prose>
      </details>
      <details className="calculus-practice"><summary>Differentiate a solve instead of replaying an iterative algorithm</summary>
        <Prose>Suppose invertible A and b determine u through Au=b. A coupled calibration or equilibrium model may need to know how u changes when a coefficient changes. Differentiate the equation: (dA)u+Adu=db, hence solve A·du=db−(dA)u. The coefficient perturbation changes the right-hand side of a sensitivity solve; it is not another independent input to the original solution.</Prose>
        <RunnableExample example={matrixCalculusExamples.solve}><Prose>The example varies A along E and b along c together. Its rate is (−0.32,0.44): for small positive t the first solution component decreases and the second increases. This is an invented numerical sensitivity scenario, not a validated physical calibration system.</Prose></RunnableExample>
        <Prose>For the inverse itself, differentiating AA⁻¹=I gives D(A⁻¹)[E]=−A⁻¹EA⁻¹. The matrix must remain invertible near the point; near singularity the derivative can be very large. Numerical implementations should use solves and factorization reuse where appropriate, rather than form inverses just to multiply vectors. Differentiating an exact solved equation and differentiating a finite number of solver iterations are different tasks.</Prose>
      </details>
      <details className="calculus-practice"><summary>Softmax: changing one score changes several outputs</summary>
        <Prose>For finite scores s, define pᵢ=exp(sᵢ)/Σⱼexp(sⱼ). The exponential derivative and quotient rule give ∂pᵢ/∂sⱼ=pᵢ(δᵢⱼ−pⱼ), where δᵢⱼ is 1 for equal indices and 0 otherwise. Thus J=diag(p)−ppᵀ. Off-diagonal entries matter because changing one score also changes the denominator shared by every output.</Prose>
        <MathBlock>{String.raw`Jv=p\odot\bigl(v-(p^\top v)\mathbf1\bigr),\qquad J\mathbf1=0`}</MathBlock>
        <Prose>Here ⊙ multiplies matching entries and 1 is the all-ones vector. Adding the same constant to every score cancels exactly from the normalized ratios. Its derivative therefore kills the common-shift direction, connecting to the null-space idea from linear algebra. The output rates sum to zero because the probabilities must keep summing to one.</Prose>
        <RunnableExample example={matrixCalculusExamples.softmax}><Prose>Subtracting the maximum score before exponentiating stabilizes this computation. The formula for J is for the smooth normalized function of finite scores; a maximum used inside an algebraically equivalent stable implementation does not make that final function nonsmooth. This example does not imply probabilities are calibrated for a real prediction task.</Prose></RunnableExample>
      </details>
      <details className="calculus-practice"><summary>Hessians: the local change of the gradient</summary>
        <Prose>For a twice continuously differentiable scalar L, its gradient is another vector-valued function. Its Jacobian is the Hessian H, with entries Hᵢⱼ=∂²L/(∂xᵢ∂xⱼ). Under this smoothness condition mixed partials agree and H is symmetric. Hv predicts how the gradient changes along v; vᵀHv describes directional curvature.</Prose>
        <MathBlock>{String.raw`L(x+\Delta x)\approx L(x)+g^\top\Delta x
+\tfrac12\Delta x^\top H\Delta x`}</MathBlock>
        <RunnableExample example={matrixCalculusExamples.hessian}><Prose>For this quadratic the second-order expression is exact in real arithmetic. General smooth functions also have higher-order remainders. A Hessian is a second derivative, not the outer product of a gradient with itself.</Prose></RunnableExample>
      </details>

      <H2>10. Practise on changed problems</H2>
      <Prose>Keep solutions closed first. State the function, evaluation point, derivative convention, shapes and assumptions before calculating. Explain at least one mistaken alternative; numerical agreement alone cannot show that you chose the intended objective.</Prose>
      <Practice title="Separate a rate, a predicted change and the new output" hint="Jv is per unit t. Multiply by t, then add the base output." solution={<Prose>For the original f at (1,−1), f=(0,−1) and J=[[2,1],[−1,1]]. With v=(1,2), the rate is (4,1). At t=0.1, predicted change=(0.4,0.1), predicted new output=(0.4,−0.9). Actual input=(1.1,−0.8) gives (0.41,−0.88), so the remainder=(0.01,0.02). It is the change that J approximates, not f itself.</Prose>}>
        <Prose>Use the original polynomial at x=(1,−1), v=(1,2), t=0.1. Find Jv, predicted change, predicted new output, actual new output and remainder. Identify which result is a rate.</Prose>
      </Practice>
      <Practice title="Reverse a new scalar objective" hint="Differentiate the outer square first, then propagate to the original inputs." solution={<Prose>Let y₁=x₁+x₂ and y₂=x₁−2x₂, so Jg=[[1,1],[1,−2]]. At (2,1), y=(3,0). For L=y₁²+3y₂, g_y=(6,3); therefore g_x=(9,0). Directly expanding L=(x₁+x₂)²+3x₁−6x₂ gives the same gradient. A zero x₂ derivative here is cancellation between paths; it does not mean L never depends on x₂.</Prose>}>
        <Prose>Let y=(x₁+x₂,x₁−2x₂) and L=y₁²+3y₂. At x=(2,1), compute the input gradient via the chain rule and by differentiating the expanded scalar. Explain the zero component.</Prose>
      </Practice>
      <Practice title="Repair a batch gradient with the right shape but the wrong scale" hint="Differentiate the exact averaging convention once, then sum the shared bias's uses." solution={<Prose>If L is mean squared error across all Nm output entries with no 1/2, then G=2(Y−T)/(Nm). For N=4,m=2, G=(Y−T)/4. The weight gradient is XᵀG, bias gradient G.sum(axis=0), input gradient GWᵀ. Taking G.mean(axis=0) adds an incorrect factor 1/4. In this particular m=2 case G numerically matches our earlier half/observation-mean objective; that coincidence disappears for other m.</Prose>}>
        <Prose>Now define L as the mean of all squared entries of Y−T, with N=4 and m=2 and no factor 1/2. Derive G and the three affine gradients. Diagnose <Code>grad_b=G.mean(axis=0)</Code>. Would the same scaling hold if m changed to 3?</Prose>
      </Practice>
      <Practice title="Design a check that can reveal a transpose mistake" hint="A symmetric Jacobian or a single unlucky seed can conceal an error." solution={<Prose>Use a nonsymmetric, nonzero fixture such as the original J=[[4,1],[3,2]]. Check two independent directions v=(1,0) and (0,1), then a mixed direction. Jv and Jᵀv disagree for each basis seed here. Compare central changes of the actual vector function at several h values, not another copy of the analytic formula. Re-run at a second point. Avoid kinks, freeze state and report absolute/relative scales; a finite sample increases confidence but is not a proof for all inputs.</Prose>}>
        <Prose>A colleague's derivative code accidentally uses Jᵀ. Propose concrete inputs, directions and acceptance checks likely to expose it. Explain why testing only at the zero input or with a symmetric example is weak.</Prose>
      </Practice>
      <Practice title="Choose a derivative product for a new application" hint="Ask what is seeded and what must be returned before considering a full Jacobian." solution={<Prose>For 3 control inputs and 1,000 sensor outputs, one chosen control direction needs one JVP; all sensitivities require three forward basis seeds, subject to implementation costs. For one scalar fitting loss and 1,000,000 parameters, reverse mode seeded with 1 naturally produces all parameter gradients. A full 1,000×3 Jacobian can be useful for analysis; the scalar-loss gradient requires only 1,000,000 values rather than pairwise parameter interactions. A gradient does not by itself provide the Hessian or guarantee an optimizer's success.</Prose>}>
        <Prose>Compare a 3-input/1,000-output sensor model with a one-loss/1,000,000-parameter training model. Choose JVP, VJP or a full Jacobian for one control-direction response, all sensor sensitivities and the training gradient. State a limitation of the dimension-based rule.</Prose>
      </Practice>
      <Callout accent="green" label="Ready to continue">You can interpret an individual Jacobian entry, derive a local prediction and its limit, propagate a direction or scalar sensitivity in the correct order, derive the batch gradients and independently diagnose a numerical check. The deeper matrix and Hessian branches extend that core; a “complete” click does not test these skills.</Callout>
      <Prose><strong>Next in this module:</strong> <a href="/learn/path/full-curriculum/tensor-algebra-einsum-notation">Tensor Algebra & Einsum Notation</a> names the indices multiplied, retained and reduced. The weight-gradient sum ΣᵢXᵢₖGᵢⱼ is a concrete starting point: keep feature/output indices, sum the observation index.</Prose>
      <Sources alternatives={<div><h4>Another explanation or guided route</h4><ul>
        <li><a href="https://www.youtube.com/watch?v=5DUQ3-Y_gX4" target="_blank" rel="noopener noreferrer">MIT 18.S096 — Derivatives as Linear Operators (video)</a>, with its <a href="https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/resources/ocw_18s096_lecture01-part2_2023jan18_mp4/" target="_blank" rel="noopener noreferrer">course page</a>. Revisit sections 1–3 after elementary calculus and linear algebra; companion notes were reviewed, not the complete recording.</li>
        <li><a href="https://www.youtube.com/watch?v=r9_5dxtDTOk" target="_blank" rel="noopener noreferrer">MIT — Differentiation on Computational Graphs (video)</a>, with the <a href="https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/resources/ocw_18s096_lecture05-part3-new_2023jan27_mp4/" target="_blank" rel="noopener noreferrer">official lecture entry</a>. A deeper graph-oriented treatment after forward/reverse mode; the associated notes support the recommendation. No timestamp or whole-video viewing claimed.</li>
      </ul></div>}>
        <ul>
          <li><a href="https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec02.pdf" target="_blank" rel="noopener noreferrer">MIT notes — Derivatives as Linear Operators</a>. Mathematical definitions and directional interpretation; a more formal continuation of the local-map explanation.</li>
          <li><a href="https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec04.pdf" target="_blank" rel="noopener noreferrer">MIT notes — Finite-Difference Approximations</a>. Independent checking, truncation/roundoff and matrix-product order.</li>
          <li><a href="https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec08.pdf" target="_blank" rel="noopener noreferrer">MIT notes — Forward and Reverse-Mode AD</a>. Dual arithmetic, graph differentiation and second-order products. The source examples use Julia; this lesson's examples use Python.</li>
          <li><a href="https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec07.pdf" target="_blank" rel="noopener noreferrer">MIT notes — Matrix Inverse and Determinant Derivatives</a>. Continue the matrix-operator branch, attending to invertibility conditions.</li>
          <li><a href="https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html" target="_blank" rel="noopener noreferrer">PyTorch — Jacobians, Hessians and function transforms</a>. Optional framework practice after section 8; tutorial requires PyTorch 2.0 or later. APIs and explanation were reviewed; its hardware timings are not reproduced or treated as universal rankings.</li>
        </ul>
      </Sources>
    </div>
};
