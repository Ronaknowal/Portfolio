// Generated from the complete prepared manuscript; all original sections and worked examples retained.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {OdeFixedLab,OdeAdaptiveLab,OdeGradientLab,OdeTopologyLab,OdeObservationLab} from '../../components/lesson-labs/NeuralOdeLabs.jsx';
import {OdeInputFlowFigure,OdeDecayFigure,OdeWorkedStages,OdeAdaptiveWorkedFigure,OdeGradientWorkedFigure,OdeOrderPreservationFigure,OdeObservationWorkedFigure,OdeArchitectureFigure,OdeAdjointFigure,OdeOrderFigure,OdeDensityFigure,OdeFlowMatchingFigure,OdeStructuredFigure,OdeSolverApiFigure} from '../../components/lesson-labs/NeuralOdeDiagrams.jsx';
import {OdeStudyFigure,OdeWorkedIrisFigure,OdeIrisLab,OdeProgram} from '../../components/lesson-labs/NeuralOdeStudy.jsx';
export default {title:'Neural ODEs and Continuous-Depth Models',readTime:'~90 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral ode-lesson">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Change a field, follow the resulting trajectory, and compare numerical error with computational work. Later investigations let you alter real measurements and observation times while keeping the learned model fixed."}</Prose>

<Prose>{"Imagine transforming a measurement by moving a point through a landscape of arrows. At its current location, an arrow tells the point which direction to move and how quickly. After a small move, the point encounters a new arrow. A whole journey emerges from repeatedly following this local rule."}</Prose>

<Prose>{"A "}<strong>{"neural ordinary differential equation"}</strong>{", or Neural ODE, uses a neural network to produce those arrows. A numerical solver follows them. The resulting journey can transform features for classification, describe a hidden state between irregular observations, or move a probability distribution into another shape."}</Prose>

<Prose>{"This division of work is the idea to keep: "}<strong>{"the network learns the rate of change; the solver approximates the accumulated change."}</strong>{" Neither piece can be understood fully by looking at the other alone."}</Prose>

<OdeInputFlowFigure/>

<Prose opening="route">{""}<strong>{"First pass."}</strong>{" Follow sections 1–5 to calculate a trajectory and understand how a model learns it. Section 6 trains small classifiers on real flower measurements; section 7 connects continuous state to irregular observations. Then do the core practice. The deeper branches on density models, flow matching and structured dynamics can wait until the main distinction between a field and its solution feels natural."}</Prose>

<Prose>{"You need vectors, derivatives, an ordinary feedforward network and backpropagation. The "}<a href={"/learn/path/full-curriculum/ordinary-differential-equations-linear-systems?module=math-foundations"}>{"ODE and linear-systems lesson"}</a>{" provides a fuller mathematical foundation. We introduce the numerical ideas needed here as we use them."}</Prose>

<H2>{"1. A derivative tells you how to move"}</H2>

<Prose>{"Let "}<InlineMath>{"z(t)"}</InlineMath>{" be a state: perhaps two coordinates on a plane, or a vector of hidden features. The equation"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{dz(t)}{dt}=f_\\theta(t,z(t)),\\qquad z(0)=z_0"}</MathBlock></div>

<Prose>{"specifies its rate of change and starting point. The symbol "}<InlineMath>{"\\theta"}</InlineMath>{" collects the network's learned weights. The right side returns a vector with the "}<strong>{"same shape as the state"}</strong>{". It is a velocity, not the next state itself."}</Prose>

<Prose>{"The variable "}<InlineMath>{"t"}</InlineMath>{" needs an interpretation. For measurements recorded over hours, it may be physical time. For a classifier transforming one image or flower measurement, it is a coordinate along "}<strong>{"representation depth"}</strong>{". The flower does not grow while the solver runs."}</Prose>

<Prose>{"Suppose the scalar rule is "}<InlineMath>{"z'=-2z"}</InlineMath>{", initially "}<InlineMath>{"z_0=1"}</InlineMath>{". The initial derivative is −2. Over a small interval of length 0.1, a first estimate of the change is "}<InlineMath>{"0.1(-2)=-0.2"}</InlineMath>{", so the next state is approximately 0.8. Now the derivative is −1.6. The state keeps decreasing, but the decrease slows as the state approaches zero."}</Prose>

<Prose>{"The exact solution is "}<InlineMath>{"z(t)=e^{-2t}"}</InlineMath>{". At "}<InlineMath>{"t=0.1"}</InlineMath>{", it is approximately 0.81873. Our estimate 0.8 was close, but it used the initial velocity throughout an interval during which velocity changed. That is the first numerical error to understand."}</Prose>

<OdeDecayFigure/>

<Prose>{"An "}<strong>{"initial-value problem"}</strong>{" asks for a trajectory given an initial state and rule. Learning asks a different question: which rule produces useful trajectories for the examples we have? In a classifier, a readout converts the final state into scores:"}</Prose>

<div className="neural-equation"><MathBlock>{"z_T=\\operatorname{Solve}(f_\\theta,z_0,0,T),\\qquad\n\\text{logits}=Wz_T+b."}</MathBlock></div>

<Prose>{"The loss compares those scores with labels. Training changes the field and readout; an ordinary inference solve holds their weights fixed."}</Prose>

<H3>{"Why the connection to residual networks matters"}</H3>

<Prose>{"A residual update has the form"}</Prose>

<div className="neural-equation"><MathBlock>{"z_{k+1}=z_k+h f_\\theta(t_k,z_k)."}</MathBlock></div>

<Prose>{"This is also the forward Euler numerical method. It adds a scaled local change to the current state. If we refine the time grid while evaluating a consistently defined field, the discrete updates can approximate a continuous trajectory under the usual existence and numerical-convergence conditions."}</Prose>

<Prose>{"An arbitrary ResNet with unrelated weights at each layer does not automatically become a particular ODE just because it has many layers. We must define how its layer weights correspond to a function of time and what remains fixed as the grid changes."}</Prose>

<Prose>{"This gives Neural ODEs a useful form of parameter sharing. Evaluating one field 16 times need not introduce 16 sets of weights. It still consumes computation. “Continuous depth” describes the mathematical model; the machine performs finitely many operations. The "}<a href={"https://arxiv.org/html/1806.07366v5"}>{"original Neural ODE paper"}</a>{" develops this connection and several applications."}</Prose>

<OdeArchitectureFigure/>

<H2>{"2. The solver is part of the computation"}</H2>

<Prose>{"A solver chooses where to evaluate the field and how to combine those evaluations. It does not learn the field's weights during a forward pass."}</Prose>

<H3>{"Euler: one arrow per step"}</H3>

<Prose>{"With "}<InlineMath>{"h=T/N"}</InlineMath>{", Euler uses one field evaluation for each of "}<InlineMath>{"N"}</InlineMath>{" steps. Halving the step size usually reduces its global error by approximately a factor of two in the smooth, sufficiently resolved regime. That statement concerns a convergence regime, not every possible coarse step."}</Prose>

<Prose>{"For the two-dimensional rotation"}</Prose>

<div className="neural-equation"><MathBlock>{"f(x,y)=(-y,x),\\qquad z(0)=(1,0),"}</MathBlock></div>

<Prose>{"the exact path is "}<InlineMath>{"(\\cos t,\\sin t)"}</InlineMath>{". The point turns at one radian per unit time and keeps radius one. Euler does something revealing:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{bmatrix}x_{k+1}\\\\y_{k+1}\\end{bmatrix}\n=\n\\begin{bmatrix}1&-h\\\\h&1\\end{bmatrix}\n\\begin{bmatrix}x_k\\\\y_k\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Squaring and adding gives "}<InlineMath>{"x_{k+1}^2+y_{k+1}^2=(1+h^2)(x_k^2+y_k^2)"}</InlineMath>{". Euler's point spirals outward even though the exact dynamics conserve radius. We can identify the artifact algebraically rather than blaming a learned model."}</Prose>

<H3>{"Classical RK4: sample the turn inside a step"}</H3>

<Prose>{"Classical fourth-order Runge–Kutta evaluates four velocities:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{aligned}\nk_1&=f(t,z),\\\\\nk_2&=f(t+h/2,z+hk_1/2),\\\\\nk_3&=f(t+h/2,z+hk_2/2),\\\\\nk_4&=f(t+h,z+hk_3),\\\\\nz_{\\rm next}&=z+\\frac h6(k_1+2k_2+2k_3+k_4).\n\\end{aligned}"}</MathBlock></div>

<Prose>{"The middle evaluations probe where the state may be halfway through the interval. Their weighted combination cancels lower-order errors. For a sufficiently smooth, stable problem, the global error is "}<InlineMath>{"O(h^4)"}</InlineMath>{": halving "}<InlineMath>{"h"}</InlineMath>{" can reduce it by about 16, before rounding or other errors dominate. This is polynomial convergence, not exponential convergence."}</Prose>

<OdeWorkedStages/>

<Prose>{"Our executed rotation calculation integrates to "}<InlineMath>{"T=1"}</InlineMath>{" in float64:"}</Prose>

<NeuralTable caption={"Classical RK4: sample the turn inside a step"} headers={[<>{"Method"}</>,<>{"Steps"}</>,<>{"Field evaluations"}</>,<>{"Endpoint error, Euclidean norm"}</>]} rows={[[<>{"Euler"}</>,<>{"4"}</>,<>{"4"}</>,<>{"0.130661"}</>],[<>{"Euler"}</>,<>{"16"}</>,<>{"16"}</>,<>{"0.0317081"}</>],[<>{"Classical RK4"}</>,<>{"4"}</>,<>{"16"}</>,<>{"0.0000325318"}</>],[<>{"Classical RK4"}</>,<>{"16"}</>,<>{"64"}</>,<>{"0.000000127152"}</>]]} />

<Prose>{"At four RK4 steps, the radius is 0.99999327, not exactly one. A small error is still an error; RK4 is not generally a method that exactly conserves energy."}</Prose>

<OdeFixedLab/>

<H3>{"A readable differentiable implementation"}</H3>

<Prose>{"The downloadable "}<a href={"/learn-assets/neural-ode-continuous-depth-models/neural_ode_study.py"}>{"training program"}</a>{" contains the field, integrator, models, split and complete fitting loop. This is its core classical RK4 update, expressed with tensor operations so automatic differentiation can follow every stage:"}</Prose>

<CodeBlock language={"python"}>{"def rk4_step(field, time, state, step_size):\n    first = field(time, state)\n    second = field(time + step_size/2, state + step_size*first/2)\n    third = field(time + step_size/2, state + step_size*second/2)\n    fourth = field(time + step_size, state + step_size*third)\n    return state + step_size*(first + 2*second + 2*third + fourth)/6"}</CodeBlock>

<Prose>{"To build the trajectory, repeat this update at times "}<InlineMath>{"0,h,\\ldots,T-h"}</InlineMath>{". The complete program retains all states and uses that same expression inline. Its field concatenates the fixed depth coordinate to the state, applies a 16-unit tanh layer, and returns a derivative vector. Time is not a learnable parameter in this example; differentiating learned event times would require a different treatment."}</Prose>

<Prose>{"In a practical library, method names alone are insufficient for exact reproduction. The current "}<a href={"https://github.com/rtqichen/torchdiffeq"}>{"torchdiffeq README"}</a>{" specifies a 3/8-rule implementation for its fixed-step RK4; our displayed program uses classical RK4. Both are fourth-order methods, but their intermediate stages differ. Its callable uses the argument order (time, state), and fixed-step resolution is configured separately from the requested output times."}</Prose>

<H2>{"3. Adapt the steps to estimated error"}</H2>

<Prose>{"Taking tiny steps everywhere can waste work. An adaptive method estimates the error in a proposed step and decides whether to accept it."}</Prose>

<Prose>{"A simple teaching method combines Euler with "}<strong>{"Heun's method"}</strong>{", which averages the velocity at the beginning and the Euler-predicted endpoint:"}</Prose>

<div className="neural-equation"><MathBlock>{"z_E=z+h f(t,z),\\qquad\nz_H=z+\\frac h2\\{f(t,z)+f(t+h,z_E)\\}."}</MathBlock></div>

<Prose>{"The difference "}<InlineMath>{"z_H-z_E"}</InlineMath>{" is an error indicator. In our demonstration, coordinate "}<InlineMath>{"i"}</InlineMath>{" has scale"}</Prose>

<div className="neural-equation"><MathBlock>{"s_i=\\mathrm{atol}+\\mathrm{rtol}\\max(|z_i|,|z_{H,i}|),\n\\quad\nr=\\sqrt{\\frac1d\\sum_i[(z_{H,i}-z_{E,i})/s_i]^2}."}</MathBlock></div>

<Prose>{"Accept when "}<InlineMath>{"r\\leq1"}</InlineMath>{"; otherwise retry from the same state with a smaller step. The demonstration proposes the next step using a safety factor "}<InlineMath>{"0.9r^{-1/2}"}</InlineMath>{", clamped between 0.1 and 5. The exponent reflects this embedded difference's order. It is not a universal controller for all solvers."}</Prose>

<OdeAdaptiveWorkedFigure/>

<Prose>{"For the synthetic two-scale system "}<InlineMath>{"z'=\\operatorname{diag}(-1,-100)z"}</InlineMath>{", starting at "}<InlineMath>{"(1,1)"}</InlineMath>{" and ending at 0.4, our executed method uses 58 accepted and four rejected steps with relative tolerance 0.01 and absolute tolerance 0.0001. Tightening both tolerances tenfold produces 144 accepted and five rejected steps. Every attempt evaluates the field twice: 124 versus 298 evaluations."}</Prose>

<Prose>{"These are measured counts from this specified teaching solver, not a promised cost for Neural ODEs. Production methods such as Dormand–Prince use different formulas, interpolation and reuse strategies."}</Prose>

<OdeAdaptiveLab/>

<H3>{"Stiffness can hide beside a smooth trajectory"}</H3>

<Prose>{"Consider "}<InlineMath>{"y'=-\\kappa(y-\\cos t)-\\sin t"}</InlineMath>{", with "}<InlineMath>{"y(0)=1"}</InlineMath>{". The exact solution is "}<InlineMath>{"\\cos t"}</InlineMath>{" for every positive "}<InlineMath>{"\\kappa"}</InlineMath>{". It looks equally smooth when "}<InlineMath>{"\\kappa=5"}</InlineMath>{" and when "}<InlineMath>{"\\kappa=1000"}</InlineMath>{"."}</Prose>

<Prose>{"A perturbation away from that solution obeys "}<InlineMath>{"e'=-\\kappa e"}</InlineMath>{". Large "}<InlineMath>{"\\kappa"}</InlineMath>{" creates a rapidly decaying mode that can restrict an explicit solver's stable step size even when the desired solution changes slowly. That separation between fast stability constraints and slower behavior is a characteristic stiffness problem."}</Prose>

<Prose>{"Our SciPy run to time one, with relative tolerance "}<InlineMath>{"10^{-6}"}</InlineMath>{" and absolute tolerance "}<InlineMath>{"10^{-9}"}</InlineMath>{", illustrates it:"}</Prose>

<NeuralTable caption={"Stiffness can hide beside a smooth trajectory"} headers={[<>{""}<InlineMath>{"\\kappa"}</InlineMath>{""}</>,<>{"RK45 field evaluations"}</>,<>{"Radau field evaluations"}</>,<>{"Radau matrix factorizations"}</>]} rows={[[<>{"5"}</>,<>{"92"}</>,<>{"99"}</>,<>{"8"}</>],[<>{"1000"}</>,<>{"2162"}</>,<>{"84"}</>,<>{"16"}</>]]} />

<Prose>{"Both solved the stated problem successfully. A field evaluation is not a unit of equal total cost across these methods: implicit Radau also solves algebraic systems. These numbers do not establish a wall-clock speedup. The "}<a href={"https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html"}>{"SciPy solver documentation"}</a>{" explains explicit/implicit choices, Jacobians and event handling."}</Prose>

<Prose>{"High evaluation counts deserve diagnosis. Tight tolerances, poor coordinate scaling, nonsmooth fields and difficult transients can also increase work. “More evaluations” does not by itself prove stiffness or that an input is semantically harder."}</Prose>

<H2>{"4. Learn through a trajectory"}</H2>

<Prose>{"We now have a computation whose output depends on learned weights. A loss can differentiate that dependence in two main ways."}</Prose>

<H3>{"Differentiate the numerical program"}</H3>

<Prose>{"For fixed Euler steps, write "}<InlineMath>{"F_k=z_k+h f_\\theta(t_k,z_k)"}</InlineMath>{". Ordinary backpropagation applies the chain rule through each "}<InlineMath>{"F_k"}</InlineMath>{". With a final-state loss "}<InlineMath>{"L(z_N)"}</InlineMath>{", let "}<InlineMath>{"a_k"}</InlineMath>{" be the column vector "}<InlineMath>{"\\partial L/\\partial z_k"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"a_k=(I+hJ_zf_k)^\\top a_{k+1},\n\\qquad\n\\nabla_\\theta L=\\sum_k h(J_\\theta f_k)^\\top a_{k+1}."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"J_z f"}</InlineMath>{" is the matrix of field derivatives with respect to the state, and "}<InlineMath>{"J_\\theta f"}</InlineMath>{" collects derivatives with respect to weights. Their transposes carry a loss sensitivity backward from outputs to inputs. If the readout or initial state depends on parameters, add those paths too."}</Prose>

<Prose>{"RK4 has more intermediate operations, but automatic differentiation handles the same principle. It computes a gradient of the "}<strong>{"implemented finite computation"}</strong>{", subject to floating-point effects and any nondifferentiable branches. This is often called "}<strong>{"discretize then optimize"}</strong>{". Keeping every stage's graph consumes memory; checkpointing stores selected states and recomputes parts of the forward calculation during backward propagation."}</Prose>

<H3>{"Derive a continuous adjoint"}</H3>

<Prose>{"Alternatively, derive sensitivity equations for the continuous model first. For a terminal loss with no direct parameter dependence and parameter-independent "}<InlineMath>{"z_0"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{da}{dt}=-J_z f_\\theta(t,z(t))^\\top a(t),\n\\qquad a(T)=\\nabla_{z(T)}L,"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"\\nabla_\\theta L=\\int_0^T\nJ_\\theta f_\\theta(t,z(t))^\\top a(t)\\,dt."}</MathBlock></div>

<Prose>{"The adjoint says how much the final loss cares about a small change in the state at each time. A backward solve can accumulate this integral without retaining the entire forward computation graph."}</Prose>

<Prose>{""}<strong>{"Keep the integration direction explicit."}</strong>{" A parameter accumulator "}<InlineMath>{"g"}</InlineMath>{" initialized at "}<InlineMath>{"g(T)=0"}</InlineMath>{" and integrated from "}<InlineMath>{"T"}</InlineMath>{" down to zero uses "}<InlineMath>{"g'=-J_\\theta f^\\top a"}</InlineMath>{". The negative right side combined with backward limits yields the positive forward-time integral above."}</Prose>

<OdeAdjointFigure/>

<Prose>{"For losses at several observation times, the adjoint receives a jump from each observation's loss gradient. A running integral cost also contributes a term between observations. These extensions matter in time-series training; the terminal-only formula is not the full recipe for every objective."}</Prose>

<section data-lesson-teaching="" className="lesson-teaching-section">

<h3 className="lesson-teaching-section__title">Deeper: why the adjoint equation cancels state sensitivity</h3>

<Prose>{"Let "}<InlineMath>{"S(t)=\\partial z(t)/\\partial\\theta"}</InlineMath>{". Differentiating the state equation gives "}<InlineMath>{"S'=J_zf\\,S+J_\\theta f"}</InlineMath>{", with "}<InlineMath>{"S(0)=0"}</InlineMath>{" under our assumptions. The product rule gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{d}{dt}(a^\\top S)=a'^\\top S+a^\\top S'\n=-a^\\top J_zf S+a^\\top(J_zf S+J_\\theta f)\n=a^\\top J_\\theta f."}</MathBlock></div>

<Prose>{"Integrate from zero to "}<InlineMath>{"T"}</InlineMath>{". The left endpoint vanishes; the final term is "}<InlineMath>{"\\nabla_{z(T)}L^\\top S(T)"}</InlineMath>{", precisely the chain rule for the parameter gradient. This also explains the adjoint's negative sign. Forward sensitivities store a state-by-parameter matrix; the adjoint is attractive when there are many parameters and a scalar objective."}</Prose>

</section>

<H3>{"The two gradients need not be equal at finite resolution"}</H3>

<Prose>{"Use "}<InlineMath>{"z'=\\theta z"}</InlineMath>{", "}<InlineMath>{"z(0)=1.2"}</InlineMath>{", "}<InlineMath>{"T=1.3"}</InlineMath>{", target 0.4, and loss "}<InlineMath>{"\\frac12(z(T)-0.4)^2"}</InlineMath>{", evaluated at "}<InlineMath>{"\\theta=-0.7"}</InlineMath>{"."}</Prose>

<Prose>{"The exact state is "}<InlineMath>{"1.2e^{1.3\\theta}"}</InlineMath>{". Differentiating it gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{dL}{d\\theta}=(z(T)-0.4)\\,1.3z(T)=0.05213709."}</MathBlock></div>

<Prose>{"Four Euler steps instead produce "}<InlineMath>{"z_4=1.2(1+0.325\\theta)^4=0.42734163"}</InlineMath>{". Their autograd gradient is 0.01966276. Central differences of that same four-step loss agree with autograd. They do not agree with the continuous answer because they differentiate a different forward function."}</Prose>

<Prose>{"Four classical RK4 steps give 0.05213852, much closer here. A backward RK4 solve of the continuous adjoint initialized at the numerical forward endpoint gives 0.05214299. It introduces its own approximation; it is not exactly the RK4 program's reverse-mode derivative."}</Prose>

<OdeGradientWorkedFigure/>

<Prose>{"This is why “the gradient passed a check” needs a named reference. "}<a href={"https://arxiv.org/html/2005.13420v2"}>{"Onken and Ruthotto"}</a>{" study the distinction and solver changes in continuous models. Our scalar example is independently calculated rather than reproduced from their experiments."}</Prose>

<OdeGradientLab/>

<H3>{"Why saving no forward trajectory can be fragile"}</H3>

<Prose>{"The decay "}<InlineMath>{"z'=-20z"}</InlineMath>{", "}<InlineMath>{"z(0)=1"}</InlineMath>{", reaches "}<InlineMath>{"e^{-20}\\approx2.0612\\times10^{-9}"}</InlineMath>{" at time one. Reverse the exact equation from an endpoint perturbed by "}<InlineMath>{"10^{-8}"}</InlineMath>{". The recovered initial value is approximately 5.85165 instead of one: reverse evolution multiplies the perturbation by "}<InlineMath>{"e^{20}"}</InlineMath>{"."}</Prose>

<Prose>{"The forward process is stable but its inverse reconstruction is ill-conditioned. A more accurate solver helps numerical error; it cannot remove this mathematical amplification of an already present endpoint error. Checkpoints can reduce the length over which state must be reconstructed."}</Prose>

<Prose>{"“Constant memory” for a backsolve usually concerns dependence on the number of internal forward steps. Parameters, parameter gradients, requested output states, batches and local network activations still occupy memory. Current "}<a href={"https://docs.kidger.site/diffrax/api/adjoints/"}>{"Diffrax adjoint documentation"}</a>{" describes checkpointed differentiation of the numerical solution and distinguishes it from approximate continuous backsolves. The correct choice depends on the objective, solver, accuracy needs and available memory."}</Prose>

<H3>{"Control an actual solver API with the same field"}</H3>

<Prose>{"The mechanism owner is "}<code>{"integrate"}</code>{" in "}<a href={"/learn-assets/neural-ode-continuous-depth-models/neural_ode_study.py"}>{"neural_ode_study.py"}</a>{": it builds Euler and classical RK4 updates from tensor operations, keeping the computation graph for direct differentiation. "}<code>{"VectorField"}</code>{" and "}<code>{"DepthClassifier"}</code>{" later turn that mechanism into a learned classifier. Here we isolate the solver/library bridge on z′=a z, where both the trajectory and its derivatives are known. This avoids mistaking a similar classification score for a correct solver interface."}</Prose>

<Prose>{"Keep "}<a href={"/learn-assets/neural-ode-continuous-depth-models/solver_library_bridge.py"}>{"solver_library_bridge.py"}</a>{" beside that program. With the earlier PyTorch environment, install "}<code>{"torchdiffeq==0.2.5"}</code>{", then run "}<code>{"python solver_library_bridge.py"}</code>{". The target for these new authored examples is PyTorch 2.14.0, CPU float64. The local bridge has now been executed with torchdiffeq 0.2.5 and PyTorch 2.14.0 on CPU float64. The fixed-step Euler endpoint and both initial-state and rate gradients match the scratch program; adaptive direct and continuous-adjoint results pass the stated analytic tolerances. The isolated replay also re-evaluated every saved classifier on its declared data roles."}</Prose>

<Prose>{""}<code>{"odeint(field, initial, times)"}</code>{" returns a tensor with requested time as its first axis. Those times request output values; an adaptive solver may take many internal steps between them. "}<code>{"method='euler', options={'step_size': 1/8}"}</code>{" gives the same eight Euler steps as our scratch route. We compare both final states and both kinds of derivative: with respect to the initial state and the field parameter. In contrast, the package's "}<code>{"rk4"}</code>{" uses the 3/8 tableau, while our scratch program uses classical RK4. Their order alone does not justify step-by-step equality. The solver choices and adjoint interface are documented in "}<a href={"https://github.com/rtqichen/torchdiffeq"}>{"the author's repository"}</a>{"."}</Prose>

<OdeSolverApiFigure/>

<Prose>{"For endpoint time one and loss L=½Σz(1)², exact calculus gives z(1)=exp(a)z₀, ∂L/∂z₀=exp(2a)z₀, and ∂L/∂a=exp(2a)Σz₀². The Euler solution instead uses factor (1+a/8)⁸. The adaptive direct and adjoint solves target the continuous answer within stated tolerances; we do "}<strong>{"not"}</strong>{" require them to equal the eight-step Euler answer. "}<code>{"odeint_adjoint"}</code>{" receives an "}<code>{"nn.Module"}</code>{" so its trainable field parameter can be found, and we set forward and backward tolerances explicitly."}</Prose>

<section data-lesson-teaching="" className="lesson-teaching-section"><h3 className="lesson-teaching-section__title">Read the complete runnable solver-library bridge</h3><CodeBlock language={"python"}>{"\"\"\"A matched Euler program, adaptive solve, and continuous-adjoint comparison.\n\nAuthoring targets: torch 2.14.0, torchdiffeq 0.2.5. CPU float64.\nRun beside neural_ode_study.py; no fitting or dataset download occurs.\n\"\"\"\nimport torch\nfrom torch import nn\nfrom torchdiffeq import odeint, odeint_adjoint\nfrom neural_ode_study import integrate\n\n\nclass Decay(nn.Module):\n    def __init__(self, rate=-.7):\n        super().__init__()\n        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float64))\n\n    def forward(self, time, state):\n        return self.rate * state\n\n\ndef value_and_derivatives(route):\n    field = Decay()\n    initial = torch.tensor([[1.5], [-.5]], dtype=torch.float64, requires_grad=True)\n    times = torch.tensor([0., 1.], dtype=torch.float64)\n    if route == \"scratch_euler\":\n        result = integrate(field, initial, steps=8, method=\"euler\")[-1]\n    elif route == \"library_euler\":\n        result = odeint(field, initial, times, method=\"euler\",\n                        options={\"step_size\": 1/8})[-1]\n    elif route == \"adaptive\":\n        result = odeint(field, initial, times, method=\"dopri5\", rtol=1e-9, atol=1e-11)[-1]\n    elif route == \"adjoint\":\n        result = odeint_adjoint(field, initial, times, method=\"dopri5\",\n                                rtol=1e-9, atol=1e-11, adjoint_method=\"dopri5\",\n                                adjoint_rtol=1e-9, adjoint_atol=1e-11)[-1]\n    else:\n        raise ValueError(route)\n    loss = result.square().sum()/2\n    initial_gradient, rate_gradient = torch.autograd.grad(loss, (initial, field.rate))\n    return result.detach(), initial_gradient.detach(), rate_gradient.detach()\n\n\ndef main():\n    scratch, library = [value_and_derivatives(route)\n                        for route in (\"scratch_euler\", \"library_euler\")]\n    for left, right in zip(scratch, library):\n        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)\n    initial = torch.tensor([[1.5], [-.5]], dtype=torch.float64)\n    factor = torch.exp(torch.tensor(-.7, dtype=torch.float64))\n    exact = (initial*factor, initial*factor.square(), initial.square().sum()*factor.square())\n    for route in (\"adaptive\", \"adjoint\"):\n        actual = value_and_derivatives(route)\n        for value, reference in zip(actual, exact):\n            torch.testing.assert_close(value, reference, rtol=2e-7, atol=2e-9)\n        print(route, \"endpoint/initial-gradient/rate-gradient:\", *actual)\n    print(\"Euler endpoint, initial gradient, rate gradient:\", *library)\n\n\nif __name__ == \"__main__\":\n    main()"}</CodeBlock></section>

<Prose>{"The continuous adjoint changes how derivatives are computed; it does not erase reconstruction error, intermediate saved output states, parameter gradients or library overhead. Direct differentiation through K accepted numerical steps generally retains work proportional to K, while a continuous backsolve trades recomputation and its own numerical sensitivity for less retained internal-step history. Smoothness, solver stability and state dimension still matter. The current example has no events, stochastic layers or discontinuous observation jumps; inserting those changes the problem."}</Prose>

<Prose>{""}<strong>{"Take control."}</strong>{" Change a to −4 and compare Euler at 1, 4, 8 and 32 steps with the same adaptive solve. Keep the parameter fixed: do not retrain it to compensate for a changed solver. Then tighten only the adjoint tolerances and inspect which derivative discrepancy changes."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"One Euler step multiplies the state by −3, which reverses sign and magnifies it even though the continuous system decays. Four steps multiply by zero at every step, incorrectly erasing the state. Eight give factor (½)⁸; as steps increase, (1−4/K)^K approaches exp(−4). For a finite K, direct differentiation correctly differentiates that discrete factor, not exp(a). Tightening backward tolerance can improve the continuous-adjoint approximation but cannot repair an intentionally inaccurate forward Euler trajectory. Report endpoint, initial-state gradient and parameter gradient separately; a small final loss alone misses this distinction."}</Prose>

</details>

<H2>{"5. What continuous flow can—and cannot—rearrange"}</H2>

<Prose>{"Suppose a field is continuous in time, locally Lipschitz in state, and solutions exist uniquely over the interval of interest. Starting at a specified state and time determines one trajectory. Locally Lipschitz means that, near each state, small state changes cannot cause an unbounded ratio of change in the field; it is a useful condition for uniqueness."}</Prose>

<Prose>{"Two distinct trajectories cannot meet at the "}<strong>{"same state and same time"}</strong>{" and then have different pasts within that common interval: uniqueness applied backward would contradict their different starts. Crossing lines in a two-dimensional projection of a higher-dimensional state are not a violation. Neither is visiting the same position at different times."}</Prose>

<Prose>{"This has a concrete consequence in one dimension. If "}<InlineMath>{"z_A(0)<z_B(0)"}</InlineMath>{", their ordering cannot reverse along a well-defined unique flow. Reversing order would require a meeting."}</Prose>

<Prose>{"Take inputs −1, 0, 1, where the outer two belong to class one and the middle belongs to class zero. A one-dimensional unique flow followed by a single linear threshold cannot solve this arrangement. A threshold separates a line into two intervals; preserving order leaves the middle between the outer points. A nonlinear readout could change the conclusion, so the readout assumption belongs in the explanation."}</Prose>

<OdeOrderPreservationFigure/>

<H3>{"Augment the state with room to move"}</H3>

<Prose>{"Append a zero coordinate: "}<InlineMath>{"(x,0)"}</InlineMath>{". Now use the two-dimensional rule "}<InlineMath>{"x'=0,\\ y'=x^2"}</InlineMath>{". At depth "}<InlineMath>{"T"}</InlineMath>{", the state is "}<InlineMath>{"(x,Tx^2)"}</InlineMath>{". At "}<InlineMath>{"T=1"}</InlineMath>{", the three inputs become "}<InlineMath>{"(-1,1),(0,0),(1,1)"}</InlineMath>{". A linear readout testing "}<InlineMath>{"y>0.5"}</InlineMath>{" separates the classes."}</Prose>

<Prose>{"No two full trajectories meet. The added coordinate gives the representation another direction in which to separate examples. It is a constructive illustration of augmentation, not a trained experiment or a proof that every augmented model learns easily. "}<a href={"https://arxiv.org/html/1904.01681v1"}>{"Augmented Neural ODEs"}</a>{" investigates this idea and its computational consequences."}</Prose>

<OdeTopologyLab/>

<Prose>{"A coarse numerical method need not preserve an exact flow's properties. For "}<InlineMath>{"z'=-2z"}</InlineMath>{", one Euler step with "}<InlineMath>{"h=1"}</InlineMath>{" maps "}<InlineMath>{"z"}</InlineMath>{" to "}<InlineMath>{"-z"}</InlineMath>{", reversing order. The exact map multiplies by "}<InlineMath>{"e^{-2}>0"}</InlineMath>{" and preserves order. A discrete model that exploits a coarse solver's behavior may change substantially when evaluated with finer steps."}</Prose>

<OdeOrderFigure/>

<Prose>{"For finite sampled rings or point clouds, gaps can also let trajectories thread between examples. A theorem about an entire continuous region is stronger than a claim about a finite training set. Both distinctions prevent us from mistaking an attractive picture for a general impossibility result."}</Prose>

<H2>{"6. Train a small continuous-depth classifier on real measurements"}</H2>

<Prose>{"Our data are the 150 Iris measurements distributed by UCI: sepal length, sepal width, petal length and petal width, all in centimeters, with three species labels. The "}<a href={"https://archive.ics.uci.edu/dataset/53/iris"}>{"dataset"}</a>{" is credited to R. A. Fisher and licensed CC BY 4.0. The retained "}<a href={"/learn-assets/neural-ode-continuous-depth-models/iris.csv"}>{"CSV"}</a>{" uses the corrected values documented in its "}<a href={"/learn-assets/neural-ode-continuous-depth-models/data-provenance.md"}>{"provenance"}</a>{"."}</Prose>

<Prose>{"This is a feature-transformation experiment. Each flower supplies an initial four-dimensional state after standardization. Solver time describes representation depth, not elapsed botanical time."}</Prose>

<H3>{"Keep the comparison interpretable"}</H3>

<Prose>{"Identical four-feature rows are grouped before assigning data roles; source rows 102 and 143 are duplicates. We keep all original rows in the download but score each unique vector once. A fixed stratified split assigns 90 unique examples to fitting, 30 to validation and 29 to assessment. The fitting examples alone determine the feature means and population standard deviations."}</Prose>

<Prose>{"We compare four models:"}</Prose>

<NeuralTable caption={"Keep the comparison interpretable"} headers={[<>{"Model"}</>,<>{"Computation before a linear three-class head"}</>,<>{"Trainable parameters"}</>]} rows={[[<>{"Linear"}</>,<>{"No learned feature transformation"}</>,<>{"15"}</>],[<>{"Residual"}</>,<>{"Four distinct 16-unit tanh fields, each used for one step"}</>,<>{"671"}</>],[<>{"Neural ODE"}</>,<>{"One shared 16-unit tanh field; four classical RK4 steps"}</>,<>{"179"}</>],[<>{"Augmented ODE"}</>,<>{"Append two zeros; one six-dimensional field; four RK4 steps"}</>,<>{"251"}</>]]} />

<Prose>{"The two ODE models perform 16 field evaluations per example. Their parameter counts stay fixed if we change solver resolution. The residual model has separate weights at its four blocks. These models have different capacities; this is not a parameter-matched benchmark."}</Prose>

<Prose>{"The complete "}<a href={"/learn-assets/neural-ode-continuous-depth-models/neural_ode_study.py"}>{"study program"}</a>{" runs three declared seeds—13, 37 and 61—for each model. Every run uses 300 full-batch AdamW updates, learning rate 0.01 and weight decay 0.001, in float64 on CPU. Validation loss is measured after update one and every 25 updates. We select the lowest-validation-loss checkpoint within each run, then evaluate that checkpoint on assessment data. All 12 runs and their curves remain in "}<a href={"/learn-assets/neural-ode-continuous-depth-models/study-results.json"}>{"the result file"}</a>{"; "}<a href={"/learn-assets/neural-ode-continuous-depth-models/fitted-models.json"}>{"selected weights"}</a>{" are retained."}</Prose>

<H3>{"What actually happened"}</H3>

<NeuralTable caption={"What actually happened"} headers={[<>{"Model"}</>,<>{"Assessment cross-entropy, seeds 13 / 37 / 61"}</>,<>{"Correct out of 29, same order"}</>]} rows={[[<>{"Linear"}</>,<>{"0.22045 / 0.20028 / 0.21245"}</>,<>{"28 / 28 / 28"}</>],[<>{"Residual"}</>,<>{"0.16763 / 0.10539 / 0.05133"}</>,<>{"28 / 28 / 29"}</>],[<>{"Neural ODE"}</>,<>{"0.15239 / 0.10335 / 0.06212"}</>,<>{"28 / 28 / 29"}</>],[<>{"Augmented ODE"}</>,<>{"0.04204 / 0.10549 / 0.03648"}</>,<>{"29 / 28 / 29"}</>]]} />

<Prose>{"The uniform three-class predictor has cross-entropy "}<InlineMath>{"\\log3\\approx1.09861"}</InlineMath>{". A fixed class-zero choice scores 10/29. The linear model is already strong: a small reduction in classification errors cannot explain every difference in cross-entropy. Probability assigned to the true class matters even when the winning label stays unchanged."}</Prose>

<Prose>{"All nonlinear runs selected update 50, while the linear runs selected update 300. Later training loss improvement therefore did not automatically mean better validation performance. There was no post-assessment search for a more impressive configuration."}</Prose>

<OdeStudyFigure/>

<H3>{"Follow one actual flower through the learned field"}</H3>

<Prose>{"For validation source row 64, the seed-37 Neural ODE assigns probabilities approximately "}<InlineMath>{"(0.01744,0.93374,0.04882)"}</InlineMath>{" in the order setosa, versicolor, virginica. Increasing its petal length by 0.6 cm while holding the other measurements fixed changes them to "}<InlineMath>{"(0.00730,0.67440,0.31830)"}</InlineMath>{". The winning label stays versicolor, but the evidence shifts substantially toward virginica."}</Prose>

<Prose>{"These are full forward passes through the saved learned model, not a hand-drawn boundary. The input edit is a hypothetical measurement intervention; it does not establish a biological causal effect."}</Prose>

<OdeWorkedIrisFigure/>

<OdeIrisLab/>

<Prose>{"For unchanged row 64, using 4, 16 and 64 RK4 steps gives versicolor probabilities 0.9337389, 0.9335432 and 0.9335421. Four Euler steps give 0.9442828. The weights are identical; the numerical realization changed. Agreement under refinement is useful evidence that the model is behaving like its continuous formulation locally. It does not prove accuracy everywhere or improve the training data automatically."}</Prose>

<Prose>{"Independent NumPy inference reproduces the saved PyTorch models' logits and traces within "}<InlineMath>{"8.9\\times10^{-16}"}</InlineMath>{" over the checked input/method/step cases. Input derivatives also agree with central differences. These author calculations support the manuscript and future interactive model."}</Prose>

<H3>{"Run the complete study"}</H3>

<Prose>{"Keep "}<a href={"/learn-assets/neural-ode-continuous-depth-models/iris.csv"}>{"iris.csv"}</a>{" beside the "}<a href={"/learn-assets/neural-ode-continuous-depth-models/neural_ode_study.py"}>{"study program"}</a>{". With Python, NumPy and PyTorch installed, run the program from that directory:"}</Prose>

<CodeBlock language={"text"}>{"python neural_ode_study.py"}</CodeBlock>

<Prose>{"It writes the complete results and selected weights. Install SciPy as well to run the separate "}<a href={"/learn-assets/neural-ode-continuous-depth-models/ode_calculations.py"}>{"calculation program"}</a>{", which imports the study's model definitions and reuses the saved weights. That second command does not train again. The "}<a href={"/learn-assets/neural-ode-continuous-depth-models/data-provenance.md"}>{"provenance"}</a>{" records the versions used here."}</Prose>

<section data-lesson-teaching="" className="lesson-teaching-section">

<h3 className="lesson-teaching-section__title">Complete training program: data roles, models, selection and saved weights</h3>

<CodeBlock language={"python"}>{"\"\"\"Declared real Iris study: learned depth is not botanical time.\"\"\"\nfrom copy import deepcopy\nfrom pathlib import Path\nimport csv\nimport hashlib\nimport json\nimport numpy as np\nimport torch\nfrom torch import nn\n\nDIRECTORY = Path(__file__).resolve().parent\ntorch.set_num_threads(1)\ntorch.set_default_dtype(torch.float64)\n\n\ndef load_data():\n    with (DIRECTORY / \"iris.csv\").open(newline=\"\") as handle:\n        records = list(csv.DictReader(handle))\n    columns = [\"sepal_length_cm\", \"sepal_width_cm\", \"petal_length_cm\", \"petal_width_cm\"]\n    names = [\"setosa\", \"versicolor\", \"virginica\"]\n    features = np.array([[float(row[key]) for key in columns] for row in records])\n    labels = np.array([names.index(row[\"species\"]) for row in records])\n    groups = {}\n    for index, row in enumerate(features):\n        groups.setdefault(tuple(row), []).append(index)\n    for members in groups.values():\n        assert len(set(labels[members])) == 1\n    kept = np.array(sorted(members[0] for members in groups.values()))\n    roles = {key: [] for key in (\"fit\", \"validation\", \"assessment\")}\n    generator = np.random.default_rng(926)\n    for label in range(3):\n        indices = generator.permutation(kept[labels[kept] == label])\n        for key, subset in zip(roles, (indices[:30], indices[30:40], indices[40:])):\n            roles[key].extend(subset.tolist())\n    mean, scale = features[roles[\"fit\"]].mean(0), features[roles[\"fit\"]].std(0)\n    metadata = dict(columns=columns, classes=names, mean=mean.tolist(), scale=scale.tolist(),\n        roles={key: [index+1 for index in indices] for key, indices in roles.items()},\n        duplicate_groups=[[i+1 for i in group] for group in groups.values() if len(group)>1],\n        unique_count=len(kept))\n    return torch.tensor((features-mean)/scale), torch.tensor(labels), roles, metadata\n\n\nclass VectorField(nn.Module):\n    def __init__(self, dimension):\n        super().__init__()\n        self.input = nn.Linear(dimension+1, 16)\n        self.output = nn.Linear(16, dimension)\n\n    def forward(self, time, state):\n        clock = torch.full_like(state[:, :1], float(time))\n        return self.output(torch.tanh(self.input(torch.cat((state, clock), -1))))\n\n\ndef integrate(field, initial, steps=4, method=\"rk4\", endpoint=1.):\n    state = initial\n    trace = [state]\n    step_size = endpoint/steps\n    for step in range(steps):\n        time = step*step_size\n        first = field(time, state)\n        if method == \"euler\":\n            state = state + step_size*first\n        elif method == \"rk4\":\n            second = field(time+step_size/2, state+step_size*first/2)\n            third = field(time+step_size/2, state+step_size*second/2)\n            fourth = field(time+step_size, state+step_size*third)\n            state = state + step_size*(first+2*second+2*third+fourth)/6\n        else:\n            raise ValueError(method)\n        trace.append(state)\n    return torch.stack(trace)\n\n\nclass DepthClassifier(nn.Module):\n    def __init__(self, kind):\n        super().__init__()\n        self.kind = kind\n        self.dimension = 6 if kind == \"augmented_ode\" else 4\n        if kind in (\"neural_ode\", \"augmented_ode\"):\n            self.field = VectorField(self.dimension)\n        if kind == \"residual\":\n            self.blocks = nn.ModuleList([VectorField(4) for _ in range(4)])\n        self.readout = nn.Linear(self.dimension, 3)\n\n    def forward(self, inputs, steps=4, method=\"rk4\", endpoint=1., return_trace=False):\n        state = torch.cat((inputs, inputs.new_zeros((len(inputs), self.dimension-4))), -1)\n        if self.kind in (\"neural_ode\", \"augmented_ode\"):\n            trace = integrate(self.field, state, steps, method, endpoint)\n            state = trace[-1]\n        else:\n            states = [state]\n            if self.kind == \"residual\":\n                for index, block in enumerate(self.blocks):\n                    state = state + .25*block(index/4, state)\n                    states.append(state)\n            trace = torch.stack(states)\n        logits = self.readout(state)\n        return (logits, trace) if return_trace else logits\n\n\ndef evaluate(model, features, labels):\n    with torch.no_grad():\n        logits = model(features)\n    return dict(cross_entropy=float(nn.functional.cross_entropy(logits, labels)),\n                correct=int((logits.argmax(-1)==labels).sum()), count=len(labels))\n\n\ndef main():\n    features, labels, roles, metadata = load_data()\n    reports, snapshots = [], []\n    for kind in [\"linear\", \"residual\", \"neural_ode\", \"augmented_ode\"]:\n        for seed in [13, 37, 61]:\n            torch.manual_seed(seed)\n            model = DepthClassifier(kind)\n            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)\n            best_loss, selected, curves = float(\"inf\"), None, []\n            for step in range(1, 301):\n                optimizer.zero_grad(set_to_none=True)\n                loss = nn.functional.cross_entropy(model(features[roles[\"fit\"]]), labels[roles[\"fit\"]])\n                loss.backward()\n                optimizer.step()\n                if step==1 or step%25==0:\n                    fit = evaluate(model, features[roles[\"fit\"]], labels[roles[\"fit\"]])\n                    validation = evaluate(model, features[roles[\"validation\"]], labels[roles[\"validation\"]])\n                    curves.append(dict(step=step, fit=fit, validation=validation))\n                    if validation[\"cross_entropy\"] < best_loss:\n                        best_loss = validation[\"cross_entropy\"]\n                        selected = deepcopy(model.state_dict())\n                        selected_step = step\n            model.load_state_dict(selected)\n            report = dict(kind=kind, seed=seed, parameters=sum(p.numel() for p in model.parameters()),\n                selected_step=selected_step, curves=curves,\n                fit=evaluate(model,features[roles[\"fit\"]],labels[roles[\"fit\"]]),\n                validation=evaluate(model,features[roles[\"validation\"]],labels[roles[\"validation\"]]),\n                assessment=evaluate(model,features[roles[\"assessment\"]],labels[roles[\"assessment\"]]))\n            reports.append(report)\n            snapshots.append(dict(kind=kind,seed=seed,selected_step=selected_step,\n                state={key:value.tolist() for key,value in selected.items()}))\n            print(kind,seed,selected_step,report[\"assessment\"],flush=True)\n    result=dict(data=metadata, updates=300, learning_rate=.01,weight_decay=.001,\n        seeds=[13,37,61], dtype=\"float64\", torch=torch.__version__, numpy=np.__version__,\n        dataset_sha256=hashlib.sha256((DIRECTORY/\"iris.csv\").read_bytes()).hexdigest(),runs=reports)\n    (DIRECTORY/\"study-results.json\").write_text(json.dumps(result,indent=2,allow_nan=False)+\"\\n\")\n    (DIRECTORY/\"fitted-models.json\").write_text(json.dumps(snapshots,allow_nan=False)+\"\\n\")\n\n\nif __name__ == \"__main__\":\n    main()"}</CodeBlock>

<Prose>{"Read the model definition and integrator first, then follow one fitting update. The remaining loop repeats that update under the declared selection protocol. The saved weights let you change inputs or solver resolution without rerunning the fits."}</Prose>

</section>

<H2>{"7. Irregular observations: evolving state is not new evidence"}</H2>

<Prose>{"Suppose a sensor reports at times 0.2, 0.9 and 1.3. A model should distinguish “nothing new was observed” from “the sensor observed zero.” A continuous hidden state gives us a way to evolve between those moments, but the ODE alone does not decide how to incorporate arriving measurements."}</Prose>

<Prose>{"Three constructions answer different questions:"}</Prose>

<NeuralTable caption={"7. Irregular observations: evolving state is not new evidence"} headers={[<>{"Construction"}</>,<>{"What happens between observations?"}</>,<>{"How does a new observation affect state?"}</>]} rows={[[<>{"Plain Neural ODE"}</>,<>{"Integrate from an initial state"}</>,<>{"It does not, unless an input/update mechanism is added"}</>],[<>{"ODE-RNN"}</>,<>{"Integrate hidden state across each time gap"}</>,<>{"Apply a recurrent update at each observed time"}</>],[<>{"Latent ODE"}</>,<>{"Evolve a sampled latent initial state"}</>,<>{"An encoder infers a distribution over that initial state from a chosen observation set"}</>]]} />

<Prose>{"A time-aware ordinary RNN is also a legitimate baseline: it can receive elapsed time as an input or use a decay rule. Continuity is one modeling choice, not a prerequisite for handling irregular timestamps."}</Prose>

<H3>{"Work through an ODE-RNN-shaped calculation"}</H3>

<Prose>{"Use a deliberately simple hidden state: between observations, "}<InlineMath>{"h'=-0.5h"}</InlineMath>{". At an observation with value "}<InlineMath>{"x_i"}</InlineMath>{", update "}<InlineMath>{"h^+=0.7h^-+0.3x_i"}</InlineMath>{". Start at zero."}</Prose>

<Prose>{"For values "}<InlineMath>{"(1,-0.5,0.8)"}</InlineMath>{" at times "}<InlineMath>{"(0.2,0.9,1.3)"}</InlineMath>{":"}</Prose>

<ul><li>{"At 0.2, the update gives 0.3."}</li><li>{"Just before 0.9, decay gives "}<InlineMath>{"0.3e^{-0.35}=0.211406"}</InlineMath>{". The observation changes it to −0.002016."}</li><li>{"Just before 1.3, it is −0.001650. The new observation changes it to 0.238845."}</li><li>{"At a query time of 1.6, the state is "}<InlineMath>{"0.238845e^{-0.15}=0.205576"}</InlineMath>{"."}</li></ul>

<OdeObservationWorkedFigure/>

<Prose>{"Omitting the middle observation gives 0.310853 at 1.6. Observing zero instead gives 0.279568. The latter still applies the update's 0.7 retention. Missingness and zero are different computations."}</Prose>

<OdeObservationLab/>

<Prose>{"The "}<a href={"https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf"}>{"Latent ODE paper"}</a>{" combines continuous dynamics with observation-dependent inference. Its generative model samples "}<InlineMath>{"z_0"}</InlineMath>{", integrates "}<InlineMath>{"z(t)"}</InlineMath>{", and decodes observation distributions. An encoder approximates "}<InlineMath>{"q(z_0\\mid\\{t_i,x_i\\})"}</InlineMath>{"; training balances expected reconstruction log-likelihood against divergence from a prior."}</Prose>

<Prose>{"Interpolation may condition on observations on both sides of a query. Forecasting must restrict the encoder to information available by the forecast origin. The paper also models observation timing with a Poisson process; a basic value-only model does not automatically handle informative missingness. The lesson's scalar calculation explains the mechanism without claiming to reproduce a clinical study or validate a medical system."}</Prose>

<H2>{"8. Deeper branch: move density as well as points"}</H2>

<Prose>{"A normalizing flow transforms samples from a simple distribution into samples from a more useful one while tracking how density changes. Imagine stretching a small patch containing a fixed amount of probability. If its volume grows, its density must decrease."}</Prose>

<Prose>{"For a differentiable vector field and a well-defined invertible flow,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{d}{dt}\\log p_t(z(t))\n=-\\nabla\\cdot f(t,z(t))\n=-\\operatorname{tr}J_zf(t,z(t))."}</MathBlock></div>

<Prose>{"The "}<strong>{"divergence"}</strong>{" is the sum of the field's same-coordinate derivatives. It measures instantaneous local volume expansion. The negative sign converts expansion into a density decrease."}</Prose>

<Prose>{"For "}<InlineMath>{"f(z)=Az"}</InlineMath>{", the exact flow is "}<InlineMath>{"z(T)=e^{TA}z_0"}</InlineMath>{", and its volume multiplier is "}<InlineMath>{"\\det(e^{TA})=e^{T\\operatorname{tr}A}"}</InlineMath>{". Take"}</Prose>

<div className="neural-equation"><MathBlock>{"A=\\begin{bmatrix}0.2&2\\\\0.4&-0.1\\end{bmatrix},\\qquad T=2."}</MathBlock></div>

<Prose>{"The trace is 0.1. Volume multiplies by "}<InlineMath>{"e^{0.2}=1.221403"}</InlineMath>{"; density along the trajectory multiplies by "}<InlineMath>{"e^{-0.2}=0.818731"}</InlineMath>{". The off-diagonal terms change the patch's shape and trajectory even though they do not appear directly in the trace."}</Prose>

<OdeDensityFigure/>

<Prose>{"A "}<strong>{"continuous normalizing flow"}</strong>{", or CNF, integrates both state and log-density. For likelihood evaluation we also need the base density and the correct direction of integration. For sampling, moving a base sample may suffice without computing its density."}</Prose>

<H3>{"Estimate a trace without constructing a whole Jacobian"}</H3>

<Prose>{"For a random vector "}<InlineMath>{"\\epsilon"}</InlineMath>{" with mean zero and covariance identity,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[\\epsilon^\\top J\\epsilon]=\\operatorname{tr}J."}</MathBlock></div>

<Prose>{"This follows by expanding the sum: cross-coordinate terms have zero expectation and diagonal terms have unit second moment. Jacobian-vector or vector-Jacobian products can evaluate the quadratic form without explicitly storing all "}<InlineMath>{"d^2"}</InlineMath>{" entries. "}<a href={"https://arxiv.org/html/1810.01367v2"}>{"FFJORD"}</a>{" uses this strategy in continuous density models."}</Prose>

<Prose>{"Our matrix above makes the randomness visible. The four equally likely sign vectors "}<InlineMath>{"(\\pm1,\\pm1)"}</InlineMath>{" produce trace estimates 2.5, −2.3, −2.3 and 2.5. Their mean is 0.1. One estimate can be far from the trace, even with the wrong sign. The estimator is useful because of its expectation and computational cost, not because every draw is exact."}</Prose>

<Prose>{"Keep a probe fixed over an individual integration solve so the augmented right-hand side is a consistent function for the solver. A finite-sample trace estimate and numerical integration introduce distinct errors. An unbiased ideal log-density estimate does not imply an unbiased density after exponentiation. Full maximum-likelihood training, bottleneck trace identities and model evaluation belong in the "}<a href={"/learn/path/full-curriculum/normalizing-flows-realnvp-glow-neural-ode?module=generative-models"}>{"normalizing-flows lesson"}</a>{"."}</Prose>

<H2>{"9. Deeper branch: learn a velocity without solving during every training example"}</H2>

<Prose>{"Suppose we pair a noise sample "}<InlineMath>{"x_0"}</InlineMath>{" with a data example "}<InlineMath>{"x_1"}</InlineMath>{", choose a time uniformly, and form"}</Prose>

<div className="neural-equation"><MathBlock>{"x_t=(1-t)x_0+tx_1,\\qquad v_{\\rm target}=x_1-x_0."}</MathBlock></div>

<Prose>{"The chosen straight path and its velocity can be calculated directly. Train a network to predict this velocity from "}<InlineMath>{"(t,x_t)"}</InlineMath>{" using squared error. At generation time, solve "}<InlineMath>{"x'=v_\\theta(t,x)"}</InlineMath>{" from a noise sample."}</Prose>

<Prose>{"This is a simple endpoint-interpolation form of "}<strong>{"conditional flow matching"}</strong>{". Some formulations retain a small final noise level; their path and target velocity change correspondingly. The key computational benefit is that the ordinary regression training objective does not require integrating the learned ODE for each target. Generating a new sample still requires a solve unless an additional approximation or distillation changes that process."}</Prose>

<Prose>{"Why can different training paths provide useful targets at the same place? Squared-error regression learns their conditional mean velocity. Consider two synthetic pairs: "}<InlineMath>{"0\\to2"}</InlineMath>{" and "}<InlineMath>{"2\\to0"}</InlineMath>{". At time 0.5, both are at position one, but their target velocities are +2 and −2. A single model receiving only that position and time cannot return both. Its least-squares prediction is zero: mean loss four, compared with eight if it always predicts +2."}</Prose>

<OdeFlowMatchingFigure/>

<Prose>{"This local discrete example explains averaging; it is not a full smooth-density generative model. In the continuous theory, the conditional mean field transports the marginal probability path under appropriate regularity assumptions. Straight conditional training paths do not guarantee straight generated trajectories or a globally optimal transport map. "}<a href={"https://arxiv.org/html/2210.02747v2"}>{"Flow Matching for Generative Modeling"}</a>{" states that distinction explicitly."}</Prose>

<Prose>{"This connection helps interpret modern generative systems without assuming that all ODE-trained models use the same loss. CNF likelihood training uses density change; flow-matching regression uses target velocities. A variational latent ODE uses reconstruction and a prior penalty. They share differential-equation machinery but optimize different objectives. Continue the generative route through "}<a href={"/learn/path/full-curriculum/rectified-flow-flow-matching?module=generative-models"}>{"Rectified Flow and Flow Matching"}</a>{"."}</Prose>

<H2>{"10. Deeper branch: put useful structure into a learned rule"}</H2>

<Prose>{"A generic field can model many relationships, but useful restrictions can make a problem easier to learn and interpret. Start with the structure of the task."}</Prose>

<Prose>{""}<strong>{"Known physics plus a learned correction."}</strong>{" If a mechanistic model explains most of a process but misses one force or reaction term, write "}<InlineMath>{"z'=f_{\\rm known}(t,z)+f_\\theta(t,z)"}</InlineMath>{". The learned part need not rediscover everything. It still must be tested outside the observations used for fitting: many different fields can match one short trajectory. Fitting a trajectory is not proof that the underlying physical mechanism has been identified."}</Prose>

<Prose>{""}<strong>{"Energy-based dynamics."}</strong>{" For position "}<InlineMath>{"q"}</InlineMath>{", momentum "}<InlineMath>{"p"}</InlineMath>{" and a learned energy "}<InlineMath>{"H_\\theta(q,p)"}</InlineMath>{", Hamilton's equations use "}<InlineMath>{"q'=\\partial H/\\partial p"}</InlineMath>{", "}<InlineMath>{"p'=-\\partial H/\\partial q"}</InlineMath>{". Along the exact autonomous dynamics,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{dH}{dt}\n=\\frac{\\partial H}{\\partial q}^{\\!\\top}\\frac{\\partial H}{\\partial p}\n-\\frac{\\partial H}{\\partial p}^{\\!\\top}\\frac{\\partial H}{\\partial q}=0."}</MathBlock></div>

<Prose>{"That cancellation is an architectural property. A numerical solver may still drift in energy; a model of a damped system may need dissipation instead of an exact conservation constraint. The earlier rotation example made the solver issue visible before this more advanced application."}</Prose>

<Prose>{""}<strong>{"Learn a solution versus learn its derivative."}</strong>{" A physics-informed neural network can parameterize a solution "}<InlineMath>{"u_\\theta(t,x)"}</InlineMath>{" and penalize its equation residual at sampled points. A Neural ODE parameterizes a vector field and numerically integrates it from initial conditions. They can be combined, but one is not simply another name for the other. The "}<a href={"/learn/path/full-curriculum/physics-informed-neural-networks-pinns?module=frontier-research"}>{"PINN lesson"}</a>{" owns residual losses and boundary-condition design."}</Prose>

<Prose>{""}<strong>{"Events and interventions."}</strong>{" An event may stop integration when a state reaches a threshold. A sensor update or physical impact may change the state discontinuously. These require event/root handling or explicit jump rules; an ordinary smooth solve will not invent them. Event-time gradients can become delicate near tangencies or changes in which event occurs first. A threshold crossing also needs a detection policy: a solver that checks signs only at step boundaries can miss multiple crossings inside one step."}</Prose>

<Prose>{""}<strong>{"Controlled and stochastic dynamics."}</strong>{" If a stream continuously drives the state, a controlled differential equation makes that input path part of the dynamics. If uncertainty is modeled by a stochastic differential equation, the driving noise and stochastic calculus alter the solver and gradient problem. A deterministic ODE with newly sampled dropout on every field evaluation is not an automatically valid substitute for either construction."}</Prose>

<Prose>{"These applications are interesting because they change the model's assumptions, not simply because they attach a new industry name to the same diagram. The author talk in the resources offers another route into constrained dynamics; advanced applications require their own data, assumptions and evaluation."}</Prose>

<OdeStructuredFigure/>

<H2>{"11. Diagnose the right layer of a problem"}</H2>

<Prose>{"A useful implementation begins with a small, deterministic field and a checkable solver. Add complexity after establishing which part of the system needs it."}</Prose>

<NeuralTable caption={"11. Diagnose the right layer of a problem"} headers={[<>{"Observation"}</>,<>{"A question to investigate"}</>,<>{"Useful controlled check"}</>]} rows={[[<>{"Endpoint changes under finer resolution"}</>,<>{"Was training exploiting a coarse numerical map?"}</>,<>{"Keep weights/input fixed; compare methods and step sizes"}</>],[<>{"Gradient differs from a reference"}</>,<>{"Are both differentiating the same objective and discretization?"}</>,<>{"Compare autograd with finite differences of the exact executed loss"}</>],[<>{"Backward reconstruction fails"}</>,<>{"Does reversing dynamics amplify endpoint error?"}</>,<>{"Test an analytic system; compare stored/checkpointed states"}</>],[<>{"Adaptive solver takes many steps"}</>,<>{"Is the cause scaling, tolerances, transients, smoothness or stiffness?"}</>,<>{"Normalize coordinates; inspect rejection/error traces; compare an appropriate implicit method"}</>],[<>{"More input examples change a batch member's result"}</>,<>{"Are states coupled by BatchNorm or a shared adaptive error controller?"}</>,<>{"Evaluate examples individually and inspect the batch error norm"}</>],[<>{"Trajectory seems to cross itself"}</>,<>{"Is it a projection or a visit at another time?"}</>,<>{"Inspect full state with time labels"}</>],[<>{"Forecast looks too accurate"}</>,<>{"Did the encoder see observations beyond the forecast origin?"}</>,<>{"Rebuild the information-availability mask before fitting"}</>],[<>{"Density behaves implausibly"}</>,<>{"Is the divergence sign/direction right, and is trace noise large?"}</>,<>{"Use an exact linear flow with known determinant first"}</>]]} />

<Prose>{"Batching deserves particular care. A shared adaptive controller can use a norm over a whole batch. Many easy coordinates may dilute one difficult coordinate under an RMS norm; a maximum-based norm behaves differently. Even when the mathematical fields are independent, step decisions can depend on which examples are batched together."}</Prose>

<Prose>{"For neural fields, smooth activations often make high-order integration easier to use. ReLU is Lipschitz, so it does not by itself destroy uniqueness, but its kinks can affect differentiability and numerical-order behavior. Repeated random dropout masks or mutable normalization statistics can change the field between evaluations; decide what fixed function the solver is supposed to integrate."}</Prose>

<Prose>{"Performance depends on field cost, forward and backward evaluations, state size, saved outputs, solver/controller overhead and device utilization. Record those factors before making a timing claim. More accurate integration need not improve statistical prediction if model or data error dominates."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"12. Practice: transfer the mechanism"}</H2>

<Prose>{"Work out a prediction before opening a hint. These inputs differ from the worked calculations. Numerical answers refer to the stated finite computation or exact equation, as named."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. A different decay"}</H3>

<Prose>{"Start at "}<InlineMath>{"z_0=2"}</InlineMath>{", with "}<InlineMath>{"z'=-3z"}</InlineMath>{". Take two Euler steps of length 0.1. Compare the result with the exact state at time 0.2. Explain the sign of the error."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Each Euler step multiplies the state by "}<InlineMath>{"1-3(0.1)"}</InlineMath>{". Compare that factor with the exact exponential factor."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"Euler gives "}<InlineMath>{"2(0.7)^2=0.98"}</InlineMath>{". The exact state is "}<InlineMath>{"2e^{-0.6}=1.097623272"}</InlineMath>{", so the error, numerical minus exact, is −0.117623272. The start-of-step tangent keeps decreasing at its initial slope while the exact positive state's decay slows. For this step size, Euler therefore decreases too far."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Resolution without new weights"}</H3>

<Prose>{"A four-dimensional field takes the state plus time through a 16-unit tanh hidden layer and a four-output affine layer. A three-class affine head reads the final state. Count the parameters. Does increasing classical RK4 steps from four to twelve triple them?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"An affine map from "}<InlineMath>{"a"}</InlineMath>{" inputs to "}<InlineMath>{"b"}</InlineMath>{" outputs has "}<InlineMath>{"ab+b"}</InlineMath>{" parameters. Count field and readout separately from evaluations."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"The field has "}<InlineMath>{"5(16)+16+16(4)+4=164"}</InlineMath>{" parameters. The readout has "}<InlineMath>{"4(3)+3=15"}</InlineMath>{", totaling 179. Four versus twelve RK4 steps require 16 versus 48 field evaluations, but both use the same 179 parameters. The forward computation grows; the parameter set does not."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. A local estimate is not a final guarantee"}</H3>

<Prose>{"An embedded step has current state 0.5, Heun state 0.49 and Euler state 0.48. Let relative tolerance be 0.02 and absolute tolerance 0.001. Is it accepted by the lesson's scalar controller? Does acceptance prove endpoint error below 0.001?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Use the maximum magnitude of current and Heun states in the denominator."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"The scale is "}<InlineMath>{"0.001+0.02(0.5)=0.011"}</InlineMath>{". The normalized difference is "}<InlineMath>{"0.01/0.011=0.90909"}</InlineMath>{", so this step is accepted. That is a comparison of two local approximations. Accumulation, estimator accuracy and conditioning separate it from a guarantee about final global error."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Differentiate one Euler step"}</H3>

<Prose>{"Use "}<InlineMath>{"z'=\\theta z"}</InlineMath>{", "}<InlineMath>{"z_0=1.5"}</InlineMath>{", one step of length 0.2, target 1, and half squared error. At "}<InlineMath>{"\\theta=-1"}</InlineMath>{", calculate the prediction and parameter gradient of this one-step program."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Differentiate "}<InlineMath>{"z_1=z_0(1+h\\theta)"}</InlineMath>{", then apply the loss derivative."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"The prediction is 1.2; "}<InlineMath>{"dz_1/d\\theta=0.3"}</InlineMath>{". Therefore "}<InlineMath>{"dL/d\\theta=(1.2-1)(0.3)=0.06"}</InlineMath>{". A small gradient-descent step decreases "}<InlineMath>{"\\theta"}</InlineMath>{", reducing the numerical prediction toward the target. The exact continuous prediction "}<InlineMath>{"1.5e^{-0.2}"}</InlineMath>{" defines a different loss; substituting it halfway through this calculation would mix objectives."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Separate invertibility from a readout"}</H3>

<Prose>{"Can a unique one-dimensional continuous flow followed by one linear threshold label inputs −2 and 3 as class one and input 0 as class zero? Construct an augmented solution and name a numerical failure that could confuse this test."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Preserve ordering in one dimension, then use a second coordinate proportional to the square of the first."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"The one-dimensional flow keeps 0 between the other points, so a single threshold cannot select both outer points alone. Starting at "}<InlineMath>{"(x,0)"}</InlineMath>{", the field "}<InlineMath>{"x'=0,y'=x^2"}</InlineMath>{" reaches "}<InlineMath>{"(-2,4),(0,0),(3,9)"}</InlineMath>{" at time one; "}<InlineMath>{"y>1"}</InlineMath>{" works. A coarse explicit solver can reverse or collapse order even where the exact flow does not. A nonlinear readout also changes the representational question and must not be silently substituted."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Observation deletion"}</H3>

<Prose>{"Start at zero. Let "}<InlineMath>{"h'=-0.5h"}</InlineMath>{", and update by "}<InlineMath>{"h^+=0.7h^-+0.3x"}</InlineMath>{" at times 0.4 and 1.0 with observations 2 and 0. Query at 1.2. Compare with omitting the second observation."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The second observed value is zero, but its update still scales existing state by 0.7."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"After the first observation, state is 0.6. With the zero observation, the final state is "}<InlineMath>{"0.6e^{-0.3}(0.7)e^{-0.1}=0.42e^{-0.4}=0.281534419"}</InlineMath>{". Omitting it gives "}<InlineMath>{"0.6e^{-0.4}=0.402192028"}</InlineMath>{". Zero is evidence processed through an update; deletion removes that operation."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Expansion and density"}</H3>

<Prose>{"A two-dimensional linear field has matrix "}<InlineMath>{"\\begin{bmatrix}0.4&3\\\\0&-0.1\\end{bmatrix}"}</InlineMath>{". Over 1.5 units of time, what are the log-density change and volume multiplier? Does a large off-diagonal entry invalidate the trace calculation?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The trace is the diagonal sum. Density and volume change inversely along the flow."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"The trace is 0.3. Log-density changes by −0.45, and volume multiplies by "}<InlineMath>{"e^{0.45}=1.568312185"}</InlineMath>{". Density multiplies by "}<InlineMath>{"e^{-0.45}=0.637628152"}</InlineMath>{". The off-diagonal entry alters shape and motion; the determinant identity still uses the trace for this constant linear field."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. What can a velocity regressor know?"}</H3>

<Prose>{"Two equally likely training pairs are "}<InlineMath>{"−1\\to3"}</InlineMath>{" and "}<InlineMath>{"3\\to−1"}</InlineMath>{". At time 0.5, what location and target velocities do they present? What prediction minimizes mean squared loss at that location? Why should you not draw both pair paths as solutions of a single unique field?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"A model receiving only position and time cannot distinguish the pair identity."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"Both present location one, with velocities +4 and −4. Their conditional mean is zero; the mean squared loss there is 16. Always choosing +4 would give mean loss 32. Under uniqueness, one field cannot choose two velocities at the identical state and time. Pair-conditioned training paths and trajectories of the learned marginal field are different objects."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Design an honest comparison"}</H3>

<Prose>{"A report trains a neural flow and a residual classifier, tries new tolerances after reading test accuracy, and publishes only the best seed. It calls a two-coordinate projection proof that trajectories never cross. Propose a corrected protocol."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Separate data roles, statistical variation, numerical sensitivity and the full-state mathematical claim."}</Prose>

</details>

<details>

<summary>Solution and reasoning</summary>

<Prose>{"Declare splits, preprocessing, architectures, budgets, seeds and model-selection rules before assessment. Fit transformations only on training data and select with validation; retain all declared seeds and report actual parameter/evaluation costs. Perform solver-sensitivity checks on a specified diagnostic or validation set with fixed weights, then assess under a fixed protocol. A projection is a visualization of selected coordinates, not a proof about full-state uniqueness. State theorem assumptions and use analytic counterexamples or full-state checks for the claim actually being made."}</Prose>

</details>

<Prose>{"You are ready to continue when you can distinguish a field from a solved trajectory, compute and compare numerical updates, name the objective behind a gradient, explain augmentation with its readout assumption, and keep observation availability separate from evaluation time. You should also be able to read an experiment's counts and numerical limits without needing every method to win."}</Prose></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"13. Continue and learn another way"}</H2>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals"}>{"Hybrid SSM–Transformer Architectures (Jamba)"}</a>{". It returns to discrete token sequences and combines different memory operations. Our current lesson adds another distinction to that comparison: a continuous dynamical model still needs a concrete numerical execution method."}</Prose>

<Prose>{"Useful routes into the subject:"}</Prose>

<ul><li>{""}<a href={"https://arxiv.org/html/1806.07366v5"}>{"Neural Ordinary Differential Equations"}</a>{", Chen and colleagues. Primary introduction to learned dynamics, adjoints and applications. Read the mathematical model first; revisit memory claims with the numerical qualifications in this lesson."}</li><li>{""}<a href={"https://arxiv.org/html/2005.13420v2"}>{"Discretize-Optimize vs. Optimize-Discretize"}</a>{", Onken and Ruthotto. An intermediate numerical perspective on gradients, rediscretization and extrapolation. Its experiments are specific evidence, not universal timing ratios."}</li><li>{""}<a href={"https://arxiv.org/html/1904.01681v1"}>{"Augmented Neural ODEs"}</a>{", Dupont and colleagues. Read the one-dimensional construction and appendix uniqueness argument after section 5; the linear readout and exact-flow assumptions are essential."}</li><li>{""}<a href={"https://github.com/rtqichen/torchdiffeq"}>{"torchdiffeq"}</a>{" and its "}<a href={"https://github.com/rtqichen/torchdiffeq/blob/master/FAQ.md"}>{"FAQ"}</a>{". Practical PyTorch API and solver/adjoint guidance. Repository documentation was reviewed and the downloadable local solver bridge was executed with torchdiffeq 0.2.5; the repository’s separate official example suite was not executed. The downloadable local study is self-contained with PyTorch."}</li><li>{""}<a href={"https://docs.kidger.site/diffrax/api/adjoints/"}>{"Diffrax adjoints"}</a>{". A JAX-oriented explanation of checkpointed differentiation and continuous backsolves. Read after the scalar gradient experiment rather than starting with every API class."}</li><li>{""}<a href={"https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf"}>{"Latent ODEs for Irregularly-Sampled Time Series"}</a>{". Primary methods for encoding observations and separating interpolation from extrapolation. Useful after the observation-jump investigation."}</li><li>{""}<a href={"https://arxiv.org/html/1810.01367v2"}>{"FFJORD"}</a>{" and "}<a href={"https://arxiv.org/html/2210.02747v2"}>{"Flow Matching"}</a>{". Two different training routes for continuous generative models. Their objectives deserve separate study, even though both generate with learned dynamics."}</li><li>{""}<a href={"https://anucvml.github.io/ddn-cvprw2020/talk2.html"}>{"Ricky T. Q. Chen's CVPR 2020 workshop talk"}</a>{". An alternate video route into constraints, physical dynamics and probabilistic models. The organizer's title, speaker and abstract were verified; the embedded recording was not watched, and no timestamp is claimed. Use the paper and checked calculations for technical details."}</li><li>{""}<a href={"/learn-assets/neural-ode-continuous-depth-models/ode_calculations.py"}>{"Author calculations"}</a>{", "}<a href={"/learn-assets/neural-ode-continuous-depth-models/calculated-inputs.json"}>{"calculated inputs and outcomes"}</a>{", "}<a href={"/learn-assets/neural-ode-continuous-depth-models/neural_ode_study.py"}>{"full training program"}</a>{", "}<a href={"/learn-assets/neural-ode-continuous-depth-models/data-provenance.md"}>{"data and provenance"}</a>{". These reproduce this lesson's small examples and fitted study offline. No external large-model checkpoint is required."}</li></ul>
<OdeProgram file="ode_calculations.py"/></section>
</div>};
