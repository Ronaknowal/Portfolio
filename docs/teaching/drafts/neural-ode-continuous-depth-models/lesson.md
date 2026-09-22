# Neural ODEs: learn a rule for change, then follow it

**Explore as you read.** Edit vector-field parameters, initial state, step/tolerance, differentiation route, augmentation and supported real measurements. Show field arrows, accepted/rejected solver stages, current trajectory/error, derivative target and class output immediately or through bounded process steps. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose a solver/tolerance or representation from error, work and topology, distinguishing numerical approximation from the continuous equation.


Imagine transforming a measurement by moving a point through a landscape of arrows. At its current location, an arrow tells the point which direction to move and how quickly. After a small move, the point encounters a new arrow. A whole journey emerges from repeatedly following this local rule.

A **neural ordinary differential equation**, or Neural ODE, uses a neural network to produce those arrows. A numerical solver follows them. The resulting journey can transform features for classification, describe a hidden state between irregular observations, or move a probability distribution into another shape.

This division of work is the idea to keep: **the network learns the rate of change; the solver approximates the accumulated change.** Neither piece can be understood fully by looking at the other alone.

[Figure O01 — A measured input enters a vector field, follows a trajectory and reaches a readout. Label the field's output “change per unit depth,” and the readout's output “class scores.” Put solver evaluation points on the curve without drawing a new set of learned weights at each point.]

**First pass.** Follow sections 1–5 to calculate a trajectory and understand how a model learns it. Section 6 trains small classifiers on real flower measurements; section 7 connects continuous state to irregular observations. Then do the core practice. The deeper branches on density models, flow matching and structured dynamics can wait until the main distinction between a field and its solution feels natural.

You need vectors, derivatives, an ordinary feedforward network and backpropagation. The [ODE and linear-systems lesson](/learn/path/full-curriculum/ordinary-differential-equations-linear-systems?module=math-foundations) provides a fuller mathematical foundation. We introduce the numerical ideas needed here as we use them.

## 1. A derivative tells you how to move

Let \(z(t)\) be a state: perhaps two coordinates on a plane, or a vector of hidden features. The equation

\[
\frac{dz(t)}{dt}=f_\theta(t,z(t)),\qquad z(0)=z_0
\]

specifies its rate of change and starting point. The symbol \(\theta\) collects the network's learned weights. The right side returns a vector with the **same shape as the state**. It is a velocity, not the next state itself.

The variable \(t\) needs an interpretation. For measurements recorded over hours, it may be physical time. For a classifier transforming one image or flower measurement, it is a coordinate along **representation depth**. The flower does not grow while the solver runs.

Suppose the scalar rule is \(z'=-2z\), initially \(z_0=1\). The initial derivative is −2. Over a small interval of length 0.1, a first estimate of the change is \(0.1(-2)=-0.2\), so the next state is approximately 0.8. Now the derivative is −1.6. The state keeps decreasing, but the decrease slows as the state approaches zero.

The exact solution is \(z(t)=e^{-2t}\). At \(t=0.1\), it is approximately 0.81873. Our estimate 0.8 was close, but it used the initial velocity throughout an interval during which velocity changed. That is the first numerical error to understand.

[Figure O02 — Exponential decay with one tangent-based step. Show the exact endpoint and approximate endpoint at the same time, joined by a vertical error segment. The tangent is an approximation over the interval, not a second physical trajectory.]

An **initial-value problem** asks for a trajectory given an initial state and rule. Learning asks a different question: which rule produces useful trajectories for the examples we have? In a classifier, a readout converts the final state into scores:

\[
z_T=\operatorname{Solve}(f_\theta,z_0,0,T),\qquad
\text{logits}=Wz_T+b.
\]

The loss compares those scores with labels. Training changes the field and readout; an ordinary inference solve holds their weights fixed.

### Why the connection to residual networks matters

A residual update has the form

\[
z_{k+1}=z_k+h f_\theta(t_k,z_k).
\]

This is also the forward Euler numerical method. It adds a scaled local change to the current state. If we refine the time grid while evaluating a consistently defined field, the discrete updates can approximate a continuous trajectory under the usual existence and numerical-convergence conditions.

An arbitrary ResNet with unrelated weights at each layer does not automatically become a particular ODE just because it has many layers. We must define how its layer weights correspond to a function of time and what remains fixed as the grid changes.

This gives Neural ODEs a useful form of parameter sharing. Evaluating one field 16 times need not introduce 16 sets of weights. It still consumes computation. “Continuous depth” describes the mathematical model; the machine performs finitely many operations. The [original Neural ODE paper](https://arxiv.org/html/1806.07366v5) develops this connection and several applications.

## 2. The solver is part of the computation

A solver chooses where to evaluate the field and how to combine those evaluations. It does not learn the field's weights during a forward pass.

### Euler: one arrow per step

With \(h=T/N\), Euler uses one field evaluation for each of \(N\) steps. Halving the step size usually reduces its global error by approximately a factor of two in the smooth, sufficiently resolved regime. That statement concerns a convergence regime, not every possible coarse step.

For the two-dimensional rotation

\[
f(x,y)=(-y,x),\qquad z(0)=(1,0),
\]

the exact path is \((\cos t,\sin t)\). The point turns at one radian per unit time and keeps radius one. Euler does something revealing:

\[
\begin{bmatrix}x_{k+1}\\y_{k+1}\end{bmatrix}
=
\begin{bmatrix}1&-h\\h&1\end{bmatrix}
\begin{bmatrix}x_k\\y_k\end{bmatrix}.
\]

Squaring and adding gives \(x_{k+1}^2+y_{k+1}^2=(1+h^2)(x_k^2+y_k^2)\). Euler's point spirals outward even though the exact dynamics conserve radius. We can identify the artifact algebraically rather than blaming a learned model.

### Classical RK4: sample the turn inside a step

Classical fourth-order Runge–Kutta evaluates four velocities:

\[
\begin{aligned}
k_1&=f(t,z),\\
k_2&=f(t+h/2,z+hk_1/2),\\
k_3&=f(t+h/2,z+hk_2/2),\\
k_4&=f(t+h,z+hk_3),\\
z_{\rm next}&=z+\frac h6(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

The middle evaluations probe where the state may be halfway through the interval. Their weighted combination cancels lower-order errors. For a sufficiently smooth, stable problem, the global error is \(O(h^4)\): halving \(h\) can reduce it by about 16, before rounding or other errors dominate. This is polynomial convergence, not exponential convergence.

[Figure O03 — An expanded RK4 step with four numbered velocity arrows, tentative stage locations and a separate weighted final move. Stage locations are calculations, not four successive states along the final accepted trajectory.]

Our executed rotation calculation integrates to \(T=1\) in float64:

| Method | Steps | Field evaluations | Endpoint error, Euclidean norm |
| --- | ---: | ---: | ---: |
| Euler | 4 | 4 | 0.130661 |
| Euler | 16 | 16 | 0.0317081 |
| Classical RK4 | 4 | 16 | 0.0000325318 |
| Classical RK4 | 16 | 64 | 0.000000127152 |

At four RK4 steps, the radius is 0.99999327, not exactly one. A small error is still an error; RK4 is not generally a method that exactly conserves energy.

**Investigation O-I1 — Follow the field.** Start from a different point, \((0.6,0.8)\), and integrate to 1.2. Inspect the direction of the radius error before comparing Euler and RK4. Then edit the field's matrix or starting point. Keep an exact matrix-exponential reference beside the numerical path. Use the residual panel to see small differences without distorting the path itself.

### A readable differentiable implementation

The downloadable [training program](neural_ode_study.py) contains the field, integrator, models, split and complete fitting loop. This is its core classical RK4 update, expressed with tensor operations so automatic differentiation can follow every stage:

~~~python
def rk4_step(field, time, state, step_size):
    first = field(time, state)
    second = field(time + step_size/2, state + step_size*first/2)
    third = field(time + step_size/2, state + step_size*second/2)
    fourth = field(time + step_size, state + step_size*third)
    return state + step_size*(first + 2*second + 2*third + fourth)/6
~~~

To build the trajectory, repeat this update at times \(0,h,\ldots,T-h\). The complete program retains all states and uses that same expression inline. Its field concatenates the fixed depth coordinate to the state, applies a 16-unit tanh layer, and returns a derivative vector. Time is not a learnable parameter in this example; differentiating learned event times would require a different treatment.

In a practical library, method names alone are insufficient for exact reproduction. The current [torchdiffeq README](https://github.com/rtqichen/torchdiffeq) specifies a 3/8-rule implementation for its fixed-step RK4; our displayed program uses classical RK4. Both are fourth-order methods, but their intermediate stages differ. Its callable uses the argument order (time, state), and fixed-step resolution is configured separately from the requested output times.

## 3. Adapt the steps to estimated error

Taking tiny steps everywhere can waste work. An adaptive method estimates the error in a proposed step and decides whether to accept it.

A simple teaching method combines Euler with **Heun's method**, which averages the velocity at the beginning and the Euler-predicted endpoint:

\[
z_E=z+h f(t,z),\qquad
z_H=z+\frac h2\{f(t,z)+f(t+h,z_E)\}.
\]

The difference \(z_H-z_E\) is an error indicator. In our demonstration, coordinate \(i\) has scale

\[
s_i=\mathrm{atol}+\mathrm{rtol}\max(|z_i|,|z_{H,i}|),
\quad
r=\sqrt{\frac1d\sum_i[(z_{H,i}-z_{E,i})/s_i]^2}.
\]

Accept when \(r\leq1\); otherwise retry from the same state with a smaller step. The demonstration proposes the next step using a safety factor \(0.9r^{-1/2}\), clamped between 0.1 and 5. The exponent reflects this embedded difference's order. It is not a universal controller for all solvers.

[Figure O04 — A timeline with accepted intervals as solid segments and rejected attempts as hollow overlays. A rejected attempt spends field evaluations but does not advance the state. Show both counts.]

For the synthetic two-scale system \(z'=\operatorname{diag}(-1,-100)z\), starting at \((1,1)\) and ending at 0.4, our executed method uses 58 accepted and four rejected steps with relative tolerance 0.01 and absolute tolerance 0.0001. Tightening both tolerances tenfold produces 144 accepted and five rejected steps. Every attempt evaluates the field twice: 124 versus 298 evaluations.

These are measured counts from this specified teaching solver, not a promised cost for Neural ODEs. Production methods such as Dormand–Prince use different formulas, interpolation and reuse strategies.

**Investigation O-I2 — Spend an error budget.** Use fresh rates \((-2,-50)\), initial state \((1,0.3)\), endpoint 0.4 and the same controller. Predict what tightening tolerance will change. Inspect the first rejected step, alter one rate or tolerance, and compare endpoint error with the number of evaluations. A tolerance is a control on local estimates; it is not a guarantee that global error or a downstream class probability meets the same numerical threshold.

### Stiffness can hide beside a smooth trajectory

Consider \(y'=-\kappa(y-\cos t)-\sin t\), with \(y(0)=1\). The exact solution is \(\cos t\) for every positive \(\kappa\). It looks equally smooth when \(\kappa=5\) and when \(\kappa=1000\).

A perturbation away from that solution obeys \(e'=-\kappa e\). Large \(\kappa\) creates a rapidly decaying mode that can restrict an explicit solver's stable step size even when the desired solution changes slowly. That separation between fast stability constraints and slower behavior is a characteristic stiffness problem.

Our SciPy run to time one, with relative tolerance \(10^{-6}\) and absolute tolerance \(10^{-9}\), illustrates it:

| \(\kappa\) | RK45 field evaluations | Radau field evaluations | Radau matrix factorizations |
| --- | ---: | ---: | ---: |
| 5 | 92 | 99 | 8 |
| 1000 | 2162 | 84 | 16 |

Both solved the stated problem successfully. A field evaluation is not a unit of equal total cost across these methods: implicit Radau also solves algebraic systems. These numbers do not establish a wall-clock speedup. The [SciPy solver documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) explains explicit/implicit choices, Jacobians and event handling.

High evaluation counts deserve diagnosis. Tight tolerances, poor coordinate scaling, nonsmooth fields and difficult transients can also increase work. “More evaluations” does not by itself prove stiffness or that an input is semantically harder.

## 4. Learn through a trajectory

We now have a computation whose output depends on learned weights. A loss can differentiate that dependence in two main ways.

### Differentiate the numerical program

For fixed Euler steps, write \(F_k=z_k+h f_\theta(t_k,z_k)\). Ordinary backpropagation applies the chain rule through each \(F_k\). With a final-state loss \(L(z_N)\), let \(a_k\) be the column vector \(\partial L/\partial z_k\). Then

\[
a_k=(I+hJ_zf_k)^\top a_{k+1},
\qquad
\nabla_\theta L=\sum_k h(J_\theta f_k)^\top a_{k+1}.
\]

Here \(J_z f\) is the matrix of field derivatives with respect to the state, and \(J_\theta f\) collects derivatives with respect to weights. Their transposes carry a loss sensitivity backward from outputs to inputs. If the readout or initial state depends on parameters, add those paths too.

RK4 has more intermediate operations, but automatic differentiation handles the same principle. It computes a gradient of the **implemented finite computation**, subject to floating-point effects and any nondifferentiable branches. This is often called **discretize then optimize**. Keeping every stage's graph consumes memory; checkpointing stores selected states and recomputes parts of the forward calculation during backward propagation.

### Derive a continuous adjoint

Alternatively, derive sensitivity equations for the continuous model first. For a terminal loss with no direct parameter dependence and parameter-independent \(z_0\),

\[
\frac{da}{dt}=-J_z f_\theta(t,z(t))^\top a(t),
\qquad a(T)=\nabla_{z(T)}L,
\]

\[
\nabla_\theta L=\int_0^T
J_\theta f_\theta(t,z(t))^\top a(t)\,dt.
\]

The adjoint says how much the final loss cares about a small change in the state at each time. A backward solve can accumulate this integral without retaining the entire forward computation graph.

**Keep the integration direction explicit.** A parameter accumulator \(g\) initialized at \(g(T)=0\) and integrated from \(T\) down to zero uses \(g'=-J_\theta f^\top a\). The negative right side combined with backward limits yields the positive forward-time integral above.

[Figure O05 — Two lanes. The discrete lane reverses actual Euler/RK4 operations. The continuous lane solves state, adjoint and parameter-accumulator equations backward. Both lanes name the function whose gradient is being approximated.]

For losses at several observation times, the adjoint receives a jump from each observation's loss gradient. A running integral cost also contributes a term between observations. These extensions matter in time-series training; the terminal-only formula is not the full recipe for every objective.

<details>
<summary>Deeper: why the adjoint equation cancels state sensitivity</summary>

Let \(S(t)=\partial z(t)/\partial\theta\). Differentiating the state equation gives \(S'=J_zf\,S+J_\theta f\), with \(S(0)=0\) under our assumptions. The product rule gives

\[
\frac{d}{dt}(a^\top S)=a'^\top S+a^\top S'
=-a^\top J_zf S+a^\top(J_zf S+J_\theta f)
=a^\top J_\theta f.
\]

Integrate from zero to \(T\). The left endpoint vanishes; the final term is \(\nabla_{z(T)}L^\top S(T)\), precisely the chain rule for the parameter gradient. This also explains the adjoint's negative sign. Forward sensitivities store a state-by-parameter matrix; the adjoint is attractive when there are many parameters and a scalar objective.

</details>

### The two gradients need not be equal at finite resolution

Use \(z'=\theta z\), \(z(0)=1.2\), \(T=1.3\), target 0.4, and loss \(\frac12(z(T)-0.4)^2\), evaluated at \(\theta=-0.7\).

The exact state is \(1.2e^{1.3\theta}\). Differentiating it gives

\[
\frac{dL}{d\theta}=(z(T)-0.4)\,1.3z(T)=0.05213709.
\]

Four Euler steps instead produce \(z_4=1.2(1+0.325\theta)^4=0.42734163\). Their autograd gradient is 0.01966276. Central differences of that same four-step loss agree with autograd. They do not agree with the continuous answer because they differentiate a different forward function.

Four classical RK4 steps give 0.05213852, much closer here. A backward RK4 solve of the continuous adjoint initialized at the numerical forward endpoint gives 0.05214299. It introduces its own approximation; it is not exactly the RK4 program's reverse-mode derivative.

[Figure O06 — Gradient values on a common axis, with a zoomed residual panel for the RK4 difference. Label exact continuous, discrete autograd, discrete finite difference and numerical continuous backsolve separately.]

This is why “the gradient passed a check” needs a named reference. [Onken and Ruthotto](https://arxiv.org/html/2005.13420v2) study the distinction and solver changes in continuous models. Our scalar example is independently calculated rather than reproduced from their experiments.

**Investigation O-I3 — Which loss are you differentiating?** Start with fresh \(\theta=0.3\), \(z_0=0.8\), target 1, \(T=0.7\), three steps. Inspect the gradient sign, compare the four methods, then increase steps. Change the target and check that feedback recomputes from the new loss. Closer numerical agreement is evidence about this fixture, not a universal equivalence theorem.

### Why saving no forward trajectory can be fragile

The decay \(z'=-20z\), \(z(0)=1\), reaches \(e^{-20}\approx2.0612\times10^{-9}\) at time one. Reverse the exact equation from an endpoint perturbed by \(10^{-8}\). The recovered initial value is approximately 5.85165 instead of one: reverse evolution multiplies the perturbation by \(e^{20}\).

The forward process is stable but its inverse reconstruction is ill-conditioned. A more accurate solver helps numerical error; it cannot remove this mathematical amplification of an already present endpoint error. Checkpoints can reduce the length over which state must be reconstructed.

“Constant memory” for a backsolve usually concerns dependence on the number of internal forward steps. Parameters, parameter gradients, requested output states, batches and local network activations still occupy memory. Current [Diffrax adjoint documentation](https://docs.kidger.site/diffrax/api/adjoints/) describes checkpointed differentiation of the numerical solution and distinguishes it from approximate continuous backsolves. The correct choice depends on the objective, solver, accuracy needs and available memory.

### Control an actual solver API with the same field

The mechanism owner is `integrate` in [neural_ode_study.py](neural_ode_study.py): it builds Euler and classical RK4 updates from tensor operations, keeping the computation graph for direct differentiation. `VectorField` and `DepthClassifier` later turn that mechanism into a learned classifier. Here we isolate the solver/library bridge on z′=a z, where both the trajectory and its derivatives are known. This avoids mistaking a similar classification score for a correct solver interface.

Keep [solver_library_bridge.py](solver_library_bridge.py) beside that program. With the earlier PyTorch environment, install `torchdiffeq==0.2.5`, then run `python solver_library_bridge.py`. The target for these new authored examples is PyTorch 2.14.0, CPU float64. Package execution and final numeric output capture remain phase-two work; the analytic answers below are derived.

`odeint(field, initial, times)` returns a tensor with requested time as its first axis. Those times request output values; an adaptive solver may take many internal steps between them. `method='euler', options={'step_size': 1/8}` gives the same eight Euler steps as our scratch route. We compare both final states and both kinds of derivative: with respect to the initial state and the field parameter. In contrast, the package's `rk4` uses the 3/8 tableau, while our scratch program uses classical RK4. Their order alone does not justify step-by-step equality. The solver choices and adjoint interface are documented in [the author's repository](https://github.com/rtqichen/torchdiffeq).

For endpoint time one and loss L=½Σz(1)², exact calculus gives z(1)=exp(a)z₀, ∂L/∂z₀=exp(2a)z₀, and ∂L/∂a=exp(2a)Σz₀². The Euler solution instead uses factor (1+a/8)⁸. The adaptive direct and adjoint solves target the continuous answer within stated tolerances; we do **not** require them to equal the eight-step Euler answer. `odeint_adjoint` receives an `nn.Module` so its trainable field parameter can be found, and we set forward and backward tolerances explicitly.

```python
"""A matched Euler program, adaptive solve, and continuous-adjoint comparison.

Authoring targets: torch 2.14.0, torchdiffeq 0.2.5. CPU float64.
Run beside neural_ode_study.py; no fitting or dataset download occurs.
"""
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from neural_ode_study import integrate


class Decay(nn.Module):
    def __init__(self, rate=-.7):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float64))

    def forward(self, time, state):
        return self.rate * state


def value_and_derivatives(route):
    field = Decay()
    initial = torch.tensor([[1.5], [-.5]], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0., 1.], dtype=torch.float64)
    if route == "scratch_euler":
        result = integrate(field, initial, steps=8, method="euler")[-1]
    elif route == "library_euler":
        result = odeint(field, initial, times, method="euler",
                        options={"step_size": 1/8})[-1]
    elif route == "adaptive":
        result = odeint(field, initial, times, method="dopri5", rtol=1e-9, atol=1e-11)[-1]
    elif route == "adjoint":
        result = odeint_adjoint(field, initial, times, method="dopri5",
                                rtol=1e-9, atol=1e-11, adjoint_method="dopri5",
                                adjoint_rtol=1e-9, adjoint_atol=1e-11)[-1]
    else:
        raise ValueError(route)
    loss = result.square().sum()/2
    initial_gradient, rate_gradient = torch.autograd.grad(loss, (initial, field.rate))
    return result.detach(), initial_gradient.detach(), rate_gradient.detach()


def main():
    scratch, library = [value_and_derivatives(route)
                        for route in ("scratch_euler", "library_euler")]
    for left, right in zip(scratch, library):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    initial = torch.tensor([[1.5], [-.5]], dtype=torch.float64)
    factor = torch.exp(torch.tensor(-.7, dtype=torch.float64))
    exact = (initial*factor, initial*factor.square(), initial.square().sum()*factor.square())
    for route in ("adaptive", "adjoint"):
        actual = value_and_derivatives(route)
        for value, reference in zip(actual, exact):
            torch.testing.assert_close(value, reference, rtol=2e-7, atol=2e-9)
        print(route, "endpoint/initial-gradient/rate-gradient:", *actual)
    print("Euler endpoint, initial gradient, rate gradient:", *library)


if __name__ == "__main__":
    main()
```

The continuous adjoint changes how derivatives are computed; it does not erase reconstruction error, intermediate saved output states, parameter gradients or library overhead. Direct differentiation through K accepted numerical steps generally retains work proportional to K, while a continuous backsolve trades recomputation and its own numerical sensitivity for less retained internal-step history. Smoothness, solver stability and state dimension still matter. The current example has no events, stochastic layers or discontinuous observation jumps; inserting those changes the problem.

**Take control.** Change a to −4 and compare Euler at 1, 4, 8 and 32 steps with the same adaptive solve. Keep the parameter fixed: do not retrain it to compensate for a changed solver. Then tighten only the adjoint tolerances and inspect which derivative discrepancy changes.

<details><summary>Hint and reasoned solution</summary>

One Euler step multiplies the state by −3, which reverses sign and magnifies it even though the continuous system decays. Four steps multiply by zero at every step, incorrectly erasing the state. Eight give factor (½)⁸; as steps increase, (1−4/K)^K approaches exp(−4). For a finite K, direct differentiation correctly differentiates that discrete factor, not exp(a). Tightening backward tolerance can improve the continuous-adjoint approximation but cannot repair an intentionally inaccurate forward Euler trajectory. Report endpoint, initial-state gradient and parameter gradient separately; a small final loss alone misses this distinction.

</details>

## 5. What continuous flow can—and cannot—rearrange

Suppose a field is continuous in time, locally Lipschitz in state, and solutions exist uniquely over the interval of interest. Starting at a specified state and time determines one trajectory. Locally Lipschitz means that, near each state, small state changes cannot cause an unbounded ratio of change in the field; it is a useful condition for uniqueness.

Two distinct trajectories cannot meet at the **same state and same time** and then have different pasts within that common interval: uniqueness applied backward would contradict their different starts. Crossing lines in a two-dimensional projection of a higher-dimensional state are not a violation. Neither is visiting the same position at different times.

This has a concrete consequence in one dimension. If \(z_A(0)<z_B(0)\), their ordering cannot reverse along a well-defined unique flow. Reversing order would require a meeting.

Take inputs −1, 0, 1, where the outer two belong to class one and the middle belongs to class zero. A one-dimensional unique flow followed by a single linear threshold cannot solve this arrangement. A threshold separates a line into two intervals; preserving order leaves the middle between the outer points. A nonlinear readout could change the conclusion, so the readout assumption belongs in the explanation.

[Figure O07 — Three ordered tracks over continuous depth. Below them show a single movable threshold and the impossible desired outer/middle split. Do not draw arbitrary paths crossing and call them valid solutions.]

### Augment the state with room to move

Append a zero coordinate: \((x,0)\). Now use the two-dimensional rule \(x'=0,\ y'=x^2\). At depth \(T\), the state is \((x,Tx^2)\). At \(T=1\), the three inputs become \((-1,1),(0,0),(1,1)\). A linear readout testing \(y>0.5\) separates the classes.

No two full trajectories meet. The added coordinate gives the representation another direction in which to separate examples. It is a constructive illustration of augmentation, not a trained experiment or a proof that every augmented model learns easily. [Augmented Neural ODEs](https://arxiv.org/html/1904.01681v1) investigates this idea and its computational consequences.

**Investigation O-I4 — Lift the middle out.** Use fresh inputs \((-2,0,1)\), threshold 0.5 and depth 0.4. Inspect which points the readout selects. Increase depth to 0.7, then change the threshold. Show the complete two-dimensional trajectories and the one-dimensional readout scores together.

A coarse numerical method need not preserve an exact flow's properties. For \(z'=-2z\), one Euler step with \(h=1\) maps \(z\) to \(-z\), reversing order. The exact map multiplies by \(e^{-2}>0\) and preserves order. A discrete model that exploits a coarse solver's behavior may change substantially when evaluated with finer steps.

For finite sampled rings or point clouds, gaps can also let trajectories thread between examples. A theorem about an entire continuous region is stronger than a claim about a finite training set. Both distinctions prevent us from mistaking an attractive picture for a general impossibility result.

## 6. Train a small continuous-depth classifier on real measurements

Our data are the 150 Iris measurements distributed by UCI: sepal length, sepal width, petal length and petal width, all in centimeters, with three species labels. The [dataset](https://archive.ics.uci.edu/dataset/53/iris) is credited to R. A. Fisher and licensed CC BY 4.0. The retained [CSV](iris.csv) uses the corrected values documented in its [provenance](data-provenance.md).

This is a feature-transformation experiment. Each flower supplies an initial four-dimensional state after standardization. Solver time describes representation depth, not elapsed botanical time.

### Keep the comparison interpretable

Identical four-feature rows are grouped before assigning data roles; source rows 102 and 143 are duplicates. We keep all original rows in the download but score each unique vector once. A fixed stratified split assigns 90 unique examples to fitting, 30 to validation and 29 to assessment. The fitting examples alone determine the feature means and population standard deviations.

We compare four models:

| Model | Computation before a linear three-class head | Trainable parameters |
| --- | --- | ---: |
| Linear | No learned feature transformation | 15 |
| Residual | Four distinct 16-unit tanh fields, each used for one step | 671 |
| Neural ODE | One shared 16-unit tanh field; four classical RK4 steps | 179 |
| Augmented ODE | Append two zeros; one six-dimensional field; four RK4 steps | 251 |

The two ODE models perform 16 field evaluations per example. Their parameter counts stay fixed if we change solver resolution. The residual model has separate weights at its four blocks. These models have different capacities; this is not a parameter-matched benchmark.

The complete [study program](neural_ode_study.py) runs three declared seeds—13, 37 and 61—for each model. Every run uses 300 full-batch AdamW updates, learning rate 0.01 and weight decay 0.001, in float64 on CPU. Validation loss is measured after update one and every 25 updates. We select the lowest-validation-loss checkpoint within each run, then evaluate that checkpoint on assessment data. All 12 runs and their curves remain in [the result file](study-results.json); [selected weights](fitted-models.json) are retained.

### What actually happened

| Model | Assessment cross-entropy, seeds 13 / 37 / 61 | Correct out of 29, same order |
| --- | --- | --- |
| Linear | 0.22045 / 0.20028 / 0.21245 | 28 / 28 / 28 |
| Residual | 0.16763 / 0.10539 / 0.05133 | 28 / 28 / 29 |
| Neural ODE | 0.15239 / 0.10335 / 0.06212 | 28 / 28 / 29 |
| Augmented ODE | 0.04204 / 0.10549 / 0.03648 | 29 / 28 / 29 |

The uniform three-class predictor has cross-entropy \(\log3\approx1.09861\). A fixed class-zero choice scores 10/29. The linear model is already strong: a small reduction in classification errors cannot explain every difference in cross-entropy. Probability assigned to the true class matters even when the winning label stays unchanged.

All nonlinear runs selected update 50, while the linear runs selected update 300. Later training loss improvement therefore did not automatically mean better validation performance. There was no post-assessment search for a more impressive configuration.

[Figure O08 — All 12 validation curves and paired assessment points. Keep seeds visible, parameter counts adjacent and accuracy as counts with denominator 29. Do not draw confidence intervals from three seeds or use a leaderboard trophy.]

### Follow one actual flower through the learned field

For validation source row 64, the seed-37 Neural ODE assigns probabilities approximately \((0.01744,0.93374,0.04882)\) in the order setosa, versicolor, virginica. Increasing its petal length by 0.6 cm while holding the other measurements fixed changes them to \((0.00730,0.67440,0.31830)\). The winning label stays versicolor, but the evidence shifts substantially toward virginica.

These are full forward passes through the saved learned model, not a hand-drawn boundary. The input edit is a hypothetical measurement intervention; it does not establish a biological causal effect.

[Figure O09 — Raw centimeters → fixed standardization → four-dimensional state trace → three logits → probabilities. Display all four coordinates as small time-series panels; a selectable two-coordinate path is a projection, not the complete state.]

**Investigation O-I5 — Edit a measurement, keep the model.** Start on fresh validation row 70. Observe whether adding 0.6 cm to petal length will raise or lower the model's versicolor probability. Run the saved seed-37 model, then compare the augmented model and the opposite edit. Local behavior need not be monotone or shared across models. The lab calculates the edited input through all weights and solver stages and updates the linked output as the bounded computation completes.

For unchanged row 64, using 4, 16 and 64 RK4 steps gives versicolor probabilities 0.9337389, 0.9335432 and 0.9335421. Four Euler steps give 0.9442828. The weights are identical; the numerical realization changed. Agreement under refinement is useful evidence that the model is behaving like its continuous formulation locally. It does not prove accuracy everywhere or improve the training data automatically.

Independent NumPy inference reproduces the saved PyTorch models' logits and traces within \(8.9\times10^{-16}\) over the checked input/method/step cases. Input derivatives also agree with central differences. These author calculations support the manuscript and future interactive model.

### Run the complete study

Keep [iris.csv](iris.csv) beside the [study program](neural_ode_study.py). With Python, NumPy and PyTorch installed, run the program from that directory:

~~~text
python neural_ode_study.py
~~~

It writes the complete results and selected weights. Install SciPy as well to run the separate [calculation program](ode_calculations.py), which imports the study's model definitions and reuses the saved weights. That second command does not train again. The [provenance](data-provenance.md) records the versions used here.

<details>
<summary>Complete training program: data roles, models, selection and saved weights</summary>

~~~python
"""Declared real Iris study: learned depth is not botanical time."""
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch
from torch import nn

DIRECTORY = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def load_data():
    with (DIRECTORY / "iris.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    names = ["setosa", "versicolor", "virginica"]
    features = np.array([[float(row[key]) for key in columns] for row in records])
    labels = np.array([names.index(row["species"]) for row in records])
    groups = {}
    for index, row in enumerate(features):
        groups.setdefault(tuple(row), []).append(index)
    for members in groups.values():
        assert len(set(labels[members])) == 1
    kept = np.array(sorted(members[0] for members in groups.values()))
    roles = {key: [] for key in ("fit", "validation", "assessment")}
    generator = np.random.default_rng(926)
    for label in range(3):
        indices = generator.permutation(kept[labels[kept] == label])
        for key, subset in zip(roles, (indices[:30], indices[30:40], indices[40:])):
            roles[key].extend(subset.tolist())
    mean, scale = features[roles["fit"]].mean(0), features[roles["fit"]].std(0)
    metadata = dict(columns=columns, classes=names, mean=mean.tolist(), scale=scale.tolist(),
        roles={key: [index+1 for index in indices] for key, indices in roles.items()},
        duplicate_groups=[[i+1 for i in group] for group in groups.values() if len(group)>1],
        unique_count=len(kept))
    return torch.tensor((features-mean)/scale), torch.tensor(labels), roles, metadata


class VectorField(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.input = nn.Linear(dimension+1, 16)
        self.output = nn.Linear(16, dimension)

    def forward(self, time, state):
        clock = torch.full_like(state[:, :1], float(time))
        return self.output(torch.tanh(self.input(torch.cat((state, clock), -1))))


def integrate(field, initial, steps=4, method="rk4", endpoint=1.):
    state = initial
    trace = [state]
    step_size = endpoint/steps
    for step in range(steps):
        time = step*step_size
        first = field(time, state)
        if method == "euler":
            state = state + step_size*first
        elif method == "rk4":
            second = field(time+step_size/2, state+step_size*first/2)
            third = field(time+step_size/2, state+step_size*second/2)
            fourth = field(time+step_size, state+step_size*third)
            state = state + step_size*(first+2*second+2*third+fourth)/6
        else:
            raise ValueError(method)
        trace.append(state)
    return torch.stack(trace)


class DepthClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.dimension = 6 if kind == "augmented_ode" else 4
        if kind in ("neural_ode", "augmented_ode"):
            self.field = VectorField(self.dimension)
        if kind == "residual":
            self.blocks = nn.ModuleList([VectorField(4) for _ in range(4)])
        self.readout = nn.Linear(self.dimension, 3)

    def forward(self, inputs, steps=4, method="rk4", endpoint=1., return_trace=False):
        state = torch.cat((inputs, inputs.new_zeros((len(inputs), self.dimension-4))), -1)
        if self.kind in ("neural_ode", "augmented_ode"):
            trace = integrate(self.field, state, steps, method, endpoint)
            state = trace[-1]
        else:
            states = [state]
            if self.kind == "residual":
                for index, block in enumerate(self.blocks):
                    state = state + .25*block(index/4, state)
                    states.append(state)
            trace = torch.stack(states)
        logits = self.readout(state)
        return (logits, trace) if return_trace else logits


def evaluate(model, features, labels):
    with torch.no_grad():
        logits = model(features)
    return dict(cross_entropy=float(nn.functional.cross_entropy(logits, labels)),
                correct=int((logits.argmax(-1)==labels).sum()), count=len(labels))


def main():
    features, labels, roles, metadata = load_data()
    reports, snapshots = [], []
    for kind in ["linear", "residual", "neural_ode", "augmented_ode"]:
        for seed in [13, 37, 61]:
            torch.manual_seed(seed)
            model = DepthClassifier(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
            best_loss, selected, curves = float("inf"), None, []
            for step in range(1, 301):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(features[roles["fit"]]), labels[roles["fit"]])
                loss.backward()
                optimizer.step()
                if step==1 or step%25==0:
                    fit = evaluate(model, features[roles["fit"]], labels[roles["fit"]])
                    validation = evaluate(model, features[roles["validation"]], labels[roles["validation"]])
                    curves.append(dict(step=step, fit=fit, validation=validation))
                    if validation["cross_entropy"] < best_loss:
                        best_loss = validation["cross_entropy"]
                        selected = deepcopy(model.state_dict())
                        selected_step = step
            model.load_state_dict(selected)
            report = dict(kind=kind, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                selected_step=selected_step, curves=curves,
                fit=evaluate(model,features[roles["fit"]],labels[roles["fit"]]),
                validation=evaluate(model,features[roles["validation"]],labels[roles["validation"]]),
                assessment=evaluate(model,features[roles["assessment"]],labels[roles["assessment"]]))
            reports.append(report)
            snapshots.append(dict(kind=kind,seed=seed,selected_step=selected_step,
                state={key:value.tolist() for key,value in selected.items()}))
            print(kind,seed,selected_step,report["assessment"],flush=True)
    result=dict(data=metadata, updates=300, learning_rate=.01,weight_decay=.001,
        seeds=[13,37,61], dtype="float64", torch=torch.__version__, numpy=np.__version__,
        dataset_sha256=hashlib.sha256((DIRECTORY/"iris.csv").read_bytes()).hexdigest(),runs=reports)
    (DIRECTORY/"study-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (DIRECTORY/"fitted-models.json").write_text(json.dumps(snapshots,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
~~~

Read the model definition and integrator first, then follow one fitting update. The remaining loop repeats that update under the declared selection protocol. The saved weights let you change inputs or solver resolution without rerunning the fits.

</details>


## 7. Irregular observations: evolving state is not new evidence

Suppose a sensor reports at times 0.2, 0.9 and 1.3. A model should distinguish “nothing new was observed” from “the sensor observed zero.” A continuous hidden state gives us a way to evolve between those moments, but the ODE alone does not decide how to incorporate arriving measurements.

Three constructions answer different questions:

| Construction | What happens between observations? | How does a new observation affect state? |
| --- | --- | --- |
| Plain Neural ODE | Integrate from an initial state | It does not, unless an input/update mechanism is added |
| ODE-RNN | Integrate hidden state across each time gap | Apply a recurrent update at each observed time |
| Latent ODE | Evolve a sampled latent initial state | An encoder infers a distribution over that initial state from a chosen observation set |

A time-aware ordinary RNN is also a legitimate baseline: it can receive elapsed time as an input or use a decay rule. Continuity is one modeling choice, not a prerequisite for handling irregular timestamps.

### Work through an ODE-RNN-shaped calculation

Use a deliberately simple hidden state: between observations, \(h'=-0.5h\). At an observation with value \(x_i\), update \(h^+=0.7h^-+0.3x_i\). Start at zero.

For values \((1,-0.5,0.8)\) at times \((0.2,0.9,1.3)\):

- At 0.2, the update gives 0.3.
- Just before 0.9, decay gives \(0.3e^{-0.35}=0.211406\). The observation changes it to −0.002016.
- Just before 1.3, it is −0.001650. The new observation changes it to 0.238845.
- At a query time of 1.6, the state is \(0.238845e^{-0.15}=0.205576\).

[Figure O10 — Continuous decay arcs interrupted by visible observation jumps. Query markers sample the curve but cause no jump. A separate availability strip shows which observations existed before the query.]

Omitting the middle observation gives 0.310853 at 1.6. Observing zero instead gives 0.279568. The latter still applies the update's 0.7 retention. Missingness and zero are different computations.

**Investigation O-I6 — Separate observation from query.** Use fresh times \((0.1,0.7,1.4)\) and values \((1,-1,0.5)\). Inspect the effect of deleting the middle observation, then compare deletion with replacing its value by zero. Move the query to time one: the observation at 1.4 must become unavailable to a causal forecast. Adding extra query markers must leave the hidden dynamics unchanged.

The [Latent ODE paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf) combines continuous dynamics with observation-dependent inference. Its generative model samples \(z_0\), integrates \(z(t)\), and decodes observation distributions. An encoder approximates \(q(z_0\mid\{t_i,x_i\})\); training balances expected reconstruction log-likelihood against divergence from a prior.

Interpolation may condition on observations on both sides of a query. Forecasting must restrict the encoder to information available by the forecast origin. The paper also models observation timing with a Poisson process; a basic value-only model does not automatically handle informative missingness. The lesson's scalar calculation explains the mechanism without claiming to reproduce a clinical study or validate a medical system.

## 8. Deeper branch: move density as well as points

A normalizing flow transforms samples from a simple distribution into samples from a more useful one while tracking how density changes. Imagine stretching a small patch containing a fixed amount of probability. If its volume grows, its density must decrease.

For a differentiable vector field and a well-defined invertible flow,

\[
\frac{d}{dt}\log p_t(z(t))
=-\nabla\cdot f(t,z(t))
=-\operatorname{tr}J_zf(t,z(t)).
\]

The **divergence** is the sum of the field's same-coordinate derivatives. It measures instantaneous local volume expansion. The negative sign converts expansion into a density decrease.

For \(f(z)=Az\), the exact flow is \(z(T)=e^{TA}z_0\), and its volume multiplier is \(\det(e^{TA})=e^{T\operatorname{tr}A}\). Take

\[
A=\begin{bmatrix}0.2&2\\0.4&-0.1\end{bmatrix},\qquad T=2.
\]

The trace is 0.1. Volume multiplies by \(e^{0.2}=1.221403\); density along the trajectory multiplies by \(e^{-0.2}=0.818731\). The off-diagonal terms change the patch's shape and trajectory even though they do not appear directly in the trace.

[Figure O11 — A small parallelogram transported by the exact matrix exponential. Label area ratio, conserved probability mass and inverse density ratio. Show a second rotation field with zero divergence: density can be transported without local volume change.]

A **continuous normalizing flow**, or CNF, integrates both state and log-density. For likelihood evaluation we also need the base density and the correct direction of integration. For sampling, moving a base sample may suffice without computing its density.

### Estimate a trace without constructing a whole Jacobian

For a random vector \(\epsilon\) with mean zero and covariance identity,

\[
\mathbb E[\epsilon^\top J\epsilon]=\operatorname{tr}J.
\]

This follows by expanding the sum: cross-coordinate terms have zero expectation and diagonal terms have unit second moment. Jacobian-vector or vector-Jacobian products can evaluate the quadratic form without explicitly storing all \(d^2\) entries. [FFJORD](https://arxiv.org/html/1810.01367v2) uses this strategy in continuous density models.

Our matrix above makes the randomness visible. The four equally likely sign vectors \((\pm1,\pm1)\) produce trace estimates 2.5, −2.3, −2.3 and 2.5. Their mean is 0.1. One estimate can be far from the trace, even with the wrong sign. The estimator is useful because of its expectation and computational cost, not because every draw is exact.

Keep a probe fixed over an individual integration solve so the augmented right-hand side is a consistent function for the solver. A finite-sample trace estimate and numerical integration introduce distinct errors. An unbiased ideal log-density estimate does not imply an unbiased density after exponentiation. Full maximum-likelihood training, bottleneck trace identities and model evaluation belong in the [normalizing-flows lesson](/learn/path/full-curriculum/normalizing-flows-realnvp-glow-neural-ode?module=generative-models).

## 9. Deeper branch: learn a velocity without solving during every training example

Suppose we pair a noise sample \(x_0\) with a data example \(x_1\), choose a time uniformly, and form

\[
x_t=(1-t)x_0+tx_1,\qquad v_{\rm target}=x_1-x_0.
\]

The chosen straight path and its velocity can be calculated directly. Train a network to predict this velocity from \((t,x_t)\) using squared error. At generation time, solve \(x'=v_\theta(t,x)\) from a noise sample.

This is a simple endpoint-interpolation form of **conditional flow matching**. Some formulations retain a small final noise level; their path and target velocity change correspondingly. The key computational benefit is that the ordinary regression training objective does not require integrating the learned ODE for each target. Generating a new sample still requires a solve unless an additional approximation or distillation changes that process.

Why can different training paths provide useful targets at the same place? Squared-error regression learns their conditional mean velocity. Consider two synthetic pairs: \(0\to2\) and \(2\to0\). At time 0.5, both are at position one, but their target velocities are +2 and −2. A single model receiving only that position and time cannot return both. Its least-squares prediction is zero: mean loss four, compared with eight if it always predicts +2.

[Figure O12 — Two conditional paths meet at a training point; opposing target arrows average to zero. Next to them show a separately labeled learned marginal field. Do not depict every individual straight path as a trajectory of that single field.]

This local discrete example explains averaging; it is not a full smooth-density generative model. In the continuous theory, the conditional mean field transports the marginal probability path under appropriate regularity assumptions. Straight conditional training paths do not guarantee straight generated trajectories or a globally optimal transport map. [Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2) states that distinction explicitly.

This connection helps interpret modern generative systems without assuming that all ODE-trained models use the same loss. CNF likelihood training uses density change; flow-matching regression uses target velocities. A variational latent ODE uses reconstruction and a prior penalty. They share differential-equation machinery but optimize different objectives. Continue the generative route through [Rectified Flow and Flow Matching](/learn/path/full-curriculum/rectified-flow-flow-matching?module=generative-models).

## 10. Deeper branch: put useful structure into a learned rule

A generic field can model many relationships, but useful restrictions can make a problem easier to learn and interpret. Start with the structure of the task.

**Known physics plus a learned correction.** If a mechanistic model explains most of a process but misses one force or reaction term, write \(z'=f_{\rm known}(t,z)+f_\theta(t,z)\). The learned part need not rediscover everything. It still must be tested outside the observations used for fitting: many different fields can match one short trajectory. Fitting a trajectory is not proof that the underlying physical mechanism has been identified.

**Energy-based dynamics.** For position \(q\), momentum \(p\) and a learned energy \(H_\theta(q,p)\), Hamilton's equations use \(q'=\partial H/\partial p\), \(p'=-\partial H/\partial q\). Along the exact autonomous dynamics,

\[
\frac{dH}{dt}
=\frac{\partial H}{\partial q}^{\!\top}\frac{\partial H}{\partial p}
-\frac{\partial H}{\partial p}^{\!\top}\frac{\partial H}{\partial q}=0.
\]

That cancellation is an architectural property. A numerical solver may still drift in energy; a model of a damped system may need dissipation instead of an exact conservation constraint. The earlier rotation example made the solver issue visible before this more advanced application.

**Learn a solution versus learn its derivative.** A physics-informed neural network can parameterize a solution \(u_\theta(t,x)\) and penalize its equation residual at sampled points. A Neural ODE parameterizes a vector field and numerically integrates it from initial conditions. They can be combined, but one is not simply another name for the other. The [PINN lesson](/learn/path/full-curriculum/physics-informed-neural-networks-pinns?module=frontier-research) owns residual losses and boundary-condition design.

**Events and interventions.** An event may stop integration when a state reaches a threshold. A sensor update or physical impact may change the state discontinuously. These require event/root handling or explicit jump rules; an ordinary smooth solve will not invent them. Event-time gradients can become delicate near tangencies or changes in which event occurs first. A threshold crossing also needs a detection policy: a solver that checks signs only at step boundaries can miss multiple crossings inside one step.

**Controlled and stochastic dynamics.** If a stream continuously drives the state, a controlled differential equation makes that input path part of the dynamics. If uncertainty is modeled by a stochastic differential equation, the driving noise and stochastic calculus alter the solver and gradient problem. A deterministic ODE with newly sampled dropout on every field evaluation is not an automatically valid substitute for either construction.

These applications are interesting because they change the model's assumptions, not simply because they attach a new industry name to the same diagram. The author talk in the resources offers another route into constrained dynamics; advanced applications require their own data, assumptions and evaluation.

## 11. Diagnose the right layer of a problem

A useful implementation begins with a small, deterministic field and a checkable solver. Add complexity after establishing which part of the system needs it.

| Observation | A question to investigate | Useful controlled check |
| --- | --- | --- |
| Endpoint changes under finer resolution | Was training exploiting a coarse numerical map? | Keep weights/input fixed; compare methods and step sizes |
| Gradient differs from a reference | Are both differentiating the same objective and discretization? | Compare autograd with finite differences of the exact executed loss |
| Backward reconstruction fails | Does reversing dynamics amplify endpoint error? | Test an analytic system; compare stored/checkpointed states |
| Adaptive solver takes many steps | Is the cause scaling, tolerances, transients, smoothness or stiffness? | Normalize coordinates; inspect rejection/error traces; compare an appropriate implicit method |
| More input examples change a batch member's result | Are states coupled by BatchNorm or a shared adaptive error controller? | Evaluate examples individually and inspect the batch error norm |
| Trajectory seems to cross itself | Is it a projection or a visit at another time? | Inspect full state with time labels |
| Forecast looks too accurate | Did the encoder see observations beyond the forecast origin? | Rebuild the information-availability mask before fitting |
| Density behaves implausibly | Is the divergence sign/direction right, and is trace noise large? | Use an exact linear flow with known determinant first |

Batching deserves particular care. A shared adaptive controller can use a norm over a whole batch. Many easy coordinates may dilute one difficult coordinate under an RMS norm; a maximum-based norm behaves differently. Even when the mathematical fields are independent, step decisions can depend on which examples are batched together.

For neural fields, smooth activations often make high-order integration easier to use. ReLU is Lipschitz, so it does not by itself destroy uniqueness, but its kinks can affect differentiability and numerical-order behavior. Repeated random dropout masks or mutable normalization statistics can change the field between evaluations; decide what fixed function the solver is supposed to integrate.

Performance depends on field cost, forward and backward evaluations, state size, saved outputs, solver/controller overhead and device utilization. Record those factors before making a timing claim. More accurate integration need not improve statistical prediction if model or data error dominates.

## 12. Practice: transfer the mechanism

Work out a prediction before opening a hint. These inputs differ from the worked calculations. Numerical answers refer to the stated finite computation or exact equation, as named.

### 1. A different decay

Start at \(z_0=2\), with \(z'=-3z\). Take two Euler steps of length 0.1. Compare the result with the exact state at time 0.2. Explain the sign of the error.

<details>
<summary>Hint</summary>

Each Euler step multiplies the state by \(1-3(0.1)\). Compare that factor with the exact exponential factor.

</details>

<details>
<summary>Solution and reasoning</summary>

Euler gives \(2(0.7)^2=0.98\). The exact state is \(2e^{-0.6}=1.097623272\), so the error, numerical minus exact, is −0.117623272. The start-of-step tangent keeps decreasing at its initial slope while the exact positive state's decay slows. For this step size, Euler therefore decreases too far.

</details>

### 2. Resolution without new weights

A four-dimensional field takes the state plus time through a 16-unit tanh hidden layer and a four-output affine layer. A three-class affine head reads the final state. Count the parameters. Does increasing classical RK4 steps from four to twelve triple them?

<details>
<summary>Hint</summary>

An affine map from \(a\) inputs to \(b\) outputs has \(ab+b\) parameters. Count field and readout separately from evaluations.

</details>

<details>
<summary>Solution and reasoning</summary>

The field has \(5(16)+16+16(4)+4=164\) parameters. The readout has \(4(3)+3=15\), totaling 179. Four versus twelve RK4 steps require 16 versus 48 field evaluations, but both use the same 179 parameters. The forward computation grows; the parameter set does not.

</details>

### 3. A local estimate is not a final guarantee

An embedded step has current state 0.5, Heun state 0.49 and Euler state 0.48. Let relative tolerance be 0.02 and absolute tolerance 0.001. Is it accepted by the lesson's scalar controller? Does acceptance prove endpoint error below 0.001?

<details>
<summary>Hint</summary>

Use the maximum magnitude of current and Heun states in the denominator.

</details>

<details>
<summary>Solution and reasoning</summary>

The scale is \(0.001+0.02(0.5)=0.011\). The normalized difference is \(0.01/0.011=0.90909\), so this step is accepted. That is a comparison of two local approximations. Accumulation, estimator accuracy and conditioning separate it from a guarantee about final global error.

</details>

### 4. Differentiate one Euler step

Use \(z'=\theta z\), \(z_0=1.5\), one step of length 0.2, target 1, and half squared error. At \(\theta=-1\), calculate the prediction and parameter gradient of this one-step program.

<details>
<summary>Hint</summary>

Differentiate \(z_1=z_0(1+h\theta)\), then apply the loss derivative.

</details>

<details>
<summary>Solution and reasoning</summary>

The prediction is 1.2; \(dz_1/d\theta=0.3\). Therefore \(dL/d\theta=(1.2-1)(0.3)=0.06\). A small gradient-descent step decreases \(\theta\), reducing the numerical prediction toward the target. The exact continuous prediction \(1.5e^{-0.2}\) defines a different loss; substituting it halfway through this calculation would mix objectives.

</details>

### 5. Separate invertibility from a readout

Can a unique one-dimensional continuous flow followed by one linear threshold label inputs −2 and 3 as class one and input 0 as class zero? Construct an augmented solution and name a numerical failure that could confuse this test.

<details>
<summary>Hint</summary>

Preserve ordering in one dimension, then use a second coordinate proportional to the square of the first.

</details>

<details>
<summary>Solution and reasoning</summary>

The one-dimensional flow keeps 0 between the other points, so a single threshold cannot select both outer points alone. Starting at \((x,0)\), the field \(x'=0,y'=x^2\) reaches \((-2,4),(0,0),(3,9)\) at time one; \(y>1\) works. A coarse explicit solver can reverse or collapse order even where the exact flow does not. A nonlinear readout also changes the representational question and must not be silently substituted.

</details>

### 6. Observation deletion

Start at zero. Let \(h'=-0.5h\), and update by \(h^+=0.7h^-+0.3x\) at times 0.4 and 1.0 with observations 2 and 0. Query at 1.2. Compare with omitting the second observation.

<details>
<summary>Hint</summary>

The second observed value is zero, but its update still scales existing state by 0.7.

</details>

<details>
<summary>Solution and reasoning</summary>

After the first observation, state is 0.6. With the zero observation, the final state is \(0.6e^{-0.3}(0.7)e^{-0.1}=0.42e^{-0.4}=0.281534419\). Omitting it gives \(0.6e^{-0.4}=0.402192028\). Zero is evidence processed through an update; deletion removes that operation.

</details>

### 7. Expansion and density

A two-dimensional linear field has matrix \(\begin{bmatrix}0.4&3\\0&-0.1\end{bmatrix}\). Over 1.5 units of time, what are the log-density change and volume multiplier? Does a large off-diagonal entry invalidate the trace calculation?

<details>
<summary>Hint</summary>

The trace is the diagonal sum. Density and volume change inversely along the flow.

</details>

<details>
<summary>Solution and reasoning</summary>

The trace is 0.3. Log-density changes by −0.45, and volume multiplies by \(e^{0.45}=1.568312185\). Density multiplies by \(e^{-0.45}=0.637628152\). The off-diagonal entry alters shape and motion; the determinant identity still uses the trace for this constant linear field.

</details>

### 8. What can a velocity regressor know?

Two equally likely training pairs are \(−1\to3\) and \(3\to−1\). At time 0.5, what location and target velocities do they present? What prediction minimizes mean squared loss at that location? Why should you not draw both pair paths as solutions of a single unique field?

<details>
<summary>Hint</summary>

A model receiving only position and time cannot distinguish the pair identity.

</details>

<details>
<summary>Solution and reasoning</summary>

Both present location one, with velocities +4 and −4. Their conditional mean is zero; the mean squared loss there is 16. Always choosing +4 would give mean loss 32. Under uniqueness, one field cannot choose two velocities at the identical state and time. Pair-conditioned training paths and trajectories of the learned marginal field are different objects.

</details>

### 9. Design an honest comparison

A report trains a neural flow and a residual classifier, tries new tolerances after reading test accuracy, and publishes only the best seed. It calls a two-coordinate projection proof that trajectories never cross. Propose a corrected protocol.

<details>
<summary>Hint</summary>

Separate data roles, statistical variation, numerical sensitivity and the full-state mathematical claim.

</details>

<details>
<summary>Solution and reasoning</summary>

Declare splits, preprocessing, architectures, budgets, seeds and model-selection rules before assessment. Fit transformations only on training data and select with validation; retain all declared seeds and report actual parameter/evaluation costs. Perform solver-sensitivity checks on a specified diagnostic or validation set with fixed weights, then assess under a fixed protocol. A projection is a visualization of selected coordinates, not a proof about full-state uniqueness. State theorem assumptions and use analytic counterexamples or full-state checks for the claim actually being made.

</details>

You are ready to continue when you can distinguish a field from a solved trajectory, compute and compare numerical updates, name the objective behind a gradient, explain augmentation with its readout assumption, and keep observation availability separate from evaluation time. You should also be able to read an experiment's counts and numerical limits without needing every method to win.

## 13. Continue and learn another way

The next topic in this module is [Hybrid SSM–Transformer Architectures (Jamba)](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals). It returns to discrete token sequences and combines different memory operations. Our current lesson adds another distinction to that comparison: a continuous dynamical model still needs a concrete numerical execution method.

Useful routes into the subject:

- [Neural Ordinary Differential Equations](https://arxiv.org/html/1806.07366v5), Chen and colleagues. Primary introduction to learned dynamics, adjoints and applications. Read the mathematical model first; revisit memory claims with the numerical qualifications in this lesson.
- [Discretize-Optimize vs. Optimize-Discretize](https://arxiv.org/html/2005.13420v2), Onken and Ruthotto. An intermediate numerical perspective on gradients, rediscretization and extrapolation. Its experiments are specific evidence, not universal timing ratios.
- [Augmented Neural ODEs](https://arxiv.org/html/1904.01681v1), Dupont and colleagues. Read the one-dimensional construction and appendix uniqueness argument after section 5; the linear readout and exact-flow assumptions are essential.
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) and its [FAQ](https://github.com/rtqichen/torchdiffeq/blob/master/FAQ.md). Practical PyTorch API and solver/adjoint guidance. Repository documentation was reviewed; the package and its official examples were not executed for this packet. The downloadable local study is self-contained with PyTorch.
- [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/). A JAX-oriented explanation of checkpointed differentiation and continuous backsolves. Read after the scalar gradient experiment rather than starting with every API class.
- [Latent ODEs for Irregularly-Sampled Time Series](https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf). Primary methods for encoding observations and separating interpolation from extrapolation. Useful after the observation-jump investigation.
- [FFJORD](https://arxiv.org/html/1810.01367v2) and [Flow Matching](https://arxiv.org/html/2210.02747v2). Two different training routes for continuous generative models. Their objectives deserve separate study, even though both generate with learned dynamics.
- [Ricky T. Q. Chen's CVPR 2020 workshop talk](https://anucvml.github.io/ddn-cvprw2020/talk2.html). An alternate video route into constraints, physical dynamics and probabilistic models. The organizer's title, speaker and abstract were verified; the embedded recording was not watched, and no timestamp is claimed. Use the paper and checked calculations for technical details.
- [Author calculations](ode_calculations.py), [calculated inputs and outcomes](calculated-inputs.json), [full training program](neural_ode_study.py), [data and provenance](data-provenance.md). These reproduce this lesson's small examples and fitted study offline. No external large-model checkpoint is required.
