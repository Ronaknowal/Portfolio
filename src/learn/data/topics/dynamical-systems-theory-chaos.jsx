import { Callout, Code, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Dynamical Systems Theory & Chaos",
  readTime: "~43 min",
  content: () => <div>
    <H2>1. The question: how does a state evolve under a rule?</H2>
    <Prose>
      A dynamical system specifies how a state changes over time. The state may be a pendulum's position and velocity, a population, a chemical concentration, a hidden state in a recurrent model, or a control system. The rule may be continuous in time, expressed as a differential equation, or discrete, expressed as an update map. Dynamics helps us reason about trajectories, long-run behaviour, stability, and the limits of prediction.
    </Prose>
    <MathBlock>{`\\frac{dx}{dt}=f(x,t)\\quad\\text{(continuous time)}, \\qquad x_{t+1}=F(x_t)\\quad\\text{(discrete time)}`}</MathBlock>
    <Prose>
      The key distinction from a static model is feedback: tomorrow's state depends on today's state. A small modelling error can therefore compound through repeated updates, even if every individual update looks reasonable.
    </Prose>

    <H2>2. Fixed points and stability are the first things to inspect</H2>
    <Prose>
      A fixed point (or equilibrium) is a state that does not change under the dynamics. For a discrete map it satisfies <Code>x* = F(x*)</Code>; for an autonomous differential equation it satisfies <Code>f(x*) = 0</Code>. The important follow-up is stability: does a small perturbation return toward the equilibrium or move away from it?
    </Prose>
    <MathBlock>{`|F'(x^*)|&lt;1\\Rightarrow\\text{locally stable discrete fixed point}, \\qquad \\operatorname{Re}(\\lambda(J_f(x^*)))&lt;0\\Rightarrow\\text{locally stable continuous equilibrium}`}</MathBlock>
    <Prose>
      The Jacobian <Code>J_f</Code> is the multidimensional derivative. Its eigenvalues describe local expansion, contraction, rotation, and oscillation. Linearisation is a local approximation, so it can be misleading far from the equilibrium or near strongly non-linear transitions.
    </Prose>

    <H2>3. The logistic map shows feedback becoming complicated</H2>
    <Prose>
      The logistic map is a tiny deterministic population model: x is a scaled population between zero and one, r controls growth, and the factor <Code>(1 - x)</Code> represents limited resources. Its simplicity makes it ideal for seeing bifurcations and chaos without hiding behind a complex simulator.
    </Prose>
    <MathBlock>{`x_{t+1}=r x_t(1-x_t)`}</MathBlock>
    <Prose>
      It has fixed points 0 and <Code>(r - 1) / r</Code>. The nonzero fixed point has derivative <Code>2 - r</Code>, so it is stable for 1 &lt; r &lt; 3. As r rises, the system moves through period doubling—one stable value becomes a two-cycle, then four, then more complex behaviour. For many r values beyond about 3.57 it is chaotic, though periodic windows still occur.
    </Prose>

    <H2>4. Chaos is deterministic but rapidly unpredictable</H2>
    <Prose>
      Chaotic systems have sensitive dependence on initial conditions: two states that begin almost identically can separate exponentially fast. There is no random number generator in the rule. The unpredictability comes from limited measurement precision and repeated nonlinear feedback, which create a finite horizon for accurate individual forecasts.
    </Prose>
    <CodeBlock language="python">{`def logistic(x, r):
    return r * x * (1 - x)

r = 3.9
x, y = 0.5, 0.500001  # initial states differ by one millionth
for _ in range(50):
    x = logistic(x, r)
    y = logistic(y, r)

print(round(x, 6), round(y, 6), round(abs(x - y), 6))`}</CodeBlock>
    <CodeBlock language="output">{`0.241355 0.800844 0.559489`}</CodeBlock>
    <Prose>
      A Lyapunov exponent quantifies average local separation; a positive largest exponent is a hallmark of chaos. One dramatic trajectory is not proof of chaos. Estimate sensitivity across states and time, rule out numerical artifacts, and distinguish deterministic nonlinearity from stochastic noise.
    </Prose>

    <H2>5. Attractors, cycles, and basins of attraction</H2>
    <Prose>
      Long-run trajectories can approach a fixed point, a periodic orbit (limit cycle), a more complicated attractor, or escape the model's meaningful region. A basin of attraction is the set of initial states that lead to the same attractor. Multiple basins matter in optimisation and control: two nearly identical starting conditions can settle into qualitatively different outcomes if they sit on opposite sides of a boundary.
    </Prose>
    <Prose>
      Phase portraits visualise trajectories against one another—such as position versus velocity—rather than against time. They often expose cycles, equilibria, and unstable regions that a single time-series plot obscures. In high dimensions, projections can be useful but must not be mistaken for the full state-space geometry.
    </Prose>

    <H2>6. Connections to ML, control, and simulation</H2>
    <Prose>
      Recurrent networks and iterative optimisation are discrete dynamical systems; exploding or vanishing gradients correspond to repeated expansion or contraction. Neural ODEs define continuous-time dynamics learned from data. Model-predictive control plans actions using a dynamics model, while reinforcement learning must account for state transitions. In all these cases, stability and error propagation are engineering concerns, not just theoretical decoration.
    </Prose>
    <Callout accent="green" label="A learned one-step model can still fail long term">
      Low one-step prediction error does not guarantee accurate rollouts. Reusing a model's own output as its next input exposes it to states absent from training and compounds small bias. Evaluate multi-step trajectories, conservation laws, stability regions, and uncertainty—not only single-step loss.
    </Callout>

    <H2>7. Numerical and modelling pitfalls</H2>
    <Prose>
      Numerical integrators introduce their own dynamics: a large step size can manufacture instability or damp a real oscillation. Compare step sizes and, when relevant, use integrators that preserve important structure. Estimate parameters on held-out trajectories, respect measurement noise and unobserved inputs, and avoid interpreting correlation in a time series as evidence of a complete state model. External forcing can make an apparently autonomous system time-varying.
    </Prose>
    <Callout label="Practice">
      Simulate the logistic map for r values 2.5, 3.2, 3.5, and 3.9 from several nearby initial states. For each, decide whether trajectories approach a fixed point, a cycle, or show sensitive behaviour. What finite-time evidence would you collect before claiming the last case is chaotic?
    </Callout>
  </div>,
};

export default content;
