import { Callout, Code, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Stochastic Processes (Markov Chains, Brownian Motion, Poisson)",
  readTime: "~44 min",
  content: () => <div>
    <H2>1. Why a distribution is not enough when time matters</H2>
    <Prose>
      A single random variable describes one uncertain quantity. A stochastic process describes a sequence or continuum of uncertain quantities indexed by time, space, or another ordering. Demand tomorrow depends on demand today; arrivals accumulate; prices fluctuate; a sensor drifts. The process specifies not only what values are likely, but how values relate across time.
    </Prose>
    <MathBlock>{`\\{X_t:t\\in T\\}`}</MathBlock>
    <Prose>
      Choosing a process means choosing assumptions about memory, increments, state space, and time scale. The models below are useful baselines, not default truths for every timestamped dataset.
    </Prose>

    <H2>2. Markov chains model state transitions</H2>
    <Prose>
      A discrete-time Markov chain has the Markov property: once you know the current state, the next state does not depend on earlier history. It does <em>not</em> mean observations are independent; the current state carries dependence forward. A transition matrix P contains the chance of moving between states, with each row summing to one.
    </Prose>
    <MathBlock>{`P(X_{t+1}=j|X_t=i,X_{t-1},\\ldots)=P(X_{t+1}=j|X_t=i)=P_{ij}`}</MathBlock>
    <Prose>
      With row-vector convention, the state distribution evolves as <Code>pi_(t+1) = pi_t P</Code>. In the weather example below, a sunny day remains sunny with probability 0.8 and a rainy day becomes sunny with probability 0.3. Starting sunny, after five steps the distribution is close to the long-run distribution of 60% sunny and 40% rainy.
    </Prose>
    <CodeBlock language="python">{`P = [[0.8, 0.2], [0.3, 0.7]]  # rows: current state, columns: next state
distribution = [1.0, 0.0]       # start sunny

for _ in range(5):
    distribution = [
        distribution[0] * P[0][j] + distribution[1] * P[1][j]
        for j in range(2)
    ]

print([round(x, 3) for x in distribution])`}</CodeBlock>
    <CodeBlock language="output">{`[0.613, 0.388]`}</CodeBlock>
    <Prose>
      A stationary distribution pi satisfies <Code>pi = pi P</Code>. Convergence to it needs conditions such as irreducibility and aperiodicity. Absorbing states, cycles, and disconnected state regions can prevent the simple "eventually settles" intuition from applying.
    </Prose>

    <H2>3. Poisson processes model independent arrivals</H2>
    <Prose>
      A homogeneous Poisson process counts events over continuous time. Its rate lambda is the expected event count per unit time. It assumes independent increments and a constant rate: the number of requests in one time window is independent of the count in a disjoint window, and only the window length matters.
    </Prose>
    <MathBlock>{`N(t)\\sim\\operatorname{Poisson}(\\lambda t), \\qquad P(N(t)=k)=e^{-\\lambda t}\\frac{(\\lambda t)^k}{k!}`}</MathBlock>
    <Prose>
      If lambda is 2.5 arrivals per minute, three minutes have expected count 7.5. The chance of no arrivals is exp(-7.5), about 0.0006. Inter-arrival times are exponential, which is memoryless: after waiting five minutes, the remaining wait has the same distribution as a fresh wait.
    </Prose>
    <CodeBlock language="python">{`import math

rate_per_minute = 2.5
minutes = 3
p_zero_arrivals = math.exp(-rate_per_minute * minutes)

print(round(p_zero_arrivals, 4))
print(round(1 - p_zero_arrivals, 4))`}</CodeBlock>
    <CodeBlock language="output">{`0.0006
0.9994`}</CodeBlock>
    <Prose>
      Poisson processes fail for bursts, daily seasonality, feedback, aftershocks, and capacity limits. Use non-homogeneous Poisson processes for time-varying rates, Hawkes processes for self-excitation, or queueing/state-space models when the process has memory or congestion.
    </Prose>

    <H2>4. Brownian motion models continuous random fluctuation</H2>
    <Prose>
      Standard Brownian motion, W(t), starts at zero, has independent Gaussian increments, and has variance that grows with elapsed time. Over a time interval of length delta, the increment is Normal(0, delta). Its paths are continuous but almost surely nowhere differentiable—a key reason ordinary calculus needs modification for stochastic differential equations.
    </Prose>
    <MathBlock>{`W(0)=0, \\qquad W(t)-W(s)\\sim\\mathcal{N}(0,t-s)\\quad(t&gt;s)`}</MathBlock>
    <Prose>
      The following is a five-step simulation with step size one. It is one possible path, not an estimate of the process mean; across many paths, the expected position at each time remains zero while uncertainty spreads out.
    </Prose>
    <CodeBlock language="python">{`import random

rng = random.Random(11)
position, path = 0.0, []
for _ in range(5):
    position += rng.gauss(0, 1)  # Normal(0, step_size) with step_size = 1
    path.append(round(position, 3))

print(path)`}</CodeBlock>
    <CodeBlock language="output">{`[-1.224, -0.846, 0.149, -0.365, -1.694]`}</CodeBlock>
    <Prose>
      Brownian motion is a mathematical building block for diffusion, noisy physical systems, latent continuous-time models, and finance. Raw financial returns are not automatically Brownian: volatility clustering, jumps, changing regimes, and market microstructure can violate its assumptions.
    </Prose>

    <H2>5. How the models relate—and do not relate</H2>
    <Prose>
      Markov chains evolve through discrete states; Poisson processes count events; Brownian motion evolves continuously in value and time. Each has an appropriate state and observation model. A hidden Markov model, for example, uses a Markov chain for unobserved states and a separate distribution for noisy observations. Do not force continuous sensor values into arbitrary bins merely to use a Markov chain, or treat every count series as a Poisson process because it contains integers.
    </Prose>
    <Callout accent="green" label="Time scale changes the model">
      A process that is approximately memoryless at hourly resolution can have strong dependence at minute resolution. Always specify units, sampling interval, and whether timestamps record event time, ingestion time, or a delayed proxy. Many apparent model failures are really time-definition failures.
    </Callout>

    <H2>6. A practical modelling workflow</H2>
    <Prose>
      Define the state, event, and time index. Plot counts, durations, transitions, and autocorrelation before choosing a process. Test simple baseline assumptions: do transition probabilities drift, do increments cluster, does variance grow with time, are there structural breaks? Fit on the past and validate on future windows. For simulations, compare generated trajectories—not only marginal histograms—to real ones. For decisions, propagate uncertainty through the downstream cost rather than reporting only an average path.
    </Prose>
    <Callout label="Practice">
      A support team receives tickets with a strong weekday pattern and bursts after outages. Explain why a homogeneous Poisson process is inadequate. Name one model extension for the schedule and one for burstiness, then describe a held-out time-based check you would use.
    </Callout>
  </div>,
};

export default content;
