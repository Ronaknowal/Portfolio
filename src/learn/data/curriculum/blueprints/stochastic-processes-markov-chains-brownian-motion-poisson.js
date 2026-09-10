export default {
  summary: 'Build and question models of dependence through time, connecting state transitions, arrival clocks and continuous random paths to exact calculations, repeatable simulation and temporal validation.',
  outcomes: [
    'Distinguish a path, a time-slice distribution and a joint process law using concrete counterexamples',
    'Propagate a finite Markov-chain distribution and calculate path, first-hit and absorption quantities',
    'Separate stationary distributions, convergence and occupation averages with their required conditions',
    'Connect Poisson event times, waiting times and interval counts with consistent rate and exposure units',
    'Derive independent splitting and integrated-intensity extensions and recognize when their assumptions fail',
    'Interpret a finite continuous-time generator and distinguish jump frequencies from elapsed-time occupation',
    'Calculate Brownian increment laws, covariance, drift and scale with non-unit time steps',
    'Explain sampled-path limits, coupled refinement, conditional bridges and quadratic variation',
    'Simulate and diagnose changed temporal data without mistaking a matching histogram for a validated process'
  ],
  prerequisites: [
    "Probability Distributions & Bayes' Theorem",
    'Vectors, Matrices & Tensor Operations'
  ],
  sequence: [
    'Read paths and time slices',
    'Move probability through a state graph',
    'Distinguish equilibrium, convergence and time averages',
    'Solve first-passage questions',
    'Connect arrivals, gaps and counts',
    'Route events and change the intensity clock',
    'Separate jump transitions from holding times',
    'Construct and scale Brownian increments',
    'Inspect refinement, bridges and path limits',
    'Validate a process and practise changed assumptions'
  ],
  visual: {
    type: 'Path-law strips, linked transition-mass flows, first-hit survival lanes, arrival/count timelines, holding-time clocks and coupled Brownian paths',
    question: 'What relationship across time does this model assert, and what can a finite observation or simulation actually establish?',
    interaction: 'Apply bounded transition, intensity or drift/scale changes; inspect consistent exact laws and repeatable trajectories; advance first-hit mass, route arrivals and reveal a finer observation grid without changing existing sampled points.'
  },
  practice: {
    task: 'Calculate changed transition, boundary, arrival and Brownian questions; repair false stationarity or independence claims; build a reproducible temporal model report with an explicit assumption check.',
    success: 'Probability, time and scale conventions agree with independently checked values; conditions and censoring are explicit; a sampled trajectory, a distribution and a path-level event are distinguished.'
  },
  misconceptions: [
    'Equal one-time distributions determine the same process',
    'The Markov property means observations are independent or transitions must be time homogeneous',
    'A unique stationary distribution guarantees marginal convergence from every start',
    'Stationary increments make the process stationary',
    'A finite simulation cutoff means the target will never be hit',
    'Poisson cumulative counts at different times are independent',
    'Any routing rule produces independent Poisson streams',
    'Generator entries are per-step probabilities',
    'Sampling at jumps estimates time occupation without weighting durations',
    'Brownian normal increments have standard deviation equal to elapsed time',
    'A finite smooth plot is the continuous Brownian path',
    'Pointwise probability bands give the same coverage for an entire path',
    'Any almost-surely finite stopping rule preserves an unconditional mean'
  ],
  sources: [
    'https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf',
    'https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3a19ce0e02d0008877351bfa24f3716a_MIT6_262S11_chap02.pdf',
    'https://www.columbia.edu/~ks20/4106-18-Fall/Notes-BM.pdf',
    'https://www.columbia.edu/~mh2078/MonteCarlo/MCS_SDEs_MasterSlides.pdf'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/STOCHASTIC-PROCESSES-LESSON-DESIGN.md',
  reviewFocus: 'Row orientation; finite versus countable chain assumptions; stationarity versus marginal and occupation convergence; first-hit boundaries and censoring; independent marks and deterministic intensity; clock units; Brownian covariance and sqrt-time scaling; conditional bridges, exact skeletons, partition convergence and fitted-model limitations.'
};
