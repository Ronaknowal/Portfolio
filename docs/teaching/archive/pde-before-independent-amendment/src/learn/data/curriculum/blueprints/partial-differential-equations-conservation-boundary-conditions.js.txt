export default {
  summary: 'Turn a local transport or force balance into a spatial differential model, supply physically consistent data, solve representative problems and explain what the solution does and does not guarantee.',
  outcomes: [
    'Define a spatial field, its units, domain and time interval before writing an equation',
    'Derive conservation, divergence and diffusion locally, including variable conductivity and outward-flux signs',
    'Choose initial, inflow, Dirichlet, Neumann, Robin or periodic data appropriate to the mechanism',
    'Trace transport characteristics and identify which initial or boundary datum determines an observation',
    'Derive boundary-selected heat modes and explain smoothing, loss of heat and a retained constant mode',
    'Use the wave equation with both displacement and velocity data and explain finite dependence and conserved energy',
    'Solve Poisson and harmonic examples while checking source/flux compatibility, uniqueness and the Neumann nullspace',
    'Distinguish classical equations from a locally derived weak balance and admissible shocks',
    'Use explicit energy, maximum and mode arguments to separate forward stability from an ill-posed inverse',
    'Complete and verify a forced diffusion model, then pass a well-specified target to numerical discretization'
  ],
  prerequisites: [
    'Multivariate Calculus & Gradients',
    'Ordinary Differential Equations & Linear Systems',
    'Complex Numbers, Fourier & Laplace Transforms'
  ],
  sequence: [
    'Observe a field and derive a balance on a short interval before introducing differential operators',
    'Specify the domain, initial state and physically signed boundary information',
    'Follow transport characteristics to initial data or an inflow boundary',
    'Derive diffusion modes from boundary conditions and compare the same initial heat profile',
    'Use energy, maximum and kernel arguments to explain smoothing and stability',
    'Derive waves, their dependence cone and the separate role of initial velocity',
    'Solve steady source and boundary problems, including a genuinely two-dimensional harmonic field',
    'Derive a weak point-source identity and explain why a nonlinear shock needs admissibility',
    'Connect inverse diffusion and periodic penetration to the already derived mechanisms',
    'Formulate, solve, diagnose and assess a changed forced rod before Numerical PDEs'
  ],
  visual: {
    type: 'Flux control volumes, space-time characteristics, boundary-selected modes, dependence cones, harmonic fields and weak jump balances',
    question: 'Which data determine this field, and what changes when information can leave, reflect or diffuse through its boundary?',
    interaction: 'Trace a selected observation to its data, compare heat profiles with different physical boundaries, inspect a wave cone, and change source or flux data while checking the same conservation identity'
  },
  practice: {
    task: 'Specify and solve a changed diffusion or wave problem, verify the PDE and all data, and diagnose an incompatible boundary or unjustified numerical claim',
    success: 'Symbols and units are defined; flux signs and initial/boundary conditions agree; the solution is verified independently; conserved or dissipated quantities and approximation limits are stated'
  },
  misconceptions: [
    'An equation alone specifies a unique physical problem',
    'Zero temperature and zero outward heat flux are the same condition',
    'The transport equation needs independently prescribed data at every endpoint',
    'Diffusion always conserves total heat regardless of the boundary',
    'The wave equation needs only an initial shape',
    'A Neumann Poisson problem has a unique solution without compatibility or a mean constraint',
    'A classical derivative exists at a point-source kink or shock',
    'A weak jump balance alone selects the physically admissible nonlinear solution',
    'A sampled plot proves convergence, stability or a general maximum principle'
  ],
  sources: [
    'https://web.stanford.edu/class/math220a/handouts/firstorder.pdf',
    'https://web.stanford.edu/class/math220b/handouts/heateqn.pdf',
    'https://web.stanford.edu/class/math220a/handouts/waveequation1.pdf',
    'https://web.stanford.edu/class/math220b/handouts/laplace.pdf',
    'https://web.stanford.edu/class/math220a/handouts/conservation.pdf',
    'https://www.jirka.org/diffyqs/html/slproblems_section.html',
    'https://ocw.mit.edu/courses/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/resources/heat-equation/',
    'https://www.3blue1brown.com/lessons/pdes/'
  ],
  depth: 'specialist',
  reviewFocus: 'Follow docs/teaching/PARTIAL-DIFFERENTIAL-EQUATIONS-LESSON-DESIGN.md. Check operator domains, boundary signs, compatibility, classical versus weak meaning, convergence hypotheses and analytic-versus-floating error. The brief and independently checked proposed fixtures are design only until a complete lesson and actual native/browser evidence exist.'
};
