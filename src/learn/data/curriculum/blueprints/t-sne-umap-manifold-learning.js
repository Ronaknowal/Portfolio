export default {
  summary: 'Build and inspect a low-dimensional map by making its preserved relationships explicit: graph routes, neighbor probabilities, local reconstruction recipes and measured neighborhood retention.',
  outcomes: [
    'Distinguish intrinsic dimension from ambient coordinates and recognize when a graph introduces a shortcut or loses connectivity',
    'Calculate retained, missing and false neighbors and distinguish retention from trustworthiness and continuity',
    'Normalize a Gaussian affinity row, interpret perplexity and explain its limits under equal-distance ties',
    'Construct symmetric t-SNE probabilities and connect its normalized map affinities to attraction, repulsion and the exact gradient',
    'Separate UMAP local graph calibration, fuzzy union and sampled layout optimization from an ideal full-pair cost',
    'Audit actual handwritten images using consistent source identities, pixel distances and saved map settings',
    'Keep training and validation in one fitted transformation and evaluate pixels, PCA and UMAP as complete prediction pipelines',
    'Derive classical MDS on a line, compute an LLE reconstruction and explain what Isomap and spectral alternatives change',
    'Demonstrate why topology and metric structure need not survive an embedding with a calculated square projection'
  ],
  prerequisites: ['PCA & Dimensionality Reduction'],
  sequence: [
    'Start from 300 handwritten images and a concrete inspection question',
    'Build a graph on a U-shaped route; change one connection and measure its effect',
    'Define what a map keeps and audit identities rather than interpreting empty plot space',
    'Turn distances into a Gaussian row and choose bandwidth through entropy',
    'Symmetrize probabilities, normalize a Student kernel and move coordinates by the gradient',
    'Calibrate and merge UMAP memberships before considering a sampled layout',
    'Compare actual saved digit maps and investigate an individual exception to the aggregate ranking',
    'Fit transforms for new images and assess held-out task performance',
    'Return for MDS, Isomap, LLE, graph eigenvectors and a topology counterexample',
    'Derive and run a tiny exact optimizer; transfer the ideas to changed practice problems'
  ],
  visual: {
    type: 'Metric route and straightened strip; identity-matched false/missing neighbors; Gaussian normalization bars; signed pair forces; directed memberships and fuzzy union; real image maps and measured retention; double centering; local reconstruction; square and projected Rips complexes',
    question: 'Which relationship will survive this edit, and what can be measured in the input space to check the claim?',
    interaction: 'Four distinct investigations: edit graph geometry and trace a route; edit distances and bandwidth; build a fuzzy edge and inspect an ideal pair cost; select an actual digit image and predict retained neighbors before revealing identity-matched lists.'
  },
  practice: {
    task: 'Seven changed problems on a sparse corridor, entropy after merging events, repairing an incorrect metric interpretation, a changed fuzzy union, MDS and LLE on a new line, a broken deployment pipeline and an independent real-image audit.',
    success: 'State the input geometry and tie rule; calculate the actual quantities; keep a graph distinct from a layout; use train-fitted transforms; explain at least one local outcome that differs from a global ranking.'
  },
  misconceptions: [
    'A two-dimensional map preserves all distances, densities, groups or holes',
    'More neighbors always produce a better surface estimate',
    'Trustworthiness is the fraction of retained neighbors',
    'Perplexity is an exact neighbor cutoff and every requested entropy is attainable',
    'Early exaggeration is the same normalized KL objective throughout training',
    'A fuzzy membership is a calibrated probability that a topological feature exists',
    'UMAP min_dist is a hard minimum separation',
    'Negative sampling is an unbiased shortcut for the displayed full-pair cost',
    'Separately fitting training and test maps produces interchangeable coordinates',
    'A more attractive two-dimensional plot guarantees a better classifier'
  ],
  sources: [
    'https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf',
    'https://arxiv.org/html/1802.03426v3',
    'https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf',
    'https://scikit-learn.org/stable/modules/manifold.html',
    'https://umap-learn.readthedocs.io/en/latest/how_umap_works.html',
    'https://distill.pub/2016/misread-tsne/',
    'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/MANIFOLD-LEARNING-LESSON-DESIGN.md',
  reviewFocus: 'Preserve all nine specified figures and four distinct investigations; verify metric aspect ratios and annotation bounds; carry source identities through real maps and tiles; test disconnected routes, tied affinities, fuzzy-union boundaries, coincident saved seeds and the source-row 30 local ranking reversal; verify four complete Python programs and distinguish saved reference maps from new fits; preserve exact MDS/LLE/Rips calculations and all deeper mathematical coverage.'
};
