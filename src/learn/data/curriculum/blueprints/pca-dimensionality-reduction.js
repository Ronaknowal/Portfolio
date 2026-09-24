export default {
  summary: 'Learn a coordinate system from centered variation, keep the directions that matter for a stated purpose, and inspect exactly what the discarded coordinates lose.',
  outcomes: [
    'Explain a principal direction, an observation score and a mean-restored reconstruction on the same point, and calculate one projection and its squared loss by hand',
    'State why the maximum-variance and minimum-reconstruction-loss definitions agree, using the conserved total of retained plus residual squared length',
    'Fit, transform and inverse-transform with NumPy SVD and scikit-learn, keeping fitted means and directions fixed for new observations',
    'Predict how a change of unit or standardization changes the leading direction, and choose a scaling from the task rather than by default',
    'Choose a component count for an explicit purpose with a train-only fit, a mean-only baseline and a stated error budget, then diagnose individual reconstructed records',
    'Distinguish direction coefficients, feature–score correlations and biplot arrows, and read a labeled score plot without treating explained variance as retained meaning',
    'Connect PCA to pairwise distances, label information, sampling variation, whitening, solver choice and alternative reduction methods, and know when each matters'
  ],
  prerequisites: ['K-Means & Hierarchical Clustering'],
  sequence: [
    'Ask which few numbers can stand for many measurements, on real wines and a four-point fixture',
    'Center, choose a unit direction, read scores and reconstruct with the mean restored',
    'Show that retained plus residual squared length is conserved, so variance and loss agree',
    'Fit once with SVD and the library; transform new observations with fixed quantities',
    'Change units and standardize; see the leading direction move and tie',
    'Split, fit on training rows, set a budget and choose k against a mean baseline',
    'Read coefficients, correlations, scores and biplots correctly',
    'Connect to distances, labels and prediction; practise changed cases; deeper eigen/SVD, degeneracies, sampling, computation and alternatives'
  ],
  visual: {
    type: 'Projection workbench with editable points, rotating ruler, residual right triangles and a score strip; conserved-total bars; matrix-shape trace; unit-and-metric rectangle experiment with variance bars; real Wine variance, score and signed-coefficient views; validation error-budget curve with per-record residual inspection; variance-versus-label strip; simulated Gaussian spectrum; residual-alarm schematic',
    question: "Which direction loses least, what changed when the unit changed, which count meets my budget, and does the retained coordinate still carry the label?",
    interaction: "Rotate or fit a direction on editable points and follow scores, residual segments and squared loss immediately; change units or standardization and inspect the leading direction; move an error budget and inspect the first qualifying count and individual validation wines; vary spreads and label rules and inspect retained-coordinate collisions."
  },
  practice: {
    task: 'Compute a new fit and new-observation reconstruction; retain the second direction instead of the first; repair a misleading scaling conclusion; rerun the Wine budget at 6%; diagnose a pipeline leak; audit a storage promise; separate uncorrelated from independent; interpret a noise-only spectrum; complete a one-purpose mini-project on the supplied Wine data.',
    success: 'Hand calculations match the recorded outputs; reconstructions restore the mean; scaling and evaluation claims name their denominators and fitting data; changed-budget and changed-seed answers are reproduced; explanations separate variance retained from task information retained.'
  },
  misconceptions: [
    'Explained variance is the fraction of meaning or label information preserved',
    'PCA standardizes its inputs automatically, or standardization is always correct',
    'A sign-flipped component is a different or wrong model',
    'Minimizing validation reconstruction error chooses a useful k without a budget',
    'Loadings mean the same numbers in every library and biplot',
    'Uncorrelated scores are independent',
    'A leading sample component proves a hidden low-dimensional cause',
    'Forming the covariance matrix is how SVD-based PCA must be computed',
    'Kernel PCA cannot transform new points and UMAP cannot either',
    'Autoencoders require labels or a GPU'
  ],
  sources: [
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/',
    'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
    'https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html',
    'https://archive.ics.uci.edu/dataset/109/wine',
    'https://scikit-learn.org/stable/common_pitfalls.html#data-leakage'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/PCA-LESSON-DESIGN.md',
  reviewFocus: 'Mean restoration in every reconstruction; n−1 versus n denominators; row-orientation of directions; raw versus standardized Wine ratios and the 133/45 split; validation ratio thresholds 10%/6%/11%; label-collision and tie fixtures; no claim that variance retained equals information retained; current solver names and kernel/UMAP transform facts.'
};
