export default {
  summary: 'Group observations through a graph of locally crowded neighbourhoods, decide what to do with the rows a density rule leaves out, and know when one radius cannot answer the question.',
  outcomes: [
    'Compute core, border and noise types with a closed, self-counting neighbourhood and explain why only core rows transmit expansion',
    'Distinguish order-invariant types and core components from an order-dependent shared-border assignment',
    'Read the fourth-neighbour distance plot as the exact inverse of the count test and choose a scale to investigate rather than a scale to trust',
    'Tell a unit change from a metric change and convert radius and coordinates consistently',
    'Run and report a real-data density analysis with coverage, conditional scores and the retained row IDs',
    'Prove with intervals when no single radius recovers intended groups of different density, and repair the data or move to a hierarchy',
    'Read OPTICS reachability and HDBSCAN stability correctly, and state a frozen-reference policy for new observations'
  ],
  prerequisites: ['Clustering Evaluation & Validation (Silhouette, ARI, NMI)'],
  sequence: [
    'Count a closed neighbourhood including the row itself on a ten-row trail',
    'Build the core graph, attach borders, and show why a border cannot transmit',
    'Name reachability and connectivity; implement a complete small DBSCAN',
    'Change radius and count; read the sorted fourth-neighbour distances',
    'Units versus metric with a paired conversion and a faulty one',
    'Fit Iris offline and report coverage beside conditional scores',
    'Prove incompatible radius intervals and repair the fixture',
    'Deeper: OPTICS ordering, HDBSCAN mutual reachability and stability, cost, new observations'
  ],
  visual: {
    type: 'Trail strips with closed intervals and rosters; core graph with a struck-through border bridge; sorted c₄ curve; incompatible interval bars; OPTICS reachability bars with core-start marks; stability lifetime tree; concentric rings versus a bisector; equal-unit neighbourhood graphs in four labs',
    question: 'Which rows are crowded enough to transmit, what changes when a radius, count, unit or visiting order changes, how much of a real collection survives a setting, and can any radius satisfy two density requirements at once? Predict before each control.',
    interaction: 'Record a prediction; edit any trail coordinate, radius, count or order; multiply axes and radius; choose an Iris radius and count and compare snapshots on common rows; edit group offset and spacing and test a radius against the analytic interval.'
  },
  practice: {
    task: 'Twelve changed tasks: a new five-row trail, a shared border moved off the line, a false transitivity proof, duplicates and the count convention, a faulty unit conversion, repairing the incompatible fixture, a perfect score on survivors, an OPTICS cluster start, non-overlapping branch selection, dense-output memory, an independent Iris report and a frozen-reference policy.',
    success: 'Types and components match the closed self-counting rule; borders are never used to transmit; coverage and the retained IDs accompany every conditional score; interval arguments name both thresholds; reports state representation, parameters and what the result does not claim.'
  },
  misconceptions: [
    'A row with fewer than m neighbours is noise',
    'A border row can connect two components',
    'Density connectivity through any row is transitive',
    'The neighbour-distance elbow identifies the correct radius',
    'Standardizing coordinates is a unit conversion',
    'A higher conditional score on fewer rows is a better clustering',
    'Every OPTICS bar above the cut is noise',
    'Every locally stable HDBSCAN branch should be selected',
    'A tree index makes a dense radius query logarithmic',
    'scikit-learn DBSCAN can predict new rows'
  ],
  sources: [
    'https://file.biolab.si/papers/1996-DBSCAN-KDD.pdf',
    'https://www.jstatsoft.org/article/view/v091i01',
    'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html',
    'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html',
    'https://archive.ics.uci.edu/dataset/53/iris'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/DBSCAN-LESSON-DESIGN.md',
  reviewFocus: 'Closed ≤ neighbourhood counting self; border attachment versus transmission; order invariance of types and components; c_m indexing with kneighbors(X); the 0.375 / 0.75 interval fixture; the seven Iris settings and their coverage; OPTICS core-start rule; sklearn versus contrib HDBSCAN min_samples convention; no predict on sklearn DBSCAN.'
};
