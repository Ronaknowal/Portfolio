export default {
  summary: 'Choose a meaningful geometry, learn representative centers or a merge hierarchy, and evaluate what the resulting groups actually support.',
  outcomes: [
    'Define observations, features, distance and a useful clustering decision without treating group labels as truth',
    'Calculate and implement Lloyd assignments, mean updates, initialization and explicit stopping contracts',
    'Explain how units, weights, outliers and representation change the squared-distance objective',
    'Compute D-squared seeding probabilities and qualify the vanilla k-means++ expectation guarantee',
    'Build and interpret single, complete, average and Ward hierarchies, including tied cuts and the Ward height scale',
    'Compare reproducible current-library fits, distinguish diagnostics from validation and report a held-out result',
    'Apply weighted clustering to a color palette and connect representatives to streaming summaries and search routing'
  ],
  prerequisites: ['K-Nearest Neighbors (KNN)'],
  sequence: ['Define similarity and the useful grouping question', 'Trace assignment and mean movement', 'Choose units and derive the objective', 'Seed and stop with explicit contracts', 'Construct linkage merges and read a dendrogram', 'Derive Ward merge cost and height', 'Use current APIs and choose k with qualified diagnostics', 'Apply quantization and resource-aware variants', 'Diagnose failure, practise changed cases and report a complete experiment'],
  visual: {
    type: 'Real Old Faithful scatter and forty-leaf Ward dendrogram; point-to-center assignment map with the nearest-center boundary; exhaustive two-group geometry table; D-squared probability strip with sequential draws and seeded frequencies; linked dendrogram and membership map on a tied six-point fixture and a chain fixture; three-image color-palette reconstruction; log-scale inertia and silhouette diagnostics',
    question: 'What changed the grouping: the geometry, the current representatives, the random seed, the linkage rule or the cut? Change the inputs and inspect the mechanism, visual state and computed result immediately.',
    interaction: 'Choose any two seed rows and scrub, step or run Lloyd; compare feature weights against every possible split; distinguish exact D² probabilities from seeded frequencies; explore linkage and merge/count cuts directly; and compare calculated image palettes within the stated teaching cap.'
  },
  practice: {
    task: 'Solve changed centroid, seeding, Ward and cut cases; diagnose units, initialization and metric misuse, including a changed-unit geyser question; produce a frozen held-out clustering report with a supplied changed-seed self-check and a weighted palette.',
    success: 'Calculations match declared ties, units and cut conventions; results distinguish optimization from usefulness; code is reproducible and explanations include representation and validation limits.'
  },
  misconceptions: ['Every unlabeled dataset has a unique natural clustering', 'K in KNN and k-means has the same role', 'Each Lloyd half-step strictly decreases error or convergence establishes the global optimum', 'K-means++ is deterministic farthest-first or always recovers the intended groups', 'Standardization is always the correct geometry', 'Ward merge cost equals SciPy dendrogram height squared', 'Every horizontal tree cut can attain every k despite tied heights', 'Lowest inertia or largest silhouette proves the correct number of groups', 'Cluster IDs are ordered classes or distant observations are automatically rejected', 'A faster approximation, a nonlinear projection or a biologically named cluster is automatically valid'],
  sources: ['https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html', 'https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf', 'https://scikit-learn.org/stable/modules/clustering.html'],
  depth: 'core',
  designRecord: 'docs/teaching/K-MEANS-HIERARCHICAL-LESSON-DESIGN.md',
  reviewFocus: 'Complete assignment/update and stopping contracts; duplicate and empty clusters; vanilla versus greedy seeding; metric/weight units; Ward Δ and sqrt(2Δ) height; tied cuts; label-invariant comparisons; current native outputs; calculated visual data; meaningful accessible topic-native diagrams; independently changed practice and qualified applications.'
};
