export default {
  summary: 'Rank observations worth investigating under a declared reference population, explain the comparison that ranked them, and turn a score into a threshold, a workload and an honest report.',
  outcomes: [
    'State the unit, the context, the reference population and the intended response before choosing any detector',
    'Keep fitting, scoring and thresholding as separate decisions taken from separate periods, with annotations reaching only the evaluation',
    'Integrate first-cut intervals exactly, correct a truncated isolation path with c(m) and normalise by the sample actually fitted',
    'Apply the reachability floor to the neighbour that owns it and explain a farther query taking the lower local outlier factor',
    'Tell a LOF training factor from a new-query score at the same coordinate and choose the fitting mode the task requires',
    'Derive the two-anchor kernel boundary, put its midpoint outside the region, and read the nu bounds as training statements only',
    'Convert prevalence and two conditional rates into an expected review queue and an undefined-when-empty precision',
    'Choose a calibration quantile on a real series and report unmatched workload, window hits and what the annotations do not establish'
  ],
  prerequisites: ['DBSCAN & Density-Based Clustering'],
  sequence: [
    'Name the observation, its context and the response; separate point, contextual and collective cases',
    'Split reference, calibration and later periods; fix one score orientation and a strict threshold',
    'Isolation Forest: first-cut intervals, the leaf correction, the exact five-position expectations',
    'LOF: radius, reachability floor, reciprocal mean reach and the dimensionless ratio',
    'Fitting mode as mathematics: a training row excludes its identity, a new query does not',
    'One-Class SVM: RBF similarity, the weighted sum and rho, a disconnected accepted region',
    'Match mechanism to task; four applications where representation decides what is visible',
    'Decision policy: population arithmetic, tie-limited percentiles, rows versus events',
    'Deeper: primal, dual and the nu bounds on strict violators and support vectors',
    'Real monitoring: 22,671 NAB temperature rows, a fixed protocol, two predeclared quantiles',
    'Deeper: LOF local bounds, resource costs and the rules for updating a reference'
  ],
  visual: {
    type: 'Provenance lanes with a struck-out arrow from later rows back to the scaler; first-cut interval bars over five positions; two-group reach diagram with per-row radii; tie ruler on sorted calibration scores; isolation tree strip; neighbourhood number lines; kernel curves with the rho line and signed difference; population bars; a 480-bin temperature overview with window bands, threshold and alert marks',
    question: "Which position do random cuts separate first, whose radius sets each floor, is the midpoint inside the accepted region, what fraction of alerts are faults, and what workload does this threshold create on a real series?",
    interaction: "Edit positions, depth cap, reference rows, k, query, gamma, anchor, prevalence, sensitivity, false-positive rate and review budget to follow paths, densities, scores and workloads live. On the real series change method, calibration quantile or threshold and inspect the synchronized timeline, annotated windows and review counts; keep fit, calibration and held-out time roles intact."
  },
  practice: {
    task: 'Ten changed tasks: a different isolation gap, a truncated path with the wrong normaliser, a rescaled reach calculation, the invalid training-versus-query comparison, widened kernel anchors, a 200-slot review budget, a training bound mistaken for a test promise, an honest temperature recommendation, an event metric hiding repeated work, and a stuck sensor needing a causal feature.',
    success: 'Every probability is a length ratio over the stated span; every normaliser names the sample actually fitted; floors are attributed to the neighbour; precision is computed from separate fault and non-fault populations; real-data claims name the split, the settings, the quantile, the unmatched workload and what the annotations do not establish.'
  },
  misconceptions: [
    'An anomaly score is a probability of failure',
    'A flagged observation is a fault',
    'A percentile threshold flags exactly that fraction of later rows',
    'contamination or nu discovers the true fraction of anomalies',
    'LOF above 1 means anomalous and exactly 1 means normal',
    'The reachability floor belongs to the query rather than to the neighbour',
    'Scoring the training array as queries recovers the training LOF factors',
    'A One-Class SVM accepted region is one connected blob',
    'The nu bounds promise a future false-alert rate',
    'Hitting all four annotation windows means the detector works',
    'Outside-window alerts are verified false positives',
    'A requested max_samples is the normaliser even when fewer rows were fitted'
  ],
  sources: [
    'https://arindam.cs.illinois.edu/papers/09/anomaly.pdf',
    'https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf',
    'https://sigmodrecord.org/publications/sigmodRecord/0006/pdfs/LOF_%20Identifying%20Density-Based%20Local%20Outliers.pdf',
    'https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf',
    'https://scikit-learn.org/stable/modules/outlier_detection.html',
    'https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/ANOMALY-DETECTION-LESSON-DESIGN.md',
  reviewFocus: 'Strict greater-than thresholds and tie behaviour; c(m) with the fitted sample size; exact five-position expectations at depth cap 3; the neighbour-owned reachability floor and the 35/24 versus 35/32 contrast; training factor 4/3 against new-query 7/8; rho and the disconnected positive region at gamma 1; precision undefined when nothing is flagged; the 885 / 1,152 / 20,634 split, 2,268 inside and 18,366 outside, and the eight published quantile rows; annotations never used to select a setting.'
};
