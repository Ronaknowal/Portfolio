export default {
  summary: 'Collect evidence about a proposed grouping, name the population, representation and comparison rule behind every score, and decide whether that evidence supports the intended use.',
  outcomes: [
    'Compute a silhouette from actual distances, explain a negative value geometrically, and read a sorted silhouette plot as a distribution with named weighting',
    'Show that a relabeling leaves a partition unchanged, count together and apart pairs through a contingency table, and derive the Rand and adjusted Rand indices with their fixed-margin chance baseline',
    'Compute entropy, mutual information, arithmetic NMI and AMI, and explain what a finite fixed-margin null does and does not remove',
    'Diagnose which reference groups a candidate splits or merges using purity, homogeneity, completeness and Fowlkes–Mallows, and state what each omits',
    'Separate rescoring a frozen partition under a changed geometry from refitting, and report the disagreement between silhouette and species agreement on real Iris data',
    'Report coverage and conditional scores when observations are rejected, and compare methods on a named common population',
    'Design a stability experiment that keeps observation IDs aligned, and write a frozen fit/selection/report protocol with a baseline'
  ],
  prerequisites: ['K-Means & Hierarchical Clustering', 'PCA & Dimensionality Reduction'],
  sequence: [
    'Ask which result to keep when two groups score cleaner and three match species better',
    'Fix observation identity before comparing groupings',
    'Compute one silhouette from its distances, then read the distribution',
    'Recognize that a score evaluates a geometry as well as a partition',
    'Count pairs, correct for fixed-margin chance, derive ARI',
    'Measure shared information; enumerate the exact chance experiment',
    'Diagnose splits and merges; connect purity, homogeneity, completeness and VI',
    'Reproduce the real Iris disagreement and rescore a frozen partition',
    'Handle rejected observations; design stability; freeze a report protocol; practise changed cases'
  ],
  visual: {
    type: 'Identity strips across relabelings; distance fan and genuine silhouette bars; ring-versus-slice contrast; information strips; real Iris snapshot and workspace; rejection lanes with coverage; weighted probe ruler; fit/selection/report flow',
    question: 'Did the memberships change or only the names? Which averages set a and b? What is the chance baseline? Which population is behind this score? Predict before applying each edit.',
    interaction: 'Edit coordinates and memberships, relabel or refine a candidate partition, construct two binary labelings and enumerate their null, rescore a frozen Iris partition with feature weights, change training multiplicities on fixed probes; every lab records a prediction and compares it with the calculation.'
  },
  practice: {
    task: 'Compute a foreign-group minimum correctly; derive RI, ARI and NMI for crossed and singleton partitions; weigh a small group; design a fair rejection comparison; repair a misaligned stability script; solve a changed exact split and its scaled version; plan a new-context evaluation.',
    success: 'Exact values match (s = 0.5, ARI −1/6, NMI 0.5, means 0.68 versus 0.2, centers 1 and 9 with cost 4 then 3 and 27 with cost 36); explanations name the population, geometry, null and weighting behind each number.'
  },
  misconceptions: [
    'A high silhouette or a fixed threshold certifies the right number of groups',
    'Different label numbers mean a different partition',
    'The Rand index of a random partition is zero',
    'NMI equal to one requires identical partitions under every normalizer',
    'AMI is always preferable to NMI',
    'A negative silhouette flags a misclassified observation',
    'Purity rewards the right thing when a candidate fragments the reference',
    'A cleaner conditional score after rejecting cases is a like-for-like improvement',
    'Comparing bootstrap label arrays by position compares the same observations',
    'Selecting a pipeline with the final report leaves the report unbiased'
  ],
  sources: [
    'https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation',
    'https://jmlr.org/papers/volume11/vinh10a/vinh10a.pdf',
    'https://arxiv.org/pdf/1007.1075',
    'https://archive.ics.uci.edu/dataset/53/iris',
    'https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap17.pdf'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/CLUSTERING-EVALUATION-LESSON-DESIGN.md',
  reviewFocus: 'b as a minimum of group averages; singleton and undefined conventions; RI expectation under fixed margins; arithmetic versus geometric NMI and constant-labeling conventions; exact 70-assignment null; eight Iris fits with n_init 20 seed 17; frozen-weight rescoring leaves ARI/AMI fixed; coverage denominators; exact contiguous split solver ties; 90/30/30 report fixture.'
};
