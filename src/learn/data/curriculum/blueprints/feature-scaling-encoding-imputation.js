export default {
  summary: 'Turn a mixed table into model coordinates on purpose: choose a ruler and watch the nearest neighbour change, give categories a geometry instead of an invented order, tell an absent measurement apart from a measured zero, and fit every transformation on the training rows alone before applying it frozen to held-out records.',
  outcomes: [
    'Show that a chosen divisor, not the data, decided which of two candidates was nearer, and read each feature’s contribution to the squared distance',
    'Fit standard, min–max and robust rulers on one training column and apply each to a later value outside the fitted range without clipping it',
    'Separate centring from scaling: a shared centre cancels in a difference, while the divisor changes the relative feature weights',
    'State what standardization does and does not do, including that it does not make a skewed feature Gaussian',
    'Read the chosen geometry of a one-hot block, and say what dropping a reference column does to distances and to an unknown value',
    'Keep missing, unknown and rare apart as three different states with three different policies',
    'Compute a nan-aware donor distance with its m/q overlap adjustment, and decide which incomplete rows may donate',
    'Name the MCAR/MAR/MNAR distinction as a claim about a collection process that no imputer can establish from a blank cell',
    'Hold the fit/transform boundary: transform an edited held-out record without moving a single fitted statistic',
    'Trace one real record through a fitted imputer, scaler and encoder to its seven named output coordinates',
    'Explain how a row’s own target can reach its own target encoding, and how fold-specific sums, counts and priors stop it',
    'Distinguish one completed dataset from multiple imputation, and pool analysis estimates rather than filled tables'
  ],
  prerequisites: ['Non-Negative Matrix Factorization (NMF)'],
  sequence: [
    'Scaling, encoding and imputation answer three different questions about one row',
    'A divisor is a weight: the nearest neighbour flips when the ruler changes, and a common rescaling changes nothing',
    'Fit a ruler from training data; an outlier makes standard, min–max and robust visibly different',
    'Column scaling is not row normalization, and a tree cares much less about either',
    'One-hot coordinates as a chosen geometry; the reference column, the unknown value and the real ordinal case',
    'A missing value is a question: a transparent estimate, a missingness indicator, and the mechanism behind the gap',
    'Nearest-neighbour imputation with incomplete donors, overlap distances and an explicit fallback',
    'Fit learns; transform applies. Split first, and never let a later row redefine the ruler',
    'A complete offline experiment on 344 penguin observations: four rulers, one split, one baseline',
    'Deeper: logs, Box–Cox and Yeo–Johnson, rank versus magnitude, bins, polynomials and circular features',
    'Deeper: smoothed target encoding, cross-fitting including the prior, and signed hashing with collisions',
    'Deeper: completing data is not representing uncertainty; Rubin’s pooling rule'
  ],
  visual: {
    type: 'One record forked into numeric, categorical and absent branches with the target on its own rail; five identified observations on three fitted rulers with real ticks and a magnified inset; a category simplex beside a true equal-aspect reference plane; a fitting-boundary diagram with a crossed backward arrow; the actual fitted bundle and one held-out row through it; a zero-origin count axis for five preparations; linked source/rank/log axes plus a labelled unit circle; a donor-dependency graph separating numerator edges from prior edges; and three symbolic completions feeding within/between variance bars',
    question: "Which candidate is nearer once you choose the divisors, which incomplete rows may donate a missing measurement, what does one held-out record become under the frozen fitted bundle, and which target can reach which encoded row?",
    interaction: "Edit query coordinates and divisors and inspect distance contributions live; clear donor measurements, change target column and neighbour count and inspect eligible donors and imputation; select or edit a held-out record and inspect its transformation under the frozen training-fitted bundle; change categories, targets, folds or smoothing and follow the donor graph without crossing held-out-label boundaries."
  },
  practice: {
    task: 'Nine changed tasks: a neighbour that flips under new divisors, a ruler fitted once and applied to a later value, what L2 normalization discarded, an unknown category against a fitted vocabulary, two separate donor edits with one null, a real held-out record with its mass deleted, a target-encoding edit that moves another row through the prior alone, the Yeo–Johnson identity at λ = 1 on both signs, and Rubin pooling of four analyses.',
    success: 'Every divisor sits inside the square; fitted statistics come from training rows only and are never refitted on a later batch; an imputed cell is named as an estimate rather than a measurement; an unknown category is distinguished from an absent one; a cross-fitted value never depends on its own row’s target, including through the prior; and pooled variance adds a between-completion term rather than averaging tables.'
  },
  misconceptions: [
    'Standardization makes a feature Gaussian',
    'Standardization establishes the assumptions behind a regression confidence interval',
    'Centring changes the Euclidean distance between two rows',
    'A constant training column must be divided by its zero standard deviation, or dropped',
    'Min–max scaling guarantees that future values fall inside [0, 1]',
    'Robust scaling removes outliers',
    'Row normalization is a kind of column scaling',
    'Integer codes are a harmless representation for unordered categories',
    'Dropping the first category is always the right one-hot setting',
    'An all-zero unknown block is a neutral representation',
    'A missing value and an unknown category are the same state',
    'An imputed value is a recovered measurement',
    'A missingness indicator repairs MNAR',
    'A nearest-neighbour imputer needs complete donor rows',
    'Fitting the scaler on all rows before splitting is harmless because no labels were read',
    'A pipeline makes leakage impossible',
    'A quantile transform is invertible because it preserves order',
    'Yeo–Johnson applies a square root to negative inputs',
    'Smoothing alone makes target encoding safe',
    'Running an imputer with several random seeds is multiple imputation'
  ],
  sources: [
    'https://scikit-learn.org/stable/modules/preprocessing.html',
    'https://scikit-learn.org/stable/modules/impute.html',
    'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html',
    'https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html',
    'https://allisonhorst.github.io/palmerpenguins/',
    'https://stefvanbuuren.name/fimd/sec-MCAR.html',
    'https://amices.org/mice/reference/pool.html',
    'https://arxiv.org/pdf/0902.2206'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/drafts/feature-scaling-encoding-imputation/design.md',
  reviewFocus: 'The raw-versus-scaled totals 10,001/10 and 2/9.0001 with the 4,400 g contrast (17) and the doubled-divisor null (0.5 / 2.250025); the five-value fixture rows −0.5383/0/−1 through 1.9993/1/48.5 and the later 150 at 3.2810/1.5051/73.5; √2 in the full one-hot block against 1 and √2 after dropping red; the overlap distances 7.5, 3 and 12 giving 200, with 300 and 220 under the two edits and the labelled column-mean fallback 300; the fitted medians [45, 17.3, 197, 4000], means [43.9306, 17.0992, 200.6589, 4190.9884] and scales [5.4000, 1.9387, 14.2307, 806.6876], all fitted on 258 training rows; source row 309 becoming [1.3091, 0.8773, 0.1645, −0.1128, 0, 1, 0] and its mass-deleted variant −0.2368; the correct counts 38, 67, 84, 85, 85 out of 86 with no winner declared; the cross-fitted array [5/9, 7/9, 2/9, 4/9, 2/9, 7/9] and its changed-target counterpart [5/9, 2/9, 2/9, 2/9, 2/9, 5/9]; and the pooled T = 16/3 with standard error 2.309 against 13/3 in practice 9.'
};
