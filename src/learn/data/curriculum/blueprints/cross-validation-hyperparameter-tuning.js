export default {
  summary: 'Give every observation a turn as held-out evidence, use that evidence to choose a learning procedure, and keep the score that made a choice separate from the score that assesses it.',
  outcomes: [
    'Name which of three questions a reported score answers: a fitted object, a learning recipe at a stated training size, or the whole select-and-refit process',
    'Build a fold plan in which every row is assessed exactly once, with no dropped remainder and a declared tie rule',
    'Place every learned imputer, scaler and encoder inside the fold that fitted it, refitting the whole procedure per fold',
    'Distinguish an unweighted fold mean from a pooled row accuracy, and state the unit being weighted',
    'Choose a split from a concrete future use: unseen groups, later records from known groups, a forward horizon with late-maturing labels',
    'Explain why a plan that is valid for scoring can still fail the exactly-once contract a one-prediction-per-row table needs',
    'Show selection raising a validation score without any predictive signal, and show a duplicate candidate raising nothing',
    'Run nested cross-validation, and report the outer result as an assessment of the procedure at the outer training size rather than of an oracle-best setting',
    'Separate sampling mass under a search distribution from proximity to an optimal score',
    'Refute the claim that overlapping training sets determine the correlation of held-out losses',
    'Compute expected improvement and distinguish it from expected loss',
    'Account for a halving schedule exactly, and separate what it paid to observe from hindsight'
  ],
  prerequisites: ['Feature Scaling, Encoding & Imputation'],
  sequence: [
    'Three different questions that all sound like "how good is the model?"; parameters against hyperparameters; roles as uses, not properties of rows',
    'A fold is a role assignment: a fresh copy of the whole procedure per fold, a remainder that is never dropped, and one held-out prediction per row',
    'Fold mean against pooled accuracy, and the identity that connects them by fold size',
    'Splits that match the future use: independence and rare classes, unseen groups against later records, forward horizons with label-availability dates',
    'The exactly-once contract a one-prediction-per-row table requires, and what a forward plan legitimately cannot supply',
    'Selection on a criterion with no signal: four fair bits, an exhaustive sixteen-pattern enumeration, and the duplicate-candidate null',
    'Two valid separations: development plus a reserved test, or nested cross-validation with a refit inside each outer fold',
    'Grid against random search: conditional branches, log-uniform distributions, coverage seen from one coordinate, and the hit-probability calculation',
    'A complete nested experiment on 344 real penguins: six candidates, three outer folds, three inner folds, a declared tie rule and a majority baseline',
    'Deeper: training size inside the estimand, exact mean-predictor risks, and why overlap is not a variance formula',
    'Deeper: surrogates and expected improvement, what TPE actually models, successive halving and early ranking',
    'Budgeting a comparison honestly: exact fit counts, parallelism limits, close candidates and visible failures'
  ],
  visual: {
    type: 'A two-loop data-flow diagram with the assessment bundle outside both loops; two patient-record panels for the group and time questions plus a label-availability timeline and a held-out-coverage contrast; a five-stage nested expansion drawn with the real experiment\'s row IDs; a grid-against-random coverage projection beside the analytic hit curve; conditional and expected error diagrams with exact mean-predictor curves, a covariance grid and a no-arrow counterexample; discrete probability masses with a shaded improvement region and a TPE density-ratio branch',
    question: "Which rows was this fit allowed to use, which score chose the setting, which score assessed it, and what did the schedule actually pay to observe?",
    interaction: "Edit features, labels and fold assignments and inspect the held-out model outputs immediately; change fair-label cases or candidate rules and follow selected validation accuracy and fresh-label performance; change a nested fixture and trace which training labels can influence the selected neighbour count and protected-row model output; edit loss trajectories, budget and survival factor and inspect the paid-for schedule. Invalid configurations display errors; hindsight remains an explicitly identified diagnostic outside the actual schedule."
  },
  practice: {
    task: 'Ten changed tasks: unequal-fold weighting, a dropped remainder, a group-versus-time boundary pair, an impossible out-of-fold array, a new candidate pattern against a duplicate, an epoch chosen by the assessment set, required draws for a stated sampling mass, an acquisition value that does not move, a full halving ledger from scratch and on resume, and the variance of a fixed rule under leave-one-out.',
    success: 'Every row keeps a validation turn; the unit being weighted is stated; a split is justified by a named future use; a selection score is never quoted as assessment evidence; sampling mass is never reported as score proximity; and a hindsight comparison is labelled as information the decision did not have.'
  },
  misconceptions: [
    'A selected validation score estimates the selected setting\'s performance',
    'Ten folds is the correct number of folds',
    'Leave-one-out always has the highest variance because its training sets overlap most',
    'The standard deviation of fold scores divided by the square root of K is a valid standard error',
    'Overlapping fold intervals are a significance test',
    'Stratification corrects the uncertainty caused by a rare class',
    'A timestamp column means the evaluation must be forward in time',
    'Naming a splitter GroupKFold is the same as writing the required boundary',
    'Any valid evaluation plan can produce a cross_val_predict table',
    'Nested cross-validation gives an unbiased estimate of the final all-data model',
    'The most frequently selected outer-fold setting should be deployed',
    'Sixty random draws reach within 5% of the best achievable score',
    'Random search always beats grid search',
    'A bootstrap sample contains all n distinct rows',
    'Every bootstrap score should be multiplied by .632',
    'Monotonically improving loss curves make early elimination safe',
    'A halving finalist always receives the full resource'
  ],
  sources: [
    'https://scikit-learn.org/stable/modules/cross_validation.html',
    'https://scikit-learn.org/stable/modules/grid_search.html',
    'https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf',
    'https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf',
    'https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf',
    'https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf',
    'https://www.jmlr.org/papers/volume18/16-558/16-558.pdf',
    'https://allisonhorst.github.io/palmerpenguins/'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/drafts/cross-validation-hyperparameter-tuning/design.md',
  reviewFocus: 'The selection/assessment boundary everywhere a number is quoted; the seven-row trace (predictions [1,1,1], [0,1], [1,1]; 0/3, 1/2, 2/2; fold mean .5 against pooled 3/7 = 0.42857142857142855); the exhaustive sixteen-pattern enumeration giving .6875 against a future accuracy of .5, and the duplicate-candidate null; the sixteen-row nested fixture, where relabelling row 3 moves the inner means to .5 and .625 and the selected count to 3 while the protected predictions stay [0,0,0,0,0,1,1,1] at 7/8, relabelling row 2 changes only correctness, and relabelling row 7 changes the fitting path in one outer fold and only the correctness in the other; the real experiment\'s selections 11/Standard, 3/Standard and 3/Robust with 114/115, 114/115 and 112/114 against baselines 51, 51 and 50, fold mean 0.9883549 against pooled 340/344 = 0.9883721, the five-way tie in outer fold 3 resolved by first enumeration, and the final all-data selection score 0.9913043 kept as selection evidence; hit probabilities 0.9539302 and 0.4528434 at 60 draws and 149 draws for mass .02 at 95%; expected new loss 4.5 at m = 8 against 4.3333 at n = 12 with optimism 2σ²/n; EI .02 and .075 against mean losses .18 and .25; the halving ledger 270 nominal, 810 full allocation and 210 on genuine resume; and that the Optuna block is presented as one executed run in a pinned environment rather than a property of TPE.'
};
