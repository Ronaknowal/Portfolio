# Neural ODE data and calculation provenance

Prepared 13 September 2026. These inputs support a content-first manuscript and future implementation.

## Real measurements

Source: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), Fisher, R. (1936), DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source page's four centimeter measurements, 150 rows, 50 examples per class, correction notes and license were read this session.

The CSV is an exact byte copy of the corrected snapshot already retained in the GMM packet, originally produced from scikit-learn 1.9.1's Iris data. See its [inherited provenance](../gaussian-mixture-models-gmm-em-algorithm/data-provenance.md). It is not a newly downloaded physiological time series.

Local iris.csv SHA256: c6fd24e7f41dd55405cbc30f344e648c2f31eedfb24f9dcaba6290998bc26eb7. All 150 source rows and one-based IDs are retained. Corrections match UCI's documented 35th row [4.9,3.1,1.5,0.2] and 38th row [4.9,3.6,1.4,0.1]. Features are sepal_length_cm, sepal_width_cm, petal_length_cm and petal_width_cm. Classes are setosa, versicolor and virginica.

The author groups identical full feature vectors and verifies consistent labels. Rows 102 and 143 duplicate; score only their first representative, leaving 149 unique vectors. NumPy default_rng(926) permutes unique IDs separately per class: first 30 fit, next 10 validation, remainder assessment. Exact 90/30/29 IDs are in study-results.data.roles; assessment class counts are 10/10/9. The split/seed declaration precedes fitting in design.md.

Fit-only mean: [5.89,3.045555555555556,3.793333333333333,1.2177777777777776]. Population SD: [0.9057900173636023,0.4392319334452182,1.8423897524682449,0.7836634450207108]. No source sample is a temporal trajectory; integration time is representation depth.

## Actual executed study

Environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, SciPy 1.18.1, one CPU thread and torch float64.

Four models with 15/671/179/251 parameters were fitted for seeds 13/37/61: 12 fits, 300 full 90-example updates each, AdamW learning rate 0.01, weight decay 0.001 and default betas 0.9/0.999. Validate at update one and every 25 updates; select minimum validation cross-entropy per run, then assess selected weights. All nonlinear models selected update 50; linear models selected update 300. No post-assessment tuning or discarded seeds. Full curves and selected weights are retained. The 29-example assessment and unequal capacities do not support broad architecture-superiority claims.

From repository root:

~~~text
scratch/lesson-tools/Scripts/python.exe -B docs/teaching/drafts/neural-ode-continuous-depth-models/neural_ode_study.py
scratch/lesson-tools/Scripts/python.exe -B docs/teaching/drafts/neural-ode-continuous-depth-models/ode_calculations.py
~~~

On another machine, use Python with NumPy, PyTorch and SciPy to run those same scripts in the packet directory. The first reproduces 12 fits and writes study-results.json/fitted-models.json. The second loads saved weights without retraining and writes calculated-inputs.json. Selected model states support inference, not optimizer resumption.

## Synthetic and numerical inputs

Rotation, scalar gradients, adaptive Heun, stiffness, augmentation, observation jumps, linear CNF traces and flow-matching pairs are explicitly hand-defined mathematical fixtures. They are not real measurements or trained application results. Matrix exponentials and analytical solutions supply reference answers; native autograd and central differences check fixed-solver gradients. SciPy RK45/Radau results are executed version-specific evaluation/Jacobian/factorization counts, not wall-clock timings.

The real-model diagnostics load full seed-37 ODE/augmented weights and use validation IDs 64/70/109, petal-length changes 0/+0.6/−0.6 cm, and RK4 with 4/16/64 steps or Euler with four steps. Independent NumPy full-state/logit inference differs from PyTorch by at most 8.881784197001252e−16. Input derivatives agree with central differences within 2e−8. These checks are author evidence, not browser evidence or formal independent review.

## Retention and implementation

Keep the CSV, actual fits, complete programs, fixtures, manuscript, specifications and design. Phase two extracts only necessary runtime arrays with hashes and provenance; do not ship the entire author packet eagerly. Preserve attribution in downloads and near real-data figures. No disposable screenshot or downloaded media was created for this topic, and no unrelated scratch cleanup is needed.
