# Input provenance and observed outputs

The included digits-400.csv contains real UCI Optical Recognition of Handwritten Digits samples, E. Alpaydin and C. Kaynak(1998), DOI[10.24432/C50P49](https://doi.org/10.24432/C50P49), [UCI source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), CC BY4.0. Retain attribution when redistributing rows. This is not MNIST.

It is a byte-identical copy of the Perceptrons packet's asset, SHA256 a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672. The previously inspected extraction used sklearn load_digits, the1797-row copy of UCI's historical test partition, selecting the first40 rows per digit0–9 and concatenating by digit. source_id is the original one-based row; pixel_0 through pixel_63 are row-major8×8 integers0–16; digit is the original class.

The local teaching split repartitions these400 specimens into280 training and120 validation rows using original-digit stratification and seed22. All source IDs are retained in JSON; pixels are divided by16. This is neither the official historical UCI test protocol nor a writer-independent benchmark, and no untouched final test was used.

Actual author execution on2026-09-12 used Python3.12.14, NumPy2.3.5, scikit-learn1.9.1 and torch2.14.0+cpu, with one CPU thread. Twelve matched runs use seeds1/2/3, no norm/BN/LN/RMS, a64→32→tanh→10 network, epsilon1e-5, identity norm affine initialization, default torch Linear initialization, SGD.1, fifty epochs, batches28, and shuffle seed1000+model seed. Initial weights and ordering match within each seed. No individual variant tuning was performed.

Recorded losses and correct counts at epochs0/1/5/20/50 are actual observations. Evaluation mode is used for all reported metrics. No individual prediction arrays, timings, or universal architecture ranking are claimed. Mechanism arrays are separate synthetic mathematical fixtures, not handwriting activations. State updates, forward/gradient comparisons and the affine update are actual author calculations. The separate low-precision promotion example was also executed; its result is a small arithmetic check, not a precision benchmark.
