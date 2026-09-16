# Offline digit data for the NMF lesson

E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, UCI Machine
Learning Repository, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49),
[source page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits),
inspected 12 September 2026. UCI specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain this attribution and the subset description with the CSV and any derived factor data; the
original collectors do not endorse this lesson.

The source question is handwritten digit recognition. The collection reduced normalized 32×32
handwritten bitmaps into nonoverlapping 4×4 blocks, counting foreground pixels in each. The 64
features therefore form 8×8 block-count images with values 0…16. This is not MNIST, a clinical
dataset or an unprocessed camera image.

`digits-300.csv` is the byte-identical established teaching subset: from scikit-learn 1.9.1
`load_digits()` (1,797 rows), take the first 30 rows of each digit label 0…9 and sort their loader
indices. Scikit-learn's own documentation, not the UCI record cited above, is what identifies those
1,797 rows as the collection's test-set portion; the UCI page reports the combined 5,620 instances. Labels therefore affect subset selection; the
early-row subset is not population-representative. SHA-256:
`d93f963c4b2610eb07122a312eec3ddceac18a031477370d71e33835eced728e`.

Columns: `source_row` is the original zero-based loader index, `digit` is a diagnostic target, then
`pixel_0`…`pixel_63` are row-major block counts. Only those 64 counts enter the factorization,
divided by 16; there is no centering, no learned per-feature scale and no imputation. The split uses
`train_test_split` with seed 19 and stratification: first 60/40%, then the remainder half and half,
giving 180 training, 60 validation and 60 test rows. Writer IDs are unavailable in this extract, so
this is a withheld-image split within one collection, not a new-writer evaluation.

Recorded fits use k = 1, 4, 8, 16 with seeds 7 and 19, random initialization, the coordinate-descent
solver, Frobenius loss, no penalty, `max_iter = 2000` and `tol = 1e-5`. The inspection case k = 8,
seed = 19 was declared for readable images, not selected by validation. PCA with eight components
and full SVD, plus the training-mean image, are measured on the same reserved rows. No elapsed time,
classification accuracy, biological identity or optimal component count is claimed. Random-seed
repeatability is tied to Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1.

Everything else in the lesson — the three-by-three matrices, the two exact factorizations, the
loss comparison table, the zero-lock case, the nonnegative-rank matrix, the three-band spectrum and
the five-document corpus — is an original constructed teaching fixture, not this real dataset.
