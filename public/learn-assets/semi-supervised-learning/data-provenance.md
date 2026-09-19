# Banknote Authentication teaching subset

Source: Volker Lohweg (2012), Banknote Authentication, UCI Machine Learning Repository. DOI [10.24432/C55P57](https://doi.org/10.24432/C55P57). Dataset information and license inspected at [UCI](https://archive.ics.uci.edu/dataset/267/banknote+authentication) on 2026-09-12.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as specified by UCI. Credit Volker Lohweg and UCI, retain the source/DOI and license link, and state that this file is a reordered subset with source-row and split annotations. No endorsement is implied.

Downloaded archive: https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip

- Archive SHA256: 1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227.
- Member: data_banknote_authentication.txt.
- Original member SHA256: d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9.
- Retrieval date: 2026-09-12.
- Original rows: 1,372, with four numerical features and a binary class code. The source reports no missing values.

The source describes measurements extracted from wavelet-transformed images of banknote-like specimens. Features are variance, skewness, curtosis (source spelling) and entropy. Retain the numeric values without rounding changes. These are derived image features with no physical measurement unit supplied in the retained dataset description. This packet contains numerical measurements, not specimen images.

Transformation: form NumPy default_rng(23).permutation(1372), keep its first480 positions, and retain those source rows in that order. Add source_row as the original one-based text-file row. Rows0–319 in the retained file are pool,320–399 development,400–479 test. The CSV therefore has480 data rows plus its header. Preserve this order for the instructional programs.

The six seed labels are the first three occurrences of each class in the320-row pool, corresponding to zero-based pool indices[0,1,2,4,6,7] (the code concatenates class-specific lists; order is immaterial to the row mask). This deliberately uses an oracle to ensure both classes appear. Keep that policy explicit. It is not a claim that a random six-label sample is representative.

Training access: all320 pool feature vectors, only six pool labels. Development80 labels choose among predefined candidates. Test80 labels evaluate only the chosen candidate. Standardization fits only the320 pool inputs and is shared by all candidates. Development labels are not added to the final fit. The manuscript states the total evaluation-label cost separately.

The CSV retains every target for reproducibility. Code must pass only the six-label mask to fitting and pass no hidden truth to promotion decisions. The post-training wrong-pseudo-label counts are a retrospective benchmark audit. This audit must never become an oracle inside an interactive fitting rule or be described as an available signal on genuinely unlabeled data.

Source class semantics are preserved as codes0 and1. This packet does not infer which code corresponds to genuine or forged specimens.

Author-generated derivatives: checked-results.json contains exact split indices, seed source rows, scaler parameters, development predictions, selected final test predictions and pseudo-label traces. Those results are a single specified teaching experiment, not a general benchmark ranking.

The raw archive was read in memory for extraction and is not retained as redundant scratch. The small CSV and source receipt are required assets for prepared content and should remain until their supported implementation is integrated. Dataset source refresh is not needed merely because authoring continues; reopen the source only if a provenance/value question arises.
