# ICA recording input and attribution

The file r01-first20s.csv contains data from the **Abdominal and Direct Fetal ECG Database, version 1.0.0**, made available at [PhysioNet](https://physionet.org/content/adfecgdb/1.0.0/) under the [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/). This derivative extract retains that license. Retrieved 12 September 2026.

Please credit the original researchers:

Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R. *Determination of the fetal heart rate from abdominal signals: evaluation of beat-to-beat accuracy in relation to the direct fetal electrocardiogram.* Biomedical Engineering/Biomedizinische Technik. 2012;57(5):383–394. [DOI:10.1515/bmt-2011-0130](https://doi.org/10.1515/bmt-2011-0130).

Dataset version DOI: [10.13026/C2RP4B](https://doi.org/10.13026/C2RP4B). PhysioNet's dataset page also requests its current platform citation: Pollard T and colleagues, *PhysioNet as a global platform for biomedical research*, Nature Health (2026), [DOI:10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z). The author read the dataset's citation and acquisition metadata; no clinical claim is inferred from the platform citation.

## What was collected

The database contains five-minute recordings from five women in labor, acquired at the Medical University of Silesia using the KOMPOREL system. Each record has four differential abdominal signals and a simultaneous direct fetal ECG reference. The research question concerned heartbeat measurement from abdominal recordings compared with that direct reference. The lesson uses only record r01, chosen before running the comparison, and makes no patient-level or clinical performance estimate.

Dataset metadata reports synchronous 1,000 Hz sampling, 16-bit resolution, a 1–150 Hz bandwidth and additional filtering for power-line interference and baseline drift. The EDF channel prefilter field says “LP:150Hz, CombF:50Hz”. The lesson preserves the provided signal; it does not perform additional filtering, resampling, alignment, interpolated sample insertion or missing-value replacement.

## Original file and extraction

- Original [r01.edf](https://physionet.org/files/adfecgdb/1.0.0/r01.edf), 3,061,792 bytes.
- SHA-256: 7549bbd378ea23851c20c0b7924f0a1f9fd909a3a3683c2334144a4c156dcb62, matching the publisher's [SHA256SUMS.txt](https://physionet.org/content/adfecgdb/1.0.0/SHA256SUMS.txt).
- EDF header: 1,792 bytes; 6 signal entries; 60 records; each record lasts 5 seconds. The first five channels have 5,000 samples per record; the EDF annotation channel has 500 16-bit samples per record.
- Original channel order: Direct_1, Abdomen_1, Abdomen_2, Abdomen_3, Abdomen_4, EDF Annotations.
- The extract retains the first four complete data records, all samples from the first five signals, in chronological row order. It omits the annotation channel and subsequent 280 seconds. No annotation-derived outcome is evaluated in this lesson.
- Retained CSV: 20,000 data rows × 5 numeric columns, plus one header; 390,723 bytes; UTF-8/ASCII header, decimal integers, comma-delimited, LF lines.
- CSV SHA-256: 7c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc.

Each numeric row is one simultaneous instant, with sample index i starting at zero and time i/1000 seconds. First sample is t=0; last is t=19.999 seconds. Columns:

| Column | Meaning | Stored unit | Use in lesson |
| --- | --- | --- | --- |
| direct_adc | Direct fetal ECG channel Direct_1 | signed ADC counts | reference only |
| abdomen1_adc | Abdomen_1 | signed ADC counts | decomposition input |
| abdomen2_adc | Abdomen_2 | signed ADC counts | decomposition input |
| abdomen3_adc | Abdomen_3 | signed ADC counts | decomposition input |
| abdomen4_adc | Abdomen_4 | signed ADC counts | decomposition input |

All five header calibrations are the same: digital minimum −32768, digital maximum 32767, physical minimum −3276.8, physical maximum 3276.8, physical unit uV. Convert each digital value d by

$$
p=(d-(-32768))\frac{3276.8-(-3276.8)}{32767-(-32768)}-3276.8
  =(d+32768)\frac{6553.6}{65535}-3276.8.
$$

The small offset follows the recorded endpoint calibration; replacing it with an assumed exact 0.1 µV/count is not the specified conversion. Pearson correlation is invariant to these positive affine calibrations, but trace units and reconstruction still use the actual header.

The retained [author-calculations.py](author-calculations.py) reconstructs the CSV with its --extract option from a hash-checked EDF at scratch/ica-content/r01.edf. That optional source file can be downloaded from the exact URL above if regeneration is needed; the lesson and author probes normally read the provided CSV and work offline. Data extraction uses NumPy little-endian signed 16-bit reading, record/channel/sample reshaping, no specialist EDF dependency. The source EDF was downloaded and its header inspected; the published hash was independently compared before extraction.

## Fixed comparison protocol and interpretation

Training is samples 0:12000; development 12000:16000; test 16000:20000, all half-open. Only four abdominal channels enter PCA and ICA fitting. The direct reference is first used to choose the coordinate with largest absolute development correlation separately for raw channels, PCA and ICA. Those choices remain fixed at test. Means/whitening are fitted to training samples; transform uses the fitted model.

ICA: 4 components, parallel algorithm, logcosh, unit-variance whitening, SVD whiten solver, seed 7, tolerance 1e−5, maximum 1,000 iterations. PCA: 4 components, full SVD. No scaling sweep, seed search, filter tuning or interval search was performed to improve the result. Every method has four candidate coordinates. The selected coordinate may differ from its one-based label under an equivalent sign/permutation convention in another implementation; retain the actual selected score and reference comparison.

The test diagnostic is absolute Pearson correlation to a direct waveform, not physiological-source recovery, beat-detection accuracy or a clinical endpoint. The finite 4-second interval and shared participant permit a narrow within-recording comparison. The provider's preprocessing is offline; this is not a demonstrated real-time causal signal-processing pipeline. Temporal autocorrelation, source dependence, reference morphology and nonstationarity are relevant limits.

Author calculation snapshot: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1. Development/test absolute correlation: raw channel 3 = .201203/.119806; PCA coordinate 4 = .184580/.450178; ICA coordinate 2 = .169804/.343966; ICA 14 iterations. Exact details and phase boundaries are in [the design record](../../ICA-LESSON-DESIGN.md).

## Publication handoff

During phase two, offer the CSV with this attribution/calibration alongside it. Preserve the source URL, dataset version, license link, authors and modification statement. If embedding a compact data module, derive it from this same CSV and verify byte/numeric agreement; do not hand-copy a selected subset to make a favorable plot. No production data/module/download location has been written by this content-only task.
