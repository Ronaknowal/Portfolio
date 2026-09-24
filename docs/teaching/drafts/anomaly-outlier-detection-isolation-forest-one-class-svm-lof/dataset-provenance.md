# Data for the anomaly-detection manuscript

This is a content-stage data packet, not a benchmark release or proof of prospective detection performance.

## Original source, license and pinned bytes

Repository: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB). Pinned commit: ea702d75cc2258d9d7dd35ca8e5e2539d71f3140. Files were fetched on 12 September 2026 local time; the machine-readable [source record](dataset-source.json) retains the UTC retrieval timestamp, exact URLs, byte lengths and hashes.

The pinned repository's license is **MIT**. Its full [license notice](NAB-LICENSE.txt) is supplied verbatim. The inspected source tree/data README did not identify a separate exception for this series. Do not replace this pinned license with an older tutorial's AGPL description. Attribution: Numenta's NAB dataset, original repository paths below; preprocessing and scoring are this lesson's derived work.

| Supplied file | Upstream relative path | SHA256 |
|---|---|---|
| [machine_temperature_system_failure.csv](machine_temperature_system_failure.csv) | data/realKnownCause/machine_temperature_system_failure.csv | 92bf5b87fc7f9bba8ca0b7ec63ccaac8cb4a1371a258e8c29a10ae9c018d82a4 |
| [nab-event-windows.json](nab-event-windows.json) | labels/combined_windows.json | 1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1 |
| [nab-source-readme.md](nab-source-readme.md) | data/README.md | 1d32963e44a52323b44affff374a1222ebe5ef19ef8f7f6c6fd7a747aa52db30 |
| [NAB-LICENSE.txt](NAB-LICENSE.txt) | LICENSE.txt | 0a0b4d0b10cb1f7ed9ab2993ef93defc03447e6eba9daca1315dd32dae4877e3 |

The annotation file is retained as downloaded, including unrelated dataset keys; only realKnownCause/machine_temperature_system_failure.csv is selected. Those other keys are not additional experiments or scope.

The CSV begins at 2013-12-02 21:15:00 and ends at 2014-02-19 15:25:00. It describes an industrial machine's internal component temperature. The inspected metadata do not specify temperature units or timestamp timezone: do not label the values Celsius/Fahrenheit or convert the timestamps to UTC by assumption.

## Observations, duplicates and causal features

Raw rows: 22,695. Unique timestamps: 22,683. Duplicate excess rows: 12. The upstream [duplicate-timestamp issue](https://github.com/numenta/NAB/issues/376) documents the issue; our counts come from the supplied bytes.

Our explicit policy is to average all measurements with the same timestamp, retaining the original raw file. We are not interpreting the row sequence as 22,695 unique evenly spaced instants. This aggregation is an illustrative measurement policy. Other defensible policies should be assessed using provenance, not silently substituted.

For each unique timestamp t, form level v(t) and change v(t)−v(t−one hour), using exact timestamp lookup. Missing lag values are dropped, not interpolated. Twelve rows are removed, leaving 22,671 feature rows. Temporal features use no future measurement. For actual online operation, same-timestamp records must be available before the observation closes; late arrivals would need a closing delay or revision policy. This packet does not establish that deployment condition.

Intervals:

* Reference: t before 2013-12-06, 885 rows.
* Calibration: 2013-12-06 through before 2013-12-10, 1,152 rows.
* Test/inspection: t from 2013-12-10, 20,634 rows.

The first interval is an assumed reference period, not a verified all-normal label. Scaling means and standard deviations are fitted on that interval only. Features are not tuned using annotation windows.

## What annotations mean here

Use inclusive endpoints from the pinned window file:

| Window | Start | End |
|---:|---|---|
| 1 | 2013-12-10 06:25:00 | 2013-12-12 05:35:00 |
| 2 | 2013-12-15 17:50:00 | 2013-12-17 17:00:00 |
| 3 | 2014-01-27 14:20:00 | 2014-01-29 13:30:00 |
| 4 | 2014-02-07 14:55:00 | 2014-02-09 14:05:00 |

The source's prose mentions a planned shutdown and failure-related behavior, while this label file has four windows. Do not invent a one-to-one mapping, precise fault onset or individual binary fault labels. There are 2,268 test timestamps inside these windows and 18,366 outside.

In the manuscript, a window hit means at least one row alert inside its interval. The row counts distinguish inside-window from outside-window alerts. Outside-window alerts are unmatched workload, not established false positives; inside-window alerts are not all known true positives. No point adjustment, NAB official scoring, incident collapse or prospective early-warning claim is made.

## Derived inputs for the future visual

[nab-derived-scores.csv](nab-derived-scores.csv) holds the exact feature rows and anomaly-oriented scores used for the proposed real-data visual. SHA256 ae1222f93dc6c5098a8e97c2e402c2ca4c59594cc9ed3f4e4cef02efebb80cc6. It has 22,671 rows and these columns:

* timestamp, level, one_hour_change;
* period: reference, calibration or test;
* baseline, isolation, one_class_svm, lof: anomaly-oriented scores;
* window: 0 outside annotations, otherwise 1–4.

Reference score fields are deliberately blank: those rows fitted the models and are not mixed into the new-query score distribution. Blank is missing, not zero. Baseline uses level only; learned detectors use standardized level and change. All parameters and the complete program appear in lesson.md.

The [visual-input calculation record](visual-input-calculation.json) binds the actual program, output and CSV hash. [prepare-temperature-visual-inputs.py](prepare-temperature-visual-inputs.py) reran the fixed manuscript program once to supply real row-level curves and a hidden-to-learner q=.975 fixture; aggregate counts alone would not justify drawing a score trace. This was a new data-preparation purpose, not a repeated formal verification campaign or hyperparameter search.

The earlier [author calculations](author-calculations.json) retain the original elementary fixture and fixed comparison outputs, with their own source hash. [manuscript calculations](manuscript-calculations.json) separately record four compact teaching programs and changed arithmetic. Their timestamps and scopes remain distinct.

Required future checks: original/derived hash match; exact chronological feature reconstruction; score direction; decimal parsing round-trip; quantile and strict-tie behavior; event endpoints; full-population counts despite plot downsampling; displayed code/output; responsive readable trace and keyboard inspection. None has yet been certified in a production implementation.
