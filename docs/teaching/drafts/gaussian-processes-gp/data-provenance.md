# NOAA monthly CO₂ subset

- Collector/source: NOAA Global Monitoring Laboratory, Carbon Cycle Greenhouse Gases group, Mauna Loa monthly average atmospheric CO₂.
- Source CSV: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
- Data explanation: https://gml.noaa.gov/ccgg/trends/data.html
- Public-domain and attribution conditions read: https://gml.noaa.gov/about/disclaimer.html
- Retrieved 2026-09-12. Source header file creation 2026-09-05. Full downloaded-source SHA256 dcf0198658c87ebb5e3b2ce807fe3a52e1d5f000aa0479fb1bf334afb7adde4d; machine-readable record in data-source.json.
- Retained 120 rows, January 1990 through December 1999 inclusive. Fields: year, month, decimal_date, co2_ppm. The last is source average; decimal_date preserves source decimal date. No synthetic data inserted, measurements altered, smoothing or labels added. NOAA's monthly averages are an already-aggregated scientific product.
- Source explains monthly means are derived from daily means and adjusted to the center of the month to address asymmetric missing days. Source marks interpolated NOAA months with negative sdev/uncertainty. Author inspected those source fields for this range: **zero flagged months**. Fields were checked before discarding non-used spread/count columns. No original raw daily data are claimed.
- Source includes Scripps observations for 1958–1974; the chosen range is later NOAA data. The 2022–2023 alternate-site interval is outside this extract.
- Learning program time input (year−1990)+(month−.5)/12 is a constructed evenly spaced monthly coordinate. It does not reuse source decimal date or pretend each observation occurred at that exact instant.
- Noise variance .09 ppm² is the lesson's modeling assumption, not supplied NOAA uncertainty. Tiny pipe examples and kernel draws are constructed mathematical examples, separate from measured CO₂.
- Offline execution: complete manuscript programs use local arrays or this CSV; no runtime data downloads. Phase2 can package the program and CSV together, maintaining this provenance.
- Attribution: NOAA Global Monitoring Laboratory, Boulder, Colorado, Carbon Cycle Greenhouse Gases group, source link and retrieval date. The public-domain notice permits US government material use absent another annotation; do not imply NOAA endorses instructional models or that derived forecasts are official NOAA products. No contact/email transmission was made.

Fixed evaluation: 1990–1995 train, 1996–1997 development, 1998–1999 final test. Family selected by development MAE, then initialized/refitted on 96 observations using the same predeclared family. Seasonal-naive repeats 1997's monthly profile into both future years. It does not use observed 1998 outcomes for 1999. Full predictions, uncertainty, fitted hyperparameters, source subset and author calculations are retained.

Explorer outputs condition on longer prefixes using kernel parameters frozen from 1990–1995. These are exploratory results and must not be confused with the final refitted model. Their horizon-prefix null property was author-checked.
