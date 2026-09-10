export const pandasPracticeExamples={
  nullable:{code:`import pandas as pd

raw = pd.Series(["10", "", "bad", "0"], dtype="string")
value = pd.to_numeric(raw, errors="coerce").astype("Float64")
positive = value > 0
print("raw:", raw.tolist())
print("values:", value.tolist())
print("positive mask:", positive.tolist())
print("known rows:", value.loc[value.notna()].index.tolist())
print("positive rows:", value.loc[positive].index.tolist())
invalid = raw.ne("") & value.isna()
print("invalid source rows:", raw.loc[invalid].index.tolist())`,output:`raw: ['10', '', 'bad', '0']
values: [10.0, <NA>, <NA>, 0.0]
positive mask: [True, <NA>, <NA>, False]
known rows: [0, 3]
positive rows: [0]
invalid source rows: [2]`},
  temporal:{code:`import pandas as pd

readings = pd.DataFrame({"time_ms": [2, 5, 9], "raw_value": [10, 12, 15]})
calibration = pd.DataFrame({"calibration_ms": [0, 4], "offset": [1, 3]})
matched = pd.merge_asof(
    readings.sort_values("time_ms"),
    calibration.sort_values("calibration_ms"),
    left_on="time_ms", right_on="calibration_ms",
    direction="backward", tolerance=3, allow_exact_matches=True,
)
matched["corrected"] = matched["raw_value"] - matched["offset"]
for row in matched.itertuples(index=False):
    if pd.isna(row.offset):
        print(f"t={row.time_ms}: no recent calibration; corrected unknown")
    else:
        age = int(row.time_ms - row.calibration_ms)
        print(f"t={row.time_ms}: offset={row.offset:g}, age={age}, corrected={row.corrected:g}")
assert matched["corrected"].iloc[:2].tolist() == [9, 9]
assert pd.isna(matched["corrected"].iloc[2])`,output:`t=2: offset=1, age=2, corrected=9
t=5: offset=3, age=1, corrected=9
t=9: no recent calibration; corrected unknown`},
  uptime:{code:`import pandas as pd

samples = pd.DataFrame({
    "sample_id": [1, 2, 3, 4, 5, 6],
    "device": ["A", "A", "A", "B", "B", "C"],
    "watts": pd.Series([0, 4, None, 0, 0, None], dtype="Float64"),
})
if samples["sample_id"].duplicated().any():
    raise ValueError("duplicate sample ID")
samples["active"] = samples["watts"].gt(0).astype("Int64")
summary = samples.groupby("device", as_index=False).agg(
    scheduled=("sample_id", "size"),
    observed=("watts", "count"),
    active=("active", "sum"),
)
summary["active_fraction"] = summary["active"] / summary["observed"].replace(0, pd.NA)
for row in summary.itertuples(index=False):
    fraction = "unknown" if pd.isna(row.active_fraction) else f"{row.active_fraction:.2f}"
    print(f"{row.device}: {row.observed}/{row.scheduled} observed, active fraction={fraction}")
assert summary["observed"].tolist() == [2, 2, 0]
assert samples["watts"].isna().sum() == 2`,output:`A: 2/3 observed, active fraction=0.50
B: 2/2 observed, active fraction=0.00
C: 0/1 observed, active fraction=unknown`},
};
