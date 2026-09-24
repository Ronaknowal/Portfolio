// Each example is independently runnable. Outputs are checked against pandas 3.0.1.
export const pandasExamples = {
  inspect: {
    code: `import pandas as pd

orders = pd.DataFrame({
    "order_id": [101, 102, 103],
    "customer": pd.Series(["C1", "C2", "C1"], dtype="string"),
    "amount": pd.Series([10, 20, None], dtype="Int64"),
})
print(orders.to_string(index=False))
print("shape:", orders.shape)
print("types:", orders.dtypes.astype(str).to_dict())
print("missing:", orders.isna().sum().to_dict())
print(type(orders["amount"]).__name__, type(orders[["amount"]]).__name__)`,
    output: ` order_id customer  amount
      101       C1      10
      102       C2      20
      103       C1    <NA>
shape: (3, 3)
types: {'order_id': 'int64', 'customer': 'string', 'amount': 'Int64'}
missing: {'order_id': 0, 'customer': 0, 'amount': 1}
Series DataFrame`,
  },
  select: {
    code: `import pandas as pd

orders = pd.DataFrame({"order_id": [101, 102, 103], "amount": [10, 20, 30]},
                      index=["a", "b", "c"])
print(orders.loc["a":"b", "order_id"].tolist())
print(orders.iloc[0:1, 0].tolist())
mask = (orders["amount"] >= 20) & (orders["amount"] < 30)
print(orders.loc[mask, ["order_id", "amount"]].to_dict("records"))
orders.loc[mask, "amount"] = 25
print(orders["amount"].tolist())
print(orders.at["b", "amount"], orders.iat[1, 1])`,
    output: `[101, 102]
[101]
[{'order_id': 102, 'amount': 20}]
[10, 25, 30]
25 25`,
  },
  alignment: {
    code: `import pandas as pd

orders = pd.DataFrame({"amount": [10, 20]}, index=["b", "a"])
fees = pd.Series([1, 2], index=["a", "b"])
orders["total"] = orders["amount"] + fees
print(orders.to_dict("index"))
orders["position_only"] = fees.to_numpy()
print(orders["position_only"].tolist())
print(orders.reindex(["a", "b"])["total"].tolist())
print(orders.reset_index(names="row_label").columns.tolist())

part = orders[["amount"]]
part.loc["b", "amount"] = 99
print("original:", orders.loc["b", "amount"], "derived:", part.loc["b", "amount"])`,
    output: `{'b': {'amount': 10, 'total': 12}, 'a': {'amount': 20, 'total': 21}}
[1, 2]
[21, 12]
['row_label', 'amount', 'total', 'position_only']
original: 10 derived: 99`,
  },
  cleaning: {
    code: `import pandas as pd

raw = pd.DataFrame({
    "customer": pd.Series([" c1 ", "C2", None, ""], dtype="string"),
    "amount_text": pd.Series(["10", "bad", None, "0"], dtype="string"),
})
raw["customer"] = raw["customer"].str.strip().str.upper().replace("", pd.NA)
raw["amount"] = pd.to_numeric(raw["amount_text"], errors="coerce").astype("Float64")
invalid = raw["amount_text"].notna() & raw["amount"].isna()
print("invalid source values:", raw.loc[invalid, "amount_text"].tolist())
print("missing:", raw[["customer", "amount"]].isna().sum().to_dict())
print("known nonnegative:", raw["amount"].ge(0).fillna(False).tolist())
print("presentation only:", raw["customer"].fillna("UNKNOWN").tolist())
print("sum all missing:", pd.Series([pd.NA], dtype="Int64").sum(min_count=1))`,
    output: `invalid source values: ['bad']
missing: {'customer': 2, 'amount': 2}
known nonnegative: [True, False, False, True]
presentation only: ['C1', 'C2', 'UNKNOWN', 'UNKNOWN']
sum all missing: <NA>`,
  },
  dates: {
    code: `import pandas as pd

events = pd.DataFrame({
    "time": ["2026-01-01T23:00:00+00:00", "invalid", "2026-01-02T01:00:00+00:00"],
    "amount": [10, 999, 20],
})
events["time"] = pd.to_datetime(events["time"], format="ISO8601", utc=True, errors="coerce")
print("invalid timestamps:", int(events["time"].isna().sum()))
valid = events.dropna(subset=["time"]).set_index("time").sort_index()
daily = valid["amount"].resample("D").sum(min_count=1)
print(list(zip(daily.index.strftime("%Y-%m-%d"), daily.tolist())))
print(valid.index.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M").tolist())`,
    output: `invalid timestamps: 1
[('2026-01-01', 10), ('2026-01-02', 20)]
['2026-01-02 04:30', '2026-01-02 06:30']`,
  },
  duplicates: {
    code: `import pandas as pd

orders = pd.DataFrame({"order_id": [101, 101, 102, 102], "amount": [10, 10, 20, 25]})
print("exact duplicate rows:", int(orders.duplicated().sum()))
deduped = orders.drop_duplicates()
conflicts = deduped.loc[deduped.duplicated("order_id", keep=False)]
print("conflicting IDs:", conflicts["order_id"].unique().tolist())
indexed = deduped.set_index("order_id")
if not indexed.index.is_unique:
    print("order_id is not unique")
parts = [pd.DataFrame({"order_id": [101]}), pd.DataFrame({"order_id": [102]})]
print(pd.concat(parts, ignore_index=True)["order_id"].tolist())`,
    output: `exact duplicate rows: 1
conflicting IDs: [102]
order_id is not unique
[101, 102]`,
  },
  merge: {
    code: `import pandas as pd

orders = pd.DataFrame({"order_id": [101, 102, 103], "customer": ["C1", "C2", "C9"]})
customers = pd.DataFrame({"customer": ["C1", "C2", "C3"], "region": ["North", "South", "West"]})
joined = orders.merge(customers, on="customer", how="left", validate="many_to_one", indicator=True)
print(joined[["order_id", "region", "_merge"]].to_string(index=False))
print("unmatched orders:", joined.loc[joined["_merge"] == "left_only", "order_id"].tolist())
for how in ["inner", "left", "outer"]:
    print(how, len(orders.merge(customers, on="customer", how=how)))
bad_lookup = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)
try:
    orders.merge(bad_lookup, on="customer", validate="many_to_one")
except pd.errors.MergeError:
    print("duplicate lookup key rejected")`,
    output: ` order_id region    _merge
      101  North      both
      102  South      both
      103    NaN left_only
unmatched orders: [103]
inner 2
left 3
outer 4
duplicate lookup key rejected`,
  },
  nullJoin: {
    code: `import pandas as pd

left = pd.DataFrame({"key": pd.Series([pd.NA], dtype="string"), "value": [1]})
right = pd.DataFrame({"key": pd.Series([pd.NA], dtype="string"), "label": ["missing key"]})
matched = left.merge(right, on="key", how="inner")
print("null-key matches:", len(matched))
safe = left.dropna(subset=["key"]).merge(right.dropna(subset=["key"]), on="key", how="inner")
print("matches after rejecting null keys:", len(safe))`,
    output: `null-key matches: 1
matches after rejecting null keys: 0`,
  },
  group: {
    code: `import pandas as pd

orders = pd.DataFrame({
    "region": pd.Series(["North", "North", "South", pd.NA], dtype="string"),
    "amount": pd.Series([10, pd.NA, 30, 40], dtype="Int64"),
})
summary = orders.groupby("region", dropna=False, observed=True, as_index=False).agg(
    rows=("amount", "size"), known=("amount", "count"), mean=("amount", "mean")
)
print(summary.to_string(index=False))
orders["region_mean"] = orders.groupby("region", dropna=False)["amount"].transform("mean")
print("per-row means:", orders["region_mean"].tolist())
print("rows without missing-key group:", orders.groupby("region").size().sum())`,
    output: `region  rows  known  mean
 North     2      1  10.0
 South     1      1  30.0
  <NA>     1      1  40.0
per-row means: [10.0, 10.0, 30.0, 40.0]
rows without missing-key group: 3`,
  },
  reshape: {
    code: `import pandas as pd

long = pd.DataFrame({
    "region": ["North", "North", "South", "South"],
    "month": ["Jan", "Feb", "Jan", "Feb"],
    "amount": [10, 20, 30, 40],
})
wide = long.pivot(index="region", columns="month", values="amount").reindex(columns=["Jan", "Feb"])
print(wide.to_string())
tidy = wide.reset_index().melt(id_vars="region", var_name="month", value_name="amount")
print("long shape:", tidy.shape)
duplicated = pd.concat([long, long.iloc[[0]]], ignore_index=True)
try:
    duplicated.pivot(index="region", columns="month", values="amount")
except ValueError:
    print("pivot rejects duplicate cells")
totals = duplicated.pivot_table(index="region", columns="month", values="amount", aggfunc="sum", observed=True)
print("North/Jan with explicit sum:", totals.loc["North", "Jan"])`,
    output: `month   Jan  Feb
region          
North    10   20
South    30   40
long shape: (4, 3)
pivot rejects duplicate cells
North/Jan with explicit sum: 20`,
  },
  windows: {
    code: `import pandas as pd

orders = pd.DataFrame({
    "customer": ["C1", "C2", "C1", "C2"],
    "day": [1, 1, 2, 2], "amount": [10, 20, 30, 40],
}).sort_values(["customer", "day"], kind="stable")
orders["previous"] = orders.groupby("customer")["amount"].shift(1)
orders["running"] = orders.groupby("customer")["amount"].cumsum()
print(orders.to_string(index=False))
print("two-row moving mean:", pd.Series([10, 30, 50]).rolling(2, min_periods=1).mean().tolist())`,
    output: `customer  day  amount  previous  running
      C1    1      10       NaN       10
      C1    2      30      10.0       40
      C2    1      20       NaN       20
      C2    2      40      20.0       60
two-row moving mean: [10.0, 20.0, 40.0]`,
  },
  project: {
    code: `from io import StringIO
import pandas as pd

csv = """order_id,customer,amount,status
101, c1 ,10,paid
102,C2,20,paid
103,C1,bad,paid
104,,30,cancelled
105,C9,40,paid
101, c1 ,10,paid
"""
raw = pd.read_csv(StringIO(csv), dtype="string")
orders = raw.drop_duplicates().copy()
if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
    raise ValueError("missing or conflicting order ID")
orders["customer"] = orders["customer"].str.strip().str.upper().replace("", pd.NA)
orders["amount_cents"] = pd.to_numeric(orders["amount"], errors="coerce").astype("Int64")
invalid = orders["amount_cents"].isna() | orders["amount_cents"].lt(0).fillna(False)
rejected = orders.loc[invalid].copy()
valid = orders.loc[~invalid].copy()
paid = valid.loc[valid["status"].eq("paid")].copy()
customers = pd.DataFrame({
    "customer": pd.Series(["C1", "C2"], dtype="string"),
    "region": pd.Series(["North", "South"], dtype="string"),
})
if customers["customer"].isna().any():
    raise ValueError("lookup keys must not be missing")
joined = paid.merge(customers, on="customer", how="left", validate="many_to_one", indicator=True)
unmatched = joined.loc[joined["_merge"].eq("left_only")]
accepted = joined.loc[joined["_merge"].eq("both")]
summary = accepted.groupby("region", as_index=False, observed=True).agg(
    orders=("order_id", "size"), revenue_cents=("amount_cents", "sum")
).sort_values("region")
audit = {
    "raw": len(raw), "exact_duplicates": len(raw) - len(orders),
    "invalid_amount": len(rejected), "not_paid": len(valid) - len(paid),
    "unmatched": len(unmatched), "accepted": len(accepted),
}
print(audit)
print(summary.to_csv(index=False, lineterminator="\\n").strip())
assert len(joined) == len(paid)
assert sum(audit[key] for key in ["exact_duplicates", "invalid_amount", "not_paid", "unmatched", "accepted"]) == len(raw)
assert summary["revenue_cents"].sum() == accepted["amount_cents"].sum()
assert unmatched["order_id"].tolist() == ["105"]
expected = pd.DataFrame({
    "region": pd.Series(["North", "South"], dtype="string"),
    "orders": [1, 1], "revenue_cents": pd.Series([10, 20], dtype="Int64"),
})
pd.testing.assert_frame_equal(summary.reset_index(drop=True), expected)
buffer = StringIO()
summary.to_csv(buffer, index=False)
buffer.seek(0)
restored = pd.read_csv(buffer, dtype={"region": "string", "orders": "int64", "revenue_cents": "Int64"})
pd.testing.assert_frame_equal(restored, expected)
print("audit and round-trip checks passed")`,
    output: `{'raw': 6, 'exact_duplicates': 1, 'invalid_amount': 1, 'not_paid': 1, 'unmatched': 1, 'accepted': 2}
region,orders,revenue_cents
North,1,10
South,1,20
audit and round-trip checks passed`,
  },
};
