export const joinOrders = [
  { order: 101, customer: "C1" },
  { order: 102, customer: "C2" },
  { order: 103, customer: "C9" },
];
export function lookupRows(duplicate = false) {
  const rows = [
    { customer: "C1", region: "North" },
    { customer: "C2", region: "South" },
    { customer: "C3", region: "West" },
  ];
  return duplicate ? [...rows, { customer: "C1", region: "North copy" }] : rows;
}
// Deliberately bounded teaching model: non-null string keys, inner/left/outer joins.
export function modelJoin(how, duplicate, validate) {
  const lookup = lookupRows(duplicate);
  if (validate && new Set(lookup.map(r => r.customer)).size !== lookup.length) {
    return { error: "MergeError: right-hand customer keys are not unique.", rows: [] };
  }
  const rows = [];
  const used = new Set();
  for (const order of joinOrders) {
    const matches = lookup.map((row, i) => ({ ...row, i })).filter(row => row.customer === order.customer);
    for (const match of matches) {
      rows.push({ ...order, region: match.region, match: "both" });
      used.add(match.i);
    }
    if (!matches.length && how !== "inner") rows.push({ ...order, region: null, match: "left_only" });
  }
  if (how === "outer") lookup.forEach((row, i) => {
    if (!used.has(i)) rows.push({ order: null, ...row, match: "right_only" });
  });
  return { error: null, rows };
}
