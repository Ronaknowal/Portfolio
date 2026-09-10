export const pivotRegions = ['North', 'South'];

export const pivotMonths = ['Jan', 'Feb'];

export function pivotModel(variation = 'unique', operation = 'pivot') {
  const rows = [
    { id: 'r1', region: 'North', month: 'Jan', amount: 10 },
    { id: 'r2', region: 'North', month: 'Feb', amount: 20 },
    { id: 'r3', region: 'South', month: 'Jan', amount: 30 },
    { id: 'r4', region: 'South', month: 'Feb', amount: 40 },
  ];
  if (variation === 'duplicate') rows.push({ ...rows[0], id: 'r5' });
  if (variation === 'missing') rows.pop();
  const cells = pivotRegions.flatMap(region => pivotMonths.map(month => {
    const sources = rows.filter(row => row.region === region && row.month === month);
    return { region, month, sources, conflict: operation === 'pivot' && sources.length > 1,
      value: sources.length ? sources.reduce((total, row) => total + row.amount, 0) / (operation === 'mean' ? sources.length : 1) : null };
  }));
  return { rows, cells, blocked: cells.some(cell => cell.conflict) };
}
