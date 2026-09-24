export const measurementFields = ['sample_id', 'temperature', 'unit', 'site'];

export const measurementCases = [
  { id: 'valid', label: 'Valid: an ID, a missing value and zero' },
  { id: 'number', label: 'Broken: a number spelled eighteen' },
  { id: 'nonfinite', label: 'Broken: NaN is not a finite reading' },
  { id: 'unit', label: 'Broken: Fahrenheit in a Celsius batch' },
  { id: 'duplicate', label: 'Broken: a repeated sample ID' },
];

export function measurementRows(caseId = 'valid') {
  const rows = [['001', '18.5', 'C', 'room,north'], ['002', '', 'C', 'room south'], ['003', '0', 'C', 'room south']];
  if (caseId === 'number') rows[0][1] = 'eighteen';
  if (caseId === 'nonfinite') rows[0][1] = 'NaN';
  if (caseId === 'unit') rows[2][2] = 'F';
  if (caseId === 'duplicate') rows[2][0] = '001';
  return rows;
}

export function measurementModel(caseId = 'valid') {
  const rows = measurementRows(caseId);
  const csv = [measurementFields, ...rows].map(row => row.map(cell => cell.includes(',') ? `"${cell}"` : cell).join(',')).join('\n') + '\n';
  const typed = rows.map(([id, raw, unit, site], index) => {
    const value = raw === '' ? null : Number(raw);
    const conversionError = raw !== '' && Number.isNaN(value) && raw !== 'NaN';
    return { id, value, unit, site, index, conversionError };
  });
  const conversionErrors = typed.filter(row => row.conversionError).map(row => `Record ${row.index + 1}: temperature cannot be converted to a number.`);
  const seen = new Set();
  const validationErrors = [];
  for (const row of typed) {
    if (!/^\d{3}$/.test(row.id) || seen.has(row.id)) validationErrors.push(`Record ${row.index + 1}: sample_id must be three digits and unique.`);
    seen.add(row.id);
    if (!row.conversionError && row.value !== null && (!Number.isFinite(row.value) || row.value < -80 || row.value > 80)) validationErrors.push(`Record ${row.index + 1}: temperature must be finite and between -80 and 80.`);
    if (row.unit !== 'C') validationErrors.push(`Record ${row.index + 1}: expected unit C; received ${row.unit}.`);
  }
  return { csv, rows, typed, conversionErrors, validationErrors, valid: conversionErrors.length === 0 && validationErrors.length === 0 };
}

export const oldMeasurementFile = 'sample_id,temperature\n001,18\n002,20\n';

export const newMeasurementFile = 'sample_id,temperature\n001,22\n002,24\n';

export function publicationTrace(strategy = 'replace', fail = false) {
  const prefix = 'sample_id,temperature\n001,22\n';
  const staged = strategy === 'replace';
  const snapshots = [
    { label: 'Before the update', destination: oldMeasurementFile, temporary: null, note: 'A new reader sees the complete old file.' },
    { label: staged ? 'Open a separate temporary file' : 'Open the destination with mode w', destination: staged ? oldMeasurementFile : '', temporary: staged ? '' : null, note: staged ? 'The published path is untouched. The new file has its own name in the same directory.' : 'Opening with w truncates the destination immediately; the old contents are gone.' },
    { label: 'Write the header and first record', destination: staged ? oldMeasurementFile : prefix, temporary: staged ? prefix : null, note: staged ? 'Only the staging file is partial. Readers of the published path still see the old complete file.' : 'The destination is now partial. A fresh reader can see only one of the two new records.' },
  ];
  if (fail) return [...snapshots, { ...snapshots[2], label: 'Writer fails here', note: staged ? 'The old published file remains usable. Discard the abandoned temporary file after investigating the failure.' : 'Failure does not reconstruct the old file. The published path contains an incomplete replacement.' }];
  snapshots.push({ label: 'Finish, close and validate the new file', destination: staged ? oldMeasurementFile : newMeasurementFile, temporary: staged ? newMeasurementFile : null, note: 'The new data has both expected records. A successful write still needs schema and round-trip checks.' });
  if (staged) snapshots.push({ label: 'Replace the published name', destination: newMeasurementFile, temporary: null, note: 'A successful same-filesystem replace switches the name to the complete new file. This alone is not a power-loss durability protocol.' });
  return snapshots;
}
