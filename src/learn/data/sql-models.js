export const sqlSensors = [{ id: 'A', room: 'north' }, { id: 'B', room: 'south' }, { id: 'C', room: 'spare' }];

export const sqlReadings = [
  { id: 1, sensor: 'A', minute: 0, value: 18 },
  { id: 2, sensor: 'A', minute: 10, value: 22 },
  { id: 3, sensor: 'B', minute: 0, value: null },
];

export function joinModel({ join = 'left', duplicate = false, cutoff = 10, placement = 'on' } = {}) {
  const sensors = duplicate ? [...sqlSensors, { id: 'A', room: 'annex' }] : sqlSensors;
  const pairs = [];
  sensors.forEach((sensor, sensorIndex) => {
    const matches = sqlReadings.filter(reading => reading.sensor === sensor.id && (placement === 'where' || reading.minute <= cutoff));
    if (matches.length) matches.forEach(reading => pairs.push({ sensor, sensorIndex, reading }));
    else if (join === 'left') pairs.push({ sensor, sensorIndex, reading: null });
  });
  const result = placement === 'where' ? pairs.filter(pair => pair.reading !== null && pair.reading.minute <= cutoff) : pairs;
  const grouped = [...new Set(result.map(pair => pair.sensor.id))].map(id => {
    const group = result.filter(pair => pair.sensor.id === id);
    const measured = group.filter(pair => pair.reading?.value != null).map(pair => pair.reading.value);
    return { id, rows: group.length, readings: group.filter(pair => pair.reading !== null).length, measured: measured.length, mean: measured.length ? measured.reduce((a,b) => a+b,0) / measured.length : null };
  });
  return { sensors, readings: sqlReadings, pairs: result, grouped };
}

export function transactionTrace({ atomic = true, fail = true } = {}) {
  // Two compute-credit accounts; the invariant is A + B = 10 credits.
  const states = [{ label: 'Initial committed state', writer: [6, 4], committed: [6, 4], note: 'Move 2 credits from A to B. The total should stay 10.' }];
  states.push({ label: atomic ? 'BEGIN' : 'Use separate statements', writer: [6, 4], committed: [6, 4], note: atomic ? 'Both updates will belong to one explicit transaction.' : 'Each statement will commit independently.' });
  states.push({ label: 'Debit A by 2', writer: [4, 4], committed: atomic ? [6, 4] : [4, 4], note: atomic ? 'The writer sees its pending debit. The independently committed state is still unchanged.' : 'The first update is already committed. An incomplete transfer is now visible.' });
  if (fail) {
    states.push({ label: 'Second statement violates a constraint', writer: [4, 4], committed: atomic ? [6, 4] : [4, 4], note: 'This statement failed. Do not assume that every database error automatically undoes every previous statement.' });
    states.push({ label: atomic ? 'Application calls ROLLBACK' : 'Application calls ROLLBACK too late', writer: atomic ? [6, 4] : [4, 4], committed: atomic ? [6, 4] : [4, 4], note: atomic ? 'The pending debit is discarded; both balances return to the committed state.' : 'There is no open transaction containing the earlier debit. Repair requires a new, carefully checked transaction.' });
  } else {
    states.push({ label: 'Credit B by 2', writer: [4, 6], committed: atomic ? [6, 4] : [4, 6], note: atomic ? 'The writer has finished both pending changes; the committed state is still the original pair.' : 'The second independent update is committed. The final total is correct, but an intermediate state was exposed.' });
    states.push({ label: atomic ? 'COMMIT' : 'Both individual commits finished', writer: [4, 6], committed: [4, 6], note: 'The completed transfer has A=4, B=6 and total=10.' });
  }
  return states;
}
