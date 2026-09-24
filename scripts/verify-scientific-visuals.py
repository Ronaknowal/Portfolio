import csv
import json
import sqlite3
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist

with open(sys.argv[1], encoding='utf-8') as handle:
    fixture = json.load(handle)

assert next(csv.reader([fixture['csv']['source']])) == fixture['csv']['fields']
assert len(fixture['csv']['source'].split(',')) == 5
array = np.array(fixture['sensorValues']).reshape(3, 2)
for key, actual in [('transpose', array.T), ('reshape', array.reshape(2, 3))]:
    assert actual.tolist() == np.array(fixture[key]['values']).reshape(fixture[key]['shape']).tolist()
assert array.T[0, 1] == 24 and array.reshape(2, 3)[0, 2] == 24

for case in fixture['pivots']:
    frame = pd.DataFrame(case['rows'])
    if case['blocked']:
        try:
            frame.pivot(index='region', columns='month', values='amount')
        except ValueError:
            pass
        else:
            raise AssertionError('Native pivot unexpectedly accepted a collision')
        continue
    if case['operation'] == 'pivot':
        wide = frame.pivot(index='region', columns='month', values='amount')
    else:
        wide = frame.pivot_table(index='region', columns='month', values='amount', aggfunc=case['operation'], observed=True)
    for cell in case['cells']:
        actual = wide.loc[cell['region'], cell['month']]
        assert pd.isna(actual) if cell['value'] is None else actual == cell['value']
        native_sources = frame.loc[(frame.region == cell['region']) & (frame.month == cell['month']), 'id'].tolist()
        assert native_sources == [row['id'] for row in cell['sources']]

db = sqlite3.connect(':memory:')
db.executescript('PRAGMA foreign_keys=ON; CREATE TABLE sensors(id TEXT PRIMARY KEY, room TEXT); CREATE TABLE readings(id INTEGER PRIMARY KEY, sensor TEXT REFERENCES sensors(id), minute INTEGER, value REAL);')
db.executemany('INSERT INTO sensors VALUES (?,?)', [(row['id'], row['room']) for row in fixture['sqlSensors']])
db.executemany('INSERT INTO readings VALUES (?,?,?,?)', [(row['id'], row['sensor'], row['minute'], row['value']) for row in fixture['sqlReadings']])
assert db.execute('SELECT sensor, COUNT(*) FROM readings GROUP BY sensor ORDER BY sensor').fetchall() == [('A', 2), ('B', 1)]
assert db.execute('SELECT COUNT(*) FROM readings WHERE value IS NULL').fetchone()[0] == 1
db.close()

figure, axes = plt.subplots()
line, = axes.plot([0, 1, 2], [0, 1, 2])
assert axes in figure.axes and line in axes.lines
assert axes.xaxis.axes is axes and axes.yaxis.axes is axes
assert all(isinstance(obj, Artist) for obj in [figure, axes, axes.xaxis, axes.yaxis, line])
plt.close(figure)
print(json.dumps({'passed': True, 'checks': ['CSV boundaries', 'NumPy transpose/reshape coordinates', '9 native Pandas pivot configurations and source identities', 'SQLite key relationships', 'Matplotlib object ownership'], 'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__, 'sqlite': sqlite3.sqlite_version}}))
