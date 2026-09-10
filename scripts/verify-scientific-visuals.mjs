import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { csvBoundarySource, csvBoundaryFields } from '../src/learn/data/csv-parsing-model.js';
import { pivotModel } from '../src/learn/data/pandas-pivot-model.js';
import { selectionResult, sensorValues } from '../src/learn/data/numpy-foundations-model.js';
import { sqlSensors } from "../src/learn/data/sql-models.js";
import { sqlReadings } from "../src/learn/data/sql-models.js";

const fixture = { csv: { source: csvBoundarySource, fields: csvBoundaryFields }, sensorValues,
  transpose: selectionResult('transpose'), reshape: selectionResult('reshape'), sqlSensors, sqlReadings,
  pivots: ['unique', 'duplicate', 'missing'].flatMap(variation => ['pivot', 'sum', 'mean'].map(operation => ({ variation, operation, ...pivotModel(variation, operation) }))) };
const directory = 'scratch/scientific-visual-review';
fs.mkdirSync(directory, { recursive: true });
const fixturePath = path.resolve(directory, 'native-fixture.json');
fs.writeFileSync(fixturePath, JSON.stringify(fixture));
const result = spawnSync(process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-scientific-visuals.py', fixturePath], {
  encoding: 'utf8', env: { ...process.env, MPLBACKEND: 'Agg', MPLCONFIGDIR: path.resolve(directory, 'mpl') },
});
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0, result.error?.message || 'Native scientific visual checks failed');
fs.writeFileSync(path.join(directory, 'native-results.json'), result.stdout);
