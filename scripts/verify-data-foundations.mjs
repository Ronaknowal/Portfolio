import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { fileExamples } from "../src/learn/data/scientific-file-examples.js";
import { sqlExamples } from "../src/learn/data/sql-examples.js";
import { measurementValidator } from "../src/learn/data/scientific-file-examples.js";
import { sensorDatabase } from "../src/learn/data/sql-examples.js";
import { measurementCases } from "../src/learn/data/scientific-file-models.js";
import { measurementModel } from "../src/learn/data/scientific-file-models.js";
import { publicationTrace } from "../src/learn/data/scientific-file-models.js";
import { joinModel } from "../src/learn/data/sql-models.js";
import { transactionTrace } from "../src/learn/data/sql-models.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'scratch/first-five-review');
fs.mkdirSync(output, { recursive: true });
const joins = [];
for (const join of ['left','inner']) for (const duplicate of [false,true]) for (const cutoff of [0,10]) for (const placement of ['on','where']) {
  const options = { join, duplicate, cutoff, placement };
  const model = joinModel(options);
  joins.push({ options, pairs: model.pairs.map(p => [p.sensorIndex + 1, p.sensor.id, p.reading?.id ?? null, p.reading?.value ?? null]), grouped: model.grouped.map(g => [g.id,g.rows,g.readings,g.measured,g.mean]) });
}
const fixture = {
  examples: { ...Object.fromEntries(Object.entries(fileExamples).map(([k,v])=>['file-'+k,v])), ...Object.fromEntries(Object.entries(sqlExamples).map(([k,v])=>['sql-'+k,v])) },
  modules: { 'measurement_io.py': measurementValidator, 'sensor_db.py': sensorDatabase },
  measurements: measurementCases.map(c => ({ id:c.id, ...measurementModel(c.id) })),
  publications: ['direct','replace'].flatMap(strategy => [false,true].map(fail => ({strategy,fail,trace:publicationTrace(strategy,fail)}))),
  joins,
  transactions: [false,true].flatMap(atomic => [false,true].map(fail => ({atomic,fail,trace:transactionTrace({atomic,fail})}))),
};
const fixturePath = path.join(output, 'data-native-fixtures.json');
fs.writeFileSync(fixturePath, JSON.stringify(fixture));
const result = spawnSync(process.env.LESSON_PYTHON || 'python', [path.join(root,'scripts/verify-data-foundations.py'), fixturePath, path.join(output,'data-native-results.json')], { cwd:root, encoding:'utf8', maxBuffer:5*1024*1024 });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status || 1);
