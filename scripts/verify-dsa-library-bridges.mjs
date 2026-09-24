import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parse } from '@babel/parser';
import { collectExamples } from './build-dsa-library-bridges.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.LESSON_PYTHON || path.join(root, 'scratch/lesson-tools/Scripts/python.exe');
const outputPath = path.join(root, 'docs/teaching/implementation-depth/dsa-remediation-native.json');
const previous = fs.existsSync(outputPath) ? JSON.parse(fs.readFileSync(outputPath, 'utf8')) : null;
const records = [];
const sourceHashes = {};
const files = {
  tree: ['ordered_set_library.py'], 'graph-traversal': ['graph_library.py'],
  'range-query': ['range_library_reference.py'], 'weighted-graph': ['weighted_graph_library.py'],
  'string-matching': ['string_search_library.py'], 'randomized-algorithm': ['sampling_library.py'],
  'network-flow': ['flow_library.py'], 'computational-geometry': ['hull_library.py'],
  'persistent-structures': ['persistent_map_library.py'],
  'external-memory': ['sqlite_index_library.py', 'external_sort_stream.py'],
};
const components = {
  tree: 'TreeLibraryBridge', 'graph-traversal': 'GraphTraversalLibraryBridge',
  'range-query': 'RangeQueryLibraryBridge', 'weighted-graph': 'WeightedGraphLibraryBridge',
  'string-matching': 'StringSearchLibraryBridge', 'randomized-algorithm': 'SamplingLibraryBridge',
  'network-flow': 'FlowLibraryBridge', 'computational-geometry': 'GeometryLibraryBridge',
  'persistent-structures': 'PersistentMapLibraryBridge', 'external-memory': 'ExternalMemoryLibraryBridge',
};
const normalize = text => text.replaceAll('\r\n', '\n').trim();
function run(command, args, input) {
  const result = spawnSync(command, args, { cwd: root, input, encoding: 'utf8', timeout: 120000,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' }, maxBuffer: 8 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(result.error?.message || result.stderr || result.stdout);
  return normalize(result.stdout);
}
function hash(file) {
  sourceHashes[file] = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, file))).digest('hex');
}
function save(status, error) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify({ status, scope: 'author native and source checks; independent/browser/integration separate',
    runtime: run(python, ['-c', 'import sys,numpy,scipy,networkx,immutables,sqlite3; print(sys.version.split()[0], "NumPy",numpy.__version__,"SciPy",scipy.__version__,"NetworkX",networkx.__version__,"immutables",immutables.__version__,"SQLite",sqlite3.sqlite_version)']),
    records, sourceHashes, ...(error ? { error } : {}) }, null, 2) + '\n');
}
save('running');
try {
  run(process.execPath, ['scripts/build-dsa-library-bridges.mjs', '--check']);
  for (const { id, stem, owner, examples } of await collectExamples()) {
    hash(owner);
    let originals = 0;
    const reuse = process.argv.includes('--reuse-originals') && previous?.status === 'passed'
      && previous.sourceHashes[owner] === sourceHashes[owner]
      && previous.records.find(record => record.id === id && record.originalPrograms)?.exactOutputs;
    if (reuse) originals = Object.keys(examples).length;
    else for (const [key, example] of Object.entries(examples)) {
      const actual = run(python, ['-X', 'utf8', '-B', '-'], example.code);
      if (actual !== normalize(example.expected)) throw new Error(`${stem}.${key}: recorded original output mismatch`);
      originals += 1;
    }
    const body = `src/learn/data/topics/${id}.jsx`;
    parse(fs.readFileSync(path.join(root, body), 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
    hash(body);
    const component = `src/learn/components/lesson-labs/${components[stem]}.jsx`;
    parse(fs.readFileSync(path.join(root, component), 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
    hash(component);
    if (!fs.readFileSync(path.join(root, body), 'utf8').includes(`<${components[stem]} />`)) throw new Error(`Missing integration: ${component}`);
    const directory = `public/learn-assets/${id}`;
    const output = {};
    for (const file of files[stem]) {
      output[file] = run(python, ['-X', 'utf8', '-B', `${directory}/${file}`]);
      hash(`${directory}/${file}`);
    }
    if (stem === 'range-query') {
      const scratch = path.join(root, 'scratch/dsa-remediation');
      fs.mkdirSync(scratch, { recursive: true });
      const executable = path.join(scratch, 'range-library.exe');
      run(process.env.LESSON_CXX || 'C:/msys64/ucrt64/bin/g++.exe', ['-std=c++17', '-O2',
        `-I${directory}/ac-library-v1.6`, `${directory}/range_library.cpp`, '-o', executable]);
      output['range_library.cpp'] = run(executable, []);
      if (output['range_library.cpp'] !== output['range_library_reference.py']) throw new Error('C++/Python range comparison mismatch');
      hash(`${directory}/range_library.cpp`);
      hash(`${directory}/ac-library-provenance.json`);
      const oracle = path.join(scratch, 'range-library-oracle.exe');
      run(process.env.LESSON_CXX || 'C:/msys64/ucrt64/bin/g++.exe', ['-std=c++17', '-O2',
        `-I${directory}/ac-library-v1.6`, `-I${directory}`, 'scripts/fixtures/range-library-oracle.cpp', '-o', oracle]);
      const oracleResult = run(oracle, []);
      if (oracleResult !== 'C++ direct-array range checks: 14000; ordered folds: 21') throw new Error(`Unexpected C++ oracle result: ${oracleResult}`);
      records.push({ id, check: 'C++ adapters against direct-array oracle', result: oracleResult });
      hash('scripts/fixtures/range-library-oracle.cpp');
      const provenance = JSON.parse(fs.readFileSync(path.join(root, directory, 'ac-library-provenance.json'), 'utf8'));
      for (const file of provenance.files) {
        const owned = `${directory}/ac-library-v1.6/${file.path}`;
        hash(owned);
        if (sourceHashes[owned] !== file.sha256) throw new Error(`Header provenance mismatch: ${owned}`);
      }
      hash(`${directory}/ac-library-v1.6-range-headers.zip`);
    }
    hash(`${directory}/${stem.replaceAll('-', '_')}_mechanisms.py`);
    const displayedOutput = `src/learn/data/${stem}-library-output.json`;
    if (process.argv.includes('--update-output')) fs.writeFileSync(path.join(root, displayedOutput), JSON.stringify(output, null, 2) + '\n');
    else if (JSON.stringify(JSON.parse(fs.readFileSync(path.join(root, displayedOutput)))) !== JSON.stringify(output)) throw new Error(`Stale output: ${displayedOutput}`);
    hash(displayedOutput);
    records.push({ id, originalPrograms: originals, originalProgramsReusedFromMatchingHash: Boolean(reuse), bridgePrograms: Object.keys(output), exactOutputs: true, bodyParsed: true });
    console.log(`${id}: ${originals} original + ${Object.keys(output).length} comparison programs passed`);
  }
  hash('scripts/build-dsa-library-bridges.mjs');
  hash('scripts/verify-dsa-library-bridges.mjs');
  save('passed');
} catch (error) {
  save('failed', error.message);
  throw error;
}
