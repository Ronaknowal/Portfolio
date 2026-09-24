"""Independent set/ordering, reuse-distance, page-length and recovery oracles."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import hashlib
import json
import math
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/external-memory-verification'
OUT.mkdir(parents=True, exist_ok=True)

def node(source):
    return json.loads(subprocess.check_output(['node', '--input-type=module', '-e', source], cwd=ROOT, text=True, encoding='utf-8'))

examples = node("import {externalMemoryExamples} from './src/learn/data/external-memory-examples.js'; console.log(JSON.stringify(externalMemoryExamples));")
native = {}
for name, example in examples.items():
    path = OUT / f'{name}.py'
    path.write_text(example['code'], encoding='utf-8')
    actual = subprocess.check_output([sys.executable, '-X', 'utf8', '-I', str(path)], text=True, encoding='utf-8')
    assert actual.strip() == example['expected'], (name, actual)
    namespace = {}
    with redirect_stdout(StringIO()):
        exec(compile(example['code'], str(path), 'exec'), namespace)
    native[name] = namespace

rng = random.Random(38261)
tree_fixtures = [[degree, preset, [[rng.choice(['insert', 'delete']), rng.randrange(30)] for _ in range(12)]]
                 for degree in range(2, 5) for preset in ['insert', 'delete', 'duplicates'] for _ in range(30)]
(OUT / 'fixtures.json').write_text(json.dumps(tree_fixtures), encoding='utf-8')
data = node(r"""
import fs from 'node:fs';
import * as model from './src/learn/data/external-memory-models.js';
const fixtures = JSON.parse(fs.readFileSync('scratch/external-memory-verification/fixtures.json','utf8'));
const buffers = [], ranges = [], sorts = [];
for (let size=1;size<=8;size++) for (let frames=1;frames<=6;frames++) for (const pattern of ['sequential','strided','reuse','writes']) buffers.push(model.bufferTrace(size,frames,pattern));
for (let capacity=2;capacity<=4;capacity++) for (let fanout=2;fanout<=4;fanout++) for(let low=0;low<=40;low++) for(let high=0;high<=40;high++) {
  const state=model.bplusRange(low,high,capacity,fanout);
  ranges.push({low,high,capacity,fanout,result:state.result,visited:state.visited});
}
for(let n=0;n<=96;n++) for(let b=1;b<=8;b++) for(let p=3;p<=8;p++) {
  const state=model.mergePlan(n,b,p);
  sorts.push({n,b,p,result:state.result,stages:state.stages.map(stage=>({lengths:stage.runs.map(run=>run.length),reads:stage.reads,writes:stage.writes})),reads:state.totalReads,writes:state.totalWrites});
}
const trees=fixtures.map(args=>{
  const state=model.btreeTrace(...args);
  return {completed:state.completed,events:state.frames.map(frame=>frame.action),searches:Array.from({length:35},(_,key)=>state.search(key))};
});
const invalid=[];
for (const run of [()=>model.bufferTrace(0),()=>model.bufferTrace(4,0),()=>model.bufferTrace(4,2,'bad'),()=>model.btreeTrace(1),()=>model.btreeTrace(2,'insert',[['bad',4]]),()=>model.bplusRange(-1),()=>model.mergePlan(1,4,2),()=>model.mergePlan(1.5)]) {
  try {run(); invalid.push(false);} catch(error) {invalid.push(error instanceof RangeError);}
}
console.log(JSON.stringify({buffers,ranges,sorts,trees,invalid,crashes:[model.shadowCommitTrace(false),model.shadowCommitTrace(true)]}));
""")
assert all(data['invalid'])

# Independent LRU representation: last-use timestamps and dirty membership.
for case in data['buffers']:
    last_used, dirty = {}, set()
    reads = writes = hits = 0
    for index, (address, operation) in enumerate(case['requests']):
        page = address // case['pageSize']
        if page in last_used:
            hits += 1
        else:
            reads += 1
            if len(last_used) == case['capacity']:
                evicted = min(last_used, key=last_used.get)
                del last_used[evicted]
                writes += evicted in dirty
                dirty.discard(evicted)
        last_used[page] = index
        if operation == 'write':
            dirty.add(page)
        frame = case['frames'][index + 1]
        assert (frame['reads'], frame['writes'], frame['hits']) == (reads, writes, hits)
        assert [(item['id'], item['dirty']) for item in frame['resident']] == [(key, key in dirty) for key in sorted(last_used, key=last_used.get)]
    final = case['frames'][-1]
    assert (final['reads'], final['writes'], final['hits']) == (reads, writes + len(dirty), hits)
    native_counts = native['locality']['buffer_counts'](case['requests'], case['pageSize'], case['capacity'])
    assert native_counts == (reads, writes, len(dirty), hits)

def page_dict(page):
    return {'keys': list(page.keys), 'children': [page_dict(child) for child in page.children]}

def verify_tree(root, degree, expected):
    depths, values = [], []
    def visit(page, lower, upper, depth, is_root):
        keys, children = page['keys'], page['children']
        assert keys == sorted(set(keys)) and all(lower < key < upper for key in keys)
        assert len(keys) <= 2 * degree - 1
        assert len(keys) >= (int(bool(children)) if is_root else degree - 1)
        values.extend(keys)
        if children:
            assert len(children) == len(keys) + 1
            bounds = [lower, *keys, upper]
            for index, child in enumerate(children):
                visit(child, bounds[index], bounds[index + 1], depth + 1, False)
        else:
            depths.append(depth)
    visit(root, -math.inf, math.inf, 0, True)
    assert len(set(depths)) == 1
    assert sorted(values) == sorted(expected)

js_states = 0
for (degree, preset, extra), case in zip(tree_fixtures, data['trees']):
    expected = set(range(1, 17)) if preset == 'delete' else ({10, 5, 20} if preset == 'duplicates' else set())
    for frame in case['completed']:
        if frame['operation'] == 'insert': expected.add(frame['key'])
        else: expected.discard(frame['key'])
        verify_tree(frame['root'], degree, expected)
        js_states += 1
    for key, query in enumerate(case['searches']):
        assert query['found'] == (key in expected) and len(query['visited']) >= 1

native_operations, repair_events = 0, set()
for degree in [2, 3, 4, 8]:
    for _ in range(20):
        tree = native['btreeUpdates']['BTree'](degree)
        expected = set()
        operations = [('insert', rng.randrange(80)) for _ in range(100)]
        operations += [(rng.choice(['insert', 'delete']), rng.randrange(100)) for _ in range(150)]
        operations += [('delete', key) for key in range(100)]
        for operation, key in operations:
            before = key in expected
            actual = tree.insert(key) if operation == 'insert' else tree.remove(key)
            assert actual == (not before if operation == 'insert' else before)
            if operation == 'insert': expected.add(key)
            else: expected.discard(key)
            verify_tree(page_dict(tree.root), degree, expected)
            assert tree.ordered() == sorted(expected)
            assert tree.search(key)[0] == (key in expected)
            native_operations += 1
        repair_events.update(tree.events)
assert repair_events == {'split', 'merge', 'predecessor', 'successor', 'borrow-left', 'borrow-right', 'shrink-root'}

records = list(range(2, 36, 3))
for case in data['ranges']:
    expected = [value for value in records if case['low'] <= value <= case['high']]
    assert case['result'] == expected
    root, leaves = native['leafRanges']['bulk_index'](records, case['capacity'], case['fanout'])
    actual, visits = native['leafRanges']['range_query'](root, leaves, case['low'], case['high'])
    assert actual == expected and visits == case['visited']
    assert len(visits) == len(set(visits))
    if case['low'] > case['high']: assert visits == []

def sort_oracle(n, block, frames):
    if not n: return []
    memory, fan_in = block * frames, frames - 1
    lengths = [min(memory, n-offset) for offset in range(0, n, memory)]
    pages = lambda runs: sum((length + block - 1)//block for length in runs)
    stages = [{'lengths': lengths, 'reads': (n+block-1)//block, 'writes': pages(lengths)}]
    while len(lengths)>1:
        next_lengths = [sum(lengths[offset:offset+fan_in]) for offset in range(0,len(lengths),fan_in)]
        stages.append({'lengths': next_lengths, 'reads': pages(lengths), 'writes': pages(next_lengths)})
        lengths = next_lengths
    return stages

for case in data['sorts']:
    expected = sort_oracle(case['n'],case['b'],case['p'])
    assert case['stages'] == expected
    assert case['result'] == list(range(case['n']))
    assert case['reads'] == sum(stage['reads'] for stage in expected)
    assert case['writes'] == sum(stage['writes'] for stage in expected)

file_cases = 0
for size in [0,1,2,7,12,13,32,33,35,48,65,96]:
    for block, frames in [(2,3),(4,3),(4,4),(8,5)]:
        values = [rng.randrange(-20,21) for _ in range(size)]
        result, stages, counts = native['externalSort']['external_sort'](values,block,frames)
        oracle = sort_oracle(size,block,frames)
        assert result == sorted(values)
        assert stages == [(len(stage['lengths']),stage['reads'],stage['writes']) for stage in oracle]
        assert counts == {'reads':sum(stage['reads'] for stage in oracle),'writes':sum(stage['writes'] for stage in oracle)}
        file_cases += 1

for case in data['crashes']:
    for frame in case['frames']:
        root_id = frame['committedRoot']
        reachable = {root_id, case['pages'][root_id]['child']}
        complete = reachable <= set(frame['durable'])
        expected = case['pages'][case['pages'][root_id]['child']]['value'] if complete else None
        assert frame['recoveredValue'] == expected
        assert bool(frame['missing']) == (not complete)
    expected_values = [5,5,None,None,9,9] if case['earlyRoot'] else [5,5,5,5,9,9]
    assert [frame['recoveredValue'] for frame in case['frames']] == expected_values
    python_values = [value for _,value in native['crashRoots']['crash_states'](case['earlyRoot'])]
    assert python_values == [value if value is not None else 'invalid root reachability' for value in expected_values]

# Changed local exercises independently executed, not copied output snapshots.
assert native['locality']['buffer_counts']([(0,'write'),(1,'write'),(2,'read'),(0,'read')],2,1) == (3,1,0,1)
assert 24+16*62 <= 1024 < 24+16*63
assert 2*3**3-1 == 53
Page, BTree = native['btreeUpdates']['Page'], native['btreeUpdates']['BTree']
for left, expected_root, expected_children in [([2,5],[5],[[2],[10]]),([5],[5,10],[])]:
    tree = BTree(2); tree.root = Page([10],[Page(left),Page([12])]); tree.remove(12)
    assert tree.root.keys == expected_root and [child.keys for child in tree.root.children] == expected_children
for n,b,p,total in [(35,4,4,36),(35,4,3,54),(33,4,3,54),(13,2,3,42)]:
    assert sum(stage['reads']+stage['writes'] for stage in sort_oracle(n,b,p)) == total
changed = [9,1,7,1,4,8,2,6,5,3,0,9,2]
result,_,counts = native['externalSort']['external_sort'](changed,2,3)
assert result == [0,1,1,2,2,3,4,5,6,7,8,9,9] and sum(counts.values()) == 42
tree = BTree(2)
for key in result: tree.insert(key)
for key in [1,5,9]: tree.remove(key)
verify_tree(page_dict(tree.root),2,{0,2,3,4,6,7,8})

files = [ROOT/'src/learn/data/external-memory-models.js', ROOT/'src/learn/data/external-memory-examples.js']
record = {'checkedAt':datetime.now(timezone.utc).isoformat(),'nativePrograms':list(examples),
          'bufferCases':len(data['buffers']),'javascriptCompletedTreeStates':js_states,
          'nativeTreeOperations':native_operations,'nativeRepairEvents':sorted(repair_events),
          'bplusRanges':len(data['ranges']),'mergeSchedules':len(data['sorts']),
          'actualTemporaryFileSortCases':file_cases,'crashFrames':12,'changedExerciseGroups':7,
          'invalidModelInputs':len(data['invalid']),
          'sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
(OUT/'results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
