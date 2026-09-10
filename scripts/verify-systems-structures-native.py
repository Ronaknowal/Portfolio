"""Run exported lesson programs and independent stdlib oracles; no host changes."""
import contextlib
import io
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import unicodedata
from collections import deque
from pathlib import Path

fixtures = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
linux_only = '--linux-only' in sys.argv
if linux_only and (sys.platform != 'linux' or os.geteuid() == 0):
    raise RuntimeError('Linux verification must run as an ordinary Linux user')
env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
ran, skipped = [], []
for name, ex in fixtures['examples'].items():
    if (ex.get('platform') == 'linux' and sys.platform != 'linux') or (linux_only and name not in ['os.mappings', 'os.child', 'os.worker']):
        skipped.append(name)
        continue
    process = subprocess.run([sys.executable, '-c', ex['code']], text=True, encoding='utf-8', capture_output=True, env=env, timeout=15)
    assert process.returncode == 0, (name, process.stderr)
    assert process.stdout.strip() == ex['output'].strip(), (name, process.stdout, ex['output'])
    assert process.stderr == '', (name, process.stderr)
    ran.append(name)
if linux_only:
    print(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'uid':os.geteuid(),'passed':ran,'scope':'Actual child identity, result channels and private/shared mmap writes; no physical-frame measurements.'},indent=2))
    sys.exit(0)

def namespace(name):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(fixtures['examples'][name]['code'], name, 'exec'), ns)
    return ns

# Compare model output with Python's actual ordered sequence operations.
for case in fixtures['arrays']:
    items = ['A', 'B', 'C']
    index = {'front':0, 'insert':1, 'append':3, 'delete':1}[case['operation']]
    if case['operation'] == 'delete':
        items.pop(index)
        count = 3-index-1
    else:
        items.insert(index, 'X')
        count = (3 if case['full'] else 0) + (3-index) + 1
    end = case['trace'][-1]
    assert end['cells'][:end['length']] == items
    assert end['writes'] == count
    assert all(x is None for x in end['cells'][end['length']:])
    for previous, current in zip(case['trace'],case['trace'][1:]):
        delta = current['writes']-previous['writes']
        assert delta in (0,1)
        if delta:
            assert current['to'] is not None
            if current['from'] is not None:
                source = previous['oldCells'] or previous['cells']
                assert current['cells'][current['to']] == source[current['from']]

for case in fixtures['text']:
    m = case['model']
    text = unicodedata.normalize('NFC',m['original']) if case['normalize'] else m['original']
    assert m['text'] == text and m['length'] == len(text)
    assert m['bytes'] == list(text.encode('utf-8'))
    for index, unit in enumerate(m['units']):
        assert unit['char'] == text[index]
        assert unit['code'] == f'U+{ord(text[index]):04X}'
        assert unit['byteOffset'] == len(text[:index].encode('utf-8'))
        assert unit['bytes'] == list(text[index].encode('utf-8'))
for case in fixtures['hashes']:
    table = {10:2,14:5,18:1}
    key, capacity = case['key'],case['capacity']
    candidates = [k for k in table if k % capacity == key % capacity]
    comparisons = candidates.index(key)+1 if key in candidates else len(candidates)
    if case['operation'] == 'set':
        table[key] = 99
    end = case['trace'][-1]
    observed = {e['key']:e['value'] for bucket in end['buckets'] for e in bucket}
    assert observed == table and end['result'] == table.get(key,'absent')
    assert end['comparisons'] == comparisons
    assert sum(map(len,end['buckets'])) == len(table)
    for bucket, entries in enumerate(end['buckets']):
        assert all(e['key'] % capacity == bucket for e in entries)

translator = namespace('os.translation')
for m in fixtures['translations']:
    page, offset = divmod(m['address'],16)
    assert (m['page'],m['offset']) == (page,offset)
    if page == 3:
        expected,kind = 'unmapped','unmapped'
    elif page == 0 and m['access'] == 'write':
        expected,kind = 'protection fault','protection'
    elif page == 2:
        expected,kind = 'valid page: fault service required','demand'
    else:
        frame = 0 if page == 0 else (3 if m['process'] == 'A' else 5)
        expected,kind = frame*16+offset,'resident'
    assert translator['translate'](m['process'],m['address'],m['access']=='write') == expected
    assert m['kind'] == kind
    assert m['physical'] == (expected if isinstance(expected,int) else 6*16+offset if kind=='demand' else None)
translator['PAGE_SIZE'] = 32
assert translator['translate']('A',38) == 102
worker = namespace('os.worker')['run_job']
assert worker('-2') == {'ok':True,'value':4}
assert worker('') == {'ok':False,'status':2,'error':'invalid integer'}
assert worker('0') == {'ok':True,'value':0}

# Independent grammar recognizer: recursively parse groups, without a mutable stack.
def grammar(text):
    closers = {'(':')','[':']','{':'}'}
    def group(position, expected=None):
        while position < len(text):
            char = text[position]
            if char in closers:
                position, ok = group(position+1,closers[char])
                if not ok:
                    return position, False
            elif char in ')]}':
                return position+1, char == expected
            else:
                position += 1
        return position, expected is None
    position, ok = group(0)
    return ok and position == len(text)
balanced = namespace('linked.brackets')['balanced']
for case in fixtures['brackets']:
    expected = grammar(case['text'])
    assert case['result'] == expected
    assert balanced(case['text']) == expected
for text in ['{[()]}','[x]','{]','hello','"("']:
    assert balanced(text) == grammar(text)

for case in fixtures['rings']:
    logical, served = deque(), []
    for (op,*argument), state in zip(fixtures['queueEvents'],case['trace'][1:]):
        if op == 'put' and len(logical) < case['capacity']:
            logical.append(argument[0])
        elif op == 'take' and logical:
            served.append(logical.popleft())
        assert state['logical'] == list(logical) and state['output'] == served
        assert [state['cells'][(state['head']+i)%case['capacity']] for i in range(state['size'])] == list(logical)

rev = namespace('linked.reverse')
remove = namespace('linked.remove')
unique = namespace('arrays.unique')['first_unique']
sequence_cases = 0
def node_ids(head):
    ids = []
    while head is not None:
        assert id(head) not in ids
        ids.append(id(head))
        head = head.next
    return ids
for length in range(6):
    for seq in itertools.product(range(3),repeat=length):
        seq = list(seq)
        sequence_cases += 1
        head = rev['build'](seq)
        before = node_ids(head)
        head = rev['reverse'](head)
        assert rev['values'](head) == seq[::-1] and node_ids(head) == before[::-1]
        expected = next((v for v in seq if seq.count(v)==1),None)
        assert unique(seq) == expected
        for target in range(4):
            head = remove['build'](seq)
            before = node_ids(head)
            wanted, remaining = list(seq), list(before)
            if target in wanted:
                index = wanted.index(target)
                wanted.pop(index); remaining.pop(index)
            head = remove['remove_first'](head,target)
            assert remove['values'](head) == wanted and node_ids(head) == remaining

ring = namespace('linked.ring')['RingQueue']
window = namespace('linked.window')['recent_means']
for invalid in [0,-1,True,False,1.5,'3']:
    for make in [lambda:ring(invalid),lambda:window([],invalid)]:
        try:
            make()
        except ValueError:
            pass
        else:
            raise AssertionError(('accepted invalid capacity/width',invalid))
rng = random.Random(37)
for capacity in range(1,8):
    q, reference = ring(capacity),deque()
    for _ in range(500):
        previous = (list(q.data),q.head,q.size)
        if rng.random()<.58:
            value = rng.choice([None,0,1,2,'A'])
            if len(reference)==capacity:
                try:q.put(value)
                except OverflowError:pass
                else:raise AssertionError('accepted full enqueue')
                assert (q.data,q.head,q.size)==previous
            else:q.put(value);reference.append(value)
        elif reference:
            assert q.get()==reference.popleft()
        else:
            try:q.get()
            except IndexError:pass
            else:raise AssertionError('accepted empty dequeue')
            assert (q.data,q.head,q.size)==previous
        assert q.size==len(reference)
        assert [q.data[(q.head+i)%capacity] for i in range(q.size)]==list(reference)
for width in range(1,9):
    for readings in [[],[0],[2,4,8,10],[1,-1,2,0,8,-3]]:
        oracle = [sum(readings[max(0,i-width+1):i+1])/len(readings[max(0,i-width+1):i+1]) for i in range(len(readings))]
        assert all(math.isclose(a,b) for a,b in zip(window(readings,width),oracle))
        assert len(window(readings,width))==len(oracle)
lru=namespace('linked.lru')
before=list(lru['cache'].items())
try:lru['read']('missing')
except KeyError:pass
else:raise AssertionError('invented cache value')
assert list(lru['cache'].items())==before
print(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'programsPassed':ran,'skippedPlatformPrograms':skipped,'bracketStrings':len(fixtures['brackets']),'exhaustiveSequences':sequence_cases,'ringOperations':3500,'oracles':'Python list/dict/Unicode, divmod, recursive grammar, preserved Node identities, deque, direct window means'},indent=2))
