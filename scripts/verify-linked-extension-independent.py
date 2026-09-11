"""Independent native/object checks for the frozen linked/stack extension."""
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

directory = Path('scratch/linked-extension-independent')
payload = json.loads((directory / 'payload.json').read_text(encoding='utf-8'))
namespaces, outputs = {}, []
for name, example in payload['examples'].items():
    namespace = {'__name__': '__main__'}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['filename'], 'exec'), namespace)
    assert output.getvalue().strip() == example['output'].strip()
    namespaces[name] = namespace
    outputs.append({'name': name, 'stdout': output.getvalue().strip(), 'codeSha256': hashlib.sha256(example['code'].encode()).hexdigest()})

cycle = namespaces['cycle']
for case in payload['graphs']:
    nodes = [cycle['Node'](f'n{i}', 7) for i in range(len(case['next']))]
    for index, target in enumerate(case['next']):
        nodes[index].next = None if target is None else nodes[target]
    before = tuple(node.next for node in nodes)
    head = None if case['head'] is None else nodes[case['head']]
    entry = cycle['cycle_entry'](head)
    expected = None if case['entry'] is None else nodes[case['entry']]
    assert entry is expected
    assert cycle['entry_by_seen'](head) is expected
    assert cycle['cycle_size'](entry) == case['cycleLength']
    assert all(node.next is old for node, old in zip(nodes, before))

middle = namespaces['middle']
long_lengths = [0, 1, 2, 3, 6, 7, 41, 42, 1001, 1002, 10000]
for length in long_lengths:
    head, nodes = middle['make_chain']([7] * length)
    first, second = middle['middle'](head, True), middle['middle'](head)
    assert first is (nodes[(length - 1) // 2] if length else None)
    assert second is (nodes[length // 2] if length else None)
    before = tuple(node.next for node in nodes)
    left, right = middle['split_left_heavy'](head)
    collected = []
    for start in (left, right):
        current, part = start, []
        while current is not None:
            assert current not in part
            part.append(current)
            current = current.next
        collected.append(part)
    assert collected[0] + collected[1] == nodes
    assert len(collected[0]) == (length + 1) // 2
    assert len(collected[1]) == length // 2
    if length > 1:
        assert [i for i, (node, previous) in enumerate(zip(nodes, before)) if node.next is not previous] == [(length - 1) // 2]
    else:
        assert all(node.next is previous for node, previous in zip(nodes, before))

for case in payload['arrays']:
    assert namespaces['greater']['next_distances'](case['values'])[0] == case['strict']
    assert namespaces['greater']['next_distances'](case['values'], True)[0] == case['inclusive']
    left, right = namespaces['histogram']['smaller_boundaries'](case['heights'])
    assert left == case['left'] and right == case['right']
    area, witness = namespaces['histogram']['largest_rectangle'](case['heights'])
    assert area == case['area']
    if area:
        start, end, height = witness
        assert (end - start) * height == area
        assert height <= min(case['heights'][start:end])
    else:
        assert witness is None

changed = {
    'strict': namespaces['greater']['next_distances']([-5, -5, 0, -1, 0, 1])[0],
    'inclusive': namespaces['greater']['next_distances']([-5, -5, 0, -1, 0, 1], True)[0],
    'histogram': namespaces['histogram']['largest_rectangle']([0, 3, 3, 1, 4, 4, 0]),
    'bothBoundaries': namespaces['histogram']['smaller_boundaries']([0, 3, 3, 1, 4, 4, 0]),
}
assert changed['strict'] == [2, 1, 3, 1, 1, 0]
assert changed['inclusive'] == [1, 1, 2, 1, 1, 0]
assert changed['histogram'] == (8, (4, 6, 4))
for source in payload['sources']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'sources': payload['sources'],
          'authorPacketSha256': payload['authorPacketSha256'], 'programs': outputs,
          'checks': {'allSuccessorGraphsAndHeadsThroughFiveNodes': len(payload['graphs']),
                     'middleModelStates': len(payload['middleCases']), 'nativeLongLengths': long_lengths,
                     'lengthSevenTernaryHistogramsAndSignedFutureArrays': len(payload['arrays']),
                     'frameMeaningAndStackPrefixInvariants': True, 'nodeIdentityAndNonmutation': True},
          'changedBrowserContracts': changed,
          'limits': 'Complementary exhaustive/object/witness checks support independently read proofs. No judge submissions or runtime benchmark.'}
(directory / 'native-results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Independent linked/stack native and frame/witness checks passed.')
