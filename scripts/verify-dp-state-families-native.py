from collections import Counter
import contextlib
from datetime import datetime, timezone
from functools import cache
import hashlib
import io
from itertools import permutations
import json
import math
from pathlib import Path

data = json.loads(Path('scratch/dp-state-families-verification/fixtures.json').read_text(encoding='utf-8'))
checks = Counter()
environments = {}
for name, example in data['examples'].items():
    environment, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], f'actual-{name}.py', 'exec'), environment)
    assert output.getvalue().rstrip() == example['expected']
    environments[name] = environment
    checks['complete_new_program_stdout'] += 1

@cache
def shapes(size):
    if size == 1:
        return (None,)
    return tuple((left, first, second) for left in range(1, size)
                 for first in shapes(left) for second in shapes(size-left))

def expression_cost(tree, dimensions, first=0):
    if tree is None:
        return 0, first+1
    length, left, right = tree
    first_cost, split = expression_cost(left, dimensions, first)
    second_cost, last = expression_cost(right, dimensions, split)
    return first_cost+second_cost+dimensions[first]*dimensions[split]*dimensions[last], last

for case in data['matrices']:
    dimensions = case['dimensions']
    expected = min(expression_cost(tree, dimensions)[0] for tree in shapes(len(dimensions)-1))
    actual, expression, operations = environments['matrix']['matrix_chain'](dimensions)
    assert expected == actual == case['result']
    live = {(index,index+1) for index in range(len(dimensions)-1)}
    total = 0
    for left, split, right in operations:
        assert (left,split) in live and (split,right) in live
        live.remove((left,split)); live.remove((split,right)); live.add((left,right))
        total += dimensions[left]*dimensions[split]*dimensions[right]
    assert live == {(0,len(dimensions)-1)} and total == actual
    assert [(row['left'],row['split'],row['right']) for row in case['operations']] == operations
    assert case['expression'].replace('×','*') == expression, (dimensions, case['expression'], expression)
    checks['all_parenthesizations_and_actual_witness'] += 1

def removal_score(values, order):
    sequence = [(index,value) for index,value in enumerate(values)]
    score = 0
    for target in order:
        position = next(index for index,entry in enumerate(sequence) if entry[0] == target)
        before = sequence[position-1][1] if position else 1
        after = sequence[position+1][1] if position+1<len(sequence) else 1
        score += before*sequence[position][1]*after
        del sequence[position]
    return score

for case in data['balloons']:
    values = case['values']
    expected = max(removal_score(values, order) for order in permutations(range(len(values))))
    native, order = environments['balloons']['balloon_plan'](values)
    assert native == expected == case['result']
    assert sorted(order) == list(range(len(values))) and order == case['order']
    assert removal_score(values,order) == expected
    assert case['replay'][-1]['total'] == expected
    checks['all_live_removal_orders_and_actual_witness'] += 1

def subset_optimum(weights, edges, allowed, forbidden=None):
    allowed = sorted(allowed)
    best = 0
    for mask in range(1 << len(allowed)):
        selected = {node for bit,node in enumerate(allowed) if mask & (1 << bit)}
        if forbidden in selected or any(first in selected and second in selected for first,second in edges):
            continue
        best = max(best,sum(weights[node] for node in selected))
    return best

for case in data['trees']:
    weights, edges = case['weights'], case['edges']
    for query in case['queries']:
        allowed, stack = set(), [query['node']]
        while stack:
            current = stack.pop(); allowed.add(current); stack.extend(case['children'][current])
        expected = subset_optimum(weights,edges,allowed,query['node'] if query['parent'] else None)
        selected = query['selected']
        assert len(selected) == len(set(selected)) and set(selected) <= allowed
        assert not any(first in selected and second in selected for first,second in edges)
        assert not query['parent'] or query['node'] not in selected
        assert query['result'] == expected == sum(weights[node] for node in selected)
        checks['exhaustive_subtree_boundary_witness'] += 1
    for parent in [False,True]:
        expected = subset_optimum(weights,edges,set(range(len(weights))),case['root'] if parent else None)
        value,witness = environments['tree']['independent_tree'](weights,edges,case['root'],parent)
        assert value == expected == sum(weights[node] for node in witness)
        assert not any(first in witness and second in witness for first,second in edges)
        checks['actual_native_tree_roots_and_boundaries'] += 1

count_distinct = environments['digits']['count_distinct']
count_range = environments['digits']['count_range']
cumulative = 0
for case in data['digits']:
    bound = case['bound']
    if bound > 0 and len(set(str(bound))) == len(str(bound)):
        cumulative += 1
    assert case['count'] == cumulative == count_distinct(bound)
    checks['all_small_bounds_actual_native_and_model'] += 1
for case in data['prefixes']:
    values = [value for value in range(1,case['bound']+1)
              if len(set(str(value))) == len(str(value))
              and str(value).zfill(len(case['digits'])).startswith(case['prefix'])]
    assert case['remaining'] == len(values)
    assert case['completion'] == (min(values) if values else None)
    for branch in case['branches']:
        selected = [value for value in values if str(value).zfill(len(case['digits']))[len(case['prefix'])] == str(branch['digit'])]
        assert len(selected) == branch['count']
    checks['exact_prefix_sets_and_each_digit_partition'] += 1
for length in range(1,19):
    expected = sum(9*math.perm(9,size-1) for size in range(1,min(length,10)+1))
    assert count_distinct(10**length-1) == expected
    checks['large_all_nines_combinatorial_length_counts'] += 1
for first in range(0,250,11):
    for last in range(first,251,23):
        expected = sum(value>0 and len(set(str(value)))==len(str(value)) for value in range(first,last+1))
        assert count_range(first,last) == expected
        checks['actual_changed_inclusive_ranges'] += 1

# Changed local tasks and deep native structures, outside tiny display limits.
assert environments['matrix']['matrix_chain']([10**40]*4)[0] == 2*10**120
checks['large_integer_matrix_arithmetic'] += 1
assert environments['tree']['independent_tree']([1]*10000,[(index,index+1) for index in range(9999)])[0] == 5000
checks['actual_10000_node_chain_no_recursion'] += 1
for first,last,expected in [(100,130,19),(0,0,1)]:
    valid = [value for value in range(first,last+1) if all(a!=b for a,b in zip(str(value),str(value)[1:]))]
    assert len(valid) == expected
    checks['changed_adjacent_property_and_zero'] += 1
assert subset_optimum([4,3,3,3],[[0,1],[0,2],[0,3],[1,2]],set(range(4))) == 6
checks['changed_cross_child_conflict'] += 1
invalid = [lambda:count_distinct(-1),lambda:count_distinct(True),lambda:count_distinct(10**18+1),
           lambda:count_range(10,1),lambda:count_range(0,1.5),
           lambda:environments['tree']['independent_tree']([1,2,3,4],[(0,1),(1,2),(2,0)]),
           lambda:environments['tree']['independent_tree']([1,2],[(0,0)]),
           lambda:environments['tree']['independent_tree']([1,2,3],[(0,1),(1,0)]),
           lambda:environments['tree']['independent_tree']([1,True],[(0,1)]),
           lambda:environments['tree']['independent_tree']([1],[],parent_selected=1),
           lambda:environments['matrix']['matrix_chain']([1,0]),
           lambda:environments['matrix']['matrix_chain']([True,3]),
           lambda:environments['balloons']['balloon_plan']([-1])]
for call in invalid:
    try: call()
    except ValueError: checks['native_rejected_inputs'] += 1
    else: raise AssertionError('Invalid native input accepted')
checks['model_rejected_inputs'] = data['rejectedInputs']
sources = ['src/learn/data/dp-state-families-models.js','src/learn/data/dp-state-families-examples.js']
record = {'checkedAt':datetime.now(timezone.utc).isoformat(),'checks':dict(checks),
          'sourceHashes':{source:hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources},
          'scope':'Exact finite oracles, actual displayed-program execution and changed-input checks; not a proof of all algorithms or browser verification.'}
Path('docs/teaching/evidence/dp-state-families-native-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
