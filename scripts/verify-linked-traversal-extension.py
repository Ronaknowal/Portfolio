"""Independently enumerate paths, forward candidates and histogram intervals."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
directory = root / "scratch/linked-traversal-extension-verification"
payload = json.loads((directory / "payload.json").read_text(encoding="utf-8"))
spaces = {}
for name, program in (payload["originalPrograms"] | payload["programs"]).items():
    run = subprocess.run([sys.executable, "-c", program["code"]], capture_output=True, text=True, encoding="utf-8", check=True)
    assert run.stdout.strip() == program["output"].strip(), name
    assert not run.stderr
    namespace = {}
    with redirect_stdout(StringIO()):
        exec(compile(program["code"], name, "exec"), namespace)
    spaces[name] = namespace


def orbit(next_nodes, head):
    visited = {}
    order = []
    current = head
    while current is not None and current not in visited:
        visited[current] = len(order)
        order.append(current)
        current = next_nodes[current]
    return current, visited.get(current), 0 if current is None else len(order) - visited[current], order


def follow(next_nodes, head, count):
    for _ in range(count):
        if head is None:
            break
        head = next_nodes[head]
    return head


for row in payload["cycles"]:
    next_nodes, head, actual = row["next"], row["head"], row["actual"]
    entry, prefix, length, order = orbit(next_nodes, head)
    assert (actual["entry"], actual["prefixLength"], actual["cycleLength"]) == (entry, prefix, length)
    assert actual["next"] == next_nodes
    for frame in actual["frames"]:
        if frame["phase"] == "detect":
            t = frame["rounds"]
            assert frame["slow"] == follow(next_nodes, head, t)
            assert frame["fast"] == follow(next_nodes, head, 2*t)
            assert frame["via"] == follow(next_nodes, head, 2*t-1)
    if entry is not None:
        t = next(frame["rounds"] for frame in actual["frames"] if frame["phase"] == "meeting")
        assert t > 0 and t % length == 0 and t <= (prefix + length)
        assert all(follow(next_nodes, head, before) != follow(next_nodes, head, 2*before) for before in range(1,t))
    Node = spaces["cycle"]["Node"]
    nodes = [Node(f"n{i}", 7) for i in range(len(next_nodes))]
    for node, target in zip(nodes, next_nodes):
        node.next = None if target is None else nodes[target]
    start = None if head is None else nodes[head]
    before = tuple(node.next for node in nodes)
    result = spaces["cycle"]["cycle_entry"](start)
    assert result is (None if entry is None else nodes[entry])
    assert spaces["cycle"]["entry_by_seen"](start) is result
    assert spaces["cycle"]["cycle_size"](result) == length
    assert all(node.next is old for node, old in zip(nodes,before))


def distances(values, inclusive):
    return [next((j-i for j in range(i+1,len(values)) if values[j] >= value if inclusive), 0) if inclusive else next((j-i for j in range(i+1,len(values)) if values[j] > value), 0) for i,value in enumerate(values)]


for row in payload["arrays"] + payload["signed"]:
    values = row["values"]
    for inclusive in [False, True]:
        expected = distances(values, inclusive)
        assert row["inclusive" if inclusive else "strict"] == expected
        native, pushes, pops = spaces["greater"]["next_distances"](values, inclusive)
        assert native == expected and pushes == len(values) and pops <= pushes
    if "area" not in row:
        continue
    left = [next((j for j in range(i-1,-1,-1) if values[j] < value), -1) for i,value in enumerate(values)]
    right = [next((j for j in range(i+1,len(values)) if values[j] < value), len(values)) for i,value in enumerate(values)]
    assert row["left"] == left and row["right"] == right
    candidates = [(min(values[start:end])*(end-start),start,end) for start in range(len(values)) for end in range(start+1,len(values)+1)]
    optimum = max((area for area,_,_ in candidates), default=0)
    assert row["area"] == optimum
    native_area, witness = spaces["histogram"]["largest_rectangle"](values)
    assert native_area == optimum
    assert spaces["histogram"]["smaller_boundaries"](values) == (left,right)
    if optimum:
        start,end,height = witness
        assert min(values[start:end]) >= height and height*(end-start) == optimum
        model = row["witness"]
        assert min(values[model["start"]:model["end"]]) >= model["height"]
        assert model["area"] == optimum
    else:
        assert witness is None and row["witness"] is None

for state in payload["middles"]:
    n = state["length"]
    expected = None if n == 0 else (n-1)//2 if state["policy"] == "first" else n//2
    assert state["middle"] == expected
    assert state["leftSize"] == (n+1)//2 and state["rightSize"] == n//2
    if state["cut"] and n:
        assert state["next"][(n-1)//2] is None
        assert state["rightHead"] == ((n+1)//2 if n > 1 else None)
for n in range(41):
    head, nodes = spaces["middle"]["make_chain"]([7]*n)
    assert spaces["middle"]["middle"](head) is (nodes[n//2] if n else None)
    assert spaces["middle"]["middle"](head,True) is (nodes[(n-1)//2] if n else None)
    left,right = spaces["middle"]["split_left_heavy"](head)
    def identities(start):
        result = []
        while start is not None:
            assert start not in result
            result.append(start)
            start = start.next
        return result
    assert identities(left) == nodes[:(n+1)//2]
    assert identities(right) == nodes[(n+1)//2:]
for n in range(1,20):
    for entry in range(n):
        head,nodes = spaces["middle"]["make_chain"]([7]*n,entry)
        old = tuple(node.next for node in nodes)
        try:
            spaces["middle"]["split_left_heavy"](head)
            raise AssertionError("cyclic split accepted")
        except ValueError:
            pass
        assert all(node.next is link for node,link in zip(nodes,old))

result = {"checkedAt":datetime.now(timezone.utc).isoformat(),"status":"passed","productionSources":payload["sources"],"actualPrograms":10,"originalProgramsConserved":6,"originalTeachingElementsConserved":payload["originalTeachingElements"],"existingPracticePlacementsConserved":10,"newPracticePlacements":[142,876,739,84],"successorCases":len(payload["cycles"]),"exhaustiveNonnegativeArrays":len(payload["arrays"]),"signedArrays":len(payload["signed"]),"middleModelStates":len(payload["middles"]),"nativeMiddleLengths":41,"cyclicSplitRejections":190,"invalidModels":payload["rejected"],"oracleMethods":["independent first-visit orbit and direct repeated successor evaluation","forward nearest qualifying scans","all histogram intervals with direct minimum","actual Node identity partition and mutation rejection"],"limits":"Bounded exhaustive and changed actual-source checks, not a proof of arbitrary runtime inputs; production integration and browser review separate."}
(directory / "results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
