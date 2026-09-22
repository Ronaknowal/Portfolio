"""Independent complementary review; preserve author evidence and outputs."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
manifest_path = ROOT / "docs/teaching/implementation-depth/dsa-remediation.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
records = []
hashes = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name, condition=True):
    assert condition, name
    records.append(name)


def load(topic, filename):
    path = ROOT / "public/learn-assets" / topic / filename
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


for topic in manifest["topics"]:
    for path, expected in topic["sourceHashes"].items():
        actual = sha(ROOT / path)
        assert actual == expected, f"Source changed after author freeze: {path}"
        hashes[path] = actual
    # This comparator does not invoke the author's extraction implementation.
    owner = topic["unchangedMechanismOwners"][0]
    script = f"import * as m from './{owner['path']}'; console.log(JSON.stringify(Object.values(m)[0]));"
    examples = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT,
                                         capture_output=True, text=True, check=True).stdout)
    wanted = {}
    for key in owner["exampleKeys"]:
        for statement in ast.parse(examples[key]["code"]).body:
            if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
                definition = ast.dump(statement, include_attributes=False)
                if statement.name in wanted:
                    assert wanted[statement.name] == definition
                wanted[statement.name] = definition
    download = next(path for path in topic["changedFiles"] if path.endswith("_mechanisms.py"))
    downloaded = {node.name: ast.dump(node, include_attributes=False)
                  for node in ast.parse((ROOT/download).read_text()).body
                  if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    record(topic["id"] + ": all extracted definitions/decorators match displayed owners", downloaded == wanted)

# A source permutation changes tie choices, but distances remain the graph property.
graph = load("graphs-representations-bfs-dfs", "graph_library.py")
vertices = [("node", 1), "other", 7, "isolate"]
edges = [(vertices[0], "other"), (vertices[0], 7), ("other", 7), (7, 7)]
first = dict(graph.compare(vertices, edges, vertices[0], True))
second = dict(graph.compare(vertices, list(reversed(edges)), vertices[0], True))
record("Traversal mixed hashable labels and neighbor-order invariant distances", first == second == {vertices[0]: 0, "other": 1, 7: 1, "isolate": None})

weighted = load("shortest-paths-spanning-trees-topological-ordering", "weighted_graph_library.py")
import networkx as nx
edges = [(0,1,2), (2,3,-2), (3,2,1)]
graph = weighted.weighted_graph(5, edges)
local, _ = weighted.bellman_ford(5, edges, 0)
library = nx.single_source_bellman_ford_path_length(graph, 0, weight="weight")
record("Bellman-Ford ignores a negative cycle unreachable from this source", local == [0,2,math.inf,math.inf,math.inf] and library == {0: 0, 1: 2})

strings = load("string-matching-prefix-functions-rolling-hashes", "string_search_library.py")
text, pattern = "🙂e\u0301🙂e\u0301🙂", "🙂e\u0301🙂"
matcher = strings.StreamMatcher(pattern)
streamed = []
for chunk in ["", *text, ""]:
    streamed.extend(matcher.feed(chunk))
expected = [index for index in range(len(text)+1) if text.startswith(pattern,index)]
record("Unicode one-code-point chunks preserve overlap coordinates without normalization", streamed == strings.builtin_occurrences(text,pattern) == expected == [0,3])

flow = load("network-flow-minimum-cuts-bipartite-matching", "flow_library.py")
base = [(0,1,5), (1,2,3), (0,2,1), (1,0,2)]
split = [(0,1,2), (0,1,3), (1,2,3), (0,2,1), (1,0,2), (0,0,100)]
record("Flow parallel splitting and self-loop addition preserve certified value", flow.compare(4,base,0,2) == flow.compare(4,split,0,2) == 4)

persistent = load("persistent-data-structures-structural-sharing-versioned-queries", "persistent_map_library.py")
root = persistent.build([1,2,3,4])
branch = persistent.assign(root,0,9)
record("Persistent point update preserves off-path identity and historical total", branch.right is root.right and persistent.range_sum(root,0,4) == 10 and persistent.range_sum(branch,0,4) == 18)

# Inspect actual sorter resource lifetime under failures absent from the happy path.
sorter = load("external-memory-algorithms-b-trees-i-o-complexity", "external_sort_stream.py")
with tempfile.TemporaryDirectory(prefix="dsa-review-") as directory:
    directory = Path(directory)
    source, destination = directory/"source.bin", directory/"new.bin"
    with source.open("wb") as stream:
        sorter.write_records(stream, [9,1,4,1,-2,8], 2)
    original_bytes = source.read_bytes()
    original_open = Path.open
    handles = []

    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name.startswith("run-"):
            handles.append(handle)
        return handle

    def fail_merge(*iterators):
        for iterator in iterators:
            next(iterator)
        raise OSError("injected merge-read failure")
        yield 0

    with patch.object(Path, "open", tracked_open), patch.object(sorter, "merge", fail_merge):
        try:
            sorter.sort_integer_file(source, destination, chunk_records=2, fan_in=2, buffer_records=1)
        except OSError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("fault did not run")
    record("External merge fault closes every opened run and cleans its workspace",
           handles and all(handle.closed for handle in handles) and not list(directory.glob("sorted-runs-*")) and not destination.exists() and source.read_bytes() == original_bytes)

    def competing_open(path, mode="r", *args, **kwargs):
        if path == destination and mode == "xb":
            path.write_bytes(b"another writer")
        return original_open(path, mode, *args, **kwargs)

    with patch.object(Path, "open", competing_open):
        try:
            sorter.sort_integer_file(source, destination, chunk_records=2, fan_in=2, buffer_records=1)
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive creation did not protect competing output")
    record("External final exclusive-create protects a destination appearing during sort",
           destination.read_bytes() == b"another writer" and not list(directory.glob("sorted-runs-*")))
    destination.unlink()

    def fail_copy(incoming, outgoing, length):
        outgoing.write(incoming.read(8))
        raise OSError("injected final-copy failure")

    with patch.object(sorter, "copyfileobj", fail_copy):
        try:
            sorter.sort_integer_file(source, destination, chunk_records=2, fan_in=2, buffer_records=1)
        except OSError:
            pass
    record("External final-copy failure matches documented partial-output boundary",
           destination.exists() and destination.stat().st_size == 8 and not list(directory.glob("sorted-runs-*")) and source.read_bytes() == original_bytes)

range_dir = ROOT/"public/learn-assets/segment-trees-fenwick-trees-range-queries"
provenance = json.loads((range_dir/"ac-library-provenance.json").read_text())
with zipfile.ZipFile(range_dir/"ac-library-v1.6-range-headers.zip") as archive:
    for entry in provenance["files"]:
        zipped = archive.read("ac-library-v1.6/" + entry["path"])
        assert hashlib.sha256(zipped).hexdigest() == entry["sha256"]
        assert sha(range_dir/"ac-library-v1.6"/entry["path"]) == entry["sha256"]
record("Every packaged AtCoder header/license equals its pinned provenance hash")
scratch = ROOT/"scratch/dsa-independent-review"
scratch.mkdir(exist_ok=True)
executable = scratch/"affine-action.exe"
fixture = "scripts/fixtures/dsa-range-action-independent.cpp"
subprocess.run([os.environ.get("LESSON_CXX", "C:/msys64/ucrt64/bin/g++.exe"), "-std=c++17", "-O2",
                f"-I{range_dir}/ac-library-v1.6", f"-I{range_dir}", str(ROOT/fixture), "-o", str(executable)],
               check=True, capture_output=True, text=True)
run = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
record("Compiled actual AtCoder adapter passes affine composition/action/empty laws and negative multiplier updates", "1215" in run.stdout)
hashes[fixture] = sha(ROOT/fixture)
hashes["scripts/review-dsa-depth-remediation.py"] = sha(Path(__file__))

result = {"date": "2026-09-22", "reviewer": "independent mathematics author; not DSA author",
          "status": "source-and-complementary-native-review-pass; production browser remains root-owned",
          "manifest": {"path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(manifest_path)},
          "checks": records, "count": len(records), "sourceHashes": hashes, "cppOutput": run.stdout.strip(),
          "findings": [{"id": "DSA-R1", "status": "resolved", "scope": "String Matching section 11 prose",
                        "resolution": "Replaced stale prediction-first instruction with direct run/inspect/compare; final source inspected."}]}
report = "docs/teaching/implementation-depth/DSA-REMEDIATION-INDEPENDENT.md"
if (ROOT/report).exists():
    result["reviewReport"] = {"path": report, "sha256": sha(ROOT/report)}
destination = ROOT/"docs/teaching/implementation-depth/DSA-REMEDIATION-INDEPENDENT.json"
destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(f"Independent DSA review: {len(records)} complementary groups passed")
