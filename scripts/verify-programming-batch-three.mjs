import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {collectLessonExamples} from './lib/lesson-examples.mjs';
import { plottingExamples } from "../src/learn/data/plotting-examples.js";
import { notebookExamples } from "../src/learn/data/notebook-examples.js";
import { apiExamples } from "../src/learn/data/api-design-examples.js";

const python = path.resolve(process.env.LESSON_PYTHON || "scratch/lesson-tools/Scripts/python.exe");
const runtime = path.resolve("scratch/programming-three-runtime");
const assets = path.resolve("public/learn-assets/plots");
const notebookAssets = path.resolve("public/learn-assets/notebooks");
fs.mkdirSync(path.join(runtime, "kernels/python3"), { recursive: true });
fs.mkdirSync(assets, { recursive: true });
fs.mkdirSync(notebookAssets, { recursive: true });
fs.writeFileSync(path.join(runtime, "kernels/python3/kernel.json"), JSON.stringify({ argv: [python, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name: "Lesson verification", language: "python" }));
const env = { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: "", MPLBACKEND: "Agg", MPLCONFIGDIR: path.join(runtime, "mpl"), JUPYTER_PATH: runtime, JUPYTER_RUNTIME_DIR: path.join(runtime, "jupyter"), IPYTHONDIR: path.join(runtime, "ipython") };
const normal = text => text.replace(/\r\n/g, "\n").trimEnd();
function run(code, cwd, timeout = 90000) {
  const result = spawnSync(python, ["-B", "-c", code], { cwd, env, encoding: "utf8", timeout });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  return result.stdout;
}
const groups = [
  ["matplotlib-scientific-plotting", plottingExamples],
  ["reproducible-notebooks-experiment-structure", notebookExamples],
  ["code-documentation-type-hints-api-design", apiExamples],
];
const plotChecks = {
  curves: "assert len(ns['ax'].lines) == 2\nnp.testing.assert_array_equal(ns['ax'].lines[1].get_ydata(), ns['valid'])\nassert ns['ax'].get_xlabel() == 'Epoch'",
  bars: "assert [bar.get_height() for bar in ns['bars']] == [10, 20]\nassert ns['ax'].get_ylim()[0] == 0",
  scatter: "np.testing.assert_array_equal(ns['ax'].collections[0].get_offsets(), np.column_stack([ns['size'], ns['latency']]))",
  histogram: "assert sum(ns['counts']) == len(ns['latency'])\nnp.testing.assert_allclose(sum(ns['density'] * np.diff(ns['edges'])), 1)\nnp.testing.assert_array_equal([patch.get_height() for patch in ns['axes'][0].patches], ns['counts'])",
  uncertainty: "np.testing.assert_array_equal(ns['ax'].containers[-1].lines[0].get_ydata(), ns['mean'])\nnp.testing.assert_array_equal(ns['ax'].containers[-1].lines[2][0].get_segments(), [[[0, 10], [0, 14]], [[1, 13], [1, 15]]])",
  heatmap: "np.testing.assert_array_equal(ns['image'].get_array(), ns['counts'])\nassert ns['image'].get_clim() == (0, 10)\nassert ns['image'].origin == 'upper'",
  scales: "assert ns['axes'][0].get_yscale() == 'linear'\nassert ns['axes'][1].get_yscale() == 'log'\nnp.testing.assert_array_equal(ns['axes'][1].lines[0].get_ydata(), ns['error'])",
};
for (const [slug, examples] of groups) {
  for (const [id, example] of Object.entries(examples)) {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lesson-three-"));
    fs.writeFileSync(path.join(directory, "example.py"), example.code);
    let check = "import runpy\nns = runpy.run_path('example.py', run_name='__main__')";
    if (example.artifact) check += "\nimport numpy as np\n" + plotChecks[id] + "\nns['fig'].savefig('preview.png', dpi=120)\nassert len(ns['fig'].axes) >= 1";
    const output = run(check, directory);
    assert.equal(normal(output), normal(example.output), id + " output mismatch");
    if (id === "execute") run(`from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
notebook = nbformat.read("offset-analysis.ipynb", as_version=4)
runner = ExecutePreprocessor(timeout=60, kernel_name="python3", allow_errors=False)
runner.preprocess(notebook, {"metadata": {"path": str(Path.cwd())}})
assert notebook.cells[-1].outputs[0].text.strip() == "18.0"
notebook.cells[1].source = notebook.cells[1].source.replace("offset = 2", "offset = 5")
try:
    runner.preprocess(notebook, {"metadata": {"path": str(Path.cwd())}})
except CellExecutionError:
    pass
else:
    raise AssertionError("Stale expectation did not fail")
notebook.cells[-1].source = notebook.cells[-1].source.replace("18.0", "15.0")
runner.preprocess(notebook, {"metadata": {"path": str(Path.cwd())}})
assert notebook.cells[-1].outputs[0].text.strip() == "15.0"
try:
    exec(Path("example.py").read_text(encoding="utf-8"))
except FileExistsError:
    pass
else:
    raise AssertionError("Existing notebook would be overwritten")`, directory);
    for (const artifact of [example.artifact, ...(example.artifacts || [])].filter(Boolean)) {
      const source = path.join(directory, artifact);
      assert.ok(fs.statSync(source).size > 0);
      fs.copyFileSync(source, path.join(artifact.endsWith(".ipynb") ? notebookAssets : assets, artifact));
    }
    if (example.artifact) fs.copyFileSync(path.join(directory, "preview.png"), path.join(runtime, id + ".png"));
    // Only these freshly created temporary directories are removed.
    fs.rmSync(directory, { recursive: true });
    console.log(slug + "/" + id + ": exact output verified");
  }
  const source = fs.readFileSync("src/learn/data/topics/" + slug + ".jsx", "utf8");
  const refs = (await collectLessonExamples('src/learn/data/topics/'+slug+'.jsx')).filter(reference=>Object.hasOwn(examples,reference.key)).map(reference=>reference.key);
  assert.deepEqual(refs.sort(), Object.keys(examples).sort(), "Example coverage: " + slug);
}

const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lesson-contract-"));
fs.writeFileSync(path.join(directory, "scores.py"), apiExamples.loader.code);
fs.writeFileSync(path.join(directory, "mean_api.py"), apiExamples.contract.code.split('if __name__ == "__main__":')[0]);
run(`from pathlib import Path
from scores import load_scores
from mean_api import mean_ms
import json
import math
path = Path("scores.json")
for invalid in ['[]', '{"x": -1}', '{"x": 2}', '{"x": null}', '{"x": true}', '{"x": "0.5"}', '{" ": 0.5}', '{"x": NaN}']:
    path.write_text(invalid, encoding="utf-8")
    try:
        load_scores(path, minimum=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(invalid)
path.write_text('{"zero": 0, "boundary": 0.85, "one": 1}', encoding="utf-8")
assert load_scores(path, minimum=0.85) == {"boundary": 0.85, "one": 1.0}
assert load_scores(path)["zero"] == 0.0
for invalid in [-1, 2, True, math.inf, math.nan, 10**1000]:
    try:
        load_scores(path, minimum=invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(invalid)
assert mean_ms([10., 20., 30.], offset_ms=5.) == 15.
assert mean_ms([4.], offset_ms=4.) == 0.
print("API boundary checks passed")`, directory);
fs.writeFileSync(path.join(directory, "bad_types.py"), apiExamples.hints.code);
const bad = spawnSync(python, ["-m", "mypy", "--strict", "--cache-dir", path.join(directory, "cache"), "bad_types.py"], { cwd: directory, env, encoding: "utf8", timeout: 60000 });
assert.equal(bad.status, 1, bad.stderr);
assert.match(bad.stdout, /arg-type/);
// Separate modules avoid unrelated duplicate imports when checking all examples.
for (const id of ["contract", "optional", "defaults", "result", "loader"]) fs.writeFileSync(path.join(directory, id + ".py"), apiExamples[id].code.split('if __name__ == "__main__":')[0]);
const good = spawnSync(python, ["-m", "mypy", "--strict", "--cache-dir", path.join(directory, "cache"), ...["contract", "optional", "defaults", "result", "loader"].map(id => id + ".py")], { cwd: directory, env, encoding: "utf8", timeout: 60000 });
assert.equal(good.status, 0, good.stdout + good.stderr);
fs.rmSync(directory, { recursive: true });
console.log("19 runnable outputs, 7 real charts, notebook execution, API boundaries and positive/negative type checks verified.");
