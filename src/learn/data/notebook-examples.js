export const notebookExamples = {
  state: {
    code: `# A small simulation of a notebook's shared Python namespace.
cells = ["rate = 2", "cost = rate * 10", "print(cost)"]
kernel = {}
for cell in cells:
    exec(cell, kernel)
exec("rate = 3", kernel)
exec(cells[2], kernel)  # cost was not recalculated
fresh_kernel = {}
try:
    exec(cells[2], fresh_kernel)
except NameError:
    print("fresh kernel: cost is not defined")
for cell in ["rate = 3", cells[1], cells[2]]:
    exec(cell, fresh_kernel)`,
    output: "20\n20\nfresh kernel: cost is not defined\n30",
  },
  pure: {
    code: `def summarise(values, *, offset):
    if not values:
        raise ValueError("at least one measurement is required")
    adjusted = [value - offset for value in values]
    return {"count": len(adjusted), "mean": sum(adjusted) / len(adjusted)}

raw = [10, 20, 30]
first = summarise(raw, offset=2)
second = summarise(raw, offset=2)
print(first)
print("repeat agrees:", first == second)
print("input unchanged:", raw)`,
    output: "{'count': 3, 'mean': 18.0}\nrepeat agrees: True\ninput unchanged: [10, 20, 30]",
  },
  random: {
    code: `import numpy as np

def draw(seed):
    return np.random.default_rng(seed).integers(0, 100, size=5)

print("fresh generators agree:", np.array_equal(draw(7), draw(7)))
rng = np.random.default_rng(7)
first = rng.integers(0, 100, size=5)
second = rng.integers(0, 100, size=5)
print("successive draws agree:", np.array_equal(first, second))
split_seed, model_seed = np.random.SeedSequence(7).spawn(2)
split_rng = np.random.default_rng(split_seed)
model_rng = np.random.default_rng(model_seed)
saved = model_rng.bit_generator.state
split_rng.normal(size=100)  # does not consume model_rng
print("model stream untouched:", model_rng.bit_generator.state == saved)`,
    output: "fresh generators agree: True\nsuccessive draws agree: False\nmodel stream untouched: True",
  },
  leakage: {
    code: `import numpy as np

train = np.array([10., 20., 30.])
validation = np.array([100.])
train_mean = train.mean()
print("training mean:", float(train_mean))
print("validation centred with training mean:", (validation - train_mean).tolist())
leaked_mean = np.concatenate([train, validation]).mean()
print("leaked mean:", float(leaked_mean))
print("incorrect centred validation:", (validation - leaked_mean).tolist())`,
    output: "training mean: 20.0\nvalidation centred with training mean: [80.0]\nleaked mean: 40.0\nincorrect centred validation: [60.0]",
  },
  manifest: {
    code: `from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import platform
import tempfile

@dataclass(frozen=True)
class Config:
    offset: float = 2.0
    seed: int = 7

def run(data, config):
    values = json.loads(data)
    if not values:
        raise ValueError("empty input")
    return {"count": len(values), "mean": sum(values) / len(values) - config.offset}

data = b"[10, 20, 30]"
config = Config()
result = run(data, config)
manifest = {
    "config": asdict(config), "data_sha256": sha256(data).hexdigest(),
    "python": platform.python_version(), "numpy": version("numpy"),
    "code_revision": "lesson-example-v1", "result": result,
}
# Temporary storage keeps this standalone demonstration safe to rerun.
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    saved = json.loads(path.read_text(encoding="utf-8"))
    replay = run(data, Config(**saved["config"]))
    print("saved keys:", sorted(saved))
    print("same data:", sha256(data).hexdigest() == saved["data_sha256"])
    print("replay:", replay)
    print("same result:", replay == saved["result"])`,
    output: "saved keys: ['code_revision', 'config', 'data_sha256', 'numpy', 'python', 'result']\nsame data: True\nreplay: {'count': 3, 'mean': 18.0}\nsame result: True",
  },
  execute: {
    code: `from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell("# Offset-adjusted measurements\\nOne row is one measurement; units are milliseconds."),
    nbformat.v4.new_code_cell("values = [10, 20, 30]\\noffset = 2"),
    nbformat.v4.new_code_cell("adjusted = [x - offset for x in values]\\nmean = sum(adjusted) / len(adjusted)"),
    nbformat.v4.new_code_cell("assert len(adjusted) == 3\\nassert mean == 18.0\\nprint(mean)"),
])
source = Path("offset-analysis.ipynb")
destination = Path("offset-analysis.executed.ipynb")
if source.exists() or destination.exists():
    raise FileExistsError("choose a new folder; refusing to overwrite a notebook")
nbformat.write(notebook, source)
runner = ExecutePreprocessor(timeout=60, kernel_name="python3", allow_errors=False)
runner.preprocess(notebook, {"metadata": {"path": str(Path.cwd())}})
nbformat.write(notebook, destination)
print("cell output:", notebook.cells[-1].outputs[0].text.strip())
print("executed code cells:", [c.execution_count for c in notebook.cells if c.cell_type == "code"])
print("saved:", destination.name)`,
    output: "cell output: 18.0\nexecuted code cells: [1, 2, 3]\nsaved: offset-analysis.executed.ipynb",
    artifacts: ["offset-analysis.ipynb", "offset-analysis.executed.ipynb"],
  },
};
