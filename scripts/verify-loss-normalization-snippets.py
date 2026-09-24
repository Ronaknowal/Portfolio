"""Execute every displayed Python fence in the two manuscripts, in fresh namespaces."""
import contextlib
import hashlib
import io
import json
import math
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = []
pending_outputs = {}
namespaces = []
for topic in ["loss-functions-ce-mse-focal-contrastive-triplet", "batch-layer-group-rms-normalization"]:
    manuscript = root / "docs/teaching/drafts" / topic / "lesson.md"
    snippets = re.findall(r"```python\n(.*?)\n```", manuscript.read_text(encoding="utf-8"), re.S)
    outputs = []
    for index, code in enumerate(snippets):
        output = io.StringIO()
        namespace = {}
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"{topic}-snippet-{index + 1}", "exec"), namespace)
        namespaces.append(namespace)
        outputs.append(f"Snippet {index + 1}\n{output.getvalue()}")
        records.append(dict(topic=topic, index=index + 1, code_sha256=hashlib.sha256(code.encode()).hexdigest(), output=output.getvalue()))
    target = root / "public/learn-assets" / topic / "snippet-output.txt"
    pending_outputs[target] = "\n".join(outputs)
assert len(records) == 5
assert "1000.6566" in records[0]["output"]
binary_expected = sum(max(z, 0) - y * z + math.log1p(math.exp(-abs(z))) for z, y in zip([-2, 1.5, .2, -.4], [0, 1, 1, 0])) / 4
assert f"{binary_expected:.4f}" in records[0]["output"]
assert "0.5600" in records[1]["output"] and "0.8000" in records[1]["output"]
assert math.isclose(float(namespaces[2]["loss"]), 1.225170469096215, abs_tol=1e-12)
assert abs(namespaces[2]["logit_gradient"].sum(axis=1)).max() < 1e-12
assert "0.8486" in records[3]["output"] and "1.1318" in records[3]["output"]
assert abs(namespaces[4]["dx"].sum(axis=1)).max() < 1e-12
assert abs(namespaces[4]["h"].mean(axis=1)).max() < 1e-12
for target, output in pending_outputs.items():
    target.write_text(output, encoding="utf-8")
(root / "docs/teaching/evidence/loss-normalization-snippets.json").write_text(json.dumps(dict(passed=True, records=records), indent=2), encoding="utf-8")
print("PASS five complete displayed Python snippets and their visible numerical outputs.")
