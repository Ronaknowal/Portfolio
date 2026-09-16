"""Close bounded content evidence; no browser or production campaign."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import re

import numpy as np
import torch

ROOT = Path(__file__).parent
REPOSITORY = ROOT.parents[3]


def main():
    lesson = (ROOT / "lesson.md").read_text(encoding="utf8")
    spec = (ROOT / "visual-specifications.md").read_text(encoding="utf8")
    report = {"phase": "content-author checks only", "programs": [], "links": []}
    blocks = re.findall(r"```python\n(.*?)```", lesson, flags=re.S)
    assert len(blocks) == 3
    for index, program in enumerate(blocks):
        namespace = {"torch": torch}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(program, f"manuscript-program-{index + 1}", "exec"), namespace)
        if index == 0:
            assert np.allclose(namespace["fast"], [1, 2.5, 4.25, 6.125])
        elif index == 1:
            values = torch.tensor([[[1.], [2.], [3.], [4.]]], dtype=torch.float64)
            kernel = torch.tensor([[1.], [.5], [.25], [.125]], dtype=torch.float64)
            actual = namespace["causal_convolution"](values, kernel)
            assert torch.allclose(actual.flatten(), torch.tensor([1, 2.5, 4.25, 6.125], dtype=torch.float64))
        else:
            assert np.allclose(namespace["outputs"], [1, -1.8, .275, 2.81875, -.4109375, 2.299609375])
        report["programs"].append({"index": index + 1, "executed": True, "output": output.getvalue()})
    figures = re.findall(r"\*\*Figure (H\d\d) —", lesson)
    assert figures == [f"H{i:02}" for i in range(1, 21)]
    assert all(f"| {figure} " in spec for figure in figures)
    assert lesson.count("<details>") == lesson.count("</details>") == 20
    assert "<details open" not in lesson
    assert re.findall(r"\*\*Investigation (H[A-D]) —", lesson) == ["HA", "HB", "HC", "HD"]
    for target in re.findall(r"\]\(([^)]+)\)", lesson):
        if target.startswith("http"):
            continue
        if target.startswith("/learn/"):
            topic = target.split("/learn/path/full-curriculum/")[1].split("?")[0]
            assert "module=deep-learning-fundamentals" in target
            assert (REPOSITORY / "src/learn/data/topics" / f"{topic}.jsx").exists()
        else:
            assert (ROOT / target).exists()
        report["links"].append(target)
    original = REPOSITORY / "src/learn/data/topics/hyena-long-convolution-models.jsx"
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    assert source_hash == "621f9daadcbdebb6a38eaf8035daf0518b5026e996c5eb997086fd554cacb93c"
    results = json.loads((ROOT / "splice-results.json").read_text())
    for name, expected in results["data"]["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for fit in results["fits"]:
        for role in ["fit", "validation", "assessment"]:
            matrix = np.array(fit[role]["confusion"])
            assert matrix.sum() == fit[role]["count"]
            assert matrix.sum() - np.trace(matrix) == fit[role]["errors"]
        assert min(fit["history"], key=lambda row: row[1])[0] == fit["selected_epoch"]
    report.update({"figures": figures, "investigations": ["HA", "HB", "HC", "HD"],
                   "closed_hint_solution_disclosures": 20, "original_sha256": source_hash,
                   "data_hashes_checked": True, "confusions_and_epoch_selection_checked": True,
                   "implementation": "not started"})
    (ROOT / "author-checks.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
