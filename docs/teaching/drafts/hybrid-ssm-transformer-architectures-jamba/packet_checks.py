"""Final bounded author arithmetic and packet integrity checks, without fitting."""
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from hybrid_mechanisms import cache_bytes, memory_read, route
from inspect_stroke import run_trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    report = {"scope": "author packet integrity and small null fixtures; no training or browser checks"}
    original = ROOT / "src/learn/data/topics/hybrid-ssm-transformer-architectures-jamba.jsx"
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    assert original_hash == "e71d445f9077cc2dcf9f265b749d47a631a5022e29f3a93f696e87a8a5d457d8"
    report["original_sha256"] = original_hash

    programs = sorted(HERE.glob("*.py"))
    for path in programs:
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    report["syntax_checked_programs"] = [path.name for path in programs]
    manuscript = (HERE / "lesson.md").read_text(encoding="utf-8")
    specifications = (HERE / "visual-specifications.md").read_text(encoding="utf-8")
    figures = re.findall(r"\[Figure (J\d{2}):", manuscript)
    assert figures == [f"J{i:02}" for i in range(1, 19)]
    assert all(f"| {figure}," in specifications for figure in figures)
    assert manuscript.count("<details>") == manuscript.count("</details>") == 20
    assert not re.search(r"<details\s+open", manuscript)
    assert len(re.findall(r"^```", manuscript, flags=re.M)) % 2 == 0
    relative_links = []
    internal_ids = []
    navigation = (ROOT / "src/learn/data/generated/navigation.js").read_text(encoding="utf-8")
    known_ids = set(re.findall(r'"id":"([^"]+)"', navigation))
    for link in re.findall(r"\]\(([^)]+)\)", manuscript):
        if link.startswith("/learn/path/"):
            topic_id = link.split("?")[0].rsplit("/", 1)[1]
            assert topic_id in known_ids, topic_id
            assert "module=deep-learning-fundamentals" in link
            internal_ids.append(topic_id)
        elif not link.startswith(("https://", "http://", "#")):
            assert (HERE / link).is_file(), link
            relative_links.append(link)
    report["manuscript"] = {"figures": figures, "closed_disclosures": 20,
                             "relative_links": relative_links, "internal_ids": sorted(set(internal_ids))}

    values, keys = [3, 8, 1, 5], ["A", "B", "A", "C"]
    boundary_reads = {
        "zero_decay": memory_read(values, keys, "A", decay=0),
        "unit_decay": memory_read(values, keys, "A", decay=1),
        "zero_score_gap": memory_read(values, keys, "A", beta=0),
    }
    assert boundary_reads["zero_decay"]["summary"] == 5
    assert boundary_reads["unit_decay"]["summary"] == 17 / 4
    np.testing.assert_allclose(boundary_reads["zero_score_gap"]["weights"], [.25] * 4)
    assert boundary_reads["zero_score_gap"]["attention"] == 17 / 4
    report["boundary_reads"] = boundary_reads

    logits = [0, math.log(3), math.log(6), math.log(2)]
    outputs = [[1, 2], [3, 0], [-1, 4], [2, -2]]
    original_output = route(logits, outputs)["output"]
    changed_outputs = [row[:] for row in outputs]
    changed_outputs[0] = [-10, 10]
    unselected_output = route(logits, changed_outputs)["output"]
    shifted_output = route([x + 2 for x in logits], outputs)["output"]
    all_retained = route(logits, outputs, k=4)["output"]
    all_renormalized = route(logits, outputs, k=4, renormalize=True)["output"]
    np.testing.assert_allclose(original_output, unselected_output, atol=0, rtol=0)
    np.testing.assert_allclose(original_output, shifted_output, atol=1e-14, rtol=0)
    np.testing.assert_allclose(all_retained, all_renormalized, atol=1e-14, rtol=0)
    report["router_nulls"] = {"original": original_output, "unselected_value_edit": unselected_output,
                              "common_logit_shift": shifted_output,
                              "all_retained": all_retained, "all_renormalized": all_renormalized}

    memory = cache_bytes(8192, batch=2, layers=24, attention_layers=6, kv_heads=4, head_width=64)
    assert memory["kv"] == 100663296
    assert (4 + 4 * 4) * 3 * 16 * 32 == 30720
    assert (4 + 4 * 2) * 3 * 16 * 32 == 18432
    assert 8 * 3 * 16 * 32 == 12288
    report["practice_counts"] = {"question_2_kv_bytes": memory["kv"],
                                 "question_8_stored_active_dense": [30720, 18432, 12288]}

    author = json.loads((HERE / "author-results.json").read_text())
    null_differences = {}
    for name in ["worked", "fresh"]:
        coordinates = author["fixtures"][name]["traces"]["carry"]["coordinates"]
        for boundary, mode in [(0, "recurrent-reset"), (8, "kv-reset")]:
            carry = run_trace(coordinates, break_at=boundary)
            fault = run_trace(coordinates, break_at=boundary, mode=mode)
            difference = float(np.max(np.abs(np.array(carry["branch_logits"]) - np.array(fault["branch_logits"]))))
            assert difference == 0
            null_differences[f"{name}_boundary_{boundary}"] = difference
    report["matching_carry_boundary_nulls"] = null_differences
    report["program_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in programs}
    (HERE / "packet-checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
