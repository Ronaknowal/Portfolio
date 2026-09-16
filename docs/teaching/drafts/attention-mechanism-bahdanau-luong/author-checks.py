"""Bounded content-author closure. Does not refit models or run the website."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import re
import runpy
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    lesson = (ROOT/"lesson.md").read_text(encoding="utf-8")
    specifications = (ROOT/"visual-specifications.md").read_text(encoding="utf-8")
    for text in (lesson, specifications):
        assert not any(bad in text for bad in ("\u00e2\u20ac", "\u00e2\u02c6", "\ufffd"))
    blocks = re.findall(r"```python\n([\s\S]*?)```", lesson)
    assert len(blocks) == 2
    assert blocks[1].strip() == (ROOT/"attentive-inflection.py").read_text(encoding="utf-8").strip()
    capture = io.StringIO()
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(capture):
            exec(compile(blocks[0], "lesson-saved-model-example", "exec"), {})
    finally:
        os.chdir(previous_directory)
    assert capture.getvalue().strip() == "lactated True"
    practice = lesson.split("## 9. Practice:", 1)[1].split("## 10.", 1)[0]
    assert practice.count("<details>") == 16 and practice.count("</details>") == 16
    assert practice.count("<summary>Hint</summary>") == 8
    assert practice.count("<summary>Solution</summary>") == 8
    assert "<details open" not in practice

    source = REPOSITORY/"src/learn/data/topics/attention.jsx"
    assert digest(source) == "eb481258c4aaf13bd9f1a3d697802190174b7e1f55c6ff62b47f825255fd5e36"
    assert digest(ROOT/"english-inflections.csv") == "eb56afb2415f267586809803c67574c4fff7998dbf02a70551fd784cf86dbe14"
    previous = ROOT.parent/"sequence-to-sequence-encoder-decoder"
    for name in ("english-inflections.csv", "data-extraction.json", "prepare-inflection-data.py"):
        assert (ROOT/name).read_bytes() == (previous/name).read_bytes()
    report = json.loads((ROOT/"calculated-inputs.json").read_text())
    mechanics = json.loads((ROOT/"mechanics-results.json").read_text())
    assert len(report["runs"]) == 6
    assert mechanics["prior_baseline"]["source_sha256"] == digest(previous/"calculated-inputs.json")
    api = runpy.run_path(str(ROOT/"attentive-inflection.py"))
    train, development = api["load_records"]()
    for run in report["runs"]:
        summary = api["summarize"](development, run["development_predictions"])
        assert all(run["development"][key] == value for key, value in summary.items())
        assert summary["reference_characters"] == 3490 and summary["no_eos"] == 0
        assert run["parameters"] == (62080 if run["kind"] == "additive" else 49760)
    for kind, rows in mechanics["traces"].items():
        for name, trace in rows.items():
            if not isinstance(trace, dict):
                continue
            previous_state = np.array(trace["memory"])[len(trace["source_ids"])-1]
            for row in trace["rows"]:
                assert np.isclose(sum(row["attention"]), 1., atol=1e-12)
                assert np.isclose(sum(row["probabilities"]), 1., atol=1e-12)
                assert all(row["probabilities"][index] == 0 for index in (0,1,3,4,5))
                query = np.array(row["query"])
                assert np.allclose(query, previous_state if kind == "additive" else row["state"], atol=1e-12)
                previous_state = np.array(row["state"])
        assert rows["fresh"]["input"] != rows["worked"]["input"]
        assert rows["fresh"]["prediction"] == rows["fresh_broken_mask"]["prediction"]
    calculations = runpy.run_path(str(ROOT/"attention-calculations.py"))
    analytic = json.loads((ROOT/"analytic-results.json").read_text())
    values = analytic["fixtures"]
    assert np.allclose(values["fresh"]["class_probabilities"], values["fresh_value_edit"]["class_probabilities"])
    assert not np.allclose(values["fresh"]["class_probabilities"], values["fresh_value_contrast"]["class_probabilities"])
    assert np.allclose(values["fresh_zero_rate"]["query"], values["fresh_zero_rate"]["next_query"])
    for value in values.values():
        assert np.allclose(value["query_gradient"], value["finite_difference"], atol=1e-9)
    local = analytic["local_fresh"]
    scores = np.array(local["scores"]); valid = np.array(local["valid"])
    weights = []
    for excluded_value in (2., -3.):
        edited = scores.copy(); edited[4] = excluded_value
        masked = np.where(valid, edited, -np.inf)
        base = np.exp(masked-np.max(masked)); base /= base.sum()
        weights.append(base*np.exp(-(np.arange(1,6)-2.5)**2/(2*.75**2)))
    assert np.allclose(weights[0], weights[1])
    assert np.allclose(weights[0], local["weights"])
    local_setting_count = 0
    for center in np.arange(1., 5.01, .5):
        for radius in np.arange(1., 3.01, .5):
            for renormalize in (False, True):
                actual = calculations["local"](float(center), float(radius), renormalize)
                assert np.isfinite(actual["context"]) and 0 < actual["sum"] <= 1+1e-12
                if renormalize:
                    assert np.isclose(actual["sum"], 1)
                local_setting_count += 1
    try:
        calculations["calculate"]([.2,.4], valid=[0,0,0])
    except ValueError:
        all_masked_rejected = True
    else:
        raise AssertionError("All-masked constructed read was accepted.")
    practice1 = calculations["calculate"]([0.,1.])
    assert np.allclose(practice1["context"], [.21194155761708544,1.3641753271487436])
    for fixture in (analytic["pointer"],analytic["pointer"]["fresh"]):
        words = set(fixture["source"]) | set(fixture["vocabulary"])
        actual = {word: fixture["p_generate"]*fixture["vocabulary"].get(word,0)
                  +(1-fixture["p_generate"])*sum(weight for token,weight in zip(fixture["source"],fixture["attention"]) if token==word)
                  for word in words}
        assert all(abs(actual[word]-fixture["output"][word]) < 1e-12 for word in words)
    scope = (REPOSITORY/"docs/teaching/DEEP-LEARNING-ARCHITECTURES-CONTENT.md").read_text(encoding="utf-8")
    for topic in re.findall(r"/learn/path/full-curriculum/([^?)]+)", lesson):
        assert topic in scope
    result = {"source_sha256": digest(source), "data_sha256": digest(ROOT/"english-inflections.csv"),
              "training_program_sha256": digest(ROOT/"attentive-inflection.py"), "full_program_bound": True,
              "saved_model_displayed_code_output": capture.getvalue().splitlines(),
              "source_extract_exact_copies": True, "six_final_metric_rows_reconciled": True,
              "query_schedule_and_probability_invariants": True, "closed_practice_hint_solution_blocks": 16,
              "fresh_local_excluded_score_null": True, "local_settings_checked": local_setting_count,
              "all_masked_rejected": all_masked_rejected, "pointer_outputs_reconstructed": True,
              "practice1_context": practice1["context"],
              "maximum_native_numpy_probability_error": mechanics["max_native_numpy_probability_error"],
              "fit_count_this_packet": 6, "phase_two_review_or_browser_performed": False}
    (ROOT/"author-check-results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

