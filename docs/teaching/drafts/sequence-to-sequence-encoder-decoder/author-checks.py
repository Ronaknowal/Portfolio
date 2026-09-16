"""Bounded source/data/manuscript consistency checks; no repeated fits or UI work."""
from pathlib import Path
import contextlib
import csv
import hashlib
import io
import json
import math
import re
import runpy
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]
sys.dont_write_bytecode = True
torch.set_num_threads(1)


def main():
    manuscript = (ROOT/"lesson.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n([\s\S]*?)\n```", manuscript)
    assert len(blocks) == 2
    assert blocks[0].strip() == (ROOT/"inflection-seq2seq.py").read_text().strip()
    compile(blocks[0], "full-program", "exec")
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(blocks[1], "saved-model-example", "exec"), {"__file__": str(ROOT/"saved-model-example.py")})
    assert capture.getvalue().splitlines() == ["3 lac False", "16 lactated True"]
    assert "<!--" not in manuscript
    practice = manuscript.split("## 10. Practice", 1)[1].split("## 11.", 1)[0]
    assert practice.count("<details>") == practice.count("</details>") == 16
    assert "<details open" not in practice and "**Solution" not in practice
    assert manuscript.count("<details>") == manuscript.count("</details>")
    source_hash = hashlib.sha256((REPOSITORY/"src/learn/data/topics/sequence-to-sequence-encoder-decoder.jsx").read_bytes()).hexdigest()
    assert source_hash == "7ad7db0588632290b7b78a6d628f97c76cb685876b601357f551fe716bbe6005"
    source_record = json.loads((ROOT/"data-extraction.json").read_text())
    data_hash = hashlib.sha256((ROOT/"english-inflections.csv").read_bytes()).hexdigest()
    assert data_hash == source_record["extract_sha256"] == "eb56afb2415f267586809803c67574c4fff7998dbf02a70551fd784cf86dbe14"
    api = runpy.run_path(str(ROOT/"inflection-seq2seq.py"))
    train, development = api["load_records"]()
    for records in (train, development):
        x, lengths, decoder_input, target = api["batch"](records)
        expected = [api["source_ids"](row["lemma"], row["feature"]) for row in records]
        old_x = torch.zeros(len(records), max(map(len, expected)), dtype=torch.long)
        for row, values in enumerate(expected):
            old_x[row, :len(values)] = torch.tensor(values)
        assert torch.equal(x, old_x) and lengths.tolist() == list(map(len, expected))
        assert bool((decoder_input[:, 0] == api["BOS"]).all())
        assert torch.equal(decoder_input[:, 1:], target[:, :-1])
    report = json.loads((ROOT/"calculated-inputs.json").read_text())
    assert len(report["runs"]) == 3 and report["protocol"]["updates"] == 1200
    for run in report["runs"]:
        metrics = api["summarize"](development, run["development_predictions"])
        assert all(run["development"][key] == value for key, value in metrics.items())
        assert metrics["reference_characters"] == 3490 and metrics["no_eos"] == 0
        assert run["parameters"] == 37408
    model = api["Inflector"]().eval()
    weights = report["runs"][0]["weights"]
    model.load_state_dict({key: torch.tensor(value, dtype=torch.bool if key == "invalid_output" else torch.float32) for key, value in weights.items()})
    queries = [{"lemma": row["lemma"], "feature": row["feature"]} for row in development]
    predictions = api["greedy"](model, queries)
    assert [row["tokens"] for row in predictions] == [row["tokens"] for row in report["runs"][0]["development_predictions"]]
    mechanics = json.loads((ROOT/"mechanics-results.json").read_text())
    mechanism_api = runpy.run_path(str(ROOT/"sequence-mechanics.py"))
    trace_function = mechanism_api["manual_trace"]
    original = trace_function(development[0], weights)
    assert original == trace_function(development[0], weights, context_override=original["context"])
    assert original == trace_function({**development[0], "display_label": "renamed only"}, weights)
    assert original == trace_function(development[0], weights)
    assert mechanics["scalar_zero_rate"]["updated_weight"] == .7
    assert mechanics["scalar_zero_rate"]["updated_mean_loss"] == mechanics["scalar_zero_rate"]["mean_loss"]
    for name in ("scalar_worked", "scalar_fresh", "scalar_fresh_input_edit"):
        row = mechanics[name]
        assert abs(row["central_difference"]-row["encoder_input_weight_gradient"]) < 1e-9
    fresh_length = [{"length": length, "log_probability": logp, "alpha0": logp,
                     "alpha1": logp/((5+length)/6)} for length, logp in ((2,-1.2),(5,-1.4))]
    assert fresh_length[0]["alpha0"] > fresh_length[1]["alpha0"]
    assert fresh_length[0]["alpha1"] < fresh_length[1]["alpha1"]
    practice2 = {"sequence_probability": .8*.5*.25, "valid_mean_nll": -math.log(.8*.5*.25)/3,
                 "incorrect_padded_mean": -math.log(.8*.5*.25)/5}
    assert abs(practice2["valid_mean_nll"]-.7675283643313485) < 1e-12
    source_ids = json.loads((REPOSITORY/"src/learn/data/lesson-manifest.json").read_text())
    # Route existence is also recorded in current inventory; avoid assuming JSON value layout here.
    manifest_text = json.dumps(source_ids)
    for topic in ("rnns-lstms-grus", "backpropagation-automatic-differentiation", "attention-mechanism-bahdanau-luong"):
        assert topic in manifest_text
    result = {"source_sha256": source_hash, "data_sha256": data_hash,
              "full_program_bound": True, "saved_model_block_actual_output": capture.getvalue().splitlines(),
              "source_only_generation_matches_saved_all_447": True,
              "source_batch_extraction_preserves_all_arrays": True,
              "same_context_repeated_source_label_nulls": True,
              "practice_closed_hint_solution_blocks": 16,
              "final_three_run_metric_reconciliation": True,
              "fresh_length_score": fresh_length, "practice2": practice2,
              "phase_two_review_or_browser_performed": False}
    (ROOT/"author-check-results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
