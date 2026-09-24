"""Fresh prediction-gated fixtures, evaluated without training or downloads.

Run beside the two original programs and forecast-model.json.
Answers belong to author/phase-two evidence and remain hidden before commit.
"""
from pathlib import Path
import importlib.util
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def load_module(name, filename):
    specification = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(1)
    mechanism = load_module("mla_mechanism", "mechanism-calculations.py")
    study = load_module("mla_study", "author-calculations.py")
    q = np.array([[.5, 1.], [-1., 2.]])
    c = np.array([[2., -1.], [.5, 1.], [-.5, 2.]])
    uk = np.array([[[1., .5], [0., 2.]], [[1., -1.], [.5, 1.]]])
    uv = np.array([[[1., 0.], [.5, 1.]], [[1., 1.], [-1., 2.]]])
    raw_qr = np.array([[1., .5], [-.5, 1.]])
    raw_kr = np.array([[1., 0.], [.5, 1.], [1., -1.]])
    qr = mechanism.rotate_rows(raw_qr, [3, 3], math.pi / 3)
    kr = mechanism.rotate_rows(raw_kr, [0, 1, 3], math.pi / 3)
    base = mechanism.attention(q, c, uk, uv, qr, kr, .5)
    expanded = mechanism.attention(q, c, uk, uv, qr, kr, .5, expanded=True)
    changed_c = c.copy(); changed_c[1, 0] += .75
    changed = mechanism.attention(q, changed_c, uk, uv, qr, kr, .5)
    changed_uv = uv.copy(); changed_uv[0, 0, 0] += .5
    value_edit = mechanism.attention(q, c, uk, changed_uv, qr, kr, .5)
    assert np.allclose(base["head_outputs"], expanded["head_outputs"], atol=1e-12)
    assert np.array_equal(base["weights"], value_edit["weights"])
    assert np.array_equal(base["head_outputs"][1], value_edit["head_outputs"][1])
    angle = math.pi / 3
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    projection = np.array([[1., 1.], [0., 2.]])
    vector = np.array([1., -1.])
    matrix = np.array([[3., 1.], [0., 2.]])
    input_vector = np.array([2., -3.])
    _, singular, right = np.linalg.svd(matrix)
    retained = right[0]
    reduced = matrix @ np.outer(retained, retained)
    control = retained * np.linalg.norm(input_vector)
    assert np.allclose(matrix @ control, reduced @ control, atol=1e-12)
    saved = json.loads((ROOT / "forecast-model.json").read_text(encoding="utf-8"))
    model = study.LatentForecaster()
    model.load_state_dict({name: torch.tensor(value) for name, value in saved["weights"].items()})
    model.eval()
    points, _, _, _ = study.load_records()
    prefix = points[76:77, :27]
    changed_prefix = prefix.clone(); changed_prefix[:, 19, 1] *= -1
    traces = {}
    with torch.no_grad():
        for label, basis in (("full", None), ("rank4", torch.tensor(saved["rank4_basis"]))):
            output, cache, trace = model(prefix, basis=basis, return_trace=True)
            changed_output, _ = model(changed_prefix, basis=basis)
            expanded_output, _ = model(prefix, basis=basis, mode="expanded")
            assert torch.equal(output[:, :19], changed_output[:, :19])
            assert torch.allclose(output, expanded_output, atol=2e-6, rtol=2e-6)
            traces[label] = {"prediction": (output[0, -1] + 1) / 2,
                "changed_prediction": (changed_output[0, -1] + 1) / 2,
                "earlier_output_max_error": (output[:, :19]-changed_output[:, :19]).abs().max(),
                "expanded_absorbed_max_error": (output-expanded_output).abs().max(),
                "cache_bytes": (cache[0].numel()+cache[1].numel())*4, "trace": trace}
    result = {"author_only_expected_results": True,
        "I1": {"content_queries": q, "latents": c, "key_up": uk, "value_up": uv,
            "raw_rotary_queries": raw_qr, "raw_rotary_keys": raw_kr,
            "memory_positions": [0, 1, 3], "query_position": 3,
            "frequency": math.pi/3, "scale": .5, "base": base,
            "proposed_latent_edit": {"record": 1, "coordinate": 0, "add": .75},
            "changed_latent": changed, "value_only_control": value_edit},
        "I2": {"projection": projection, "input": vector, "angle": angle,
            "project_then_rotate": rotation @ projection @ vector,
            "rotate_then_project": projection @ rotation @ vector,
            "isotropic_control": 2 * rotation @ vector},
        "I3": {"B": 3, "N": 24, "L": 8192, "H": 24, "dk": 64, "dv": 64,
            "dc": 192, "dr": 32, "bytes_per_number": 2,
            "payload_bytes": 3*24*8192*(192+32)*2,
            "proposed_H": 48, "new_payload_bytes": 3*24*8192*(192+32)*2,
            "expanded_core_ops": 2*3*24*8192*(64+32+64),
            "absorbed_core_ops": 2*3*24*8192*(2*192+32)},
        "I4": {"matrix": matrix, "input": input_vector, "retained_direction": retained,
            "singular_values": singular, "reduced_matrix": reduced,
            "matrix_squared_error": float(np.sum((matrix-reduced)**2)),
            "original_output": matrix @ input_vector, "reduced_output": reduced @ input_vector,
            "retained_input_control": control, "retained_output": matrix @ control},
        "I5": {"source_row": 77, "prefix_length": 27, "edit_frame": 19,
            "edit": "reflect y with 1-y", "observed_points": (prefix+1)/2,
            "true_next_point": (points[76, 27]+1)/2, "traces": traces}}
    result = mechanism.lists(study.as_lists(result))
    (ROOT / "practice-fixtures.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"I1_heads": result["I1"]["base"]["head_outputs"],
        "I2": result["I2"], "I3": result["I3"], "I4": result["I4"],
        "I5": {key: {k: v for k, v in value.items() if k != "trace"}
               for key, value in result["I5"]["traces"].items()}}, indent=2))


if __name__ == "__main__":
    main()
