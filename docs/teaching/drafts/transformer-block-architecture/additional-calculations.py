"""Independent hand/finite-difference checks; no model fitting or network access."""
import json
import math
from pathlib import Path
import numpy as np

PACKET = Path(__file__).resolve().parent
EPSILON = 1e-5


def layer_norm(x):
    centered = x - x.mean()
    return centered / np.sqrt(np.mean(centered ** 2) + EPSILON)


def rms_norm(x):
    return x / np.sqrt(np.mean(x ** 2) + EPSILON)


if __name__ == "__main__":
    x = np.array([1., 2., 5., 8.])
    centered = x - x.mean()
    scale = np.sqrt(np.mean(centered ** 2) + EPSILON)
    jacobian = (np.eye(4) - np.ones((4, 4)) / 4 - np.outer(centered, centered) / (4 * scale ** 2)) / scale
    probe = np.array([1., -1., 0., 0.])
    analytic = jacobian.T @ probe
    step = 1e-5
    finite = np.array([(layer_norm(x + step * np.eye(4)[i]) @ probe -
                        layer_norm(x - step * np.eye(4)[i]) @ probe) / (2 * step) for i in range(4)])
    assert np.max(np.abs(analytic - finite)) < 1e-9
    assert np.max(np.abs(jacobian @ np.ones(4))) < 1e-14
    tight = np.array([-.03, -.01, .01, .03])
    tight_centered = tight - tight.mean()
    tight_scale = np.sqrt(np.mean(tight_centered ** 2) + EPSILON)
    tight_jacobian = (np.eye(4) - np.ones((4, 4)) / 4 -
                      np.outer(tight_centered, tight_centered) / (4 * tight_scale ** 2)) / tight_scale
    singular_values = np.linalg.svd(tight_jacobian, compute_uv=False)
    practice = np.array([0., 2., 4., 6.])
    up = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0.]])
    down = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, -.5]])
    ffn_inputs = np.array([[1, 2, -1, -2], [-1, 0, 1, 0.]])
    original = np.maximum(0, ffn_inputs @ up) @ down
    changed_inputs = ffn_inputs.copy()
    changed_inputs[1, 0] = 3
    changed = np.maximum(0, changed_inputs @ up) @ down
    assert np.array_equal(original[0], changed[0]) and not np.array_equal(original[1], changed[1])
    practice_hidden = np.maximum(0, np.array([2., -1., 0., 1.]) @ up)
    changed_down = down.copy()
    changed_down[2] = [0, 0, 1, 0]
    values = {"ln_probe": {"analytic": analytic.tolist(), "central_difference": finite.tolist(),
                           "max_error": float(np.max(np.abs(analytic - finite))),
                           "step": step, "null_max_error": float(np.max(np.abs(jacobian @ np.ones(4)))),
                           "tight_input": tight.tolist(), "tight_jacobian_singular_values": singular_values.tolist()},
              "practice_norm": {"input": practice.tolist(), "ln": layer_norm(practice).tolist(),
                                "rms": rms_norm(practice).tolist(), "shifted_rms": rms_norm(practice + 3).tolist()},
              "ffn_independence": {"inputs": ffn_inputs.tolist(), "original": original.tolist(),
                                   "changed_inputs": changed_inputs.tolist(), "changed": changed.tolist()},
              "practice_ffn": {"hidden": practice_hidden.tolist(), "original": (practice_hidden @ down).tolist(),
                               "changed": (practice_hidden @ changed_down).tolist()},
              "practice_parameter_counts": {"original": 4 * 96 ** 2 + 2 * 96 * 240 + 240 + 9 * 96,
                                            "gated": 4 * 96 ** 2 + 4 * 96 + 3 * 96 * 160 + 4 * 96},
              "practice_copy_floor": 6 / 14 * math.log(8),
              "derived_mac_counts": [{"length": length, "linear_maps": 12 * length * 512 ** 2,
                                      "pairwise_attention": 2 * length ** 2 * 512} for length in (512, 4096)]}
    (PACKET / "additional-fixtures.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(values, indent=2))
