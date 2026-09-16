"""Independent small-input arithmetic and unsolved edit fixtures, not browser QA."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import torch
from capsule_learning_import import load_learning

ROOT = Path(__file__).resolve().parent
learning = load_learning()
spec = importlib.util.spec_from_file_location("capsule_mechanics", ROOT / "capsule-mechanics.py")
mechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanics)


def convolution(image, weights, bias, stride=1):
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)))
    height = (padded.shape[1] - 3) // stride + 1
    width = (padded.shape[2] - 3) // stride + 1
    result = np.empty((len(weights), height, width))
    for row in range(height):
        for column in range(width):
            patch = padded[:, row * stride:row * stride + 3, column * stride:column * stride + 3]
            for output_channel in range(len(weights)):
                result[output_channel, row, column] = np.sum(patch * weights[output_channel]) + bias[output_channel]
    return result


def independent_encode(image, state, iterations):
    features = np.maximum(convolution(image, state["features.weight"], state["features.bias"]), 0)
    channels = convolution(features, state["primary.weight"], state["primary.bias"], stride=2)
    primary = np.empty((64, 4))
    for row in range(4):
        for column in range(4):
            for kind in range(4):
                primary[(row * 4 + column) * 4 + kind] = channels[kind * 4:kind * 4 + 4, row, column]
    primary = mechanics.squash(primary)
    votes = np.empty((64, 10, 8))
    for child in range(64):
        for parent in range(10):
            votes[child, parent] = state["transforms"][child, parent] @ primary[child]
    return mechanics.routing(votes, iterations)[0]


def decode(capsules, label, state):
    masked = np.zeros_like(capsules)
    masked[label] = capsules[label]
    values = masked.reshape(-1)
    for index in (0, 2, 4):
        values = state[f"decoder.{index}.weight"] @ values + state[f"decoder.{index}.bias"]
        values = np.maximum(values, 0) if index < 4 else 1 / (1 + np.exp(-values))
    return values.reshape(8, 8)


def main():
    torch.set_num_threads(1)
    saved = json.loads((ROOT / "calculated-inputs.json").read_text(encoding="utf-8"))
    fixtures = []
    for count in (1, 3):
        state = {name: np.array(value) for name, value in saved["saved_models"][str(count)].items()}
        model = learning.TinyCapsules()
        model.load_state_dict({name: torch.tensor(value, dtype=torch.float32) for name, value in state.items()})
        model.eval()
        run = next(item for item in saved["runs"] if item["seed"] == 1 and item["training_iterations"] == count)
        for example_index, raw_image in enumerate(run["first_two_examples"]["images"]):
            image = np.array(raw_image)
            capsule = independent_encode(image[None], state, count)
            actual = np.array(run["first_two_examples"]["capsules"][example_index])
            label = int(np.linalg.norm(capsule, axis=1).argmax())
            reconstruction = decode(capsule, label, state)
            tensor = torch.tensor(image[None, None], dtype=torch.float32)
            with torch.no_grad():
                native_capsules = model.encode(tensor, count)[0]
                native_reconstruction = model.reconstruct(native_capsules, torch.tensor([label]))[0, 0].numpy()
            edited = image.copy()
            edited[3, 3] = 1 - edited[3, 3]
            edited_capsules = independent_encode(edited[None], state, count)
            coordinate_results = []
            for delta in (-0.1, 0.0, 0.1):
                changed_capsule = capsule.copy()
                changed_capsule[label, 0] += delta
                output = decode(changed_capsule, label, state)
                coordinate_results.append({"delta": delta, "reconstruction": output.tolist(),
                    "maximum_image_change": float(np.max(np.abs(output - reconstruction)))})
            absent_label = (label + 1) % 10
            masked_edit = capsule.copy()
            masked_edit[absent_label, 0] += 0.1
            fixtures.append({"training_iterations": count,
                "source_id": run["first_two_examples"]["source_ids"][example_index],
                "capsule_error": float(np.max(np.abs(capsule - actual))),
                "reconstruction_error": float(np.max(np.abs(reconstruction - native_reconstruction))),
                "predicted_class": label, "original_lengths": np.linalg.norm(capsule, axis=1).tolist(),
                "pixel_edit": {"row": 3, "column": 3, "before": float(image[3, 3]), "after": float(edited[3, 3]),
                    "lengths": np.linalg.norm(edited_capsules, axis=1).tolist(),
                    "predicted_class": int(np.linalg.norm(edited_capsules, axis=1).argmax()),
                    "maximum_capsule_change": float(np.max(np.abs(edited_capsules - capsule)))},
                "coordinate_edits": coordinate_results,
                "masked_capsule_edit_reconstruction_error": float(np.max(np.abs(decode(masked_edit, label, state) - reconstruction)))})
    summary = []
    for run in saved["runs"]:
        predictions = run["inference_iterations"]
        reference = np.array(predictions[str(run["training_iterations"])]["predictions"])
        summary.append({"seed": run["seed"], "trained_routing": run["training_iterations"],
            "counts_by_inference_iterations": {count: value["correct"] for count, value in predictions.items()},
            "changed_predictions": {count: int((np.array(value["predictions"]) != reference).sum()) for count, value in predictions.items()},
            "shifted": {direction: value["correct"] for direction, value in run["shifted"].items()},
            "training_correct": run["trajectory"][-1]["train"]["correct"]})
    result = {"fixtures": fixtures, "comparison_summary": summary,
              "maximum_capsule_error": max(item["capsule_error"] for item in fixtures),
              "maximum_reconstruction_error": max(item["reconstruction_error"] for item in fixtures)}
    (ROOT / "author-check-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "fixtures"}, indent=2))


if __name__ == "__main__":
    main()
