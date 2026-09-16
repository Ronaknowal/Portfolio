"""Constructed capsule fixtures: routing, geometry, derivatives and diagonal EM."""
from pathlib import Path
import json
import numpy as np
import torch
from capsule_learning_import import load_learning

ROOT = Path(__file__).resolve().parent
learning = load_learning()


def softmax(values, axis=-1):
    exp_values = np.exp(values - np.max(values, axis=axis, keepdims=True))
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def squash(values):
    radius = np.sqrt((values * values).sum(axis=-1, keepdims=True))
    return values * radius / (1 + radius * radius)


def routing(votes, iterations=3):
    logits = np.zeros(votes.shape[:2], dtype=float)
    history = []
    for step in range(iterations):
        coupling = softmax(logits, axis=1)
        sums = np.zeros(votes.shape[1:], dtype=float)
        for child in range(votes.shape[0]):
            for parent in range(votes.shape[1]):
                sums[parent] += coupling[child, parent] * votes[child, parent]
        outputs = squash(sums)
        agreement = (votes * outputs[None]).sum(axis=2)
        history.append({"step": step + 1, "logits": logits.copy().tolist(),
                        "coupling": coupling.tolist(), "sums": sums.tolist(),
                        "outputs": outputs.tolist(), "lengths": np.linalg.norm(outputs, axis=1).tolist(),
                        "agreement": agreement.tolist(),
                        "mean_row_entropy": float(-(coupling * np.log(np.maximum(coupling, 1e-300))).sum(axis=1).mean())})
        logits += agreement
    return outputs, history


def diagonal_em(votes, child_activation, iterations=3, variance_floor=0.01):
    """Small declared EM-capsule illustration; no claim of full paper reproduction."""
    responsibility = np.full(votes.shape[:2], 1 / votes.shape[1])
    beta_u, beta_a = 0.0, 0.0
    history = []
    for step in range(iterations):
        effective = responsibility * child_activation[:, None]
        mass = effective.sum(axis=0)
        means = (effective[..., None] * votes).sum(axis=0) / np.maximum(mass[:, None], 1e-12)
        variance = (effective[..., None] * (votes - means[None]) ** 2).sum(axis=0)
        variance = np.maximum(variance / np.maximum(mass[:, None], 1e-12), variance_floor)
        cost = ((beta_u + 0.5 * np.log(variance)) * mass[:, None]).sum(axis=1)
        inverse_temperature = 0.5 + 0.25 * step
        activation = 1 / (1 + np.exp(-inverse_temperature * (beta_a - cost)))
        log_density = -0.5 * (np.log(2 * np.pi * variance)[None] +
                              (votes - means[None]) ** 2 / variance[None]).sum(axis=2)
        updated = softmax(log_density + np.log(np.maximum(activation, 1e-300))[None], axis=1)
        history.append({"step": step + 1, "responsibility_in": responsibility.tolist(),
                        "effective_weights": effective.tolist(), "mass": mass.tolist(),
                        "means": means.tolist(), "variance": variance.tolist(), "activation": activation.tolist(),
                        "responsibility_out": updated.tolist(), "inverse_temperature": inverse_temperature})
        responsibility = updated
    return history


def central_gradient(function, values, step=1e-6):
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def main():
    torch.set_num_threads(1)
    votes = np.array([[[2., 0.], [0., 1.]],
                      [[2., 0.], [0., -1.]],
                      [[0., 1.], [0., 2.]]])
    result = {"constructed": True, "votes": votes.tolist(), "routing": routing(votes, 8)[1]}
    edited = votes.copy()
    edited[1, 0] = [-2, 0]
    result["opposing_edit"] = routing(edited, 3)[1]
    zeros = np.zeros_like(votes)
    result["zero_votes"] = routing(zeros, 3)[1]
    identical = np.tile(np.array([[[1., 0.], [1., 0.]]]), (3, 1, 1))
    result["identical_parents"] = routing(identical, 3)[1]
    result["child_permutation_error"] = float(np.max(np.abs(routing(votes[[2, 0, 1]])[0] - routing(votes)[0])))
    comparisons = []
    for array in (votes, edited, zeros, identical):
        for count in (1, 2, 3, 8):
            tensor = torch.tensor(array[None], dtype=torch.float64)
            actual = learning.route(tensor, count)[0][0].numpy()
            comparisons.append(float(np.max(np.abs(actual - routing(array, count)[0]))))
    result["numpy_torch_routing_max_error"] = max(comparisons)
    probe = np.array([[0.3, 0.4], [0., 0.], [3., 4.]])
    result["squash_probes"] = []
    for vector in probe:
        radius = np.linalg.norm(vector)
        radial = 2 * radius / (1 + radius ** 2) ** 2
        tangential = radius / (1 + radius ** 2)
        numerical = np.stack([central_gradient(lambda x: squash(x)[axis], vector) for axis in range(2)])
        analytic = tangential * np.eye(2)
        if radius:
            unit = vector / radius
            analytic += (radial - tangential) * np.outer(unit, unit)
        result["squash_probes"].append({"input": vector.tolist(), "output": squash(vector).tolist(),
            "radius": float(radius), "radial_gradient": float(radial), "tangential_gradient": float(tangential),
            "jacobian": analytic.tolist(), "central_difference_error": float(np.max(np.abs(analytic - numerical)))})
    target = np.array([[0.2, -0.1], [0.5, 0.3]])
    def scalar_loss(array):
        return float(((routing(array, 3)[0] - target) ** 2).sum() / 2)
    numerical = central_gradient(scalar_loss, votes)
    gradients = {}
    for detach in (False, True):
        tensor = torch.tensor(votes[None], dtype=torch.float64, requires_grad=True)
        output = learning.route(tensor, 3, stop_gradient=detach)[0][0]
        loss = ((output - torch.tensor(target)) ** 2).sum() / 2
        loss.backward()
        gradients[str(detach)] = {"loss": float(loss.detach()), "gradient": tensor.grad[0].tolist(),
                                 "error_against_full_function": float(np.max(np.abs(tensor.grad[0].numpy() - numerical)))}
    result["gradient_paths"] = gradients
    rotation = np.array([[0., -1.], [1., 0.]])
    vector = np.array([1., 2.])
    arbitrary = np.diag([2., 1.])
    result["geometry"] = {"rotation_squash_error": float(np.max(np.abs(squash(rotation @ vector) - rotation @ squash(vector)))),
        "scale_squash_error": float(np.max(np.abs(squash(2 * vector) - 2 * squash(vector)))),
        "unconstrained_vote_equivariance_error": float(np.max(np.abs(arbitrary @ rotation @ vector - rotation @ arbitrary @ vector)))}
    pose = np.array([[1., 0., 2.], [0., 1., 3.], [0., 0., 1.]])
    relation = np.array([[1., 0., -1.], [0., 1., 0.], [0., 0., 1.]])
    transform = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result["geometry"].update(pose=pose.tolist(), relation=relation.tolist(), transform=transform.tolist(),
        whole_vote=(pose @ relation).tolist(), transformed_vote=((transform @ pose) @ relation).tolist(),
        matrix_associativity_error=float(np.max(np.abs((transform @ pose) @ relation - transform @ (pose @ relation)))))
    em_votes = np.array([[[0., 0.], [0., 0.]], [[0.2, 0.], [3., 0.]], [[2., 0.], [3.2, 0.]]])
    result["em"] = {"votes": em_votes.tolist(), "activation": [1., 1., 0.5],
                    "history": diagonal_em(em_votes, np.array([1., 1., 0.5]))}
    removed = diagonal_em(em_votes, np.array([1., 1., 0.]))
    changed_em = em_votes.copy()
    changed_em[2] = 100
    result["em"]["inactive_child_edit_error"] = float(np.max(np.abs(np.array(removed[-1]["means"]) -
        np.array(diagonal_em(changed_em, np.array([1., 1., 0.]))[-1]["means"]))))
    result["counts"] = {"conv1_parameters": 256 * (81 + 1),
        "primary_parameters": 256 * (256 * 81 + 1), "vote_parameters": 1152 * 10 * 16 * 8,
        "decoder_parameters": (160 + 1) * 512 + (512 + 1) * 1024 + (1024 + 1) * 784,
        "conv1_macs": 20 * 20 * 256 * 81, "primary_macs": 6 * 6 * 256 * 256 * 81,
        "vote_macs": 1152 * 10 * 16 * 8, "three_step_routing_product_sums": (3 + 2) * 1152 * 10 * 16,
        "votes_float32_bytes_batch32": 32 * 1152 * 10 * 16 * 4,
        "naive_224_children": 32 * 104 * 104,
        "naive_224_1000class_vote_parameters": 32 * 104 * 104 * 1000 * 16 * 8}
    result["counts"]["total_parameters"] = sum(result["counts"][name] for name in
        ("conv1_parameters", "primary_parameters", "vote_parameters", "decoder_parameters"))
    (ROOT / "mechanics-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"routing_error": result["numpy_torch_routing_max_error"],
                      "gradient_paths": gradients, "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
