"""NumPy objectives and explicit derivatives; PyTorch is only the reference.

Run: python loss-mechanisms.py (numpy 2.3.5, torch 2.14.0+cpu).
All arrays are float64. Functions return MEAN loss and its input gradient.
"""
import json
import numpy as np
import torch
from torch.nn import functional as F


def regression(prediction, target, kind="mse", delta=1., quantile=.7):
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("use equal nonempty shapes")
    if delta <= 0 or not 0 < quantile < 1:
        raise ValueError("delta must be positive and quantile between zero and one")
    residual = prediction - target
    magnitude = np.abs(residual)
    if kind == "mse":
        values, gradient = residual ** 2, 2 * residual
    elif kind == "mae":
        values, gradient = magnitude, np.sign(residual)
    elif kind == "huber":
        values = np.where(magnitude <= delta, .5 * residual ** 2,
                          delta * (magnitude - .5 * delta))
        gradient = np.clip(residual, -delta, delta)
    elif kind == "quantile":
        values = np.maximum(-quantile * residual, (1 - quantile) * residual)
        gradient = np.where(residual < 0, -quantile, 1 - quantile)
        gradient = np.where(residual == 0, 0., gradient)  # chosen subgradient
    else:
        raise ValueError("unknown objective")
    return values.mean(), gradient / residual.size


def cross_entropy(logits, targets):
    """N by C logits, N hard class indices, unweighted mean."""
    if logits.ndim != 2 or min(logits.shape) == 0 or targets.shape != (len(logits),):
        raise ValueError("use nonempty N by C logits and N class indices")
    if not np.issubdtype(targets.dtype, np.integer) or np.any((targets < 0) | (targets >= logits.shape[1])):
        raise ValueError("targets must be integer indices in [0, C)")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    partition = exponential.sum(axis=1, keepdims=True)
    log_probability = shifted - np.log(partition)
    gradient = exponential / partition
    rows = np.arange(len(logits))
    loss = -log_probability[rows, targets].mean()
    gradient[rows, targets] -= 1
    return loss, gradient / len(logits)


def binary_focal(logits, targets, gamma=0.):
    """gamma=0 is BCE; no alpha or positive-class weighting."""
    if gamma < 0 or logits.shape != targets.shape or logits.size == 0:
        raise ValueError("use gamma >= 0 and equal nonempty shapes")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("this focal implementation uses binary hard labels")
    signed = (2 * targets - 1) * logits
    # logaddexp supplies only a stable scalar primitive, not a loss API.
    log_correct = -np.logaddexp(0., -signed)
    log_wrong = -np.logaddexp(0., signed)
    correct, wrong = np.exp(log_correct), np.exp(log_wrong)
    modulation = np.exp(gamma * log_wrong)
    values = -modulation * log_correct
    gradient = (2 * targets - 1) * modulation * (
        gamma * correct * log_correct - wrong)
    return values.mean(), gradient / logits.size


def squared_triplet(anchor, positive, negative, margin=1.):
    """Rows are triples; distance is SQUARED Euclidean, no mining here."""
    ap, an = anchor - positive, anchor - negative
    raw = (ap ** 2).sum(1) - (an ** 2).sum(1) + margin
    active = (raw > 0)[:, None] / len(anchor)
    return np.maximum(raw, 0).mean(), (
        2 * (negative - positive) * active,
        -2 * ap * active, 2 * an * active)


def paired_info_nce(queries, keys, temperature=.2):
    """One-way paired rows; nonzero, representable norms; no epsilon floor."""
    if temperature <= 0 or queries.ndim != 2 or queries.shape != keys.shape or min(queries.shape) == 0:
        raise ValueError("use positive temperature and equal nonempty N by D arrays")
    q_length = np.linalg.norm(queries, axis=1, keepdims=True)
    k_length = np.linalg.norm(keys, axis=1, keepdims=True)
    if np.any(q_length == 0) or np.any(k_length == 0):
        raise ValueError("zero vectors have no cosine direction")
    q, k = queries / q_length, keys / k_length
    loss, score_gradient = cross_entropy(q @ k.T / temperature,
                                         np.arange(len(q)))
    gq, gk = score_gradient @ k / temperature, score_gradient.T @ q / temperature
    # Backprop through length normalization, rather than treating q,k as raw.
    gq = (gq - q * (gq * q).sum(1, keepdims=True)) / q_length
    gk = (gk - k * (gk * k).sum(1, keepdims=True)) / k_length
    return loss, (gq, gk)


def main():
    torch.set_num_threads(1)
    results = {}

    def compare(name, arrays, scratch, library):
        inputs = [torch.tensor(x, dtype=torch.float64, requires_grad=True)
                  for x in arrays]
        expected = library(*inputs)
        expected.backward()
        value, gradients = scratch(*arrays)
        if not isinstance(gradients, tuple):
            gradients = (gradients,)
        np.testing.assert_allclose(value, expected.item(), rtol=1e-11, atol=1e-11)
        errors = []
        for manual, tensor in zip(gradients, inputs):
            actual = tensor.grad.numpy()
            np.testing.assert_allclose(manual, actual, rtol=1e-10, atol=1e-10)
            errors.append(float(np.max(np.abs(manual - actual))))
        results[name] = dict(loss=float(value), max_gradient_error=max(errors))

    prediction = np.array([-.3, .2, 2.4, 5.1])
    target = np.array([.4, -.4, 1.1, 1.])
    target_tensor = torch.tensor(target)
    for kind, api in (("mse", F.mse_loss), ("mae", F.l1_loss), ("huber", F.huber_loss)):
        compare(kind, [prediction], lambda p: regression(p, target, kind),
                lambda p: api(p, target_tensor))
    compare("quantile", [prediction], lambda p: regression(p, target, "quantile"),
            lambda p: torch.maximum(.7 * (target_tensor-p), -.3 * (target_tensor-p)).mean())
    logits = np.array([[1., 2., -.5], [-.2, 1., .6]])
    targets = np.array([0, 2])
    compare("cross_entropy", [logits], lambda z: cross_entropy(z, targets),
            lambda z: F.cross_entropy(z, torch.tensor(targets)))
    compare("extreme_cross_entropy", [np.array([[1000., -1000.]])],
            lambda z: cross_entropy(z, np.array([1])),
            lambda z: F.cross_entropy(z, torch.tensor([1])))
    binary_logits = np.array([-1000., -2., .4, 2., 1000., -1000., 1000.])
    binary_targets = np.array([1., 0., 1., 0., 0., 0., 1.])
    for gamma in (0., .5, 2.):
        def focal_reference(z):
            y = torch.tensor(binary_targets)
            log_miss = -F.softplus((2*y-1)*z)
            return ((gamma * log_miss).exp() * F.binary_cross_entropy_with_logits(
                z, y, reduction="none")).mean()
        compare(f"focal_gamma_{gamma}", [binary_logits],
                lambda z: binary_focal(z, binary_targets, gamma), focal_reference)
    a = np.array([[.2, -.4], [1., .3]])
    p = np.array([[.5, .8], [.9, .2]])
    n = np.array([[.4, .1], [4., 2.]])
    triplet = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda u, v: (u-v).square().sum(-1), margin=1.)
    compare("squared_triplet", [a, p, n], squared_triplet, triplet)
    keys = np.array([[.8, .3], [-.4, .9]])
    compare("paired_info_nce", [a, keys], paired_info_nce,
            lambda q, k: F.cross_entropy(F.normalize(q, dim=1, eps=0.) @
                F.normalize(k, dim=1, eps=0.).T / .2, torch.arange(len(q))))
    # Same-state affine classifier: manual CE gradient -> manual parameter step.
    features = np.array([[1., -2.], [.5, 1.], [-1., .2]])
    weight = np.array([[.2, -.1, .4], [.5, .3, -.2]])  # D by C
    bias = np.array([.1, -.2, .3])
    labels = np.array([0, 2, 1])
    before, dz = cross_entropy(features @ weight + bias, labels)
    next_weight = weight - .1 * features.T @ dz
    next_bias = bias - .1 * dz.sum(0)
    layer = torch.nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight.T)); layer.bias.copy_(torch.tensor(bias))
    optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
    optimizer.zero_grad()
    F.cross_entropy(layer(torch.tensor(features)), torch.tensor(labels)).backward()
    optimizer.step()
    np.testing.assert_allclose(next_weight.T, layer.weight.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(next_bias, layer.bias.detach().numpy(), atol=1e-12)
    after, _ = cross_entropy(features @ next_weight + next_bias, labels)
    results["one_classifier_update"] = dict(before=float(before), after=float(after),
                                          parameters_match=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
