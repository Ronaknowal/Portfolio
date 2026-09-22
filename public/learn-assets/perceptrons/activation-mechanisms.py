"""NumPy activation rules beside their PyTorch counterparts; no dataset needed."""
import math
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Perceptron
from scipy.special import erfc


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    magnitude = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + magnitude), magnitude / (1 + magnitude))


def activation(name, z):
    """Return float64 forward values and local slopes, preserving input shape."""
    z = np.asarray(z, dtype=np.float64)
    if name in ("sigmoid", "silu"):
        magnitude = np.exp(-np.abs(z))
        s = np.where(z >= 0, 1 / (1 + magnitude), magnitude / (1 + magnitude))
        slope = magnitude / (1 + magnitude)**2
        return (s, slope) if name == "sigmoid" else (z * s, s + z * slope)
    if name == "tanh":
        value = np.tanh(z)
        return value, 1 - value**2
    if name == "relu":
        return np.maximum(z, 0), (z > 0).astype(float)
    if name == "leaky_relu":
        return np.where(z >= 0, z, .1 * z), np.where(z > 0, 1., .1)
    if name == "elu":
        negative = np.minimum(z, 0)
        value = np.where(z >= 0, z, np.expm1(negative))
        return value, np.where(z >= 0, 1., np.exp(negative))
    if name == "gelu":
        cdf = .5 * erfc(-z / math.sqrt(2))
        density = np.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
        return z * cdf, cdf + z * density
    if name == "gelu_tanh":
        scale = math.sqrt(2 / math.pi)
        gate = np.tanh(scale * (z + .044715 * z**3))
        slope = .5 * (1 + gate) + .5 * z * (1 - gate**2) * scale * (1 + 3 * .044715 * z**2)
        return .5 * z * (1 + gate), slope
    if name == "mish":
        softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
        gate = np.tanh(softplus)
        return z * gate, gate + z * (1 - gate**2) * sigmoid(z)
    raise ValueError(name)


def run():
    torch.set_num_threads(1)
    modules = {
        "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(.1), "elu": nn.ELU(1.),
        "gelu": nn.GELU(approximate="none"),
        "gelu_tanh": nn.GELU(approximate="tanh"),
        "silu": nn.SiLU(), "mish": nn.Mish(),
    }
    scores = np.array([-6., -2., -.5, 0., .5, 2., 6.])
    for name, module in modules.items():
        values, slopes = activation(name, scores)
        z = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
        result = module(z)
        library_slopes = torch.autograd.grad(result.sum(), z)[0].detach().numpy()
        value_error = np.max(np.abs(values - result.detach().numpy()))
        slope_error = np.max(np.abs(slopes - library_slopes))
        print(name, "value_max_error", f"{value_error:.3e}",
              "slope_max_error", f"{slope_error:.3e}")

    # Rows are examples; this weight layout matches nn.Linear: output × input.
    x = np.array([[2., -1.], [1., 3.]])
    weight = np.array([[1.5, -2.], [-.5, 1.]])
    bias = np.array([-1., .5])
    scores = x @ weight.T + bias
    layer = nn.Linear(2, 2, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(bias))
    print("affine_scores", scores.tolist())
    print("affine_max_error", float(np.max(np.abs(scores - layer(torch.from_numpy(x)).detach().numpy()))))
    for incoming_weight in (0., .5, 4.):
        _, slope = activation("sigmoid", np.array([0.]))
        input_value = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        torch.sigmoid(incoming_weight * input_value).backward()
        print("sigmoid_at_zero_input_sensitivity", incoming_weight,
              float(incoming_weight * slope[0]), input_value.grad.item())

    gate_weight = np.array([[.5, 1.], [-1., .25]])
    value_weight = np.array([[1., -.5], [.3, .8]])
    output_weight = np.array([[1., 2.], [-.5, 1.]])
    gate_scores = x @ gate_weight.T
    gate = gate_scores * sigmoid(gate_scores)
    gated_values = gate * (x @ value_weight.T)
    numpy_output = gated_values @ output_weight.T
    gate_layer, value_layer, output_layer = [nn.Linear(2, 2, bias=False, dtype=torch.float64) for _ in range(3)]
    with torch.no_grad():
        for layer, weights in zip((gate_layer, value_layer, output_layer), (gate_weight, value_weight, output_weight)):
            layer.weight.copy_(torch.from_numpy(weights))
        library_output = output_layer(nn.SiLU()(gate_layer(torch.from_numpy(x))) * value_layer(torch.from_numpy(x)))
    print("swiglu_output", np.round(numpy_output, 6).tolist())
    print("swiglu_max_error", f"{np.max(np.abs(numpy_output - library_output.numpy())):.3e}")

    # Match the earlier NumPy perceptron's row order, step and zero initialization.
    binary_inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    for name, labels in (("AND", [-1, -1, -1, 1]), ("XOR", [-1, 1, 1, -1])):
        classifier = Perceptron(max_iter=12, tol=None, shuffle=False, eta0=1., penalty=None)
        classifier.fit(binary_inputs, labels)
        print("library_perceptron", name, "weights", classifier.coef_.tolist(),
              "bias", classifier.intercept_.tolist(), "predictions", classifier.predict(binary_inputs).tolist())


if __name__ == "__main__":
    run()
