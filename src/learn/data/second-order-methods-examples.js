// Complete independently runnable Python programs; native runtime versions are in the lesson.
export const secondOrderMethodsExamples = {
  quadraticSolve: {
    title: "An exact Newton solve, then a rotated valley",
    code: String.raw`import numpy as np

point = np.array([1.0, 1.0])
gradient = np.array([200.0, 2.0])
hessian = np.diag([200.0, 2.0])
step_to_subtract = np.linalg.solve(hessian, gradient)
print("step to subtract:", step_to_subtract)
print("new point:", point - step_to_subtract)

angle = np.pi / 6
rotation = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
hessian = rotation @ hessian @ rotation.T
gradient = hessian @ point
direction = np.linalg.solve(hessian, -gradient)
rate = 1.6 / 200
iterate = point.copy()
for _ in range(12):
    iterate -= rate * (hessian @ iterate)
print("rotated Newton direction:", np.round(direction, 6))
print("solve residual:", np.linalg.norm(hessian @ direction + gradient) < 1e-12)
print("GD after 12:", np.round(iterate, 6))
print("condition number:", round(np.linalg.cond(hessian), 6))
`,
    expected: "step to subtract: [1. 1.]\nnew point: [0. 0.]\nrotated Newton direction: [-1. -1.]\nsolve residual: True\nGD after 12: [-0.148232  0.262693]\ncondition number: 100.0",
  },
  safeguardedNewton: {
    title: "Check the direction before accepting a Newton step",
    code: String.raw`import numpy as np

def objective(point):
    x, y = point
    return (x*x - 1)**2 / 4 + y*y / 2

def attempt(point, damping):
    x, y = point
    gradient = np.array([x * (x*x - 1), y])
    hessian = np.diag([3*x*x - 1, 1.0])
    system = hessian + damping * np.eye(2)
    if np.min(np.linalg.eigvalsh(system)) <= 0:
        raw = np.linalg.solve(system, -gradient)
        return {"status": "indefinite: reject as SPD model", "raw": raw,
                "slope": gradient @ raw, "full_loss": objective(point + raw)}
    direction = np.linalg.solve(system, -gradient)
    slope = gradient @ direction
    if slope >= 0:
        return {"status": "no strict descent direction"}
    for reductions in range(20):
        alpha = 0.5**reductions
        candidate = point + alpha * direction
        if objective(candidate) <= objective(point) + 1e-4 * alpha * slope:
            return {"status": "accepted", "alpha": alpha, "point": candidate,
                    "loss": objective(candidate), "reductions": reductions}
    raise RuntimeError("Backtracking budget exhausted; inspect the model.")

point = np.array([0.2, 0.5])
raw = attempt(point, 0)
print(raw["status"])
print("raw direction:", np.round(raw["raw"], 6))
print("raw full-step loss:", round(raw["full_loss"], 6))
for point, damping in [(point, 1.2), (np.array([0.6, 0.5]), 0.01)]:
    result = attempt(point, damping)
    print("start:", point, "damping:", damping)
    print(result["status"], "alpha:", result["alpha"],
          "point:", np.round(result["point"], 6), "loss:", round(result["loss"], 6))
`,
    expected: "indefinite: reject as SPD model\nraw direction: [-0.218182 -0.5     ]\nraw full-step loss: 0.249835\nstart: [0.2 0.5] damping: 1.2\naccepted alpha: 1.0 point: [0.8      0.272727] loss: 0.06959\nstart: [0.6 0.5] damping: 0.01\naccepted alpha: 0.125 point: [1.133333 0.438119] loss: 0.116201",
  },
  limitedMemory: {
    title: "Apply an inverse without storing it: the two-loop recursion",
    code: String.raw`import numpy as np

def two_loop(gradient, pairs, gamma):
    q = gradient.copy()
    coefficients = []
    for step, change in reversed(pairs):
        curvature = step @ change
        if curvature <= 1e-12:
            raise ValueError("This teaching implementation requires positive curvature.")
        alpha = (step @ q) / curvature
        coefficients.append(alpha)
        q -= alpha * change
    result = gamma * q
    for (step, change), alpha in zip(pairs, reversed(coefficients)):
        beta = (change @ result) / (step @ change)
        result += step * (alpha - beta)
    return result

def dense_reference(pairs, gamma, size):
    inverse = gamma * np.eye(size)
    for step, change in pairs:
        rho = 1 / (step @ change)
        correction = np.eye(size) - rho * np.outer(step, change)
        inverse = correction @ inverse @ correction.T + rho * np.outer(step, step)
    return inverse

hessian = np.array([[4., 1.], [1., 2.]])
steps = [np.array(value, dtype=float) for value in [[1, 0], [0, 1], [1, -1]]]
history = [(step, hessian @ step) for step in steps]
gradient = np.array([3., 1.])
for memory in [1, 2, 3]:
    pairs = history[-memory:]
    step, change = pairs[-1]
    gamma = (step @ change) / (change @ change)
    transformed = two_loop(gradient, pairs, gamma)
    inverse = dense_reference(pairs, gamma, 2)
    assert np.allclose(transformed, inverse @ gradient, atol=1e-12)
    assert np.allclose(inverse @ change, step, atol=1e-12)
    print("memory:", memory, "direction:", np.round(-transformed, 6))
print("exact Newton:", np.round(np.linalg.solve(hessian, -gradient), 6))
try:
    two_loop(gradient, [(steps[-1], -history[-1][1])], 1)
except ValueError:
    print("negative-curvature pair: rejected")
`,
    expected: "memory: 1 direction: [-0.8 -0.4]\nmemory: 2 direction: [-0.725 -0.175]\nmemory: 3 direction: [-0.713281 -0.139844]\nexact Newton: [-0.714286 -0.142857]\nnegative-curvature pair: rejected",
  },
  lbfgsFit: {
    title: "Fit a decay curve using an actual PyTorch L-BFGS closure",
    code: String.raw`import torch

torch.set_default_dtype(torch.float64)
times = torch.linspace(0, 4, 21)
# Deterministic small measurement perturbations, fixed for every closure call.
noise = 0.01 * torch.sin(3 * times)
observations = 2 * torch.exp(-0.7 * times) + 0.3 + noise
# Positive amplitude/rate are represented by their logarithms.
parameters = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
optimizer = torch.optim.LBFGS([parameters], lr=1, max_iter=80,
                            history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)
evaluations = 0

def loss_value():
    amplitude, rate = parameters[:2].exp()
    prediction = amplitude * torch.exp(-rate * times) + parameters[2]
    return ((prediction - observations)**2).mean()

def closure():
    global evaluations
    optimizer.zero_grad()
    loss = loss_value()
    loss.backward()
    evaluations += 1
    return loss

returned_loss = optimizer.step(closure)
with torch.no_grad():
    fitted = torch.cat((parameters[:2].exp(), parameters[2:]))
    final_loss = loss_value()
print("amplitude, rate, offset:", [round(value, 6) for value in fitted.tolist()])
print("returned initial loss:", round(returned_loss.item(), 6))
print("reevaluated final loss:", round(final_loss.item(), 8))
print("closure evaluated more than once:", evaluations > 1)
print("final mean squared error below 0.0001:", final_loss.item() < 0.0001)
`,
    expected: "amplitude, rate, offset: [2.009861, 0.698043, 0.295096]\nreturned initial loss: 0.612318\nreevaluated final loss: 4.328e-05\nclosure evaluated more than once: True\nfinal mean squared error below 0.0001: True",
  },
  probabilityGeometry: {
    title: "Calculate Fisher from model outcomes, then compare coordinates",
    code: String.raw`import numpy as np

p, target, fraction = 0.2, 0.8, 0.25
outcome_probabilities = np.array([1-p, p])
scores = np.array([-1/(1-p), 1/p])
fisher = outcome_probabilities @ scores**2
empirical = np.array([1-target, target]) @ scores**2
gradient = (p-target) / (p*(1-p))
natural_probability_direction = -gradient / fisher
logit = np.log(p/(1-p))
natural_logit_direction = (target-p) / (p*(1-p))
direct_next = p + fraction * natural_probability_direction
logit_next = 1 / (1 + np.exp(-(logit + fraction * natural_logit_direction)))

def kl(next_probability):
    return p*np.log(p/next_probability) + (1-p)*np.log((1-p)/(1-next_probability))

print("Fisher, observed-label outer product:", round(fisher, 6), round(empirical, 6))
print("probability tangent:", round(natural_probability_direction, 6))
print("mapped logit tangent:", round(p*(1-p)*natural_logit_direction, 6))
print("finite endpoints:", round(direct_next, 6), round(logit_next, 6))
print("exact KLs:", round(kl(direct_next), 6), round(kl(logit_next), 6))
print("local probability KL estimate:", round(0.5*fisher*(direct_next-p)**2, 6))
`,
    expected: "Fisher, observed-label outer product: 6.25 20.3125\nprobability tangent: 0.6\nmapped logit tangent: 0.6\nfinite endpoints: 0.35 0.389647\nexact KLs: 0.054188 0.083075\nlocal probability KL estimate: 0.070312",
  },
  fisherFactors: {
    title: "Enumerate a small layer's true Fisher and inspect its factorization",
    code: String.raw`import itertools
import numpy as np

inputs = np.array([[1., -1.], [1., 2.]])
labels = np.array([[1., 0.], [0., 1.]])
weights = np.array([[0.8, -0.5], [-0.4, 0.6]])
input_factor = np.zeros((2, 2))
output_factor = np.zeros((2, 2))
fisher = np.zeros((4, 4))
enumerated = np.zeros_like(fisher)
gradient = np.zeros_like(weights)
for input_vector, label in zip(inputs, labels):
    probabilities = 1 / (1 + np.exp(-(weights @ input_vector)))
    input_outer = np.outer(input_vector, input_vector)
    covariance = np.diag(probabilities*(1-probabilities))
    input_factor += input_outer / 2
    output_factor += covariance / 2
    fisher += np.kron(input_outer, covariance) / 2
    gradient += np.outer(probabilities-label, input_vector) / 2
    for outcome in itertools.product([0, 1], repeat=2):
        outcome = np.array(outcome)
        mass = np.prod(np.where(outcome, probabilities, 1-probabilities))
        score = np.outer(outcome-probabilities, input_vector).ravel(order="F")
        enumerated += mass * np.outer(score, score) / 2
factored = np.kron(input_factor, output_factor)
assert np.allclose(fisher, enumerated, atol=1e-14)
flat_gradient = gradient.ravel(order="F")
damping = 0.1
exact_direction = np.linalg.solve(fisher + damping*np.eye(4), -flat_gradient)
factored_direction = np.linalg.solve(factored + damping*np.eye(4), -flat_gradient)
factor_damped = np.kron(input_factor + np.sqrt(damping)*np.eye(2),
                       output_factor + np.sqrt(damping)*np.eye(2))
print("model-outcome enumeration agrees:", np.allclose(fisher, enumerated))
print("factorization error:", round(np.linalg.norm(factored-fisher), 6))
print("exact damped direction:", np.round(exact_direction, 6))
print("factored damped direction:", np.round(factored_direction, 6))
print("factor-damping is a different matrix:", not np.allclose(factor_damped, factored+damping*np.eye(4)))
# With no damping, factors can be applied by two smaller solves.
left_solved = np.linalg.solve(output_factor, gradient)
matrix_result = np.linalg.solve(input_factor, left_solved.T).T
assert np.allclose(matrix_result.ravel(order="F"), np.linalg.solve(factored, flat_gradient))
print("two factor solves agree with Kronecker solve:", True)
`,
    expected: "model-outcome enumeration agrees: True\nfactorization error: 0.105332\nexact damped direction: [ 0.059705 -0.216708 -0.834831  0.749927]\nfactored damped direction: [-0.08446  -0.186955 -0.884957  0.756229]\nfactor-damping is a different matrix: True\ntwo factor solves agree with Kronecker solve: True",
  },
  shampooReplay: {
    title: "Apply original matrix Shampoo to a declared gradient replay",
    code: String.raw`import numpy as np

gradients = [np.array(value, dtype=float) for value in
             [[[1, 2], [0, 1]], [[2, 0], [1, -1]], [[0.5, -1], [2, 0.5]]]]

def inverse_quarter(matrix):
    values, vectors = np.linalg.eigh(matrix)
    if values.min() <= 0:
        raise ValueError("Positive definite accumulators are required.")
    return (vectors * values**(-0.25)) @ vectors.T

def replay(sequence, epsilon=0.1):
    rows, columns = sequence[0].shape
    left, right = epsilon*np.eye(rows), epsilon*np.eye(columns)
    parameter = np.zeros((rows, columns))
    directions = []
    for gradient in sequence:
        left += gradient @ gradient.T
        right += gradient.T @ gradient
        direction = inverse_quarter(left) @ gradient @ inverse_quarter(right)
        parameter -= 0.1 * direction
        directions.append(direction)
    return parameter, directions, left, right

parameter, directions, left, right = replay(gradients)
angle = np.pi / 5
rotation = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
rotated_parameter, rotated_directions, _, _ = replay([rotation @ g for g in gradients])
assert all(np.allclose(actual, rotation @ original, atol=1e-12)
           for actual, original in zip(rotated_directions, directions))
print("first preconditioned gradient:", np.round(directions[0], 6))
print("final left accumulator:", np.round(left, 6))
print("final right accumulator:", np.round(right, 6))
print("parameter after three replay updates:", np.round(parameter, 6))
print("row rotation commutes with the update:", np.allclose(rotated_parameter, rotation @ parameter))
`,
    expected: "first preconditioned gradient: [[ 0.631578  0.729922]\n [-0.533234  0.631578]]\nfinal left accumulator: [[10.35  4.5 ]\n [ 4.5   7.35]]\nfinal right accumulator: [[10.35  1.5 ]\n [ 1.5   7.35]]\nparameter after three replay updates: [[-0.140974 -0.04254 ]\n [-0.05231  -0.02647 ]]\nrow rotation commutes with the update: True",
  },
  hessianVector: {
    title: "Solve using Hessian-vector products, without building the Hessian",
    code: String.raw`import torch

torch.set_default_dtype(torch.float64)
point = torch.linspace(-0.8, 1.2, 6, requires_grad=True)

def objective(value):
    curvature = torch.arange(1, value.numel()+1, dtype=value.dtype)
    return (0.5*curvature*value**2 + 0.05*value**4).sum() + 0.1*((value[1:]-value[:-1])**2).sum()

gradient = torch.autograd.grad(objective(point), point)[0].detach()

def hessian_times(vector):
    _, product = torch.autograd.functional.hvp(objective, point, vector)
    return product.detach()

def conjugate_gradient(apply, right_hand_side, tolerance=1e-10, budget=30):
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    search = residual.clone()
    for iteration in range(budget):
        if torch.linalg.vector_norm(residual) <= tolerance:
            return solution, iteration
        product = apply(search)
        curvature = torch.dot(search, product)
        if curvature <= 0:
            raise ValueError("This solver requires positive definite curvature.")
        old_norm_squared = torch.dot(residual, residual)
        alpha = old_norm_squared / curvature
        solution += alpha*search
        residual -= alpha*product
        beta = torch.dot(residual, residual) / old_norm_squared
        search = residual + beta*search
    raise RuntimeError("Residual tolerance was not reached within the budget.")

direction, iterations = conjugate_gradient(hessian_times, -gradient)
print("CG iterations:", iterations)
print("direction:", [round(value, 6) for value in direction.tolist()])
print("relative solve residual below 1e-10:",
      (torch.linalg.vector_norm(hessian_times(direction)+gradient) / torch.linalg.vector_norm(gradient)).item() < 1e-10)
print("descent:", torch.dot(gradient, direction).item() < 0)
# A tiny independent check only: never build this matrix in the matrix-free solve above.
dense_hessian = torch.autograd.functional.hessian(objective, point)
print("six-variable dense oracle agrees:", torch.allclose(direction, torch.linalg.solve(dense_hessian, -gradient), atol=1e-10))
`,
    expected: "CG iterations: 6\ndirection: [0.668069, 0.379109, -0.000794, -0.392603, -0.760915, -1.101045]\nrelative solve residual below 1e-10: True\ndescent: True\nsix-variable dense oracle agrees: True",
  },
  changedFitComparison: {
    title: "One reproducible changed-data comparison",
    code: String.raw`import torch

torch.set_default_dtype(torch.float64)
times = torch.linspace(0, 6, 21)
observations = 1.5*torch.exp(-0.4*times) + 0.2 + 0.01*torch.sin(3*times)
tolerance = 1e-6
maximum_gradient_evaluations = 1000

def loss_value(parameters):
    amplitude, rate = parameters[:2].exp()
    predictions = amplitude*torch.exp(-rate*times) + parameters[2]
    return ((predictions-observations)**2).mean()

for method in ["L-BFGS", "gradient descent"]:
    parameters = torch.zeros(3, requires_grad=True)
    gradient_evaluations = 0
    if method == "L-BFGS":
        optimizer = torch.optim.LBFGS(
            [parameters], lr=1, max_iter=300,
            max_eval=maximum_gradient_evaluations-1,
            history_size=10, line_search_fn="strong_wolfe",
            tolerance_grad=tolerance, tolerance_change=1e-14)

        def closure():
            global gradient_evaluations
            optimizer.zero_grad()
            loss = loss_value(parameters)
            loss.backward()
            gradient_evaluations += 1
            return loss

        optimizer.step(closure)
    else:
        # One declared rate, without searching a grid against this fixture.
        for _ in range(maximum_gradient_evaluations-1):
            loss = loss_value(parameters)
            gradient = torch.autograd.grad(loss, parameters)[0]
            gradient_evaluations += 1
            if gradient.abs().max().item() <= tolerance:
                break
            with torch.no_grad():
                parameters -= 0.2*gradient
    # Include the independent final objective/gradient evaluation in the count.
    final_loss = loss_value(parameters)
    final_gradient = torch.autograd.grad(final_loss, parameters)[0]
    gradient_evaluations += 1
    with torch.no_grad():
        fit = torch.cat((parameters[:2].exp(), parameters[2:]))
    print(method)
    print("fit:", [round(value, 6) for value in fit.tolist()])
    print("final MSE:", round(final_loss.item(), 8))
    print("objective/gradient evaluations:", gradient_evaluations)
    print("gradient infinity norm:", round(final_gradient.abs().max().item(), 8))
    print("gradient tolerance reached:", final_gradient.abs().max().item() <= tolerance)
    assert gradient_evaluations <= maximum_gradient_evaluations
    assert torch.isfinite(fit).all() and final_loss.item() < 0.0001
`,
    expected: "L-BFGS\nfit: [1.506372, 0.399816, 0.197327]\nfinal MSE: 4.71e-05\nobjective/gradient evaluations: 15\ngradient infinity norm: 6e-08\ngradient tolerance reached: True\ngradient descent\nfit: [1.503948, 0.402745, 0.201367]\nfinal MSE: 4.78e-05\nobjective/gradient evaluations: 1000\ngradient infinity norm: 0.00014212\ngradient tolerance reached: False",
  },
};
