// Each exported program is independently executable. Shared definitions are
// included inside each complete string, rather than assumed from earlier cells.
const optimizerDefinition = `import numpy as np

class SmallOptimizer:
    """Dense float64 educational rules; epsilon is outside the square root."""
    def __init__(self, parameters, method="adamw", lr=0.1,
                 beta1=0.9, beta2=0.99, epsilon=1e-8, decay=0.0):
        allowed = {"sgd", "momentum", "adagrad", "rmsprop", "adam", "adamw"}
        if method not in allowed:
            raise ValueError("Unknown method")
        if not all(np.isfinite(x) for x in (lr, beta1, beta2, epsilon, decay)):
            raise ValueError("Hyperparameters must be finite")
        if lr <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or epsilon <= 0 or decay < 0:
            raise ValueError("Invalid optimizer hyperparameter")
        self.theta = np.array(parameters, dtype=float, copy=True)
        if self.theta.ndim != 1 or self.theta.size == 0 or not np.isfinite(self.theta).all():
            raise ValueError("Use a nonempty finite parameter vector")
        self.method, self.lr = method, lr
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon, self.decay = epsilon, decay
        self.first = np.zeros_like(self.theta)
        self.second = np.zeros_like(self.theta)
        self.t = 0

    def step(self, data_gradient):
        gradient = np.array(data_gradient, dtype=float, copy=True)
        if gradient.shape != self.theta.shape or not np.isfinite(gradient).all():
            raise ValueError("Gradient must match the parameter vector and be finite")
        # Commit a complete finite state; a failed arithmetic step changes nothing.
        theta, first, second = self.theta.copy(), self.first.copy(), self.second.copy()
        t = self.t
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if self.method != "adamw":
                gradient += self.decay * theta  # Coupled L2 where requested.
            t += 1
            if self.method == "sgd":
                direction = gradient
            elif self.method == "momentum":
                first = self.beta1 * first + gradient
                direction = first
            elif self.method == "adagrad":
                second += gradient**2
                direction = gradient / (np.sqrt(second) + self.epsilon)
            elif self.method == "rmsprop":
                second = self.beta2 * second + (1 - self.beta2) * gradient**2
                direction = gradient / (np.sqrt(second) + self.epsilon)
            else:
                first = self.beta1 * first + (1 - self.beta1) * gradient
                second = self.beta2 * second + (1 - self.beta2) * gradient**2
                corrected_first = first / (1 - self.beta1**t)
                corrected_second = second / (1 - self.beta2**t)
                direction = corrected_first / (np.sqrt(corrected_second) + self.epsilon)
            if self.method == "adamw":
                theta *= 1 - self.lr * self.decay
            theta -= self.lr * direction
            if not all(np.isfinite(value).all() for value in (theta, first, second)):
                raise FloatingPointError("Optimizer state exceeded finite float64 arithmetic")
        self.theta, self.first, self.second, self.t = theta, first, second, t
        return self.theta.copy()`;

const blockDefinition = `import numpy as np

def block_step(weights, gradient, first, second, velocity, step, method,
               lr=0.1, beta1=0.9, beta2=0.99, decay=0.0,
               coefficient=0.1, epsilon=1e-8):
    """LARS: LR inside velocity. LAMB: identity scaling, no ratio clipping.
    If weight norm or ratio denominator is zero, use ratio 1.
    All blocks participate in decay/scaling here; no bias/norm exclusions.
    """
    if method not in {"lars", "lamb"} or type(step) is not int or step < 1:
        raise ValueError("Choose a method and positive integer update count")
    arrays = [np.array(value, dtype=float, copy=True)
              for value in (weights, gradient, first, second, velocity)]
    weights, gradient, first, second, velocity = arrays
    if weights.ndim != 1 or weights.size == 0 or any(value.shape != weights.shape for value in arrays):
        raise ValueError("All states must match a nonempty vector")
    if not all(np.isfinite(value).all() for value in arrays) or (second < 0).any():
        raise ValueError("Finite states and nonnegative second moments required")
    if not all(np.isfinite(x) for x in (lr, beta1, beta2, decay, coefficient, epsilon)):
        raise ValueError("Finite hyperparameters required")
    if lr <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or decay < 0 or coefficient <= 0 or epsilon <= 0:
        raise ValueError("Invalid hyperparameter")
    weight_norm = np.linalg.norm(weights)
    if method == "lars":
        direction = gradient + decay * weights
        denominator = np.linalg.norm(gradient) + decay * weight_norm
        ratio = coefficient * weight_norm / denominator if weight_norm > 0 and denominator > 0 else 1.0
        velocity = beta1 * velocity + lr * ratio * direction
        updated = weights - velocity
    else:
        first = beta1 * first + (1 - beta1) * gradient
        second = beta2 * second + (1 - beta2) * gradient**2
        corrected_first = first / (1 - beta1**step)
        corrected_second = second / (1 - beta2**step)
        direction = corrected_first / (np.sqrt(corrected_second) + epsilon) + decay * weights
        denominator = np.linalg.norm(direction)
        ratio = weight_norm / denominator if weight_norm > 0 and denominator > 0 else 1.0
        updated = weights - lr * ratio * direction
    return updated, first, second, velocity, float(ratio)`;

export const gradientVariantsExamples = {
  batch: {
    title: 'Enumerate the gradients before drawing a random batch',
    code: `import itertools
import numpy as np

measurements = np.array([-3.0, -1.0, 1.0, 3.0])
theta, rate = 1.0, 0.2
gradient = theta - measurements
objective = lambda value: np.mean((value - measurements)**2) / 2
print("individual gradients:", gradient.tolist())
for size in (1, 2, 4):
    means = [gradient[list(batch)].mean()
             for batch in itertools.combinations(range(4), size)]
    print("batch", size, "mean", round(float(np.mean(means)), 6),
          "variance", round(float(np.var(means)), 6))
selected_gradient = gradient[3]
next_theta = theta - rate * selected_gradient
print("selected observation:", measurements[3])
print("theta:", round(theta, 3), "->", round(next_theta, 3))
print("full loss:", round(objective(theta), 3), "->", round(objective(next_theta), 3))`,
    expected: "individual gradients: [4.0, 2.0, 0.0, -2.0]\nbatch 1 mean 1.0 variance 5.0\nbatch 2 mean 1.0 variance 1.666667\nbatch 4 mean 1.0 variance 0.0\nselected observation: 3.0\ntheta: 1.0 -> 1.4\nfull loss: 3.0 -> 3.48",
  },
  momentum: {
    title: 'Trace three momentum steps, then evaluate lookahead',
    code: `theta, velocity = 5.0, 0.0
rate, beta = 0.1, 0.9
for step in range(1, 4):
    gradient = 2 * theta  # f(theta)=theta²
    velocity = beta * velocity + gradient
    theta -= rate * velocity
    print(step, "gradient", round(gradient, 3),
          "buffer", round(velocity, 3), "theta", round(theta, 3))

theta, velocity = 5.0, 0.0
for step in range(1, 4):
    lookahead = theta - rate * beta * velocity
    gradient = 2 * lookahead
    velocity = beta * velocity + gradient
    theta -= rate * velocity
    print("lookahead", step, round(lookahead, 6), "theta", round(theta, 6))`,
    expected: "1 gradient 10.0 buffer 10.0 theta 4.0\n2 gradient 8.0 buffer 17.0 theta 2.3\n3 gradient 4.6 buffer 19.9 theta 0.31\nlookahead 1 5.0 theta 4.0\nlookahead 2 3.1 theta 2.48\nlookahead 3 1.112 theta 0.8896",
  },
  adaptive: {
    title: 'Run complete adaptive rules on the same supplied gradient history',
    code: `${optimizerDefinition}

history = [[2, 0], [2, 0], [2, 4], [2, 0], [0, 0]]
for method in ("adagrad", "rmsprop", "adam"):
    optimizer = SmallOptimizer([0, 0], method=method, lr=0.1, beta2=0.9)
    for gradient in history:
        optimizer.step(gradient)
    print(method, "parameters", np.round(optimizer.theta, 6).tolist(),
          "square history", np.round(optimizer.second, 6).tolist())
print("These are supplied gradients; no shared training objective was evaluated.")`,
    expected: "adagrad parameters [-0.278446, -0.1] square history [16.0, 16.0]\nrmsprop parameters [-0.908262, -0.316228] square history [1.23804, 1.296]\nadam parameters [-0.486937, -0.156377] square history [1.23804, 1.296]\nThese are supplied gradients; no shared training objective was evaluated.",
  },
  correction: {
    title: 'Check Adam correction against an explicit weighted-history sum',
    code: `import numpy as np

history = np.array([2.0, -1.0, 3.0])
beta1, beta2 = 0.9, 0.99
first = second = 0.0
for step, gradient in enumerate(history, 1):
    first = beta1 * first + (1 - beta1) * gradient
    second = beta2 * second + (1 - beta2) * gradient**2
powers = np.arange(len(history) - 1, -1, -1)
weights = (1 - beta1) * beta1**powers
explicit = np.dot(weights, history)
print("first raw:", round(first, 6), "weighted sum:", round(float(explicit), 6))
print("weight sum:", round(float(weights.sum()), 6))
print("corrected first:", round(first / (1 - beta1**len(history)), 6))
print("corrected second:", round(second / (1 - beta2**len(history)), 6))
print("current gradient:", history[-1])
# Constant gradients give exact correction of the initial zero contribution.
for step in (1, 3, 10):
    raw = 2 * (1 - beta1**step)
    print("constant", step, "raw", round(raw, 6),
          "corrected", round(raw / (1 - beta1**step), 6))`,
    expected: "first raw: 0.372 weighted sum: 0.372\nweight sum: 0.271\ncorrected first: 1.372694\ncorrected second: 4.683479\ncurrent gradient: 3.0\nconstant 1 raw 0.2 corrected 2.0\nconstant 3 raw 0.542 corrected 2.0\nconstant 10 raw 1.302643 corrected 2.0",
  },
  decay: {
    title: 'Observe coupled L2 and direct AdamW shrinkage separately',
    code: `${optimizerDefinition}

for method in ("adam", "adamw"):
    optimizer = SmallOptimizer([2, 10], method=method, lr=0.1, decay=0.1)
    optimizer.step([0, 0])
    print(method, "theta", np.round(optimizer.theta, 6).tolist(),
          "first", np.round(optimizer.first, 6).tolist(),
          "second", np.round(optimizer.second, 6).tolist())
plain = np.array([2.0, 10.0])
gradient = np.array([1.0, -2.0])
coupled = plain - 0.1 * (gradient + 0.1 * plain)
decoupled = (1 - 0.1 * 0.1) * plain - 0.1 * gradient
print("plain SGD agrees:", bool(np.allclose(coupled, decoupled)))`,
    expected: "adam theta [1.9, 9.9] first [0.02, 0.1] second [0.0004, 0.01]\nadamw theta [1.98, 9.9] first [0.0, 0.0] second [0.0, 0.0]\nplain SGD agrees: True",
  },
  blocks: {
    title: 'Run fully defined LARS and LAMB block steps, including zero norms',
    code: `${blockDefinition}

for method in ("lars", "lamb"):
    for scale in (1.0, 0.1, 0.0):
        initial = scale * np.array([3.0, 4.0])
        weights = initial.copy()
        first = np.zeros(2)
        second = np.zeros(2)
        velocity = np.zeros(2)
        for step in range(1, 3):
            previous = weights.copy()
            weights, first, second, velocity, ratio = block_step(
                weights, [0.6, 0.8], first, second, velocity, step, method)
            relative = (np.linalg.norm(weights - previous) / np.linalg.norm(previous)
                        if np.linalg.norm(previous) else None)
            print(method, "scale", scale, "step", step, "ratio", round(ratio, 6),
                  "theta", np.round(weights, 6).tolist(),
                  "relative", None if relative is None else round(float(relative), 6))`,
    expected: "lars scale 1.0 step 1 ratio 0.5 theta [2.97, 3.96] relative 0.01\nlars scale 1.0 step 2 ratio 0.495 theta [2.9133, 3.8844] relative 0.019091\nlars scale 0.1 step 1 ratio 0.05 theta [0.297, 0.396] relative 0.01\nlars scale 0.1 step 2 ratio 0.0495 theta [0.29133, 0.38844] relative 0.019091\nlars scale 0.0 step 1 ratio 1.0 theta [-0.06, -0.08] relative None\nlars scale 0.0 step 2 ratio 0.01 theta [-0.1146, -0.1528] relative 0.91\nlamb scale 1.0 step 1 ratio 3.535534 theta [2.646447, 3.646447] relative 0.1\nlamb scale 1.0 step 2 ratio 3.185926 theta [2.327854, 3.327854] relative 0.1\nlamb scale 0.1 step 1 ratio 0.353553 theta [0.264645, 0.364645] relative 0.1\nlamb scale 0.1 step 2 ratio 0.318593 theta [0.232785, 0.332785] relative 0.1\nlamb scale 0.0 step 1 ratio 1.0 theta [-0.1, -0.1] relative None\nlamb scale 0.0 step 2 ratio 0.1 theta [-0.11, -0.11] relative 0.1",
  },
  experiment: {
    title: 'Fit the same complete tiny model under a declared comparison budget',
    code: `${optimizerDefinition}

def make_data():
    # Deterministic synthetic calibration data: intercept, input, noisy output.
    x = np.linspace(-2, 2, 32)
    design = np.column_stack([np.ones_like(x), x])
    target = 1 + 2*x + 0.1*np.sin(3*x)
    train = np.arange(32) % 4 != 0
    return design[train], target[train], design[~train], target[~train]

def mean_loss(design, target, theta):
    return float(np.mean((design @ theta - target)**2) / 2)

train_x, train_y, validation_x, validation_y = make_data()
rates = {"sgd": 0.05, "momentum": 0.01, "adamw": 0.05}
# These are declared candidate rates, not a universal recommendation or tuning proof.
for method, rate in rates.items():
    optimizer = SmallOptimizer([0, 0], method=method, lr=rate, decay=0)
    rng = np.random.default_rng(73)  # Same epoch permutations for each method.
    updates = 0
    for epoch in range(40):
        permutation = rng.permutation(len(train_y))
        for start in range(0, len(train_y), 6):
            indices = permutation[start:start + 6]
            batch_x, batch_y = train_x[indices], train_y[indices]
            residual = batch_x @ optimizer.theta - batch_y
            gradient = batch_x.T @ residual / len(indices)
            optimizer.step(gradient)
            updates += 1
    full_gradient = train_x.T @ (train_x @ optimizer.theta - train_y) / len(train_y)
    print(method, "updates", updates, "theta", np.round(optimizer.theta, 6).tolist(),
          "train", round(mean_loss(train_x, train_y, optimizer.theta), 6),
          "validation", round(mean_loss(validation_x, validation_y, optimizer.theta), 6),
          "gradient norm", round(float(np.linalg.norm(full_gradient)), 6))
baseline, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
print("least-squares reference:", np.round(baseline, 6).tolist())
print("No test set, wall-clock benchmark or neural-network ranking is claimed.")`,
    expected: "sgd updates 160 theta [1.001107, 1.975698] train 0.002114 validation 0.002191 gradient norm 0.000503\nmomentum updates 160 theta [1.00103, 1.974582] train 0.002114 validation 0.002196 gradient norm 0.001152\nadamw updates 160 theta [1.0005, 1.975536] train 0.002114 validation 0.002189 gradient norm 0.000812\nleast-squares reference: [1.001297, 1.975354]\nNo test set, wall-clock benchmark or neural-network ranking is claimed.",
  },
  resume: {
    title: 'Restart parameters alone, then compare with complete numerical state',
    code: `${optimizerDefinition}
import copy

def update(optimizer, rng):
    # f(theta)=||theta||²/2 with a fresh artificial zero-mean perturbation.
    gradient = optimizer.theta + rng.choice([-0.2, 0.2], size=2)
    optimizer.step(gradient)

original = SmallOptimizer([2, -1], method="adamw", lr=0.1, decay=0.01)
rng = np.random.default_rng(81)
for _ in range(5):
    update(original, rng)
# An in-memory numerical checkpoint; durable file serialization is separate.
resumed = copy.deepcopy(original)
resumed_rng = np.random.default_rng()
resumed_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
weights_only = SmallOptimizer(original.theta, method="adamw", lr=0.1, decay=0.01)
restart_rng = np.random.default_rng()
restart_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
for _ in range(5):
    update(original, rng)
    update(resumed, resumed_rng)
    update(weights_only, restart_rng)
print("same full state:", bool(np.array_equal(original.theta, resumed.theta)))
print("weights only agree:", bool(np.array_equal(original.theta, weights_only.theta)))
print("continuous:", np.round(original.theta, 6).tolist(), "t", original.t)
print("weights only:", np.round(weights_only.theta, 6).tolist(), "t", weights_only.t)`,
    expected: "same full state: True\nweights only agree: False\ncontinuous: [1.006609, -0.08404] t 10\nweights only: [0.987727, -0.028698] t 5",
  },
  accumulation: {
    title: 'Unequal microbatches must preserve the intended full mean',
    code: `import numpy as np

measurements = np.array([-3.0, -1.0, 3.0])
theta = 1.0
gradients = theta - measurements
batches = [gradients[:2], gradients[2:]]
wrong = np.mean([batch.mean() for batch in batches])
weighted = sum(batch.size * batch.mean() for batch in batches) / gradients.size
print("per-observation gradients:", gradients.tolist())
print("microbatch means:", [float(batch.mean()) for batch in batches])
print("wrong unweighted mean:", round(float(wrong), 6))
print("correct weighted mean:", round(float(weighted), 6))
print("direct full mean:", round(float(gradients.mean()), 6))
rate = 0.1
print("one full update:", round(theta - rate * weighted, 6))
changed = theta
for observations in (measurements[:2], measurements[2:]):
    changed -= rate * np.mean(changed - observations)
print("two separate updates:", round(float(changed), 6))
print("Accumulate at one parameter state, then call the optimizer once.")`,
    expected: "per-observation gradients: [4.0, 2.0, -2.0]\nmicrobatch means: [3.0, -2.0]\nwrong unweighted mean: 0.5\ncorrect weighted mean: 1.333333\ndirect full mean: 1.333333\none full update: 0.866667\ntwo separate updates: 0.93\nAccumulate at one parameter state, then call the optimizer once.",
  },
};
