export const nonconvexLandscapeExamples = {
  wells: {
    title: "Find every stationary point before comparing minima",
    code: String.raw`import numpy as np

def well(x, tilt):
    return (x*x - 1)**2 / 4 + tilt*x

def stationary_points(tilt):
    roots = np.roots([1.0, 0.0, -1.0, tilt])
    return sorted(float(root.real) for root in roots if abs(root.imag) < 1e-10)

tilt = 0.15
for x in stationary_points(tilt):
    curvature = 3*x*x - 1
    kind = "minimum" if curvature > 0 else "maximum"
    print(f"x={x:.6f} cost={well(x, tilt):.6f} H={curvature:.6f} {kind}")
for initial in [-0.8, 0.8]:
    x = initial
    for _ in range(60):
        x -= 0.12 * (x**3 - x + tilt)
    print(f"start={initial:+.1f} end={x:.6f} cost={well(x, tilt):.6f}")`,
    expected: String.raw`x=-1.067923 cost=-0.155256 H=2.421379 minimum
x=0.153626 cost=0.261383 H=-0.929197 maximum
x=0.914297 cost=0.143874 H=1.507819 minimum
start=-0.8 end=-1.067923 cost=-0.155256
start=+0.8 end=0.914296 cost=0.143874`,
  },
  stationary: {
    title: "A zero Hessian leaves the test undecided",
    code: String.raw`import numpy as np

GEOMETRIES = {
    "bowl": ([2, 2], lambda x,y: x*x + y*y),
    "cap": ([-2, -2], lambda x,y: -x*x - y*y),
    "saddle": ([2, -2], lambda x,y: x*x - y*y),
    "quartic minimum": ([0, 0], lambda x,y: x**4 + y**4),
    "quartic saddle": ([0, 0], lambda x,y: x**4 - y**4),
    "flat valley": ([2, 0], lambda x,y: x*x),
}
for name, (diagonal, function) in GEOMETRIES.items():
    eigenvalues = np.linalg.eigvalsh(np.diag(diagonal))
    changes = [function(0.2, 0), function(0, 0.2)]
    print(name, "eigenvalues", eigenvalues.astype(int).tolist(),
          "axis changes", [round(value, 4) for value in changes])
# Sampling two axes supplies counterexamples, not a general minimum proof.
# For the quartic minimum, the proof is x**4 + y**4 > 0 off the origin.`,
    expected: String.raw`bowl eigenvalues [2, 2] axis changes [0.04, 0.04]
cap eigenvalues [-2, -2] axis changes [-0.04, -0.04]
saddle eigenvalues [-2, 2] axis changes [0.04, -0.04]
quartic minimum eigenvalues [0, 0] axis changes [0.0016, 0.0016]
quartic saddle eigenvalues [0, 0] axis changes [0.0016, -0.0016]
flat valley eigenvalues [0, 2] axis changes [0.04, 0]`,
  },
  saddle: {
    title: "Trace the stable and unstable coordinates separately",
    code: String.raw`import numpy as np

def gradient_descent(initial, rate, steps):
    point = np.array(initial, dtype=float)
    history = [point.copy()]
    for _ in range(steps):
        gradient = np.array([2*point[0], -2*point[1]])
        point = point - rate*gradient
        history.append(point.copy())
    return np.array(history)

rate, steps = 0.1, 20
for y0 in [0.0, 0.001]:
    end = gradient_descent([0.6, y0], rate, steps)[-1]
    closed_form = np.array([0.6*(1-2*rate)**steps, y0*(1+2*rate)**steps])
    assert np.allclose(end, closed_form)
    print(f"y0={y0:.3f}: x20={end[0]:.6f}, y20={end[1]:.6f}")`,
    expected: String.raw`y0=0.000: x20=0.006918, y20=0.000000
y0=0.001: x20=0.006918, y20=0.038338`,
  },
  noise: {
    title: "Average zero does not mean every trajectory stays zero",
    code: String.raw`from itertools import product
import numpy as np

def noisy_saddle(initial, rate, amplitude, direction, signs):
    point = np.array(initial, dtype=float)
    direction = np.asarray(direction, dtype=float)
    history = [point.copy()]
    for sign in signs:
        gradient = np.array([2*point[0], -2*point[1]])
        point = point - rate*(gradient + sign*amplitude*direction)
        history.append(point.copy())
    return np.array(history)

rate, amplitude, steps = 0.1, 0.2, 8
for name, direction in [("x only", [1,0]), ("y only", [0,1])]:
    endpoints = np.array([
        noisy_saddle([0.6,0], rate, amplitude, direction, signs)[-1]
        for signs in product([-1,1], repeat=steps)
    ])
    second_moment = np.mean(endpoints[:,1]**2)
    predicted = (rate*amplitude*direction[1])**2 * sum(
        (1+2*rate)**(2*j) for j in range(steps))
    assert np.isclose(second_moment, predicted)
    print(f"{name}: mean y={abs(endpoints[:,1].mean()):.6f}, E[y²]={second_moment:.6f}")
print("Every one of the 256 sign sequences is weighted equally.")`,
    expected: String.raw`x only: mean y=0.000000, E[y²]=0.000000
y only: mean y=0.000000, E[y²]=0.015899
Every one of the 256 sign sequences is weighted equally.`,
  },
  factors: {
    title: "Rescale an identical predictor and measure its curvature",
    code: String.raw`import numpy as np

def loss(point):
    a, b = point
    return 0.5*(a*b - 1)**2

def gradient(point):
    a, b = point
    return (a*b-1)*np.array([b,a])

def hessian(point):
    a, b = point
    return np.array([[b*b, 2*a*b-1], [2*a*b-1, a*a]])

epsilon = 0.1
for scale in [1.0, 4.0]:
    point = np.array([scale, 1/scale])
    a, b = point
    normal = np.array([b,a])/np.linalg.norm(point)
    tangent = np.array([a,-b])/np.linalg.norm(point)
    eigenvalues = np.linalg.eigvalsh(hessian(point))
    print(f"s={scale:g}: ab={a*b:g}, eigenvalues={eigenvalues.tolist()}")
    print(f"  normal loss={loss(point+epsilon*normal):.8f}; tangent loss={loss(point+epsilon*tangent):.8f}")
    print("  predictions:", (a*b*np.array([-2.0,0.0,3.0])).tolist())`,
    expected: String.raw`s=1: ab=1, eigenvalues=[0.0, 2.0]
  normal loss=0.01071961; tangent loss=0.00001250
  predictions: [-2.0, 0.0, 3.0]
s=4: ab=1, eigenvalues=[0.0, 16.0625]
  normal loss=0.08056221; tangent loss=0.00000019
  predictions: [-2.0, 0.0, 3.0]`,
  },
  relu: {
    title: "Move a whole hidden unit, including its bias",
    code: String.raw`import numpy as np

def network(inputs, incoming, bias, outgoing):
    hidden = np.maximum(0.0, np.asarray(inputs)[:,None]*incoming + bias)
    return hidden @ outgoing

inputs = np.array([-1.0,0.0,1.0,1.5,2.0,3.0])
w = np.array([1.0,-1.0])
b = np.array([-1.0,2.0])
v = np.array([2.0,-1.0])
original = network(inputs,w,b,v)
permutation = [1,0]
swapped = network(inputs,w[permutation],b[permutation],v[permutation])
scales = np.array([3.0,0.5])  # strictly positive
scaled = network(inputs,w*scales,b*scales,v/scales)
wrong = network(inputs,w*scales,b,v/scales)  # forgot bias scaling
assert np.allclose(swapped, original)
assert np.allclose(scaled, original)
print("original:", original.tolist())
print("swap error:", float(np.max(np.abs(swapped-original))))
print("positive scaling error:", float(np.max(np.abs(scaled-original))))
print("forgot-bias maximum error:", round(float(np.max(np.abs(wrong-original))), 6))`,
    expected: String.raw`original: [-3.0, -2.0, -1.0, 0.5, 2.0, 4.0]
swap error: 0.0
positive scaling error: 5.551115123125783e-17
forgot-bias maximum error: 2.0`,
  },
  paths: {
    title: "One interpolation is not every possible path",
    code: String.raw`import numpy as np

def loss(a,b):
    return 0.5*(a*b-1)**2

for t in [0.0,0.25,0.5,0.75,1.0]:
    straight = loss(1+t, 1-t/2)
    curved = loss(1+t, 1/(1+t))
    opposite = loss(1-2*t, 1-2*t)
    print(f"t={t:.2f}: straight={straight:.8f} curved={curved:.8f} opposite={opposite:.8f}")
# The table samples paths; the algebra in the text proves their full behavior.
# Any continuous path from (1,1) to (-1,-1) crosses a=0, where loss=1/2.`,
    expected: String.raw`t=0.00: straight=0.00000000 curved=0.00000000 opposite=0.00000000
t=0.25: straight=0.00439453 curved=0.00000000 opposite=0.28125000
t=0.50: straight=0.00781250 curved=0.00000000 opposite=0.50000000
t=0.75: straight=0.00439453 curved=0.00000000 opposite=0.28125000
t=1.00: straight=0.00000000 curved=0.00000000 opposite=0.00000000`,
  },
  generalization: {
    title: "Same flat training objective, different held-out answers",
    code: String.raw`import numpy as np

def predict(parameter, inputs):
    return inputs + parameter*(inputs*inputs-1)

def half_mse(predictions, targets):
    return float(np.mean((predictions-targets)**2)/2)

train_x = np.array([-1.0,1.0])
train_y = train_x.copy()
test_x = np.array([0.0])
test_y = np.array([0.0])  # declared target rule y=x
for parameter in [-1.0,0.0,1.0,2.0]:
    train = half_mse(predict(parameter,train_x),train_y)
    test = half_mse(predict(parameter,test_x),test_y)
    print(f"a={parameter:+.0f}: training={train:.1f}, held-out={test:.1f}")
print("Changing the assumed target rule can change the preferred predictor.")`,
    expected: String.raw`a=-1: training=0.0, held-out=0.5
a=+0: training=0.0, held-out=0.0
a=+1: training=0.0, held-out=0.5
a=+2: training=0.0, held-out=2.0
Changing the assumed target rule can change the preferred predictor.`,
  },
  dead: {
    title: "Diagnose an inactive unit before changing the learning rate",
    code: String.raw`import numpy as np

def train_unit(initial, rate, steps):
    point = np.array(initial, dtype=float)  # [w,b], input x=1, target y=1
    rows = []
    for step in range(steps+1):
        preactivation = point.sum()
        prediction = max(0.0, preactivation)
        # At the kink, choose derivative 0 as an implementation convention.
        active = preactivation > 0
        gradient = np.ones(2)*(prediction-1)*active
        rows.append((step, prediction, 0.5*(prediction-1)**2,
                     float(np.linalg.norm(gradient)), int(active)))
        if step < steps:
            point -= rate*gradient
    return rows

for name, initial in [("inactive", [0,-1]), ("active", [0,0.5])]:
    for step, prediction, loss, norm, active in train_unit(initial,0.1,3):
        print(f"{name} k={step}: prediction={prediction:.3f} loss={loss:.6f} grad={norm:.6f} active={active}")`,
    expected: String.raw`inactive k=0: prediction=0.000 loss=0.500000 grad=0.000000 active=0
inactive k=1: prediction=0.000 loss=0.500000 grad=0.000000 active=0
inactive k=2: prediction=0.000 loss=0.500000 grad=0.000000 active=0
inactive k=3: prediction=0.000 loss=0.500000 grad=0.000000 active=0
active k=0: prediction=0.500 loss=0.125000 grad=0.707107 active=1
active k=1: prediction=0.600 loss=0.080000 grad=0.565685 active=1
active k=2: prediction=0.680 loss=0.051200 grad=0.452548 active=1
active k=3: prediction=0.744 loss=0.032768 grad=0.362039 active=1`,
  },
  slice: {
    title: "A positive-looking slice can miss negative curvature",
    code: String.raw`import numpy as np

def objective(point):
    x,y,z = point
    return x*x + y*y - 0.1*z*z

hessian = np.diag([2.0,2.0,-0.2])
visible_basis = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])
slice_hessian = visible_basis.T @ hessian @ visible_basis
print("visible slice eigenvalues:", np.linalg.eigvalsh(slice_hessian).tolist())
print("full Hessian eigenvalues:", np.linalg.eigvalsh(hessian).tolist())
for point in [[0.2,0,0],[0,0.2,0],[0,0,0.2]]:
    print(point, "change", round(objective(point),6))`,
    expected: String.raw`visible slice eigenvalues: [2.0, 2.0]
full Hessian eigenvalues: [-0.2, 2.0, 2.0]
[0.2, 0, 0] change 0.04
[0, 0.2, 0] change 0.04
[0, 0, 0.2] change -0.004`,
  },
};
