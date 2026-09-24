export const multivariateCalculusExamples = {
  localExpansion: {
    title: 'Separate the exact change from its tangent-plane prediction',
    code: `import numpy as np

def f(point):
    x, y = point
    return x*x + 2*y*y

point = np.array([1., 1.])
gradient = np.array([2*point[0], 4*point[1]])
for scale in (1., .5, .1):
    change = scale * np.array([.1, -.05])
    predicted = gradient @ change
    actual = f(point + change) - f(point)
    remainder = change[0]**2 + 2*change[1]**2
    assert np.isclose(actual, predicted + remainder)
    print("change:", np.round(change, 4).tolist())
    print("prediction, actual, remainder:",
          [round(float(value), 6) for value in (predicted, actual, remainder)])`,
    expected: "change: [0.1, -0.05]\nprediction, actual, remainder: [0.0, 0.015, 0.015]\nchange: [0.05, -0.025]\nprediction, actual, remainder: [0.0, 0.00375, 0.00375]\nchange: [0.01, -0.005]\nprediction, actual, remainder: [0.0, 0.00015, 0.00015]",
  },
  directions: {
    title: 'Distinguish rate per distance from rate per time',
    code: `import numpy as np

point = np.array([3., -2.])
gradient = np.array([2*point[0], 4*point[1]])
velocity = np.array([3., 4.])
speed = np.linalg.norm(velocity)
unit = velocity / speed
per_distance = gradient @ unit
per_time = gradient @ velocity
print("unit direction:", unit.tolist())
print("rate per distance:", round(float(per_distance), 6))
print("rate per time:", round(float(per_time), 6))
print("speed times distance rate:", round(float(speed*per_distance), 6))
steepest_up = gradient / np.linalg.norm(gradient)
print("steepest unit direction:", steepest_up.tolist())
print("maximum unit-direction rate:", float(gradient @ steepest_up))`,
    expected: "unit direction: [0.6, 0.8]\nrate per distance: -2.8\nrate per time: -14.0\nspeed times distance rate: -14.0\nsteepest unit direction: [0.6, -0.8]\nmaximum unit-direction rate: 10.0",
  },
  approachPaths: {
    title: 'Evaluate a function along different approaches to the origin',
    code: `def g(x, y):
    if x == 0 and y == 0:
        return 0.
    return x*x*y / (x**4 + y*y)

for t in (1., .1, .01, .001):
    print("t:", t, "axis, line, parabola:",
          [round(g(t, 0), 6), round(g(t, t), 6), round(g(t, t*t), 6)])
print("value at origin:", g(0, 0))
print("axis partials:", (g(.001, 0)-g(0, 0))/.001,
      (g(0, .001)-g(0, 0))/.001)
print("line directional quotient near origin:", round(g(.0001, .0001)/.0001, 6))`,
    expected: "t: 1.0 axis, line, parabola: [0.0, 0.5, 0.5]\nt: 0.1 axis, line, parabola: [0.0, 0.09901, 0.5]\nt: 0.01 axis, line, parabola: [0.0, 0.009999, 0.5]\nt: 0.001 axis, line, parabola: [0.0, 0.001, 0.5]\nvalue at origin: 0.0\naxis partials: 0.0 0.0\nline directional quotient near origin: 1.0",
  },
  chainMotion: {
    title: 'Check the chain rule by differentiating the composed function',
    code: `import numpy as np

# All inputs are dimensionless here. r(t)=(t,t*t), f(x,y)=x*x+2*y.
def f(point):
    x, y = point
    return x*x + 2*y

def position(t):
    return np.array([t, t*t])

for t in (-1., 0., 2.):
    gradient = np.array([2*t, 2.])
    velocity = np.array([1., 2*t])
    chain_rate = gradient @ velocity
    composed_rate = 6*t  # f(r(t))=3*t*t
    h = 1e-5
    numerical = (f(position(t+h)) - f(position(t-h))) / (2*h)
    assert np.isclose(chain_rate, composed_rate)
    assert np.isclose(numerical, chain_rate, atol=1e-8)
    print("t, chain, direct, numerical:",
          [round(float(v), 6) for v in (t, chain_rate, composed_rate, numerical)])`,
    expected: "t, chain, direct, numerical: [-1.0, -6.0, -6.0, -6.0]\nt, chain, direct, numerical: [0.0, 0.0, 0.0, 0.0]\nt, chain, direct, numerical: [2.0, 12.0, 12.0, 12.0]",
  },
  circleConstraint: {
    title: 'Find both circle-constrained candidates and certify the extremes',
    code: `import numpy as np

gradient = np.array([2., 1.])
maximum = gradient / np.linalg.norm(gradient)
minimum = -maximum
for name, point in (("maximum", maximum), ("minimum", minimum)):
    tangent = np.array([-point[1], point[0]])
    value = gradient @ point
    multiplier = value / 2  # grad f = lambda * grad(x*x+y*y)
    assert np.allclose(gradient, multiplier * 2 * point)
    print(name, "point:", np.round(point, 6).tolist())
    print("value, tangent rate, lambda:",
          [round(float(v), 6) for v in (value, gradient @ tangent, multiplier)])
print("global bounds by Cauchy-Schwarz:",
      [-round(float(np.linalg.norm(gradient)), 6), round(float(np.linalg.norm(gradient)), 6)])`,
    expected: "maximum point: [0.894427, 0.447214]\nvalue, tangent rate, lambda: [2.236068, 0.0, 1.118034]\nminimum point: [-0.894427, -0.447214]\nvalue, tangent rate, lambda: [-2.236068, 0.0, -1.118034]\nglobal bounds by Cauchy-Schwarz: [-2.236068, 2.236068]",
  },
  curvature: {
    title: 'Compute a mixed-term Hessian and expose an inconclusive test',
    code: `import numpy as np

# f=x*x+2*x*y+2*y*y = 0.5 * point.T @ H @ point.
H = np.array([[2., 2.], [2., 4.]])
point = np.array([1., -1.])
change = np.array([.2, .1])
def f(p):
    return p[0]**2 + 2*p[0]*p[1] + 2*p[1]**2

linear = (H @ point) @ change
quadratic = .5 * change @ H @ change
print("H eigenvalues:", np.round(np.linalg.eigvalsh(H), 6).tolist())
print("linear, quadratic, exact change:",
      [round(float(v), 6) for v in (linear, quadratic, f(point+change)-f(point))])
assert np.isclose(f(point+change)-f(point), linear+quadratic)
for name, sign in (("quartic minimum", 1), ("quartic saddle", -1)):
    probes = [x**4 + sign*y**4 for x, y in ((.1, 0), (0, .1))]
    print(name, "origin Hessian: zero; axis values:", [round(v, 6) for v in probes])`,
    expected: "H eigenvalues: [0.763932, 5.236068]\nlinear, quadratic, exact change: [-0.2, 0.1, -0.1]\nquartic minimum origin Hessian: zero; axis values: [0.0001, 0.0001]\nquartic saddle origin Hessian: zero; axis values: [0.0001, -0.0001]",
  },
  descent: {
    title: 'Run the original two-parameter update and its changed rates',
    code: `import numpy as np

def trajectory(rate, steps):
    point = np.array([3., -2.])
    for _ in range(steps):
        gradient = np.array([2*point[0], 4*point[1]])
        point = point - rate*gradient
    return point

print("initial gradient:", [6., -8.])
print("first step at .1:", np.round(trajectory(.1, 1), 6).tolist())
for rate in (.01, .1, .5, .6):
    point = trajectory(rate, 12)
    closed_form = np.array([3*(1-2*rate)**12, -2*(1-4*rate)**12])
    assert np.allclose(point, closed_form)
    loss = point[0]**2 + 2*point[1]**2
    print("rate:", rate, "point:", np.round(point, 6).tolist(), "loss:", round(float(loss), 6))`,
    expected: "initial gradient: [6.0, -8.0]\nfirst step at .1: [2.4, -1.2]\nrate: 0.01 point: [2.35415, -1.22542] loss: 8.545329\nrate: 0.1 point: [0.206158, -0.004354] loss: 0.042539\nrate: 0.5 point: [0.0, -2.0] loss: 8.0\nrate: 0.6 point: [0.0, -113.387825] loss: 25713.597603",
  },
  unitsAndScaling: {
    title: 'Convert sensitivity units and test first-order cancellation',
    code: `import numpy as np

# Ideal resistor: P=V*V/R. Inputs V in volts and R in ohms; P in watts.
voltage, resistance = 10., 5.
gradient = np.array([2*voltage/resistance, -voltage**2/resistance**2])
change = np.array([.1, .1])
initial = voltage**2 / resistance
actual = (voltage+change[0])**2 / (resistance+change[1]) - initial
print("P (W):", initial)
print("dP/dV (W/V), dP/dR (W/ohm):", gradient.tolist())
print("linear change (W):", float(gradient @ change))
print("actual change (W):", round(float(actual), 9))
print("dP/d millivolt (W/mV):", gradient[0] / 1000)

# Coordinate scaling x=z1, y=z2/10 for f=x*x+100*y*y.
point = np.array([1., 1.])
z = point * np.array([1., 10.])
gradient_original = np.array([2*point[0], 200*point[1]])
gradient_z = 2*z
raw_step = point - .1*gradient_original
scaled_step = (z-.1*gradient_z) / np.array([1., 10.])
print("raw versus rescaled-coordinate step:", raw_step.tolist(), scaled_step.tolist())`,
    expected: "P (W): 20.0\ndP/dV (W/V), dP/dR (W/ohm): [4.0, -4.0]\nlinear change (W): 0.0\nactual change (W): 0.001960784\ndP/d millivolt (W/mV): 0.004\nraw versus rescaled-coordinate step: [0.8, -19.0] [0.8, 0.8]",
  },
  automaticDerivative: {
    title: 'Execute a small complete forward automatic derivative',
    code: `import math
import numpy as np

class Dual:
    """Value and two input derivatives; only the operations used below."""
    def __init__(self, value, derivative):
        self.value = float(value)
        self.derivative = np.asarray(derivative, dtype=float)

    @staticmethod
    def lift(other):
        return other if isinstance(other, Dual) else Dual(other, [0., 0.])

    def __add__(self, other):
        other = self.lift(other)
        return Dual(self.value+other.value, self.derivative+other.derivative)
    __radd__ = __add__

    def __mul__(self, other):
        other = self.lift(other)
        return Dual(self.value*other.value,
                    self.derivative*other.value+self.value*other.derivative)
    __rmul__ = __mul__

    def sin(self):
        return Dual(math.sin(self.value), math.cos(self.value)*self.derivative)

x, y = Dual(1.2, [1., 0.]), Dual(-.7, [0., 1.])
result = x*x*y + x.sin()
manual = np.array([2*x.value*y.value+math.cos(x.value), x.value**2])
def f(a, b):
    return a*a*b+math.sin(a)
h = 1e-5
finite_difference = np.array([(f(1.2+h, -.7)-f(1.2-h, -.7))/(2*h),
                              (f(1.2, -.7+h)-f(1.2, -.7-h))/(2*h)])
assert np.allclose(result.derivative, manual)
assert np.allclose(result.derivative, finite_difference, atol=1e-8)
print("value:", round(result.value, 6))
print("AD, manual, central difference:")
for vector in (result.derivative, manual, finite_difference):
    print(np.round(vector, 6).tolist())
print("40 repeated local factors .5 or 2:", f"{.5**40:.3e}", f"{2.**40:.3e}")`,
    expected: "value: -0.075961\nAD, manual, central difference:\n[-1.317642, 1.44]\n[-1.317642, 1.44]\n[-1.317642, 1.44]\n40 repeated local factors .5 or 2: 9.095e-13 1.100e+12",
  },
  practice: {
    title: 'Check the independent quadratic and directional tasks',
    code: `import numpy as np

# New function: f=x*y+y*y at (2,-1), change=(.03,.02).
point = np.array([2., -1.])
change = np.array([.03, .02])
def f(p):
    return p[0]*p[1]+p[1]**2
gradient = np.array([point[1], point[0]+2*point[1]])
print("gradient:", gradient.tolist())
print("predicted and actual change:",
      [round(float(v), 6) for v in (gradient @ change, f(point+change)-f(point))])
direction = np.array([-3., 4.]) / 5
print("unit-direction rate:", round(float(gradient @ direction), 6))

# New loss q=2*x*x+3*y*y: eigenvalues of H are 4 and 6.
for rate in (.2, 1/3, .4):
    factors = np.array([1-4*rate, 1-6*rate])
    point = np.array([1., 1.]) * factors**6
    print("rate:", round(rate, 6), "factors:", np.round(factors, 6).tolist(),
          "six-step point:", np.round(point, 6).tolist())`,
    expected: "gradient: [-1.0, 0.0]\npredicted and actual change: [-0.03, -0.029]\nunit-direction rate: 0.6\nrate: 0.2 factors: [0.2, -0.2] six-step point: [6.4e-05, 6.4e-05]\nrate: 0.333333 factors: [-0.333333, -1.0] six-step point: [0.001372, 1.0]\nrate: 0.4 factors: [-0.6, -1.4] six-step point: [0.046656, 7.529536]",
  },
};
