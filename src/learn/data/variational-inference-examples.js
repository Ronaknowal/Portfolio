export const variationalExamples = {
  finite: {
    title: 'Calculate an ELBO and the best member of a restricted family',
    code: `import math

joint = [0.08, 0.20, 0.12]
evidence = sum(joint)
posterior = [weight / evidence for weight in joint]

def inspect(q):
    elbo = sum(prob * (math.log(weight) - math.log(prob))
               for prob, weight in zip(q, joint) if prob > 0)
    kl = sum(prob * math.log(prob / target)
             for prob, target in zip(q, posterior) if prob > 0)
    return elbo, kl

# Under qB=qC, differentiate the one-variable KL and solve for qA.
scale = posterior[0] + 2 * math.sqrt(posterior[1] * posterior[2])
first = posterior[0] / scale
restricted = [first, (1 - first) / 2, (1 - first) / 2]
print("posterior", [round(p, 6) for p in posterior])
for label, q in [("uniform", [1/3] * 3),
                 ("restricted", restricted), ("exact", posterior)]:
    elbo, kl = inspect(q)
    print(label, [round(p, 6) for p in q],
          round(elbo, 6), round(kl, 6))
    assert math.isclose(elbo + kl, math.log(evidence), abs_tol=1e-12)
`,
    expected: "posterior [0.2, 0.5, 0.3]\nuniform [0.333333, 0.333333, 0.333333] -0.986531 0.07024\nrestricted [0.205213, 0.397393, 0.397393] -0.942022 0.025732\nexact [0.2, 0.5, 0.3] -0.916291 0.0"
  },
  covariance: {
    title: 'Check the uncertainty of two different decisions',
    code: `import math

rho = 0.8
variance = 1 - rho * rho
target = [[1.0, rho], [rho, 1.0]]
mean_field = [[variance, 0.0], [0.0, variance]]

def linear_variance(weights, covariance):
    return sum(weights[i] * covariance[i][j] * weights[j]
               for i in range(2) for j in range(2))

print("marginal variances", target[0][0], round(variance, 3))
for label, weights in [("sum", [1, 1]), ("difference", [1, -1])]:
    print(label, round(linear_variance(weights, target), 3),
          round(linear_variance(weights, mean_field), 3))
print("best mean-field KL", round(-0.5 * math.log(variance), 6))
print("radius-one mass in 2D", round(1 - math.exp(-0.5), 6))
`,
    expected: "marginal variances 1.0 0.36\nsum 3.6 0.72\ndifference 0.4 0.72\nbest mean-field KL 0.510826\nradius-one mass in 2D 0.393469"
  },
  coordinates: {
    title: 'Run exact factor updates and retain the family gap',
    code: `import math

rho = 0.8
target_mean = [1.0, -1.0]
mean = [-2.0, 2.0]
variance = 1 - rho * rho
family_gap = -0.5 * math.log(variance)

def kl(values):
    dx, dy = [values[i] - target_mean[i] for i in range(2)]
    return family_gap + (dx*dx - 2*rho*dx*dy + dy*dy) / (2*variance)

previous = kl(mean)
for update in range(1, 25):
    coordinate = (update - 1) % 2
    other = 1 - coordinate
    mean[coordinate] = (target_mean[coordinate]
                        + rho * (mean[other] - target_mean[other]))
    current = kl(mean)
    assert current <= previous + 1e-12
    previous = current
    if update in [1, 2, 4, 24]:
        print(update, [round(value, 6) for value in mean],
              round(current, 6))
print("limit KL", round(family_gap, 6))
`,
    expected: "1 [3.4, 2.0] 5.010826\n2 [3.4, 0.92] 3.390826\n4 [2.536, 0.2288] 1.690474\n24 [1.017709, -0.985833] 0.510982\nlimit KL 0.510826"
  },
  mixture: {
    title: 'Preserve the two-mode comparison with stable log-density integration',
    code: `import math

def log_normal(x, mean=0.0, sd=1.0):
    return -math.log(sd) - 0.5*math.log(2*math.pi) - 0.5*((x-mean)/sd)**2

def log_target(x):
    first, second = log_normal(x, -3), log_normal(x, 3)
    top = max(first, second)
    return top + math.log(math.exp(first-top) + math.exp(second-top)) - math.log(2)

def reverse_kl(mean, sd, intervals=1440):
    # E under q becomes integration against standard-normal noise.
    step = 18 / intervals
    total = 0.0
    for index in range(intervals + 1):
        noise = -9 + index * step
        value = mean + sd * noise
        weight = 1 if index in (0, intervals) else (4 if index % 2 else 2)
        total += weight * math.exp(log_normal(noise)) * (
            log_normal(value, mean, sd) - log_target(value))
    return total * step / 3

for mean, sd in [(0, 3), (3, 1), (-3, 1)]:
    kl = reverse_kl(mean, sd)
    assert abs(kl - reverse_kl(mean, sd, 2880)) < 1e-8
    right = 0.5 * (1 + math.erf(mean / (sd * math.sqrt(2))))
    print(mean, sd, round(kl, 3), round(right, 6))
print("target probability of positive value", 0.5)
`,
    expected: "0 3 0.877 0.5\n3 1 0.689 0.99865\n-3 1 0.689 0.00135\ntarget probability of positive value 0.5"
  },
  optimizer: {
    title: 'Fit a Gaussian using actual reparameterized stochastic gradients',
    code: `import math
import random

# The target is known here so we can validate inference independently.
target_mean, target_sd = 1.5, 0.7
mean, log_sd = 0.0, math.log(1.2)
rng = random.Random(7)
first_moment, second_moment = [0.0, 0.0], [0.0, 0.0]

def exact_kl(mean, log_sd):
    sd = math.exp(log_sd)
    return (math.log(target_sd/sd)
            + (sd*sd + (mean-target_mean)**2)/(2*target_sd**2) - 0.5)

for iteration in range(1, 2001):
    sd = math.exp(log_sd)
    gradients = [0.0, 0.0]
    for _ in range(64):
        noise = rng.gauss(0, 1)
        value = mean + sd * noise
        slope = (target_mean - value) / target_sd**2
        gradients[0] += slope / 64
        gradients[1] += (1 + sd * noise * slope) / 64
    # Adam ascent: maximize ELBO, hence the plus sign.
    parameters = [mean, log_sd]
    rate = 0.03 / math.sqrt(1 + iteration / 200)
    for i, gradient in enumerate(gradients):
        first_moment[i] = 0.9*first_moment[i] + 0.1*gradient
        second_moment[i] = 0.999*second_moment[i] + 0.001*gradient**2
        corrected_first = first_moment[i] / (1 - 0.9**iteration)
        corrected_second = second_moment[i] / (1 - 0.999**iteration)
        parameters[i] += rate * corrected_first / (math.sqrt(corrected_second) + 1e-8)
    mean, log_sd = parameters
    assert math.isfinite(mean) and math.isfinite(log_sd)
    if iteration in [100, 500, 2000]:
        print(iteration, round(mean, 4), round(math.exp(log_sd), 4),
              round(exact_kl(mean, log_sd), 6))
print("exact mean and sd", target_mean, target_sd)
`,
    expected: "100 1.4879 0.7133 0.000507\n500 1.514 0.7122 0.0005\n2000 1.468 0.6858 0.001462\nexact mean and sd 1.5 0.7"
  },
  minibatches: {
    title: 'Prove a minibatch gradient is unbiased by checking every possible batch',
    code: `from itertools import combinations
from statistics import mean

data = [1.0, 2.0, 4.0, 5.0]
q_mean, q_sd = 1.0, 0.5
size, batch_size = len(data), 2
# Prior N(0,1), observations independent N(theta,1).
full_mean_gradient = sum(x - q_mean for x in data) - q_mean
full_log_sd_gradient = 1 - q_sd**2 * (1 + size)
estimates, wrong_estimates = [], []
for indices in combinations(range(size), batch_size):
    likelihood_gradient = sum(data[i] - q_mean for i in indices)
    estimates.append(size/batch_size * likelihood_gradient - q_mean)
    # Wrong: the prior is counted too many times as well.
    wrong_estimates.append(size/batch_size * (likelihood_gradient - q_mean))
print("all batch mean-gradients", estimates)
print("full and mean batch", full_mean_gradient, mean(estimates))
print("wrong mean batch", mean(wrong_estimates))
print("full log-sd gradient", full_log_sd_gradient)
print("exact posterior mean and variance", sum(data)/(1+size), 1/(1+size))
`,
    expected: "all batch mean-gradients [1.0, 5.0, 7.0, 7.0, 9.0, 13.0]\nfull and mean batch 7.0 7.0\nwrong mean batch 6.0\nfull log-sd gradient -0.25\nexact posterior mean and variance 2.4 0.2"
  },
  amortization: {
    title: 'A shared inference rule can output different posteriors and predictions',
    code: `from math import sqrt

# Independent cases: theta ~ N(0,1); one observation x | theta ~ N(theta,1).
def encoder(x):
    return x / 2, sqrt(0.5)

for observed in [-2, 0, 2]:
    mean, sd = encoder(observed)
    posterior_variance = sd * sd
    # A new measurement of the same latent theta adds unit observation variance.
    predictive_variance = 1 + posterior_variance
    print(observed, mean, round(posterior_variance, 3),
          round(predictive_variance, 3))

# These three observations are separate cases, not three updates of one theta.
# A fixed mean-zero encoder misses the conditional means for +/-2.
print("restricted mean-zero encoder KL for x=2", (0-1)**2 / (2*0.5))
`,
    expected: "-2 -1.0 0.5 1.5\n0 0.0 0.5 1.5\n2 1.0 0.5 1.5\nrestricted mean-zero encoder KL for x=2 1.0"
  }
};
