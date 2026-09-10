// Complete independent programs; captured stdout is checked by native verification.
export const learningRateScheduleExamples = {
  finiteEndpoints: {
    title: "Give each used update an explicit endpoint",
    code: String.raw`import math

def warmup_cosine(update, total, warmup, peak, minimum):
    if not (isinstance(total, int) and total >= 2
            and isinstance(warmup, int) and 0 <= warmup < total
            and isinstance(update, int) and 0 <= update < total
            and math.isfinite(peak) and math.isfinite(minimum)
            and 0 <= minimum <= peak):
        raise ValueError("invalid finite schedule")
    if warmup and update < warmup:
        return peak * (update + 1) / warmup
    progress = ((update - warmup + 1) / (total - warmup)
                if warmup else update / (total - 1))
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2

def previous_convention(update, total, warmup, peak, minimum):
    if update < warmup:
        return peak * (update + 1) / warmup
    progress = (update - warmup) / (total - warmup)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2

for label, function in (("previous", previous_convention), ("declared endpoints", warmup_cosine)):
    values = [function(u, 10, 2, .001, .00001) for u in range(10)]
    print(label, [f"{value:.6f}" for value in values])
assert warmup_cosine(1, 10, 2, .001, .00001) == .001
assert warmup_cosine(9, 10, 2, .001, .00001) == .00001
print("zero warmup", [round(warmup_cosine(u, 3, 0, .2, .01), 6) for u in range(3)])
try:
    warmup_cosine(10, 10, 2, .001, .00001)
except ValueError:
    print("completed budget: rejected")`,
    expected: "previous ['0.000500', '0.001000', '0.001000', '0.000962', '0.000855', '0.000694', '0.000505', '0.000316', '0.000155', '0.000048']\ndeclared endpoints ['0.000500', '0.001000', '0.000962', '0.000855', '0.000694', '0.000505', '0.000316', '0.000155', '0.000048', '0.000010']\nzero warmup [0.2, 0.105, 0.01]\ncompleted budget: rejected",
  },
  noiseMoments: {
    title: "Check the noise recurrence against every sign sequence",
    code: String.raw`from itertools import product
import math

rates = [.2, .15, .08, .02]
curvature, sigma, initial = 2.0, 1.0, 3.0
mean, variance = initial, 0.0
for rate in rates:
    multiplier = 1 - rate * curvature
    mean = multiplier * mean
    variance = multiplier**2 * variance + rate**2 * sigma**2

outcomes = []
for signs in product((-1, 1), repeat=len(rates)):
    error = initial
    for rate, sign in zip(rates, signs):
        error -= rate * (curvature * error + sigma * sign)
    outcomes.append(error)
exact_mean = math.fsum(outcomes) / len(outcomes)
exact_square = math.fsum(value * value for value in outcomes) / len(outcomes)
assert math.isclose(mean, exact_mean, abs_tol=1e-14)
assert math.isclose(mean**2 + variance, exact_square, abs_tol=1e-14)
print("all equally likely paths", len(outcomes))
print("mean", round(mean, 6), "variance", round(variance, 6))
print("expected squared error", round(exact_square, 6))
rate = .2
floor = rate * sigma**2 / (curvature * (2 - rate * curvature))
print("constant-rate limiting variance", round(floor, 6))`,
    expected: "all equally likely paths 16\nmean 1.016064 variance 0.033675\nexpected squared error 1.066061\nconstant-rate limiting variance 0.0625",
  },
  oneCycleRuntime: {
    title: "Log the rate and momentum actually used by OneCycleLR",
    code: String.raw`import torch

parameter = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
optimizer = torch.optim.SGD([parameter], lr=.02, momentum=.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=.2, total_steps=12, pct_start=.25,
    div_factor=10, final_div_factor=100,
    base_momentum=.85, max_momentum=.95, three_phase=False)
used = []
for update in range(12):
    optimizer.zero_grad()
    loss = .5 * (parameter - 2)**2
    loss.backward()
    rate = optimizer.param_groups[0]["lr"]
    momentum = optimizer.param_groups[0]["momentum"]
    used.append((rate, momentum))
    optimizer.step()
    scheduler.step()  # prepares a later value; it is not used by the finished update
    print(update, f"used={rate:.6f}", f"momentum={momentum:.6f}")
assert abs(used[0][0] - .02) < 1e-12
assert abs(used[2][0] - .2) < 1e-12
assert abs(used[-1][0] - .0002) < 1e-12
print("prepared after budget", f'{optimizer.param_groups[0]["lr"]:.6f}')
print("No thirteenth optimizer update is run.")`,
    expected: "0 used=0.020000 momentum=0.950000\n1 used=0.110000 momentum=0.900000\n2 used=0.200000 momentum=0.850000\n3 used=0.193975 momentum=0.853015\n4 used=0.176628 momentum=0.861698\n5 used=0.150050 momentum=0.875000\n6 used=0.117447 momentum=0.891318\n7 used=0.082753 momentum=0.908682\n8 used=0.050150 momentum=0.925000\n9 used=0.023572 momentum=0.938302\n10 used=0.006225 momentum=0.946985\n11 used=0.000200 momentum=0.950000\nprepared after budget 0.006225\nNo thirteenth optimizer update is run.",
  },
  cosineAndRestarts: {
    title: "Separate cosine call counts from warm restarts",
    code: String.raw`import torch

for horizon in (4, 3):
    weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon, eta_min=.01)
    rates = []
    for _ in range(4):
        rates.append(optimizer.param_groups[0]["lr"])
        weight.grad = torch.ones_like(weight)
        optimizer.step()
        scheduler.step()
    print("T_max", horizon, "four used rates", [round(value, 6) for value in rates])

weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
optimizer = torch.optim.SGD([weight], lr=.2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1, eta_min=.01)
rates = []
for _ in range(8):
    rates.append(optimizer.param_groups[0]["lr"])
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    scheduler.step()
print("restart used rates", [round(value, 6) for value in rates])
print("weight retained through restarts", round(weight.item(), 6))
assert torch.isclose(weight.detach(), torch.tensor(-sum(rates), dtype=torch.float64))`,
    expected: "T_max 4 four used rates [0.2, 0.172175, 0.105, 0.037825]\nT_max 3 four used rates [0.2, 0.1525, 0.0575, 0.01]\nrestart used rates [0.2, 0.172175, 0.105, 0.037825, 0.2, 0.172175, 0.105, 0.037825]\nweight retained through restarts -1.03",
  },
  plateauRuntime: {
    title: "Observe strict improvement, patience and cooldown",
    code: String.raw`import torch

weight = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([weight], lr=.2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=.5, patience=1,
    threshold=.02, threshold_mode="abs", cooldown=1,
    min_lr=.025, eps=1e-8)
losses = [1.0, .9, .9, .89, .88, .88, .9, .87, .87, .87, .87, .87]
for observation, loss in enumerate(losses):
    # These are invented validation observations, not results of training this weight.
    scheduler.step(loss)
    print(observation, f"metric={loss:.2f}", f"best={scheduler.best:.2f}",
          f"bad={scheduler.num_bad_epochs}", f"cooldown={scheduler.cooldown_counter}",
          f'next_lr={optimizer.param_groups[0]["lr"]:.3f}')`,
    expected: "0 metric=1.00 best=1.00 bad=0 cooldown=0 next_lr=0.200\n1 metric=0.90 best=0.90 bad=0 cooldown=0 next_lr=0.200\n2 metric=0.90 best=0.90 bad=1 cooldown=0 next_lr=0.200\n3 metric=0.89 best=0.90 bad=0 cooldown=1 next_lr=0.100\n4 metric=0.88 best=0.90 bad=0 cooldown=0 next_lr=0.100\n5 metric=0.88 best=0.90 bad=1 cooldown=0 next_lr=0.100\n6 metric=0.90 best=0.90 bad=0 cooldown=1 next_lr=0.050\n7 metric=0.87 best=0.87 bad=0 cooldown=0 next_lr=0.050\n8 metric=0.87 best=0.87 bad=1 cooldown=0 next_lr=0.050\n9 metric=0.87 best=0.87 bad=0 cooldown=1 next_lr=0.025\n10 metric=0.87 best=0.87 bad=0 cooldown=0 next_lr=0.025\n11 metric=0.87 best=0.87 bad=1 cooldown=0 next_lr=0.025",
  },
  accumulationAndSkip: {
    title: "Advance an update schedule only when this optimizer actually runs",
    code: String.raw`import math
import torch

targets = [1, 3, 2, 4, 0, 2, 3, 1, 4, 2, 1, 3]
accumulation, total_updates = 2, 5  # six attempts; deliberately reject attempt two
rates = [.01 + .19 * (1 + math.cos(math.pi * u / 4)) / 2 for u in range(5)]
weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
optimizer = torch.optim.SGD([weight], lr=rates[0])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: rates[min(u, 4)] / rates[0])
scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
clock = {"committed": 0}
def record_step(optimizer, args, kwargs):
    clock["committed"] += 1
hook = optimizer.register_step_post_hook(record_step)
optimizer.zero_grad()
for microbatch, target in enumerate(targets):
    # Equal-sized microbatches: average their means over this effective batch.
    loss = .5 * (weight - target)**2 / accumulation
    scaler.scale(loss).backward()
    if (microbatch + 1) % accumulation:
        continue
    attempt = (microbatch + 1) // accumulation
    if attempt == 2:
        weight.grad.fill_(float("inf"))  # explicit controlled skipped-update fixture
    before = clock["committed"]
    rate = optimizer.param_groups[0]["lr"]
    scaler.step(optimizer)
    scaler.update()
    committed = clock["committed"] > before
    if committed and clock["committed"] < total_updates:
        scheduler.step()
    optimizer.zero_grad()
    print("attempt", attempt, "committed", committed,
          "used", f"{rate:.6f}" if committed else "none",
          "updates", clock["committed"], "weight", round(weight.item(), 6))
hook.remove()
assert clock["committed"] == total_updates
print("CPU GradScaler event contract checked; no GPU throughput claim.")`,
    expected: "attempt 1 committed True used 0.200000 updates 1 weight 0.4\nattempt 2 committed False used none updates 1 weight 0.4\nattempt 3 committed True used 0.172175 updates 2 weight 0.503305\nattempt 4 committed True used 0.105000 updates 3 weight 0.660458\nattempt 5 committed True used 0.037825 updates 4 weight 0.748951\nattempt 6 committed True used 0.010000 updates 5 weight 0.761461\nCPU GradScaler event contract checked; no GPU throughput claim.",
  },
  checkpointResume: {
    title: "Save the schedule, momentum and sampling state together",
    code: String.raw`import copy
import io
import torch

torch.set_num_threads(1)
features = torch.tensor([[1., 0.], [1., 1.], [1., 2.], [1., 3.]], dtype=torch.float64)
targets = torch.tensor([1., 2., 2., 4.], dtype=torch.float64)
def create():
    weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=.08, momentum=.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=11, eta_min=.002)
    generator = torch.Generator().manual_seed(17)
    return weight, optimizer, scheduler, generator

def advance(state, count):
    weight, optimizer, scheduler, generator = state
    rows = []
    for _ in range(count):
        sample = int(torch.randint(4, (1,), generator=generator))
        optimizer.zero_grad()
        loss = .5 * (features[sample] @ weight - targets[sample])**2
        loss.backward()
        rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        rows.append((sample, rate, weight.detach().clone()))
    return rows

original = create()
advance(original, 5)
weight, optimizer, scheduler, generator = original
buffer = io.BytesIO()
torch.save({"weight": weight.detach(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
            "completed": 5, "budget": 12}, buffer)
buffer.seek(0)
saved = torch.load(buffer, weights_only=True)
expected = advance(original, 7)

restored = create()  # initialize scheduler before loading optimizer rates
new_weight, new_optimizer, new_scheduler, new_generator = restored
with torch.no_grad():
    new_weight.copy_(saved["weight"])
new_scheduler.load_state_dict(saved["scheduler"])
new_optimizer.load_state_dict(saved["optimizer"])
new_generator.set_state(saved["generator"])
actual = advance(restored, saved["budget"] - saved["completed"])
assert all(a[:2] == b[:2] and torch.equal(a[2], b[2]) for a, b in zip(expected, actual))
print("identical remaining updates", len(actual))
print("final coefficients", [round(value, 6) for value in new_weight.tolist()])

incomplete = create()
with torch.no_grad():
    incomplete[0].copy_(saved["weight"])
incomplete[3].set_state(saved["generator"])
wrong = advance(incomplete, 7)
print("weights-only resume matches", torch.equal(wrong[-1][2], expected[-1][2]))
print("This checks one CPU process and a saved generator, not a distributed data loader.")`,
    expected: "identical remaining updates 7\nfinal coefficients [0.260373, 0.893214]\nweights-only resume matches False\nThis checks one CPU process and a saved generator, not a distributed data loader.",
  },
  decayAndMomentum: {
    title: "Calculate two effects that a rate-only plot hides",
    code: String.raw`import math

rates = [.2, .1, .05, .01]
decay = .3
weight = 2.0
for rate in rates:
    weight *= 1 - rate * decay  # zero adaptive displacement; parameter participates each time
expected = 2 * math.prod(1 - rate * decay for rate in rates)
assert weight == expected
print("cumulative decayed weight", round(weight, 6))
print("first-order exponential approximation", round(2 * math.exp(-decay * sum(rates)), 6))

gradient_buffer = 0.0
scaled_velocity = 0.0
first_weight = second_weight = 0.0
for rate, gradient in zip((.2, .05), (1.0, 1.0)):
    gradient_buffer = .9 * gradient_buffer + gradient
    first_weight -= rate * gradient_buffer
    scaled_velocity = .9 * scaled_velocity + rate * gradient
    second_weight -= scaled_velocity
print("gradient-storage momentum", round(first_weight, 6))
print("scaled-contribution velocity", round(second_weight, 6))
print("Changed learning rate prevents naive identification of these state conventions.")`,
    expected: "cumulative decayed weight 1.790857\nfirst-order exponential approximation 1.795255\ngradient-storage momentum -0.295\nscaled-contribution velocity -0.43\nChanged learning rate prevents naive identification of these state conventions.",
  },
  controlledExperiment: {
    title: "Compare documented configurations on the same finite data",
    code: String.raw`import math
import numpy as np

data_rng = np.random.default_rng(31)
train_x = np.linspace(-1, 1, 40)
train_y = .7 + 1.5 * train_x + data_rng.normal(0, .3, len(train_x))
validation_x = np.linspace(-.95, .95, 30)
validation_y = .7 + 1.5 * validation_x + data_rng.normal(0, .3, len(validation_x))
X = np.column_stack((np.ones(len(train_x)), train_x))
V = np.column_stack((np.ones(len(validation_x)), validation_x))
updates, peak, minimum = 120, .2, .01

def rate_for(policy, update):
    if policy == "constant":
        return peak
    if policy == "linear":
        return peak + (minimum - peak) * update / (updates - 1)
    warmup = 12
    if update < warmup:
        return peak * (update + 1) / warmup
    progress = (update - warmup + 1) / (updates - warmup)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2

for policy in ("constant", "linear", "warmup-cosine"):
    validation_errors = []
    for seed in (2, 7, 19):
        # Reset initialization and the complete sample stream for each policy.
        sample_stream = np.random.default_rng(seed).integers(len(X), size=updates)
        weight = np.zeros(2)
        for update, sample in enumerate(sample_stream):
            prediction_error = X[sample] @ weight - train_y[sample]
            gradient = prediction_error * X[sample]  # half-squared training loss
            weight -= rate_for(policy, update) * gradient
        validation_errors.append(float(np.mean((V @ weight - validation_y)**2)))
    print(policy, "validation MSE", [round(value, 6) for value in validation_errors],
          "mean", round(float(np.mean(validation_errors)), 6))
print("Each configuration: same 120 sample-gradient updates, same data and three matched streams.")
print("Peak and schedules were fixed, not independently tuned; this is not a universal ranking.")`,
    expected: "constant validation MSE [0.076153, 0.189687, 0.08856] mean 0.118133\nlinear validation MSE [0.075919, 0.066271, 0.076825] mean 0.073005\nwarmup-cosine validation MSE [0.073893, 0.065025, 0.078241] mean 0.072386\nEach configuration: same 120 sample-gradient updates, same data and three matched streams.\nPeak and schedules were fixed, not independently tuned; this is not a universal ranking.",
  }
};
