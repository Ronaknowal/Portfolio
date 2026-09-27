"""Small NumPy teaching rules; algorithms and exact scope are documented in lesson.md.

All arrays are float64. Prodigy is paper Algorithm 4, without bias correction or
weight decay. Sophia uses the paper's batch-scaled GNB estimate. These are
complete single-array algorithms, not production PyTorch optimizer wrappers.
"""
import numpy as np


class Optimizer:
    def __init__(self, initial, method, rate, warmup=20):
        self.method, self.rate, self.warmup = method, rate, warmup
        self.parameters = np.array(initial, dtype=np.float64, copy=True)
        self.state = {"step": 0}
        if method in ("adamw", "adamw_cosine", "lion", "sophia_g", "prodigy"):
            self.state["momentum"] = np.zeros_like(self.parameters)
        if method in ("adamw", "adamw_cosine", "schedule_free", "prodigy"):
            self.state["second"] = np.zeros_like(self.parameters)
        if method == "sophia_g":
            self.state["curvature"] = np.zeros_like(self.parameters)
        if method == "prodigy":
            self.state.update(initial=self.parameters.copy(),
                              displacement_sum=np.zeros_like(self.parameters),
                              numerator=0.0, distance=1e-6)
        if method == "schedule_free":
            self.state.update(average=self.parameters.copy(),
                              fast=self.parameters.copy(), weight_sum=0.0)

    def evaluation_parameters(self):
        return self.state["average"] if self.method == "schedule_free" else self.parameters

    def step(self, gradient, curvature=None, horizon=400, weight_decay=0.0):
        state, parameters = self.state, self.parameters
        state["step"] += 1
        step = state["step"]
        rate = self.rate
        diagnostics = {}
        if self.method in ("adamw", "adamw_cosine"):
            if self.method == "adamw_cosine":
                warm = min(1.0, step / self.warmup)
                progress = max(0.0, (step - self.warmup) / (horizon - self.warmup))
                rate *= warm * (1 + np.cos(np.pi * min(progress, 1.0))) / 2
            state["momentum"] = .9 * state["momentum"] + .1 * gradient
            state["second"] = .999 * state["second"] + .001 * gradient**2
            first = state["momentum"] / (1 - .9**step)
            second = state["second"] / (1 - .999**step)
            parameters *= 1 - rate * weight_decay
            parameters -= rate * first / (np.sqrt(second) + 1e-8)
        elif self.method == "lion":
            direction = np.sign(.9 * state["momentum"] + .1 * gradient)
            parameters *= 1 - rate * weight_decay
            parameters -= rate * direction
            state["momentum"] = .99 * state["momentum"] + .01 * gradient
        elif self.method == "sophia_g":
            if curvature is not None:
                state["curvature"] = .99 * state["curvature"] + .01 * curvature
            state["momentum"] = .965 * state["momentum"] + .035 * gradient
            ratio = state["momentum"] / np.maximum(.04 * state["curvature"], 1e-12)
            diagnostics["clipped_fraction"] = float(np.mean(np.abs(ratio) > 1))
            parameters *= 1 - rate * weight_decay
            parameters -= rate * np.clip(ratio, -1, 1)
        elif self.method == "prodigy":
            if weight_decay:
                raise ValueError("This teaching rule is paper Algorithm 4 without weight decay")
            distance = state["distance"]
            beta = np.sqrt(.999)
            weight = (1 - beta) * rate * distance**2
            state["numerator"] = beta * state["numerator"] + weight * float(
                np.sum(gradient * (state["initial"] - parameters)))
            state["displacement_sum"] = beta * state["displacement_sum"] + weight * gradient
            state["momentum"] = .9 * state["momentum"] + .1 * distance * gradient
            state["second"] = .999 * state["second"] + .001 * distance**2 * gradient**2
            denominator = float(np.abs(state["displacement_sum"]).sum())
            estimate = state["numerator"] / denominator if denominator > 0 else distance
            state["distance"] = max(distance, estimate)
            parameters -= rate * distance * state["momentum"] / (
                np.sqrt(state["second"]) + distance * 1e-8)
            diagnostics.update(distance_used=distance, distance_next=state["distance"])
        elif self.method == "schedule_free":
            rate *= min(1.0, step / self.warmup)
            state["second"] = .999 * state["second"] + .001 * gradient**2
            second = state["second"] / (1 - .999**step)
            state["fast"] -= rate * (gradient / (np.sqrt(second) + 1e-8)
                                     + weight_decay * parameters)
            state["weight_sum"] += rate**2
            coefficient = rate**2 / state["weight_sum"]
            state["average"] += coefficient * (state["fast"] - state["average"])
            parameters[:] = .9 * state["average"] + .1 * state["fast"]
            diagnostics["averaging_coefficient"] = coefficient
        else:
            raise ValueError(self.method)
        diagnostics["rate"] = rate
        return diagnostics


def probabilities(features, parameters):
    scores = features @ parameters
    scores -= scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def gradient(features, labels, parameters):
    residual = probabilities(features, parameters)
    residual[np.arange(len(labels)), labels] -= 1
    return features.T @ residual / len(labels)


def metrics(features, labels, parameters):
    scores = features @ parameters
    maximum = scores.max(axis=1)
    logsum = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
    return {"cross_entropy": float(np.mean(logsum - scores[np.arange(len(labels)), labels])),
            "correct": int(np.sum(scores.argmax(axis=1) == labels)), "count": len(labels)}
