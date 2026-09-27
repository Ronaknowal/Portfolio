"""Exact constructed attention fixtures; no fitted benchmark is generated here."""
from pathlib import Path
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
def softmax(scores):
    shifted = np.asarray(scores, dtype=float)-np.max(scores)
    return np.exp(shifted)/np.exp(shifted).sum()

KEYS = np.array([[1., 0.], [0., 1.], [-1., 0.]])
VALUES = np.array([[2., 0.], [0., 2.], [-1., 1.]])

def calculate(query, keys=KEYS, values=VALUES, valid=None, rate=.1):
    query = np.asarray(query, dtype=float)
    valid = np.ones(len(keys), dtype=bool) if valid is None else np.array(valid, dtype=bool)
    if not valid.any():
        raise ValueError("No valid memory position.")
    scores = keys@query
    masked = np.where(valid, scores, -np.inf)
    attention = softmax(masked)
    context = attention@values
    probabilities = softmax(context)
    loss = -np.log(probabilities[1])
    context_gradient = probabilities-np.array([0., 1.])
    score_gradient = attention*((values-context)@context_gradient)
    query_gradient = keys.T@score_gradient
    next_query = query-rate*query_gradient
    next_attention = softmax(np.where(valid, keys@next_query, -np.inf))
    next_loss = -np.log(softmax(next_attention@values)[1])
    differentiable_query = torch.tensor(query, requires_grad=True)
    torch_attention = (torch.tensor(keys)@differentiable_query).masked_fill(~torch.tensor(valid), -torch.inf).softmax(-1)
    native_loss = torch.nn.functional.cross_entropy((torch_attention@torch.tensor(values))[None], torch.tensor([1]))
    native_loss.backward()
    assert np.allclose(differentiable_query.grad.numpy(), query_gradient, atol=1e-12)
    finite = []
    for index in range(len(query)):
        offset = np.zeros_like(query); offset[index] = 1e-5
        def loss_at(q):
            c = softmax(np.where(valid, keys@q, -np.inf))@values
            return -np.log(softmax(c)[1])
        finite.append((loss_at(query+offset)-loss_at(query-offset))/(2e-5))
    assert np.allclose(finite, query_gradient, atol=1e-9)
    return {"query": query.tolist(), "keys": keys.tolist(), "values": values.tolist(), "valid": valid.tolist(),
            "scores": scores.tolist(), "attention": attention.tolist(), "context": context.tolist(),
            "class_probabilities": probabilities.tolist(), "loss": float(loss),
            "context_gradient": context_gradient.tolist(), "score_gradient": score_gradient.tolist(),
            "query_gradient": query_gradient.tolist(), "finite_difference": finite,
            "rate": rate, "next_query": next_query.tolist(), "next_loss": float(next_loss)}

def local(center, radius, renormalize=False):
    positions = np.arange(1, 6)
    scores = np.array([0., .5, 1., -.5, 2.])
    valid = abs(positions-center) <= radius
    base = softmax(np.where(valid, scores, -np.inf))
    weights = base*np.exp(-((positions-center)**2)/(2*(radius/2)**2))
    if renormalize:
        weights /= weights.sum()
    return {"center": center, "radius": radius, "positions": positions.tolist(), "scores": scores.tolist(),
            "valid": valid.tolist(), "base": base.tolist(), "weights": weights.tolist(),
            "sum": float(weights.sum()), "context": float(weights@positions), "renormalized": renormalize}

def main():
    torch.set_num_threads(1)
    changed_values = VALUES.copy(); changed_values[1] = [1., 3.]
    contrast_values = VALUES.copy(); contrast_values[1] = [1., 2.]
    changed_keys = KEYS.copy(); changed_keys[1] = [2., 1.]
    same_values = np.tile([.5, .5], (3, 1))
    fixtures = {"worked": calculate([1., 0.]), "fresh": calculate([.3, -.4]),
        "fresh_key_edit": calculate([.3, -.4], keys=changed_keys),
        "fresh_value_edit": calculate([.3, -.4], values=changed_values),
        "fresh_value_contrast": calculate([.3, -.4], values=contrast_values),
        "fresh_mask_edit": calculate([.3, -.4], valid=[1, 0, 1]),
        "fresh_zero_rate": calculate([.3, -.4], rate=0.),
        "equal_values": calculate([.3, -.4], values=same_values),
        "equal_values_query_changed": calculate([1.2, .5], values=same_values)}
    assert fixtures["fresh"]["attention"] == fixtures["fresh_value_edit"]["attention"]
    assert np.allclose(fixtures["equal_values"]["query_gradient"], 0.)
    padded_keys = np.vstack([KEYS, [1e4, -1e4]])
    padded_values = np.vstack([VALUES, [1e3, 1e3]])
    padded = calculate([.3, -.4], keys=padded_keys, values=padded_values, valid=[1, 1, 1, 0])
    assert np.allclose(padded["context"], fixtures["fresh"]["context"])
    fixtures["masked_extra_null"] = padded
    q1, q2, scalar_keys = .2, 1.1, np.array([-.7, .1, 1.3])
    linear1 = 2*q1+scalar_keys
    linear2 = 2*q2+scalar_keys
    nonlinear1 = np.tanh(2*q1+scalar_keys)
    nonlinear2 = np.tanh(2*q2+scalar_keys)
    cancellation = {"keys": scalar_keys.tolist(), "queries": [q1, q2],
        "linear_attention": [softmax(linear1).tolist(), softmax(linear2).tolist()],
        "nonlinear_attention": [softmax(nonlinear1).tolist(), softmax(nonlinear2).tolist()],
        "constant_shift": softmax(linear1+37).tolist()}
    assert np.allclose(softmax(linear1), softmax(linear2))
    assert not np.allclose(softmax(nonlinear1), softmax(nonlinear2))
    alternative_values = np.array([[1., 0.], [0., 1.], [.5, .5]])
    ambiguity = {"values": alternative_values.tolist(), "weights_a": [.4, .4, .2], "weights_b": [.2, .2, .6],
                 "context_a": (np.array([.4, .4, .2])@alternative_values).tolist(),
                 "context_b": (np.array([.2, .2, .6])@alternative_values).tolist()}
    # Fresh independent values and two distributions retain a deliberate exact affine dependence.
    fresh_values = np.array([[2., 1.], [0., 3.], [1., 2.]])
    ambiguity["fresh"] = {"values": fresh_values.tolist(), "weights_a": [.3, .3, .4], "weights_b": [.1, .1, .8],
                         "contexts": [(np.array(a)@fresh_values).tolist() for a in ([.3,.3,.4], [.1,.1,.8])]}
    pointer = {"source": ["Ada", "met", "Ada"], "attention": [.2, .3, .5], "p_generate": .4,
               "vocabulary": {"Ada": .1, "met": .6, "left": .3},
               "output": {"Ada": .46, "met": .42, "left": .12},
               "fresh": {"source": ["red", "blue", "red"], "attention": [.15, .25, .6],
                         "p_generate": .2, "vocabulary": {"blue": .5, "green": .5},
                         "output": {"red": .6, "blue": .3, "green": .1}}}
    report = {"fixtures": fixtures, "cancellation": cancellation, "nonidentifiability": ambiguity,
              "local_worked": local(3, 2), "local_worked_renormalized": local(3, 2, True),
              "local_fresh": local(2.5, 1.5), "local_fresh_shift": local(3.5, 1.5),
              "local_fresh_renormalized": local(2.5, 1.5, True),
              "pointer": pointer, "score_parameters_no_bias": {"query": 96, "memory": 192, "additive_hidden": 64,
                  "additive": 96*64+192*64+64, "general": 96*192, "difference": 64}}
    (ROOT/"analytic-results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"worked": fixtures["worked"], "fresh": fixtures["fresh"],
                      "local": report["local_fresh"]}, indent=2))

if __name__ == "__main__":
    main()
