"""Bounded author arithmetic evidence, not website/runtime verification."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import runpy
from fractions import Fraction
import numpy as np
import torch
from torch.nn import functional as F
import train_iris

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def gradient(x, y, weights=None, included=None):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    weights = np.ones(len(x)) if weights is None else np.array(weights, dtype=float)
    included = np.ones(len(x)) if included is None else np.array(included, dtype=float)
    masses = weights * included
    return float(np.dot(masses, -x * y)), float(masses.sum())


def scalar_paths(x, y, groups, lr=0.1, momentum=0.0):
    x, y = np.array(x), np.array(y)
    parts = [-float(np.dot(x[g], y[g])) / len(x) for g in groups]
    early_w = early_momentum = 0.0
    for g in groups:
        early_momentum = momentum * early_momentum + float(np.dot(x[g], early_w * x[g] - y[g])) / len(x)
        early_w -= lr * early_momentum
    return {"correct": -lr * sum(parts), "clear_each": -lr * parts[-1], "step_each": early_w, "contributions": parts}


def normalization_case(x, y, groups):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    eps = 1e-5
    mean, variance = float(x.mean()), float(x.var())
    def result(z):
        return {"z": z.tolist(), "loss": float(np.mean(0.5 * (z-y)**2)), "gradient": float(np.mean(z*(z-y)))}
    full_z = (x - mean) / np.sqrt(variance + eps)
    local_z = np.zeros_like(x)
    running_mean = 0.0
    for group in groups:
        values = x[group]
        local_z[group] = (values - values.mean()) / np.sqrt(values.var() + eps)
        running_mean = 0.9 * running_mean + 0.1 * float(values.mean())
    full = result(full_z)
    split = result(local_z)
    frozen = result((x - mean)/np.sqrt(variance+eps))
    batch_norm = torch.nn.BatchNorm1d(1, affine=False, eps=eps, momentum=0.1)
    torch_z = batch_norm(torch.tensor(x).reshape(-1, 1)).flatten().detach().numpy()
    assert np.allclose(full_z, torch_z, atol=1e-12)
    local_bn = torch.nn.BatchNorm1d(1, affine=False, eps=eps, momentum=0.1)
    for g in groups:
        observed = local_bn(torch.tensor(x[g]).reshape(-1, 1)).flatten().detach().numpy()
        assert np.allclose(local_z[g], observed, atol=1e-12)
    assert abs(float(local_bn.running_mean[0]) - running_mean) < 1e-12
    return {"full": full, "local": split, "frozen_full_statistics": frozen, "no_normalization": result(x), "full_running_mean": 0.1*mean, "split_running_mean": running_mean}


def main():
    results = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "device": "CPU", "threads": 1, "dtype": "float64"}}
    results["worked_trace"] = scalar_paths([1,2,3], [2,0,1], [[0,1],[2]])
    assert abs(results["worked_trace"]["correct"] - 1/6) < 1e-12
    results["state_lab_fresh"] = scalar_paths([1,2,4], [1,2,1], [[0,1],[2]])
    results["state_lab_fresh_momentum"] = scalar_paths([1,2,4], [1,2,1], [[0,1],[2]], momentum=0.9)
    results["state_lab_single_group_null"] = scalar_paths([1,2,4], [1,2,1], [[0,1,2]])
    results["state_lab_single_group_momentum_null"] = scalar_paths([1,2,4], [1,2,1], [[0,1,2]], momentum=0.9)
    results["state_lab_zero_rate_null"] = scalar_paths([1,2,4], [1,2,1], [[0,1],[2]], lr=0, momentum=0.9)
    assert len(set(round(results["state_lab_single_group_null"][k],12) for k in ["correct","clear_each","step_each"])) == 1
    numerator, denominator = gradient([1,2,3,1], [1,1,2,4], [1,1,3,0], [1,1,1,0])
    results["denominator_lab_fresh"] = {"numerator": numerator, "denominator": denominator, "weighted_mean": numerator/denominator, "naive_microbatch_mean": (-1.5-6)/2, "eligible_count_divisor": numerator/3, "all_slots_divisor": numerator/4}
    # Equal denominator null: first partition contributes -3 / 2; second -12 / 2.
    results["denominator_equal_mass_null"] = {"weighted_mean": -15/4, "naive_microbatch_mean": (-1.5-6)/2}
    assert -15/4 == (-1.5-6)/2
    results["denominator_zero_mass"] = dict(zip(["numerator","mass"], gradient([1,2], [1,1], [1,3], [0,0])))
    assert results["denominator_zero_mass"] == {"numerator":0.0,"mass":0.0}
    logits = torch.tensor([[0.,0.],[1.,0.],[0.,2.],[3.,-1.]], requires_grad=True)
    targets = torch.tensor([0,1,1,-100])
    weights = torch.tensor([1.,3.])
    full_loss = F.cross_entropy(logits, targets, weight=weights, ignore_index=-100)
    full_grad, = torch.autograd.grad(full_loss, logits)
    accumulated = 0.0
    for g in [[0,1],[2,3]]:
        accumulated = accumulated + F.cross_entropy(logits[g], targets[g], weight=weights, ignore_index=-100, reduction="sum") / 7
    accumulated_grad, = torch.autograd.grad(accumulated, logits)
    assert torch.allclose(full_grad, accumulated_grad, atol=1e-12)
    probability_targets = torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
    probability_sum = F.cross_entropy(logits, probability_targets, weight=weights, reduction="sum")
    probability_mean = F.cross_entropy(logits, probability_targets, weight=weights, reduction="mean")
    assert torch.allclose(probability_mean, probability_sum/4, atol=1e-12)
    results["cross_entropy"] = {"class_index_denominator":7, "class_index_weighted_loss":float(full_loss.detach()), "gradient_gap":float((full_grad-accumulated_grad).abs().max()), "probability_target_denominator":4, "probability_target_loss":float(probability_mean.detach()), "all_ignored_sum":float(F.cross_entropy(logits, torch.full((4,),-100), reduction="sum").detach())}
    results["normalization_contrast"] = normalization_case([0,2,10,12], [0,0,1,1], [[0,1],[2,3]])
    results["normalization_lab_fresh"] = normalization_case([1,3,7,9], [0,1,0,2], [[0,1],[2,3]])
    results["normalization_equal_stats_null"] = normalization_case([0,2,0,2], [0,1,0,1], [[0,1],[2,3]])
    results["normalization_constant_null"] = normalization_case([1,1,1,1], [0,0,1,1], [[0,1],[2,3]])
    results["normalization_one_group_null"] = normalization_case([1,3,7,9], [0,1,0,2], [[0,1,2,3]])
    results["normalization_noncontiguous"] = normalization_case([1,3,7,9], [0,1,0,2], [[0,2],[1,3]])
    contrast = results["normalization_contrast"]
    assert abs(contrast["full"]["gradient"] - contrast["local"]["gradient"]) > 0.4
    assert abs(contrast["full"]["gradient"] - contrast["frozen_full_statistics"]["gradient"]) < 1e-12
    equal = results["normalization_equal_stats_null"]
    assert abs(equal["full"]["gradient"] - equal["local"]["gradient"]) < 1e-12
    # Fixed dropout masks, rather than library RNG call shape, isolate the mechanism.
    x, y = np.array([1.,2.,3.,4.]), np.array([1.,0.,1.,0.])
    def dropout_gradient(mask):
        z = x*np.array(mask)*2
        return float(np.mean(z*(z-y)))
    results["dropout"] = {"aligned_mask_gradient":dropout_gradient([1,0,1,0]), "different_mask_gradient":dropout_gradient([0,1,0,1]), "disabled_gradient":float(np.mean(x*(x-y)))}
    results["clipping"] = {"contributions":[3.,-2.5], "clip_sum_at_1":float(np.clip(3-2.5,-1,1)), "sum_clipped_at_1":float(np.clip(3,-1,1)+np.clip(-2.5,-1,1))}
    results["partial_groups"] = {"N":10, "microbatch_size":4, "K":2, "microbatch_sizes":[4,4,2], "group_denominators":[8,2], "updates":2, "final_wrong_fixed_K_factor":0.5}
    results["ddp_arithmetic"] = {"rank_mean_gradients":[-1.,-3.], "unweighted_rank_average":-2., "global_item_mean":-5/3, "rank_scaled_numerator_gradients":[-4/3,-2.], "scaled_rank_average":(-4/3-2)/2}
    assert abs(results["ddp_arithmetic"]["scaled_rank_average"]+5/3)<1e-12
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        results["iris"] = train_iris.main()
    (HERE / "iris-output.txt").write_text(output.getvalue(), encoding="utf-8")
    assert results["iris"]["parameter_gap"] < 1e-12
    assert results["iris"]["momentum_gap"] < 1e-12
    full, full_optimizer, _, _ = train_iris.train(32)
    changed, changed_optimizer, _, calls = train_iris.train(7)
    results["iris_microbatch_7_transfer"] = {
        "calls": calls,
        "parameter_gap": max(float((a.detach()-b.detach()).abs().max()) for a,b in zip(full.parameters(),changed.parameters())),
        "momentum_gap": max(float((full_optimizer.state[a]["momentum_buffer"]-changed_optimizer.state[b]["momentum_buffer"]).abs().max()) for a,b in zip(full.parameters(),changed.parameters()))}
    assert calls == 380
    assert results["iris_microbatch_7_transfer"]["parameter_gap"] < 1e-12
    assert results["iris_microbatch_7_transfer"]["momentum_gap"] < 1e-12
    post_loss = sum(Fraction(1,2)*(Fraction(x,6)-y)**2 for x,y in zip([1,2,3],[2,0,1]))/3
    assert post_loss == Fraction(67,108)
    results["manual_arithmetic"] = {"post_update_loss":str(post_loss), "momentum_second_state":str(Fraction(9,10)*Fraction(-5,3)+1), "momentum_second_weight":str(Fraction(1,6)-Fraction(1,10)*Fraction(-1,2)), "practice_2_gradient":str(Fraction(-10,3)), "practice_2_weight":str(Fraction(1,3)), "practice_3_token_mean":str(Fraction(11,4)), "practice_3_sequence_mean":str((Fraction(6,3)+5)/2)}
    one = [Fraction(-2),Fraction(0),Fraction(-3)]
    two = [(one[0]+one[1])/2,(one[0]+one[2])/2,(one[1]+one[2])/2]
    mean = sum(one)/3
    results["gradient_sampling"] = {"mean":str(mean), "one_row_variance":str(sum((g-mean)**2 for g in one)/3), "two_row_without_replacement_means":[str(g) for g in two], "two_row_variance":str(sum((g-mean)**2 for g in two)/3)}
    assert results["gradient_sampling"]["one_row_variance"] == "14/9"
    assert results["gradient_sampling"]["two_row_variance"] == "7/18"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runpy.run_path(str(HERE / "trace_update.py"), run_name="__main__")
    (HERE / "trace-output.txt").write_text(output.getvalue(), encoding="utf-8")
    results["input_sha256"] = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ["iris.csv","train_iris.py","trace_update.py","author_checks.py"]}
    (HERE / "author-results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print("Author calculations passed; evidence written in packet.")


if __name__ == "__main__":
    main()
