"""Independent direct-loop, geometry and gradient calculations for the manuscript."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


def cross_correlation(image, weights, dilation=1, padding=0, stride=1, groups=1):
    """One C×H×W image, O×(C/groups)×kh×kw kernel; explicit zero padding."""
    channels, height, width = image.shape
    outputs, per_group, kh, kw = weights.shape
    if channels % groups or outputs % groups or per_group != channels // groups:
        raise ValueError("invalid group partition")
    out_h = (height + 2*padding - dilation*(kh-1) - 1)//stride + 1
    out_w = (width + 2*padding - dilation*(kw-1) - 1)//stride + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("kernel has no valid output")
    result = np.zeros((outputs, out_h, out_w), dtype=np.result_type(image, weights))
    for output in range(outputs):
        group = output // (outputs // groups)
        for row in range(out_h):
            for column in range(out_w):
                for local_channel in range(per_group):
                    channel = group * per_group + local_channel
                    for u in range(kh):
                        for v in range(kw):
                            i = row*stride - padding + dilation*u
                            j = column*stride - padding + dilation*v
                            if 0 <= i < height and 0 <= j < width:
                                result[output,row,column] += image[channel,i,j]*weights[output,local_channel,u,v]
    return result


def support(rates):
    sites = {0}
    for rate in rates:
        sites = {site + offset*rate for site in sites for offset in (-1,0,1)}
    return sorted(sites)


def sample(signal, position, dilation):
    return sum(signal[index] for index in (position-dilation, position, position+dilation)
               if 0 <= index < len(signal))


def saved_model_logits(state, image, dilation, factors=None):
    stem = cross_correlation(image[None], np.array(state["stem.weight"]), padding=1)
    stem = np.maximum(stem + np.array(state["stem.bias"])[:,None,None], 0)
    if factors is None:
        spatial = cross_correlation(stem, np.array(state["spatial.weight"]),
                                    dilation=dilation, padding=dilation)
        spatial += np.array(state["spatial.bias"])[:,None,None]
    else:
        filtered = cross_correlation(stem, np.array(factors["0.weight"]),
                                     dilation=dilation, padding=dilation, groups=8)
        spatial = cross_correlation(filtered, np.array(factors["1.weight"]))
        spatial += np.array(factors["1.bias"])[:,None,None]
    features = np.maximum(spatial, 0)
    pooled = features.reshape(12,4,2,4,2).mean(axis=(2,4)).reshape(-1)
    return np.array(state["head.weight"]) @ pooled + np.array(state["head.bias"])


def main():
    rng = np.random.default_rng(14)
    image = rng.normal(size=(4,5,6))
    comparisons = []
    for groups in (1,2,4):
        weights = rng.normal(size=(8,4//groups,3,3))
        for dilation, stride in ((1,1),(2,1),(2,2)):
            actual = cross_correlation(image, weights, dilation, dilation, stride, groups)
            reference = F.conv2d(torch.tensor(image)[None], torch.tensor(weights),
                                 padding=dilation, dilation=dilation, stride=stride, groups=groups)[0].numpy()
            comparisons.append({"groups":groups,"dilation":dilation,"stride":stride,
                                "shape":list(actual.shape),
                                "max_error":float(np.max(np.abs(actual-reference)))})
            assert np.allclose(actual, reference, atol=1e-12, rtol=1e-12)
    # Two input channels, two-tap valid spatial filters; first-position half squared loss.
    inputs = torch.tensor([[1.,2.,3.],[2.,0.,1.]], dtype=torch.float64)
    depthwise = torch.tensor([[1.,-1.],[.5,1.]], dtype=torch.float64, requires_grad=True)
    mixing = torch.tensor([2.,-1.], dtype=torch.float64, requires_grad=True)
    features = (inputs[:,:2] * depthwise).sum(1)
    output = features @ mixing
    loss = .5 * (output-1)**2
    loss.backward()
    with torch.no_grad():
        updated_d = depthwise-.01*depthwise.grad
        updated_p = mixing-.01*mixing.grad
        updated_output = (inputs[:,:2]*updated_d).sum(1) @ updated_p
        large_step_output = ((inputs[:,:2]*(depthwise-.1*depthwise.grad)).sum(1)
                             @ (mixing-.1*mixing.grad))
    # Rank restriction: each output's spatial filter for one channel is a multiple of one filter.
    exact_matrix = np.eye(2)
    rank_one = np.array([[1.,0.],[0.,0.]])
    patch = np.array([2.,3.])
    signal = np.arange(1.,10.)
    touched = signal.copy()
    touched[6] = 20
    untouched = signal.copy()
    untouched[5] = 20
    geometries = []
    for rates in ((2,2,2),(1,2,4),(1,4),(1,2,5),(1,2,9),(1,3,9)):
        sites = support(rates)
        bound = 1+2*sum(rates)
        geometries.append({"rates":rates,"sites":sites,"bound":bound,
                           "one_dimensional_count":len(sites),"two_dimensional_count":len(sites)**2,
                           "two_dimensional_box_count":bound**2})
    # Boundary counts for a 3x3 stencil on a real 8x8 feature-map geometry.
    boundary = []
    for dilation in (1,2,4,8):
        counts = [[sum(0 <= i+dilation*u < 8 and 0 <= j+dilation*v < 8
                       for u in (-1,0,1) for v in (-1,0,1))
                   for j in range(8)] for i in range(8)]
        boundary.append({"dilation":dilation,"valid_tap_counts":counts})
    ci, co, side, kernel = 64,128,14,3
    dense = kernel**2*ci*co
    separated = ci*(kernel**2+co)
    experiment = json.loads((HERE/"calculated-inputs.json").read_text())
    saved_checks = []
    for run in experiment["runs"]:
        if run["seed"] != 1:
            continue
        for example in run["examples"]:
            state = run["dense_state"]
            input_image = np.array(example["input"])
            dense_logits = saved_model_logits(state, input_image, run["dilation"])
            error = float(np.max(np.abs(dense_logits-np.array(example["dense_logits"]))))
            assert error < 2e-5
            factor_rows = []
            for item in run["factorizations"]:
                logits = saved_model_logits(state, input_image, run["dilation"],
                                             item["factorized_spatial_state"])
                expected = item["development"]["predictions"][example["development_index"]]
                assert int(logits.argmax()) == expected
                factor_rows.append({"multiplier":item["multiplier"],"logits":logits.tolist(),
                                    "predicted":int(logits.argmax())})
            edited = input_image.copy()
            edited[0,0] = 1
            edited_logits = saved_model_logits(state, edited, run["dilation"])
            saved_checks.append({
                "dilation":run["dilation"],"source_id":example["source_id"],
                "dense_loop_error":error,"factorized_outputs":factor_rows,
                "edit":{"row":0,"column":0,"old":float(input_image[0,0]),"new":1,
                        "dense_logits":edited_logits.tolist(),"predicted":int(edited_logits.argmax()),
                        "actual_class_logit_change":float(edited_logits[example["actual"]]-dense_logits[example["actual"]])}
            })
    def parallel_summary(values):
        branches = [float(values[4])] + [sample(values,4,rate) for rate in (1,2,4)] + [float(np.mean(values))]
        return {"branches":branches,"projected":float(np.dot(branches,[0,.1,.2,.3,1]))}
    distant_edit = signal.copy()
    distant_edit[8] = 20
    rearrangement = signal.copy()
    rearrangement[0], rearrangement[1] = rearrangement[1], rearrangement[0]
    result = {
        "loop_against_torch":comparisons,
        "saved_model_checks":saved_checks,
        "parallel":{"base":parallel_summary(signal),"index8_to20":parallel_summary(distant_edit),
                    "swap_indices0_1":parallel_summary(rearrangement)},
        "gradient":{"filtered":features.tolist(),"output":output.item(),"half_squared_loss":loss.item(),
                    "mixing_gradient":mixing.grad.tolist(),"depthwise_gradient":depthwise.grad.tolist(),
                    "updated_output":updated_output.item(),"updated_loss":float(.5*(updated_output-1)**2),
                    "large_step_rate":.1,"large_step_output":float(large_step_output),
                    "large_step_loss":float(.5*(large_step_output-1)**2)},
        "rank":{"dense_output":(exact_matrix@patch).tolist(),"rank_one_output":(rank_one@patch).tolist(),
                "null_patch":[2,0],"null_dense":(exact_matrix@np.array([2,0])).tolist(),
                "null_rank_one":(rank_one@np.array([2,0])).tolist()},
        "sampling":{"signal":signal.tolist(),"center_index":4,"dilation":2,
                    "base":sample(signal,4,2),"touched_index6":sample(touched,4,2),
                    "untouched_index5":sample(untouched,4,2),
                    "dilation1_base":sample(signal,4,1),"dilation1_edited_index5":sample(untouched,4,1),
                    "dilation8_base":sample(signal,4,8)},
        "geometry":geometries,"boundary":boundary,
        "cost":{"dense_weights":dense,"separable_weights":separated,
                "dense_macs":dense*side**2,"separable_macs":separated*side**2,
                "separable_over_dense":separated/dense,
                "single_output_separable_over_dense":(9+1)/9,
                "mbconv32_t6_weights":2*6*32**2+9*6*32,
                "mbconv32_t6_macs14":(2*6*32**2+9*6*32)*14**2,
                "mbconv32_t6_stride2_macs14to7":14**2*32*192+7**2*(9*192+192*32)},
        "practice":{"group96_to128_g32_weights":9*96*128//32,
                    "ci16_co24_m2_weights":16*2*(9+24),"dense16_24_weights":9*16*24,
                    "rf_r7_j2_k3_d3_s2":{"r":19,"jump":4},
                    "hard_swish_neg2":-2/3,"hard_swish_2":5/3,
                    "maximum_dense_bound_four_layers":3**4},
    }
    (HERE/"author-check-results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"max_loop_error":max(row["max_error"] for row in comparisons),
                      "saved_model_max_error":max(row["dense_loop_error"] for row in saved_checks),
                      "edits":[(r["source_id"],r["edit"]["actual_class_logit_change"]) for r in saved_checks],
                      "gradient":result["gradient"],"cost":result["cost"]},indent=2))


if __name__ == "__main__":
    main()
