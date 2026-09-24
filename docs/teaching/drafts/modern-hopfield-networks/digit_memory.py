"""Reproduce a modest handwritten-digit association experiment, entirely offline.

pip install numpy torch
python digit_memory.py
Uses the three neighboring original UCI data files; CPU, no pretrained downloads.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

ROOT = Path(__file__).parent


def unit_rows(values):
    return functional.normalize(values, dim=-1, eps=1e-12)


def load_roles():
    training = np.loadtxt(ROOT / "optdigits.tra", delimiter=",", dtype=np.int64)
    testing = np.loadtxt(ROOT / "optdigits.tes", delimiter=",", dtype=np.int64)
    # Audit exact features before making role assignments.
    seen = set()
    unique_training = []
    for index, row in enumerate(training):
        key = tuple(row[:-1])
        if key not in seen:
            unique_training.append(index)
            seen.add(key)
    test_indices = [index for index, row in enumerate(testing) if tuple(row[:-1]) not in seen]
    rng = np.random.default_rng(113)
    roles = {"memory": [], "fit_queries": [], "validation": []}
    for label in range(10):
        members = np.array([index for index in unique_training if training[index,-1] == label])
        members = rng.permutation(members)
        roles["memory"].extend(members[:20].tolist())
        roles["fit_queries"].extend(members[20:100].tolist())
        roles["validation"].extend(members[100:130].tolist())
    result = {}
    for name, indices in roles.items():
        result[name] = (torch.tensor(training[indices,:64]/16, dtype=torch.float32),
                        torch.tensor(training[indices,64], dtype=torch.int64))
    result["test"] = (torch.tensor(testing[test_indices,:64]/16, dtype=torch.float32),
                      torch.tensor(testing[test_indices,64], dtype=torch.int64))
    metadata = {"seed":113, "training_rows":len(training), "test_rows":len(testing),
                "unique_training_features":len(unique_training),
                "test_excluded_matching_training_features":len(testing)-len(test_indices),
                "roles_training_source_ids":{name:[i+1 for i in indices] for name,indices in roles.items()},
                "test_source_ids":[i+1 for i in test_indices],
                "hashes":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ["optdigits.tra","optdigits.tes","optdigits.names"]}}
    return result, metadata


def obscure(values):
    result = values.clone().reshape(-1,8,8)
    result[:,:,3:5] = 0
    return result.reshape(-1,64)


def read_memory(queries, memories, labels, beta, projection=None):
    query_keys = unit_rows(queries if projection is None else projection(queries))
    memory_keys = unit_rows(memories if projection is None else projection(memories))
    log_weights = functional.log_softmax(beta * query_keys @ memory_keys.T, dim=-1)
    # Class probabilities sum association mass, without exposing query labels to the read.
    log_class = torch.stack([torch.logsumexp(log_weights[:,labels == digit], dim=1)
                             for digit in range(10)], dim=1)
    return log_class, log_weights.exp()


def summarize(log_probabilities, labels):
    prediction = log_probabilities.argmax(dim=1)
    confusion = torch.zeros(10,10,dtype=torch.int64)
    for truth, guess in zip(labels, prediction):
        confusion[truth,guess] += 1
    return {"count":len(labels), "errors":int((prediction != labels).sum()),
            "cross_entropy":float(functional.nll_loss(log_probabilities,labels)),
            "confusion":confusion.tolist()}


def main():
    torch.set_num_threads(1)
    roles, metadata = load_roles()
    memory, memory_labels = roles["memory"]
    queries, query_labels = roles["fit_queries"]
    validation, validation_labels = roles["validation"]
    results = {"data": metadata, "fixed_candidates": [], "learned": []}
    arrays = {"memory_pixels":memory.numpy(), "memory_labels":memory_labels.numpy(),
              "validation_pixels":validation.numpy(), "validation_labels":validation_labels.numpy()}
    for beta in [4.,16.,64.,256.]:
        with torch.no_grad():
            report = {"beta":beta}
            for name in ["validation","test"]:
                values, labels = roles[name]
                for condition in ["clean","occluded"]:
                    inputs = values if condition == "clean" else obscure(values)
                    log_probabilities, weights = read_memory(inputs,memory,memory_labels,beta)
                    report[f"{name}_{condition}"] = summarize(log_probabilities,labels)
            results["fixed_candidates"].append(report)
    chosen = min(results["fixed_candidates"], key=lambda row:row["validation_clean"]["cross_entropy"])
    results["selected_fixed_beta"] = chosen["beta"]
    results["nearest_memory"] = {}
    with torch.no_grad():
        for name in ["validation","test"]:
            values, labels = roles[name]
            for condition in ["clean","occluded"]:
                inputs = values if condition == "clean" else obscure(values)
                nearest = (unit_rows(inputs)@unit_rows(memory).T).argmax(dim=1)
                results["nearest_memory"][f"{name}_{condition}"] = {
                    "count":len(labels),"errors":int((memory_labels[nearest] != labels).sum())}
    for seed in [17,41]:
        torch.manual_seed(seed)
        projection = nn.Linear(64,16,bias=False)
        optimizer = torch.optim.Adam(projection.parameters(),lr=.005)
        best = float("inf")
        training_curve = []
        for epoch in range(1,101):
            optimizer.zero_grad()
            log_probabilities, _ = read_memory(queries,memory,memory_labels,16.,projection)
            loss = functional.nll_loss(log_probabilities,query_labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_log, _ = read_memory(validation,memory,memory_labels,16.,projection)
                val_loss = float(functional.nll_loss(val_log,validation_labels))
            training_curve.append({"epoch":epoch,"fit_loss_before_update":float(loss.detach()),
                                   "validation_loss_after_update":val_loss})
            if val_loss < best:
                best = val_loss
                selected_epoch = epoch
                selected_state = copy.deepcopy(projection.state_dict())
        projection.load_state_dict(selected_state)
        report = {"seed":seed,"beta":16.,"parameters":1024,"selected_epoch":selected_epoch,
                  "training_curve":training_curve}
        with torch.no_grad():
            for name in ["fit_queries","validation","test"]:
                values, labels = roles[name]
                for condition in ["clean","occluded"]:
                    inputs = values if condition == "clean" else obscure(values)
                    log_probabilities, weights = read_memory(inputs,memory,memory_labels,16.,projection)
                    entry = summarize(log_probabilities,labels)
                    reconstructed = weights @ memory
                    entry["reconstruction_mse_to_clean"] = float(((reconstructed-values)**2).mean())
                    entry["input_mse_to_clean"] = float(((inputs-values)**2).mean())
                    report[f"{name}_{condition}"] = entry
                    if name == "validation":
                        arrays[f"seed{seed}_{condition}_log_class"] = log_probabilities.numpy()
                        arrays[f"seed{seed}_{condition}_weights"] = weights.numpy()
            arrays[f"seed{seed}_projection"] = projection.weight.detach().numpy()
            results["learned"].append(report)
    # Select the presentation seed in advance (17), not by its test score.
    with torch.no_grad():
        for condition in ["clean","occluded"]:
            inputs = validation if condition == "clean" else obscure(validation)
            log_class, weights = read_memory(inputs,memory,memory_labels,chosen["beta"])
            arrays[f"fixed_{condition}_log_class"] = log_class.numpy()
            arrays[f"fixed_{condition}_weights"] = weights.numpy()
    np.savez_compressed(ROOT/"digit-memory-fits.npz",**arrays)
    (ROOT/"digit-results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({"selected_beta":chosen["beta"],"data":{k:v for k,v in metadata.items() if "ids" not in k},
       "fixed":[{"beta":r["beta"],**{k:v["errors"] for k,v in r.items() if isinstance(v,dict)}} for r in results["fixed_candidates"]],
       "nearest":results["nearest_memory"],
       "learned":[{"seed":r["seed"],"epoch":r["selected_epoch"],**{k:v["errors"] for k,v in r.items() if isinstance(v,dict)}} for r in results["learned"]]},indent=2))


if __name__ == "__main__":
    main()
