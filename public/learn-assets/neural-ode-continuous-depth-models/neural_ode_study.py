"""Declared real Iris study: learned depth is not botanical time."""
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch
from torch import nn

DIRECTORY = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def load_data():
    with (DIRECTORY / "iris.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    names = ["setosa", "versicolor", "virginica"]
    features = np.array([[float(row[key]) for key in columns] for row in records])
    labels = np.array([names.index(row["species"]) for row in records])
    groups = {}
    for index, row in enumerate(features):
        groups.setdefault(tuple(row), []).append(index)
    for members in groups.values():
        assert len(set(labels[members])) == 1
    kept = np.array(sorted(members[0] for members in groups.values()))
    roles = {key: [] for key in ("fit", "validation", "assessment")}
    generator = np.random.default_rng(926)
    for label in range(3):
        indices = generator.permutation(kept[labels[kept] == label])
        for key, subset in zip(roles, (indices[:30], indices[30:40], indices[40:])):
            roles[key].extend(subset.tolist())
    mean, scale = features[roles["fit"]].mean(0), features[roles["fit"]].std(0)
    metadata = dict(columns=columns, classes=names, mean=mean.tolist(), scale=scale.tolist(),
        roles={key: [index+1 for index in indices] for key, indices in roles.items()},
        duplicate_groups=[[i+1 for i in group] for group in groups.values() if len(group)>1],
        unique_count=len(kept))
    return torch.tensor((features-mean)/scale), torch.tensor(labels), roles, metadata


class VectorField(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.input = nn.Linear(dimension+1, 16)
        self.output = nn.Linear(16, dimension)

    def forward(self, time, state):
        clock = torch.full_like(state[:, :1], float(time))
        return self.output(torch.tanh(self.input(torch.cat((state, clock), -1))))


def integrate(field, initial, steps=4, method="rk4", endpoint=1.):
    state = initial
    trace = [state]
    step_size = endpoint/steps
    for step in range(steps):
        time = step*step_size
        first = field(time, state)
        if method == "euler":
            state = state + step_size*first
        elif method == "rk4":
            second = field(time+step_size/2, state+step_size*first/2)
            third = field(time+step_size/2, state+step_size*second/2)
            fourth = field(time+step_size, state+step_size*third)
            state = state + step_size*(first+2*second+2*third+fourth)/6
        else:
            raise ValueError(method)
        trace.append(state)
    return torch.stack(trace)


class DepthClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.dimension = 6 if kind == "augmented_ode" else 4
        if kind in ("neural_ode", "augmented_ode"):
            self.field = VectorField(self.dimension)
        if kind == "residual":
            self.blocks = nn.ModuleList([VectorField(4) for _ in range(4)])
        self.readout = nn.Linear(self.dimension, 3)

    def forward(self, inputs, steps=4, method="rk4", endpoint=1., return_trace=False):
        state = torch.cat((inputs, inputs.new_zeros((len(inputs), self.dimension-4))), -1)
        if self.kind in ("neural_ode", "augmented_ode"):
            trace = integrate(self.field, state, steps, method, endpoint)
            state = trace[-1]
        else:
            states = [state]
            if self.kind == "residual":
                for index, block in enumerate(self.blocks):
                    state = state + .25*block(index/4, state)
                    states.append(state)
            trace = torch.stack(states)
        logits = self.readout(state)
        return (logits, trace) if return_trace else logits


def evaluate(model, features, labels):
    with torch.no_grad():
        logits = model(features)
    return dict(cross_entropy=float(nn.functional.cross_entropy(logits, labels)),
                correct=int((logits.argmax(-1)==labels).sum()), count=len(labels))


def main():
    features, labels, roles, metadata = load_data()
    reports, snapshots = [], []
    for kind in ["linear", "residual", "neural_ode", "augmented_ode"]:
        for seed in [13, 37, 61]:
            torch.manual_seed(seed)
            model = DepthClassifier(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
            best_loss, selected, curves = float("inf"), None, []
            for step in range(1, 301):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(features[roles["fit"]]), labels[roles["fit"]])
                loss.backward()
                optimizer.step()
                if step==1 or step%25==0:
                    fit = evaluate(model, features[roles["fit"]], labels[roles["fit"]])
                    validation = evaluate(model, features[roles["validation"]], labels[roles["validation"]])
                    curves.append(dict(step=step, fit=fit, validation=validation))
                    if validation["cross_entropy"] < best_loss:
                        best_loss = validation["cross_entropy"]
                        selected = deepcopy(model.state_dict())
                        selected_step = step
            model.load_state_dict(selected)
            report = dict(kind=kind, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                selected_step=selected_step, curves=curves,
                fit=evaluate(model,features[roles["fit"]],labels[roles["fit"]]),
                validation=evaluate(model,features[roles["validation"]],labels[roles["validation"]]),
                assessment=evaluate(model,features[roles["assessment"]],labels[roles["assessment"]]))
            reports.append(report)
            snapshots.append(dict(kind=kind,seed=seed,selected_step=selected_step,
                state={key:value.tolist() for key,value in selected.items()}))
            print(kind,seed,selected_step,report["assessment"],flush=True)
    result=dict(data=metadata, updates=300, learning_rate=.01,weight_decay=.001,
        seeds=[13,37,61], dtype="float64", torch=torch.__version__, numpy=np.__version__,
        dataset_sha256=hashlib.sha256((DIRECTORY/"iris.csv").read_bytes()).hexdigest(),runs=reports)
    (DIRECTORY/"study-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (DIRECTORY/"fitted-models.json").write_text(json.dumps(snapshots,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
