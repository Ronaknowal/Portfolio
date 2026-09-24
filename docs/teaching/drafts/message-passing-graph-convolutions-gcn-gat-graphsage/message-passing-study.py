"""Complete CPU study and exact fixtures; reads the adjacent real graph offline."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def graph_operators(adjacency):
    closed = adjacency + torch.eye(len(adjacency), dtype=adjacency.dtype)
    degree = closed.sum(-1)
    symmetric = degree.rsqrt()[:, None] * closed * degree.rsqrt()[None, :]
    mean = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
    return symmetric, mean, closed.bool()


class GraphLayer(nn.Module):
    def __init__(self, incoming, outgoing, kind):
        super().__init__()
        self.kind = kind
        self.linear = nn.Linear(incoming, outgoing, bias=False)
        self.bias = nn.Parameter(torch.zeros(outgoing))
        if kind == "sage":
            self.self_linear = nn.Linear(incoming, outgoing, bias=False)
        if kind == "gat":
            self.receiver_score = nn.Parameter(torch.randn(outgoing) * .1)
            self.sender_score = nn.Parameter(torch.randn(outgoing) * .1)

    def forward(self, features, operators):
        symmetric, mean, allowed = operators
        transformed = self.linear(features)
        weights = None
        if self.kind == "gcn":
            output = symmetric @ transformed
        elif self.kind == "sage":
            output = mean @ transformed + self.self_linear(features)
        elif self.kind == "gat":
            scores = transformed @ self.receiver_score
            scores = scores[:, None] + (transformed @ self.sender_score)[None, :]
            scores = nn.functional.leaky_relu(scores, .2).masked_fill(~allowed, -torch.inf)
            weights = scores.softmax(-1)
            output = weights @ transformed
        else:
            output = transformed
        return output + self.bias, weights


class NodeClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.first = GraphLayer(3, 16, kind)
        self.last = GraphLayer(16, 2, kind)

    def forward(self, features, operators):
        hidden, weights = self.first(features, operators)
        hidden = hidden.relu()
        logits, _ = self.last(hidden, operators)
        return logits, hidden, weights


def exact_cases():
    # A three-node unequal-degree path: 0--1--2, with self-loops only in GCN.
    adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
    symmetric, mean, allowed = graph_operators(adjacency)
    features = torch.tensor([[1.], [2.], [4.]], dtype=torch.float64)
    direct = torch.zeros_like(features)
    degrees = (adjacency + torch.eye(3)).sum(-1)
    for receiver in range(3):
        for sender in range(3):
            if allowed[receiver, sender]:
                direct[receiver] += features[sender] / torch.sqrt(degrees[receiver] * degrees[sender])
    assert torch.allclose(direct, symmetric @ features, atol=1e-12)
    p = torch.tensor([2, 0, 1])
    permuted, _, _ = graph_operators(adjacency[p][:, p])
    assert torch.allclose(permuted @ features[p], (symmetric @ features)[p])
    direction = degrees.sqrt(); direction /= direction.norm()
    limit = direction[:, None] * (direction @ features)[None, :]
    propagated = features.clone(); steps = []
    for step in range(41):
        if step in (0, 1, 2, 4, 10, 40):
            steps.append({"step": step, "raw": propagated.flatten().tolist(), "degree_divided": (propagated[:, 0] / degrees.sqrt()).tolist()})
        propagated = symmetric @ propagated
    assert torch.allclose(propagated, limit, atol=1e-10)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    # Bias before propagation is not interchangeable with bias after it.
    before = symmetric @ (features + 1)
    after = symmetric @ features + 1
    # One scalar shared weight and one observed target on node 0.
    weight = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    prediction = (symmetric @ features)[0, 0] * weight
    loss = .5 * (prediction - 1).square()
    loss.backward()
    updated = .5 - .1 * weight.grad.item()
    return {"adjacency": adjacency.tolist(), "features": features.tolist(), "symmetric": symmetric.tolist(), "row_sums": symmetric.sum(-1).tolist(), "first_step": direct.flatten().tolist(), "second_step": (symmetric @ direct).flatten().tolist(), "eigenvalues": eigenvalues.tolist(), "propagation": steps, "limit": limit.flatten().tolist(), "bias_before": before.flatten().tolist(), "bias_after": after.flatten().tolist(), "single_update": {"weight": .5, "prediction": prediction.item(), "loss": loss.item(), "gradient": weight.grad.item(), "new_weight": updated, "new_prediction": (symmetric @ features)[0, 0].item() * updated}, "max_loop_matrix_error": (direct - symmetric @ features).abs().max().item()}


def main():
    data = json.loads((HERE / "karate-club.json").read_text(encoding="utf-8"))
    count = len(data["nodes"])
    adjacency = torch.zeros(count, count)
    for edge in data["edges"]:
        adjacency[edge[0], edge[1]] = adjacency[edge[1], edge[0]] = 1
    features = torch.tensor([[1., node["degree"] / 33., node["clustering"]] for node in data["nodes"]])
    targets = torch.tensor([node["club"] for node in data["nodes"]])
    operators = graph_operators(adjacency)
    rng = np.random.default_rng(133)
    roles = {"fit": [], "development": [], "assessment": []}
    for label in (0, 1):
        order = rng.permutation(np.flatnonzero(targets.numpy() == label)).tolist()
        roles["fit"].extend(order[:5]); roles["development"].extend(order[5:8]); roles["assessment"].extend(order[8:])
    roles = {role: sorted(ids) for role, ids in roles.items()}
    # A non-neural graph baseline: repeatedly average known-label scores and clamp fit labels.
    label_scores = torch.zeros(count, 2)
    fit = roles["fit"]
    label_scores[fit] = nn.functional.one_hot(targets[fit], 2).float()
    transition = (adjacency + torch.eye(count)) / (adjacency.sum(-1, keepdim=True) + 1)
    for _ in range(200):
        label_scores = transition @ label_scores
        label_scores[fit] = nn.functional.one_hot(targets[fit], 2).float()
    baseline = {role: int((label_scores[ids].argmax(-1) == targets[ids]).sum()) for role, ids in roles.items()}
    measured = []
    for kind in ("mlp", "gcn", "sage", "gat"):
        for seed in (11, 29, 47):
            torch.manual_seed(seed)
            model = NodeClassifier(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.01)
            history = []
            for epoch in range(300):
                logits, _, _ = model(features, operators)
                loss = nn.functional.cross_entropy(logits[fit], targets[fit])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if epoch in (0, 9, 49, 99, 299):
                    with torch.no_grad():
                        output, _, _ = model(features, operators)
                        history.append({"epoch": epoch + 1, "fit_loss": nn.functional.cross_entropy(output[fit], targets[fit]).item(), "development_correct": int((output[roles["development"]].argmax(-1) == targets[roles["development"]]).sum())})
            with torch.no_grad():
                logits, hidden, weights = model(features, operators)
                record = {"model": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "history": history, "correct": {role: int((logits[ids].argmax(-1) == targets[ids]).sum()) for role, ids in roles.items()}, "probabilities": logits.softmax(-1).tolist()}
                removed, _, _ = model(features, graph_operators(torch.zeros_like(adjacency)))
                record["propagation_removed_assessment_correct"] = int((removed[roles["assessment"]].argmax(-1) == targets[roles["assessment"]]).sum())
                permutation = torch.tensor(np.random.default_rng(91).permutation(count))
                permuted_output, _, _ = model(features[permutation], graph_operators(adjacency[permutation][:, permutation]))
                record["permutation_max_error"] = (permuted_output - logits[permutation]).abs().max().item()
                assert record["permutation_max_error"] < 1e-4
                if seed == 11:
                    record["hidden"] = hidden.tolist()
                    record["state_dict"] = {key: tensor.tolist() for key, tensor in model.state_dict().items()}
                    if weights is not None:
                        record["attention"] = weights.tolist()
                measured.append(record)
    result = {"versions": {"torch": torch.__version__, "numpy": np.__version__}, "roles": roles, "graph": {"nodes": count, "undirected_edges": len(data["edges"]), "same_label_edges": sum(int(targets[a] == targets[b]) for a, b, _ in data["edges"])}, "label_propagation_correct": baseline, "measurements": measured, "exact": exact_cases()}
    (HERE / "calculated-inputs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"roles": {k: len(v) for k, v in roles.items()}, "baseline": baseline, "measurements": [{"model": r["model"], "seed": r["seed"], "correct": r["correct"], "removed": r["propagation_removed_assessment_correct"]} for r in measured], "exact": result["exact"]}, indent=2))


if __name__ == "__main__":
    main()
