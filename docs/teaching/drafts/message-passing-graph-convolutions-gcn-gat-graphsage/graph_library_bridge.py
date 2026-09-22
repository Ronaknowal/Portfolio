"""Sparse graph mechanisms and matched PyG layers; no training dataset required.

Python 3.12+, torch 2.14; optional torch-geometric 2.9.0 for the library checks.
Run --scratch-only without PyG. The default compares values, derivatives and SGD.
Simple unweighted graph, unique (source,target) edges, no input self-loops.
"""
import argparse
import copy
import json
import torch
from torch import nn


class SparseGraphLayer(nn.Module):
    def __init__(self, incoming, outgoing, kind):
        super().__init__()
        if kind not in ("gcn", "sage", "gat"):
            raise ValueError("Choose gcn, sage or gat")
        self.kind = kind
        self.linear = nn.Linear(incoming, outgoing, bias=False)
        self.bias = nn.Parameter(torch.zeros(outgoing))
        if kind == "sage":
            self.self_linear = nn.Linear(incoming, outgoing, bias=False)
        if kind == "gat":
            self.sender_score = nn.Parameter(torch.randn(outgoing) * .1)
            self.receiver_score = nn.Parameter(torch.randn(outgoing) * .1)

    def forward(self, features, edge_index):
        n = len(features)
        source, target = edge_index
        if self.kind != "sage":
            loops = torch.arange(n, device=edge_index.device)
            source, target = torch.cat((source, loops)), torch.cat((target, loops))
        transformed = self.linear(features)
        degree = features.new_zeros(n).index_add_(0, target, features.new_ones(len(target)))
        if self.kind == "gcn":
            # PyG source_to_target uses receiving degree for this directed operator.
            inverse = degree.rsqrt()
            coefficients = inverse[source] * inverse[target]
        elif self.kind == "sage":
            coefficients = degree[target].reciprocal()
        else:
            scores = nn.functional.leaky_relu(
                transformed[source] @ self.sender_score +
                transformed[target] @ self.receiver_score, negative_slope=.2)
            maxima = scores.new_full((n,), -torch.inf)
            maxima.scatter_reduce_(0, target, scores.detach(), reduce="amax", include_self=True)
            masses = (scores - maxima[target]).exp()
            totals = masses.new_zeros(n).index_add_(0, target, masses)
            coefficients = masses / totals[target]
        output = transformed.new_zeros((n, transformed.shape[1]))
        output.index_add_(0, target, coefficients[:, None] * transformed[source])
        if self.kind == "sage":
            output = output + self.self_linear(features)
        return output + self.bias


def dense_reference(layer, features, edges):
    n = len(features)
    adjacency = features.new_zeros((n, n))
    adjacency[edges[1], edges[0]] = 1
    projected = layer.linear(features)
    if layer.kind == "sage":
        mean = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
        result = mean @ projected + layer.self_linear(features)
    else:
        adjacency = adjacency + torch.eye(n, dtype=features.dtype, device=features.device)
        if layer.kind == "gcn":
            inverse = adjacency.sum(-1).rsqrt()
            result = (inverse[:, None] * adjacency * inverse[None, :]) @ projected
        else:
            scores = projected @ layer.receiver_score
            scores = scores[:, None] + (projected @ layer.sender_score)[None, :]
            weights = nn.functional.leaky_relu(scores, .2).masked_fill(adjacency == 0, -torch.inf).softmax(-1)
            result = weights @ projected
    return result + layer.bias


def paired_library(layer):
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    incoming, outgoing = layer.linear.in_features, layer.linear.out_features
    if layer.kind == "gcn":
        other = GCNConv(incoming, outgoing, cached=False, normalize=True,
                        add_self_loops=True, improved=False).double()
        pairs = [(layer.linear.weight, other.lin.weight), (layer.bias, other.bias)]
    elif layer.kind == "sage":
        other = SAGEConv(incoming, outgoing, aggr="mean", normalize=False,
                         project=False, root_weight=True).double()
        pairs = [(layer.linear.weight, other.lin_l.weight), (layer.bias, other.lin_l.bias),
                 (layer.self_linear.weight, other.lin_r.weight)]
    else:
        other = GATConv(incoming, outgoing, heads=1, concat=True, negative_slope=.2,
                        dropout=0., add_self_loops=True, residual=False).double()
        pairs = [(layer.linear.weight, other.lin.weight), (layer.bias, other.bias),
                 (layer.sender_score, other.att_src), (layer.receiver_score, other.att_dst)]
    with torch.no_grad():
        for ours, theirs in pairs:
            theirs.copy_(ours.reshape_as(theirs))
    return other, pairs


def compare(layer, features, edges, library):
    other, pairs = paired_library(layer) if library else (copy.deepcopy(layer), None)
    if pairs is None:
        pairs = list(zip(layer.parameters(), other.parameters(), strict=True))
    left = features.clone().requires_grad_()
    right = features.clone().requires_grad_()
    actual = layer(left, edges)
    expected = other(right, edges) if library else dense_reference(other, right, edges)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    probe = torch.linspace(-.7, .9, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-9, rtol=1e-9)
    gradient_error = 0.
    with torch.no_grad():
        for ours, theirs in pairs:
            mapped = theirs.grad.reshape_as(ours)
            torch.testing.assert_close(ours.grad, mapped, atol=1e-9, rtol=1e-9)
            gradient_error = max(gradient_error, float((ours.grad - mapped).abs().max()))
            ours.add_(ours.grad, alpha=-.03)
            theirs.add_(theirs.grad, alpha=-.03)
    updated = other(features, edges) if library else dense_reference(other, features, edges)
    torch.testing.assert_close(layer(features, edges), updated, atol=1e-9, rtol=1e-9)
    return {"output_error": float((actual-expected).abs().max().detach()),
            "parameter_gradient_error": gradient_error, "matched_sgd_step": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(73)
    torch.set_num_threads(1)
    features = torch.tensor([[1., 2.], [-1., .3], [.4, -2.], [3., .1]], dtype=torch.float64)
    graphs = {"path_and_isolate": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
              "edited_directed": torch.tensor([[0, 1, 2], [1, 2, 0]]),
              "no_edges": torch.empty((2, 0), dtype=torch.long)}
    results = {}
    for name, edges in graphs.items():
        for kind in ("gcn", "sage", "gat"):
            layer = SparseGraphLayer(2, 3, kind).double()
            results[f"{name}/{kind}"] = compare(layer, features, edges, not args.scratch_only)
    print(json.dumps({"route": "dense scratch oracle" if args.scratch_only else "PyG 2.9 contract",
                      "torch": torch.__version__, "checks": results}, indent=2))


if __name__ == "__main__":
    main()
