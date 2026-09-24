"""Manual GPS composition versus PyG, using the earlier sparse graph owner.

Place graph_library_bridge.py from the Message Passing packet beside this file.
Requires torch 2.14 and torch-geometric 2.9.0. No dataset or GPU required.
"""
import json
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GPSConv
from graph_library_bridge import SparseGraphLayer, paired_library


class LocalGlobalBlock(nn.Module):
    def __init__(self, width=4, heads=2):
        super().__init__()
        self.local = SparseGraphLayer(width, width, "gcn")
        self.global_attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.feedforward = nn.Sequential(nn.Linear(width, 2*width), nn.ReLU(),
                                         nn.Dropout(0.), nn.Linear(2*width, width), nn.Dropout(0.))

    def forward(self, features, edges, batch):
        # Processing each graph independently makes the information boundary explicit.
        global_update = torch.zeros_like(features)
        for graph in torch.unique(batch, sorted=True):
            indices = (batch == graph).nonzero(as_tuple=True)[0]
            x = features[indices][None]
            update, _ = self.global_attention(x, x, x, need_weights=False)
            global_update = global_update.index_copy(0, indices, update[0])
        mixed = (features + self.local(features, edges)) + (features + global_update)
        return mixed + self.feedforward(mixed)


def matched_package(manual):
    local, local_pairs = paired_library(manual.local)
    package = GPSConv(4, local, heads=2, dropout=0., act="relu", norm=None,
                      attn_type="multihead", attn_kwargs={"dropout": 0.}).double()
    package.attn.load_state_dict(manual.global_attention.state_dict())
    package.mlp.load_state_dict(manual.feedforward.state_dict())
    pairs = local_pairs + list(zip(manual.global_attention.parameters(), package.attn.parameters(), strict=True))
    pairs += list(zip(manual.feedforward.parameters(), package.mlp.parameters(), strict=True))
    return package, pairs


def return_feature(count, edges):
    # One two-step return probability per node, in the receiver-row convention.
    adjacency = torch.zeros(count, count, dtype=torch.float64)
    adjacency[edges[1], edges[0]] = 1
    transition = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
    return (transition @ transition).diagonal()[:, None]


def main():
    torch.manual_seed(31)
    torch.set_num_threads(1)
    edge_a = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_b = torch.tensor([[0, 1], [1, 0]])
    raw = [torch.tensor([[1., 0., .2], [-1., .3, 1.], [.4, -.7, 2.]], dtype=torch.float64),
           torch.tensor([[.3, 1., -.5], [2., -.4, .7]], dtype=torch.float64)]
    graphs = [Data(x=torch.cat((x, return_feature(len(x), e)), -1), edge_index=e)
              for x, e in zip(raw, (edge_a, edge_b), strict=True)]
    packed = Batch.from_data_list(graphs)
    manual = LocalGlobalBlock().double().eval()
    package, pairs = matched_package(manual)
    package.eval()
    left = packed.x.detach().clone().requires_grad_()
    right = packed.x.detach().clone().requires_grad_()
    actual = manual(left, packed.edge_index, packed.batch)
    expected = package(right, packed.edge_index, batch=packed.batch)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    probe = torch.linspace(-.6, .8, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    (actual * probe).sum().backward(); (expected * probe).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-9, rtol=1e-9)
    with torch.no_grad():
        for ours, theirs in pairs:
            torch.testing.assert_close(ours.grad, theirs.grad.reshape_as(ours), atol=1e-9, rtol=1e-9)
            ours.add_(ours.grad, alpha=-.02); theirs.add_(theirs.grad, alpha=-.02)
        updated = package(packed.x, packed.edge_index, batch=packed.batch)
        torch.testing.assert_close(manual(packed.x, packed.edge_index, packed.batch), updated,
                                   atol=1e-9, rtol=1e-9)
        separate = torch.cat([package(g.x, g.edge_index, batch=torch.zeros(len(g.x), dtype=torch.long))
                              for g in graphs])
        torch.testing.assert_close(updated, separate, atol=1e-10, rtol=1e-10)
        changed = packed.x.clone(); changed[3:] += 4
        changed_output = package(changed, packed.edge_index, batch=packed.batch)
        torch.testing.assert_close(changed_output[:3], updated[:3], atol=1e-10, rtol=1e-10)
        # Relabel each graph without breaking the sorted batch-ID requirement.
        permutation = torch.tensor([2, 0, 1, 4, 3])
        inverse = torch.argsort(permutation)
        relabeled = package(packed.x[permutation], inverse[packed.edge_index], batch=packed.batch)
        torch.testing.assert_close(relabeled, updated[permutation], atol=1e-10, rtol=1e-10)
    print(json.dumps({"nodes": len(packed.x), "features": packed.x.shape[1],
                      "manual_package_error": float((actual-expected).abs().max().detach()),
                      "input_gradient_and_update": "matched", "batch_isolation": "matched",
                      "relabeling": "matched", "norm": None, "dropout": 0.}, indent=2))


if __name__ == "__main__":
    main()
