"""Position mechanisms composed with Embedding and SDPA. PyTorch 2.14 CPU.

Standard adjacent-pair RoPE and additive ALiBi; no checkpoint-scaling variant.
"""
import torch
from torch import nn
from torch.nn import functional as F


def rotate(values, positions):
    width = values.shape[-1]
    if width % 2:
        raise ValueError("An even rotary width is required")
    frequency = 10000.**(-torch.arange(0, width, 2, dtype=values.dtype)/width)
    angles = positions.to(values.dtype)[:, None]*frequency
    even, odd = values[..., ::2], values[..., 1::2]
    return torch.stack((even*angles.cos()-odd*angles.sin(),
                        even*angles.sin()+odd*angles.cos()), -1).flatten(-2)


def read(query, key, value, query_ids, key_ids, mode, library):
    if mode == "rope":
        query, key = rotate(query, query_ids), rotate(key, key_ids)
    legal = key_ids[None, :] <= query_ids[:, None]
    bias = query.new_zeros((len(query_ids), len(key_ids)))
    if mode == "alibi":
        bias = -.4*(query_ids[:, None]-key_ids[None, :]).to(query.dtype)
    bias = bias.masked_fill(~legal, -torch.inf)
    if library:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=0.)
    return (query@key.T/query.shape[-1]**.5+bias).softmax(-1)@value


def main():
    torch.manual_seed(59)
    torch.set_num_threads(1)
    queries = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    keys = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    q_ids, k_ids = torch.tensor([9, 10]), torch.tensor([7, 8, 9, 10])
    for mode in ("rope", "alibi"):
        direct = read(queries, keys, values, q_ids, k_ids, mode, False)
        native = read(queries, keys, values, q_ids, k_ids, mode, True)
        torch.testing.assert_close(direct, native, atol=1e-11, rtol=1e-11)
        left = torch.autograd.grad(direct.square().sum(), (queries, keys, values))
        right = torch.autograd.grad(native.square().sum(), (queries, keys, values))
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        print(mode, "output and gradient agreement: True")
    table = nn.Embedding(12, 4).double()
    ids = torch.tensor([7, 9, 7])
    output = table(ids)
    torch.testing.assert_close(output, table.weight[ids])
    output.sum().backward()
    assert torch.equal(table.weight.grad[7], torch.full((4,), 2., dtype=torch.float64))
    assert torch.equal(table.weight.grad[9], torch.ones(4, dtype=torch.float64))
    assert table.weight.grad[[0, 1, 8, 10, 11]].abs().sum() == 0
    print("repeated position ID accumulates two gradients: True")


if __name__ == "__main__":
    main()
