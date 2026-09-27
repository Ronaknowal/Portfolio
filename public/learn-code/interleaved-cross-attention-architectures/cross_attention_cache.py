"""Project fixed cross-attention memory once; compare against ordinary MHA.

Inference only: a cache belongs to one memory and parameter revision.
torch 2.14, CPU float64. All fixtures are declared constructed inputs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def split_heads(values, heads):
    batch, length, width = values.shape
    return values.reshape(batch, length, heads, width//heads).transpose(1, 2)


def project_memory(memory, layer, identity):
    width = layer.embed_dim
    weights = layer.in_proj_weight.chunk(3)
    biases = layer.in_proj_bias.chunk(3)
    keys = split_heads(F.linear(memory, weights[1], biases[1]), layer.num_heads)
    values = split_heads(F.linear(memory, weights[2], biases[2]), layer.num_heads)
    return {"keys": keys, "values": values, "identity": identity}


def read_memory(queries, layer, cache, allowed, identity):
    if identity != cache["identity"]:
        raise ValueError("Memory, preprocessing or model revision changed; rebuild the cache")
    if not allowed.any(-1).all():
        raise ValueError("Each query needs an available memory entry")
    width = layer.embed_dim
    projected = F.linear(queries, layer.in_proj_weight[:width], layer.in_proj_bias[:width])
    q = split_heads(projected, layer.num_heads)
    scores = q @ cache["keys"].transpose(-1, -2) / math.sqrt(width//layer.num_heads)
    output = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ cache["values"]
    merged = output.transpose(1, 2).reshape(queries.shape)
    return layer.out_proj(merged)


def main():
    torch.manual_seed(53)
    layer = nn.MultiheadAttention(8, 2, dropout=0., batch_first=True).double().eval()
    memory = torch.randn(1, 5, 8, dtype=torch.float64)
    queries = torch.randn(1, 3, 8, dtype=torch.float64)
    # A query-dependent availability mask, not a square language-causal mask.
    allowed = torch.arange(5)[None, :] < torch.tensor([2, 4, 5])[:, None]
    identity = ("image-17/crop-2", "weights-0")
    with torch.no_grad():
        cache = project_memory(memory, layer, identity)
        manual = read_memory(queries, layer, cache, allowed, identity)
        reference, _ = layer(queries, memory, memory, attn_mask=~allowed, need_weights=False)
        streamed = torch.cat([read_memory(queries[:, i:i+1], layer, cache, allowed[i:i+1], identity)
                              for i in range(3)], dim=1)
        torch.testing.assert_close(manual, reference, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(streamed, reference, atol=1e-11, rtol=1e-11)
        changed = memory.clone(); changed[:, -1] += 2
        new_identity = ("image-17/crop-3", "weights-0")
        refreshed = project_memory(changed, layer, new_identity)
        edited = read_memory(queries, layer, refreshed, allowed, new_identity)
        torch.testing.assert_close(edited[:, :2], manual[:, :2], atol=1e-11, rtol=1e-11)
        assert not torch.allclose(edited[:, -1], manual[:, -1])
        try:
            read_memory(queries, layer, cache, allowed, new_identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Stale cache was not rejected")
    print("full/streamed/library error:", float((manual-reference).abs().max()))
    print("forbidden-memory null and stale-identity rejection: passed")


if __name__ == "__main__":
    main()
