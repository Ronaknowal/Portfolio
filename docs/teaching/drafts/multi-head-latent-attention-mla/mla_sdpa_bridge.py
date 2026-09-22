"""Absorbed MLA through ordinary PyTorch SDPA, including matching gradients.

The latent/rotary tensors are already projected, normalized and positioned.
This is the local MLA operator, not a DeepSeek checkpoint or fused MLA backend.
"""
import math
import torch
from torch.nn import functional as F


def absorbed(query, latent, key_up, value_up, rotary_query, rotary_key, allowed, native=False):
    # Q [B,H,T,P], C [B,S,C], U_K [H,P,C], U_V [H,V,C].
    effective = torch.einsum("bhtp,hpc->bhtc", query, key_up)
    joined_query = torch.cat((effective, rotary_query), -1)
    joined_key = torch.cat((latent, rotary_key), -1)[:, None]
    scale = 1 / math.sqrt(query.shape[-1] + rotary_query.shape[-1])
    if native:
        mixture = F.scaled_dot_product_attention(joined_query, joined_key, latent[:, None],
            attn_mask=allowed, dropout_p=0., is_causal=False, scale=scale, enable_gqa=True)
    else:
        scores = (joined_query @ joined_key.transpose(-1, -2)) * scale
        mixture = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ latent[:, None]
    return torch.einsum("bhtc,hvc->bhtv", mixture, value_up)


def main():
    torch.manual_seed(43)
    torch.set_num_threads(1)
    dimensions = [(1, 2, 2, 3), (1, 4, 5), (2, 3, 5), (2, 2, 5), (1, 2, 2, 2), (1, 4, 2)]
    inputs = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in dimensions]
    allowed = torch.arange(4)[None, :] <= torch.tensor([2, 3])[:, None]
    direct = absorbed(*inputs, allowed)
    native = absorbed(*inputs, allowed, native=True)
    torch.testing.assert_close(direct, native, atol=1e-11, rtol=1e-11)
    probe = torch.linspace(-.8, 1., direct.numel(), dtype=direct.dtype).reshape_as(direct)
    left = torch.autograd.grad((direct*probe).sum(), inputs)
    right = torch.autograd.grad((native*probe).sum(), inputs)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    updated_direct = [x.detach()-.02*g for x, g in zip(inputs, left, strict=True)]
    updated_native = [x.detach()-.02*g for x, g in zip(inputs, right, strict=True)]
    torch.testing.assert_close(absorbed(*updated_direct, allowed),
                               absorbed(*updated_native, allowed, native=True), atol=1e-10, rtol=1e-10)
    print("output error:", float((direct-native).abs().max().detach()))
    print("gradient error:", max(float((a-b).abs().max()) for a, b in zip(left, right)))
    print("matching one-step change: True")


if __name__ == "__main__":
    main()
