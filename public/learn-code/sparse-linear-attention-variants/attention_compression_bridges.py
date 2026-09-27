"""Trainable sequence-compression mechanisms and ordinary tensor/library routes.

Linformer and Nyström here are bidirectional operators, not causal decoders.
Python 3.12+, torch 2.14. Small float64 probes; no training benchmark.
"""
import math
import torch
from torch.nn import functional as F


def linformer(query, key, value, key_projection, value_projection, native=False):
    compressed_key = key_projection @ key
    compressed_value = value_projection @ value
    if native:
        return F.scaled_dot_product_attention(query, compressed_key, compressed_value, dropout_p=0.)
    score = query @ compressed_key.T / math.sqrt(query.shape[-1])
    return score.softmax(-1) @ compressed_value


def segment_landmarks(values, count):
    if not 1 <= count <= len(values):
        raise ValueError("Use between one landmark and one per position")
    # tensor_split covers a non-divisible trailing segment without dropping tokens.
    return torch.stack([part.mean(0) for part in torch.tensor_split(values, count)])


def nystrom(query, key, value, landmarks, pseudoinverse_rtol=1e-10):
    q_bar, k_bar = segment_landmarks(query, landmarks), segment_landmarks(key, landmarks)
    scale = 1 / math.sqrt(query.shape[-1])
    front = (query @ k_bar.T * scale).softmax(-1)
    middle = (q_bar @ k_bar.T * scale).softmax(-1)
    back = (q_bar @ key.T * scale).softmax(-1)
    # Parenthesization avoids materializing the L by L approximate attention.
    reduced = back @ value
    return front @ (torch.linalg.pinv(middle, rtol=pseudoinverse_rtol) @ reduced)


def gathered_window(query, key, value, width):
    if width < 1:
        raise ValueError("Window must include at least the current position")
    outputs = []
    for position, q in enumerate(query):
        start = max(0, position-width+1)
        scores = key[start:position+1] @ q / math.sqrt(query.shape[-1])
        outputs.append(scores.softmax(0) @ value[start:position+1])
    return torch.stack(outputs)


def main():
    torch.manual_seed(31)
    torch.set_num_threads(1)
    q, k, v = [torch.randn(7, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    e, f = [torch.randn(3, 7, dtype=torch.float64, requires_grad=True) / 7**.5 for _ in range(2)]
    manual, native = linformer(q, k, v, e, f), linformer(q, k, v, e, f, True)
    torch.testing.assert_close(manual, native, atol=1e-11, rtol=1e-11)
    a = torch.autograd.grad(manual.square().sum(), (q, k, v, e, f))
    b = torch.autograd.grad(native.square().sum(), (q, k, v, e, f))
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
    positions = torch.arange(7)
    legal = (positions[:, None] >= positions[None, :]) & (positions[:, None]-positions[None, :] < 3)
    window = gathered_window(q, k, v, 3)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=legal, dropout_p=0.)
    torch.testing.assert_close(window, reference, atol=1e-11, rtol=1e-11)
    whole = nystrom(q, k, v, 7)
    dense = (q @ k.T / math.sqrt(3)).softmax(-1) @ v
    torch.testing.assert_close(whole, dense, atol=1e-8, rtol=1e-8)
    approximate = nystrom(q, k, v, 3)
    derivatives = torch.autograd.grad(approximate.square().sum(), (q, k, v))
    assert all(torch.isfinite(item).all() for item in derivatives)
    # This measures a concrete approximation error; it does not assert monotonicity.
    print("Linformer API error:", float((manual-native).abs().max().detach()))
    print("gathered-window API error:", float((window-reference).abs().max().detach()))
    print("all-landmark reconstruction error:", float((whole-dense).abs().max().detach()))
    print("three-landmark output error:", float((approximate-dense).abs().max().detach()))


if __name__ == "__main__":
    main()
