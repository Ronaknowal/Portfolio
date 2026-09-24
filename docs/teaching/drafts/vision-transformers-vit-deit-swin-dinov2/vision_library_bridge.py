"""Swin V1 parameter/layout bridge to torchvision 0.29; optional unexecuted route.

Keep vision-mechanisms.py beside this program. No pretrained download or fitting.
"""
import importlib.util
from pathlib import Path
import torch
from torchvision.models.swin_transformer import PatchMerging, ShiftedWindowAttention


def main():
    spec = importlib.util.spec_from_file_location("vision_mechanisms", Path(__file__).with_name("vision-mechanisms.py"))
    mechanisms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mechanisms)
    torch.manual_seed(61)
    torch.set_num_threads(1)
    manual = mechanisms.ShiftedWindowAttention(width=4, heads=2, window=3, shift=1).double().eval()
    library = ShiftedWindowAttention(dim=4, window_size=[3, 3], shift_size=[1, 1],
                                    num_heads=2, attention_dropout=0., dropout=0.).double().eval()
    library.qkv.load_state_dict(manual.qkv.state_dict())
    library.proj.load_state_dict(manual.output.state_dict())
    with torch.no_grad():
        manual.relative_bias.copy_(torch.randn_like(manual.relative_bias)*.1)
        library.relative_position_bias_table.copy_(manual.relative_bias.T)
    # Divisible dimensions isolate the shared operator. Padding policies differ.
    x = torch.randn(1, 6, 9, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    ours, theirs = manual(x), library(y)
    torch.testing.assert_close(ours, theirs, atol=1e-10, rtol=1e-10)
    ours.square().mean().backward(); theirs.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)
    pairs = [(manual.qkv.weight, library.qkv.weight), (manual.qkv.bias, library.qkv.bias),
             (manual.output.weight, library.proj.weight), (manual.output.bias, library.proj.bias)]
    for a, b in pairs:
        torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(manual.relative_bias.grad.T, library.relative_position_bias_table.grad,
                               atol=1e-9, rtol=1e-9)
    merging = mechanisms.PatchMerge(4).double()
    reference = PatchMerging(4).double()
    reference.norm.load_state_dict(merging.norm.state_dict())
    reference.reduction.load_state_dict(merging.reduce.state_dict())
    odd = torch.randn(1, 5, 7, 4, dtype=torch.float64)
    torch.testing.assert_close(merging(odd), reference(odd), atol=1e-11, rtol=1e-11)
    print("shifted-window output error:", float((ours-theirs).abs().max().detach()))
    print("input/parameter gradients: matched; odd-grid merge shape:", tuple(merging(odd).shape))


if __name__ == "__main__":
    main()
