"""Connect the lesson's selective recurrence to the maintained Mamba operator.

Requires a compatible PyTorch/CUDA/mamba-ssm installation. Run beside
trajectory_state_models.py. It is a prepared GPU example, not a measured result.
"""
from pathlib import Path
import runpy
import torch
from torch.nn import functional as F


def compare_scan():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires a compatible CUDA installation")
    namespace = runpy.run_path(str(Path(__file__).with_name("trajectory_state_models.py")))
    torch.manual_seed(17)
    mixer = namespace["SelectiveMixer"](width=8, state_size=4).cuda()
    inputs = torch.randn(2, 19, 8, device="cuda", requires_grad=True)
    reference = mixer(inputs)
    projected = mixer.coefficients(inputs)
    write, read, raw_step = projected.split([4, 4, 8], dim=-1)
    step = F.softplus(raw_step)
    native = selective_scan_fn(
        inputs.transpose(1, 2).contiguous(), step.transpose(1, 2).contiguous(),
        -mixer.log_decay.exp(), write.transpose(1, 2).contiguous(),
        read.transpose(1, 2).contiguous(), mixer.skip,
        delta_softplus=False,
    ).transpose(1, 2)
    torch.testing.assert_close(reference, native, atol=2e-5, rtol=2e-4)
    variables = (inputs, *mixer.parameters())
    cotangent = torch.randn_like(reference)
    manual_gradients = torch.autograd.grad((reference * cotangent).sum(), variables,
                                          retain_graph=True)
    native_gradients = torch.autograd.grad((native * cotangent).sum(), variables)
    for manual, library in zip(manual_gradients, native_gradients):
        torch.testing.assert_close(manual, library, atol=1e-4, rtol=5e-4)
    print({"max_forward_difference": float((reference-native).abs().max()),
           "parameters_compared": len(variables)})


def train_complete_blocks():
    from mamba_ssm import Mamba, Mamba2

    # These are complete blocks, unlike the isolated scan above. Parameters do
    # not map one-to-one to our small classifier, so no cross-model parity claim.
    torch.manual_seed(21)
    inputs = torch.randn(2, 64, 64, device="cuda")
    target = inputs.roll(1, dims=1)
    for constructor in (Mamba, Mamba2):
        block = constructor(d_model=64, d_state=64, d_conv=4, expand=2).cuda()
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
        block.train()
        optimizer.zero_grad(set_to_none=True)
        output = block(inputs)
        loss = F.mse_loss(output[:, 1:], target[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        optimizer.step()
        block.eval()
        with torch.inference_mode():
            evaluated = block(inputs)
        print(constructor.__name__, tuple(evaluated.shape), float(loss.detach()))


if __name__ == "__main__":
    compare_scan()
    train_complete_blocks()
