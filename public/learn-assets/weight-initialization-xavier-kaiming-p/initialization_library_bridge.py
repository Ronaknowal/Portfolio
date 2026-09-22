"""Same-state initialization bridges; run beside initialization-experiments.py.

Python 3.12 and PyTorch; `--mup` also needs Microsoft's `mup` package.
Verified with PyTorch 2.14.0+cpu and mup 1.0.0; see native-verification.json.
"""
import argparse
import math
from pathlib import Path
import runpy

import torch
from torch import nn


def orthogonal_matrix(rows, columns, gain=1.0, generator=None):
    """Return a dense semi-orthogonal matrix; QR is the reused LA primitive."""
    if rows < 1 or columns < 1:
        raise ValueError("matrix dimensions must be positive")
    # Reduced QR creates orthonormal columns; transpose for a wide output.
    tall = torch.randn(max(rows, columns), min(rows, columns),
                       dtype=torch.float64, generator=generator)
    basis, triangular = torch.linalg.qr(tall, mode="reduced")
    signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
    basis = basis * signs
    return gain * (basis if rows >= columns else basis.T)


def check_orthogonal():
    for rows, columns in ((7, 3), (3, 7), (4, 4)):
        manual = orthogonal_matrix(rows, columns, math.sqrt(2),
                                   torch.Generator().manual_seed(11))
        library = torch.empty_like(manual)
        nn.init.orthogonal_(library, gain=math.sqrt(2),
                            generator=torch.Generator().manual_seed(11))
        # Wide matrices consume random draws in a different shape. Their
        # distribution/Gram contract matches; identical entries are not promised.
        for matrix in (manual, library):
            gram = matrix.T @ matrix if rows >= columns else matrix @ matrix.T
            torch.testing.assert_close(gram, 2 * torch.eye(min(rows, columns),
                                       dtype=torch.float64), atol=1e-12, rtol=1e-12)
    print("orthogonal rectangular Gram contracts passed")


def check_mup():
    from mup import MuAdam, MuReadout, set_base_shapes

    namespace = runpy.run_path(str(Path(__file__).with_name("initialization-experiments.py")))
    WidthMLP = namespace["WidthMLP"]

    class LibraryWidthMLP(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.lower = nn.Linear(64, width, bias=False)
            self.upper = nn.Linear(width, width, bias=False)
            self.readout = MuReadout(width, 10, bias=False)

        def forward(self, inputs):
            return self.readout(torch.relu(self.upper(torch.relu(self.lower(inputs)))))

    for width in (32, 96):
        manual = WidthMLP(width, "mu", seed=7).double()
        library = LibraryWidthMLP(width).double()
        base, delta = LibraryWidthMLP(32), LibraryWidthMLP(64)
        # Record width axes, but do not rescale already-parametrized copied data.
        set_base_shapes(library, base, delta=delta, rescale_params=False)
        library.load_state_dict(manual.state_dict())
        manual_optimizer = manual.optimizer(0.003)
        library_optimizer = MuAdam(library.parameters(), lr=0.003, eps=1e-8,
                                   weight_decay=0.0)
        generator = torch.Generator().manual_seed(21)
        inputs = torch.randn(5, 64, dtype=torch.float64, generator=generator)
        targets = torch.tensor([0, 2, 4, 6, 8])
        for _ in range(2):
            torch.testing.assert_close(manual(inputs), library(inputs), atol=1e-12, rtol=1e-12)
            for model, optimizer in ((manual, manual_optimizer), (library, library_optimizer)):
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(inputs), targets).backward()
            for (_, left), (_, right) in zip(manual.named_parameters(), library.named_parameters()):
                torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)
            manual_optimizer.step()
            library_optimizer.step()
            for left, right in zip(manual.parameters(), library.parameters()):
                torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        print({"width": width, "readout_divisor": library.readout.width_mult(),
               "groups": [group["lr"] for group in library_optimizer.param_groups]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mup", action="store_true")
    arguments = parser.parse_args()
    check_orthogonal()
    if arguments.mup:
        check_mup()
