"""Complete small MBConv/ASPP-style building-block demonstration.

The blocks are randomly initialized, not trained segmentation models.
The adjacent real CSV supplies two images; no download is required.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


class InvertedResidual(nn.Module):
    def __init__(self, inputs, outputs, expansion=3, stride=1):
        super().__init__()
        if expansion < 1 or stride not in (1, 2):
            raise ValueError("use positive expansion and stride 1 or 2")
        hidden = inputs * expansion
        self.branch = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(),
            nn.Conv2d(hidden, outputs, 1, bias=False), nn.BatchNorm2d(outputs),
        )
        self.has_skip = stride == 1 and inputs == outputs

    def forward(self, inputs):
        branch = self.branch(inputs)
        return inputs + branch if self.has_skip else branch


class ParallelContext(nn.Module):
    def __init__(self, inputs, width=4, rates=(1, 2)):
        super().__init__()
        if not rates or any(rate < 1 for rate in rates):
            raise ValueError("rates must be nonempty positive integers")
        self.local = nn.ModuleList([
            nn.Sequential(nn.Conv2d(inputs, width, 1, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU())
        ] + [
            nn.Sequential(nn.Conv2d(inputs, width, 3, padding=rate,
                                    dilation=rate, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU())
            for rate in rates
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(inputs, width, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv2d((len(rates)+2)*width, width, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU())

    def forward(self, inputs):
        branches = [branch(inputs) for branch in self.local]
        global_features = self.global_branch(inputs)
        branches.append(F.interpolate(global_features, size=inputs.shape[-2:],
                                      mode="bilinear", align_corners=False))
        return self.project(torch.cat(branches, dim=1))


def main():
    torch.manual_seed(11)
    rows = np.genfromtxt(HERE/"digits-400.csv", delimiter=",", names=True)[:2]
    pixels = np.column_stack([rows[f"pixel_{index}"] for index in range(64)])
    images = torch.tensor(pixels/16, dtype=torch.float32).reshape(2,1,8,8)
    stem = nn.Conv2d(1,8,3,padding=1)
    mobile = InvertedResidual(8,8)
    context = ParallelContext(8)
    features = F.relu(stem(images))
    transformed = mobile(features)
    contextual = context(transformed)
    # A local differentiability probe, not a task objective or a training experiment.
    probe = contextual.square().mean()
    probe.backward()
    result = {
        "source_ids":rows["source_id"].astype(int).tolist(),
        "image_shape":list(images.shape),"mobile_shape":list(transformed.shape),
        "context_shape":list(contextual.shape),
        "stem_gradient_finite":bool(torch.isfinite(stem.weight.grad).all()),
        "stem_gradient_nonzero":bool((stem.weight.grad != 0).any()),
        "mode":"randomly initialized train-mode, batch two; no fitted segmentation output",
    }
    (HERE/"block-check-results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
