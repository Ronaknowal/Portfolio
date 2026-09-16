"""Shape/parameter arithmetic and small forwards; no pretrained download."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, values):
        return self.norm(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResponseNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
        self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, values):
        magnitude = torch.linalg.vector_norm(values, dim=(1, 2), keepdim=True)
        relative = magnitude / (magnitude.mean(-1, keepdim=True) + 1e-6)
        return values + self.scale * values * relative + self.shift


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("probability must lie in [0, 1)")
        self.probability = probability

    def forward(self, branch):
        if not self.training or self.probability == 0:
            return branch
        keep = 1 - self.probability
        mask = (torch.rand((len(branch),)+(1,)*(branch.ndim-1),
                           device=branch.device) < keep).to(branch.dtype)
        return branch * mask / keep


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, version=1, drop_probability=0.0):
        super().__init__()
        if version not in (1, 2):
            raise ValueError("version must be 1 or 2")
        self.spatial = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4*channels)
        self.response = ResponseNorm(4*channels) if version == 2 else nn.Identity()
        self.project = nn.Linear(4*channels, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6)) if version == 1 else None
        self.drop_path = DropPath(drop_probability)

    def forward(self, inputs):
        branch = self.spatial(inputs).permute(0, 2, 3, 1)
        branch = self.norm(branch)
        branch = self.project(self.response(F.gelu(self.expand(branch))))
        if self.layer_scale is not None:
            branch = branch * self.layer_scale
        # Computing the branch above is not avoided when its contribution is dropped.
        return inputs + self.drop_path(branch.permute(0, 3, 1, 2))


class ConvNeXt(nn.Module):
    def __init__(self, widths=(96,192,384,768), depths=(3,3,9,3),
                 version=1, classes=1000, maximum_drop=0.1, initialize=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,widths[0],4,stride=4), ChannelNorm(widths[0]))
        self.downsamples = nn.ModuleList([
            nn.Sequential(ChannelNorm(widths[i-1]),nn.Conv2d(widths[i-1],widths[i],2,stride=2))
            for i in range(1,4)])
        total = sum(depths)
        rates = [maximum_drop*i/max(1,total-1) for i in range(total)]
        offset = 0
        self.stages = nn.ModuleList()
        for width, depth in zip(widths,depths):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(width,version,rates[offset+i]) for i in range(depth)]))
            offset += depth
        self.final_norm = nn.LayerNorm(widths[-1],eps=1e-6)
        self.head = nn.Linear(widths[-1],classes)
        if initialize:
            self.apply(self.initialize)

    @staticmethod
    def initialize(layer):
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            nn.init.trunc_normal_(layer.weight,std=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, images, return_features=False):
        values = self.stem(images)
        features = []
        for stage_index,stage in enumerate(self.stages):
            if stage_index:
                values = self.downsamples[stage_index-1](values)
            values = stage(values)
            features.append(values)
        if return_features:
            return features
        return self.head(self.final_norm(values.mean((2,3))))


def costs(widths, depths, side=224, classes=1000):
    # One MAC is one multiplication followed by accumulation; no LN/GELU/residual costs.
    height = side//4
    stem = height*height*3*widths[0]*16
    stages, transitions = [], []
    for index,(width,depth) in enumerate(zip(widths,depths)):
        if index:
            height //= 2
            transitions.append(height*height*4*widths[index-1]*width)
        stages.append({"side":height,"width":width,"depth":depth,
                       "depthwise_macs":depth*height*height*49*width,
                       "pointwise_macs":depth*height*height*8*width*width})
    return {"stem_macs":stem,"stages":stages,"transition_macs":transitions,
            "head_macs":widths[-1]*classes,
            "total_convolution_linear_macs":stem+sum(transitions)+widths[-1]*classes+
            sum(x["depthwise_macs"]+x["pointwise_macs"] for x in stages)}


def main():
    configurations = {"tiny":(96,(3,3,9,3)),"small":(96,(3,3,27,3)),
                      "base":(128,(3,3,27,3)),"large":(192,(3,3,27,3)),
                      "xlarge":(256,(3,3,27,3))}
    results = {"models":[]}
    for name,(base,depths) in configurations.items():
        widths = tuple(base*2**i for i in range(4))
        for version in (1,2):
            # Meta tensors carry shapes without allocating hundreds of millions of weights.
            with torch.device("meta"):
                model = ConvNeXt(widths,depths,version,initialize=False)
                output = model(torch.empty(2,3,224,224))
                features = model(torch.empty(2,3,224,224),return_features=True)
            results["models"].append({"name":name,"version":version,
                "parameters":sum(p.numel() for p in model.parameters()),
                "feature_shapes":[list(f.shape) for f in features],
                "output_shape":list(output.shape),"costs":costs(widths,depths)})
    torch.manual_seed(7)
    for version in (1,2):
        block = ConvNeXtBlock(96,version)
        inputs = torch.randn(2,96,3,3,requires_grad=True)
        block(inputs).square().mean().backward()
        results[f"block_v{version}"] = {"parameters":sum(p.numel() for p in block.parameters()),
            "input_gradient_finite":bool(torch.isfinite(inputs.grad).all())}
    rates = torch.linspace(0,.1,18)
    results["expected_retained_tiny_branches"] = float((1-rates).sum())
    (HERE/"block-check-results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({k:v for k,v in results.items() if k!="models"}))


if __name__ == "__main__":
    main()
