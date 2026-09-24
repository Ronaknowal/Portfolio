"""Construct named CNN layouts from previously taught tensor layers.

Default mode inspects shapes on meta tensors; --compare runs one float32 CPU
forward against torchvision with explicitly copied component state. No weights
are downloaded. LeNet is the declared modern dense/tanh variant, not LeNet-5.
"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F


class ImageClassifier(nn.Module):
    def __init__(self, features, pool, head):
        super().__init__()
        self.features, self.pool, self.head = features, pool, head

    def forward(self, images):
        return self.head(self.pool(self.features(images)).flatten(1))


def lenet(classes=10):
    features = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                             nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2))
    head = nn.Sequential(nn.Linear(400, 120), nn.Tanh(), nn.Linear(120, 84),
                         nn.Tanh(), nn.Linear(84, classes))
    return ImageClassifier(features, nn.Identity(), head)


def alexnet(classes=1000):
    layers = []
    incoming = 3
    for outgoing, kernel, stride, padding, pooled in (
        (64, 11, 4, 2, True), (192, 5, 1, 2, True), (384, 3, 1, 1, False),
        (256, 3, 1, 1, False), (256, 3, 1, 1, True),
    ):
        layers.extend([nn.Conv2d(incoming, outgoing, kernel, stride, padding), nn.ReLU()])
        if pooled:
            layers.append(nn.MaxPool2d(3, 2))
        incoming = outgoing
    head = nn.Sequential(nn.Dropout(.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
                         nn.Dropout(.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, classes))
    return ImageClassifier(nn.Sequential(*layers), nn.AdaptiveAvgPool2d((6, 6)), head)


def vgg16(classes=1000):
    layers, incoming = [], 3
    for width, repeats in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
        for _ in range(repeats):
            layers.extend([nn.Conv2d(incoming, width, 3, padding=1), nn.ReLU()])
            incoming = width
        layers.append(nn.MaxPool2d(2))
    head = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(.5), nn.Linear(4096, classes))
    return ImageClassifier(nn.Sequential(*layers), nn.AdaptiveAvgPool2d((7, 7)), head)


def conv_norm(incoming, outgoing, kernel, stride=1, groups=1, activation=nn.ReLU):
    layers = [nn.Conv2d(incoming, outgoing, kernel, stride, kernel // 2,
                        groups=groups, bias=False), nn.BatchNorm2d(outgoing)]
    if activation is not None:
        layers.append(activation())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.first = conv_norm(incoming, outgoing, 3, stride)
        self.second = conv_norm(outgoing, outgoing, 3, activation=None)
        self.skip = (nn.Identity() if incoming == outgoing and stride == 1
                     else conv_norm(incoming, outgoing, 1, stride, activation=None))

    def forward(self, inputs):
        return F.relu(self.skip(inputs) + self.second(self.first(inputs)))


def resnet18(classes=1000):
    layers = [conv_norm(3, 64, 7, stride=2), nn.MaxPool2d(3, 2, 1)]
    incoming = 64
    for stage, width in enumerate((64, 128, 256, 512)):
        for block in range(2):
            stride = 2 if stage > 0 and block == 0 else 1
            layers.append(ResidualBlock(incoming, width, stride))
            incoming = width
    return ImageClassifier(nn.Sequential(*layers), nn.AdaptiveAvgPool2d(1), nn.Linear(512, classes))


class ChannelGate(nn.Module):
    def __init__(self, expanded, squeezed):
        super().__init__()
        self.reduce = nn.Conv2d(expanded, squeezed, 1)
        self.expand = nn.Conv2d(squeezed, expanded, 1)

    def forward(self, features):
        summary = features.mean((-2, -1), keepdim=True)
        gate = torch.sigmoid(self.expand(F.silu(self.reduce(summary))))
        return features * gate


class MobileBlock(nn.Module):
    def __init__(self, incoming, outgoing, expansion, kernel, stride, drop_probability):
        super().__init__()
        if not 0 <= drop_probability <= 1:
            raise ValueError("drop_probability must lie in [0, 1]")
        expanded = incoming * expansion
        branch = []
        if expansion != 1:
            branch.append(conv_norm(incoming, expanded, 1, activation=nn.SiLU))
        branch.extend([conv_norm(expanded, expanded, kernel, stride, expanded, nn.SiLU),
                       ChannelGate(expanded, max(1, incoming // 4)),
                       conv_norm(expanded, outgoing, 1, activation=None)])
        self.branch = nn.Sequential(*branch)
        self.add_skip = incoming == outgoing and stride == 1
        self.drop_probability = drop_probability

    def forward(self, inputs):
        correction = self.branch(inputs)
        if not self.add_skip:
            return correction
        if self.training and self.drop_probability:
            keep = 1 - self.drop_probability
            if keep == 0:
                return inputs + correction * 0
            mask = correction.new_empty(len(inputs), 1, 1, 1).bernoulli_(keep)
            correction = correction * mask / keep
        return inputs + correction


def efficientnet_b0(classes=1000):
    # expansion, kernel, first stride, output channels, repetitions
    stages = ((1, 3, 1, 16, 1), (6, 3, 2, 24, 2), (6, 5, 2, 40, 2),
              (6, 3, 2, 80, 3), (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1))
    layers = [conv_norm(3, 32, 3, stride=2, activation=nn.SiLU)]
    incoming, block_index, total = 32, 0, sum(stage[-1] for stage in stages)
    for expansion, kernel, first_stride, outgoing, repeats in stages:
        for index in range(repeats):
            layers.append(MobileBlock(incoming, outgoing, expansion, kernel,
                                      first_stride if index == 0 else 1, .2 * block_index / total))
            incoming, block_index = outgoing, block_index + 1
    layers.append(conv_norm(incoming, 1280, 1, activation=nn.SiLU))
    return ImageClassifier(nn.Sequential(*layers), nn.AdaptiveAvgPool2d(1),
                           nn.Sequential(nn.Dropout(.2), nn.Linear(1280, classes)))


BUILDERS = {name: globals()[name] for name in ("lenet", "alexnet", "vgg16", "resnet18", "efficientnet_b0")}


def copy_components(manual, library):
    kinds = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
    left = [layer for layer in manual.modules() if isinstance(layer, kinds)]
    right = [layer for layer in library.modules() if isinstance(layer, kinds)]
    if len(left) != len(right):
        raise ValueError("component count changed; inspect this torchvision version")
    for target, source in zip(left, right):
        if type(target) is not type(source):
            raise ValueError("component order changed")
        target.load_state_dict(source.state_dict(), strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=BUILDERS, default="resnet18")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    shape = (1, 1, 32, 32) if args.family == "lenet" else (1, 3, 224, 224)
    if not args.compare:
        with torch.device("meta"):
            model = BUILDERS[args.family]()
            output = model(torch.empty(shape))
        print(args.family, sum(p.numel() for p in model.parameters()), tuple(output.shape))
        return
    if args.family == "lenet":
        raise ValueError("No torchvision LeNet counterpart; inspect the declared local variant")
    from torchvision.models import get_model
    torch.manual_seed(9)
    library = get_model(args.family, weights=None).eval()
    manual = BUILDERS[args.family]().eval()
    copy_components(manual, library)
    image = torch.randn(shape)
    with torch.inference_mode():
        expected, actual = library(image), manual(image)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    print(args.family, "matched-state max difference", float((actual - expected).abs().max()))


if __name__ == "__main__":
    main()
