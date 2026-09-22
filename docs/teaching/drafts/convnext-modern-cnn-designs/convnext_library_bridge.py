"""Match the local V1 block to Torchvision; optionally use a pretrained model.

The block probe needs compatible torch/torchvision. --image PATH additionally
needs Pillow and downloads the explicit ImageNet checkpoint if it is not cached.
The image is an inference example, not an evaluation dataset or accuracy claim.
"""
import argparse
from pathlib import Path
import runpy
import torch


def compare_block():
    from torchvision.models.convnext import CNBlock

    namespace = runpy.run_path(str(Path(__file__).with_name("convnext-blocks.py")))
    torch.manual_seed(3)
    manual = namespace["ConvNeXtBlock"](8, version=1, drop_probability=0).double()
    native = CNBlock(8, layer_scale=1e-6, stochastic_depth_prob=0).double()
    # Logical NHWC LayerNorm and MLP correspond directly; layer-scale storage differs.
    for local, library in ((manual.spatial, native.block[0]), (manual.norm, native.block[2]),
                           (manual.expand, native.block[3]), (manual.project, native.block[5])):
        local.load_state_dict(library.state_dict())
    with torch.no_grad():
        manual.layer_scale.copy_(native.layer_scale[:, 0, 0])
    left = torch.randn(2, 8, 5, 7, dtype=torch.float64, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    manual.eval()
    native.eval()
    actual, expected = manual(left), native(right)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(manual.layer_scale.grad, native.layer_scale.grad[:, 0, 0],
                               atol=1e-12, rtol=1e-12)
    for local, library in ((manual.spatial, native.block[0]), (manual.norm, native.block[2]),
                           (manual.expand, native.block[3]), (manual.project, native.block[5])):
        for local_parameter, native_parameter in zip(local.parameters(), library.parameters()):
            torch.testing.assert_close(local_parameter.grad, native_parameter.grad, atol=1e-12, rtol=1e-12)
    print("V1 block values, input gradients and parameter gradients agree")


def classify_image(path):
    from PIL import Image
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights).eval()
    with Image.open(path) as image:
        inputs = weights.transforms()(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(-1)[0]
    values, indices = probabilities.topk(5)
    print([(weights.meta["categories"][int(index)], float(value))
           for value, index in zip(values, indices)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    arguments = parser.parse_args()
    compare_block()
    if arguments.image is not None:
        classify_image(arguments.image)
