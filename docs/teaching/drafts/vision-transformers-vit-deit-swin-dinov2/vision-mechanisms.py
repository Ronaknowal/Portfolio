"""Executed teaching mechanisms: patches, shifted windows, DINO and geometry.

No training occurs here. NumPy and PyTorch are the only dependencies.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def partition_windows(grid, window):
    batch, height, width, channels = grid.shape
    assert height % window == width % window == 0
    return grid.reshape(batch, height // window, window, width // window, window, channels).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)


def unpartition_windows(windows, window, height, width):
    batch = len(windows) // ((height // window) * (width // window))
    return windows.reshape(batch, height // window, width // window, window, window, -1).permute(
        0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class ShiftedWindowAttention(nn.Module):
    """Readable BHWC attention with actual relative bias and boundary/padding masks.

    Explicit conceptual region IDs prioritize inspection over a fused implementation.
    ``wrap=True`` deliberately removes the cyclic boundary mask for diagnosis only.
    """
    def __init__(self, width=4, heads=2, window=3, shift=1):
        super().__init__()
        assert width % heads == 0 and 0 <= shift < window
        self.width, self.heads, self.window, self.shift = width, heads, window, shift
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2 * window - 1) ** 2))

    def forward(self, grid, wrap=False):
        batch, height, width, channels = grid.shape
        window, shift = self.window, self.shift
        padded_height = ((height + window - 1) // window) * window
        padded_width = ((width + window - 1) // window) * window
        padded = F.pad(grid, (0, 0, 0, padded_width - width, 0, padded_height - height))
        row, column = torch.meshgrid(torch.arange(padded_height, device=grid.device),
                                     torch.arange(padded_width, device=grid.device), indexing='ij')
        valid = (row < height) & (column < width)
        region = torch.stack(((row - shift) // window, (column - shift) // window), -1)
        rolled = torch.roll(padded, (-shift, -shift), (1, 2))
        tokens = partition_windows(rolled, window)
        regions = partition_windows(torch.roll(region[None], (-shift, -shift), (1, 2)), window)
        validity = partition_windows(torch.roll(valid[None, ..., None], (-shift, -shift), (1, 2)), window)[..., 0]
        allowed = (regions[:, :, None] == regions[:, None, :]).all(-1)
        if wrap:
            allowed = torch.ones_like(allowed)
        allowed = allowed & validity[:, None, :]
        # Invalid query rows are discarded; give them self to avoid all-masked softmax.
        identity = torch.eye(window * window, device=grid.device, dtype=torch.bool)[None]
        allowed = torch.where(validity[:, :, None], allowed, identity)
        allowed = allowed.repeat(batch, 1, 1)
        query, key, value = self.qkv(tokens).reshape(
            len(tokens), window * window, 3, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        local_row, local_column = torch.meshgrid(torch.arange(window, device=grid.device),
                                                  torch.arange(window, device=grid.device), indexing='ij')
        coordinates = torch.stack((local_row.flatten(), local_column.flatten()), -1)
        offset = coordinates[:, None] - coordinates[None, :]
        bias_index = (offset[..., 0] + window - 1) * (2 * window - 1) + offset[..., 1] + window - 1
        scores = query @ key.transpose(-1, -2) / (channels // self.heads) ** .5
        scores = scores + self.relative_bias[:, bias_index][None]
        weights = scores.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
        mixed = (weights @ value).transpose(1, 2).reshape(len(tokens), window * window, channels)
        mixed = self.output(mixed)
        merged = unpartition_windows(mixed, window, padded_height, padded_width)
        return torch.roll(merged, (shift, shift), (1, 2))[:, :height, :width]


class SwinTeachingBlock(nn.Module):
    def __init__(self, width=4, heads=2, window=3, shift=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = ShiftedWindowAttention(width, heads, window, shift)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, grid):
        grid = grid + self.attention(self.norm1(grid))
        return grid + self.ffn(self.norm2(grid))


class PatchMerge(nn.Module):
    """Swin V1 order: concatenate TL, BL, TR, BR; LayerNorm4C; linear4C→2C."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(4 * channels)
        self.reduce = nn.Linear(4 * channels, 2 * channels, bias=False)

    def forward(self, grid):
        height, width = grid.shape[1:3]
        grid = F.pad(grid, (0, 0, 0, width % 2, 0, height % 2))
        parts = [grid[:, 0::2, 0::2], grid[:, 1::2, 0::2],
                 grid[:, 0::2, 1::2], grid[:, 1::2, 1::2]]
        return self.reduce(self.norm(torch.cat(parts, -1)))


def resize_positions(position, old_grid, new_grid, prefix_count):
    batch, sequence, width = position.shape
    assert batch == 1 and sequence == prefix_count + old_grid[0] * old_grid[1]
    if old_grid == new_grid:
        return position.clone()
    prefix = position[:, :prefix_count]
    grid = position[:, prefix_count:].reshape(1, *old_grid, width).permute(0, 3, 1, 2)
    resized = F.interpolate(grid.float(), size=new_grid, mode='bicubic', align_corners=False)
    return torch.cat((prefix, resized.to(position.dtype).flatten(2).transpose(1, 2)), 1)


def dino_cross_view_loss(student_logits, teacher_logits, center, student_temperature=.5, teacher_temperature=.2):
    """Two views, shape (2,batch,prototypes); no same-view pairs, detached teacher."""
    assert student_logits.shape == teacher_logits.shape and len(student_logits) == 2
    assert min(student_temperature, teacher_temperature) > 0
    target = ((teacher_logits.detach() - center) / teacher_temperature).softmax(-1)
    prediction = (student_logits / student_temperature).log_softmax(-1)
    return -.5 * ((target[0] * prediction[1]).sum(-1).mean() +
                 (target[1] * prediction[0]).sum(-1).mean())


@torch.no_grad()
def update_teacher(student, teacher, center, teacher_logits, momentum=.9, center_momentum=.9):
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters(), strict=True):
        teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1 - momentum)
    center.mul_(center_momentum).add_(teacher_logits.mean(dim=(0, 1)), alpha=1 - center_momentum)


def region_matrix(height, width, window, shift):
    coordinates = np.array([(row, col) for row in range(height) for col in range(width)])
    regions = (coordinates - shift) // window
    return (regions[:, None] == regions[None, :]).all(-1)


def mean_matrix(height, width, window, shift):
    allowed = region_matrix(height, width, window, shift).astype(float)
    return allowed / allowed.sum(1, keepdims=True)


def count_macs(image_height, image_width, patch, channels, width, blocks, classes=1000, prefixes=1, ratio=4):
    assert image_height % patch == image_width % patch == 0
    patches = (image_height // patch) * (image_width // patch)
    sequence = patches + prefixes
    patch_macs = patches * patch * patch * channels * width
    projection = 4 * sequence * width * width
    ffn = 2 * ratio * sequence * width * width
    pairs = 2 * sequence * sequence * width
    return {'patches': patches, 'sequence': sequence, 'projection': projection,
            'ffn': ffn, 'pairs': pairs, 'total': patch_macs + blocks * (projection + ffn + pairs) + width * classes,
            'pair_fraction_block': pairs / (projection + ffn + pairs)}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(419)
    image = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    weight = torch.tensor([[1., 0., 0., 1.], [0., 1., -1., 0.]], dtype=torch.float64)
    bias = torch.tensor([.5, 1.], dtype=torch.float64)
    unfolded = F.unfold(image, 2, stride=2).transpose(1, 2)
    linear = unfolded @ weight.T + bias
    conv = F.conv2d(image, weight.reshape(2, 1, 2, 2), bias, stride=2).flatten(2).transpose(1, 2)
    assert torch.equal(linear, conv)
    results = {'patches': unfolded.tolist(), 'embeddings': linear.tolist(), 'patch_conv_error': float((linear-conv).abs().max())}
    for size, query in [(4, (1, 1)), (6, (2, 2)), (6, (0, 0))]:
        first, second = mean_matrix(size, size, 2, 0), mean_matrix(size, size, 2, 1)
        influence = (second @ first)[query[0] * size + query[1]]
        results[f'reach_{size}_{query[0]}_{query[1]}'] = {
            'source_ids': np.flatnonzero(influence).tolist(), 'weights': influence.tolist()}
    # Independent explicit full-grid reference for nonzero relative bias, padding and reverse shift.
    module = ShiftedWindowAttention().double()
    with torch.no_grad(): module.relative_bias.copy_(torch.randn_like(module.relative_bias) * .2)
    grid = torch.randn(1, 5, 7, 4, dtype=torch.float64, requires_grad=True)
    output = module(grid)
    reference = []
    projected = module.qkv(grid).reshape(35, 3, 2, 2).permute(1, 2, 0, 3)
    query, key, value = projected
    coords = [(r, c) for r in range(5) for c in range(7)]
    for i, (row, column) in enumerate(coords):
        indices = [j for j, (r, c) in enumerate(coords) if (r-1)//3 == (row-1)//3 and (c-1)//3 == (column-1)//3]
        score = (query[:, i, None] * key[:, indices]).sum(-1) / 2 ** .5
        offsets = [(row-coords[j][0]+2)*5 + column-coords[j][1]+2 for j in indices]
        score = score + module.relative_bias[:, offsets]
        reference.append((score.softmax(-1)[..., None] * value[:, indices]).sum(1).reshape(4))
    reference = module.output(torch.stack(reference)).reshape(1, 5, 7, 4)
    error = float((output-reference).abs().max().detach())
    assert error < 1e-12
    output.square().sum().backward()
    assert torch.isfinite(grid.grad).all() and module.relative_bias.grad.abs().max() > 0
    results['window_reference_error'] = error
    results['relative_bias_gradient_max'] = module.relative_bias.grad.abs().max().item()
    with torch.no_grad():
        boundary = torch.zeros(1, 6, 6, 1, dtype=torch.float64)
        average = ShiftedWindowAttention(1, 1, 2, 1).double()
        for parameter in average.parameters(): parameter.zero_()
        average.qkv.weight[2, 0] = 1
        average.output.weight[0, 0] = 1
        edited = boundary.clone();edited[0, 5, 5, 0] = 16
        results['wrap_contrast'] = {'masked_corner': average(edited)[0, 0, 0, 0].item(),
                                   'unmasked_corner': average(edited, wrap=True)[0, 0, 0, 0].item()}
    positions = torch.arange(8*2, dtype=torch.float64).reshape(1, 8, 2)
    resized = resize_positions(positions, (2, 3), (3, 4), 2)
    assert torch.equal(resized[:, :2], positions[:, :2])
    assert torch.equal(resize_positions(positions, (2, 3), (2, 3), 2), positions)
    results['position_resize'] = resized.tolist()
    for name, teacher, center, student, teacher_temperature in [
        ('worked', [.4,.1,-.2], [.1,0,-.1], [.1,.2,-.1], .2),
        ('fresh', [.2,-.1,.4], [.1,.1,0], [0,.2,-.2], .25),
        ('changed', [.2,-.1,0], [.1,.1,0], [0,.2,-.2], .25),
    ]:
        t=torch.tensor(teacher,dtype=torch.float64);c=torch.tensor(center,dtype=torch.float64)
        s=torch.tensor(student,dtype=torch.float64,requires_grad=True)
        target=((t-c)/teacher_temperature).softmax(-1)
        probability=(s/.5).softmax(-1);loss=-(target*(s/.5).log_softmax(-1)).sum();loss.backward()
        assert torch.allclose(s.grad,(probability.detach()-target)/.5)
        assert torch.allclose(target,((t+3-c)/teacher_temperature).softmax(-1))
        results['dino_'+name]={'target':target.tolist(),'student':probability.tolist(),'loss':loss.item(),'gradient':s.grad.tolist()}
    teacher=torch.tensor([[[.4,.1,-.2]],[[.2,.3,-.1]]],dtype=torch.float64,requires_grad=True)
    student=torch.tensor([[[.1,.2,-.1]],[[0,.2,.1]]],dtype=torch.float64,requires_grad=True)
    cross_loss=dino_cross_view_loss(student,teacher,torch.zeros(3));cross_loss.backward()
    assert teacher.grad is None and student.grad.abs().max()>0
    results['cross_view_loss']=cross_loss.item()
    features=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    rotation=np.array([[0.,-1.],[1.,0.]])
    edited=features.copy();edited[2]=[0,-1]
    results['gram']={'teacher':features.tolist(),'gram':(features@features.T).tolist(),
                     'rotation_loss':float(np.square((features@rotation)@(features@rotation).T-features@features.T).sum()),
                     'edited_loss':float(np.square(edited@edited.T-features@features.T).sum())}
    results['macs']={str(resolution):count_macs(resolution,resolution,16,3,768,12) for resolution in [224,384,512,1024]}
    results['stem_parameters']=16*16*3*768 + 768 + 768 + 197*768
    # A predeclared finite search locates an explanatory counterexample, not an empirical benchmark.
    candidates=[np.array(x,dtype=float) for x in [(a,b,0) for a in range(-3,4) for b in range(-3,4)]]
    for first in candidates:
        for second in candidates:
            logits=(first+second)/2
            p1=np.exp(first-first.max());p1/=p1.sum()
            p2=np.exp(second-second.max());p2/=p2.sum();probability=(p1+p2)/2
            if logits.argmax()!=probability.argmax() and np.sort(logits)[-1]-np.sort(logits)[-2]>.1 and np.sort(probability)[-1]-np.sort(probability)[-2]>.01:
                results['fusion_counterexample']={'first_logits':first.tolist(),'second_logits':second.tolist(),
                                                 'mean_logits':logits.tolist(),'mean_probabilities':probability.tolist()}
                break
        if 'fusion_counterexample' in results:break
    (ROOT/'mechanism-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if not k.startswith('reach_') and k not in ['position_resize','macs']},indent=2))


if __name__ == '__main__':
    main()
