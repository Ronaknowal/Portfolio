"""Optional, unexecuted checkpoint application; reads local files only.

Requirements: PyTorch, Transformers, Pillow, NumPy. See lesson §8.
Example: python inspect-pretrained-features.py --checkpoint ./dinov2-small --output features.npz photo1.png photo2.png
Prepare the official facebook/dinov2-small checkpoint and processor locally first.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('images', type=Path, nargs='+')
    args = parser.parse_args()
    processor = AutoImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True).eval()
    if model.config.model_type != 'dinov2':
        raise ValueError('This program expects the non-register DINOv2 interface.')
    patch_size = model.config.patch_size
    global_features, patch_features, grid_shapes = [], [], []
    with torch.inference_mode():
        for path in args.images:
            with Image.open(path) as source:
                image = source.convert('RGB')
                inputs = processor(images=image, return_tensors='pt')
            _, _, height, width = inputs['pixel_values'].shape
            if height % patch_size or width % patch_size:
                raise ValueError('Choose processor dimensions divisible by the checkpoint patch size.')
            tokens = model(**inputs).last_hidden_state[0]
            grid = (height // patch_size, width // patch_size)
            if len(tokens) != 1 + grid[0] * grid[1]:
                raise ValueError('Unexpected prefix or patch layout; inspect the checkpoint interface.')
            global_features.append(tokens[0].cpu().numpy())
            patch_features.append(tokens[1:].cpu().numpy())
            grid_shapes.append(grid)
    # The processor must return a common size for this compact stacked artifact.
    if len(set(grid_shapes)) != 1:
        raise ValueError('Use a common processor crop size before stacking feature grids.')
    global_array = np.stack(global_features)
    norms = np.linalg.norm(global_array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('A zero feature vector has no cosine direction.')
    normalized = global_array / norms
    np.savez_compressed(args.output, cls=global_array, patches=np.stack(patch_features),
                        cosine=normalized @ normalized.T, grid=np.array(grid_shapes),
                        source_paths=np.array([str(path) for path in args.images]))
    print('CLS shape:', global_array.shape)
    print('Patch shape:', np.stack(patch_features).shape)
    print('Image cosine matrix:', normalized @ normalized.T)


if __name__ == '__main__':
    main()
