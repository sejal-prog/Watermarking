# Group Equivariant CNN for Rotation-Invariant Watermarking

## Motivation

Standard CNN-based watermarking models struggle with rotated images because
convolutions are not inherently rotation-aware. The typical workaround is
heavy rotation augmentation during training, which is expensive and only
approximates true invariance.

This module replaces standard convolutions with **group equivariant
convolutions** using the [escnn](https://github.com/QUVA-Lab/escnn) library.
The result is an extractor that is **mathematically invariant** to discrete
rotations, not just empirically robust to them.

## Key Concepts

### Group Equivariance vs Invariance

- **Equivariant** (embedder): rotating the input rotates the output in the
  same way. The watermark pattern follows the image orientation.
- **Invariant** (extractor): rotating the input does not change the output.
  The same message bits are decoded regardless of image rotation.

### Supported Groups

| Group | Symmetries                   | Use Case                      |
|-------|------------------------------|-------------------------------|
| C4    | 0°, 90°, 180°, 270°         | Standard rotation robustness  |
| C8    | 0°, 45°, 90°, ... , 315°    | Fine-grained rotation         |
| D4    | C4 rotations + reflections   | Rotation and flip robustness  |

The group is selected via the `group_type` parameter in the YAML configs.

## Architecture

### Extractor (G-ConvNeXt)