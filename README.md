# Deep Learning based Video Watermarking via Group Equivariant Convolutional Networks

> **Rotation-Invariant Video Watermarking using Group-Equivariant Convolutional Neural Networks**

This repository contains the implementation of my Master's thesis work on improving rotation robustness in video watermarking using Group-Equivariant CNNs (G-CNNs). Built upon [Meta's VideoSeal](https://github.com/facebookresearch/videoseal) framework.

## 🎯 Key Contribution

Traditional CNN-based watermarking systems struggle with rotation attacks because standard convolutions are not rotation-equivariant. This work introduces a **hybrid architecture** that combines:

- **Standard VideoSeal Embedder** — Full expressiveness for watermark embedding
- **G-ConvNeXt Extractor** — Group-equivariant ConvNeXt for rotation-invariant extraction

### Why Hybrid?

```
Original Image → [Embedder] → Watermarked → [ROTATION ATTACK] → [Extractor] → Message
                     ↑                                              ↑
              Sees ORIGINAL                                  Sees ROTATED
              (no equivariance needed)                       (needs G-CNN!)
```

**Key insight:** Rotation happens *after* embedding, *before* extraction. Only the extractor needs rotation invariance.

## 📊 Results

### Image Evaluation (COCO - 100 images)

| Model | PSNR | Identity | Rotate 30° | Rotate 45° | Rotate 90° |
|-------|------|----------|------------|------------|------------|
| VideoSeal Reference | 44.8 dB | 99.9% | 50.2% | 49.9% | 98.0% |
| Baseline (22K SA-1B) | 44.6 dB | 66.1% | 51.5% | 50.3% | 62.9% |
| **G-ConvNeXt C4** | 44.4 dB | 69.6% | 61.2% | 54.6% | 69.4% |
| **G-ConvNeXt C8** | 44.2 dB | 68.4% | 59.7% | 54.3% | 68.1% |

### Video Evaluation (SA-V - 155 videos)

| Model | PSNR | Identity | H264 | H265 |
|-------|------|----------|------|------|
| G-ConvNeXt C4 | 45.1 dB | 67.5% | 57.1% | 57.7% |
| G-ConvNeXt C8 | 43.6 dB | 61.7% | 54.4% | 55.1% |

## 🏗️ Architecture

### G-ConvNeXt Extractor

The extractor uses group-equivariant convolutions from the [escnn](https://github.com/QUVA-Lab/escnn) library:

1. **Lifting Convolution** — Maps input from Z² to group space G × Z²
2. **Group Convolutions** — Preserves equivariance through the network
3. **Group Pooling** — Converts equivariant → invariant features for bit prediction

Supported groups:
- **C4**: 4-fold rotation symmetry (0°, 90°, 180°, 270°)
- **C8**: 8-fold rotation symmetry (0°, 45°, 90°, ..., 315°)

### Parameter Count

| Model | Parameters |
|-------|------------|
| ConvNeXt-tiny (baseline) | ~3.98M |
| G-ConvNeXt C4 | ~4.7M |
| G-ConvNeXt C8 | ~9.5M |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sejal-prog/Watermarking.git
cd Watermarking

# Create conda environment
conda create -n gcnn_env python=3.10
conda activate gcnn_env

# Install PyTorch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
pip install escnn  # For group-equivariant convolutions
```

### Training

#### Image Training (G-ConvNeXt C4)

```bash
python train.py \
    --image_dataset sa-1b \
    --video_dataset none \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --img_size 256 \
    --extractor_model g_convnext \
    --embedder_model unet_small2_yuv_quant \
    --nbits 128 \
    --epochs 601 \
    --optimizer AdamW,lr=5e-4 \
    --lambda_dec 1.0 \
    --lambda_d 0.1 \
    --lambda_i 0.1 \
    --perceptual_loss yuv \
    --augmentation_config configs/all_augs.yaml
```

**Note:** Set `group_type: C4` or `group_type: C8` in `configs/extractor.yaml`

#### Video Fine-tuning

```bash
python train.py \
    --image_dataset sa-1b \
    --video_dataset sav \
    --prop_img_vid 0.0 \
    --video_start 601 \
    --frames_per_clip 16 \
    --batch_size_video 1 \
    --extractor_model g_convnext \
    --embedder_model unet_small2_yuv_quant \
    --nbits 128 \
    --epochs 801 \
    --optimizer AdamW,lr=5e-6 \
    --scaling_w 0.2
```

### Evaluation

```bash
python -m videoseal.evals.full \
    --checkpoint /path/to/checkpoint.pth \
    --dataset coco \
    --is_video false \
    --num_samples 100 \
    --output_dir outputs/
```

## 📁 Project Structure

```
├── videoseal/
│   ├── models/
│   │   ├── extractor.py          # Base extractor classes
│   │   └── g_extractor.py        # G-ConvNeXt extractor
│   ├── modules/
│   │   ├── g_convnext.py         # G-ConvNeXt blocks and stages
│   │   └── g_common.py           # Common G-CNN utilities
│   └── evals/
│       └── full.py               # Evaluation script
├── configs/
│   ├── extractor.yaml            # Extractor config (set group_type here)
│   └── all_augs.yaml             # Augmentation config
├── train.py                      # Training script
└── count_params.py               # Parameter counting utility
```

## 🔧 Configuration

### Setting Group Type

Edit `configs/extractor.yaml`:

```yaml
g_convnext:
  group_type: C4  # Options: C4, C8
  encoder:
    depths: [2, 2, 2]
    dims: [32, 64, 128]
```

## 📚 References

- [VideoSeal: Open and Efficient Video Watermarking](https://arxiv.org/abs/2412.09492)
- [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
- [escnn: E(2)-Equivariant Steerable CNNs](https://github.com/QUVA-Lab/escnn)

## 📝 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{gcnn_watermarking_2026,
  title={Rotation-Invariant Video Watermarking using Group-Equivariant CNNs},
  author={Sejal},
  year={2026},
  school={Your University}
}

@article{fernandez2024video,
  title={Video Seal: Open and Efficient Video Watermarking},
  author={Fernandez, Pierre and Elsahar, Hady and Yalniz, I. Zeki and Mourachko, Alexandre},
  journal={arXiv preprint arXiv:2412.09492},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Meta AI Research](https://github.com/facebookresearch) for the VideoSeal framework
- [QUVA Lab](https://github.com/QUVA-Lab) for the escnn library
- My thesis supervisors and colleagues
