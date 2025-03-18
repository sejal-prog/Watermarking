# :movie_camera: :seal: Video Seal: Open and Efficient Video Watermarking

Official implementation of [Video Seal](https://ai.meta.com/research/publications/video-seal-open-and-efficient-video-watermarking/).
Training and inference code for **image and video watermarking**, and **state-of-the-art open-sourced models**.

This repository includes pre-trained models, training code, inference code, and evaluation tools, all released under the MIT license, as well as baselines of state-of-the-art image watermarking models adapted for video watermarking (including MBRS, CIN, TrustMark, and WAM) allowing for free use, modification, and distribution of the code and models. 

[[`paper`](https://ai.meta.com/research/publications/video-seal-open-and-efficient-video-watermarking/)]
[[`arXiv`](https://arxiv.org/abs/2412.09492)]
[[`Colab`](https://colab.research.google.com/github/facebookresearch/videoseal/blob/main/notebooks/colab.ipynb)]
[[`Demo`](https://aidemos.meta.com/videoseal)]


## What's New

- March 2025: New image models, including 256-bit model with strong robustness and imperceptibility. Updates to the codebase for better performance and usability.
- December 2024: Initial release of Video Seal, including 96-bit model, baselines and video inference and training code.


## Quick start

```python
import torchvision
import videoseal
from videoseal.evals.metrics import bit_accuracy

# Load video and normalize to [0, 1]
video_path = "assets/videos/1.mp4"
video = torchvision.io.read_video(video_path, output_format="TCHW")
video = video.float() / 255.0

# Load the model
model = videoseal.load("videoseal")

# Video watermarking
outputs = model.embed(video, is_video=True, lowres_attenuation=True) # this will embed a random msg
video_w = outputs["imgs_w"] # the watermarked video
msgs = outputs["msgs"] # the embedded message

# Extract the watermark message
msg_extracted = model.extract_message(video_w, aggregation="avg", is_video=True)

# VideoSeal can do image Watermarking
img = video[0:1] # 1 x C x H x W
outputs = model.embed(img, is_video=False)
img_w = outputs["imgs_w"] # the watermarked image
msg_extracted = model.extract_message(imgs_w, aggregation="avg", is_video=False)
```



## Installation

### Requirements

Version of Python is 3.10 (pytorch > 2.3, torchvision 0.16.0, torchaudio 2.1.0, cuda 12.1).
Install pytorch:
```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Other dependencies:
```
pip install -r requirements.txt
```

For training, we also recommend using decord:
```
pip install decord
```
Note that there may be some issues with installing decord: https://github.com/dmlc/decord/issues/213
Everything should be working without decord for inference, but there may be issues for training in this case.

### Video Seal Models

#### Quick Model Loading
```python
# Automatically downloads and loads the default model (256-bit version)
model = videoseal.load("videoseal")
```

#### Available Models

- **Default Model (256-bit)**: 
  - Model name: `videoseal_1.0`
  - Download: [y_256b_img.pth](https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth)
  - Best balance of efficiency and robustness
  - Manual download:
    ```bash
    # For Linux/Windows:
    wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth -P ckpts/
    
    # For Mac:
    mkdir ckpts
    curl -o ckpts/y_256b_img.pth https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth
    ```

- **Legacy Model (96-bit)**: December 2024 version
  - Model name: `videoseal_0.0`
  - Download: [rgb_96b.pth](https://dl.fbaipublicfiles.com/videoseal/rgb_96b.pth)
  - More robust but more visible watermarks

Note: Video-optimized models (v1.0) will be released soon. For complete model checkpoints (with optimizer states and discriminator), see [docs/training.md](docs/training.md).


### Download the other models used as baselines

We do not own any third-party models, so you have to download them manually.
We provide a guide on how to download the models at [docs/baselines.md](docs/baselines.md).

### VMAF

We provide a guide on how to check and install VMAF at [docs/vmaf.md](docs/vmaf.md).






## Inference

### Notebooks

- [`notebooks/image_inference.ipynb`](notebooks/image_inference.ipynb)
- [`notebooks/video_inference.ipynb`](notebooks/video_inference.ipynb)
- [`notebooks/video_inference_streaming.ipynb`](notebooks/video_inference_streaming.ipynb): optimized for lower RAM usage

### Audio-visual watermarking

[`inference_av.py`](inference_av.py) 

To watermark both audio and video from a video file.
It loads the full video in memory, so it is not suitable for long videos.

Example:
```bash
python inference_av.py --input assets/videos/1.mp4 --output_dir outputs/
python inference_av.py --detect --input outputs/1.mp4
```

### Streaming embedding and extraction

[`inference_streaming.py`](inference_streaming.py) 

To watermark a video file in streaming.
It loads the video clips by clips, so it is suitable for long videos, even on laptops.

Example:
```bash
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/
```
Will output the watermarked video in `outputs/1.mp4` and the binary message in `outputs/1.txt`.


### Full evaluation

[`videoseal/evals/full.py`](videoseal/evals/full.py)

To run full evaluation of models and baselines.

Example to evaluate a trained model:
```bash
python -m videoseal.evals.full \
    --checkpoint /path/to/videoseal/checkpoint.pth \
```
or, to run a given baseline:
```bash
python -m videoseal.evals.full \
    --checkpoint baseline/wam \
``` 




## Training

We provide training code to reproduce our models or train your own models. This includes image and video training (we recommand training on image first, even if you wish to do video).
See [docs/training.md](docs/training.md) for detailed instructions on data preparation, training commands, and pre-trained model checkpoints.


## License

The model is licensed under an [MIT license](LICENSE).

## Contributing

See [contributing](.github/CONTRIBUTING.md) and the [code of conduct](.github/CODE_OF_CONDUCT.md).

## See Also

- [**AudioSeal**](https://github.com/facebookresearch/audioseal)
- [**Watermark-Anything**](https://github.com/facebookresearch/watermark-anything/)

## Maintainers and contributors

Pierre Fernandez, Hady Elsahar, Tomas Soucek, Sylvestre Rebuffi, Alex Mourachko

## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```bibtex
@article{fernandez2024video,
  title={Video Seal: Open and Efficient Video Watermarking},
  author={Fernandez, Pierre and Elsahar, Hady and Yalniz, I. Zeki and Mourachko, Alexandre},
  journal={arXiv preprint arXiv:2412.09492},
  year={2024}
}
```

