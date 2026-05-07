import sys
sys.path.insert(0, '/home/sejal/Documents/Thesis/videoseal')
import torch
import torch.nn as nn

# 1. ConvNeXt-Tiny extractor (backbone + pixel_decoder)
from videoseal.modules.convnext import ConvNeXtV2
from videoseal.modules.pixel_decoder import PixelDecoder
from videoseal.models.extractor import ConvnextExtractor

convnext = ConvNeXtV2(depths=[3,3,9,3], dims=[96,192,384,768])
pixel_decoder = PixelDecoder(embed_dim=768, nbits=32, upscale_stages=[1], pixelwise=False, sigmoid_output=False)
convnext_extractor = ConvnextExtractor(convnext, pixel_decoder)

# 2. GCNN C4 extractor
from videoseal.models.g_extractor import build_g_extractor
gcnn_c4 = build_g_extractor(nbits=32, depths=[2,2,6,2], dims=[64,128,256,512], group_type='C4')

# 3. GCNN C8 extractor
gcnn_c8 = build_g_extractor(nbits=32, depths=[2,2,6,2], dims=[64,128,256,512], group_type='C8')

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_params_by_component(model):
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        print(f'    {name}: {total:>12,} ({total/1e6:.2f}M)')

for name, model in [
    ('ConvNeXt-Tiny Extractor', convnext_extractor),
    ('GCNN C4 Extractor', gcnn_c4),
    ('GCNN C8 Extractor', gcnn_c8),
]:
    total, trainable = count_params(model)
    print(f'{name}:')
    print(f'  Total params:     {total:>12,}  ({total/1e6:.2f}M)')
    print(f'  Trainable params: {trainable:>12,}  ({trainable/1e6:.2f}M)')
    print(f'  Components:')
    count_params_by_component(model)
    print()
