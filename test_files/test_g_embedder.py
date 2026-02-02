"""
Debug script to print internal dimensions of G-CNN Embedder
Shows how group dimension is folded into channels

Run from videoseal root:
    python test_g_embedder_debug.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder


def test_internal_dimensions():
    print("=" * 70)
    print(" G-CNN Embedder Internal Dimensions Debug")
    print("=" * 70)
    
    # Build model
    model = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),  # 2 levels: 32, 64 channels
        num_blocks=1,
        group_type="C4"
    )
    model.eval()
    
    gunet = model.gunet
    group_order = gunet.group_order
    
    print(f"\n{'='*70}")
    print(f" Model Configuration")
    print(f"{'='*70}")
    print(f"  Group type: C4 (order = {group_order})")
    print(f"  z_channels: {gunet.z_channels}")
    print(f"  z_channels_mults: {gunet.z_channels_mults}")
    print(f"  z_channels_list: {gunet.z_channels_list}")
    print(f"  Message hidden_size: {model.msg_processor.hidden_size}")
    
    # Create input
    B, C, H, W = 1, 3, 64, 64
    imgs = torch.rand(B, C, H, W)
    msgs = model.get_random_msg(B)
    
    # Preprocess
    imgs_preprocessed = model.preprocess(imgs)
    
    print(f"\n{'='*70}")
    print(f" Forward Pass - Shape at Each Stage")
    print(f"{'='*70}")
    
    print(f"\n[INPUT]")
    print(f"  imgs:                {imgs.shape}")
    print(f"  imgs (preprocessed): {imgs_preprocessed.shape}")
    print(f"  msgs:                {msgs.shape}")
    
    # ===== LIFT =====
    from escnn import nn as enn
    x = enn.GeometricTensor(imgs_preprocessed, gunet.in_type)
    print(f"\n[LIFT: RGB → Group Space]")
    print(f"  Before lift:         {x.tensor.shape}  (trivial repr: {C} channels)")
    
    x = gunet.lift(x)
    lifted_channels = x.tensor.shape[1]
    print(f"  After lift:          {x.tensor.shape}  (regular repr: {lifted_channels//group_order} channels × {group_order} rotations = {lifted_channels})")
    
    # ===== ENCODER =====
    print(f"\n[ENCODER]")
    x = gunet.inc(x)
    print(f"  After inc:           {x.tensor.shape}  ({x.tensor.shape[1]//group_order} ch × {group_order} rot)")
    
    hiddens = [x]
    for i, down in enumerate(gunet.downs):
        x = down(x)
        hiddens.append(x)
        ch = x.tensor.shape[1] // group_order
        print(f"  After down[{i}]:       {x.tensor.shape}  ({ch} ch × {group_order} rot, spatial: {x.tensor.shape[2]}×{x.tensor.shape[3]})")
    
    # ===== MESSAGE INJECTION =====
    print(f"\n[MESSAGE INJECTION]")
    print(f"  Before msg inject:   {x.tensor.shape}")
    x = gunet.msg_processor(x, msgs)
    ch = x.tensor.shape[1] // group_order
    print(f"  After msg inject:    {x.tensor.shape}  ({ch} ch × {group_order} rot)")
    print(f"  Added channels:      {model.msg_processor.hidden_size} (hidden_size) × {group_order} (replicated)")
    
    # ===== BOTTLENECK =====
    print(f"\n[BOTTLENECK]")
    x = gunet.bottleneck(x)
    ch = x.tensor.shape[1] // group_order
    print(f"  After bottleneck:    {x.tensor.shape}  ({ch} ch × {group_order} rot)")
    
    # ===== DECODER =====
    print(f"\n[DECODER]")
    for i, up in enumerate(gunet.ups):
        skip = hiddens.pop()
        
        x_tensor = x.tensor
        skip_tensor = skip.tensor * gunet.connect_scale
        concat_tensor = torch.cat([x_tensor, skip_tensor], dim=1)
        
        print(f"  Up[{i}] - x:            {x_tensor.shape}  ({x_tensor.shape[1]//group_order} ch × {group_order} rot)")
        print(f"  Up[{i}] - skip:         {skip_tensor.shape}  ({skip_tensor.shape[1]//group_order} ch × {group_order} rot)")
        print(f"  Up[{i}] - concat:       {concat_tensor.shape}  ({concat_tensor.shape[1]//group_order} ch × {group_order} rot)")
        
        x = enn.GeometricTensor(concat_tensor, gunet.up_in_types[i])
        x = up(x)
        ch = x.tensor.shape[1] // group_order
        print(f"  Up[{i}] - output:       {x.tensor.shape}  ({ch} ch × {group_order} rot, spatial: {x.tensor.shape[2]}×{x.tensor.shape[3]})")
        print()
    
    # ===== PROJECT =====
    print(f"[PROJECT: Group Space → RGB]")
    print(f"  Before proj:   {x.tensor.shape}  ({x.tensor.shape[1]//group_order} ch × {group_order} rot)")
    
    x = gunet.outc(x)
    print(f"  After proj:    {x.tensor.shape}  (projected to RGB)")
    print(f"  Note: Direct R2Conv projection, NO group pooling (preserves message signal)")
    
    out = x.tensor
    if gunet.last_tanh:
        out = torch.tanh(out)
    
    print(f"\n[OUTPUT]")
    print(f"  Final output:        {out.shape}")
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print(f" Summary: Group Dimension Flow")
    print(f"{'='*70}")
    print(f"""
    Input:     [B, 3, H, W]           → No group dimension
                    ↓ LIFT
    Lifted:    [B, 32×4=128, H, W]    → 32 channels × 4 rotations
                    ↓ ENCODER
    Encoded:   [B, 64×4=256, H/2, W/2] → 64 channels × 4 rotations  
                    ↓ MSG INJECT
    +Message:  [B, 128×4=512, H/2, W/2] → (64+64) channels × 4 rotations
                    ↓ DECODER
    Decoded:   [B, 32×4=128, H, W]    → 32 channels × 4 rotations
                    ↓ PROJECT (R2Conv, NO group pooling!)
    Output:    [B, 3, H, W]           → RGB image
    
    Key: Group dimension is FOLDED into channel dimension!
         [B, C×|G|, H, W] where |G|=4 for C4 group
         
    IMPORTANT: Embedder uses R2Conv to project to RGB, NOT GroupPooling!
    This preserves the message signal in the output.
    """)


if __name__ == "__main__":
    test_internal_dimensions()