
import torch
from videoseal import load

# Load model
model = load("videoseal")

# Check embedder architecture
print("=== EMBEDDER ARCHITECTURE ===")
print(model.embedder)

# Check the actual forward pass
dummy_img = torch.randn(1, 3, 256, 256)
dummy_msg = torch.randint(0, 2, (1, 256)).float()

print("\n=== TRACING THROUGH EMBEDDING ===")
print(f"Input image shape: {dummy_img.shape}")
print(f"Input message shape: {dummy_msg.shape}")

with torch.no_grad():
    # Preprocess image
    imgs_preprocessed = model.embedder.preprocess(dummy_img)
    print(f"After preprocessing: {imgs_preprocessed.shape}")
    
    # Process message with image shape
    msgs_processed = model.embedder.msg_processor(imgs_preprocessed, dummy_msg)
    print(f"Message after processing: {msgs_processed.shape}")
    
    # Check spatial distribution
    if len(msgs_processed.shape) == 2:
        print("\n❌ Message is GLOBAL (not spatially distributed)")
    elif len(msgs_processed.shape) == 4:
        print(f"\n✅ Message is SPATIALLY DISTRIBUTED!")
        print(f"   Spatial resolution: {msgs_processed.shape[2]} x {msgs_processed.shape[3]}")
        print(f"   Channels: {msgs_processed.shape[1]}")
        
        # Check if it's repeated per pixel
        if msgs_processed.shape[2] == dummy_img.shape[2]:
            print(f"   Message is PER-PIXEL embedded!")
        else:
            print(f"   Message is embedded at {msgs_processed.shape[2]}x{msgs_processed.shape[3]} grid")
            print(f"   Downsampling factor: {dummy_img.shape[2] // msgs_processed.shape[2]}")

# Now let's see the actual source
print("\n=== MESSAGE PROCESSOR SOURCE CODE ===")
import inspect
try:
    print(inspect.getsource(model.embedder.msg_processor.forward))
except:
    print("Could not get source, checking file directly...")
    with open('videoseal/modules/msg_processor.py', 'r') as f:
        print(f.read())

#code by claude.ai