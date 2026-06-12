import torch
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.models.deepliif_stainer import DeepLIIFStainer

def test_deepliif():
    print("Initializing DeepLIIF Stainer...")
    stainer = DeepLIIFStainer(weights_path='deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth')
    
    # Create a dummy image: 1 batch, 3 channels, 512x512
    # We make it a gradient from blue to red so we can see how colors are processed
    dummy_img = torch.zeros(1, 3, 512, 512)
    for i in range(512):
        dummy_img[0, 0, :, i] = i / 512.0  # Red gradient horizontal
        dummy_img[0, 2, i, :] = i / 512.0  # Blue gradient vertical
    
    # Normalize to [-1, 1] as expected by the model
    dummy_img = dummy_img * 2.0 - 1.0

    print("Running extraction...")
    with torch.no_grad():
        # Let's bypass the mean collapse temporarily to get the raw output
        model = stainer._load_model('cpu')
        out = model(dummy_img)
        raw_output = out[:, 0:3, :, :]
        
        # Now apply our collapse logic
        collapsed_output = raw_output.mean(dim=1, keepdim=True)

    print(f"Raw DeepLIIF Output Shape: {raw_output.shape}")
    print(f"Collapsed Output Shape: {collapsed_output.shape}")

    # Convert to displayable format [0, 1]
    raw_img = (raw_output[0] + 1) / 2.0
    collapsed_img = (collapsed_output[0] + 1) / 2.0

    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # RGB display (permute to H, W, C)
    axes[0].imshow(raw_img.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title(f"Raw Output (RGB)\nShape: {raw_output.shape}")
    axes[0].axis('off')
    
    # Grayscale display
    axes[1].imshow(collapsed_img[0].clamp(0, 1).numpy(), cmap='gray')
    axes[1].set_title(f"Collapsed Output (Grayscale)\nShape: {collapsed_output.shape}")
    axes[1].axis('off')

    plt.savefig('deepliif_test_output.png', bbox_inches='tight')
    print("Saved visualization to deepliif_test_output.png")

if __name__ == '__main__':
    test_deepliif()
