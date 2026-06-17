import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.deepliif_stainer import DeepLIIFStainer
from deepliif.postprocessing import compute_final_results

def main():
    parser = argparse.ArgumentParser(description="Visualize DeepLIIF Clinical Evaluation")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing IHC images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save visual overlays")
    parser.add_argument('--deepliif_weights', type=str, required=True, help="Path to DeepLIIF_Latest_Model directory containing G1 to G55 models")
    parser.add_argument('--max_images', type=int, default=10, help="Max images to process")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[INFO] Loading DeepLIIF Model from {args.deepliif_weights}...")
    deepliif_stainer = DeepLIIFStainer(weights_path=args.deepliif_weights)
    
    image_paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    image_paths = image_paths[:args.max_images]
    
    print(f"[INFO] Found {len(image_paths)} images. Processing with DeepLIIF Segmentation...")
    
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {img_path.name}")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Normalize to [0, 1] for DeepLIIF
        img_01 = img_np.astype(np.float32) / 255.0
        
        # DeepLIIF expects [1, 3, H, W] in [-1, 1]
        img_tensor = torch.from_numpy(img_01).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        
        try:
            seg_mask = deepliif_stainer.extract_segmentation(img_tensor)
        except RuntimeError as e:
            print(f"[ERROR] Failed to extract segmentation: {e}")
            print("Make sure your --deepliif_weights points to the full DeepLIIF_Latest_Model directory containing G1 through G55 models!")
            return
            
        seg_np = ((seg_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        
        # Use DeepLIIF's official post-processing to get smooth contours and accurate splits
        overlay, refined, scoring = compute_final_results(
            orig=img_np,
            seg=seg_np,
            marker=None,  # We don't need the Ki67 modality mask for basic positive/negative
            resolution='40x', # Assume 40x for MIST dataset
            size_thresh='default',
            marker_thresh=None
        )
        
        li = scoring['percent_pos']
        total_count = scoring['num_total']
        pos_count = scoring['num_pos']
        
        # The 'overlay' is already a beautifully drawn RGB image!
        # Just write the text on top
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        text = f"DeepLIIF | Total: {total_count} | Pos: {pos_count} | LI: {li:.1f}%"
        cv2.rectangle(overlay_bgr, (0, 0), (overlay_bgr.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(overlay_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out_name = f"{img_path.stem}_DeepLIIF_LI{li:.1f}.jpg"
        out_path = Path(args.output_dir) / out_name
        
        # Convert back to RGB for saving overlay
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(out_path)
        
        # Save the RAW segmentation mask to see what DeepLIIF generated!
        raw_seg_path = Path(args.output_dir) / f"{img_path.stem}_RAW_SEG.jpg"
        Image.fromarray(seg_np).save(raw_seg_path)
            
    print(f"\n[SUCCESS] DeepLIIF Visualizations saved to {args.output_dir}")
    print("Please inspect the images. RED contours = Positive, BLUE contours = Negative.")

if __name__ == "__main__":
    main()
