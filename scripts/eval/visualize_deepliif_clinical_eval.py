import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import random

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
    
    # DeepLIIF Control Parameters (Matching Web UI Sliders)
    parser.add_argument('--size_thresh', type=str, default='default', help="Minimum size gating for nuclei (int or 'default')")
    parser.add_argument('--size_thresh_upper', type=str, default='None', help="Maximum size gating for nuclei (int or 'None')")
    parser.add_argument('--seg_thresh', type=int, default=130, help="Segmentation intensity threshold (0-255)")
    parser.add_argument('--marker_thresh', type=str, default='default', help="Marker intensity threshold for positivity (int or 'default')")
    parser.add_argument('--resolution', type=str, default='40x', help="Magnification resolution (10x, 20x, 40x)")
    parser.add_argument('--save_modalities', action='store_true', help="Save all 4 intermediate modalities (H, mpH, mpDAB, Lap2)")
    parser.add_argument('--cpu', action='store_true', help="Force execution on CPU instead of GPU")
    parser.add_argument('--seed', type=int, default=None, help="Use a specific random seed for reproducible image selection. If none is passed, a random seed is generated and printed.")
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"[INFO] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[INFO] Loading DeepLIIF Model from {args.deepliif_weights}...")
    deepliif_stainer = DeepLIIFStainer(weights_path=args.deepliif_weights)
    
    image_paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    # Random Seed Logic
    seed = args.seed if args.seed is not None else random.randint(1, 999999)
    print(f"[INFO] Using Random Seed: {seed}")
    random.seed(seed)
    random.shuffle(image_paths)
    
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
        img_tensor = torch.from_numpy(img_01).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        
        try:
            modalities = deepliif_stainer.extract_all_modalities(img_tensor)
        except RuntimeError as e:
            print(f"[ERROR] Failed to extract modalities: {e}")
            print("Make sure your --deepliif_weights points to the full DeepLIIF_Latest_Model directory containing G1 through G55 models!")
            return
            
        seg_mask = modalities['Segmentation']
        marker_mask = modalities['mpIHC_Ki67'] # G4 is Ki67 marker
            
        seg_np = ((seg_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        marker_np = ((marker_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        
        # Parse arguments (allowing for 'default' and 'None' strings)
        size_thresh = None if args.size_thresh == 'None' else (args.size_thresh if args.size_thresh == 'default' else int(args.size_thresh))
        size_thresh_upper = None if args.size_thresh_upper == 'None' else int(args.size_thresh_upper)
        marker_thresh = None if args.marker_thresh == 'None' else (args.marker_thresh if args.marker_thresh == 'default' else int(args.marker_thresh))
        
        # Use DeepLIIF's official post-processing to get smooth contours and accurate splits
        overlay, refined, scoring = compute_final_results(
            orig=img_np,
            seg=seg_np,
            marker=marker_np,
            resolution=args.resolution,
            size_thresh=size_thresh,
            size_thresh_upper=size_thresh_upper,
            seg_thresh=args.seg_thresh,
            marker_thresh=marker_thresh
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
        
        # Save the original image for easy comparison!
        orig_path = Path(args.output_dir) / f"{img_path.stem}_ORIGINAL.jpg"
        Image.fromarray(img_np).save(orig_path)
        
        # Save intermediate modalities if requested
        if args.save_modalities:
            for mod_name in ['Hematoxylin', 'mpIHC_DAPI', 'mpIHC_Lap2', 'mpIHC_Ki67']:
                mod_tensor = modalities[mod_name]
                mod_np = ((mod_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                mod_path = Path(args.output_dir) / f"{img_path.stem}_{mod_name}.jpg"
                Image.fromarray(mod_np).save(mod_path)
            
    print(f"\n[SUCCESS] DeepLIIF Visualizations saved to {args.output_dir}")
    print("Please inspect the images. RED contours = Positive, BLUE contours = Negative.")

if __name__ == "__main__":
    main()
