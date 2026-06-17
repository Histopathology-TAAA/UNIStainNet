import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Import the existing evaluator logic
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.ki67_evaluator import Ki67ClinicalEvaluator

def draw_visual_evaluation(img_np, labels, dab_map, threshold):
    """
    img_np: (H, W, 3) RGB image in [0, 255]
    labels: (H, W) int32 instance mask from StarDist
    dab_map: (H, W) float DAB optical density map
    threshold: float
    """
    # Create a copy for drawing (convert RGB to BGR for OpenCV)
    overlay = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids != 0]
    
    pos_count = 0
    total_count = len(unique_ids)
    
    for cell_id in unique_ids:
        # Create binary mask for this specific cell
        mask = (labels == cell_id).astype(np.uint8)
        
        # Calculate mean DAB
        mean_dab = dab_map[mask == 1].mean()
        
        # Determine color
        if mean_dab > threshold:
            color = (0, 0, 255)  # Red in BGR (Positive)
            pos_count += 1
        else:
            color = (255, 0, 0)  # Blue in BGR (Negative)
            
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
        
    li = (pos_count / total_count * 100) if total_count > 0 else 0.0
    
    # Add text banner
    text = f"Total: {total_count} | Pos: {pos_count} | LI: {li:.1f}% | Thr: {threshold}"
    
    # Draw a black rectangle background for the text
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert back to RGB for saving/displaying
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), li

def main():
    parser = argparse.ArgumentParser(description="Visualize Ki67 Clinical Evaluation")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing IHC images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save visual overlays")
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.10, 0.15, 0.20], help="List of DAB thresholds to test")
    parser.add_argument('--max_images', type=int, default=10, help="Max images to process")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the evaluator
    # Initialize the evaluator
    print("[INFO] Loading StarDist Model...")
    evaluator = Ki67ClinicalEvaluator(dab_threshold=0.15) # Thr doesn't matter here, we use it manually
    star_model = evaluator._load_star_model()
    
    image_paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    image_paths = image_paths[:args.max_images]
    
    print(f"[INFO] Found {len(image_paths)} images. Processing with thresholds: {args.thresholds}...")
    
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {img_path.name}")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Normalize to [0, 1] for StarDist and DAB
        img_01 = img_np.astype(np.float32) / 255.0
        
        # 1. StarDist Prediction
        labels, _ = star_model.predict_instances(img_01)
        
        # 2. Extract DAB
        dab_map = evaluator._extract_dab_channel(img_01)
        
        # 3. Generate overlays for each threshold
        for thr in args.thresholds:
            overlay_rgb, li = draw_visual_evaluation(img_np, labels, dab_map, thr)
            
            # Save
            out_name = f"{img_path.stem}_thr{thr:.2f}_LI{li:.1f}.jpg"
            out_path = Path(args.output_dir) / out_name
            Image.fromarray(overlay_rgb).save(out_path)
            
    print(f"\n[SUCCESS] Visualizations saved to {args.output_dir}")
    print("Please inspect the images. RED contours = Positive, BLUE contours = Negative.")

if __name__ == "__main__":
    main()
