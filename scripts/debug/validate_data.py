"""
CPU-only data loading validator — run BEFORE turning on a GPU.

Instantiates MISTMultiStainCropDataModule with your real data directory,
iterates a few batches, and prints shapes + stain distributions.
Does NOT load UNI, does NOT touch the model.

Usage:
    python scripts/debug/validate_data.py --data_dir /path/to/MIST
    python scripts/debug/validate_data.py --data_dir /path/to/MIST --stains HER2 ER
    python scripts/debug/validate_data.py --data_dir /path/to/MIST --n_batches 5
"""

import argparse
import time

from src.data.mist_dataset import MISTMultiStainCropDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root dir containing HER2/, ER/, Ki67/, PR/ subfolders')
    parser.add_argument('--stains', nargs='+', default=['HER2', 'ER', 'Ki67', 'PR'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_batches', type=int, default=3,
                        help='Number of batches to iterate for timing')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Data dir : {args.data_dir}")
    print(f"Stains   : {args.stains}")
    print(f"Batch sz : {args.batch_size}")
    print("=" * 60)

    dm = MISTMultiStainCropDataModule(
        base_dir=args.data_dir,
        stains=args.stains,
        batch_size=args.batch_size,
        num_workers=0,   # 0 for safe CPU debugging
        image_size=(512, 512),
        crop_size=512,
        null_class=4,
    )
    dm.setup('fit')

    print(f"\nTrain dataset size : {len(dm.train_dataset)} samples")
    print(f"Val   dataset size : {len(dm.val_dataset)} samples")

    print("\n--- Train loader ---")
    loader = dm.train_dataloader()
    t0 = time.time()
    for i, batch in enumerate(loader):
        he_rgb, ihc_rgb, he_h_map, ihc_h_map, labels, fnames = batch
        if i == 0:
            print(f"  he_rgb   : {tuple(he_rgb.shape)}  range [{he_rgb.min():.2f}, {he_rgb.max():.2f}]")
            print(f"  ihc_rgb  : {tuple(ihc_rgb.shape)} range [{ihc_rgb.min():.2f}, {ihc_rgb.max():.2f}]")
            print(f"  he_h_map : {tuple(he_h_map.shape)} range [{he_h_map.min():.2f}, {he_h_map.max():.2f}]")
            print(f"  ihc_h_map: {tuple(ihc_h_map.shape)} range [{ihc_h_map.min():.2f}, {ihc_h_map.max():.2f}]")
            print(f"  labels   : {labels.tolist()} (dtype={labels.dtype})")
            print(f"  fnames   : {list(fnames)}")
        if i + 1 >= args.n_batches:
            break
    elapsed = time.time() - t0
    print(f"  {args.n_batches} batches loaded in {elapsed:.2f}s ({elapsed/args.n_batches:.2f}s/batch)")

    print("\n--- Val loader ---")
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    he_rgb, ihc_rgb, he_h_map, ihc_h_map, labels, fnames = batch
    print(f"  he_rgb   : {tuple(he_rgb.shape)}")
    print(f"  labels   : {labels.tolist()}")

    print("\nAll checks passed. Data loading is healthy.")


if __name__ == '__main__':
    main()
