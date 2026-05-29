import torch
import sys
sys.path.insert(0, '.')
from src.models.trainer import UNIStainNetTrainer

trainer = UNIStainNetTrainer(
    case_a_prob=0.5,
    edge_encoder='v2',
    image_size=512
)

B = 2
he = torch.randn(B, 3, 512, 512)
her2 = torch.randn(B, 3, 512, 512)
he_h = torch.randn(B, 1, 512, 512)
ihc_h = torch.randn(B, 1, 512, 512)
uni = torch.randn(B, 16, 1024)
labels = torch.zeros(B, dtype=torch.long)
fnames = ["dummy1.png", "dummy2.png"]

batch = (he, her2, he_h, ihc_h, uni, labels, fnames)

opt_g, opt_d = trainer.configure_optimizers()
# fake step
try:
    trainer.training_step(batch, 0)
    print("Training step passed")
except Exception as e:
    print(f"Training step failed: {e}")

try:
    trainer.validation_step(batch, 0)
    print("Validation step passed")
except Exception as e:
    print(f"Validation step failed: {e}")

