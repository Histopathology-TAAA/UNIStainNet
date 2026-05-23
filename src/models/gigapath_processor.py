import torch
import torch.nn as nn
import torchvision.transforms as transforms

class GigaPathProcessor(nn.Module):
    """
    Wrapper for the Prov-GigaPath Tile Encoder.
    Used for extracting high-level semantic features for the CCPL Feature Distillation loss.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm to use GigaPath (pip install timm)")
            
        # Load Prov-GigaPath model
        # Requires huggingface-cli login if the model is gated, but prov-gigapath is public.
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.model.eval()
        
        # Freeze model completely
        for param in self.model.parameters():
            param.requires_grad = False
            
        # GigaPath expects 224x224 inputs normalized with ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def forward(self, x):
        """
        Extracts semantic features from an image batch.
        
        Args:
            x: [B, 3, H, W] images in range [-1, 1] or [0, 1]
            
        Returns:
            features: [B, 1536] 1D feature vector representing the tile's semantics
        """
        # Auto-convert [-1, 1] -> [0, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0
            
        # Transform to 224x224 and normalize
        x = self.transform(x)
        
        # Forward pass through frozen model
        features = self.model(x)
        return features
