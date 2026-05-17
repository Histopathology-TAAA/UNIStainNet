"""End-to-end test: SPADE-UNet + SpatialGNN forward + backward pass."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.generator import SPADEUNetGenerator
from src.models.spatial_gnn import SpatialGNN, build_grid_graph


def test_e2e_spatial_forward_backward():
    """Full forward + backward pass with spatial GNN, encoder frozen."""
    gen = SPADEUNetGenerator(
        uni_spatial_size=32, image_size=512,
        use_spatial=True, spatial_dim=64,
    )
    gnn = SpatialGNN(
        feature_dim=512, hidden_dim=256, output_dim=64, num_layers=3,
    )

    # Freeze encoder (simulate Stage 2)
    frozen_prefixes = (
        'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
        'bottleneck', 'uni_processor', 'class_embed',
    )
    for name, param in gen.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False

    # Verify only decoder is trainable
    trainable = [n for n, p in gen.named_parameters() if p.requires_grad]
    assert all(any(x in n for x in ('dec', 'output', 'spade'))
               for n in trainable), f"Wrong params trainable: {trainable}"

    # 3x3 grid batch
    B = 9
    he = torch.randn(B, 3, 512, 512)
    uni = torch.randn(B, 1024, 1024)
    labels = torch.randint(0, 5, (B,))
    positions = torch.tensor([[r, c] for r in range(3) for c in range(3)]).float()

    # Encoder forward (frozen)
    with torch.no_grad():
        enc1 = gen.enc1(he)
        enc2 = gen.enc2(enc1)
        enc3 = gen.enc3(enc2)
        enc4 = gen.enc4(enc3)
        enc5 = gen.enc5(enc4)
        gnn_input = enc5.mean(dim=[2, 3])

    # GNN forward
    spatial_emb = gnn(gnn_input, positions)
    assert spatial_emb.shape == (B, 64)

    # Generator forward with spatial
    generated = gen(he, uni, labels, spatial_emb=spatial_emb)
    assert generated.shape == (B, 3, 512, 512)

    # Backward
    loss = generated.mean()
    loss.backward()

    # GNN gradients
    gnn_grad = sum(p.grad.norm().item() for p in gnn.parameters()
                   if p.grad is not None)
    assert gnn_grad > 0, "No GNN gradients"

    # Decoder gradients
    dec_grad = sum(p.grad.norm().item() for n, p in gen.named_parameters()
                   if p.requires_grad and p.grad is not None)
    assert dec_grad > 0, "No decoder gradients"

    # Encoder frozen
    enc_grad = sum(p.grad.norm().item() for n, p in gen.named_parameters()
                   if not p.requires_grad and p.grad is not None)
    assert enc_grad == 0.0, f"Encoder has gradients: {enc_grad}"

    print(f"E2E test PASSED | GNN grad: {gnn_grad:.4f} | Decoder grad: {dec_grad:.4f}")


def test_stain_hop_configs():
    """Verify different hop counts produce valid output."""
    features = torch.randn(25, 512)
    positions = torch.tensor([[r, c] for r in range(5) for c in range(5)]).float()

    configs = [('ER', 3), ('PR', 3), ('HER2', 5), ('KI67', 7)]
    for stain, hops in configs:
        gnn = SpatialGNN(num_layers=hops)
        out = gnn(features, positions)
        assert out.shape == (25, 64), f"{stain} ({hops}h): {out.shape}"
        print(f"  {stain}: {hops}-hop OK")


def test_backward_compat():
    """Old checkpoints (use_spatial=False) still work."""
    gen = SPADEUNetGenerator(uni_spatial_size=32, image_size=512,
                              use_spatial=False)
    he = torch.randn(1, 3, 512, 512)
    uni = torch.randn(1, 1024, 1024)
    labels = torch.tensor([0])
    out = gen(he, uni, labels)
    assert out.shape == (1, 3, 512, 512)
    print("Backward compat OK")


def test_graph_building():
    """Edge index construction for various grid sizes."""
    for k in [2, 3, 5, 7]:
        n = k * k
        positions = torch.tensor([[r, c] for r in range(k) for c in range(k)]).float()
        edge_index = build_grid_graph(positions, k=min(8, n - 1))
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0
        print(f"  {k}x{k} grid: {edge_index.shape[1]} edges OK")


if __name__ == '__main__':
    test_graph_building()
    test_backward_compat()
    test_stain_hop_configs()
    test_e2e_spatial_forward_backward()
    print("\nAll integration tests PASSED")
