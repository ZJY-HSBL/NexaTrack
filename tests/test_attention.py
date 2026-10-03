import torch

from nexatrack.models.attention import ContextAwareAttention


def test_context_attention_shape_and_grad():
    torch.manual_seed(0)
    layer = ContextAwareAttention(d_model=64, num_heads=4, inner_dim=16, dropout=0.0)
    x = torch.randn(2, 20, 64, requires_grad=True)
    weights = torch.ones(2, 20)
    weights[:, :5] = torch.linspace(0.1, 1.0, 5)
    y = layer(x, x, x, key_weights=weights)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
