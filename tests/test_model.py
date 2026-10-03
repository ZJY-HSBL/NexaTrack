import torch

from nexatrack.models.model import NexaTrack


def tiny_cfg():
    return {
        "model": {
            "d_model": 64,
            "num_heads": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "ffn_dim": 128,
            "dropout": 0.0,
            "inner_dim": 16,
            "backbone_dims": [32, 64, 128],
            "backbone_depths": [1, 1, 1],
            "window_size": 4,
            "token_grid_template": [4, 4],
            "token_grid_search": [6, 6],
            "template_work_size": 64,
            "search_work_size": 96,
            "score_hidden": 64,
        }
    }


def test_model_forward():
    torch.manual_seed(0)
    model = NexaTrack(tiny_cfg()).eval()
    static = torch.randn(1, 3, 64, 64)
    dynamic = torch.randn(1, 3, 64, 64)
    search = torch.randn(1, 3, 96, 96)
    box = torch.tensor([[0.3, 0.3, 0.7, 0.7]], dtype=torch.float32)
    with torch.no_grad():
        out = model(static, dynamic, search, box, box)
    assert out["boxes"].shape == (1, 4)
    assert out["score"].shape == (1,)
    assert torch.isfinite(out["boxes"]).all()
