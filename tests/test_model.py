import torch
from main import Model_0

def test_model_forward_shape():
    model = Model_0()
    x = torch.randn(1, 1, 196, 196)
    out = model(x)

    assert out.shape == (1, 1)
