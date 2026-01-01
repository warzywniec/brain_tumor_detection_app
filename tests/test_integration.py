import torch
import tkinter as tk
from main import AppUI
from PIL import Image
import os

class DummyModel:
    def __call__(self, x):
        return torch.tensor([[2.0]])

def test_full_analysis_flow(tmp_path, monkeypatch):
    img_path = tmp_path / "test.png"
    Image.new("L", (196, 196)).save(img_path)

    root = tk.Tk()
    root.withdraw()

    main = AppUI(root)
    main.image_list = [str(img_path)]

    monkeypatch.setattr("main.model", DummyModel())

    main._analyze_folder_worker()

    assert len(main.analysis_results) == 1
    r = main.analysis_results[0]

    assert r["label"] == "Tumor"
    assert r["prob"] > 0.5
