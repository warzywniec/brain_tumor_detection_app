import os
from main import AppUI
import tkinter as tk

def test_pdf_generation(tmp_path):
    root = tk.Tk()
    root.withdraw()

    app = AppUI(root)
    app.image_folder = "test_folder"
    app.analysis_results = [
        {"path": "img1.png", "prob": 0.9, "label": "Tumor"},
        {"path": "img2.png", "prob": 0.1, "label": "Healthy"},
    ]

    pdf_path = tmp_path / "report.pdf"
    app._create_simple_pdf(str(pdf_path))

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
