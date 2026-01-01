from main import AppUI

def test_prob_to_label_tumor():
    assert AppUI.prob_to_label(0.9) == "Tumor"

def test_prob_to_label_healthy():
    assert AppUI.prob_to_label(0.1) == "Healthy"

def test_results_sorted_descending():
    results = [
        {"prob": 0.2},
        {"prob": 0.9},
        {"prob": 0.5},
    ]

    results.sort(key=lambda x: x["prob"], reverse=True)

    probs = [r["prob"] for r in results]
    assert probs == [0.9, 0.5, 0.2]



