from src.pipeline import train
def test_iris_accuracy():
    model, metrics = train()
    assert metrics["accuracy"] >= 0.85
