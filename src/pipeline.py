from __future__ import annotations
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train(seed=42):
    data = load_iris()
    xt, xe, yt, ye = train_test_split(data.data, data.target, test_size=0.3, random_state=seed, stratify=data.target)
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=400))])
    model.fit(xt, yt)
    return model, {"accuracy": float(accuracy_score(ye, model.predict(xe)))}
