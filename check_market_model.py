import pickle
import numpy as np

data  = pickle.load(open("model_v3_market.pkl", "rb"))
feats = data["features"]
imps  = None

for est in data["model"].calibrated_classifiers_:
    base = est.estimator
    if hasattr(base, "feature_importances_"):
        fi = base.feature_importances_.astype(np.float64)
        if imps is None:
            imps = fi.copy()
        else:
            imps += fi

imps /= len(data["model"].calibrated_classifiers_)
ranked = sorted(zip(feats, imps), key=lambda x: -x[1])

print(f"\n  Market-level model — top 20 features")
print(f"  {'Feature':<40} {'Score':>8}")
print(f"  {'─'*40} {'─'*8}")
for f, s in ranked[:20]:
    bar = "#" * int(s / ranked[0][1] * 20)
    print(f"  {f:<40} {s:>8.1f}  {bar}")
