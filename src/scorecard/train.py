"""Build a WoE logistic scorecard on synthetic data; report AUC, KS, Gini, calibration."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from scorecard.data import make_applicants
from scorecard.woe import fit_bins, information_value, transform

OUT = Path(__file__).resolve().parents[2] / "reports"
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50   # 600 points = 50:1 good:bad, 20 points to double odds


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def run(n=30000, seed=7, out=OUT, iv_min=0.02):
    out.mkdir(exist_ok=True)
    df = make_applicants(n, seed)
    feats = [c for c in df.columns if c != "bad"]
    tr, te = train_test_split(df, test_size=0.3, stratify=df["bad"], random_state=seed)

    tables = {f: fit_bins(tr[f], tr["bad"]) for f in feats}
    iv = {f: round(information_value(t), 4) for f, t in tables.items()}
    kept = [f for f in feats if iv[f] >= iv_min]
    W = lambda d: np.column_stack([transform(d[f], tables[f]) for f in kept])

    lr = LogisticRegression(max_iter=1000).fit(W(tr), tr["bad"])
    p = lr.predict_proba(W(te))[:, 1]

    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    score = np.round(offset - factor * lr.decision_function(W(te)))   # higher = safer

    frac, mean_pred = calibration_curve(te["bad"], p, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac, "o-", label="scorecard"); plt.plot([0, mean_pred.max()], [0, mean_pred.max()], "--", label="perfect")
    plt.xlabel("predicted PD"); plt.ylabel("observed bad rate"); plt.title("Calibration (synthetic test set)"); plt.legend()
    plt.savefig(out / "calibration.png", dpi=110); plt.close()

    rows = []
    for f, c in zip(kept, lr.coef_[0]):
        t = tables[f]
        for b, w in zip(t.index.astype(str), t["woe"]):
            rows.append({"feature": f, "bin": b, "woe": round(w, 4), "points": round(-factor * c * w, 1)})
    pd.DataFrame(rows).to_csv(out / "scorecard_points.csv", index=False)

    m = {"n_test": int(len(te)), "bad_rate": round(float(te["bad"].mean()), 4),
         "auc": round(roc_auc_score(te["bad"], p), 4),
         "gini": round(2 * roc_auc_score(te["bad"], p) - 1, 4),
         "ks": round(ks_stat(te["bad"], p), 4),
         "brier": round(brier_score_loss(te["bad"], p), 4),
         "information_value": iv, "features_kept": kept,
         "score_range_test": [int(score.min()), int(score.max())],
         "calibration_max_abs_gap": round(float(np.max(np.abs(frac - mean_pred))), 4)}
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
