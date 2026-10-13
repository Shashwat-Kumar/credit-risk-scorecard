"""Weight-of-Evidence binning and Information Value.

WoE convention: ln(%good / %bad), so higher WoE = lower risk.
"""
import numpy as np
import pandas as pd


def fit_bins(x: pd.Series, y: pd.Series, q: int = 10, eps: float = 0.5) -> pd.DataFrame:
    if x.nunique() <= q:
        edges = np.unique(np.concatenate(([-np.inf], np.sort(x.unique())[:-1] + 1e-9, [np.inf])))
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(x, edges, include_lowest=True)
    t = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=False)["y"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    pg = (t["good"] + eps) / (t["good"].sum() + eps)
    pb = (t["bad"] + eps) / (t["bad"].sum() + eps)
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    t.attrs["edges"] = edges
    return t


def transform(x: pd.Series, table: pd.DataFrame) -> np.ndarray:
    idx = np.clip(np.searchsorted(table.attrs["edges"], x, side="left") - 1, 0, len(table) - 1)
    return table["woe"].to_numpy()[idx]


def information_value(table: pd.DataFrame) -> float:
    return float(table["iv"].sum())
