"""Synthetic personal-loan applicant data. ALL DATA IS SYNTHETIC."""
import numpy as np
import pandas as pd


def make_applicants(n: int = 30000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bureau = np.clip(rng.normal(720, 60, n), 300, 900).round()
    income = np.round(rng.lognormal(10.8, 0.5, n), -2)          # monthly INR
    foir = np.clip(rng.beta(2, 4, n), 0, 0.95)                  # fixed-obligation-to-income
    vintage = rng.integers(0, 240, n)                           # months with current employer
    enquiries = rng.poisson(1.5, n)
    dpd30 = rng.binomial(1, 0.12, n)                            # any 30+ DPD in last 12m
    logit = (-2.6 - 0.018 * (bureau - 720) + 3.0 * (foir - 0.33) - 0.004 * vintage
             + 0.25 * enquiries + 1.1 * dpd30 - 0.3 * (np.log(income) - 10.8))
    bad = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(dict(bureau_score=bureau, monthly_income=income, foir=foir,
                             employer_vintage_m=vintage, enquiries_6m=enquiries,
                             dpd30_12m=dpd30, bad=bad))
