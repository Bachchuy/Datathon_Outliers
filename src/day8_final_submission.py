"""
Day 8 v3 - IMPROVED Final Submission (Vectorized, fast)
========================================================
3-model ensemble:
  1. Seasonal Profile (3 recent years, vectorized merge)
  2. LightGBM Fourier (Fourier K=4 + Tet, 49 features)
  3. LightGBM YoY     (same features + YoY anchor from same period last year)
Multi-cutoff backtest: 2021 + 2022 -> inverse-MAE weighted blend

Output: submission_final.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from check_sub import validate_submission

TRAIN_PATH   = ROOT / "Data" / "processed_train.csv"
SAMPLE_PATH  = ROOT / "Data" / "sample_submission.csv"
OUTPUT_FINAL = ROOT / "submission_final.csv"
RANDOM_SEED  = 42

TET_DATES = pd.to_datetime([
    "2012-01-23","2013-02-10","2014-01-31","2015-02-19",
    "2016-02-08","2017-01-28","2018-02-16","2019-02-05",
    "2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10",
])


# ─────────────────────────────────────────────────────────────────
#  Feature Engineering (vectorized)
# ─────────────────────────────────────────────────────────────────
def calendar_features(dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dates)
    d  = pd.DataFrame(index=range(len(dt)))
    d["month"]        = dt.dt.month.values
    d["day"]          = dt.dt.day.values
    d["dayofweek"]    = dt.dt.dayofweek.values
    d["dayofyear"]    = dt.dt.dayofyear.values
    d["quarter"]      = dt.dt.quarter.values
    d["is_weekend"]   = dt.dt.dayofweek.isin([5,6]).astype(int).values
    d["is_month_end"] = dt.dt.is_month_end.astype(int).values
    d["is_payday"]    = dt.dt.day.isin([15,30,31]).astype(int).values

    for k in [1, 2, 3, 4]:
        d[f"sin_yr_{k}"] = np.sin(2*np.pi*k*d["dayofyear"]/365.25)
        d[f"cos_yr_{k}"] = np.cos(2*np.pi*k*d["dayofyear"]/365.25)
        d[f"sin_wk_{k}"] = np.sin(2*np.pi*k*d["dayofweek"]/7)
        d[f"cos_wk_{k}"] = np.cos(2*np.pi*k*d["dayofweek"]/7)

    days_to_tet = np.full(len(dt), 999, dtype=float)
    for tet in TET_DATES:
        diff = (dt - tet).dt.days.values.astype(float)
        mask = (diff >= -30) & (diff <= 45)
        days_to_tet[mask] = diff[mask]

    d["days_to_tet"] = days_to_tet
    d["is_pre_tet"]  = ((days_to_tet >= -30) & (days_to_tet < 0)).astype(int)
    d["is_tet_week"] = ((days_to_tet >= 0)   & (days_to_tet <= 7)).astype(int)
    d["is_post_tet"] = ((days_to_tet > 7)    & (days_to_tet <= 45)).astype(int)
    d["tet_score"]   = np.where(days_to_tet==999, 0, np.exp(-0.05*np.abs(days_to_tet)))

    d["is_holiday"] = 0
    for m, day in [(1,1),(4,30),(5,1),(9,2),(12,25)]:
        d.loc[(d["month"]==m)&(d["day"]==day), "is_holiday"] = 1

    return d


# ─────────────────────────────────────────────────────────────────
#  YoY anchor (vectorized merge)
# ─────────────────────────────────────────────────────────────────
def yoy_features(df_hist: pd.DataFrame, future_dates: pd.Series, target: str) -> pd.DataFrame:
    """Lay gia tri cung ky nam ngoai bang vectorized left-join."""
    hist = df_hist[["Date", target]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.set_index("Date")[target]

    fut = pd.to_datetime(future_dates).reset_index(drop=True)

    yoy_vals  = []
    roll7_vals = []
    for dt in fut:
        # Try 365 / 364 / 366 offset for leap years
        v = np.nan
        for delta in [365, 364, 366]:
            ref = dt - pd.Timedelta(days=delta)
            if ref in hist.index:
                v = float(hist.loc[ref])
                break
        yoy_vals.append(v)

        # Rolling 7 days around same period last year
        ref365 = dt - pd.Timedelta(days=365)
        window = [float(hist.loc[ref365 + pd.Timedelta(days=off)])
                  for off in range(-3, 4)
                  if (ref365 + pd.Timedelta(days=off)) in hist.index]
        roll7_vals.append(np.mean(window) if window else np.nan)

    out = pd.DataFrame({"yoy_lag": yoy_vals, "yoy_roll7": roll7_vals})
    # Fallback: fill NaN with forward-fill then median
    out = out.ffill().bfill().fillna(out.median())
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
#  Seasonal Profile (vectorized)
# ─────────────────────────────────────────────────────────────────
def build_seasonal_profile(df: pd.DataFrame, target: str, recent_years: int = 3):
    df2 = df.copy()
    df2["year"]  = df2["Date"].dt.year
    df2["month"] = df2["Date"].dt.month
    df2["dom"]   = df2["Date"].dt.day
    max_year  = df2["year"].max()
    recent    = df2[df2["year"] >= max_year - recent_years + 1].copy()
    ann_mean  = recent.groupby("year")[target].mean()
    recent["norm"] = recent[target] / recent["year"].map(ann_mean)
    profile   = recent.groupby(["month","dom"])["norm"].mean().reset_index()
    profile.columns = ["month","dom","ratio"]
    return profile, ann_mean


def estimate_future_level(annual_mean: pd.Series):
    recent = annual_mean.iloc[-3:]
    years  = recent.index.values.astype(float)
    vals   = recent.values.astype(float)
    slope  = np.polyfit(years - years.mean(), vals, 1)[0]
    base   = float(annual_mean.iloc[-1])
    # Tinh chỉnh trend: Ép tăng trưởng >= 0 và tối đa 10% (tránh mô hình dự báo giảm sâu)
    return base, float(np.clip(slope, 0.0, base*0.10))


def predict_seasonal_vec(sub_df: pd.DataFrame, profile: pd.DataFrame,
                          base: float, slope: float) -> np.ndarray:
    """Vectorized seasonal predict via merge."""
    tmp = sub_df[["Date"]].copy()
    tmp["month"] = tmp["Date"].dt.month
    tmp["dom"]   = tmp["Date"].dt.day
    tmp["level"] = base + slope * (tmp["Date"].dt.year - 2022)
    tmp = tmp.merge(profile, on=["month","dom"], how="left")
    # Fallback: use monthly mean ratio for missing (month,dom) pairs
    month_mean = profile.groupby("month")["ratio"].mean().rename("ratio_m")
    tmp = tmp.merge(month_mean, on="month", how="left")
    tmp["ratio"] = tmp["ratio"].fillna(tmp["ratio_m"]).fillna(1.0)
    return np.clip((tmp["level"] * tmp["ratio"]).values, 0, None)


# ─────────────────────────────────────────────────────────────────
#  LightGBM helpers
# ─────────────────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective": "regression_l1", "metric": "mae",
    "learning_rate": 0.02, "num_leaves": 127,
    "subsample": 0.8, "colsample_bytree": 0.75,
    "min_child_samples": 10, "reg_alpha": 0.05, "reg_lambda": 0.1,
    "n_jobs": -1, "random_state": RANDOM_SEED, "verbosity": -1,
}


def _detrend(df: pd.DataFrame, target: str):
    df2 = df.copy()
    df2["year"] = df2["Date"].dt.year
    ann = df2.groupby("year")[target].mean()
    df2["norm"] = df2[target] / df2["year"].map(ann)
    return df2, ann


def train_lgb_fourier(df: pd.DataFrame, target: str):
    import lightgbm as lgb
    df2, ann = _detrend(df, target)
    X   = calendar_features(df2["Date"])
    ds  = lgb.Dataset(X, label=df2["norm"])
    mdl = lgb.train(LGB_PARAMS, ds, num_boost_round=3000)
    return mdl, ann, X.columns.tolist()


def train_lgb_yoy(df: pd.DataFrame, target: str):
    import lightgbm as lgb
    df2, ann = _detrend(df, target)
    cal = calendar_features(df2["Date"])
    yoy = yoy_features(df2, df2["Date"], target)
    base_scale = float(ann.iloc[-1])
    yoy["yoy_lag"]   /= base_scale
    yoy["yoy_roll7"] /= base_scale
    X = pd.concat([cal.reset_index(drop=True), yoy.reset_index(drop=True)], axis=1)
    ds  = lgb.Dataset(X, label=df2["norm"])
    mdl = lgb.train(LGB_PARAMS, ds, num_boost_round=3000)
    return mdl, ann, X.columns.tolist()


def predict_lgb(mdl, feat_cols: list, sub_df: pd.DataFrame,
                df_hist: pd.DataFrame, base: float, slope: float,
                target: str, ann: pd.Series, use_yoy: bool = False) -> np.ndarray:
    cal = calendar_features(sub_df["Date"])
    if use_yoy:
        yoy = yoy_features(df_hist, sub_df["Date"], target)
        base_scale = float(ann.iloc[-1])
        yoy["yoy_lag"]   /= base_scale
        yoy["yoy_roll7"] /= base_scale
        X = pd.concat([cal.reset_index(drop=True), yoy.reset_index(drop=True)], axis=1)
    else:
        X = cal
    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feat_cols]
    scales = np.array([base + slope*(yr-2022) for yr in sub_df["Date"].dt.year])
    return np.clip(mdl.predict(X) * scales, 0, None)


# ─────────────────────────────────────────────────────────────────
#  Multi-cutoff Backtest
# ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, target: str, cutoffs=("2021-01-01","2022-01-01")):
    from sklearn.metrics import mean_absolute_error
    all_maes = {"seasonal": [], "lgb_fourier": [], "lgb_yoy": []}

    for cutoff in cutoffs:
        tr = df[df["Date"] < cutoff].copy().reset_index(drop=True)
        va = df[df["Date"] >= cutoff].copy().reset_index(drop=True)
        print(f"  Cutoff {cutoff}: train={len(tr)} val={len(va)}")

        # Seasonal
        prof, ann = build_seasonal_profile(tr, target)
        b, s      = estimate_future_level(ann)
        p_s       = predict_seasonal_vec(va, prof, b, s)
        mae_s     = mean_absolute_error(va[target], p_s)
        all_maes["seasonal"].append(mae_s)

        # LGB Fourier
        mdl_f, ann_f, fc_f = train_lgb_fourier(tr, target)
        b_f, s_f = estimate_future_level(ann_f)
        p_f      = predict_lgb(mdl_f, fc_f, va, tr, b_f, s_f, target, ann_f, use_yoy=False)
        mae_f    = mean_absolute_error(va[target], p_f)
        all_maes["lgb_fourier"].append(mae_f)

        # LGB YoY
        try:
            mdl_y, ann_y, fc_y = train_lgb_yoy(tr, target)
            b_y, s_y = estimate_future_level(ann_y)
            p_y      = predict_lgb(mdl_y, fc_y, va, tr, b_y, s_y, target, ann_y, use_yoy=True)
            mae_y    = mean_absolute_error(va[target], p_y)
        except Exception as e:
            print(f"  [WARN] YoY failed: {e}")
            mae_y = mae_f
        all_maes["lgb_yoy"].append(mae_y)

        print(f"    Seasonal={mae_s:>12,.0f}  Fourier={mae_f:>12,.0f}  YoY={mae_y:>12,.0f}")

    avg = {k: np.mean(v) for k, v in all_maes.items() if v}
    print(f"\n  Avg MAE: " + "  ".join(f"{k}={v:,.0f}" for k,v in avg.items()))
    inv   = {k: 1/v for k, v in avg.items()}
    total = sum(inv.values())
    weights = {k: v/total for k, v in inv.items()}
    print(f"  Weights: " + "  ".join(f"{k}={v:.3f}" for k,v in weights.items()))
    return weights


# ─────────────────────────────────────────────────────────────────
#  Final forecast
# ─────────────────────────────────────────────────────────────────
def get_final_forecast(df: pd.DataFrame, sub: pd.DataFrame,
                        target: str, weights: dict) -> np.ndarray:
    preds = {}

    prof, ann = build_seasonal_profile(df, target)
    b, s = estimate_future_level(ann)
    preds["seasonal"] = predict_seasonal_vec(sub, prof, b, s)

    mdl_f, ann_f, fc_f = train_lgb_fourier(df, target)
    b_f, s_f = estimate_future_level(ann_f)
    preds["lgb_fourier"] = predict_lgb(mdl_f, fc_f, sub, df, b_f, s_f, target, ann_f, False)

    try:
        mdl_y, ann_y, fc_y = train_lgb_yoy(df, target)
        b_y, s_y = estimate_future_level(ann_y)
        preds["lgb_yoy"] = predict_lgb(mdl_y, fc_y, sub, df, b_y, s_y, target, ann_y, True)
    except Exception as e:
        print(f"  [WARN] YoY final failed: {e}. Using Fourier substitute.")
        preds["lgb_yoy"] = preds["lgb_fourier"]

    blended = sum(weights.get(k, 0) * v for k, v in preds.items())
    return np.clip(blended, 0, None)


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  DAY 8 v3 - IMPROVED FINAL SUBMISSION")
    print("  Seasonal + LGB-Fourier + LGB-YoY | Multi-cutoff backtest")
    print("="*65)

    print("\n[1/4] Loading data...")
    df  = pd.read_csv(TRAIN_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sub = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"  Train={df.shape} | {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Test ={sub.shape} | {sub['Date'].min().date()} to {sub['Date'].max().date()}")

    print("\n[2/4] Multi-cutoff backtest -> optimal weights...")
    # Phương pháp 1: Dự báo gián tiếp qua Gross Profit
    df["Gross_Profit"] = df["Revenue"] - df["COGS"]

    print("\n  -- COGS --")
    w_cog = run_backtest(df, "COGS")
    print("\n  -- Gross Profit --")
    w_gp = run_backtest(df, "Gross_Profit")

    print("\n[3/4] Final forecast on full data...")
    cogs_pred = get_final_forecast(df, sub, "COGS", w_cog)
    gp_pred   = get_final_forecast(df, sub, "Gross_Profit", w_gp)
    
    # Doanh thu = Giá vốn + Lợi nhuận gộp (đảm bảo Margin luôn >= 0)
    rev_pred = cogs_pred + gp_pred
    print(f"  Revenue: {rev_pred.min():,.0f} to {rev_pred.max():,.0f}  mean={rev_pred.mean():,.0f}")
    print(f"  COGS   : {cogs_pred.min():,.0f} to {cogs_pred.max():,.0f}  mean={cogs_pred.mean():,.0f}")

    print("\n[4/4] Save & validate...")
    out = sub[["Date"]].copy()
    out["Revenue"] = np.round(rev_pred,  2)
    out["COGS"]    = np.round(cogs_pred, 2)
    out["Date"]    = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_FINAL, index=False)
    print(f"  [SAVED] {OUTPUT_FINAL}")

    # Compare vs v9_plus
    v9p = ROOT / "submission_v9_plus.csv"
    if v9p.exists():
        v9 = pd.read_csv(v9p)
        m  = v9[["Date","Revenue"]].merge(out[["Date","Revenue"]], on="Date", suffixes=("_v9","_new"))
        print(f"  vs v9_plus: mean {v9['Revenue'].mean():,.0f} -> {out['Revenue'].mean():,.0f}")

    validate_submission(OUTPUT_FINAL, SAMPLE_PATH)
    print("\n[DONE] submission_final.csv ready for Kaggle!\n")


if __name__ == "__main__":
    main()
