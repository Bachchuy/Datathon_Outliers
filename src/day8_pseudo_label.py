import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
TRAIN_PATH = ROOT / "Data" / "processed_train.csv"
SAMPLE_PATH = ROOT / "Data" / "sample_submission.csv"
BEST_SUB_PATH = ROOT / "submission_v9_plus.csv"
OUTPUT_PATH = ROOT / "submission_v10_pseudo_labeled.csv"

# 1. Feature Engineering
def extract_features(df):
    dt = pd.to_datetime(df["Date"])
    X = pd.DataFrame(index=df.index)
    X["month"] = dt.dt.month
    X["day"] = dt.dt.day
    X["dayofweek"] = dt.dt.dayofweek
    X["dayofyear"] = dt.dt.dayofyear
    X["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    
    # Fourier
    for k in [1, 2, 3, 4]:
        X[f"sin_yr_{k}"] = np.sin(2 * np.pi * k * X["dayofyear"] / 365.25)
        X[f"cos_yr_{k}"] = np.cos(2 * np.pi * k * X["dayofyear"] / 365.25)
        X[f"sin_wk_{k}"] = np.sin(2 * np.pi * k * X["dayofweek"] / 7)
        X[f"cos_wk_{k}"] = np.cos(2 * np.pi * k * X["dayofweek"] / 7)
        
    return X

def main():
    print("Bat dau Pseudo-Labeling voi file tot nhat (v9)...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    pseudo = pd.read_csv(BEST_SUB_PATH, parse_dates=["Date"])
    
    # Căn chỉnh để tập pseudo có các cột như train (tạm thời bỏ qua các cột không cần thiết)
    pseudo["is_test"] = 1
    train["is_test"] = 0
    
    # Detrend train
    train["year"] = train["Date"].dt.year
    ann_rev = train.groupby("year")["Revenue"].mean()
    ann_cog = train.groupby("year")["COGS"].mean()
    
    train["norm_rev"] = train["Revenue"] / train["year"].map(ann_rev)
    train["norm_cog"] = train["COGS"] / train["year"].map(ann_cog)
    
    # Nội suy trend cho test (pseudo)
    base_rev = float(ann_rev.iloc[-1])
    slope_rev = np.polyfit(ann_rev.index[-3:] - np.mean(ann_rev.index[-3:]), ann_rev.values[-3:], 1)[0]
    slope_rev = np.clip(slope_rev, 0.0, base_rev * 0.10)
    
    base_cog = float(ann_cog.iloc[-1])
    slope_cog = np.polyfit(ann_cog.index[-3:] - np.mean(ann_cog.index[-3:]), ann_cog.values[-3:], 1)[0]
    slope_cog = np.clip(slope_cog, 0.0, base_cog * 0.10)
    
    pseudo["year"] = pseudo["Date"].dt.year
    pseudo["scale_rev"] = base_rev + slope_rev * (pseudo["year"] - 2022)
    pseudo["scale_cog"] = base_cog + slope_cog * (pseudo["year"] - 2022)
    
    pseudo["norm_rev"] = pseudo["Revenue"] / pseudo["scale_rev"]
    pseudo["norm_cog"] = pseudo["COGS"] / pseudo["scale_cog"]
    
    # Combine
    X_train = extract_features(train)
    X_pseudo = extract_features(pseudo)
    
    # Gộp lại (Pseudo Labeling)
    X_all = pd.concat([X_train, X_pseudo]).reset_index(drop=True)
    y_rev_all = pd.concat([train["norm_rev"], pseudo["norm_rev"]]).reset_index(drop=True)
    y_cog_all = pd.concat([train["norm_cog"], pseudo["norm_cog"]]).reset_index(drop=True)
    
    # Trọng số: Dữ liệu thật = 1.0, Pseudo = 0.5 (Tránh model quá tin vào nhãn giả)
    weights = np.where(pd.concat([train["is_test"], pseudo["is_test"]]) == 1, 0.5, 1.0)
    
    params = {
        "objective": "huber", "metric": "mae",
        "learning_rate": 0.01, "num_leaves": 63,
        "n_estimators": 1500, "random_state": 42,
        "verbosity": -1, "subsample": 0.8
    }
    
    print("Train Revenue model...")
    ds_rev = lgb.Dataset(X_all, label=y_rev_all, weight=weights)
    model_rev = lgb.train(params, ds_rev)
    
    print("Train COGS model...")
    ds_cog = lgb.Dataset(X_all, label=y_cog_all, weight=weights)
    model_cog = lgb.train(params, ds_cog)
    
    print("Predicting...")
    pred_rev = model_rev.predict(X_pseudo) * pseudo["scale_rev"].values
    pred_cog = model_cog.predict(X_pseudo) * pseudo["scale_cog"].values
    
    # Hậu xử lý (Post-processing Trick): Làm mượt các ngày cuối tuần
    out = pseudo[["Date"]].copy()
    out["Revenue"] = np.clip(pred_rev, 0, None)
    out["COGS"] = np.clip(pred_cog, 0, None)
    
    # Blend 1 lần nữa với v9 gốc để cực kỳ an toàn (70% Pseudo - 30% Gốc)
    out["Revenue"] = 0.7 * out["Revenue"] + 0.3 * pseudo["Revenue"]
    out["COGS"] = 0.7 * out["COGS"] + 0.3 * pseudo["COGS"]
    
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Da luu {OUTPUT_PATH}!")
    
if __name__ == "__main__":
    main()
