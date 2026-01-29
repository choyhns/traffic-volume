# src/eval_rolling.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# optional: xgboost
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

from sklearn.ensemble import HistGradientBoostingRegressor

# TF for GRU
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


# ----------------------------
# metrics
# ----------------------------
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def eval_regression(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    mp = mape(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mp)}


# ----------------------------
# feature engineering (make_windows 쪽과 호환되는 최소셋)
# ----------------------------
def make_time_features(dt: pd.Series) -> pd.DataFrame:
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    month = dt.dt.month
    is_weekend = (dow >= 5).astype(int)
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "is_weekend": is_weekend,
    })

def ensure_wind_dir_sincos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wind_dir_deg" in df.columns and ("wind_dir_sin" not in df.columns or "wind_dir_cos" not in df.columns):
        deg = pd.to_numeric(df["wind_dir_deg"], errors="coerce").fillna(0.0).astype(float)
        rad = np.deg2rad(deg % 360.0)
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    return df

def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))

@dataclass
class WindowBundle:
    X_raw: np.ndarray          # (N,T,F)
    y_raw: np.ndarray          # (N,)
    target_time: np.ndarray    # (N,) datetime64[ns]
    spot_id: np.ndarray        # (N,) int32
    feature_cols: list[str]


def build_feature_frame(hourly_csv: Path, include_weather=True) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(hourly_csv, dtype={"spot_num": str}, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["spot_num", "datetime"]).reset_index(drop=True)

    df = ensure_wind_dir_sincos(df)

    # spot_id mapping
    spots = sorted(df["spot_num"].unique().tolist())
    spot_to_id = {s: i for i, s in enumerate(spots)}
    df["spot_id"] = df["spot_num"].map(spot_to_id).astype(int)

    # time features
    tf = make_time_features(df["datetime"])
    for c in tf.columns:
        df[c] = tf[c].values

    # choose weather cols present
    weather_candidates = [
        "temp_c", "rain_mm", "wind_ms", "wind_dir_sin", "wind_dir_cos",
        "humidity_pct", "pressure_hpa", "snow_cm", "cloud_total_10"
    ]
    weather_cols = [c for c in weather_candidates if c in df.columns] if include_weather else []

    # d1 features (lag1) per spot
    for c in weather_cols:
        df[f"{c}_d1"] = df.groupby("spot_num")[c].shift(1)

    # fill na in weather & d1
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        df[f"{c}_d1"] = pd.to_numeric(df[f"{c}_d1"], errors="coerce").fillna(0.0).astype(float)

    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0).astype(float)

    # feature cols order (make_windows와 유사)
    feature_cols = ["vol"] + list(tf.columns) + weather_cols + [f"{c}_d1" for c in weather_cols] + ["spot_id"]

    meta = {
        "spot_to_id": spot_to_id,
        "weather_cols": weather_cols,
        "feature_cols": feature_cols,
    }
    return df, meta


def make_windows_with_time(df: pd.DataFrame, feature_cols: list[str], window: int, horizon: int, stride: int) -> WindowBundle:
    X_list, y_list, t_list, sid_list = [], [], [], []
    # spot별 독립 시퀀스에서 윈도우 생성
    for spot, g in df.groupby("spot_num"):
        g = g.sort_values("datetime").reset_index(drop=True)

        feats = g[feature_cols].to_numpy(dtype=np.float32)  # (L,F)
        vols = g["vol"].to_numpy(dtype=np.float32)
        dts = g["datetime"].to_numpy(dtype="datetime64[ns]")
        sids = g["spot_id"].to_numpy(dtype=np.int32)

        L = len(g)
        max_i = L - (window + horizon) + 1
        if max_i <= 0:
            continue

        for i in range(0, max_i, stride):
            x = feats[i:i+window]  # (T,F)
            y = vols[i+window+horizon-1]  # 1-step면 i+window
            target_t = dts[i+window+horizon-1]
            sid = sids[i]  # same for all steps

            X_list.append(x)
            y_list.append(y)
            t_list.append(target_t)
            sid_list.append(sid)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, window, len(feature_cols)), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    t = np.asarray(t_list, dtype="datetime64[ns]")
    sid = np.asarray(sid_list, dtype=np.int32)
    return WindowBundle(X_raw=X, y_raw=y, target_time=t, spot_id=sid, feature_cols=feature_cols)


# ----------------------------
# Rolling split
# ----------------------------
def rolling_folds(target_time: np.ndarray, n_folds: int, test_days: int, min_train_days: int):
    """
    Expanding window:
      train: [start ... cutoff)
      test : [cutoff ... cutoff+test_days)
    """
    ts = pd.to_datetime(target_time)
    min_t = ts.min()
    max_t = ts.max()

    # fold 시작점: 뒤에서 n_folds*test_days 만큼 거슬러 올라가며 생성
    # train 최소기간(min_train_days) 보장
    fold_ends = []
    cursor = max_t.normalize()  # day boundary
    for _ in range(n_folds):
        test_start = cursor - pd.Timedelta(days=test_days)
        fold_ends.append(test_start)
        cursor = test_start

    fold_ends = list(reversed(fold_ends))

    for k, test_start in enumerate(fold_ends, start=1):
        train_end = test_start
        train_start = train_end - pd.Timedelta(days=min_train_days)

        train_mask = (ts < train_end) & (ts >= train_start)
        test_mask = (ts >= test_start) & (ts < (test_start + pd.Timedelta(days=test_days)))

        yield k, train_mask, test_mask, train_start, train_end, test_start


# ----------------------------
# Models
# ----------------------------
def fit_predict_xgb(Xtr, ytr, Xte):
    Xtr_f = Xtr.reshape((Xtr.shape[0], -1))
    Xte_f = Xte.reshape((Xte.shape[0], -1))

    if XGB_OK:
        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=0,
        )
        model.fit(Xtr_f, ytr, verbose=False)
    else:
        model = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
        model.fit(Xtr_f, ytr)

    pred = model.predict(Xte_f).astype(np.float32)
    return model, pred


def build_gru_with_spot_embedding(timesteps: int, n_seq_features: int, num_spots: int, embed_dim=8, hidden=64, dropout=0.2):
    seq_in = layers.Input(shape=(timesteps, n_seq_features), name="seq_in")
    spot_in = layers.Input(shape=(), dtype="int32", name="spot_id")

    emb = layers.Embedding(input_dim=num_spots, output_dim=embed_dim, name="spot_embedding")(spot_in)
    emb = layers.Flatten()(emb)
    emb = layers.RepeatVector(timesteps)(emb)

    x = layers.Concatenate(axis=-1)([seq_in, emb])
    x = layers.GRU(hidden, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = models.Model(inputs=[seq_in, spot_in], outputs=out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_predict_gru_logscaled(
    Xtr_raw, ytr_raw, sid_tr,
    Xte_raw, sid_te,
    feature_cols: list[str],
    spot_embed_dim: int,
    epochs: int,
    batch_size: int,
):
    """
    GRU는 로그+표준화(훈련구간 통계) 적용해서 학습하고,
    평가는 원 스케일(y_raw)로 진행.
    """
    # --- feature-wise log1p mask: vol + weather + *_d1만 log1p (sin/cos, weekend, spot_id 제외)
    log1p_cols = []
    for c in feature_cols:
        if c == "vol":
            log1p_cols.append(True)
        elif c.endswith("_d1"):
            log1p_cols.append(True)
        elif c in ["temp_c", "rain_mm", "wind_ms", "humidity_pct", "pressure_hpa", "snow_cm", "cloud_total_10"]:
            log1p_cols.append(True)
        else:
            log1p_cols.append(False)
    log1p_mask = np.asarray(log1p_cols, dtype=np.int32)

    # --- drop spot_id from seq features (embedding으로만 사용)
    if "spot_id" in feature_cols:
        spot_idx = feature_cols.index("spot_id")
        Xtr_seq = np.delete(Xtr_raw, spot_idx, axis=2)
        Xte_seq = np.delete(Xte_raw, spot_idx, axis=2)
        seq_cols = [c for i, c in enumerate(feature_cols) if i != spot_idx]
        log1p_mask_seq = np.delete(log1p_mask, spot_idx)
    else:
        Xtr_seq, Xte_seq = Xtr_raw, Xte_raw
        seq_cols = feature_cols
        log1p_mask_seq = log1p_mask

    # --- apply log1p & z-score with TRAIN stats
    Xtr = Xtr_seq.copy()
    Xte = Xte_seq.copy()

    # log1p selected
    idxs = np.where(log1p_mask_seq == 1)[0]
    if len(idxs):
        Xtr[..., idxs] = _safe_log1p(Xtr[..., idxs])
        Xte[..., idxs] = _safe_log1p(Xte[..., idxs])

    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    # --- y transform: log1p + standardize (TRAIN stats)
    ytr_log = _safe_log1p(ytr_raw)
    y_mean = float(ytr_log.mean())
    y_std = float(ytr_log.std() if ytr_log.std() > 1e-6 else 1.0)
    ytr = (ytr_log - y_mean) / y_std

    timesteps = Xtr.shape[1]
    n_seq_features = Xtr.shape[2]
    num_spots = int(max(int(sid_tr.max()), int(sid_te.max())) + 1)

    model = build_gru_with_spot_embedding(
        timesteps=timesteps,
        n_seq_features=n_seq_features,
        num_spots=num_spots,
        embed_dim=spot_embed_dim,
        hidden=64,
        dropout=0.2,
    )

    cbs = [
        callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="loss", patience=2, factor=0.5, min_lr=1e-5),
    ]

    model.fit([Xtr, sid_tr], ytr, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cbs)

    y_pred_scaled = model.predict([Xte, sid_te], batch_size=batch_size, verbose=0).reshape(-1)

    # inverse y
    y_pred_log = y_pred_scaled * y_std + y_mean
    y_pred = np.expm1(y_pred_log).astype(np.float32)
    return model, y_pred


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly_csv", default=str(PROCESSED / "traffic_hourly.csv"))
    ap.add_argument("--window", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--test_days", type=int, default=14)        # 각 fold 테스트 2주
    ap.add_argument("--min_train_days", type=int, default=90)   # train 최소 3개월

    ap.add_argument("--gru_epochs", type=int, default=6)        # 롤링 검증용: 짧게
    ap.add_argument("--gru_batch", type=int, default=256)
    ap.add_argument("--spot_embed_dim", type=int, default=8)

    ap.add_argument("--tag", default="rolling")
    args = ap.parse_args()

    df, meta = build_feature_frame(Path(args.hourly_csv), include_weather=True)
    bundle = make_windows_with_time(df, meta["feature_cols"], args.window, args.horizon, args.stride)

    if bundle.X_raw.shape[0] == 0:
        raise RuntimeError("윈도우 샘플이 0개입니다. window/horizon/stride 또는 데이터 기간을 확인하세요.")

    rows = []
    for k, train_mask, test_mask, tr_s, tr_e, te_s in rolling_folds(bundle.target_time, args.n_folds, args.test_days, args.min_train_days):
        Xtr, ytr, sid_tr = bundle.X_raw[train_mask], bundle.y_raw[train_mask], bundle.spot_id[train_mask]
        Xte, yte, sid_te = bundle.X_raw[test_mask], bundle.y_raw[test_mask], bundle.spot_id[test_mask]

        # 너무 작으면 스킵
        if len(Xtr) < 1000 or len(Xte) < 200:
            continue

        # XGB
        xgb_model, xgb_pred = fit_predict_xgb(Xtr, ytr, Xte)
        xgb_m = eval_regression(yte, xgb_pred)
        rows.append({
            "fold": k,
            "model": "XGB",
            "train_start": str(tr_s.date()),
            "train_end": str(tr_e.date()),
            "test_start": str(te_s.date()),
            "test_days": args.test_days,
            **xgb_m
        })

        # GRU
        gru_model, gru_pred = fit_predict_gru_logscaled(
            Xtr, ytr, sid_tr,
            Xte, sid_te,
            feature_cols=bundle.feature_cols,
            spot_embed_dim=args.spot_embed_dim,
            epochs=args.gru_epochs,
            batch_size=args.gru_batch,
        )
        gru_m = eval_regression(yte, gru_pred)
        rows.append({
            "fold": k,
            "model": "GRU",
            "train_start": str(tr_s.date()),
            "train_end": str(tr_e.date()),
            "test_start": str(te_s.date()),
            "test_days": args.test_days,
            **gru_m
        })

    out = pd.DataFrame(rows)
    out_path = REPORTS / f"rolling_metrics_{args.tag}_w{args.window}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")

    # 평균 요약
    if len(out):
        summary = out.groupby("model")[["MAE","RMSE","MAPE"]].mean().reset_index()
        sum_path = REPORTS / f"rolling_metrics_{args.tag}_summary_w{args.window}.csv"
        summary.to_csv(sum_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {sum_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
