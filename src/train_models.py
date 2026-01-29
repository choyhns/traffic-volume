from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

# XGBoost optional
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Metrics
# -----------------------
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def eval_regression(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mp}


# -----------------------
# 1) Naive baseline (from traffic_hourly.csv)
# -----------------------
def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"spot_num": str}, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["spot_num", "datetime"]).reset_index(drop=True)
    return df

def time_split_hourly(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    df = df.sort_values("datetime")
    n = len(df)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:tr_end], df.iloc[tr_end:va_end], df.iloc[va_end:]

def naive_predict_last(df_test: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for spot, g in df_test.groupby("spot_num"):
        g = g.sort_values("datetime").reset_index(drop=True)
        y_true = g["vol"].to_numpy()
        y_pred = g["vol"].shift(1).to_numpy()
        mask = ~np.isnan(y_pred)
        out_rows.append(pd.DataFrame({
            "spot_num": spot,
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
        }))
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["spot_num","y_true","y_pred"])

def naive_predict_last_week(df_test: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for spot, g in df_test.groupby("spot_num"):
        g = g.sort_values("datetime").reset_index(drop=True)
        y_true = g["vol"].to_numpy()
        y_pred = g["vol"].shift(168).to_numpy()
        mask = ~np.isnan(y_pred)
        out_rows.append(pd.DataFrame({
            "spot_num": spot,
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
        }))
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["spot_num","y_true","y_pred"])

def run_naive(hourly_csv: Path):
    df = load_hourly(hourly_csv)
    _, _, test = time_split_hourly(df, 0.7, 0.15)

    last_df = naive_predict_last(test)
    week_df = naive_predict_last_week(test)

    results = []
    if len(last_df):
        results.append(("Naive_LastValue", eval_regression(last_df["y_true"], last_df["y_pred"])))
    if len(week_df):
        results.append(("Naive_LastWeek", eval_regression(week_df["y_true"], week_df["y_pred"])))
    return results


# -----------------------
# 2) NPZ loader (+ optional scaler meta)
# -----------------------
def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    feature_cols = data["feature_cols"].tolist() if "feature_cols" in data else None

    meta = {
        "target_transform": (data["target_transform"].tolist() if "target_transform" in data else ["none"]),
        "scaler_mean": float(data["scaler_mean"][0]) if "scaler_mean" in data else None,
        "scaler_std": float(data["scaler_std"][0]) if "scaler_std" in data else None,
        "scaler_vol_index": int(data["scaler_vol_index"][0]) if "scaler_vol_index" in data else None,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, meta

def flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], -1))

def inverse_target_if_needed(y_scaled: np.ndarray, meta: dict) -> np.ndarray:
    """
    make_windows.py에서 y에 log1p+standardize를 적용한 경우:
      y_scaled -> expm1(y_scaled*std + mean)
    """
    tt = meta.get("target_transform", ["none"])
    tt0 = tt[0] if isinstance(tt, list) and len(tt) else str(tt)

    if "log1p" not in str(tt0).lower():
        return np.asarray(y_scaled).reshape(-1)

    mean = meta.get("scaler_mean", None)
    std = meta.get("scaler_std", None)
    if mean is None or std is None:
        raise RuntimeError("npz에 logscaled target_transform이 있는데 scaler_mean/std가 없습니다.")

    y = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
    y = y * std + mean
    y = np.expm1(y)
    return y.astype(np.float32)


# -----------------------
# 3) XGBoost (or fallback) using RAW window npz
# -----------------------
def run_xgb_or_fallback(npz_path: Path, model_name: str):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, meta = load_npz(npz_path)

    # y is (N, horizon). We assume horizon=1
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)

    Xtr = flatten_windows(X_train)
    Xva = flatten_windows(X_val)
    Xte = flatten_windows(X_test)

    if XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=0,
            importance_type="gain",  # ✅ 중요도 산출용
        )
        model.fit(
            Xtr, y_train,
            eval_set=[(Xva, y_val)],
            verbose=False,
        )
        y_pred = model.predict(Xte)

        import joblib
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, out)
        return {"model": model_name, **eval_regression(y_test, y_pred)}
    else:
        model = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        import joblib
        out = MODELS_DIR / f"{model_name}_fallback.joblib"
        joblib.dump(model, out)
        return {"model": f"{model_name}_fallback", **eval_regression(y_test, y_pred)}


# -----------------------
# 4) LSTM / GRU with spot_id Embedding (LOGSCALED npz + RAW spot_id)
# -----------------------
def _extract_spot_id_from_raw_npz(npz_raw: Path, expected_feature_cols: list[str]):
    """
    spot_id는 '정수'로 Embedding에 넣어야 하므로
    (표준화가 섞일 수 있는) logscaled npz가 아니라 raw npz에서 뽑는다.
    """
    X_train, _, X_val, _, X_test, _, feature_cols_raw, _ = load_npz(npz_raw)
    if not feature_cols_raw:
        raise RuntimeError("raw npz에 feature_cols가 없습니다. make_windows에서 feature_cols 저장이 필요합니다.")

    if feature_cols_raw != expected_feature_cols:
        raise RuntimeError(
            "raw npz와 rnn npz의 feature_cols가 다릅니다.\n"
            f"- raw: {feature_cols_raw}\n"
            f"- rnn: {expected_feature_cols}\n"
            "같은 make_windows 설정(window/horizon/tf/sid/wx)으로 만든 파일인지 확인하세요."
        )

    if "spot_id" not in feature_cols_raw:
        return None, None, None  # spot_id 없음

    idx = feature_cols_raw.index("spot_id")

    # spot_id는 모든 timestep에 같은 값이므로 첫 timestep만 사용
    s_tr = np.rint(X_train[:, 0, idx]).astype(np.int32)
    s_va = np.rint(X_val[:, 0, idx]).astype(np.int32)
    s_te = np.rint(X_test[:, 0, idx]).astype(np.int32)
    return (s_tr, s_va, s_te)

def _drop_spot_id_feature(X: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    if "spot_id" not in feature_cols:
        return X, feature_cols
    idx = feature_cols.index("spot_id")
    X2 = np.delete(X, idx, axis=2)
    cols2 = [c for i, c in enumerate(feature_cols) if i != idx]
    return X2, cols2

def build_rnn_with_spot_embedding(
    kind: str,
    timesteps: int,
    n_seq_features: int,
    num_spots: int,
    embed_dim: int = 8,
    hidden: int = 64,
    dropout: float = 0.2,
):
    """
    입력 2개:
      - seq_in: (T, F_seq)  (spot_id 제외한 시계열 feature)
      - spot_in: () int32   (Embedding)
    spot embedding을 RepeatVector로 (T, embed_dim)로 늘려 seq feature에 concat 후 RNN 통과.
    """
    seq_in = layers.Input(shape=(timesteps, n_seq_features), name="seq_in")
    spot_in = layers.Input(shape=(), dtype="int32", name="spot_id")

    emb = layers.Embedding(input_dim=num_spots, output_dim=embed_dim, name="spot_embedding")(spot_in)
    emb = layers.Flatten()(emb)               # (B, embed_dim)
    emb = layers.RepeatVector(timesteps)(emb) # (B, T, embed_dim)

    x = layers.Concatenate(axis=-1)([seq_in, emb])  # (B, T, F_seq + embed_dim)

    if kind.lower() == "lstm":
        x = layers.LSTM(hidden, return_sequences=False)(x)
    elif kind.lower() == "gru":
        x = layers.GRU(hidden, return_sequences=False)(x)
    else:
        raise ValueError("kind must be lstm or gru")

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = models.Model(inputs=[seq_in, spot_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def build_gru_attention_with_spot_embedding(
    timesteps: int,
    n_seq_features: int,
    num_spots: int,
    embed_dim: int = 8,
    hidden: int = 64,
    dropout: float = 0.2,
):
    """
    입력 2개:
      - seq_in: (T, F_seq)  (spot_id 제외)
      - spot_in: () int32

    GRU(return_sequences=True, return_state=True) 후
    query=마지막 state, key/value=전체 시퀀스 출력으로 AdditiveAttention 적용.
    """
    seq_in = layers.Input(shape=(timesteps, n_seq_features), name="seq_in")
    spot_in = layers.Input(shape=(), dtype="int32", name="spot_id")

    emb = layers.Embedding(input_dim=num_spots, output_dim=embed_dim, name="spot_embedding")(spot_in)
    emb = layers.Flatten()(emb)                 # (B, E)
    emb = layers.RepeatVector(timesteps)(emb)   # (B, T, E)

    x = layers.Concatenate(axis=-1)([seq_in, emb])  # (B,T,F+E)

    seq_out, last_state = layers.GRU(hidden, return_sequences=True, return_state=True)(x)

    q = layers.Reshape((1, hidden))(last_state)  # (B,1,H)
    ctx = layers.AdditiveAttention()([q, seq_out])  # (B,1,H)
    ctx = layers.Flatten()(ctx)  # (B,H)

    z = layers.Concatenate()([ctx, last_state])  # (B,2H)
    z = layers.Dropout(dropout)(z)
    z = layers.Dense(64, activation="relu")(z)
    out = layers.Dense(1)(z)

    model = models.Model(inputs=[seq_in, spot_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def build_rnn(kind: str, timesteps: int, n_features: int, hidden: int = 64, dropout: float = 0.2):
    """
    (호환) spot_id가 없을 때 기존 1-input RNN
    """
    inp = layers.Input(shape=(timesteps, n_features))
    if kind.lower() == "lstm":
        x = layers.LSTM(hidden, return_sequences=False)(inp)
    elif kind.lower() == "gru":
        x = layers.GRU(hidden, return_sequences=False)(inp)
    else:
        raise ValueError("kind must be lstm or gru")

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


# -----------------------
# 5) XGB importance -> RNN feature selection helpers
# -----------------------
def aggregate_xgb_importance_by_feature(importances: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """
    flatten된 importances (T*F,) 를 feature_cols (F,) 기준으로 timestep 방향으로 합산.
    flatten 순서: [t0_f0..t0_fF-1, t1_f0..]
    """
    F = len(feature_cols)
    imp = np.asarray(importances).reshape(-1)
    if imp.size % F != 0:
        raise ValueError(f"importance size {imp.size} is not divisible by num features {F}")

    sums = np.zeros(F, dtype=np.float64)
    for idx, v in enumerate(imp):
        f = idx % F
        sums[f] += float(v)

    df = pd.DataFrame({"feature": feature_cols, "importance": sums})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    total = df["importance"].sum()
    df["importance_pct"] = (df["importance"] / total * 100.0) if total > 0 else 0.0
    return df

def choose_rnn_features_from_importance(
    imp_df: pd.DataFrame,
    topk: int,
    always_keep: tuple[str, ...] = (),
    drop: tuple[str, ...] = ("wind_dir_deg",),
) -> list[str]:
    """
    topk 기반으로 feature 선택.
    - drop: 원천적으로 제외할 컬럼
    - always_keep: topk와 무관하게 항상 유지할 컬럼
    """
    if topk <= 0:
        return []

    drop_set = set(drop)
    base = [f for f in imp_df["feature"].tolist() if f not in drop_set]

    picked = []
    for f in base:
        if f in picked:
            continue
        picked.append(f)
        if len(picked) >= topk:
            break

    # always_keep를 앞으로 보장
    for f in reversed(always_keep):
        if f in drop_set:
            continue
        if f in picked:
            picked.remove(f)
        picked.insert(0, f)

    out = []
    seen = set()
    for f in picked:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out

def filter_X_by_feature_cols(X: np.ndarray, feature_cols: list[str], keep_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    X: (N,T,F) 에서 keep_cols만 남긴다.
    """
    if not keep_cols:
        return X, feature_cols

    idxs = [feature_cols.index(c) for c in keep_cols if c in feature_cols]
    if not idxs:
        return X, feature_cols

    X2 = X[:, :, idxs]
    cols2 = [feature_cols[i] for i in idxs]
    return X2, cols2


# -----------------------
# 6) Run RNN (supports keep_cols + spot embedding + Attention GRU)
# -----------------------
def run_rnn(
    npz_rnn_path: Path,
    kind: str,
    model_name: str,
    epochs: int = 20,
    batch_size: int = 256,
    npz_raw_path_for_spot: Path | None = None,
    spot_embed_dim: int = 8,
    keep_cols: list[str] | None = None,
):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, meta = load_npz(npz_rnn_path)

    if not feature_cols:
        raise RuntimeError("rnn npz에 feature_cols가 없습니다. make_windows에서 feature_cols 저장이 필요합니다.")

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    use_spot_embedding = ("spot_id" in feature_cols) and (npz_raw_path_for_spot is not None)

    if use_spot_embedding:
        s_tr, s_va, s_te = _extract_spot_id_from_raw_npz(npz_raw_path_for_spot, feature_cols)
        if s_tr is None:
            use_spot_embedding = False
        else:
            # spot_id는 시계열 입력에서 제거(Embedding으로만 사용)
            X_train_seq, cols_seq = _drop_spot_id_feature(X_train, feature_cols)
            X_val_seq, _ = _drop_spot_id_feature(X_val, feature_cols)
            X_test_seq, _ = _drop_spot_id_feature(X_test, feature_cols)

            # ✅ XGB importance 기반 feature subset 적용
            if keep_cols:
                X_train_seq, cols_seq = filter_X_by_feature_cols(X_train_seq, cols_seq, keep_cols)
                X_val_seq, _ = filter_X_by_feature_cols(X_val_seq, cols_seq, keep_cols)
                X_test_seq, _ = filter_X_by_feature_cols(X_test_seq, cols_seq, keep_cols)

            timesteps = X_train_seq.shape[1]
            n_seq_features = X_train_seq.shape[2]
            num_spots = int(max(s_tr.max(), s_va.max(), s_te.max()) + 1)

            if kind.lower() == "gru_attn":
                model = build_gru_attention_with_spot_embedding(
                    timesteps=timesteps,
                    n_seq_features=n_seq_features,
                    num_spots=num_spots,
                    embed_dim=spot_embed_dim,
                    hidden=64,
                    dropout=0.2,
                )
            else:
                model = build_rnn_with_spot_embedding(
                    kind=kind,
                    timesteps=timesteps,
                    n_seq_features=n_seq_features,
                    num_spots=num_spots,
                    embed_dim=spot_embed_dim,
                    hidden=64,
                    dropout=0.2,
                )

            cbs = [
                callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
            ]

            model.fit(
                [X_train_seq, s_tr], y_train,
                validation_data=([X_val_seq, s_va], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=cbs,
                verbose=1,
            )

            y_pred_scaled = model.predict([X_test_seq, s_te], batch_size=batch_size, verbose=0).reshape(-1)
            y_true_scaled = y_test.reshape(-1)

            y_pred = inverse_target_if_needed(y_pred_scaled, meta)
            y_true = inverse_target_if_needed(y_true_scaled, meta)

            out = MODELS_DIR / f"{model_name}.keras"
            model.save(out)

            return {"model": model_name, **eval_regression(y_true, y_pred)}

    # ---- fallback: 기존 1-input RNN ----
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_rnn(kind, timesteps, n_features, hidden=64, dropout=0.2)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    y_pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    y_true_scaled = y_test.reshape(-1)

    y_pred = inverse_target_if_needed(y_pred_scaled, meta)
    y_true = inverse_target_if_needed(y_true_scaled, meta)

    out = MODELS_DIR / f"{model_name}.keras"
    model.save(out)

    return {"model": model_name, **eval_regression(y_true, y_pred)}


# -----------------------
# Main: run all + compare
# -----------------------
def main():
    p = argparse.ArgumentParser()

    # split npz
    p.add_argument("--npz_raw", default="", help="XGB용 raw npz (예: windows_*_raw.npz)")
    p.add_argument("--npz_rnn", default="", help="LSTM/GRU용 logscaled npz (예: windows_*_logscaled.npz)")

    # backward compat
    p.add_argument("--npz", default="", help="(호환용) 하나만 줄 때 raw/rnn 공통으로 사용")

    p.add_argument("--hourly_csv", default=str(PROCESSED / "traffic_hourly.csv"), help="traffic_hourly.csv 경로")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--tag", default="", help="결과/모델 파일명에 붙일 태그(예: w24, w168)")

    # spot embedding
    p.add_argument("--spot_embed_dim", type=int, default=8, help="spot_id embedding 차원(기본 8)")

    # NEW: feature selection + attention
    p.add_argument("--rnn_topk", type=int, default=0, help="XGB 중요도 기반 RNN feature top-k (0이면 미적용)")
    p.add_argument("--train_attn_gru", action="store_true", help="GRU+Attention 모델도 함께 학습")

    args = p.parse_args()

    # resolve npz paths
    if args.npz:
        npz_raw = Path(args.npz)
        npz_rnn = Path(args.npz)
    else:
        if not args.npz_raw or not args.npz_rnn:
            raise RuntimeError("npz 경로가 필요합니다: --npz_raw & --npz_rnn (또는 호환용 --npz)")
        npz_raw = Path(args.npz_raw)
        npz_rnn = Path(args.npz_rnn)

    hourly_csv = Path(args.hourly_csv)

    if not npz_raw.exists():
        raise FileNotFoundError(f"npz_raw not found: {npz_raw}")
    if not npz_rnn.exists():
        raise FileNotFoundError(f"npz_rnn not found: {npz_rnn}")

    rows = []

    # Naive
    naive_results = run_naive(hourly_csv)
    for name, metrics in naive_results:
        rows.append({"model": name, **metrics})

    tag = f"_{args.tag}" if args.tag else ""

    # XGB uses RAW
    rows.append(run_xgb_or_fallback(npz_raw, model_name=f"XGB{tag}"))

    # --- XGB importance -> export + RNN feature subset 선택 ---
    rnn_keep_cols = None
    try:
        import joblib
        xgb_path = MODELS_DIR / f"XGB{tag}.joblib"
        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)

            X_train_raw, _, _, _, _, _, feature_cols_raw, _ = load_npz(npz_raw)
            if feature_cols_raw and getattr(xgb_model, "feature_importances_", None) is not None:
                imp = xgb_model.feature_importances_

                # imp size should be T*F
                F = len(feature_cols_raw)
                expected = X_train_raw.shape[1] * F
                if len(imp) == expected:
                    imp_df = aggregate_xgb_importance_by_feature(imp, feature_cols_raw)
                    out_imp = REPORTS / f"xgb_feature_importance{tag}.csv"
                    imp_df.to_csv(out_imp, index=False, encoding="utf-8-sig")
                    print(f"[SAVE] {out_imp}")

                    if args.rnn_topk > 0:
                        always = ("vol", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend")
                        picked = choose_rnn_features_from_importance(
                            imp_df,
                            topk=args.rnn_topk,
                            always_keep=always,
                            drop=("spot_id",)  # ✅ RNN 시계열 입력에서는 어차피 제외
                        )
                        rnn_keep_cols = picked
                        print(f"[RNN FEATURE TOP{args.rnn_topk}] {rnn_keep_cols}")
                else:
                    print(f"[WARN] feature_importances_ size mismatch: got={len(imp)} expected={expected}")
    except Exception as e:
        print("[WARN] XGB importance export failed:", e)

    # RNN uses LOGSCALED + RAW spot_id (if present)
    rows.append(run_rnn(
        npz_rnn, "lstm",
        model_name=f"LSTM{tag}",
        epochs=args.epochs,
        batch_size=args.batch_size,
        npz_raw_path_for_spot=npz_raw,
        spot_embed_dim=args.spot_embed_dim,
        keep_cols=rnn_keep_cols,
    ))
    rows.append(run_rnn(
        npz_rnn, "gru",
        model_name=f"GRU{tag}",
        epochs=args.epochs,
        batch_size=args.batch_size,
        npz_raw_path_for_spot=npz_raw,
        spot_embed_dim=args.spot_embed_dim,
        keep_cols=rnn_keep_cols,
    ))

    if args.train_attn_gru:
        rows.append(run_rnn(
            npz_rnn, "gru_attn",
            model_name=f"GRU_ATT{tag}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            npz_raw_path_for_spot=npz_raw,
            spot_embed_dim=args.spot_embed_dim,
            keep_cols=rnn_keep_cols,
        ))

    df = pd.DataFrame(rows).sort_values("RMSE")
    out_csv = REPORTS / f"metrics{tag if tag else ''}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n=== Metrics (sorted by RMSE) ===")
    print(df.to_string(index=False))
    print(f"\n[SAVE] {out_csv}")
    print(f"[MODELS] saved to {MODELS_DIR}")
    print(f"[NPZ] raw={npz_raw}")
    print(f"[NPZ] rnn={npz_rnn}")


if __name__ == "__main__":
    main()
