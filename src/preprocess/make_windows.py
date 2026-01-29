# src/make_windows.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = PROCESSED_DIR / "traffic_hourly.csv"


# -----------------------
# Feature engineering
# -----------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df = df.copy()
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    month = dt.dt.month

    df["is_weekend"] = (dow >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


def ensure_wind_dir_sincos(df: pd.DataFrame) -> pd.DataFrame:
    """
    wind_dir_deg가 있으면 sin/cos 생성.
    (하지만 feature로는 wind_dir_deg는 사용하지 않음!)
    """
    df = df.copy()
    if "wind_dir_deg" in df.columns and ("wind_dir_sin" not in df.columns or "wind_dir_cos" not in df.columns):
        deg = pd.to_numeric(df["wind_dir_deg"], errors="coerce").fillna(0.0).astype(float)
        rad = np.deg2rad(deg % 360.0)
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    return df


# -----------------------
# Weather: 최소셋 + Δ(변화량)
# -----------------------
def weather_min_cols(df: pd.DataFrame) -> list[str]:
    """
    성능/안정성 우선의 최소 기상 feature 세트
    - wind_dir는 deg를 쓰지 않고 sin/cos만 사용
    - 너무 많은 기상 변수를 넣지 않음(노이즈/과적합 방지)
    """
    candidates = [
        "temp_c",
        "rain_mm",
        "wind_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "humidity_pct",
        "pressure_hpa",
        "snow_cm",
        "cloud_total_10",
    ]
    return [c for c in candidates if c in df.columns]


def add_delta_features_per_spot(df: pd.DataFrame, cols: list[str], suffix: str = "_d1") -> pd.DataFrame:
    """
    spot별 시간순 정렬 후 1시간 변화량(현재-이전)을 추가.
    """
    df = df.copy()
    df = df.sort_values(["spot_num", "datetime"])
    for c in cols:
        # wind_dir_sin/cos는 변화량이 의미가 약해서 제외하는 걸 추천
        if c in ("wind_dir_sin", "wind_dir_cos"):
            continue
        new_c = f"{c}{suffix}"
        df[new_c] = df.groupby("spot_num")[c].diff(1)
    return df


# -----------------------
# Windowing
# -----------------------
def make_windows_for_one_spot(
    g: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = g[feature_cols].to_numpy(dtype=np.float32)
    target = g[target_col].to_numpy(dtype=np.float32)

    n = len(g)
    xs, ys = [], []
    max_i = n - window - horizon + 1
    for i in range(0, max_i, stride):
        xs.append(values[i: i + window])
        ys.append(target[i + window: i + window + horizon])

    X = np.stack(xs) if xs else np.empty((0, window, len(feature_cols)), dtype=np.float32)
    y = np.stack(ys) if ys else np.empty((0, horizon), dtype=np.float32)
    return X, y


def make_windows(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    stride: int,
    include_time_features: bool,
    include_weather: bool,
    include_spot_id: bool,
    delta_suffix: str = "_d1",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["spot_num"] = df["spot_num"].astype(str)
    df = df.sort_values(["spot_num", "datetime"])

    # vol numeric
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0)

    # time features
    if include_time_features:
        df = add_time_features(df)

    # wind sin/cos ensure
    df = ensure_wind_dir_sincos(df)

    # weather cols (min set)
    wcols = weather_min_cols(df) if include_weather else []

    # numeric cast + missing fill (rain/snow=0, others ffill/bfill)
    rain_like = {"rain_mm", "snow_cm"}
    for c in wcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c in rain_like:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = df.groupby("spot_num")[c].ffill().bfill().fillna(0.0)

    # delta features
    if include_weather and wcols:
        df = add_delta_features_per_spot(df, wcols, suffix=delta_suffix)
        # delta도 numeric + 결측 0
        for c in [f"{c}{delta_suffix}" for c in wcols if c not in ("wind_dir_sin", "wind_dir_cos")]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # optional spot_id
    if include_spot_id:
        spot_to_id = {s: i for i, s in enumerate(sorted(df["spot_num"].unique()))}
        df["spot_id"] = df["spot_num"].map(spot_to_id).astype(int)

    # feature cols 구성
    feature_cols = ["vol"]

    if include_time_features:
        feature_cols += [
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "is_weekend",
        ]

    if include_weather:
        # ✅ wind_dir_deg는 절대 넣지 않음
        feature_cols += wcols
        # delta cols
        delta_cols = [f"{c}{delta_suffix}" for c in wcols if c not in ("wind_dir_sin", "wind_dir_cos")]
        feature_cols += [c for c in delta_cols if c in df.columns]

    if include_spot_id:
        feature_cols += ["spot_id"]

    # 혹시 남아 있는 결측/비정상 값 안정화
    for c in feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # target
    target_col = "vol"

    X_all, y_all = [], []
    for _, g in df.groupby("spot_num"):
        g = g.sort_values("datetime")
        X, y = make_windows_for_one_spot(
            g=g,
            feature_cols=feature_cols,
            target_col=target_col,
            window=window,
            horizon=horizon,
            stride=stride,
        )
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    X_all = np.concatenate(X_all, axis=0) if X_all else np.empty((0, window, len(feature_cols)), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.empty((0, horizon), dtype=np.float32)
    return X_all, y_all, feature_cols


# -----------------------
# Time split (no leakage)
# -----------------------
def time_split_by_datetime(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


# -----------------------
# RNN scaling: log1p + feature-wise standardize
# -----------------------
def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def fit_feature_scaler(
    X_train_raw: np.ndarray,
    feature_cols: list[str],
    log1p_cols: set[str],
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    메모리 폭발 방지:
      - 전체를 (N*T,F)로 한 번에 만들지 않고, chunk 단위로 sum/sumsq 누적해서 mean/std 계산
      - log1p는 지정 컬럼만 적용(비음수만)
      - spot_id는 스케일 대상에서 제외(Embedding용 정수)
    """
    X = X_train_raw  # float32 (권장)
    N, T, F = X.shape

    # log1p mask
    log1p_mask = np.array([1 if c in log1p_cols else 0 for c in feature_cols], dtype=np.int32)

    # spot_id 제외 (scale 하면 안 됨)
    if "spot_id" in feature_cols:
        sid_idx = feature_cols.index("spot_id")
        log1p_mask[sid_idx] = 0  # log1p도 제외

    sums = np.zeros(F, dtype=np.float64)
    sumsqs = np.zeros(F, dtype=np.float64)
    total = 0

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        xb = X[start:end].astype(np.float64, copy=True)  # chunk만 float64

        # log1p on selected features (non-negative only)
        for j in range(F):
            if int(log1p_mask[j]) == 1:
                xb[:, :, j] = np.log1p(np.maximum(xb[:, :, j], 0.0))

        flat = xb.reshape(-1, F)  # chunk 단위라 안전
        sums += flat.sum(axis=0)
        sumsqs += (flat * flat).sum(axis=0)
        total += flat.shape[0]

    mean = sums / max(total, 1)
    var = (sumsqs / max(total, 1)) - mean**2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    # spot_id는 표준화 제외: mean=0, std=1로 두면 transform 시 값 유지 가능
    if "spot_id" in feature_cols:
        mean[sid_idx] = 0.0
        std[sid_idx] = 1.0

    return mean.astype(np.float32), std.astype(np.float32), log1p_mask


def transform_X_scaled_inplace(
    X_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    log1p_mask: np.ndarray,
) -> np.ndarray:
    """
    메모리 절약:
      - X_raw를 복사하지 않고(in-place) log1p + 표준화 적용
    """
    X = X_raw  # float32 array (in-place)
    F = X.shape[2]

    # log1p
    for j in range(F):
        if int(log1p_mask[j]) == 1:
            np.maximum(X[:, :, j], 0.0, out=X[:, :, j])
            np.log1p(X[:, :, j], out=X[:, :, j])

    # standardize (in-place)
    X -= mean.reshape(1, 1, F)
    X /= (std.reshape(1, 1, F) + 1e-8)
    return X



def fit_target_scaler_from_y(y_train_raw: np.ndarray) -> tuple[float, float]:
    """
    y(=vol) 스케일: log1p + standardize
    """
    y = y_train_raw.reshape(-1).astype(np.float64)
    y = _safe_log1p(y)
    mean = float(y.mean())
    std = float(y.std() + 1e-8)
    return mean, std


def transform_y_scaled(y_raw: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    y = y_raw.astype(np.float64, copy=True)
    y = _safe_log1p(y)
    y = (y - y_mean) / y_std
    return y.astype(np.float32)


# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--window", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--no_time_features", action="store_true")
    p.add_argument("--no_weather", action="store_true", help="기상 feature 제외")
    p.add_argument("--include_spot_id", action="store_true")

    p.add_argument("--out_prefix", default="", help="출력 파일 prefix(비우면 자동)")

    args = p.parse_args()

    if not IN_PATH.exists():
        raise RuntimeError(f"입력 파일이 없습니다: {IN_PATH} (먼저 build_dataset.py 실행)")

    df = pd.read_csv(IN_PATH, dtype={"spot_num": str}, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])

    include_time_features = not args.no_time_features
    include_weather = not args.no_weather

    # (1) time split first (no leakage)
    train_df, val_df, test_df = time_split_by_datetime(df, train_ratio=0.7, val_ratio=0.15)

    # (2) RAW windows (for XGB / baseline)
    X_train_raw, y_train_raw, feature_cols = make_windows(
        train_df,
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        include_time_features=include_time_features,
        include_weather=include_weather,
        include_spot_id=args.include_spot_id,
        delta_suffix="_d1",
    )
    X_val_raw, y_val_raw, _ = make_windows(
        val_df,
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        include_time_features=include_time_features,
        include_weather=include_weather,
        include_spot_id=args.include_spot_id,
        delta_suffix="_d1",
    )
    X_test_raw, y_test_raw, _ = make_windows(
        test_df,
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        include_time_features=include_time_features,
        include_weather=include_weather,
        include_spot_id=args.include_spot_id,
        delta_suffix="_d1",
    )

    tf_tag = "tf" if include_time_features else "notf"
    sid_tag = "sid" if args.include_spot_id else "nosid"
    w_tag = "wx" if include_weather else "nowx"
    prefix = args.out_prefix.strip() or f"windows_w{args.window}_h{args.horizon}_{tf_tag}_{sid_tag}_{w_tag}"

    # save raw
    out_raw = PROCESSED_DIR / f"{prefix}_raw.npz"
    np.savez_compressed(
        out_raw,
        X_train=X_train_raw, y_train=y_train_raw,
        X_val=X_val_raw, y_val=y_val_raw,
        X_test=X_test_raw, y_test=y_test_raw,
        feature_cols=np.array(feature_cols, dtype=object),
        target_transform=np.array(["none"], dtype=object),
        note=np.array(["raw: wind_dir_deg excluded, weather=minset, deltas included"], dtype=object),
    )
    print(f"[SAVE] RAW windows: {out_raw}")
    print("[INFO] feature_cols:", feature_cols)

    # (3) RNN windows: log1p + feature-wise standardize
    # log1p 적용할 입력 컬럼(비음수/스케일 큰 것 위주)
    log1p_cols = {"vol", "rain_mm", "snow_cm"}  # 필요시 추가 가능
    # (주의) delta 컬럼은 음수가 가능하므로 log1p 제외

    feat_mean, feat_std, log1p_mask = fit_feature_scaler(X_train_raw, feature_cols, log1p_cols, chunk_size=512)

    # in-place 변환 (메모리 절약)
    X_train_rnn = transform_X_scaled_inplace(X_train_raw, feat_mean, feat_std, log1p_mask)
    X_val_rnn   = transform_X_scaled_inplace(X_val_raw,   feat_mean, feat_std, log1p_mask)
    X_test_rnn  = transform_X_scaled_inplace(X_test_raw,  feat_mean, feat_std, log1p_mask)


    # y scaling (log1p + standardize)
    y_mean, y_std = fit_target_scaler_from_y(y_train_raw)
    y_train_rnn = transform_y_scaled(y_train_raw, y_mean, y_std)
    y_val_rnn = transform_y_scaled(y_val_raw, y_mean, y_std)
    y_test_rnn = transform_y_scaled(y_test_raw, y_mean, y_std)

    out_rnn = PROCESSED_DIR / f"{prefix}_logscaled.npz"
    np.savez_compressed(
        out_rnn,
        X_train=X_train_rnn, y_train=y_train_rnn,
        X_val=X_val_rnn, y_val=y_val_rnn,
        X_test=X_test_rnn, y_test=y_test_rnn,
        feature_cols=np.array(feature_cols, dtype=object),

        # ✅ train_models.py가 inverse할 때 쓰는 y 스케일(기존 키 유지)
        target_transform=np.array(["log1p+standardize(all_features)"], dtype=object),
        scaler_mean=np.array([y_mean], dtype=np.float32),
        scaler_std=np.array([y_std], dtype=np.float32),

        # ✅ app/추가분석용: feature-wise 스케일러(새 키)
        feature_scaler_mean=feat_mean.astype(np.float32),
        feature_scaler_std=feat_std.astype(np.float32),
        feature_log1p_mask=log1p_mask.astype(np.int32),

        note=np.array([
            "rnn: wind_dir_deg excluded, weather=minset, deltas included, "
            "X=log1p(select)+feature-wise standardize, y=log1p+standardize"
        ], dtype=object),
    )
    print(f"[SAVE] RNN windows (scaled): {out_rnn}")
    print(f"[INFO] y_mean={y_mean:.6f}, y_std={y_std:.6f}")
    print(f"[INFO] feature scaler saved: mean/std shape = {feat_mean.shape}/{feat_std.shape}")


if __name__ == "__main__":
    main()
