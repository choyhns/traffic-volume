# app.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# optional deps
try:
    import joblib
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False

try:
    import tensorflow as tf  # noqa: F401
    TF_OK = True
except Exception:
    TF_OK = False

try:
    from pyproj import Transformer
    PYPROJ_OK = True
except Exception:
    PYPROJ_OK = False


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROCESSED_DIR / "traffic_hourly.csv"
SPOTS_PATH = ROOT / "data" / "raw" / "traffic_spots_20240101_20241231_D46_F10.csv"


# ----------------------------
# Feature helpers (time)
# ----------------------------
_TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
_ALWAYS_KEEP = ["vol"] + _TIME_FEATURES + ["wind_dir_sin", "wind_dir_cos"]


def add_time_features(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    d = df.copy()
    dt = pd.to_datetime(d[dt_col])

    hour = dt.dt.hour.astype(int)
    dow = dt.dt.dayofweek.astype(int)  # 0=Mon
    month = dt.dt.month.astype(int)

    d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    d["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    d["is_weekend"] = (dow >= 5).astype(int)
    return d


def ensure_wind_dir_sincos(df: pd.DataFrame) -> pd.DataFrame:
    """
    학습 feature_cols에 wind_dir_sin/cos가 있을 수 있으니
    원본에 wind_dir_deg가 있으면 sin/cos 생성해줌.
    """
    d = df.copy()
    if "wind_dir_deg" in d.columns:
        deg = pd.to_numeric(d["wind_dir_deg"], errors="coerce").fillna(0.0).to_numpy()
        rad = np.deg2rad(deg)
        d["wind_dir_sin"] = np.sin(rad)
        d["wind_dir_cos"] = np.cos(rad)
    return d


# ----------------------------
# Load meta from npz
# ----------------------------
def pick_npz_for_window(window: int, kind: str, want_sid: bool | None = None) -> Path | None:
    """
    kind: "raw" or "rnn"(=logscaled)
    want_sid:
      - True  : _sid_ 포함 npz 우선
      - False : _nosid_ 포함 npz 우선
      - None  : sid/nosid 상관없이 최신
    파일명 예:
      windows_w168_h1_tf_sid_wx_raw.npz
      windows_w168_h1_tf_nosid_wx_logscaled.npz
    """
    kind = kind.lower()
    if kind not in ("raw", "rnn"):
        raise ValueError("kind must be 'raw' or 'rnn'")

    suffix = "raw.npz" if kind == "raw" else "logscaled.npz"

    def _gather(patterns: list[str]) -> list[Path]:
        cands: list[Path] = []
        for pat in patterns:
            cands += list(PROCESSED_DIR.glob(pat))
        return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)

    # sid/nosid 우선순위 패턴 구성
    if want_sid is True:
        pats = [
            f"windows_w{window}_h1*_sid_*{suffix}",
            f"*w{window}*_sid_*{suffix}",
        ]
    elif want_sid is False:
        pats = [
            f"windows_w{window}_h1*_nosid_*{suffix}",
            f"*w{window}*_nosid_*{suffix}",
        ]
    else:
        pats = [
            f"windows_w{window}_h1*{suffix}",
            f"*w{window}*{suffix}",
        ]

    cands = _gather(pats)
    return cands[0] if cands else None



def load_npz_meta(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)

    feature_cols = d["feature_cols"].tolist() if "feature_cols" in d else None
    target_transform = d["target_transform"].tolist() if "target_transform" in d else ["none"]

    # y scaler (RNN logscaled일 때만 의미 있음)
    y_mean = float(d["scaler_mean"][0]) if "scaler_mean" in d else None
    y_std = float(d["scaler_std"][0]) if "scaler_std" in d else None

    # feature-wise scaler (RNN logscaled일 때만 의미 있음)
    feat_mean = d["feature_scaler_mean"] if "feature_scaler_mean" in d else None
    feat_std = d["feature_scaler_std"] if "feature_scaler_std" in d else None
    log1p_mask = d["feature_log1p_mask"] if "feature_log1p_mask" in d else None

    return {
        "npz_path": str(npz_path),
        "feature_cols": feature_cols,
        "target_transform": target_transform,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "feature_log1p_mask": log1p_mask,
    }


# ----------------------------
# Model I/O
# ----------------------------
def is_keras_model(model) -> bool:
    return "keras" in str(type(model)).lower() or "tensorflow" in str(type(model)).lower()


def load_artifact(artifact_path: str):
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found: {p}")

    if p.suffix.lower() == ".joblib":
        if not JOBLIB_OK:
            raise RuntimeError("joblib이 설치되어 있지 않습니다.")
        return joblib.load(p)

    if p.suffix.lower() == ".keras":
        if not TF_OK:
            raise RuntimeError("tensorflow가 설치되어 있지 않습니다.")
        import tensorflow as tf
        return tf.keras.models.load_model(p)

    raise RuntimeError(f"Unsupported model artifact type: {p.suffix}")


def predict_one(model, X: np.ndarray, feature_cols: list[str] | None = None) -> float:
    """
    X: (1, window, F)
    - Keras(1-input): model.predict(X)
    - Keras(2-input): [X_seq, spot_id]
    - XGB/Sklearn: flatten 후 predict
    """
    if is_keras_model(model):
        n_in = len(getattr(model, "inputs", []))

        if n_in == 1:
            y = model.predict(X, verbose=0)
            return float(np.asarray(y).reshape(-1)[0])

        if n_in == 2:
            if feature_cols is None or "spot_id" not in feature_cols:
                raise RuntimeError("2-input 모델인데 feature_cols에 spot_id가 없습니다.")
            sid_idx = feature_cols.index("spot_id")
            spot_id = int(np.rint(X[0, 0, sid_idx]))
            X_seq = np.delete(X, sid_idx, axis=2)
            y = model.predict([X_seq, np.array([spot_id], dtype=np.int32)], verbose=0)
            return float(np.asarray(y).reshape(-1)[0])

        raise RuntimeError(f"지원하지 않는 keras 입력 개수: {n_in}")

    # sklearn/xgb 계열
    Xf = X.reshape((X.shape[0], -1))
    y = model.predict(Xf)
    return float(np.asarray(y).reshape(-1)[0])


# ----------------------------
# Meta auto-align helpers
# ----------------------------
def _get_expected_feature_count_for_model(model, window: int) -> tuple[int | None, bool]:
    """
    returns: (expected_F_total, needs_spot_id)
    - Keras 1-input: expected_F_total = input_shape[-1]
    - Keras 2-input: expected_F_total = seq_F + 1(spot_id)
    - sklearn/xgb: n_features_in_ = window*F -> expected_F_total = F
    """
    if is_keras_model(model):
        try:
            n_in = len(getattr(model, "inputs", []))
        except Exception:
            n_in = 1

        if n_in == 1:
            shp = getattr(model, "input_shape", None)
            if shp is None:
                return None, False
            return int(shp[-1]), False

        if n_in == 2:
            try:
                shp0 = tuple(model.inputs[0].shape)  # (None, T, F_seq)
                seq_f = int(shp0[-1])
                return int(seq_f + 1), True
            except Exception:
                return None, True

        return None, False

    # sklearn/xgb
    n_in = getattr(model, "n_features_in_", None)
    if n_in is None:
        booster = getattr(model, "get_booster", None)
        if callable(booster):
            try:
                b = model.get_booster()
                n_in = int(b.num_features())
            except Exception:
                n_in = None

    if n_in is None:
        return None, False

    if window > 0 and int(n_in) % int(window) == 0:
        return int(int(n_in) // int(window)), False

    return int(n_in), False


def _select_feature_cols_to_match(
    meta_cols: list[str],
    needed_f: int,
    needs_spot_id: bool,
    data_cols: list[str],
) -> tuple[list[str], list[str], list[int]]:
    cols = list(meta_cols)
    dropped: list[str] = []

    # 1) spot_id 처리 (모델이 안 쓰면 제거)
    if (not needs_spot_id) and ("spot_id" in cols) and (len(cols) > needed_f):
        cols.remove("spot_id")
        dropped.append("spot_id")

    # 2) 너무 많으면: 우선 유지 후보를 제외한 것부터 제거
    keep_set = set(_ALWAYS_KEEP) | set(data_cols)
    if len(cols) > needed_f:
        removable = [c for c in cols if c not in keep_set]
        for c in removable:
            if len(cols) <= needed_f:
                break
            cols.remove(c)
            dropped.append(c)

    # 3) 그래도 많으면 뒤에서부터(vol은 최대한 유지)
    if len(cols) > needed_f:
        for c in list(reversed(cols)):
            if len(cols) <= needed_f:
                break
            if c == "vol":
                continue
            cols.remove(c)
            dropped.append(c)

    meta_idx = {c: i for i, c in enumerate(meta_cols)}
    selected_indices = [meta_idx[c] for c in cols if c in meta_idx]
    return cols, dropped, selected_indices


def prepare_meta_for_model(model, meta: dict, window: int, data_cols: list[str]) -> tuple[dict, dict]:
    meta_cols = meta.get("feature_cols") or ["vol"]
    exp_f, needs_spot_id = _get_expected_feature_count_for_model(model, window)

    info = {
        "expected_F": exp_f,
        "meta_F": len(meta_cols),
        "needs_spot_id": needs_spot_id,
        "dropped_cols": [],
        "note": "",
    }

    if exp_f is None:
        info["note"] = "모델에서 expected feature 수를 읽지 못해 meta(feature_cols) 그대로 사용합니다."
        return meta, info

    if len(meta_cols) == exp_f:
        return meta, info

    selected_cols, dropped_cols, sel_idx = _select_feature_cols_to_match(
        meta_cols=meta_cols,
        needed_f=exp_f,
        needs_spot_id=needs_spot_id,
        data_cols=data_cols,
    )
    info["dropped_cols"] = dropped_cols

    meta2 = dict(meta)
    meta2["feature_cols"] = selected_cols

    # (중요) RNN 스케일러도 같이 자르기 (길이 일치할 때만)
    fm = meta.get("feature_mean")
    fs = meta.get("feature_std")
    lm = meta.get("feature_log1p_mask")

    if fm is not None and fs is not None:
        try:
            fm = np.asarray(fm)
            fs = np.asarray(fs)
            if fm.shape[0] == len(meta_cols) and fs.shape[0] == len(meta_cols) and len(sel_idx) == len(selected_cols):
                meta2["feature_mean"] = fm[sel_idx]
                meta2["feature_std"] = fs[sel_idx]
                if lm is not None:
                    lm = np.asarray(lm).reshape(-1)
                    if lm.shape[0] == len(meta_cols):
                        meta2["feature_log1p_mask"] = lm[sel_idx]
        except Exception:
            pass

    info["note"] = f"meta feature_cols({len(meta_cols)}) -> model expected({exp_f}) 로 자동 보정"
    return meta2, info


# ----------------------------
# Scaling (RNN logscaled)
# ----------------------------
def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def apply_feature_scaling_if_needed(X: np.ndarray, meta: dict) -> np.ndarray:
    feat_mean = meta.get("feature_mean")
    feat_std = meta.get("feature_std")
    log1p_mask = meta.get("feature_log1p_mask")

    if feat_mean is None or feat_std is None:
        return X.astype(np.float32)

    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)

    B, T, F = X.shape
    if feat_mean.shape[0] != F or feat_std.shape[0] != F:
        return X.astype(np.float32)

    X = X.astype(np.float32)

    if log1p_mask is not None:
        log1p_mask = np.asarray(log1p_mask).reshape(-1)
        if log1p_mask.shape[0] == F:
            for j in range(F):
                if int(log1p_mask[j]) == 1:
                    X[:, :, j] = _safe_log1p(X[:, :, j])

    X = (X - feat_mean.reshape(1, 1, F)) / (feat_std.reshape(1, 1, F) + 1e-8)
    return X.astype(np.float32)


def inverse_y_if_needed(y_pred: float, meta: dict) -> float:
    y_mean = meta.get("y_mean")
    y_std = meta.get("y_std")
    if y_mean is None or y_std is None:
        return float(y_pred)
    y_log = float(y_pred) * float(y_std) + float(y_mean)
    return float(np.expm1(y_log))


# ----------------------------
# Build window features
# ----------------------------
def build_window_features(
    g: pd.DataFrame,
    end_dt: pd.Timestamp,
    window: int,
    expected_feature_cols: list[str],
    include_time_features: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    g2 = g.copy()
    g2["datetime"] = pd.to_datetime(g2["datetime"])
    g2 = g2.sort_values("datetime").reset_index(drop=True)

    start_dt = end_dt - pd.Timedelta(hours=window - 1)
    idx = pd.date_range(start_dt, end_dt, freq="H")

    wdf = pd.DataFrame({"datetime": idx})
    wdf = wdf.merge(g2, on="datetime", how="left")

    # vol
    if "vol" in wdf.columns:
        wdf["vol"] = pd.to_numeric(wdf["vol"], errors="coerce").fillna(0.0)
    else:
        wdf["vol"] = 0.0

    # time features
    if include_time_features:
        tfeat = add_time_features(wdf, "datetime")
        for c in _TIME_FEATURES:
            if c in expected_feature_cols:
                wdf[c] = tfeat[c].values

    wdf = ensure_wind_dir_sincos(wdf)

    # fill expected features
    for c in expected_feature_cols:
        if c not in wdf.columns:
            wdf[c] = 0.0
        wdf[c] = pd.to_numeric(wdf[c], errors="coerce").fillna(0.0)

    X = wdf[expected_feature_cols].to_numpy(dtype=np.float32).reshape(1, window, len(expected_feature_cols))
    return X, wdf


def predict_next_k_hours(
    model,
    meta: dict,
    g: pd.DataFrame,
    window: int,
    k: int = 24,
    use_gru: bool = False,
) -> pd.DataFrame:
    expected_feature_cols = meta.get("feature_cols") or ["vol"]

    g_roll = g.copy()
    g_roll["datetime"] = pd.to_datetime(g_roll["datetime"])
    g_roll = g_roll.sort_values("datetime").reset_index(drop=True)

    cur_end = pd.to_datetime(g_roll["datetime"].max())

    preds = []
    for _step in range(1, k + 1):
        X, _wdf = build_window_features(
            g_roll,
            end_dt=cur_end,
            window=window,
            expected_feature_cols=expected_feature_cols,
            include_time_features=True,
        )

        if use_gru:
            X_in = apply_feature_scaling_if_needed(X, meta)
            y_hat = predict_one(model, X_in, feature_cols=expected_feature_cols)
            y_pred = inverse_y_if_needed(y_hat, meta)
        else:
            y_hat = predict_one(model, X, feature_cols=expected_feature_cols)
            y_pred = float(y_hat)

        next_dt = cur_end + pd.Timedelta(hours=1)
        y_out = float(max(y_pred, 0.0))
        preds.append({"datetime": next_dt, "pred_vol": y_out})

        # append for rolling
        new_row = {"datetime": next_dt, "vol": y_out}
        for c in g_roll.columns:
            if c not in new_row:
                new_row[c] = np.nan
        g_roll = pd.concat([g_roll, pd.DataFrame([new_row])], ignore_index=True)
        cur_end = next_dt

    return pd.DataFrame(preds)


# ----------------------------
# Geo helper
# ----------------------------
def tm_to_wgs84(df: pd.DataFrame) -> pd.DataFrame:
    if not PYPROJ_OK or df.empty:
        return df

    d = df.copy()
    xcol = None
    ycol = None
    for cand in ["x", "tm_x", "TM_X", "X"]:
        if cand in d.columns:
            xcol = cand
            break
    for cand in ["y", "tm_y", "TM_Y", "Y"]:
        if cand in d.columns:
            ycol = cand
            break

    if xcol is None or ycol is None:
        return d

    transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)

    xs = pd.to_numeric(d[xcol], errors="coerce").fillna(0.0).to_numpy()
    ys = pd.to_numeric(d[ycol], errors="coerce").fillna(0.0).to_numpy()
    lons, lats = transformer.transform(xs, ys)
    d["lon"] = lons
    d["lat"] = lats
    return d


# ----------------------------
# Model list for sidebar select
# ----------------------------
def list_models_for_window(window: int, kind: str, require_sid: bool = False) -> list[Path]:
    """
    window/kind 기준으로 models 폴더에서 후보 모델 목록 반환.
    - require_sid=True면 파일명에 "_sid_"가 들어간 모델만 반환
    - 최신 수정 시간 순으로 정렬(최신이 맨 위)
    """
    kind = kind.upper()
    if kind == "XGB":
        ext = ".joblib"
        base_pats = ["XGB", "xgb"]
    elif kind == "GRU":
        ext = ".keras"
        base_pats = ["GRU", "gru"]
    else:
        return []

    pats: list[str] = []
    for pref in base_pats:
        if require_sid:
            pats += [
                f"{pref}*_w{window}_sid_*{ext}",
                f"{pref}*w{window}_sid_*{ext}",
                f"{pref}*{window}_sid_*{ext}",
            ]
        else:
            pats += [
                f"{pref}*_w{window}*{ext}",
                f"{pref}*w{window}*{ext}",
                f"{pref}*{window}*{ext}",
            ]

    cands: list[Path] = []
    for pat in pats:
        cands += list(MODELS_DIR.glob(pat))

    uniq = list({p.resolve(): p for p in cands}.values())
    uniq = sorted(uniq, key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq


# ----------------------------
# Streamlit main
# ----------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="Traffic Volume Forecast", layout="wide")
    st.title("교통량 예측 (XGBoost vs GRU) - horizon=1 재귀 예측")

    # ---- Load data
    if not DATA_PATH.exists():
        st.error(f"데이터가 없습니다: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    if "datetime" not in df.columns:
        st.error("traffic_hourly.csv에 datetime 컬럼이 필요합니다.")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if "spot_num" not in df.columns:
        st.error("traffic_hourly.csv에 spot_num 컬럼이 필요합니다.")
        st.stop()

    # ---- Spots
    spots_df = pd.DataFrame()
    if SPOTS_PATH.exists():
        spots_df = pd.read_csv(SPOTS_PATH)

        if "spot_num" not in spots_df.columns:
            for cand in ["SPOT_NUM", "spotId", "spot_id"]:
                if cand in spots_df.columns:
                    spots_df = spots_df.rename(columns={cand: "spot_num"})
                    break

        if "spot_name" not in spots_df.columns:
            for cand in ["spot_nm", "SPOT_NM", "spotName", "name"]:
                if cand in spots_df.columns:
                    spots_df = spots_df.rename(columns={cand: "spot_name"})
                    break

        spots_df = tm_to_wgs84(spots_df)
    else:
        st.warning(f"spots 파일이 없습니다: {SPOTS_PATH} (지도 탭은 제한될 수 있음)")

    # ---- Sidebar controls
    st.sidebar.header("설정")
    window = st.sidebar.selectbox("윈도우 길이(window)", options=[24, 48, 72, 168, 336], index=3)
    ahead = st.sidebar.slider("예측 길이(시간)", min_value=6, max_value=48, value=24, step=1)

    # ---- Pick models for this window (sidebar select)
    st.sidebar.subheader("모델 선택")
    only_sid = st.sidebar.checkbox(f"w{window}_sid 모델만 보기", value=True)

    xgb_cands = list_models_for_window(window, "XGB", require_sid=only_sid)
    gru_cands = list_models_for_window(window, "GRU", require_sid=only_sid)

    if only_sid and (not xgb_cands or not gru_cands):
        st.sidebar.warning("sid 모델이 부족해서 전체 모델에서 선택합니다.")
        xgb_cands = list_models_for_window(window, "XGB", require_sid=False)
        gru_cands = list_models_for_window(window, "GRU", require_sid=False)

    if not xgb_cands:
        st.error(f"window={window}에 대한 XGB 모델을 찾지 못했습니다. (models/*.joblib)")
        st.stop()
    if not gru_cands:
        st.error(f"window={window}에 대한 GRU 모델을 찾지 못했습니다. (models/*.keras)")
        st.stop()

    xgb_p = st.sidebar.selectbox("XGB 모델", options=xgb_cands, format_func=lambda p: p.name)
    gru_p = st.sidebar.selectbox("GRU 모델", options=gru_cands, format_func=lambda p: p.name)

    want_sid = ("_sid_" in xgb_p.name.lower()) or ("_sid_" in gru_p.name.lower())
    
    # ---- Load metas (IMPORTANT: split by model type)
    npz_xgb = pick_npz_for_window(window, "raw", want_sid=want_sid)
    npz_gru = pick_npz_for_window(window, "rnn", want_sid=want_sid)

    if npz_xgb is None:
        st.error(f"window={window} XGB raw npz를 찾지 못했습니다.")
        st.stop()
    if npz_gru is None:
        st.error(f"window={window} GRU logscaled npz를 찾지 못했습니다.")
        st.stop()

    meta_xgb_base = load_npz_meta(npz_xgb)
    meta_gru_base = load_npz_meta(npz_gru)

    if npz_xgb is None:
        st.error(f"window={window}에 해당하는 XGB 메타(raw npz)를 찾지 못했습니다. data/processed/windows_*_raw.npz 확인")
        st.stop()
    if npz_gru is None:
        st.error(f"window={window}에 해당하는 GRU 메타(rnn/log npz)를 찾지 못했습니다. data/processed/windows_*_rnn/log*.npz 확인")
        st.stop()

    meta_xgb_base = load_npz_meta(npz_xgb)
    meta_gru_base = load_npz_meta(npz_gru)
    
    @st.cache_resource(show_spinner=False)
    def load_models_cached(xgb_path: str, gru_path: str):
        return load_artifact(xgb_path), load_artifact(gru_path)

    xgb_model, gru_model = load_models_cached(str(xgb_p), str(gru_p))

    st.sidebar.caption(f"선택된 XGB: {xgb_p.name}")
    st.sidebar.caption(f"선택된 GRU: {gru_p.name}")
    st.sidebar.caption(f"메타(XGB raw): {Path(meta_xgb_base['npz_path']).name}")
    st.sidebar.caption(f"메타(GRU rnn): {Path(meta_gru_base['npz_path']).name}")

    # ---- Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["1) 데이터 개요", "2) 성능 비교", "3) 예측 비교", "4) 지도 시각화"])

    with tab1:
        st.subheader("데이터 개요")
        st.write(f"- rows: {len(df):,}")
        st.write(f"- 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
        st.write(f"- spot 수: {df['spot_num'].nunique():,}")
        st.dataframe(df.head(50), use_container_width=True)

    with tab2:
        st.subheader("모델 성능 비교 (reports/metrics*.csv)")
        metric_files = sorted(REPORTS_DIR.glob("metrics*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not metric_files:
            st.info("reports/metrics*.csv 를 찾지 못했습니다. train_models 실행 후 생성되는지 확인하세요.")
        else:
            mf = st.selectbox("metrics 파일 선택", options=metric_files, format_func=lambda p: p.name)
            mdf = pd.read_csv(mf)
            st.caption(f"로드: {mf.name}")

            model_col = "model" if "model" in mdf.columns else ("name" if "name" in mdf.columns else None)
            if model_col is None:
                st.dataframe(mdf, use_container_width=True)
            else:
                xgb_rows = mdf[mdf[model_col].astype(str).str.contains("XGB|xgb", regex=True)]
                gru_rows = mdf[mdf[model_col].astype(str).str.contains("GRU|gru", regex=True)]

                st.write("**XGB 후보**")
                st.dataframe(xgb_rows, use_container_width=True)

                st.write("**GRU 후보**")
                st.dataframe(gru_rows, use_container_width=True)

                st.info("발표용으로는 XGB가 1등, GRU는 딥러닝 대표로 비교 스토리로 가면 좋아요.")

    with tab3:
        st.subheader("XGBoost vs GRU: horizon=1 재귀 예측")

        spot_list = sorted(df["spot_num"].unique().tolist())
        spot = st.selectbox("지점(spot_num) 선택", options=spot_list, index=0)

        g = df[df["spot_num"] == spot].copy().sort_values("datetime").reset_index(drop=True)
        if g.empty:
            st.warning("선택한 지점 데이터가 없습니다.")
        else:
            st.write(f"- 선택 지점 rows: {len(g):,} / 마지막 관측: {g['datetime'].max()}")

            # ✅ 모델별 meta base가 다르다!
            meta_xgb, info_xgb = prepare_meta_for_model(xgb_model, meta_xgb_base, window, data_cols=list(g.columns))
            meta_gru, info_gru = prepare_meta_for_model(gru_model, meta_gru_base, window, data_cols=list(g.columns))

            with st.expander("디버그: feature_cols 자동 보정 상태", expanded=False):
                st.write("**XGB meta 정합**", info_xgb)
                st.write("**GRU meta 정합**", info_gru)
                if info_xgb.get("dropped_cols"):
                    st.info(f"XGB 입력 shape 맞추기 위해 drop된 cols: {info_xgb['dropped_cols']}")
                if info_gru.get("dropped_cols"):
                    st.info(f"GRU 입력 shape 맞추기 위해 drop된 cols: {info_gru['dropped_cols']}")

                # 기대 feature 수 확인용(개발)
                exp_fxgb, _ = _get_expected_feature_count_for_model(xgb_model, window)
                exp_fgru, _ = _get_expected_feature_count_for_model(gru_model, window)
                st.write({"xgb_expected_F": exp_fxgb, "gru_expected_F": exp_fgru})

            if st.button("예측 실행 (XGB + GRU)"):
                with st.spinner("예측 중..."):
                    pred_xgb = predict_next_k_hours(xgb_model, meta_xgb, g, window=window, k=ahead, use_gru=False)
                    pred_gru = predict_next_k_hours(gru_model, meta_gru, g, window=window, k=ahead, use_gru=True)

                out = pred_xgb.merge(pred_gru, on="datetime", how="left", suffixes=("_xgb", "_gru"))
                out = out.rename(columns={"pred_vol_xgb": "XGBoost", "pred_vol_gru": "GRU"})

                st.write(f"**예측 결과(미래 {ahead}시간)**")
                st.dataframe(out, use_container_width=True)
                st.line_chart(out.set_index("datetime")[["XGBoost", "GRU"]])
                st.caption("※ GRU는 scaling+inverse 적용. XGB는 raw 스케일 예측.")

    with tab4:
        st.subheader("지도 시각화 (spot별 예측값 표시)")

        if spots_df.empty or ("lat" not in spots_df.columns) or ("lon" not in spots_df.columns):
            st.warning("spots 파일에 lat/lon이 없어 지도 시각화를 할 수 없습니다. TM좌표(x,y) 또는 lat/lon 컬럼을 확인하세요.")
        else:
            which = st.radio("지도에 표시할 모델", options=["XGBoost", "GRU"], horizontal=True)
            model = xgb_model if which == "XGBoost" else gru_model
            use_gru = (which == "GRU")

            step_hour = st.slider("몇 시간 뒤 값을 지도에 표시할까요? (t+N)", min_value=1, max_value=ahead, value=1, step=1)

            if st.button("지도 예측 실행"):
                rows = []
                with st.spinner("spot별 예측 중... (spot이 많으면 시간이 걸릴 수 있어요)"):
                    for spot_num in sorted(df["spot_num"].unique().tolist()):
                        g2 = df[df["spot_num"] == spot_num].copy().sort_values("datetime").reset_index(drop=True)
                        if len(g2) < window:
                            continue

                        base_meta = meta_gru_base if use_gru else meta_xgb_base
                        meta_aligned, _info = prepare_meta_for_model(model, base_meta, window, data_cols=list(g2.columns))

                        pred_df = predict_next_k_hours(model, meta_aligned, g2, window=window, k=step_hour, use_gru=use_gru)
                        y = float(pred_df["pred_vol"].iloc[-1])
                        rows.append({"spot_num": spot_num, "pred_vol": y})

                pred_all = pd.DataFrame(rows)
                m = spots_df.merge(pred_all, on="spot_num", how="left")

                st.caption("pred_vol이 클수록 혼잡(점이 진하게 표시)")
                st.map(m[["lat", "lon", "pred_vol"]].rename(columns={"lat": "lat", "lon": "lon"}), zoom=11)

                show_cols = [c for c in ["spot_num", "spot_name", "pred_vol", "lat", "lon"] if c in m.columns]
                st.dataframe(m[show_cols].sort_values("pred_vol", ascending=False).head(50), use_container_width=True)


if __name__ == "__main__":
    main()
