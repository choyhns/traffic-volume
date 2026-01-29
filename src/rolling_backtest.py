# src/rolling_backtest.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# app.py에 있는 유틸을 그대로 가져다 쓰는 방식(중복 최소화)
# (프로젝트 구조가 src/app.py라면 from app import ... 로 바꿔도 됨)
from app import (
    load_artifact,
    find_logscaled_npz,
    load_npz_meta,
    build_window_features,
    transform_X_for_rnn,
    inverse_y_if_needed,
    predict_one,
    is_keras_model,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROCESSED_DIR / "traffic_hourly.csv"   # build_dataset 결과물(네 프로젝트에서 쓰는 경로로 맞춰줘)


def find_model(patterns: list[str]) -> Path | None:
    cands: list[Path] = []
    for pat in patterns:
        cands += list(MODELS_DIR.glob(pat))
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(y_true, 1e-6)))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}


def run_backtest(
    window: int = 168,
    horizon: int = 1,
    eval_days: int = 30,
    stride_hours: int = 1,
):
    if not DATA_PATH.exists():
        raise RuntimeError(f"missing: {DATA_PATH} (build_dataset 먼저 실행)")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["spot_num", "datetime"]).reset_index(drop=True)

    # 모델/메타 로드
    npz = find_logscaled_npz(window=window, horizon=horizon)
    if npz is None:
        raise RuntimeError("logscaled npz를 못 찾았어요. make_windows --window ... --horizon ... 실행 확인")

    meta = load_npz_meta(npz)
    feature_cols = meta["feature_cols"]
    if not feature_cols:
        raise RuntimeError("npz에 feature_cols가 없습니다.")

    xgb_path = find_model(["xgb*.joblib", "*xgb*.joblib"])
    gru_path = find_model(["gru*.keras", "*gru*.keras"])

    if xgb_path is None or gru_path is None:
        raise RuntimeError(f"모델을 못 찾았어요. models에 xgb*.joblib / gru*.keras 있는지 확인")

    xgb = load_artifact(str(xgb_path))
    gru = load_artifact(str(gru_path))

    # eval 범위 잡기 (각 spot별로 마지막 eval_days만)
    out_rows = []
    pred_rows = []

    for spot, g in df.groupby("spot_num"):
        g = g.sort_values("datetime").reset_index(drop=True)
        if len(g) < window + 24:
            continue

        end_dt = g["datetime"].max()
        start_eval = end_dt - pd.Timedelta(days=eval_days)

        # 최소 window 확보 위해 start_eval 앞 window 만큼 여유
        start_eval = max(start_eval, g["datetime"].min() + pd.Timedelta(hours=window + 1))

        eval_times = pd.date_range(start=start_eval, end=end_dt - pd.Timedelta(hours=horizon), freq=f"{stride_hours}H")

        y_true_list = []
        y_xgb_list = []
        y_gru_list = []

        for t in eval_times:
            # t 시점까지를 window로 써서 (t+horizon) 예측
            X_raw, wdf = build_window_features(
                g=g,
                end_dt=t,
                window=window,
                expected_feature_cols=feature_cols,
                include_time_features=True,
            )

            # 정답은 t+horizon의 실제 vol
            y_true = g.loc[g["datetime"] == (t + pd.Timedelta(hours=horizon)), "vol"]
            if y_true.empty:
                continue
            y_true = float(y_true.iloc[0])

            # --- XGB ---
            y_xgb = float(predict_one(xgb, X_raw))

            # --- GRU ---
            X_rnn = transform_X_for_rnn(X_raw, meta)

            # (주의) 만약 GRU가 spot_id embedding으로 2-input이면 여기서 분기 필요
            # 지금은 기본 1-input GRU 기준(네 app.py도 현재 predict_one이 1-input 기준) :contentReference[oaicite:2]{index=2}
            y_gru_scaled = float(predict_one(gru, X_rnn))
            y_gru = float(inverse_y_if_needed(y_gru_scaled, meta))

            y_true_list.append(y_true)
            y_xgb_list.append(y_xgb)
            y_gru_list.append(y_gru)

            pred_rows.append({
                "spot_num": spot,
                "t": t,
                "y_true": y_true,
                "y_xgb": y_xgb,
                "y_gru": y_gru,
            })

        if len(y_true_list) >= 10:
            m_xgb = regression_metrics(np.array(y_true_list), np.array(y_xgb_list))
            m_gru = regression_metrics(np.array(y_true_list), np.array(y_gru_list))
            out_rows.append({"spot_num": spot, "model": "XGBoost", **m_xgb, "n": len(y_true_list)})
            out_rows.append({"spot_num": spot, "model": "GRU", **m_gru, "n": len(y_true_list)})

    summary = pd.DataFrame(out_rows).sort_values(["model", "spot_num"])
    preds = pd.DataFrame(pred_rows).sort_values(["spot_num", "t"])

    summary_path = REPORTS_DIR / f"rolling_backtest_summary_w{window}_d{eval_days}.csv"
    preds_path = REPORTS_DIR / f"rolling_backtest_preds_w{window}_d{eval_days}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    preds.to_csv(preds_path, index=False, encoding="utf-8-sig")

    print("[SAVE]", summary_path)
    print("[SAVE]", preds_path)
    print(summary.groupby("model")[["MAE", "RMSE", "MAPE(%)"]].mean())


if __name__ == "__main__":
    run_backtest(window=168, horizon=1, eval_days=30, stride_hours=1)
