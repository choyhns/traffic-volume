from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import pandas as pd

from src.utils.io import load_metrics_csv, save_json, project_root


def infer_window_from_tag(name: str) -> int | None:
    """
    model명에 w24, w168 같은 태그가 있으면 window 추정
    """
    m = re.search(r"w(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def infer_model_artifact_path(models_dir: Path, model_name: str) -> Path | None:
    """
    모델 이름으로 저장된 파일(.keras/.joblib)을 찾아 반환.
    train_models.py 저장 규칙에 맞춰 최대한 찾아봄.
    """
    candidates = [
        models_dir / f"{model_name}.keras",
        models_dir / f"{model_name}.joblib",
        models_dir / f"{model_name}_fallback.joblib",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 혹시 model_name에 공백/특수문자 등이 있을 경우를 대비해 느슨하게 검색
    safe = re.sub(r"[^A-Za-z0-9_\-]", "", model_name)
    loose = list(models_dir.glob(f"*{safe}*"))
    for p in loose:
        if p.suffix.lower() in [".keras", ".joblib"]:
            return p
    return None


def choose_best(df: pd.DataFrame, metric: str = "RMSE") -> pd.Series:
    """
    기본 선정 규칙:
    1) metric 최소 (기본 RMSE)
    2) MAPE, MAE 순으로 tie-break
    """
    for col in [metric, "MAPE", "MAE"]:
        if col not in df.columns:
            raise RuntimeError(f"metrics 파일에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # float 변환
    work = df.copy()
    for c in [metric, "MAPE", "MAE"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[metric])
    if len(work) == 0:
        raise RuntimeError("유효한 metrics 행이 없습니다(모든 RMSE가 NaN).")

    # 정렬: RMSE -> MAPE -> MAE
    work = work.sort_values([metric, "MAPE", "MAE"], ascending=[True, True, True]).reset_index(drop=True)
    return work.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports", help="metrics.csv들이 있는 폴더")
    ap.add_argument("--models_dir", default="models", help="학습 모델 파일(.keras/.joblib) 폴더")
    ap.add_argument("--pattern", default="metrics*.csv", help="metrics 파일 glob 패턴")
    ap.add_argument("--metric", default="RMSE", help="선정 기준 (기본 RMSE)")
    ap.add_argument("--out", default="models/final_model.json", help="최종 모델 메타 저장 경로")
    args = ap.parse_args()

    root = project_root(Path(__file__).resolve())
    reports_dir = (root / args.reports_dir).resolve()
    models_dir = (root / args.models_dir).resolve()
    out_path = (root / args.out).resolve()

    files = sorted(reports_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"'{reports_dir}'에서 '{args.pattern}'에 매칭되는 파일이 없습니다.")

    dfs = []
    for f in files:
        df = load_metrics_csv(f)
        df["metrics_file"] = f.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    if "model" not in all_df.columns:
        raise RuntimeError(f"metrics 파일에 'model' 컬럼이 없습니다. 현재 컬럼: {list(all_df.columns)}")

    best = choose_best(all_df, metric=args.metric)
    model_name = str(best["model"])
    artifact = infer_model_artifact_path(models_dir, model_name)

    final = {
        "final_model_name": model_name,
        "artifact_path": str(artifact) if artifact else None,
        "selected_by": args.metric,
        "window": infer_window_from_tag(model_name),
        "horizon": 1,
        "metrics": {
            "MAE": float(best.get("MAE")) if pd.notna(best.get("MAE")) else None,
            "RMSE": float(best.get("RMSE")) if pd.notna(best.get("RMSE")) else None,
            "MAPE": float(best.get("MAPE")) if pd.notna(best.get("MAPE")) else None,
        },
        "source_metrics_file": str(best.get("metrics_file")),
        "selected_at": datetime.now().isoformat(timespec="seconds"),
    }

    save_json(out_path, final)

    print("=== Final model selected ===")
    print(f"- model: {final['final_model_name']}")
    print(f"- artifact: {final['artifact_path']}")
    print(f"- metric: {final['selected_by']}")
    print(f"- window: {final['window']}")
    print(f"- from: {final['source_metrics_file']}")
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
