from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def project_root(start: Path | None = None) -> Path:
    """
    프로젝트 루트를 찾기 위한 헬퍼.
    - pyproject.toml / requirements.txt / .git 중 하나가 있으면 그 폴더를 루트로 판단
    - 못 찾으면 start 기준 상위 2단계 fallback
    """
    if start is None:
        start = Path(__file__).resolve()

    markers = ["pyproject.toml", "requirements.txt", ".git"]
    cur = start.resolve()
    for _ in range(8):
        if any((cur / m).exists() for m in markers):
            return cur
        cur = cur.parent
    return start.parents[2]


def load_hourly_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"spot_num": str}, encoding="utf-8-sig")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 표준 컬럼 보정
    df.columns = [c.strip() for c in df.columns]
    if "model" in df.columns:
        df["model"] = df["model"].astype(str)
    return df


def save_metrics_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_npz_windows(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    make_windows.py 결과 npz 로드.
    반환: X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    """
    data = np.load(npz_path, allow_pickle=True)
    feature_cols = data["feature_cols"].tolist() if "feature_cols" in data else []
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
        feature_cols,
    )


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
