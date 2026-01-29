# src/build_dataset.py
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# CSV loader (robust encoding)
# -----------------------
def read_csv_robust(path: Path, dtype=None) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=dtype)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV read failed: {path} (tried utf-8-sig/cp949/euc-kr). Last error: {last_err}")


# -----------------------
# Load & concat raw traffic CSVs
# -----------------------
def load_raw_history(pattern: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise RuntimeError(f"raw 교통량 CSV 파일을 찾을 수 없습니다. pattern={pattern} in {RAW_DIR}")

    print("[LOAD] traffic files:")
    for f in files:
        print(" -", f.name)

    dfs = []
    for f in files:
        df = read_csv_robust(f, dtype={"spot_num": str})
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# -----------------------
# Aggregate lane/io_type
# -----------------------
def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    (spot_num, ymd, hh) 기준으로 lane_num, io_type 전부 합쳐 vol(시간당 총 교통량)
    """
    df = df.copy()
    df["spot_num"] = df["spot_num"].astype(str)
    df["ymd"] = df["ymd"].astype(str)
    df["hh"] = pd.to_numeric(df["hh"], errors="coerce").fillna(0).astype(int)
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)

    grouped = (
        df.groupby(["spot_num", "ymd", "hh"], as_index=False)["vol"]
          .sum()
          .rename(columns={"vol": "vol"})
    )

    grouped["datetime"] = pd.to_datetime(
        grouped["ymd"] + grouped["hh"].astype(str).str.zfill(2),
        format="%Y%m%d%H",
        errors="coerce",
    )
    grouped = grouped.dropna(subset=["datetime"])

    return grouped[["spot_num", "datetime", "vol"]]


# -----------------------
# Fill missing hours per spot
# -----------------------
def fill_missing_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    spot_num별로 시간축을 연속으로 만들고 없는 시간대는 vol=0
    """
    results = []
    for spot, g in df.groupby("spot_num"):
        g = g.sort_values("datetime").set_index("datetime")

        full_index = pd.date_range(start=g.index.min(), end=g.index.max(), freq="H")
        g = g.reindex(full_index)

        g["spot_num"] = spot
        g["vol"] = pd.to_numeric(g["vol"], errors="coerce").fillna(0.0)

        g = g.reset_index().rename(columns={"index": "datetime"})
        results.append(g)

    out = pd.concat(results, ignore_index=True)
    out["spot_num"] = out["spot_num"].astype(str)
    return out


# -----------------------
# Weather preprocessing
# -----------------------
def preprocess_weather(weather_csv: Path) -> pd.DataFrame:
    """
    입력 예시 컬럼(한글):
      지점, 지점명, 일시, 기온(°C), 강수량(mm), 풍속(m/s), 풍향(16방위), 습도(%),
      현지기압(hPa), 일조(hr), 적설(cm), 3시간신적설(cm), 전운량(10분위), 중하층운량(10분위),
      지면상태(지면상태코드), 지면온도(°C)
    """
    w = read_csv_robust(weather_csv)

    # 컬럼명 공백 제거
    w.columns = [str(c).strip() for c in w.columns]

    # datetime
    if "일시" not in w.columns:
        raise RuntimeError(f"weather csv에 '일시' 컬럼이 없습니다: {weather_csv}")
    w["datetime"] = pd.to_datetime(w["일시"], errors="coerce")
    w = w.dropna(subset=["datetime"]).copy()
    w = w.sort_values("datetime")

    # 숫자형 캐스팅(비어있는 값 -> NaN)
    def to_num(col):
        if col not in w.columns:
            return None
        w[col] = pd.to_numeric(w[col], errors="coerce")
        return col

    # rename -> 영문 스네이크로 통일(모델/앱에서 쓰기 편하게)
    rename_map = {}
    if "기온(°C)" in w.columns: rename_map["기온(°C)"] = "temp_c"
    if "강수량(mm)" in w.columns: rename_map["강수량(mm)"] = "rain_mm"
    if "풍속(m/s)" in w.columns: rename_map["풍속(m/s)"] = "wind_ms"
    if "풍향(16방위)" in w.columns: rename_map["풍향(16방위)"] = "wind_dir_deg"  # 실제는 '도(deg)'로 들어오는 경우가 많음
    if "습도(%)" in w.columns: rename_map["습도(%)"] = "humidity_pct"
    if "현지기압(hPa)" in w.columns: rename_map["현지기압(hPa)"] = "pressure_hpa"
    if "일조(hr)" in w.columns: rename_map["일조(hr)"] = "sunshine_hr"
    if "적설(cm)" in w.columns: rename_map["적설(cm)"] = "snow_cm"
    if "3시간신적설(cm)" in w.columns: rename_map["3시간신적설(cm)"] = "snow3h_cm"
    if "전운량(10분위)" in w.columns: rename_map["전운량(10분위)"] = "cloud_total_10"
    if "중하층운량(10분위)" in w.columns: rename_map["중하층운량(10분위)"] = "cloud_midlow_10"
    if "지면상태(지면상태코드)" in w.columns: rename_map["지면상태(지면상태코드)"] = "ground_state"
    if "지면온도(°C)" in w.columns: rename_map["지면온도(°C)"] = "ground_temp_c"

    w = w.rename(columns=rename_map)

    # numeric cast
    for c in [
        "temp_c","rain_mm","wind_ms","wind_dir_deg","humidity_pct","pressure_hpa","sunshine_hr",
        "snow_cm","snow3h_cm","cloud_total_10","cloud_midlow_10","ground_state","ground_temp_c"
    ]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    # 풍향이 있으면 sin/cos로 변환(주기성)
    if "wind_dir_deg" in w.columns:
        deg = w["wind_dir_deg"].astype(float)
        rad = np.deg2rad(deg % 360.0)
        w["wind_dir_sin"] = np.sin(rad)
        w["wind_dir_cos"] = np.cos(rad)

    # 결측 처리: 시간축을 연속으로 만들고 ffill/bfill (기상은 지점(서울 108) 1개라서 이렇게 해도 안전)
    w = w.set_index("datetime")
    full_idx = pd.date_range(start=w.index.min(), end=w.index.max(), freq="H")
    w = w.reindex(full_idx)

    # 강수량 같은 건 NaN이면 0이 자연스러움(미측정 != 0 일 수는 있지만 모델 안정성에 유리)
    if "rain_mm" in w.columns:
        w["rain_mm"] = w["rain_mm"].fillna(0.0)
    if "snow_cm" in w.columns:
        w["snow_cm"] = w["snow_cm"].fillna(0.0)
    if "snow3h_cm" in w.columns:
        w["snow3h_cm"] = w["snow3h_cm"].fillna(0.0)

    # 나머지는 앞/뒤 채우기
    w = w.ffill().bfill()

    w = w.reset_index().rename(columns={"index": "datetime"})

    # 최종 컬럼만 선택(존재하는 것만)
    keep = ["datetime"]
    for c in [
        "temp_c","rain_mm","wind_ms","wind_dir_deg","wind_dir_sin","wind_dir_cos",
        "humidity_pct","pressure_hpa","sunshine_hr","snow_cm","snow3h_cm",
        "cloud_total_10","cloud_midlow_10","ground_state","ground_temp_c"
    ]:
        if c in w.columns:
            keep.append(c)

    return w[keep]


# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traffic_pattern", default="traffic_volume_history*.csv", help="data/raw 내 교통량 파일 패턴")
    p.add_argument("--weather_csv", default="weather_seoul_108.csv", help="data/raw 내 기상 CSV 파일명")
    p.add_argument("--no_weather", action="store_true", help="기상 merge 하지 않음")
    p.add_argument("--out", default="traffic_hourly.csv", help="data/processed 출력 파일명")
    args = p.parse_args()

    print("[1] Load raw traffic history CSVs")
    raw = load_raw_history(args.traffic_pattern)

    print("[2] Aggregate lane/io_type -> hourly total")
    hourly = aggregate_hourly(raw)

    print("[3] Fill missing hours")
    hourly_full = fill_missing_hours(hourly)

    # weather merge
    if not args.no_weather:
        weather_path = RAW_DIR / args.weather_csv
        if not weather_path.exists():
            raise FileNotFoundError(f"weather csv not found: {weather_path}")
        print("[4] Load & preprocess weather")
        w = preprocess_weather(weather_path)

        print("[5] Merge weather onto hourly")
        hourly_full = hourly_full.merge(w, on="datetime", how="left")

        # 혹시 merge 이후 남는 결측이 있으면(기간 겹침 부족 등) 0/ffill로 안정화
        num_cols = [c for c in hourly_full.columns if c not in ("spot_num", "datetime")]
        for c in num_cols:
            if c in ("rain_mm","snow_cm","snow3h_cm"):
                hourly_full[c] = pd.to_numeric(hourly_full[c], errors="coerce").fillna(0.0)
            else:
                hourly_full[c] = pd.to_numeric(hourly_full[c], errors="coerce")
        hourly_full = hourly_full.sort_values(["spot_num", "datetime"])
        hourly_full[num_cols] = hourly_full.groupby("spot_num")[num_cols].ffill().bfill()

    # sort & save
    hourly_full = hourly_full.sort_values(["spot_num", "datetime"]).reset_index(drop=True)

    out = PROCESSED_DIR / args.out
    hourly_full.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {out}")
    print("shape:", hourly_full.shape)
    print("columns:", list(hourly_full.columns))
    print(hourly_full.head())


if __name__ == "__main__":
    main()
