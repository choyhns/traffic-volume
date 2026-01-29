from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable
from datetime import datetime, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv
import xml.etree.ElementTree as ET


# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://openapi.seoul.go.kr:8088"


# -------------------------
# Helpers
# -------------------------
def env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"환경변수 {name}가 없습니다. .env를 확인하세요.")
    return v


def daterange_ymd(start_ymd: str, end_ymd: str) -> List[str]:
    """YYYYMMDD inclusive"""
    s = datetime.strptime(start_ymd, "%Y%m%d").date()
    e = datetime.strptime(end_ymd, "%Y%m%d").date()
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def build_url(api_key: str, fmt: str, service: str, start: int, end: int, tail: str = "") -> str:
    # 서울 열린데이터광장 표준 형식:
    # http://openapi.seoul.go.kr:8088/{KEY}/{TYPE}/{SERVICE}/{START}/{END}/(OPTIONAL_TAIL)
    tail = tail.lstrip("/")
    return f"{BASE_URL}/{api_key}/{fmt}/{service}/{start}/{end}/" + (tail if tail else "")


def request_text(url: str, timeout: int = 20, retries: int = 3, backoff: float = 0.5) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            # 대부분 utf-8, 간혹 선언이 있으니 그대로 text 사용
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"요청 실패: {url}\n마지막 에러: {last_err}")


# -------------------------
# XML parsing
# -------------------------
def _child_text(elem: ET.Element, tag: str) -> Optional[str]:
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_openapi_xml(xml_text: str) -> Dict[str, Any]:
    """
    서울 열린데이터광장 XML 공통 구조:
    <SERVICE>
      <list_total_count>...</list_total_count>
      <RESULT>
        <CODE>INFO-000</CODE>
        <MESSAGE>정상 처리되었습니다</MESSAGE>
      </RESULT>
      <row>...</row>
      <row>...</row>
    </SERVICE>
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        # html 에러 페이지가 오는 경우도 있어서 일부 출력
        snippet = xml_text[:200].replace("\n", " ")
        raise RuntimeError(f"XML 파싱 실패: {e}\n응답 앞부분: {snippet}")

    # root.tag가 서비스명인 경우가 많음 (VolInfo, SpotInfo)
    service_tag = root.tag

    result = {
        "service": service_tag,
        "list_total_count": None,
        "code": None,
        "message": None,
        "rows": []
    }

    # RESULT
    res = root.find("RESULT")
    if res is not None:
        result["code"] = _child_text(res, "CODE")
        result["message"] = _child_text(res, "MESSAGE")

    # total count
    ltc = root.find("list_total_count")
    if ltc is not None and ltc.text is not None:
        try:
            result["list_total_count"] = int(ltc.text.strip())
        except Exception:
            result["list_total_count"] = None

    # rows
    rows = []
    for row in root.findall("row"):
        d: Dict[str, Any] = {}
        for child in list(row):
            # tag: 컬럼명, text: 값
            key = child.tag
            val = child.text.strip() if child.text else ""
            d[key] = val
        rows.append(d)

    result["rows"] = rows
    return result


# -------------------------
# Checkpoint helpers
# -------------------------
def load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# SpotInfo (XML)
# -------------------------
def fetch_spots_xml(
    api_key: str,
    spot_service: str,
    fmt: str = "xml",
    page_size: int = 1000,
    timeout: int = 20,
    retries: int = 3,
    sleep_sec: float = 0.05,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    SpotInfo는 보통 /SpotInfo/1/1000/ 형태로 페이지네이션 가능.
    """
    all_rows: List[Dict[str, Any]] = []
    start = 1
    end = page_size
    total_count: Optional[int] = None

    while True:
        url = build_url(api_key, fmt, spot_service, start, end)
        xml_text = request_text(url, timeout=timeout, retries=retries)
        parsed = parse_openapi_xml(xml_text)

        # 코드 체크 (정상이 아닐 때)
        if parsed["code"] and parsed["code"] != "INFO-000":
            raise RuntimeError(f"SpotInfo API 에러: {parsed['code']} / {parsed['message']}")

        if total_count is None and parsed["list_total_count"] is not None:
            total_count = parsed["list_total_count"]

        rows = parsed["rows"]
        if not rows:
            break

        all_rows.extend(rows)
        time.sleep(sleep_sec)

        if total_count and len(all_rows) >= total_count:
            break

        start += page_size
        end += page_size

    df = pd.DataFrame(all_rows)

    # 타입 정리
    if "spot_num" in df.columns:
        df["spot_num"] = df["spot_num"].astype(str)

    for c in ["grs80tm_x", "grs80tm_y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if save_path:
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] spots -> {save_path} ({len(df):,} rows)")

    return df


# -------------------------
# VolInfo (XML)
# -------------------------
def fetch_volinfo_one_xml(
    api_key: str,
    volume_service: str,
    spot_num: str,
    ymd: str,
    hh: str,
    fmt: str = "xml",
    timeout: int = 20,
    retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    샘플:
    /VolInfo/1/5/A-01/20160301/12/
    """
    tail = f"{spot_num}/{ymd}/{hh}/"
    # end는 넉넉히 1000 (대부분 row는 많지 않음)
    url = build_url(api_key, fmt, volume_service, 1, 1000, tail=tail)

    xml_text = request_text(url, timeout=timeout, retries=retries)
    parsed = parse_openapi_xml(xml_text)

    # 정상 코드가 아니면 빈 리스트 처리(쿼터/미존재 시간 등 케이스를 유연하게)
    if parsed["code"] and parsed["code"] != "INFO-000":
        return []

    return parsed["rows"]


def collect_volume_history_xml(
    api_key: str,
    volume_service: str,
    spots: pd.DataFrame,
    start_ymd: str,
    end_ymd: str,
    hours: Iterable[int],
    fmt: str = "xml",
    timeout: int = 20,
    retries: int = 3,
    sleep_sec: float = 0.05,
    out_csv: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    limit_spots: Optional[int] = None,
    spot_filter: Optional[List[str]] = None,
) -> Path:
    """
    (지점 × 날짜 × 시간) 루프 수집 후 CSV에 append.
    resume를 위해 checkpoint 저장.
    """
    if out_csv is None:
        out_csv = RAW_DIR / "traffic_volume_history.csv"
    if checkpoint_path is None:
        checkpoint_path = RAW_DIR / "traffic_volume_history.checkpoint.json"

    # 지점 목록 준비
    spot_nums = spots["spot_num"].astype(str).tolist() if "spot_num" in spots.columns else []
    if spot_filter:
        spot_set = set(map(str, spot_filter))
        spot_nums = [s for s in spot_nums if s in spot_set]
    if limit_spots is not None:
        spot_nums = spot_nums[:limit_spots]

    if not spot_nums:
        raise RuntimeError("수집할 SPOT_NUM이 없습니다. SpotInfo 수집 결과를 확인하세요.")

    dates = daterange_ymd(start_ymd, end_ymd)
    hh_list = [f"{h:02d}" for h in hours]

    # 체크포인트 로드
    ck = load_checkpoint(checkpoint_path) or {}
    start_i = ck.get("spot_index", 0)
    start_date_i = ck.get("date_index", 0)
    start_hh_i = ck.get("hour_index", 0)

    print(f"[INFO] output csv: {out_csv}")
    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] spots: {len(spot_nums)} | dates: {len(dates)} | hours: {len(hh_list)}")
    if ck:
        print(f"[RESUME] spot_index={start_i}, date_index={start_date_i}, hour_index={start_hh_i}")

    header_written = out_csv.exists()
    total_calls = 0
    total_rows = 0

    for si in range(start_i, len(spot_nums)):
        spot = spot_nums[si]

        di0 = start_date_i if si == start_i else 0
        for di in range(di0, len(dates)):
            ymd = dates[di]

            hi0 = start_hh_i if (si == start_i and di == di0) else 0
            for hi in range(hi0, len(hh_list)):
                hh = hh_list[hi]

                rows = fetch_volinfo_one_xml(
                    api_key=api_key,
                    volume_service=volume_service,
                    spot_num=spot,
                    ymd=ymd,
                    hh=hh,
                    fmt=fmt,
                    timeout=timeout,
                    retries=retries,
                )
                total_calls += 1

                if rows:
                    df = pd.DataFrame(rows)

                    # 혹시 응답에 없으면 보강
                    if "spot_num" not in df.columns:
                        df["spot_num"] = spot
                    if "ymd" not in df.columns:
                        df["ymd"] = ymd
                    if "hh" not in df.columns:
                        df["hh"] = hh

                    # 숫자 컬럼 정리(있을 때만)
                    for c in ["hh", "lane_num", "vol"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")

                    if not header_written:
                        df.to_csv(out_csv, index=False, encoding="utf-8-sig", mode="w")
                        header_written = True
                    else:
                        df.to_csv(out_csv, index=False, encoding="utf-8-sig", mode="a", header=False)

                    total_rows += len(df)

                # checkpoint 저장
                save_checkpoint(checkpoint_path, {
                    "spot_index": si,
                    "date_index": di,
                    "hour_index": hi + 1,
                    "spot_num": spot,
                    "ymd": ymd,
                    "hh": hh,
                    "total_calls": total_calls,
                    "total_rows": total_rows,
                    "updated_at": datetime.now().isoformat(timespec="seconds")
                })

                time.sleep(sleep_sec)

            # 날짜 변경 시 hour 초기화
            start_hh_i = 0
            save_checkpoint(checkpoint_path, {
                "spot_index": si,
                "date_index": di + 1,
                "hour_index": 0,
                "spot_num": spot,
                "ymd": ymd,
                "hh": None,
                "total_calls": total_calls,
                "total_rows": total_rows,
                "updated_at": datetime.now().isoformat(timespec="seconds")
            })

        # spot 변경 시 date 초기화
        start_date_i = 0
        save_checkpoint(checkpoint_path, {
            "spot_index": si + 1,
            "date_index": 0,
            "hour_index": 0,
            "spot_num": None,
            "ymd": None,
            "hh": None,
            "total_calls": total_calls,
            "total_rows": total_rows,
            "updated_at": datetime.now().isoformat(timespec="seconds")
        })

    print(f"[DONE] calls={total_calls:,}, rows={total_rows:,}")
    return out_csv


def merge_spots_into_history(history_csv: Path, spots_csv: Path, out_csv: Optional[Path] = None) -> Path:
    if out_csv is None:
        out_csv = RAW_DIR / "traffic_history_with_spots_20240101_20241231_D46_F10.csv"

    hist = pd.read_csv(history_csv, dtype={"spot_num": str}, encoding="utf-8-sig")
    spots = pd.read_csv(spots_csv, dtype={"spot_num": str}, encoding="utf-8-sig")

    merged = hist.merge(spots, on="spot_num", how="left")

    # datetime 생성
    if "YMD" in merged.columns and "HH" in merged.columns:
        merged["YMD"] = merged["YMD"].astype(str)
        merged["HH"] = merged["HH"].astype("Int64").astype(str).str.zfill(2)
        merged["datetime"] = pd.to_datetime(merged["YMD"] + merged["HH"], format="%Y%m%d%H", errors="coerce")

    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] merged -> {out_csv} ({len(merged):,} rows)")
    return out_csv


# -------------------------
# CLI
# -------------------------
def main():
    load_dotenv()

    api_key = env_required("SEOUL_API_KEY")
    volume_service = env_required("SEOUL_VOLUME_SERVICE")  # VolInfo
    spot_service = env_required("SEOUL_SPOT_SERVICE")      # SpotInfo
    fmt = os.getenv("SEOUL_API_FORMAT", "xml")             # xml 고정 추천

    p = argparse.ArgumentParser()
    p.add_argument("--start_ymd", required=True, help="YYYYMMDD (inclusive)")
    p.add_argument("--end_ymd", required=True, help="YYYYMMDD (inclusive)")
    p.add_argument("--limit_spots", type=int, default=10, help="처음엔 5~20 추천")
    p.add_argument("--hours", default="0-23", help='예: "0-23" 또는 "7-10" 또는 "0,6,12,18"')
    p.add_argument("--sleep", type=float, default=0.05, help="API 과호출 방지")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--spot_nums", default="", help='특정 지점만: "A-01,A-02"')
    args = p.parse_args()

    # hours 파싱
    if "-" in args.hours:
        a, b = args.hours.split("-", 1)
        hours = range(int(a), int(b) + 1)
    elif "," in args.hours:
        hours = [int(x.strip()) for x in args.hours.split(",") if x.strip()]
    else:
        hours = [int(args.hours)]

    spot_csv = RAW_DIR / "traffic_spots_20240101_20241231_D46_F10.csv"
    hist_csv = RAW_DIR / "traffic_volume_history_20240101_20241231_D46_F10.csv"
    ck_path = RAW_DIR / "traffic_volume_history_20240101_20241231_D46_F10.checkpoint.json"

    # 1) SpotInfo 수집
    if not spot_csv.exists():
        print("[1] Fetch SpotInfo (XML) ...")
        spots = fetch_spots_xml(
            api_key=api_key,
            spot_service=spot_service,
            fmt=fmt,
            page_size=1000,
            timeout=args.timeout,
            retries=args.retries,
            sleep_sec=args.sleep,
            save_path=spot_csv,
        )
    else:
        spots = pd.read_csv(spot_csv, dtype={"spot_num": str}, encoding="utf-8-sig")
        print(f"[1] SpotInfo exists -> {spot_csv} ({len(spots):,} rows)")

    spot_filter = [s.strip() for s in args.spot_nums.split(",") if s.strip()] if args.spot_nums else None

    # 2) VolInfo 수집
    print("[2] Collect VolInfo (XML) ...")
    out = collect_volume_history_xml(
        api_key=api_key,
        volume_service=volume_service,
        spots=spots,
        start_ymd=args.start_ymd,
        end_ymd=args.end_ymd,
        hours=hours,
        fmt=fmt,
        timeout=args.timeout,
        retries=args.retries,
        sleep_sec=args.sleep,
        out_csv=hist_csv,
        checkpoint_path=ck_path,
        limit_spots=args.limit_spots,
        spot_filter=spot_filter,
    )

    # 3) merge
    print("[3] Merge history + spots ...")
    merged = merge_spots_into_history(out, spot_csv)

    print("[DONE]")
    print(f"- spots:  {spot_csv}")
    print(f"- hist:   {hist_csv}")
    print(f"- merged: {merged}")
    print(f"- ckpt:   {ck_path}")


if __name__ == "__main__":
    main()
