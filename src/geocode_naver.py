from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


NAVER_GEOCODE_URL = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def get_naver_credentials() -> tuple[str, str]:
    load_env_file()
    client_id = (
        os.getenv("NAVER_CLIENT_ID")
        or os.getenv("NAVER_MAPS_CLIENT_ID")
        or os.getenv("NAVER_API_KEY_ID")
    )
    client_secret = (
        os.getenv("NAVER_CLIENT_SECRET")
        or os.getenv("NAVER_MAPS_CLIENT_SECRET")
        or os.getenv("NAVER_API_KEY")
    )
    if not client_id or not client_secret:
        raise RuntimeError(
            ".env에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 설정해야 합니다."
        )
    return client_id, client_secret


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def is_too_coarse_address(address: str) -> bool:
    return bool(re.fullmatch(r"서울특별시\s+\S+구", str(address).strip()))


def test_naver_geocoding(address: str = "서울특별시 중구 세종대로 110") -> dict[str, Any]:
    client_id, client_secret = get_naver_credentials()
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }
    response = requests.get(
        NAVER_GEOCODE_URL,
        headers=headers,
        params={"query": address},
        timeout=10,
    )
    try:
        body = response.json()
    except ValueError:
        body = response.text[:500]
    return {
        "client_id": mask_secret(client_id),
        "client_secret": mask_secret(client_secret),
        "status_code": response.status_code,
        "body": body,
    }


def validate_naver_geocoding() -> None:
    result = test_naver_geocoding()
    if result["status_code"] != 200:
        raise RuntimeError(f"Naver Geocoding 인증/연결 테스트 실패: {result}")


def geocode_address(
    address: str,
    client_id: str,
    client_secret: str,
    address_col: str = "full_road_address",
    timeout: int = 10,
) -> dict[str, Any]:
    if is_too_coarse_address(address):
        return {
            address_col: address,
            "geocode_status": "too_coarse",
            "matched_address": pd.NA,
            "road_address": pd.NA,
            "jibun_address": pd.NA,
            "latitude": pd.NA,
            "longitude": pd.NA,
        }

    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }
    response = requests.get(
        NAVER_GEOCODE_URL,
        headers=headers,
        params={"query": address},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    addresses = payload.get("addresses", [])

    if not addresses:
        return {
            address_col: address,
            "geocode_status": "not_found",
            "matched_address": pd.NA,
            "road_address": pd.NA,
            "jibun_address": pd.NA,
            "latitude": pd.NA,
            "longitude": pd.NA,
        }

    match = addresses[0]
    return {
        address_col: address,
        "geocode_status": "ok",
        "matched_address": match.get("roadAddress") or match.get("jibunAddress"),
        "road_address": match.get("roadAddress"),
        "jibun_address": match.get("jibunAddress"),
        "latitude": float(match["y"]),
        "longitude": float(match["x"]),
    }


def load_existing_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def geocode_unique_addresses(
    input_path: Path,
    output_path: Path,
    address_col: str = "full_road_address",
    sleep_seconds: float = 0.1,
    limit: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    source = pd.read_csv(input_path, encoding="utf-8-sig", usecols=[address_col])
    addresses = (
        source[address_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda series: series.ne("")]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if limit is not None:
        addresses = addresses[:limit]

    cache = load_existing_cache(output_path)
    cached_addresses = set()
    if not force and not cache.empty and address_col in cache.columns:
        if "geocode_status" in cache.columns:
            reusable_cache = cache[cache["geocode_status"].ne("error")]
        else:
            reusable_cache = cache
        cached_addresses = set(reusable_cache[address_col].dropna().astype(str))

    client_id, client_secret = get_naver_credentials()
    validate_naver_geocoding()
    rows = [] if force or cache.empty else cache.to_dict("records")
    pending = [address for address in addresses if address not in cached_addresses]

    for index, address in enumerate(pending, start=1):
        try:
            rows.append(
                geocode_address(
                    address,
                    client_id,
                    client_secret,
                    address_col=address_col,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    address_col: address,
                    "geocode_status": "error",
                    "matched_address": pd.NA,
                    "road_address": pd.NA,
                    "jibun_address": pd.NA,
                    "latitude": pd.NA,
                    "longitude": pd.NA,
                    "error_message": str(exc),
                }
            )

        if index % 100 == 0 or index == len(pending):
            print(f"geocoded {index:,}/{len(pending):,} pending addresses")
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        time.sleep(sleep_seconds)

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=[address_col], keep="last")
    result = result.sort_values(address_col).reset_index(drop=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def merge_coordinates(
    apartment_path: Path,
    geocoded_path: Path,
    output_path: Path,
    address_col: str = "full_road_address",
) -> pd.DataFrame:
    apartments = pd.read_csv(apartment_path, encoding="utf-8-sig")
    coordinates = pd.read_csv(geocoded_path, encoding="utf-8-sig")
    apartments[address_col] = apartments[address_col].astype(str).str.strip()
    coordinates[address_col] = coordinates[address_col].astype(str).str.strip()
    coordinate_cols = [
        address_col,
        "geocode_status",
        "matched_address",
        "latitude",
        "longitude",
    ]
    merged = apartments.merge(coordinates[coordinate_cols], on=address_col, how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naver Geocoding API로 아파트 도로명 주소 좌표를 생성합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/interim/seoul_apt_trade_2025_basic_cleaned.csv",
    )
    parser.add_argument(
        "--geocoded-output",
        type=Path,
        default=PROJECT_ROOT / "data/interim/seoul_apt_address_geocoded.csv",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/seoul_apt_trade_2025_with_coordinates.csv",
    )
    parser.add_argument("--address-col", default="full_road_address")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="이미 생성된 지오코딩 결과만 원본 거래 데이터에 병합합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.merge_only:
        geocode_unique_addresses(
            input_path=args.input,
            output_path=args.geocoded_output,
            address_col=args.address_col,
            sleep_seconds=args.sleep_seconds,
            limit=args.limit,
            force=args.force,
        )
    merged = merge_coordinates(
        apartment_path=args.input,
        geocoded_path=args.geocoded_output,
        output_path=args.merged_output,
        address_col=args.address_col,
    )
    print(f"saved: {args.geocoded_output}")
    print(f"saved: {args.merged_output}")
    print(
        "coordinate coverage:",
        f"{merged['latitude'].notna().sum():,}/{len(merged):,}",
    )


if __name__ == "__main__":
    main()
