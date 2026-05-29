from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


HOSPITAL_TARGET_TYPES = ("상급종합", "종합병원")

CBD_CENTERS = {
    "CBD": {
        "name": "도심권",
        "description": "시청/광화문 일대",
        "address": "서울특별시 중구 세종대로 110",
        "latitude": 37.5665,
        "longitude": 126.9780,
    },
    "YBD": {
        "name": "여의도권",
        "description": "여의도 일대",
        "address": "서울특별시 영등포구 의사당대로 1",
        "latitude": 37.5259,
        "longitude": 126.9209,
    },
    "GBD": {
        "name": "강남권",
        "description": "강남역 사거리 일대",
        "address": "서울특별시 강남구 강남대로 396",
        "latitude": 37.4979,
        "longitude": 127.0276,
    },
}


def normalize_filename(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def find_external_file(external_dir: Path, keyword: str) -> Path:
    matches = [
        path
        for path in external_dir.glob("*.csv")
        if keyword in normalize_filename(path.name)
    ]
    if not matches:
        raise FileNotFoundError(f"{external_dir}에서 '{keyword}' CSV 파일을 찾지 못했습니다.")
    if len(matches) > 1:
        raise ValueError(f"'{keyword}' 파일이 여러 개입니다: {[p.name for p in matches]}")
    return matches[0]


def get_hospital_quarter(path: Path) -> str:
    match = re.search(r"2025\.(\d+)", normalize_filename(path.name))
    if not match:
        raise ValueError(f"파일명에서 분기 월을 찾지 못했습니다: {path.name}")
    return match.group(1).zfill(2)


def load_target_hospitals(path: Path) -> pd.DataFrame:
    columns = [
        "암호화요양기호",
        "요양기관명",
        "종별코드명",
        "시도코드명",
        "시군구코드명",
        "주소",
        "전화번호",
        "개설일자",
        "총의사수",
        "좌표(X)",
        "좌표(Y)",
    ]
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=columns)
    df = df[
        df["시도코드명"].eq("서울")
        & df["종별코드명"].isin(HOSPITAL_TARGET_TYPES)
    ].copy()
    df["snapshot_month"] = get_hospital_quarter(path)
    df = df.rename(
        columns={
            "암호화요양기호": "hospital_id",
            "요양기관명": "hospital_name",
            "종별코드명": "hospital_type",
            "시도코드명": "sido",
            "시군구코드명": "gu",
            "주소": "address",
            "전화번호": "phone",
            "개설일자": "opened_date",
            "총의사수": "doctor_count",
            "좌표(X)": "longitude",
            "좌표(Y)": "latitude",
        }
    )
    return df[
        [
            "snapshot_month",
            "hospital_id",
            "hospital_name",
            "hospital_type",
            "sido",
            "gu",
            "address",
            "phone",
            "opened_date",
            "doctor_count",
            "longitude",
            "latitude",
        ]
    ].sort_values(["hospital_type", "gu", "hospital_name"]).reset_index(drop=True)


def compare_hospital_snapshots(hospital_by_month: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    diff_rows = []
    months = sorted(hospital_by_month)

    for month, df in hospital_by_month.items():
        counts = df["hospital_type"].value_counts()
        summary_rows.append(
            {
                "snapshot_month": month,
                "total_count": len(df),
                "tertiary_count": int(counts.get("상급종합", 0)),
                "general_count": int(counts.get("종합병원", 0)),
            }
        )

    base_month = months[0]
    base_ids = set(hospital_by_month[base_month]["hospital_id"])
    for month in months[1:]:
        current = hospital_by_month[month]
        current_ids = set(current["hospital_id"])
        added = current[current["hospital_id"].isin(current_ids - base_ids)]
        removed = hospital_by_month[base_month][
            hospital_by_month[base_month]["hospital_id"].isin(base_ids - current_ids)
        ]

        for _, row in added.iterrows():
            diff_rows.append(
                {
                    "base_month": base_month,
                    "compare_month": month,
                    "change_type": "added",
                    "hospital_name": row["hospital_name"],
                    "hospital_type": row["hospital_type"],
                    "gu": row["gu"],
                    "address": row["address"],
                }
            )
        for _, row in removed.iterrows():
            diff_rows.append(
                {
                    "base_month": base_month,
                    "compare_month": month,
                    "change_type": "removed",
                    "hospital_name": row["hospital_name"],
                    "hospital_type": row["hospital_type"],
                    "gu": row["gu"],
                    "address": row["address"],
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values("snapshot_month")
    diffs = pd.DataFrame(diff_rows)
    return summary, diffs


def clean_large_marts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    df = df[
        df["업태구분명"].eq("대형마트")
        & df["영업상태명"].eq("영업/정상")
        & df["상세영업상태명"].eq("정상영업")
    ].copy()

    df["address"] = df["도로명주소"].fillna(df["지번주소"])
    df["gu"] = df["address"].str.extract(r"서울특별시\s+([^\s]+)")[0]
    for column in ["좌표정보(X)", "좌표정보(Y)"]:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.strip().replace("", pd.NA),
            errors="coerce",
        )

    df = df.rename(
        columns={
            "관리번호": "store_id",
            "사업장명": "store_name",
            "업태구분명": "store_type",
            "영업상태명": "business_status",
            "상세영업상태명": "business_status_detail",
            "인허가일자": "licensed_date",
            "좌표정보(X)": "coord_x",
            "좌표정보(Y)": "coord_y",
        }
    )
    return df[
        [
            "store_id",
            "store_name",
            "store_type",
            "business_status",
            "business_status_detail",
            "licensed_date",
            "gu",
            "address",
            "coord_x",
            "coord_y",
        ]
    ].sort_values(["gu", "store_name"]).reset_index(drop=True)


def require_coordinate_columns(
    df: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    label: str = "dataframe",
) -> None:
    missing = [col for col in [latitude_col, longitude_col] if col not in df.columns]
    if missing:
        raise KeyError(f"{label}에 좌표 컬럼이 없습니다: {missing}")


def haversine_distance_km(
    left_latitude,
    left_longitude,
    right_latitude,
    right_longitude,
) -> np.ndarray:
    radius_km = 6371.0088
    left_latitude = np.radians(left_latitude)
    left_longitude = np.radians(left_longitude)
    right_latitude = np.radians(right_latitude)
    right_longitude = np.radians(right_longitude)

    delta_latitude = right_latitude - left_latitude
    delta_longitude = right_longitude - left_longitude
    a = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(left_latitude)
        * np.cos(right_latitude)
        * np.sin(delta_longitude / 2) ** 2
    )
    return radius_km * 2 * np.arcsin(np.sqrt(a))


def nearest_facility_distance_km(
    locations: pd.DataFrame,
    facilities: pd.DataFrame,
    location_latitude_col: str = "latitude",
    location_longitude_col: str = "longitude",
    facility_latitude_col: str = "latitude",
    facility_longitude_col: str = "longitude",
    chunk_size: int = 5000,
) -> pd.Series:
    require_coordinate_columns(
        locations, location_latitude_col, location_longitude_col, "locations"
    )
    require_coordinate_columns(
        facilities, facility_latitude_col, facility_longitude_col, "facilities"
    )
    facilities = facilities.dropna(
        subset=[facility_latitude_col, facility_longitude_col]
    )
    if facilities.empty:
        return pd.Series(np.nan, index=locations.index)

    facility_latitudes = facilities[facility_latitude_col].to_numpy(dtype=float)
    facility_longitudes = facilities[facility_longitude_col].to_numpy(dtype=float)
    nearest = pd.Series(np.nan, index=locations.index, dtype=float)
    valid_locations = locations.dropna(
        subset=[location_latitude_col, location_longitude_col]
    )

    for start in range(0, len(valid_locations), chunk_size):
        end = min(start + chunk_size, len(valid_locations))
        chunk = valid_locations.iloc[start:end]
        distances = haversine_distance_km(
            chunk[location_latitude_col].to_numpy(dtype=float)[:, None],
            chunk[location_longitude_col].to_numpy(dtype=float)[:, None],
            facility_latitudes[None, :],
            facility_longitudes[None, :],
        )
        nearest.loc[chunk.index] = np.nanmin(distances, axis=1)

    return nearest


def nearby_facility_count(
    locations: pd.DataFrame,
    facilities: pd.DataFrame,
    radius_km: float = 1.0,
    location_latitude_col: str = "latitude",
    location_longitude_col: str = "longitude",
    facility_latitude_col: str = "latitude",
    facility_longitude_col: str = "longitude",
    chunk_size: int = 5000,
) -> pd.Series:
    require_coordinate_columns(
        locations, location_latitude_col, location_longitude_col, "locations"
    )
    require_coordinate_columns(
        facilities, facility_latitude_col, facility_longitude_col, "facilities"
    )
    facilities = facilities.dropna(
        subset=[facility_latitude_col, facility_longitude_col]
    )
    if facilities.empty:
        return pd.Series(pd.NA, index=locations.index, dtype="Int64")

    facility_latitudes = facilities[facility_latitude_col].to_numpy(dtype=float)
    facility_longitudes = facilities[facility_longitude_col].to_numpy(dtype=float)
    counts = pd.Series(pd.NA, index=locations.index, dtype="Int64")
    valid_locations = locations.dropna(
        subset=[location_latitude_col, location_longitude_col]
    )

    for start in range(0, len(valid_locations), chunk_size):
        end = min(start + chunk_size, len(valid_locations))
        chunk = valid_locations.iloc[start:end]
        distances = haversine_distance_km(
            chunk[location_latitude_col].to_numpy(dtype=float)[:, None],
            chunk[location_longitude_col].to_numpy(dtype=float)[:, None],
            facility_latitudes[None, :],
            facility_longitudes[None, :],
        )
        counts.loc[chunk.index] = (distances <= radius_km).sum(axis=1)

    return counts


def add_cbd_distance_features(
    apartments: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
) -> pd.DataFrame:
    require_coordinate_columns(apartments, latitude_col, longitude_col, "apartments")
    result = apartments.copy()
    distance_columns = []

    for code, center in CBD_CENTERS.items():
        column = f"distance_to_{code.lower()}_km"
        result[column] = haversine_distance_km(
            result[latitude_col].to_numpy(dtype=float),
            result[longitude_col].to_numpy(dtype=float),
            center["latitude"],
            center["longitude"],
        )
        distance_columns.append(column)

    distances = result[distance_columns]
    result["nearest_business_district_distance_km"] = distances.min(axis=1)
    result["nearest_business_district"] = pd.NA
    valid_distances = distances.notna().any(axis=1)
    result.loc[valid_distances, "nearest_business_district"] = (
        distances.loc[valid_distances]
        .idxmin(axis=1)
        .str.replace("distance_to_", "", regex=False)
        .str.replace("_km", "", regex=False)
        .str.upper()
    )
    return result


def add_accessibility_features(
    apartments: pd.DataFrame,
    stations: pd.DataFrame,
    hospitals: pd.DataFrame,
    large_marts: pd.DataFrame | None = None,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
) -> pd.DataFrame:
    result = add_cbd_distance_features(apartments, latitude_col, longitude_col)

    result["nearest_subway_distance_km"] = nearest_facility_distance_km(
        result,
        stations,
        location_latitude_col=latitude_col,
        location_longitude_col=longitude_col,
        facility_latitude_col="latitude",
        facility_longitude_col="longitude",
    )
    result["hospital_count_within_1km"] = nearby_facility_count(
        result,
        hospitals,
        radius_km=1.0,
        location_latitude_col=latitude_col,
        location_longitude_col=longitude_col,
        facility_latitude_col="latitude",
        facility_longitude_col="longitude",
    )
    result["nearest_hospital_distance_km"] = nearest_facility_distance_km(
        result,
        hospitals,
        location_latitude_col=latitude_col,
        location_longitude_col=longitude_col,
        facility_latitude_col="latitude",
        facility_longitude_col="longitude",
    )

    if large_marts is not None:
        result["large_mart_count_within_1km"] = nearby_facility_count(
            result,
            large_marts,
            radius_km=1.0,
            location_latitude_col=latitude_col,
            location_longitude_col=longitude_col,
            facility_latitude_col="latitude",
            facility_longitude_col="longitude",
        )

    return result
