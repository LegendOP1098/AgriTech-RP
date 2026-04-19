#!/usr/bin/env python3
"""District-scale wheat validation against official DES APY ground truth.

This script pulls official district-level Wheat (Rabi) statistics from:
https://data.desagri.gov.in/website/crops-apy-report-web

It supports:
1) Building a large real ground-truth dataset (default: Punjab + Haryana, 1997-2022).
2) Joining model-predicted district wheat area with ground truth.
3) Computing validation metrics (MAE, RMSE, MAPE, R^2, bias).
4) Generating interactive district maps (ground truth map always; error map when predictions exist).

Optional:
5) Creating district-level model predictions directly from Google Earth Engine.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - handled by runtime check
    px = None

try:
    import ee
except ImportError:  # pragma: no cover - optional dependency
    ee = None


DES_REPORT_URL = "https://data.desagri.gov.in/report/crop/crops-printdraft-apy-report"
DISTRICT_GEOJSON_URL = (
    "https://raw.githubusercontent.com/divya-akula/GeoJson-Data-India/master/"
    "India_State_District.geojson"
)

STATE_CODES: Dict[str, int] = {
    "Punjab": 3,
    "Haryana": 6,
}

SEASON_CODES: Dict[str, str] = {
    "Rabi": "R",
    "Kharif": "K",
    "Autumn": "A",
    "Winter": "W",
    "Summer": "S",
    "Whole Year": "Y",
}


@dataclass
class Metrics:
    n: int
    mae: float
    rmse: float
    mape: float
    bias: float
    r2: float
    corr: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "mae_ha": self.mae,
            "rmse_ha": self.rmse,
            "mape_percent": self.mape,
            "bias_ha": self.bias,
            "r2": self.r2,
            "pearson_corr": self.corr,
        }


COMBINED_ROW_RE = re.compile(
    r"^\d+\.\s+(?P<district>.+?)\s+"
    r"(?P<y1>\d{4})\s*-\s*(?P<y2>\d{4})\s+"
    r"(?P<season>[A-Za-z ]+)\s+"
    r"(?P<area>[-\d,\.]+)\s+(?P<production>[-\d,\.]+)\s+(?P<yield>[-\d,\.]+)$"
)
DISTRICT_ONLY_RE = re.compile(r"^\d+\.\s+(?P<district>.+?)$")
YEAR_ONLY_RE = re.compile(
    r"^(?P<y1>\d{4})\s*-\s*(?P<y2>\d{4})\s+"
    r"(?P<season>[A-Za-z ]+)\s+"
    r"(?P<area>[-\d,\.]+)\s+(?P<production>[-\d,\.]+)\s+(?P<yield>[-\d,\.]+)$"
)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split()).strip()


def parse_number(value: str) -> float:
    if value is None:
        return float("nan")
    text = value.strip().replace(",", "")
    if text in {"", "-", "NA", "N/A"}:
        return float("nan")
    return float(text)


def normalize_district_name(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    aliases = {
        "gurgaon": "gurugram",
        "mewat": "nuh",
        "nawanshahr": "shahidbhagatsinghnagar",
        "sbsnagar": "shahidbhagatsinghnagar",
        "sahibzadaajitsinghnagar": "sasnagar",
        "sasnagar": "sasnagar",
        "firozepur": "firozpur",
        "ferozepur": "firozpur",
        "charkidadri": "charkidadri",
        "charkhidari": "charkidadri",
    }
    return aliases.get(base, base)


def build_des_pdf_url(
    state_code: int,
    crop_code: int,
    season_code: str,
    start_year: int,
    end_year: int,
) -> str:
    params = {
        "reportformat": "horizontal_crop_vertical_year",
        "fltrstates": str(state_code),
        "fltrdistricts": "",
        "fltrcrops": str(crop_code),
        "fltrseason": season_code,
        "fltrstartyear": str(start_year),
        "fltrendyear": str(end_year),
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{DES_REPORT_URL}?{query}"


def fetch_pdf_bytes(url: str, timeout: int = 180) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "").lower()
    if "pdf" not in content_type:
        raise RuntimeError(f"Expected PDF from {url}, got content-type={content_type!r}")
    return resp.content


def parse_des_pdf(pdf_bytes: bytes, state_name: str, source_url: str) -> pd.DataFrame:
    reader = PdfReader(BytesIO(pdf_bytes))
    records: List[Dict[str, object]] = []
    current_district: Optional[str] = None

    for page in reader.pages:
        raw_text = page.extract_text() or ""
        lines = [normalize_whitespace(line) for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("*Cotton 1 Bale"):
                continue
            if line in {
                "Crop Production Statistics",
                "State/Crop/District Year Season Area",
                "(Hectare)",
                "Production",
                "(Tonne)",
                "Yield",
                "(Tonne/Hectare)",
                state_name,
                "1. Wheat",
            }:
                continue
            if line.lower().startswith("total wheat"):
                continue

            m_combined = COMBINED_ROW_RE.match(line)
            if m_combined:
                district = normalize_whitespace(m_combined.group("district"))
                y1 = int(m_combined.group("y1"))
                y2 = int(m_combined.group("y2"))
                season = normalize_whitespace(m_combined.group("season"))
                area = parse_number(m_combined.group("area"))
                production = parse_number(m_combined.group("production"))
                yield_val = parse_number(m_combined.group("yield"))
                current_district = district
                records.append(
                    {
                        "state": state_name,
                        "district": district,
                        "district_norm": normalize_district_name(district),
                        "year_start": y1,
                        "year_end": y2,
                        "year_label": f"{y1}-{y2}",
                        "season": season,
                        "gt_area_ha": area,
                        "gt_production_tonnes": production,
                        "gt_yield_tpha": yield_val,
                        "source_url": source_url,
                    }
                )
                continue

            m_district = DISTRICT_ONLY_RE.match(line)
            if m_district:
                district = normalize_whitespace(m_district.group("district"))
                if district.lower() in {"wheat"}:
                    continue
                if district.lower().startswith("total"):
                    continue
                current_district = district
                continue

            m_year = YEAR_ONLY_RE.match(line)
            if m_year and current_district:
                y1 = int(m_year.group("y1"))
                y2 = int(m_year.group("y2"))
                season = normalize_whitespace(m_year.group("season"))
                area = parse_number(m_year.group("area"))
                production = parse_number(m_year.group("production"))
                yield_val = parse_number(m_year.group("yield"))
                records.append(
                    {
                        "state": state_name,
                        "district": current_district,
                        "district_norm": normalize_district_name(current_district),
                        "year_start": y1,
                        "year_end": y2,
                        "year_label": f"{y1}-{y2}",
                        "season": season,
                        "gt_area_ha": area,
                        "gt_production_tonnes": production,
                        "gt_yield_tpha": yield_val,
                        "source_url": source_url,
                    }
                )

    if not records:
        raise RuntimeError(f"No district rows parsed for state={state_name}.")

    df = pd.DataFrame(records).drop_duplicates(
        subset=["state", "district_norm", "year_start", "season"], keep="last"
    )
    return df.sort_values(["state", "district", "year_start"]).reset_index(drop=True)


def fetch_ground_truth(
    states: Iterable[str],
    crop_code: int,
    season_name: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if season_name not in SEASON_CODES:
        raise ValueError(f"Unsupported season={season_name!r}. Use one of: {list(SEASON_CODES)}")
    season_code = SEASON_CODES[season_name]
    frames: List[pd.DataFrame] = []
    for state in states:
        if state not in STATE_CODES:
            raise ValueError(f"Unsupported state={state!r}. Add its code in STATE_CODES.")
        code = STATE_CODES[state]
        url = build_des_pdf_url(
            state_code=code,
            crop_code=crop_code,
            season_code=season_code,
            start_year=start_year,
            end_year=end_year,
        )
        pdf_bytes = fetch_pdf_bytes(url)
        frames.append(parse_des_pdf(pdf_bytes, state_name=state, source_url=url))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["state", "district", "year_start"]).reset_index(drop=True)


def load_model_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"state", "district", "year_start", "pred_area_ha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            "Expected columns include: state,district,year_start,pred_area_ha"
        )
    out = df.copy()
    out["district_norm"] = out["district"].map(normalize_district_name)
    out["year_start"] = out["year_start"].astype(int)
    out["pred_area_ha"] = pd.to_numeric(out["pred_area_ha"], errors="coerce")
    return out


def compute_metrics(df: pd.DataFrame) -> Metrics:
    valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_area_ha", "gt_area_ha"])
    if valid.empty:
        return Metrics(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    y_true = valid["gt_area_ha"].to_numpy(dtype=float)
    y_pred = valid["pred_area_ha"].to_numpy(dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    pct = np.where(y_true != 0, abs_err / np.abs(y_true) * 100.0, np.nan)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.nanmean(pct))
    bias = float(np.mean(err))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(valid) > 1 else math.nan

    return Metrics(len(valid), mae, rmse, mape, bias, r2, corr)


def prepare_geojson(states: Iterable[str]) -> Tuple[dict, pd.DataFrame]:
    resp = requests.get(DISTRICT_GEOJSON_URL, timeout=180)
    resp.raise_for_status()
    raw = resp.json()

    selected_states = set(states)
    features = []
    feature_rows = []

    for feat in raw.get("features", []):
        props = feat.get("properties", {})
        state = props.get("NAME_1")
        district = props.get("NAME_2")
        if state not in selected_states:
            continue
        district_norm = normalize_district_name(str(district))
        feature_id = f"{state}|{district_norm}"

        features.append(
            {
                "type": "Feature",
                "geometry": feat.get("geometry"),
                "properties": {
                    "feature_id": feature_id,
                    "state": state,
                    "district_shape": district,
                    "district_norm": district_norm,
                },
            }
        )
        feature_rows.append(
            {
                "state": state,
                "district_shape": district,
                "district_norm": district_norm,
                "feature_id": feature_id,
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    feature_df = pd.DataFrame(feature_rows).drop_duplicates()
    return geojson, feature_df


def ensure_plotly():
    if px is None:
        raise RuntimeError(
            "plotly is required for map generation. Install it with: pip install plotly"
        )


def write_ground_truth_map(
    gt_df: pd.DataFrame,
    geojson: dict,
    map_year: int,
    output_html: Path,
) -> pd.DataFrame:
    ensure_plotly()
    subset = gt_df[gt_df["year_start"] == map_year].copy()
    subset["feature_id"] = subset["state"] + "|" + subset["district_norm"]
    subset = subset.sort_values(["state", "district", "year_start"])
    subset = subset.drop_duplicates(subset=["feature_id"], keep="last")

    fig = px.choropleth(
        subset,
        geojson=geojson,
        locations="feature_id",
        featureidkey="properties.feature_id",
        color="gt_area_ha",
        hover_name="district",
        hover_data={
            "state": True,
            "year_label": True,
            "gt_area_ha": ":,.0f",
            "gt_production_tonnes": ":,.0f",
            "gt_yield_tpha": ":.2f",
            "feature_id": False,
        },
        color_continuous_scale="YlGn",
        title=f"Official DES Wheat Area (ha) - {map_year}-{map_year+1} (Rabi)",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return subset


def write_validation_error_map(
    merged: pd.DataFrame,
    geojson: dict,
    map_year: int,
    output_html: Path,
) -> pd.DataFrame:
    ensure_plotly()
    subset = merged[merged["year_start"] == map_year].copy()
    subset["feature_id"] = subset["state"] + "|" + subset["district_norm"]
    subset["abs_pct_error"] = (
        np.where(subset["gt_area_ha"] != 0, np.abs(subset["error_ha"]) / subset["gt_area_ha"] * 100, np.nan)
    )
    subset["signed_pct_error"] = (
        np.where(subset["gt_area_ha"] != 0, subset["error_ha"] / subset["gt_area_ha"] * 100, np.nan)
    )

    fig = px.choropleth(
        subset,
        geojson=geojson,
        locations="feature_id",
        featureidkey="properties.feature_id",
        color="abs_pct_error",
        hover_name="district",
        hover_data={
            "state": True,
            "year_label": True,
            "pred_area_ha": ":,.0f",
            "gt_area_ha": ":,.0f",
            "error_ha": ":,.0f",
            "abs_pct_error": ":.2f",
            "signed_pct_error": ":.2f",
            "feature_id": False,
        },
        color_continuous_scale="RdYlGn_r",
        title=f"District Validation Error (%) - {map_year}-{map_year+1} (Wheat Rabi)",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return subset


def estimate_pred_area_with_gee(
    states: Iterable[str],
    start_year: int,
    end_year: int,
    gee_project: str,
    ndvi_threshold: float,
) -> pd.DataFrame:
    if ee is None:
        raise RuntimeError(
            "earthengine-api is not installed. Install it with: pip install earthengine-api"
        )

    ee.Initialize(project=gee_project)
    district_fc = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.inList("ADM1_NAME", list(states)))
    )

    all_rows: List[Dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-11-01"
        end_date = f"{year + 1}-04-30"

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(district_fc.geometry())
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        )
        ndvi = s2.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        wheat_mask = ndvi.max().gt(ndvi_threshold)
        pred_area_ha_img = wheat_mask.multiply(ee.Image.pixelArea()).divide(10000).rename("pred_area_ha")

        reduced = pred_area_ha_img.reduceRegions(
            collection=district_fc, reducer=ee.Reducer.sum(), scale=10, tileScale=4
        )
        reduced = reduced.map(
            lambda f: f.set(
                {
                    "state": f.get("ADM1_NAME"),
                    "district": f.get("ADM2_NAME"),
                    "year_start": year,
                    "year_label": f"{year}-{year + 1}",
                    "pred_area_ha": f.get("sum"),
                }
            )
        )
        features = reduced.getInfo().get("features", [])
        for feat in features:
            prop = feat.get("properties", {})
            district = str(prop.get("district", ""))
            all_rows.append(
                {
                    "state": str(prop.get("state", "")),
                    "district": district,
                    "district_norm": normalize_district_name(district),
                    "year_start": int(year),
                    "year_label": f"{year}-{year + 1}",
                    "pred_area_ha": parse_number(str(prop.get("pred_area_ha", "nan"))),
                }
            )
    return pd.DataFrame(all_rows)


def merge_validation(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt_small = gt_df[
        ["state", "district", "district_norm", "year_start", "year_label", "gt_area_ha"]
    ].copy()
    gt_small = (
        gt_small.groupby(["state", "district_norm", "year_start", "year_label"], as_index=False)
        .agg({"district": "first", "gt_area_ha": "sum"})
        .rename(columns={"district": "district_gt"})
    )
    pred_small = pred_df[
        ["state", "district", "district_norm", "year_start", "pred_area_ha"]
    ].copy()
    pred_small = (
        pred_small.groupby(["state", "district_norm", "year_start"], as_index=False)
        .agg({"district": "first", "pred_area_ha": "sum"})
        .rename(columns={"district": "district_pred"})
    )

    merged = pred_small.merge(
        gt_small,
        on=["state", "district_norm", "year_start"],
        how="inner",
        validate="many_to_one",
    )
    merged["district"] = merged["district_gt"].fillna(merged["district_pred"])
    merged["error_ha"] = merged["pred_area_ha"] - merged["gt_area_ha"]

    unmatched_gt = gt_small.merge(
        pred_small[["state", "district_norm", "year_start"]],
        on=["state", "district_norm", "year_start"],
        how="left",
        indicator=True,
    )
    unmatched_gt = unmatched_gt[unmatched_gt["_merge"] == "left_only"].drop(columns=["_merge"])
    return merged, unmatched_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate wheat model outputs against DES APY district-level ground truth."
    )
    parser.add_argument("--states", nargs="+", default=["Punjab", "Haryana"])
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--season", default="Rabi")
    parser.add_argument("--crop-code", type=int, default=2, help="DES crop code. Wheat=2")
    parser.add_argument("--map-year", type=int, default=2022)
    parser.add_argument("--output-dir", default="outputs")

    parser.add_argument(
        "--model-csv",
        type=Path,
        default=None,
        help="CSV with columns: state,district,year_start,pred_area_ha",
    )
    parser.add_argument(
        "--use-gee",
        action="store_true",
        help="Generate district predictions directly from Earth Engine instead of --model-csv.",
    )
    parser.add_argument("--gee-project", default=None)
    parser.add_argument("--ndvi-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_df = fetch_ground_truth(
        states=args.states,
        crop_code=args.crop_code,
        season_name=args.season,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    gt_csv = output_dir / "des_wheat_ground_truth.csv"
    gt_df.to_csv(gt_csv, index=False)

    geojson, geo_index = prepare_geojson(states=args.states)
    geo_index.to_csv(output_dir / "district_geo_index.csv", index=False)

    gt_map_html = output_dir / f"ground_truth_wheat_area_map_{args.map_year}.html"
    gt_map_df = write_ground_truth_map(
        gt_df=gt_df, geojson=geojson, map_year=args.map_year, output_html=gt_map_html
    )
    gt_mapped_keys = set(gt_map_df["feature_id"].dropna().unique())
    geo_keys = set(geo_index["feature_id"].dropna().unique())
    unmapped_gt_map = gt_map_df[~gt_map_df["feature_id"].isin(geo_keys)].copy()

    summary = {
        "ground_truth_rows": int(len(gt_df)),
        "ground_truth_states": sorted(gt_df["state"].unique().tolist()),
        "ground_truth_year_range": [int(gt_df["year_start"].min()), int(gt_df["year_start"].max())],
        "ground_truth_district_count": int(gt_df[["state", "district_norm"]].drop_duplicates().shape[0]),
        "ground_truth_csv": str(gt_csv),
        "ground_truth_map_html": str(gt_map_html),
        "ground_truth_map_year": int(args.map_year),
        "ground_truth_mapped_districts": int(len(gt_mapped_keys & geo_keys)),
        "ground_truth_unmapped_for_map_count": int(len(unmapped_gt_map)),
        "ground_truth_unmapped_for_map": sorted(
            (
                unmapped_gt_map[["state", "district"]]
                .drop_duplicates()
                .astype(str)
                .agg(" | ".join, axis=1)
                .tolist()
            )
        ),
    }

    pred_df = None
    if args.use_gee:
        if not args.gee_project:
            raise ValueError("--gee-project is required when --use-gee is enabled.")
        pred_df = estimate_pred_area_with_gee(
            states=args.states,
            start_year=args.start_year,
            end_year=args.end_year,
            gee_project=args.gee_project,
            ndvi_threshold=args.ndvi_threshold,
        )
        pred_df.to_csv(output_dir / "model_predictions_from_gee.csv", index=False)
    elif args.model_csv is not None:
        pred_df = load_model_predictions(args.model_csv)

    if pred_df is not None and not pred_df.empty:
        merged, unmatched_gt = merge_validation(pred_df=pred_df, gt_df=gt_df)
        merged_csv = output_dir / "district_validation_joined.csv"
        merged.to_csv(merged_csv, index=False)
        unmatched_gt.to_csv(output_dir / "ground_truth_unmatched.csv", index=False)

        all_metrics = compute_metrics(merged)
        by_state = {}
        for state, g in merged.groupby("state"):
            by_state[state] = compute_metrics(g).as_dict()

        err_map_html = output_dir / f"validation_error_map_{args.map_year}.html"
        write_validation_error_map(merged=merged, geojson=geojson, map_year=args.map_year, output_html=err_map_html)

        summary.update(
            {
                "validation_rows": int(len(merged)),
                "validation_csv": str(merged_csv),
                "validation_error_map_html": str(err_map_html),
                "validation_unmatched_ground_truth_rows": int(len(unmatched_gt)),
                "overall_metrics": all_metrics.as_dict(),
                "metrics_by_state": by_state,
            }
        )

    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
