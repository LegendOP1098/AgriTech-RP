#!/usr/bin/env python3
"""Wired AgriTech pipeline: GEE wheat detection + DES ground-truth validation.

This script turns the original project idea into one runnable flow:
1) Estimate district-wise wheat area from Sentinel-2 NDVI over Punjab/Haryana.
2) Download official DES APY district ground truth (Wheat, Rabi).
3) Validate predictions vs ground truth and generate maps/metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None

from wheat_groundtruth_validation import (
    compute_metrics,
    fetch_ground_truth,
    merge_validation,
    normalize_district_name,
    prepare_geojson,
    write_ground_truth_map,
    write_validation_error_map,
)

S2_FIRST_RABI_YEAR = 2015
PRED_COLUMNS = [
    "state",
    "district",
    "district_norm",
    "year_start",
    "year_label",
    "pred_area_ha",
]


def init_ee(project: str) -> None:
    if ee is None:
        raise RuntimeError(
            "earthengine-api is not installed. Install with: pip install earthengine-api"
        )
    ee.Initialize(project=project)


def district_fc(states: List[str]):
    return (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.inList("ADM1_NAME", states))
    )


def estimate_year_pred_area(
    fc,
    year_start: int,
    ndvi_threshold: float,
    cloud_max: float,
    reduce_scale: float,
) -> pd.DataFrame:
    start_date = f"{year_start}-11-01"
    end_date = f"{year_start + 1}-04-30"

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(fc.geometry())
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
    )
    img_count = int(s2.size().getInfo())
    if img_count == 0:
        print(
            f"[WARN] No Sentinel-2 images for {year_start}-{year_start + 1} "
            f"(cloud_max={cloud_max}). Skipping this year."
        )
        return pd.DataFrame(columns=PRED_COLUMNS)

    ndvi = s2.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    wheat_mask = ndvi.max().gt(ndvi_threshold).rename("WHEAT")
    area_img = wheat_mask.multiply(ee.Image.pixelArea()).divide(10000).rename("pred_area_ha")
    retry_pairs = [
        (float(reduce_scale), 4),
        (max(float(reduce_scale), 120.0), 8),
        (max(float(reduce_scale), 250.0), 16),
    ]
    # preserve order but remove duplicate scale/tile tuples
    retry_pairs = list(dict.fromkeys(retry_pairs))

    features = None
    last_err = None
    for scale, tile_scale in retry_pairs:
        try:
            reduced = area_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.sum(),
                scale=scale,
                tileScale=tile_scale,
            )
            features = reduced.getInfo().get("features", [])
            if (scale, tile_scale) != retry_pairs[0]:
                print(
                    f"[INFO] Year {year_start}-{year_start + 1} succeeded with "
                    f"fallback scale={scale}, tileScale={tile_scale}."
                )
            break
        except Exception as exc:  # ee raises dynamic exception types
            msg = str(exc)
            last_err = exc
            if "Computation timed out" in msg or "memory capacity exceeded" in msg:
                print(
                    f"[WARN] Timeout at scale={scale}, tileScale={tile_scale} for "
                    f"{year_start}-{year_start + 1}. Retrying with coarser settings..."
                )
                continue
            raise

    if features is None:
        raise RuntimeError(
            f"Failed to compute area for {year_start}-{year_start + 1} after retries: {last_err}"
        )

    rows: List[Dict[str, object]] = []
    for feat in features:
        prop = feat.get("properties", {})
        district = str(prop.get("ADM2_NAME", "")).strip()
        state = str(prop.get("ADM1_NAME", "")).strip()
        pred = prop.get("sum")
        rows.append(
            {
                "state": state,
                "district": district,
                "district_norm": normalize_district_name(district),
                "year_start": int(year_start),
                "year_label": f"{year_start}-{year_start + 1}",
                "pred_area_ha": float(pred) if pred is not None else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def generate_predictions(
    states: List[str],
    start_year: int,
    end_year: int,
    gee_project: str,
    ndvi_threshold: float,
    cloud_max: float,
    reduce_scale: float,
) -> pd.DataFrame:
    init_ee(gee_project)
    fc = district_fc(states)
    pred_start_year = max(start_year, S2_FIRST_RABI_YEAR)
    if end_year < pred_start_year:
        raise ValueError(
            f"No Sentinel-2 prediction window available for requested range "
            f"{start_year}-{end_year}. Use end-year >= {S2_FIRST_RABI_YEAR}."
        )

    all_frames = []
    if start_year < S2_FIRST_RABI_YEAR:
        print(
            f"[INFO] Predictions are satellite-limited. "
            f"Using {pred_start_year}-{end_year} for model prediction "
            f"(requested {start_year}-{end_year})."
        )
    for year in range(pred_start_year, end_year + 1):
        year_df = estimate_year_pred_area(
            fc=fc,
            year_start=year,
            ndvi_threshold=ndvi_threshold,
            cloud_max=cloud_max,
            reduce_scale=reduce_scale,
        )
        if not year_df.empty:
            all_frames.append(year_df)

    if not all_frames:
        raise RuntimeError(
            "Prediction dataframe is empty after filtering no-imagery years. "
            "Try a higher --cloud-max or a later year range."
        )
    pred_df = pd.concat(all_frames, ignore_index=True)
    pred_df = pred_df.sort_values(["state", "district", "year_start"]).reset_index(drop=True)
    return pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run argitechini detection + DES validation end-to-end."
    )
    parser.add_argument("--states", nargs="+", default=["Punjab", "Haryana"])
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--season", default="Rabi")
    parser.add_argument("--map-year", type=int, default=2022)
    parser.add_argument("--crop-code", type=int, default=2, help="DES crop code; Wheat=2")

    parser.add_argument("--gee-project", default="whee-486607")
    parser.add_argument("--ndvi-threshold", type=float, default=0.35)
    parser.add_argument("--cloud-max", type=float, default=60.0)
    parser.add_argument(
        "--reduce-scale",
        type=float,
        default=60.0,
        help="Scale (meters) for district area aggregation; higher is faster/coarser.",
    )
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = generate_predictions(
        states=args.states,
        start_year=args.start_year,
        end_year=args.end_year,
        gee_project=args.gee_project,
        ndvi_threshold=args.ndvi_threshold,
        cloud_max=args.cloud_max,
        reduce_scale=args.reduce_scale,
    )
    pred_csv = output_dir / "model_predictions_from_argitechini.csv"
    pred_df.to_csv(pred_csv, index=False)

    gt_df = fetch_ground_truth(
        states=args.states,
        crop_code=args.crop_code,
        season_name=args.season,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    gt_csv = output_dir / "des_wheat_ground_truth.csv"
    gt_df.to_csv(gt_csv, index=False)

    geojson, _ = prepare_geojson(states=args.states)
    gt_map_html = output_dir / f"ground_truth_wheat_area_map_{args.map_year}.html"
    write_ground_truth_map(
        gt_df=gt_df,
        geojson=geojson,
        map_year=args.map_year,
        output_html=gt_map_html,
    )

    merged, unmatched_gt = merge_validation(pred_df=pred_df, gt_df=gt_df)
    merged_csv = output_dir / "district_validation_joined.csv"
    unmatched_csv = output_dir / "ground_truth_unmatched.csv"
    merged.to_csv(merged_csv, index=False)
    unmatched_gt.to_csv(unmatched_csv, index=False)

    summary = {
        "pred_rows": int(len(pred_df)),
        "gt_rows": int(len(gt_df)),
        "validation_rows": int(len(merged)),
        "states": args.states,
        "year_range_requested": [args.start_year, args.end_year],
        "year_range_predicted": [int(pred_df["year_start"].min()), int(pred_df["year_start"].max())],
        "pred_csv": str(pred_csv),
        "gt_csv": str(gt_csv),
        "ground_truth_map_html": str(gt_map_html),
        "unmatched_gt_rows": int(len(unmatched_gt)),
        "overall_metrics": None,
        "metrics_by_state": {},
    }

    if not merged.empty:
        all_metrics = compute_metrics(merged).as_dict()
        by_state = {
            state: compute_metrics(group).as_dict()
            for state, group in merged.groupby("state")
        }
        err_map_html = output_dir / f"validation_error_map_{args.map_year}.html"
        write_validation_error_map(
            merged=merged,
            geojson=geojson,
            map_year=args.map_year,
            output_html=err_map_html,
        )
        summary["overall_metrics"] = all_metrics
        summary["metrics_by_state"] = by_state
        summary["validation_error_map_html"] = str(err_map_html)

    summary_path = output_dir / "argitechini_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
