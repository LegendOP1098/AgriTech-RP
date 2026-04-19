# AgriTech-RP

District-scale wheat monitoring and validation pipeline for Punjab and Haryana.

This project combines:
- Satellite-based wheat area estimation (Google Earth Engine, Sentinel-2 NDVI).
- Official government ground truth (DES APY reports).
- District-level comparison metrics and interactive maps.

## 1) Goal

Detect and quantify wheat area during Rabi season, then validate predictions against official district statistics at scale (not sample points).

## 2) What Is Implemented

- End-to-end runner: `argitechini.py`
  - Predicts district wheat area from Sentinel-2 NDVI.
  - Downloads and parses DES APY district reports.
  - Produces validation metrics and map outputs.
- Validation module: `wheat_groundtruth_validation.py`
  - Standalone ground-truth extraction + mapping + validation utilities.
- Legacy notebook export: `argitechini_legacy_colab.py`
  - Preserved original prototype workflow.

## 3) Data Sources

- DES APY reports (official ground truth):
  - https://data.desagri.gov.in/website/crops-apy-report-web
- District boundaries (GeoJSON):
  - https://raw.githubusercontent.com/divya-akula/GeoJson-Data-India/master/India_State_District.geojson
- Satellite imagery:
  - `COPERNICUS/S2_SR_HARMONIZED` (Google Earth Engine)

## 4) Repository Structure

- `argitechini.py` -> Main wired pipeline (prediction + validation + maps)
- `wheat_groundtruth_validation.py` -> Validation utilities and standalone runner
- `VALIDATION_README.md` -> Extra validation-specific usage notes
- `outputs/` -> Generated CSV/HTML outputs (ignored in git by default)

## 5) Setup

### 5.1 Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 5.2 Install dependencies

```powershell
pip install requests pandas numpy pypdf plotly earthengine-api
```

### 5.3 Authenticate Earth Engine (one-time)

```powershell
earthengine authenticate
```

## 6) Run Commands

### 6.1 Full pipeline (prediction + DES validation + maps)

```powershell
python .\argitechini.py `
  --states Punjab Haryana `
  --start-year 2015 `
  --end-year 2022 `
  --season Rabi `
  --map-year 2022 `
  --gee-project wheat-10987654 `
  --ndvi-threshold 0.35 `
  --cloud-max 60 `
  --reduce-scale 120 `
  --output-dir outputs
```

Notes:
- Sentinel-2 prediction is available from ~2015 onward.
- If computation times out, increase `--reduce-scale` to `250`.

### 6.2 Ground-truth only (no GEE predictions)

```powershell
python .\wheat_groundtruth_validation.py `
  --states Punjab Haryana `
  --start-year 1997 `
  --end-year 2022 `
  --season Rabi `
  --map-year 2022 `
  --output-dir outputs
```

## 7) Outputs

Typical generated files in `outputs/`:
- `des_wheat_ground_truth.csv`
- `model_predictions_from_argitechini.csv`
- `district_validation_joined.csv`
- `ground_truth_unmatched.csv`
- `ground_truth_wheat_area_map_2022.html`
- `validation_error_map_2022.html`
- `argitechini_validation_summary.json`

Open maps in browser:

```powershell
start .\outputs\ground_truth_wheat_area_map_2022.html
start .\outputs\validation_error_map_2022.html
```

## 8) Validation Metrics

Metrics reported in summary JSON:
- `MAE` (ha)
- `RMSE` (ha)
- `MAPE` (%)
- `Bias` (ha; positive means overestimation)
- `R2`
- `Pearson correlation`

## 9) Known Limitations

- Sentinel-2 does not cover pre-2015 years for this workflow.
- District naming/version mismatches can leave some rows unmatched.
- Simple NDVI thresholding can overestimate in some districts.

## 10) Recommended Next Improvements

- Add Landsat-based prediction for pre-2015 years.
- Add cloud masking + temporal smoothing.
- Use district-calibrated thresholds or supervised classification.
- Add scatter plots and year-wise residual trend charts.

