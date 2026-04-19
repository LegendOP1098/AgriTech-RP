# Wheat Ground-Truth Validation (Punjab + Haryana)

Use [`wheat_groundtruth_validation.py`](/D:/agritech/wheat_groundtruth_validation.py) to validate district-level wheat estimates against official DES APY data.

## What it does
- Downloads official DES APY wheat (Rabi) reports directly from:
  - `https://data.desagri.gov.in/report/crop/crops-printdraft-apy-report?...`
- Parses full district-year tables (not sample points).
- Builds a large ground-truth dataset (`state`, `district`, `year`, `area`, `production`, `yield`).
- Generates district map(s):
  - Ground-truth area map (always).
  - Validation error map (when model predictions are provided).
- Computes validation metrics (MAE, RMSE, MAPE, R², bias).

## Run (ground truth + map)
```powershell
& 'C:\Users\vaibh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  wheat_groundtruth_validation.py `
  --states Punjab Haryana `
  --start-year 1997 `
  --end-year 2022 `
  --season Rabi `
  --map-year 2022 `
  --output-dir outputs
```

## Run validation with your model output CSV
Model CSV must contain:
- `state`
- `district`
- `year_start`
- `pred_area_ha`

```powershell
& 'C:\Users\vaibh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  wheat_groundtruth_validation.py `
  --states Punjab Haryana `
  --start-year 1997 `
  --end-year 2022 `
  --season Rabi `
  --map-year 2022 `
  --model-csv path\to\your_model_predictions.csv `
  --output-dir outputs
```

## Run validation with Earth Engine predictions (optional)
```powershell
& 'C:\Users\vaibh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  wheat_groundtruth_validation.py `
  --states Punjab Haryana `
  --start-year 1997 `
  --end-year 2022 `
  --season Rabi `
  --map-year 2022 `
  --use-gee `
  --gee-project whee-486607 `
  --ndvi-threshold 0.35 `
  --output-dir outputs
```

## Output files
- [`outputs/des_wheat_ground_truth.csv`](/D:/agritech/outputs/des_wheat_ground_truth.csv)
- [`outputs/ground_truth_wheat_area_map_2022.html`](/D:/agritech/outputs/ground_truth_wheat_area_map_2022.html)
- [`outputs/validation_summary.json`](/D:/agritech/outputs/validation_summary.json)
- If validation is run:
  - `outputs/district_validation_joined.csv`
  - `outputs/validation_error_map_<year>.html`
