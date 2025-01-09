# Point-based benchmarking
Point based benchmarking is split in two steps:
1. Calculate metrics
2. Compare forecasts

## Calculate Metrics
This script computes metrics (RMSE only for now) by comparing forecast data against ground truth data for specified time periods and regions. Output are RMSE benchmarks for different variables and lead times in the format of the ground truth data.

### Options
- `--forecast_loc`: Location of the forecast data (required)
- `--ground_truth_loc`: Location of the ground truth data (required)
- `--start_date`: Start date for benchmarking (required)
- `--end_date`: End date for benchmarking (required)
- `--output`: Output path for benchmarks (required)
- `--region`: Region to benchmark (see `regions.py` for available regions)
- `--name_10m_wind_speed`: Name of 10m wind speed variable (optional)
- `--name_2m_temperature`: Name of 2m temperature variable (optional)
- `--name_ssrd`: Name of ssrd variable (optional)

If variable name is not provided, no metrics will be computed for that variable.

### Naming Conventions
The following naming conventions are suggested:
- `forecast_loc`: `gs://jua-hindcasts/forecasts/jua/{model}-{start_date}-{end_date}-{region}.zarr` 
- `output_loc`: `gs://jua-benchmarking/forecasts/{source}/{forecast_loc}-rmse.zarr`, where `{source}` is the source of the forecast, e.g. `jua` or `third_party`

### Commonly used datasets
- `ground_truth_loc`: `gs://jua-benchmarking/ground_truth/synoptic/synoptic-2023-1h-v3.zarr`
- high resolution IFS HRES `forecast_loc`: `gs://jua-benchmarking/forecasts/third_party/ifs-fc-2023-0012-6h-2220x4440.zarr`

### Example usage
```bash
poetry run python benchmarking/point_based/calculate_metrics.py \
    --forecast_loc gs://jua-hindcasts/forecasts/jua/EPT1-early-2023-03-18-2023-07-31-europe.zarr \
    --ground_truth_loc gs://jua-benchmarking/ground_truth/synoptic/synoptic-2023-1h-v3.zarr \
    --start_date 2023-03-18 --end_date 2023-07-31 --output_loc gs://jua-benchmarking/forecasts/jua/EPT1-early-2023-03-18-2023-07-31-europe-rmse.zarr \
    --region europe --name_10m_wind_speed "10si" --name_2m_temperature "2t"
```

## Compare forecasts

After generating the metrics, you can use the `compare_forecasts.py` script to compute metrics, create visualizations, and log the results to Weights & Biases (W&B).

### What it does

The `compare_forecasts.py` script:
1. Computes RMSE (Root Mean Square Error) and skill scores for different variables and lead time ranges.
2. Generates geographical scatter plots showing the spatial distribution of errors.
3. Creates line plots showing the temporal evolution of errors.
4. Logs all visualizations and metrics to a W&B run.

### Options
- `--evaluation_benchmarks_loc`: Path to the evaluation benchmarks (required)
- `--reference_benchmark_locs`: Dictionary of reference benchmark locations, the first one is used for skill score (required)
- `--run_name`: W&B run name (required)
- `--regions`: Comma-separated list of regions, see `regions.py` for available regions (required)

### Example
```bash
poetry run python benchmarking/point_based/compare_forecasts.py \
    --evaluation_benchmarks_loc gs://jua-benchmarking/forecasts/jua/EPT1-early-2023-03-18-2023-07-31-europe-rmse.zarr \
    --reference_benchmark_locs '{"HRES": "gs://jua-benchmarking/forecasts/jua/roberto_4_hres_rmse_v3.zarr"}' \
    --regions europe \
    --run_name test_101
```
