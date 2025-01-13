TODO: Adjust the docs

# Point-based benchmarking
Point based benchmarking is split in two steps:
1. Calculate metrics
2. Compare forecasts

## Benchmarking data

The provided benchmarking data is a subset of the [Meteostat](https://dev.meteostat.net/) dataset. It contains weather data from 2018-2024 for 10m wind speed and 2m temperature. The data is provided by the following organizations:
- Deutscher Wetterdienst
- NOAA
- Government of Canada
- MET Norway
- European Data Portal
- Offene Daten Österreich

Source: [Meteostat](https://dev.meteostat.net/) ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode))


The benchmarking data can be accessed from `ADD_DATA_PATH`.

![Map of weather stations used for benchmarking](docs/stations_2023_map.png)

![Number of stations reporting over time](docs/stations_2018-2024.png)



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

### Example usage
```bash
poetry run python stationbench/calculate_metrics.py \
    --forecast_loc forecast.zarr \
    --ground_truth_loc ground_truth.zarr \
    --start_date 2023-01-01 --end_date 2023-12-31 --output_loc forecast-rmse.zarr \
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
poetry run python stationbench/compare_forecasts.py \
    --evaluation_benchmarks_loc forecast-rmse.zarr \
    --reference_benchmark_locs '{"HRES": "hres-rmse.zarr"}' \
    --regions europe \
    --run_name wandb-run-name
```
