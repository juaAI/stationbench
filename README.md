# StationBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14671604.svg)](https://doi.org/10.5281/zenodo.14671604) [![Jua.ai](https://img.shields.io/badge/Website-jua.ai-blue)](https://www.jua.ai/) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/juaai/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StationBench is a Python library for benchmarking weather forecasts against weather station data. It provides tools to calculate metrics, visualize results, and compare different forecast models.

## Features

- Pre-processed ground truth data from 10,000+ weather stations around the world included in the package
- Calculate RMSE and other metrics between forecasts and ground truth data
- Support for multiple weather variables (temperature, wind speed, solar radiation)
- Regional analysis capabilities (Europe, North America, Global, etc.)
- Integration with Weights & Biases for experiment tracking

## Installation

```bash
pip install stationbench
```

## Documentation

Full documentation is available in the [docs/](./docs/) directory:
- [Setup](docs/setup.md) - How to setup StationBench
- [Tutorial](docs/tutorial.ipynb) - Basic usage of StationBench

## Quick Start

### Data Format Requirements

#### Forecast Data
- Must include dimensions: latitude, longitude, time
- Variables should include:
  - 10m_wind_speed (or custom name)
  - 2m_temperature (or custom name)

#### Ground Truth Data

Stationbench comes with ready-to-use weather stations from around the world. The benchmarking data is a subset of the [Meteostat](https://dev.meteostat.net/) dataset. It contains weather data from 2018-2024 for 10m wind speed and 2m temperature. The data is provided by the following organizations:
- Deutscher Wetterdienst
- NOAA
- Government of Canada
- MET Norway
- European Data Portal
- Offene Daten Österreich

Source: [Meteostat](https://dev.meteostat.net/) ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode))

The benchmarking data can be accessed from `https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr`.

![Map of weather stations used for benchmarking](docs/assets/stations_2023_map.png)

![Number of stations reporting over time](docs/assets/stations_2018-2024.png)

Besides the provided benchmarking data, you can also use your own ground truth data. The ground truth data must be in zarr format and must include the following dimensions and coordinates:
- Must include dimensions: station_id, time
- Must include coordinates: latitude, longitude

### Calculate Metrics
This script computes metrics (RMSE only for now) by comparing forecast data against ground truth data for specified time periods and regions. Output are RMSE benchmarks for different variables and lead times in the format of the ground truth data.

#### Options
- `--forecast`: Location of the forecast data (required)
- `--stations`: Location of the ground truth data (defaults to https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr)
- `--start_date`: Start date for benchmarking (required)
- `--end_date`: End date for benchmarking (required)
- `--output`: Output path for benchmarks (required)
- `--region`: Region to benchmark (see `regions.py` for available regions)
- `--name_10m_wind_speed`: Name of 10m wind speed variable (optional)
- `--name_2m_temperature`: Name of 2m temperature variable (optional)
- `--use_dask`: Enable parallel computation with Dask (recommended for datasets >10GB)

If variable name is not provided, no metrics will be computed for that variable.

#### Example usage
```bash
poetry run python stationbench/calculate_metrics.py \
    --forecast forecast.zarr \
    --start_date 2023-01-01 --end_date 2023-12-31 --output forecast-rmse.zarr \
    --region europe --name_10m_wind_speed "10si" --name_2m_temperature "2t"
```

### Compare forecasts

After generating the metrics, you can use the `compare_forecasts.py` script to compute metrics, create visualizations, and log the results to Weights & Biases (W&B).

#### What it does

The `compare_forecasts.py` script:
1. Computes RMSE (Root Mean Square Error) and skill scores for different variables and lead time ranges.
2. Generates geographical scatter plots showing the spatial distribution of errors.
3. Creates line plots showing the temporal evolution of errors.
4. Logs all visualizations and metrics to a W&B run.

#### Options
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

### Usage

StationBench can be used either as a Python package or through command-line interfaces.

#### Python Package Usage

```python
import stationbench

# Calculate metrics
stationbench.calculate_metrics(
    forecast="forecast.zarr",
    start_date="2023-01-01",
    end_date="2023-12-31",
    output="forecast-rmse.zarr",
    region="europe",
    name_10m_wind_speed="10si",
    name_2m_temperature="2t"
)

# Compare forecasts
stationbench.compare_forecasts(
    evaluation_benchmarks_loc="forecast-rmse.zarr",
    reference_benchmark_locs={"HRES": "hres-rmse.zarr"},
    run_name="my-comparison",
    regions=["europe"]
)
```

#### Command-Line Usage

Calculate metrics for a forecast dataset:

```bash
stationbench-calculate \
    --forecast path/to/forecast.zarr \
    --start_date 2023-01-01 \
    --end_date 2023-12-31 \
    --output forecast-rmse.zarr \
    --region europe \
    --name_10m_wind_speed "10si" \
    --name_2m_temperature "2t"
    [--use_dask]  # Optional: Enable parallel computation with Dask
```

For small datasets, it's recommended to run without Dask. For large datasets
(>10GB), enabling Dask with `--use_dask` can improve performance.

Compare forecasts:
```bash
stationbench-compare \
    --evaluation_benchmarks_loc forecast-rmse.zarr \
    --reference_benchmark_locs '{"HRES": "hres-rmse.zarr"}' \
    --regions europe \
    --run_name wandb-run-name
```

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Metrics

StationBench calculates the following verification metrics at the moment:

- **RMSE (Root Mean Square Error)**: Measures the average magnitude of forecast errors
- **MBE (Mean Bias Error)**: Measures the average direction and magnitude of forecast bias

We plan to add more benchmarking metrics in the future...
