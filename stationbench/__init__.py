import argparse
from datetime import datetime
from typing import Optional, Union
import json
import xarray as xr

from .calculate_metrics import main as calculate_metrics_main
from .compare_forecasts import main as compare_forecasts_main
from .utils.io import load_dataset


def calculate_metrics(
    forecast: Union[str, xr.Dataset],
    stations: Union[
        str, xr.Dataset
    ] = "https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr",
    start_date: Union[str, datetime] = None,
    end_date: Union[str, datetime] = None,
    output: Optional[str] = None,
    region: str = "europe",
    name_10m_wind_speed: Optional[str] = None,
    name_2m_temperature: Optional[str] = None,
    use_dask: bool = False,
    n_workers: int = 4,
) -> xr.Dataset:
    """Calculate metrics for a forecast dataset.

    Args:
        forecast: Forecast dataset or path
        stations: Ground truth dataset or path
        start_date: Start date for evaluation
        end_date: End date for evaluation
        output: Optional path to save results
        region: Region to evaluate
        name_10m_wind_speed: Name of wind speed variable
        name_2m_temperature: Name of temperature variable
        use_dask: Whether to use Dask for parallel computation (slower for small datasets)
        n_workers: Number of Dask workers to use (default: 4, only used if --use_dask is set and no client exists)
    Returns:
        xr.Dataset: Calculated metrics
    """
    args = argparse.Namespace(
        forecast=load_dataset(forecast),
        stations=load_dataset(stations),
        start_date=start_date,
        end_date=end_date,
        output=output,
        region=region,
        name_10m_wind_speed=name_10m_wind_speed,
        name_2m_temperature=name_2m_temperature,
        use_dask=use_dask,
        n_workers=n_workers,
    )

    return calculate_metrics_main(args)


def compare_forecasts(
    benchmark_datasets_locs: dict[str, str],
    regions: Union[str, list[str]],
    output_dir: str = "stationbench-results",
    wandb_run_name: Optional[str] = None,
) -> None:
    """Compare forecast benchmarks.

    Args:
        benchmark_datasets_locs: Dictionary of benchmark datasets locations
        regions: Regions to evaluate (string or list of strings)
        output_dir: Directory to save the results
    """
    # Handle reference_benchmark_locs as either dict or string
    if isinstance(benchmark_datasets_locs, str):
        try:
            benchmark_datasets_locs = json.loads(benchmark_datasets_locs)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for reference_benchmark_locs: {e}")

    args = argparse.Namespace(
        benchmark_datasets_locs=benchmark_datasets_locs,
        regions=regions
        if isinstance(regions, list)
        else [r.strip() for r in regions.split(",")],
        output_dir=output_dir,
        wandb_run_name=wandb_run_name,
    )
    return compare_forecasts_main(args)
