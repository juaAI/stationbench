import argparse
from datetime import datetime
from typing import Optional, Union
import json

from .calculate_metrics import main as calculate_metrics_main
from .compare_forecasts import main as compare_forecasts_main

def calculate_metrics(
    forecast_loc: str,
    ground_truth_loc: str = "https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr",
    start_date: Union[str, datetime] = None,
    end_date: Union[str, datetime] = None,
    output_loc: str = None,
    region: str = "europe",
    name_10m_wind_speed: Optional[str] = None,
    name_2m_temperature: Optional[str] = None,
) -> None:
    """Calculate metrics for a forecast dataset.

    Args:
        forecast_loc: Location of the forecast dataset
        ground_truth_loc: Location of the ground truth dataset
        start_date: Start date for evaluation
        end_date: End date for evaluation
        output_loc: Location to save the metrics
        region: Region to evaluate
        name_10m_wind_speed: Name of wind speed variable in forecast
        name_2m_temperature: Name of temperature variable in forecast
    """
    args = argparse.Namespace(
        forecast_loc=forecast_loc,
        ground_truth_loc=ground_truth_loc,
        start_date=start_date,
        end_date=end_date,
        output_loc=output_loc,
        region=region,
        name_10m_wind_speed=name_10m_wind_speed,
        name_2m_temperature=name_2m_temperature,
    )
    
    return calculate_metrics_main(args)

def compare_forecasts(
    evaluation_benchmarks_loc: str,
    reference_benchmark_locs: Union[str, dict[str, str]],
    run_name: str,
    regions: Union[str, list[str]],
) -> None:
    """Compare forecast benchmarks.

    Args:
        evaluation_benchmarks_loc: Location of the evaluation benchmarks
        reference_benchmark_locs: Dictionary of reference benchmark locations or JSON string
        run_name: Name for the W&B run
        regions: Regions to evaluate (string or list of strings)
    """
    # Handle reference_benchmark_locs as either dict or string
    if isinstance(reference_benchmark_locs, str):
        try:
            reference_benchmark_locs = json.loads(reference_benchmark_locs)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for reference_benchmark_locs: {e}")
    
    args = argparse.Namespace(
        evaluation_benchmarks_loc=evaluation_benchmarks_loc,
        reference_benchmark_locs=reference_benchmark_locs,
        run_name=run_name,
        regions=regions if isinstance(regions, list) else [r.strip() for r in regions.split(",")],
    )
    
    return compare_forecasts_main(args)
