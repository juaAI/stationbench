from typing import Optional, Dict, List
import pandas as pd

def calculate_metrics(
    forecast_loc: str,
    ground_truth_loc: str = "https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr",
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    output_loc: str = None,
    region: str = None,
    name_10m_wind_speed: Optional[str] = None,
    name_2m_temperature: Optional[str] = None
) -> None:
    """
    Calculate verification metrics for weather forecasts.
    
    Args:
        forecast_loc: Location of the forecast data
        ground_truth_loc: Location of the ground truth data
        start_date: Start date for benchmarking
        end_date: End date for benchmarking
        output_loc: Output path for benchmarks
        region: Region to benchmark
        name_10m_wind_speed: Name of 10m wind speed variable
        name_2m_temperature: Name of 2m temperature variable
    """
    from .calculate_metrics import main
    
    class Args:
        pass
    
    args = Args()
    args.forecast_loc = forecast_loc
    args.ground_truth_loc = ground_truth_loc
    args.start_date = pd.Timestamp(start_date)
    args.end_date = pd.Timestamp(end_date)
    args.output_loc = output_loc
    args.region = region
    args.name_10m_wind_speed = name_10m_wind_speed
    args.name_2m_temperature = name_2m_temperature
    
    main(args)

def compare_forecasts(
    evaluation_benchmarks_loc: str,
    reference_benchmark_locs: Dict[str, str],
    run_name: str,
    regions: List[str]
) -> None:
    """
    Compare multiple forecasts and visualize results.
    
    Args:
        evaluation_benchmarks_loc: Path to evaluation benchmarks
        reference_benchmark_locs: Dictionary of reference benchmark locations
        run_name: W&B run name
        regions: List of regions to analyze
    """
    from .compare_forecasts import main
    
    class Args:
        pass
    
    args = Args()
    args.evaluation_benchmarks_loc = evaluation_benchmarks_loc
    args.reference_benchmark_locs = str(reference_benchmark_locs)
    args.run_name = run_name
    args.regions = ",".join(regions)
    
    main(args)
