import argparse
from stationbench import calculate_metrics as calculate_metrics_api
from stationbench import compare_forecasts as compare_forecasts_api
from stationbench.calculate_metrics import get_parser as get_calculate_parser


def calculate_metrics():
    """CLI entry point for calculate_metrics"""
    parser = get_calculate_parser()
    args = parser.parse_args()

    # Call the API function with CLI arguments
    metrics = calculate_metrics_api(
        forecast=args.forecast,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        region=args.region,
        name_10m_wind_speed=args.name_10m_wind_speed,
        name_2m_temperature=args.name_2m_temperature,
    )
    return metrics


def get_compare_parser():
    parser = argparse.ArgumentParser(description="Compare forecast benchmarks")
    parser.add_argument(
        "--evaluation_benchmarks_loc",
        type=str,
        required=True,
        help="Path to evaluation benchmarks zarr dataset",
    )
    parser.add_argument(
        "--reference_benchmark_locs",
        type=str,
        required=True,
        help="JSON string of reference benchmark locations {'name': 'path'}",
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name for the W&B run"
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="global",
        help="Comma-separated list of regions to evaluate",
    )
    return parser


def compare_forecasts():
    """CLI entry point for compare_forecasts"""
    parser = get_compare_parser()
    args = parser.parse_args()

    # Call the API function with CLI arguments
    compare_forecasts_api(
        evaluation_benchmarks_loc=args.evaluation_benchmarks_loc,
        reference_benchmark_locs=args.reference_benchmark_locs,
        run_name=args.run_name,
        regions=args.regions,
    )
