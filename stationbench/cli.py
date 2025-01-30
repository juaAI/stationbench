from stationbench import calculate_metrics as calculate_metrics_api
from stationbench import compare_forecasts as compare_forecasts_api
from stationbench.calculate_metrics import get_parser as get_calculate_parser
from stationbench.compare_forecasts import get_parser as get_compare_parser


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
        use_dask=args.use_dask,
    )
    return metrics


def compare_forecasts():
    """CLI entry point for compare_forecasts"""
    parser = get_compare_parser()
    args = parser.parse_args()

    # Call the API function with CLI arguments
    compare_forecasts_api(
        benchmark_datasets_locs=args.benchmark_datasets_locs,
        run_name=args.run_name,
        regions=args.regions,
    )
