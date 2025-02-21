from stationbench import calculate_metrics as calculate_metrics_api
from stationbench import compare_forecasts as compare_forecasts_api
from stationbench.calculate_metrics import get_parser as get_calculate_parser
from stationbench.compare_forecasts import get_parser as get_compare_parser


def calculate_metrics():
    """CLI entry point for calculate_metrics"""
    parser = get_calculate_parser()
    args = parser.parse_args()

    metrics = calculate_metrics_api(**vars(args))
    return metrics


def compare_forecasts():
    """CLI entry point for compare_forecasts"""
    parser = get_compare_parser()
    args = parser.parse_args()

    compare_forecasts_api(**vars(args))
