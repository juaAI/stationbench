import sys
from stationbench.calculate_metrics import main as calculate_main
from stationbench.compare_forecasts import main as compare_main


def calculate_metrics():
    """CLI entry point for calculate_metrics"""
    calculate_main(sys.argv[1:])


def compare_forecasts():
    """CLI entry point for compare_forecasts"""
    compare_main(sys.argv[1:])
