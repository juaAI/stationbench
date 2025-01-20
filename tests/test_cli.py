import pytest
from unittest.mock import patch
import json
from stationbench.cli import calculate_metrics, compare_forecasts
import pandas as pd
import numpy as np
import xarray as xr


def test_calculate_metrics_cli(tmp_path):
    """Test the calculate_metrics CLI."""
    forecast_path = tmp_path / "forecast.zarr"
    output_path = tmp_path / "metrics.zarr"

    # Create sample forecast data
    create_sample_forecast(forecast_path)  # Helper function to create test data

    with patch(
        "sys.argv",
        [
            "stationbench-calculate",
            "--forecast",
            str(forecast_path),
            "--start_date",
            "2023-01-01",
            "--end_date",
            "2023-01-31",
            "--output",
            str(output_path),
            "--name_10m_wind_speed",
            "wind",
            "--name_2m_temperature",
            "t2m",
        ],
    ):
        calculate_metrics()

    assert output_path.exists()


def test_compare_forecasts_cli(tmp_path):
    """Test the compare_forecasts CLI."""
    eval_path = tmp_path / "eval.zarr"
    ref_path = tmp_path / "ref.zarr"

    # Create sample benchmark data
    create_sample_benchmarks(eval_path)
    create_sample_benchmarks(ref_path)

    ref_locs = json.dumps({"reference": str(ref_path)})

    with (
        patch("stationbench.cli.compare_forecasts_api") as mock_compare,
        patch(
            "sys.argv",
            [
                "stationbench-compare",
                "--evaluation_benchmarks_loc",
                str(eval_path),
                "--reference_benchmark_locs",
                ref_locs,
                "--run_name",
                "test-run",
                "--regions",
                "europe",
            ],
        ),
    ):
        compare_forecasts()

        # Verify the API was called with correct arguments
        mock_compare.assert_called_once()
        args = mock_compare.call_args[1]  # Get kwargs
        assert args["evaluation_benchmarks_loc"] == str(eval_path)
        assert args["reference_benchmark_locs"] == ref_locs  # Compare with JSON string
        assert args["run_name"] == "test-run"
        assert args["regions"] == "europe"


@pytest.fixture
def mock_wandb():
    """Mock W&B for testing."""
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
        yield


def create_sample_forecast(path):
    """Create a sample forecast dataset and save it."""
    ds = xr.Dataset(
        data_vars={
            "t2m": (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                np.random.randn(3, 3, 3, 3),
            ),  # Minimal 3x3 grid
            "wind": (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                np.random.randn(3, 3, 3, 3),
            ),
        },
        coords={
            "time": pd.date_range(
                "2023-01-01", "2023-01-02", freq="12h"
            ),  # 3 init times
            "prediction_timedelta": pd.timedelta_range(
                "0h", "24h", freq="12h"
            ),  # 3 lead times
            "latitude": np.array([40.0, 50.0, 60.0]),  # Just 3 latitudes
            "longitude": np.array([-10.0, 0.0, 10.0]),  # Just 3 longitudes
        },
    )
    ds.to_zarr(path)


def create_sample_benchmarks(path):
    """Create a sample benchmarks dataset and save it."""
    ds = xr.Dataset(
        data_vars={
            "10m_wind_speed": (("lead_time", "station_id"), np.random.randn(3, 2)),
            "2m_temperature": (("lead_time", "station_id"), np.random.randn(3, 2)),
        },
        coords={
            "lead_time": pd.timedelta_range(
                "0h", "24h", freq="12h"
            ),  # Match forecast lead times
            "station_id": ["ST1", "ST2"],  # Just 2 stations
            "latitude": ("station_id", [45.0, 55.0]),  # Within forecast domain
            "longitude": ("station_id", [-5.0, 5.0]),
        },
    )
    ds.to_zarr(path)
