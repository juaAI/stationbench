import pytest
import xarray as xr
import numpy as np
from stationbench.compare_forecasts import PointBasedBenchmarking
import wandb
import pandas as pd
import os
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_wandb_run():
    mock_run = MagicMock()
    mock_run.id = "test_run"

    # Mock the log_artifact method
    mock_artifact = MagicMock()
    mock_artifact.wait = lambda: None
    mock_run.log_artifact.return_value = mock_artifact

    # Mock the use_artifact method to raise CommError
    mock_run.use_artifact.side_effect = wandb.errors.CommError("Artifact not found")

    return mock_run


def test_identical_forecast_skill_score(mock_wandb_run, tmp_path):
    # Create a simple benchmark dataset with the correct metric dimension
    ds = xr.Dataset(
        data_vars={
            "10m_wind_speed": (
                ["metric", "lead_time", "station_id"],
                np.random.rand(2, 5, 3),
            ),  # 2 metrics: rmse, mbe
        },
        coords={
            "metric": ["rmse", "mbe"],
            "lead_time": pd.timedelta_range(start="0h", periods=5, freq="6h"),
            "station_id": range(3),
            "latitude": ("station_id", [45, 46, 47]),
            "longitude": ("station_id", [5, 6, 7]),
        },
    )

    # Save the dataset to a temporary zarr store
    ds_path = os.path.join(tmp_path, "test_data.zarr")
    ds.to_zarr(ds_path)

    # Mock wandb.Plotly to return a simple dict-like object
    class MockPlotly:
        def __init__(self, fig):
            self.fig = fig

    def mock_plotly(fig):
        return MockPlotly(fig)

    # Initialize the benchmarking class with mocked wandb
    with patch("wandb.Plotly", side_effect=mock_plotly):
        benchmarking = PointBasedBenchmarking(wandb_run=mock_wandb_run)

        # Generate metrics using the same dataset path as both evaluation and reference
        metrics = benchmarking.generate_metrics(
            evaluation_benchmarks=ds,
            reference_benchmark_locs={"reference": ds_path},
            region_names=["europe"],
        )

        # Check spatial plots
        for var in ds.data_vars:
            for lead_range_name in ["Short term (6-48 hours)", "Mid term (3-7 days)"]:
                for metric_type in ["RMSE", "MBE", "skill_score"]:
                    plot_key = f"stations_spatial_metrics/{metric_type}/{var} {lead_range_name}"
                    assert plot_key in metrics

                    # Get the plotly figure
                    fig = metrics[plot_key].fig

                    # Check that all color values (skill scores) are approximately 0
                    for trace in fig.data:
                        if hasattr(trace, "marker") and hasattr(trace.marker, "color"):
                            color_values = np.array(trace.marker.color)
                            if metric_type == "skill_score":
                                np.testing.assert_allclose(color_values, 0, atol=1e-10)

        # Check temporal plots
        for var in ds.data_vars:
            for metric_type in ["RMSE", "MBE", "skill_score"]:
                plot_key = (
                    f"stations_temporal_metrics/{metric_type}/{var}/europe_line_plot"
                )
                assert plot_key in metrics

                # Get the plotly figure
                fig = metrics[plot_key].fig

                # Check that all y values (skill scores) are approximately 0
                for trace in fig.data:
                    if hasattr(trace, "y"):
                        y_values = np.array(trace.y)
                        if metric_type == "skill_score":
                            np.testing.assert_allclose(y_values, 0, atol=1e-10)
