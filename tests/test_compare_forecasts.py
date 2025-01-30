import pytest
import xarray as xr
import numpy as np
from stationbench.compare_forecasts import (
    PointBasedBenchmarking,
    calculate_skill_scores,
)
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
        benchmarking = PointBasedBenchmarking(
            wandb_run=mock_wandb_run, region_names=["europe"]
        )

        # Generate metrics using the same dataset path as both evaluation and reference
        temporal_metrics_datasets, spatial_metrics_datasets = (
            benchmarking.process_temporal_and_spatial_metrics(
                benchmark_datasets={"evaluation": ds, "reference": ds},
            )
        )

        temporal_ss, spatial_ss = calculate_skill_scores(
            temporal_metrics_datasets, spatial_metrics_datasets
        )

        # Check that the skill scores are 0
        assert temporal_ss["10m_wind_speed"].sum() == 0
        assert spatial_ss["10m_wind_speed"].sum() == 0
