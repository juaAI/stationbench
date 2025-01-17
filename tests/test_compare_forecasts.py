import pytest
import xarray as xr
import numpy as np
from stationbench.compare_forecasts import PointBasedBenchmarking
import wandb
import pandas as pd
import os
import json
import codecs


@pytest.fixture
def mock_wandb_run():
    class MockRun:
        def __init__(self):
            self.id = "test_run"

        def log_artifact(self, artifact):
            class WaitObject:
                def wait(self):
                    pass

            return WaitObject()

        def use_artifact(self, name):
            raise wandb.errors.CommError("Artifact not found")

    return MockRun()


def test_identical_forecast_skill_score(mock_wandb_run, tmp_path):
    # Create a simple benchmark dataset
    ds = xr.Dataset(
        data_vars={
            "temperature": (["lead_time", "station_id"], np.random.rand(5, 3)),
            "wind_speed": (["lead_time", "station_id"], np.random.rand(5, 3)),
        },
        coords={
            "lead_time": pd.timedelta_range(start="0h", periods=5, freq="6h"),
            "station_id": range(3),
            "latitude": ("station_id", [45, 46, 47]),
            "longitude": ("station_id", [5, 6, 7]),
        },
    )

    # Save the dataset to a temporary zarr store
    ds_path = os.path.join(tmp_path, "test_data.zarr")
    ds.to_zarr(ds_path)

    # Initialize the benchmarking class
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
            plot_key = f"point-based-benchmarking/spatial_error/skill_score/{var} {lead_range_name}"
            assert plot_key in metrics

            # Get the plotly JSON data
            plotly_obj = metrics[plot_key]
            with codecs.open(plotly_obj._path, "r", encoding="utf-8") as f:
                fig_dict = json.load(f)

            # Check that all color values (skill scores) are approximately 0
            for trace in fig_dict["data"]:
                if "marker" in trace and "color" in trace["marker"]:
                    color_values = np.array(trace["marker"]["color"])
                    np.testing.assert_allclose(color_values, 0, atol=1e-10)

    # Check temporal plots
    for var in ds.data_vars:
        plot_key = f"point-based-benchmarking/temporal_error/skill_score/{var}/europe_line_plot"
        assert plot_key in metrics

        # Get the plotly JSON data
        plotly_obj = metrics[plot_key]
        with codecs.open(plotly_obj._path, "r", encoding="utf-8") as f:
            fig_dict = json.load(f)

        # Check that all y values (skill scores) are approximately 0
        for trace in fig_dict["data"]:
            if "y" in trace:
                y_values = np.array(trace["y"])
                np.testing.assert_allclose(y_values, 0, atol=1e-10)
