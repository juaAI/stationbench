import xarray as xr
import numpy as np
from stationbench.compare_forecasts import (
    PointBasedBenchmarking,
    calculate_skill_scores,
)
import pandas as pd
import os


def test_identical_forecast_skill_score(tmp_path):
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

    benchmarking = PointBasedBenchmarking(region_names=["europe"])

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
