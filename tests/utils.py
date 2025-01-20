import xarray as xr
import numpy as np
import pandas as pd


def create_sample_forecast(path):
    """Create a sample forecast dataset for testing."""
    ds = xr.Dataset(
        data_vars={
            "wind_speed": (
                ["time", "prediction_timedelta", "latitude", "longitude"],
                np.random.rand(5, 3, 4, 4),
            ),
            "temperature": (
                ["time", "prediction_timedelta", "latitude", "longitude"],
                np.random.rand(5, 3, 4, 4),
            ),
        },
        coords={
            "time": pd.date_range("2023-01-01", periods=5),
            "prediction_timedelta": pd.timedelta_range(
                start="0h", periods=3, freq="1h"
            ),
            "latitude": np.linspace(40, 50, 4),
            "longitude": np.linspace(0, 10, 4),
        },
    )
    ds.to_zarr(path)


def create_sample_benchmarks(path):
    """Create sample benchmark results for testing."""
    ds = xr.Dataset(
        data_vars={
            "wind_speed": (["lead_time", "station_id"], np.random.rand(3, 5)),
            "temperature": (["lead_time", "station_id"], np.random.rand(3, 5)),
        },
        coords={
            "lead_time": pd.timedelta_range(start="0h", periods=3, freq="1h"),
            "station_id": range(5),
            "metric": ["rmse", "mbe"],
        },
    )
    ds.to_zarr(path)
