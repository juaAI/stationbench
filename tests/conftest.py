import pytest
import xarray as xr
import numpy as np
import pandas as pd
from unittest.mock import patch


@pytest.fixture
def sample_forecast_dataset():
    # Create a small synthetic forecast dataset
    times = pd.date_range("2024-01-01", "2024-01-02", freq="D")
    lats = np.linspace(45, 50, 3)
    lons = np.linspace(-5, 5, 4)
    prediction_timedeltas = np.array([0, 6, 12, 24], dtype="timedelta64[h]")

    ds = xr.Dataset(
        data_vars={
            "10m_wind_speed": (
                ["init_time", "lead_time", "latitude", "longitude"],
                np.random.rand(
                    len(times), len(prediction_timedeltas), len(lats), len(lons)
                ),
            )
        },
        coords={
            "time": times,
            "prediction_timedelta": prediction_timedeltas,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


@pytest.fixture
def sample_stations_dataset():
    # Create a small synthetic ground truth dataset
    times = pd.date_range("2024-01-01", "2024-01-03", freq="h")
    stations = range(3)

    ds = xr.Dataset(
        data_vars={
            "10m_wind_speed": (
                ["time", "station_id"],
                np.random.rand(len(times), len(stations)),
            ),
            "latitude": ("station_id", [46, 47, 48]),
            "longitude": ("station_id", [-2, 0, 2]),
        },
        coords={
            "time": times,
            "station_id": stations,
        },
    )
    return ds


@pytest.fixture
def mock_wandb():
    """Mock W&B for testing."""
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
        yield
