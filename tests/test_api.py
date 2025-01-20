import pandas as pd
import numpy as np
import xarray as xr
import pytest
from stationbench import calculate_metrics


@pytest.fixture
def sample_forecast():
    """Create a sample forecast dataset."""
    # Minimal grid covering Europe
    lats = np.array([40.0, 50.0, 60.0])  # Just 3 latitudes
    lons = np.array([-10.0, 0.0, 10.0])  # Just 3 longitudes
    times = pd.date_range("2023-01-01", "2023-01-02", freq="12h")  # 3 init times
    lead_times = pd.timedelta_range("0h", "24h", freq="12h")  # 3 lead times

    ds = xr.Dataset(
        data_vars={
            "2m_temperature": (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                np.random.randn(len(times), len(lead_times), len(lats), len(lons)),
            ),
            "10m_wind_speed": (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                np.random.randn(len(times), len(lead_times), len(lats), len(lons)),
            ),
        },
        coords={
            "time": times,
            "prediction_timedelta": lead_times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


def test_calculate_metrics_with_dataset(sample_forecast, tmp_path):
    """Test calculate_metrics with xarray dataset input."""
    metrics = calculate_metrics(
        forecast=sample_forecast,
        start_date="2023-01-01",
        end_date="2023-01-31",
        name_10m_wind_speed="10m_wind_speed",
        name_2m_temperature="2m_temperature",
    )

    assert isinstance(metrics, xr.Dataset)
    assert "10m_wind_speed" in metrics.data_vars
    assert "2m_temperature" in metrics.data_vars


def test_calculate_metrics_with_path(sample_forecast, tmp_path):
    """Test calculate_metrics with file path input."""
    forecast_path = tmp_path / "forecast.zarr"
    sample_forecast.to_zarr(forecast_path)

    metrics = calculate_metrics(
        forecast=str(forecast_path),
        start_date="2023-01-01",
        end_date="2023-01-31",
        output=str(tmp_path / "metrics.zarr"),
        name_10m_wind_speed="10m_wind_speed",
        name_2m_temperature="2m_temperature",
    )

    assert isinstance(metrics, xr.Dataset)
    assert (tmp_path / "metrics.zarr").exists()
    assert "10m_wind_speed" in metrics.data_vars
    assert "2m_temperature" in metrics.data_vars
