import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stationbench.calculate_metrics import (
    generate_benchmarks,
    interpolate_to_stations,
    main,
    prepare_forecast,
    prepare_stations,
)


@pytest.fixture
def sample_forecast():
    """Create a sample forecast dataset."""
    times = pd.date_range("2022-01-01", "2022-01-02", freq="24h")  # Just 2 init times
    lead_times = pd.timedelta_range("0h", "24h", freq="24h")  # Just 2 lead times
    lats = np.array([45.0, 55.0])  # Just 2 latitudes
    lons = np.array([0.0, 10.0])  # Just 2 longitudes

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


@pytest.fixture
def sample_point_forecast():
    """Create a sample point-based forecast dataset."""
    times = pd.date_range("2022-01-01", "2022-01-02", freq="24h")  # Just 2 init times
    lead_times = pd.timedelta_range("0h", "24h", freq="24h")  # Just 2 lead times
    stations = ["ST1", "ST2"]  # Two stations
    lats = [50.0, 51.0]
    lons = [5.0, 6.0]

    ds = xr.Dataset(
        data_vars={
            "2m_temperature": (
                ("time", "prediction_timedelta", "station_id"),
                np.random.randn(len(times), len(lead_times), len(stations)),
            ),
            "10m_wind_speed": (
                ("time", "prediction_timedelta", "station_id"),
                np.random.randn(len(times), len(lead_times), len(stations)),
            ),
        },
        coords={
            "time": times,
            "prediction_timedelta": lead_times,
            "station_id": stations,
            "latitude": ("station_id", lats),
            "longitude": ("station_id", lons),
        },
    )
    return ds


@pytest.fixture
def sample_stations():
    """Create a sample stations dataset."""
    times = pd.date_range("2022-01-01", "2022-01-03", freq="24h")  # Just daily data
    stations = ["ST1"]  # Just 1 station
    lats = [50.0]
    lons = [5.0]

    ds = xr.Dataset(
        data_vars={
            "2m_temperature": (
                ("time", "station_id"),
                np.random.randn(len(times), len(stations)),
            ),
            "10m_wind_speed": (
                ("time", "station_id"),
                np.random.randn(len(times), len(stations)),
            ),
        },
        coords={
            "time": times,
            "station_id": stations,
            "latitude": ("station_id", lats),
            "longitude": ("station_id", lons),
        },
    )
    return ds


def test_prepare_forecast(sample_forecast):
    """Test forecast preparation."""
    forecast = prepare_forecast(
        forecast=sample_forecast,
        region_name="europe",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 3),
        wind_speed_name="10m_wind_speed",
        temperature_name="2m_temperature",
    )

    # Check time handling
    assert "init_time" in forecast.dims
    assert "lead_time" in forecast.dims
    assert "valid_time" in forecast.coords

    # Check region selection
    assert forecast.latitude.min() >= 36.0
    assert forecast.latitude.max() <= 72.0
    assert forecast.longitude.min() >= -15.0
    assert forecast.longitude.max() <= 45.0

    # Check variable renaming
    assert "10m_wind_speed" in forecast.data_vars
    assert "2m_temperature" in forecast.data_vars
    assert "wind" not in forecast.data_vars
    assert "t2m" not in forecast.data_vars


def test_prepare_stations(sample_stations):
    """Test stations preparation."""
    stations = prepare_stations(stations=sample_stations, region_name="europe")

    # Check region filtering
    assert stations.latitude.min() >= 36.0
    assert stations.latitude.max() <= 72.0
    assert stations.longitude.min() >= -15.0
    assert stations.longitude.max() <= 45.0


def test_full_pipeline(sample_forecast, sample_stations):
    """Test the full pipeline."""
    args = argparse.Namespace(
        forecast=sample_forecast,
        stations=sample_stations,
        region="europe",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 2),
        name_10m_wind_speed="10m_wind_speed",
        name_2m_temperature="2m_temperature",
        use_dask=False,
        output=None,
    )

    benchmarks = main(args)
    assert isinstance(benchmarks, xr.Dataset)

    assert set(benchmarks.dims) == {"lead_time", "station_id", "metric"}
    assert set(benchmarks.metric.values) == {"rmse", "mbe"}
    assert set(benchmarks.data_vars) == {"10m_wind_speed", "2m_temperature"}


def test_full_pipeline_with_point_based_forecast(
    sample_point_forecast, sample_stations
):
    """Test the full pipeline."""
    args = argparse.Namespace(
        forecast=sample_point_forecast,
        stations=sample_stations,
        region="europe",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 2),
        name_10m_wind_speed="10m_wind_speed",
        name_2m_temperature="2m_temperature",
        use_dask=False,
        output=None,
    )

    benchmarks = main(args)
    assert isinstance(benchmarks, xr.Dataset)

    assert set(benchmarks.dims) == {"lead_time", "station_id", "metric"}
    assert set(benchmarks.metric.values) == {"rmse", "mbe"}
    assert set(benchmarks.data_vars) == {"10m_wind_speed", "2m_temperature"}


def test_rmse_calculation_matches_manual(sample_forecast, sample_stations):
    """Test that the RMSE calculation matches a manual calculation for a simple case."""
    # Prepare forecast with known values
    forecast = prepare_forecast(
        forecast=sample_forecast,
        region_name="europe",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 2),
        wind_speed_name="10m_wind_speed",
        temperature_name="2m_temperature",
    )
    forecast["10m_wind_speed"][:] = 5.0  # Set all forecast values to 5.0

    # Prepare stations with known values
    stations = prepare_stations(stations=sample_stations, region_name="europe")
    stations["10m_wind_speed"][:] = 3.0  # Set all ground truth values to 3.0

    # Interpolate
    forecast_interp = interpolate_to_stations(forecast, stations)

    # Calculate RMSE using generate_benchmarks
    benchmarks = generate_benchmarks(forecast=forecast_interp, stations=stations)

    # Manual RMSE calculation
    # RMSE = sqrt(mean((forecast - stations)²))
    # In this case: sqrt(mean((5.0 - 3.0)²)) = sqrt(4) = 2.0
    expected_rmse = 2.0

    # Check if the calculated RMSE matches the expected value
    np.testing.assert_allclose(
        benchmarks.sel(metric="rmse")["10m_wind_speed"].values,
        expected_rmse,
        rtol=1e-6,
        err_msg="RMSE calculation does not match manual calculation",
    )


def test_mbe_calculation(sample_forecast, sample_stations):
    """Test that MBE calculation correctly handles both magnitude and sign."""
    # Prepare datasets
    forecast = sample_forecast.copy()
    forecast = forecast.rename({"time": "init_time"})
    forecast = forecast.rename({"prediction_timedelta": "lead_time"})
    forecast.coords["valid_time"] = forecast.init_time + forecast.lead_time
    stations = sample_stations.copy()

    # Test positive bias
    forecast["10m_wind_speed"][:] = 5.0
    stations["10m_wind_speed"][:] = 3.0
    metrics = generate_benchmarks(forecast=forecast, stations=stations)

    # Check magnitude matches manual calculation
    expected_mbe = 2.0  # 5.0 - 3.0 = 2.0
    np.testing.assert_allclose(
        metrics.sel(metric="mbe")["10m_wind_speed"].values,
        expected_mbe,
        rtol=1e-6,
        err_msg="MBE calculation does not match manual calculation for positive bias",
    )

    # Test negative bias
    forecast["10m_wind_speed"][:] = 1.0
    metrics = generate_benchmarks(forecast=forecast, stations=stations)

    # Check magnitude matches manual calculation
    expected_mbe = -2.0  # 1.0 - 3.0 = -2.0
    np.testing.assert_allclose(
        metrics.sel(metric="mbe")["10m_wind_speed"].values,
        expected_mbe,
        rtol=1e-6,
        err_msg="MBE calculation does not match manual calculation for negative bias",
    )


def test_invalid_path():
    """Test handling of invalid file paths."""
    with pytest.raises(Exception):  # Should raise some kind of file not found error
        prepare_forecast(
            forecast="invalid/path.zarr",
            region_name="europe",
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 3),
        )

    with pytest.raises(Exception):
        prepare_stations(stations="invalid/path.zarr", region_name="europe")


def test_invalid_region(sample_forecast):
    """Test handling of invalid region names."""
    with pytest.raises(KeyError):
        prepare_forecast(
            forecast=sample_forecast,
            region_name="invalid_region",
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 3),
        )


def test_invalid_dates(sample_forecast):
    """Test handling of invalid date ranges."""
    with pytest.raises(ValueError):
        prepare_forecast(
            forecast=sample_forecast,
            region_name="europe",
            start_date=datetime(2022, 2, 1),
            end_date=datetime(2022, 1, 1),
        )
