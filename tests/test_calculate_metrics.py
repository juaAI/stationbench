import xarray as xr
import numpy as np
from datetime import datetime
import pytest
import pandas as pd
import argparse

from stationbench.calculate_metrics import (
    prepare_forecast,
    prepare_stations,
    interpolate_to_stations,
    generate_benchmarks,
    main,
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
            "t2m": (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                np.random.randn(len(times), len(lead_times), len(lats), len(lons)),
            ),
            "wind": (
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
        wind_speed_name="wind",
        temperature_name="t2m",
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
        name_10m_wind_speed="wind",
        name_2m_temperature="t2m",
        use_dask=False,
        output=None,
    )

    benchmarks = main(args)

    # Check benchmark results
    assert "10m_wind_speed" in benchmarks.data_vars
    assert "2m_temperature" in benchmarks.data_vars
    assert "lead_time" in benchmarks.dims
    assert "station_id" in benchmarks.dims


def test_rmse_calculation_matches_manual(sample_forecast, sample_stations):
    """Test that the RMSE calculation matches a manual calculation for a simple case."""
    # Prepare forecast with known values
    forecast = prepare_forecast(
        forecast=sample_forecast,
        region_name="europe",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 2),
        wind_speed_name="wind",
        temperature_name="t2m",
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
        benchmarks["10m_wind_speed"].values,
        expected_rmse,
        rtol=1e-6,
        err_msg="RMSE calculation does not match manual calculation",
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
