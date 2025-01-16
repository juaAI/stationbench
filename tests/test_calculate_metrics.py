import pytest
import xarray as xr
from datetime import date
from stationbench.calculate_metrics import (
    DataType,
    preprocess_data,
    generate_benchmarks,
)
import numpy as np


def test_preprocess_data_forecast(sample_forecast_dataset, tmp_path):
    # Save sample dataset to a temporary zarr store
    forecast_path = tmp_path / "forecast.zarr"
    sample_forecast_dataset.to_zarr(forecast_path)

    processed_ds = preprocess_data(
        dataset_loc=str(forecast_path),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        region_name="europe",  # Assuming this region exists in region_dict
        wind_speed_name="10m_wind_speed",
        temperature_name=None,
        data_type=DataType.FORECAST,
    )

    assert isinstance(processed_ds, xr.Dataset)
    assert "valid_time" in processed_ds.coords
    assert "10m_wind_speed" in processed_ds.data_vars


def test_preprocess_data_ground_truth(sample_ground_truth_dataset, tmp_path):
    # Save sample dataset to a temporary zarr store
    gt_path = tmp_path / "ground_truth.zarr"
    sample_ground_truth_dataset.to_zarr(gt_path)

    processed_ds = preprocess_data(
        dataset_loc=str(gt_path),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        region_name="europe",  # Assuming this region exists in region_dict
        wind_speed_name="10m_wind_speed",
        temperature_name=None,
        data_type=DataType.GROUND_TRUTH,
    )

    assert isinstance(processed_ds, xr.Dataset)
    assert "10m_wind_speed" in processed_ds.data_vars
    assert "station_id" in processed_ds.dims


def test_generate_benchmarks(sample_forecast_dataset, sample_ground_truth_dataset):
    # Prepare datasets for benchmark generation
    forecast = sample_forecast_dataset.copy()
    forecast = forecast.rename({"time": "init_time"})
    forecast = forecast.rename({"prediction_timedelta": "lead_time"})
    forecast.coords["valid_time"] = forecast.init_time + forecast.lead_time

    ground_truth = sample_ground_truth_dataset.copy()

    benchmarks = generate_benchmarks(
        forecast=forecast,
        ground_truth=ground_truth,
    )

    assert isinstance(benchmarks, xr.Dataset)
    # Check for both metrics
    assert "metric" in benchmarks.dims
    assert "rmse" in benchmarks.metric
    assert "mbe" in benchmarks.metric
    
    # Check dataset dimensions
    assert set(benchmarks.dims) == {"lead_time", "station_id", "metric"}
    
    # Check data variable dimensions
    assert "10m_wind_speed" in benchmarks.data_vars
    print(f'Variable dimensions: {benchmarks["10m_wind_speed"].dims}')
    assert set(benchmarks["10m_wind_speed"].dims) == {"lead_time", "station_id", "metric"}


@pytest.mark.parametrize("data_type", [DataType.FORECAST, DataType.GROUND_TRUTH])
def test_preprocess_data_invalid_path(data_type):
    with pytest.raises(Exception):  # Should raise some kind of file not found error
        preprocess_data(
            dataset_loc="invalid/path.zarr",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            region_name="europe",
            wind_speed_name="10m_wind_speed",
            temperature_name=None,
            data_type=data_type,
        )


def test_rmse_calculation_matches_manual(
    sample_forecast_dataset, sample_ground_truth_dataset
):
    """Test that the RMSE calculation matches a manual calculation for a simple case."""
    # Prepare a simplified forecast dataset with known values
    forecast = sample_forecast_dataset.copy()
    forecast = forecast.rename({"time": "init_time"})
    forecast = forecast.rename({"prediction_timedelta": "lead_time"})
    forecast.coords["valid_time"] = forecast.init_time + forecast.lead_time

    # Set known values for forecast
    forecast["10m_wind_speed"][:] = 5.0  # Set all forecast values to 5.0

    # Prepare ground truth with known values
    ground_truth = sample_ground_truth_dataset.copy()
    ground_truth["10m_wind_speed"][:] = 3.0  # Set all ground truth values to 3.0

    # Calculate metrics
    metrics = generate_benchmarks(
        forecast=forecast,
        ground_truth=ground_truth,
    )

    # Manual RMSE calculation: sqrt(mean((5.0 - 3.0)²)) = sqrt(4) = 2.0
    expected_rmse = 2.0

    # Check if the calculated RMSE matches the expected value
    np.testing.assert_allclose(
        metrics.sel(metric="rmse")["10m_wind_speed"].values,
        expected_rmse,
        rtol=1e-6,
        err_msg="RMSE calculation does not match manual calculation",
    )


def test_mbe_calculation_matches_manual(
    sample_forecast_dataset, sample_ground_truth_dataset
):
    """Test that the MBE calculation matches a manual calculation for a simple case."""
    # Prepare datasets
    forecast = sample_forecast_dataset.copy()
    forecast = forecast.rename({"time": "init_time"})
    forecast = forecast.rename({"prediction_timedelta": "lead_time"})
    forecast.coords["valid_time"] = forecast.init_time + forecast.lead_time

    # Set known values
    forecast["10m_wind_speed"][:] = 5.0
    ground_truth = sample_ground_truth_dataset.copy()
    ground_truth["10m_wind_speed"][:] = 3.0

    # Calculate metrics
    metrics = generate_benchmarks(
        forecast=forecast,
        ground_truth=ground_truth,
    )

    # Manual MBE calculation: mean(forecast - ground_truth) = 5.0 - 3.0 = 2.0
    expected_mbe = 2.0

    np.testing.assert_allclose(
        metrics.sel(metric="mbe")["10m_wind_speed"].values,
        expected_mbe,
        rtol=1e-6,
        err_msg="MBE calculation does not match manual calculation",
    )
