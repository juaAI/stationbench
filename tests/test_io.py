import xarray as xr
import numpy as np
import pandas as pd
from stationbench.utils.io import load_dataset


def test_load_dataset_with_dataarray():
    """Test that load_dataset correctly handles DataArray inputs."""
    # Create a sample DataArray
    times = pd.date_range("2023-01-01", "2023-01-03", freq="D")
    lats = np.linspace(45, 50, 3)
    lons = np.linspace(-5, 5, 4)

    # Create DataArray with a name
    data_with_name = xr.DataArray(
        np.random.rand(len(times), len(lats), len(lons)),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        name="temperature",
    )

    # Create DataArray without a name
    data_without_name = xr.DataArray(
        np.random.rand(len(times), len(lats), len(lons)),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )

    # Test with named DataArray
    result_with_name = load_dataset(data_with_name)
    assert isinstance(result_with_name, xr.Dataset)
    assert "temperature" in result_with_name.data_vars
    assert result_with_name["temperature"].equals(data_with_name)

    # Test with unnamed DataArray
    result_without_name = load_dataset(data_without_name)
    assert isinstance(result_without_name, xr.Dataset)
    assert "data" in result_without_name.data_vars
    assert result_without_name["data"].equals(data_without_name)


def test_load_dataset_with_dataset():
    """Test that load_dataset correctly handles Dataset inputs."""
    # Create a sample Dataset
    times = pd.date_range("2023-01-01", "2023-01-03", freq="D")
    lats = np.linspace(45, 50, 3)
    lons = np.linspace(-5, 5, 4)

    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["time", "latitude", "longitude"],
                np.random.rand(len(times), len(lats), len(lons)),
            ),
            "wind_speed": (
                ["time", "latitude", "longitude"],
                np.random.rand(len(times), len(lats), len(lons)),
            ),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )

    # Test with Dataset
    result = load_dataset(ds)
    assert isinstance(result, xr.Dataset)
    assert result.equals(ds)

    # Ensure the original dataset is returned, not a copy
    assert result is ds
