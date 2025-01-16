import pytest
import xarray as xr
import numpy as np
from stationbench.utils.regions import region_dict, select_region_for_stations, Region

def test_region_dict_contains_required_regions():
    required_regions = ["global", "europe", "north-america",]
    for region in required_regions:
        assert region in region_dict

def test_region_boundaries():
    # Test that regions have valid lat/lon boundaries
    for name, region in region_dict.items():
        assert isinstance(region, Region)
        assert -90 <= region.lat_slice[0] <= 90
        assert -90 <= region.lat_slice[1] <= 90
        assert -180 <= region.lon_slice[0] <= 180
        assert -180 <= region.lon_slice[1] <= 180
        assert region.lat_slice[0] < region.lat_slice[1]

def test_select_region_for_stations():
    # Create test dataset
    ds = xr.Dataset(
        coords={
            "station_id": ["s1", "s2", "s3"],
            "latitude": ("station_id", [45, 50, 35]),
            "longitude": ("station_id", [10, 15, 20]),
        }
    )
    
    # Test European region selection
    region = region_dict["europe"]
    filtered_ds = select_region_for_stations(
        ds, 
        slice(region.lat_slice[0], region.lat_slice[1]),
        slice(region.lon_slice[0], region.lon_slice[1])
    )
    
    assert len(filtered_ds.station_id) == 2  # Only stations in Europe 