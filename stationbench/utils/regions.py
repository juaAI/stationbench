from dataclasses import dataclass
import xarray as xr


@dataclass
class Region:
    lat_slice: tuple[float, float]
    lon_slice: tuple[float, float]


region_dict = {
    "global": Region(
        lat_slice=(-90, 90),
        lon_slice=(-180, 180),
    ),
    "europe": Region(
        lat_slice=(36, 72),
        lon_slice=(-15, 45),
    ),
    "north-america": Region(
        lat_slice=(25, 60),
        lon_slice=(-125, -64),
    ),
    "italy": Region(
        lat_slice=(36.6, 47.1),
        lon_slice=(6.6, 18.6),
    ),
}


def get_lat_slice(region: Region) -> slice:
    return slice(region.lat_slice[0], region.lat_slice[1])


def get_lon_slice(region: Region) -> slice:
    return slice(region.lon_slice[0], region.lon_slice[1])


def select_region_for_stations(
    ds: xr.Dataset, lat_slice: slice, lon_slice: slice
) -> xr.Dataset:
    # drop all station_ids outside of the region
    mask = (
        (ds.latitude >= lat_slice.start)
        & (ds.latitude <= lat_slice.stop)
        & (ds.longitude >= lon_slice.start)
        & (ds.longitude <= lon_slice.stop)
    ).compute()
    ds = ds.isel(station_id=mask)
    return ds
