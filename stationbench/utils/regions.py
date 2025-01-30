from dataclasses import dataclass
import xarray as xr


@dataclass
class Region:
    name: str
    lat_slice: tuple[float, float]
    lon_slice: tuple[float, float]


region_dict = {
    "global": Region(
        name="global",
        lat_slice=(-90, 90),
        lon_slice=(-180, 180),
    ),
    "europe": Region(
        name="europe",
        lat_slice=(36, 72),
        lon_slice=(-15, 45),
    ),
    "north-america": Region(
        name="north-america",
        lat_slice=(25, 60),
        lon_slice=(-125, -64),
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
