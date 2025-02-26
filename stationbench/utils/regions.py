from dataclasses import dataclass
import xarray as xr
import logging

logger = logging.getLogger(__name__)


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
    "asia": Region(
        name="asia",
        lat_slice=(10, 55),
        lon_slice=(60, 150),
    ),
    "australia": Region(
        name="australia",
        lat_slice=(-45, -10),
        lon_slice=(110, 155),
    ),
    "germany": Region(
        name="germany",
        lat_slice=(47, 55),
        lon_slice=(8, 15),
    ),
    "united-kingdom": Region(
        name="united-kingdom",
        lat_slice=(50, 58),
        lon_slice=(-4, 2),
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


def add_region(
    name: str, lat_slice: tuple[float, float], lon_slice: tuple[float, float]
) -> None:
    """Add a new region to the region_dict.

    Args:
        name: Name of the region
        lat_slice: Tuple of (min_latitude, max_latitude)
        lon_slice: Tuple of (min_longitude, max_longitude)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not isinstance(name, str):
        error_msg = f"Region name must be a string, got {type(name)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(lat_slice, tuple) or len(lat_slice) != 2:
        error_msg = f"lat_slice must be a tuple of (min_lat, max_lat), got {lat_slice}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(lon_slice, tuple) or len(lon_slice) != 2:
        error_msg = f"lon_slice must be a tuple of (min_lon, max_lon), got {lon_slice}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate latitude range (-90 to 90)
    min_lat, max_lat = lat_slice
    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90) or min_lat > max_lat:
        error_msg = f"Invalid latitude range: {lat_slice}. Must be within [-90, 90] and min_lat <= max_lat"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate longitude range (-180 to 180)
    min_lon, max_lon = lon_slice
    if not (-180 <= min_lon < 180) or not (-180 <= max_lon < 180) or min_lon > max_lon:
        error_msg = f"Invalid longitude range: {lon_slice}. Must be within [-180, 180) and min_lon <= max_lon"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check if region already exists
    if name in region_dict:
        logger.warning(f"Region '{name}' already exists and will be overwritten")

    # Add the region
    region_dict[name] = Region(name=name, lat_slice=lat_slice, lon_slice=lon_slice)
    logger.info(
        f"Added region '{name}' with lat_slice={lat_slice}, lon_slice={lon_slice}"
    )
