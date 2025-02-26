import argparse
import logging
from datetime import date, datetime
from typing import Union

import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

from stationbench.utils.io import load_dataset
from stationbench.utils.logging import init_logging
from stationbench.utils.metrics import AVAILABLE_METRICS
from stationbench.utils.regions import region_dict, select_region_for_stations
from dask.distributed import get_client


logger = logging.getLogger(__name__)


def prepare_stations(
    stations: Union[str, xr.Dataset],
    region_name: str,
) -> xr.Dataset:
    """Filter ground truth stations by region."""
    logger.info("Preparing stations data")
    # Use load_dataset instead of direct loading
    stations = load_dataset(stations)
    variables = [
        "latitude",
        "longitude",
        "time",
        "station_id",
        "10m_wind_speed",
        "2m_temperature",
    ]
    stations = stations[variables]

    region = region_dict[region_name]

    # Adjust region slices if needed for longitude wrapping
    lat_slice = slice(region.lat_slice[0], region.lat_slice[1])
    lon_slice = slice(region.lon_slice[0], region.lon_slice[1])

    logger.info(
        "Selecting region: https://linestrings.com/bbox/#%s,%s,%s,%s",
        lon_slice.start,
        lat_slice.start,
        lon_slice.stop,
        lat_slice.stop,
    )

    original_stations = stations.sizes["station_id"]
    stations = select_region_for_stations(stations, lat_slice, lon_slice)
    remaining_stations = stations.sizes["station_id"]
    logger.info("Filtered stations: %s -> %s", original_stations, remaining_stations)

    return stations


def prepare_forecast(
    forecast: Union[str, xr.Dataset],
    region_name: str,
    start_date: date,
    end_date: date,
    wind_speed_name: str | None = None,
    temperature_name: str | None = None,
) -> xr.Dataset:
    """Prepare forecast data (region selection, time handling, variable renaming)."""
    logger.info("Preparing forecast dataset")

    # Validate dates
    if end_date <= start_date:
        raise ValueError(
            f"end_date ({end_date}) must be after start_date ({start_date})"
        )

    # Use load_dataset instead of direct loading
    forecast = load_dataset(
        forecast,
        chunks={
            "time": "auto",
            "prediction_timedelta": -1,
            "latitude": "auto",
            "longitude": "auto",
        },
    )

    is_point_based = "station_id" in forecast.dims

    # First handle time dimensions
    forecast = forecast.sel(time=slice(start_date, end_date))
    forecast = forecast.rename(
        {"time": "init_time", "prediction_timedelta": "lead_time"}
    )
    forecast.coords["valid_time"] = forecast.init_time + forecast.lead_time
    # Handle region selection
    region = region_dict[region_name]
    lat_slice = slice(region.lat_slice[0], region.lat_slice[1])
    lon_slice = slice(region.lon_slice[0], region.lon_slice[1])

    logger.info(
        "Selecting region: https://linestrings.com/bbox/#%s,%s,%s,%s",
        lon_slice.start,
        lat_slice.start,
        lon_slice.stop,
        lat_slice.stop,
    )
    # Longitude wrapping, if needed
    if forecast.longitude.max() > 180:
        logger.info("Converting longitudes from 0-360 to -180-180 range")
        forecast["longitude"] = xr.where(
            forecast.longitude > 180, forecast.longitude - 360, forecast.longitude
        )
    forecast = forecast.sortby("longitude")
    forecast = forecast.sortby("latitude")

    # Select region
    if is_point_based:
        forecast = select_region_for_stations(forecast, lat_slice, lon_slice)
    else:
        forecast = forecast.sel(latitude=lat_slice, longitude=lon_slice)

    # Rename variables
    if wind_speed_name:
        logger.info(
            "Renaming wind speed variable from %s to 10m_wind_speed", wind_speed_name
        )
        forecast = forecast.rename({wind_speed_name: "10m_wind_speed"})
    if temperature_name:
        logger.info(
            "Renaming temperature variable from %s to 2m_temperature", temperature_name
        )
        forecast = forecast.rename({temperature_name: "2m_temperature"})
    # Drop unwanted variables
    vars_to_drop = [
        var
        for var in forecast.data_vars
        if var not in ["10m_wind_speed", "2m_temperature"]
    ]
    forecast = forecast.drop_vars(vars_to_drop)

    return forecast


def intersect_stations(
    forecast: xr.Dataset,
    stations: xr.Dataset,
    by: str = "station_id",
    coord_tolerance: float | None = 0.01,
) -> xr.Dataset:
    """Match point-based forecast with station locations."""
    logger.info("Matching point-based forecast with stations")
    forecast_stations = forecast[by].values
    stations_ids = stations[by].values
    common_stations = np.intersect1d(forecast_stations, stations_ids)

    if len(common_stations) == 0:
        raise ValueError("No common stations found between forecast and stations")
    logger.info("Found %d common stations", len(common_stations))

    forecast = forecast.sel(**{by: common_stations})
    stations_subset = stations.sel(**{by: common_stations})

    # Validate coordinates match within tolerance
    if coord_tolerance is not None:
        lat_diff = np.abs(forecast.latitude.values - stations_subset.latitude.values)
        lon_diff = np.abs(forecast.longitude.values - stations_subset.longitude.values)
        if np.any(lat_diff > coord_tolerance) or np.any(lon_diff > coord_tolerance):
            raise ValueError(
                f"Coordinate mismatch between forecast and stations exceeds tolerance of {coord_tolerance} degrees"
            )

    return forecast


def interpolate_to_stations(forecast: xr.Dataset, stations: xr.Dataset) -> xr.Dataset:
    """Interpolate forecast to station locations."""
    logger.info("Interpolating forecast to station locations")
    forecast_interp = forecast.interp(
        latitude=stations.latitude,
        longitude=stations.longitude,
        method="linear",
    )
    return forecast_interp


def generate_benchmarks(
    *,
    forecast: xr.Dataset,
    stations: xr.Dataset,
) -> xr.Dataset:
    """Generate benchmarks by comparing forecast against ground truth.

    Computes the following metrics:
    - RMSE: Root Mean Square Error
    - MBE: Mean Bias Error

    Args:
        forecast: Forecast dataset
        stations: Ground truth dataset

    Returns:
        xr.Dataset with metrics for each variable
    """
    logger.info("Aligning stations with forecast valid times")
    stations = stations.sel(time=forecast.valid_time)

    logger.info("Calculating metrics")
    metrics_list = []

    # Calculate each metric
    for metric in AVAILABLE_METRICS.values():
        metrics_list.append(metric.compute(forecast, stations))
    # Merge all metrics into one dataset
    return xr.merge(metrics_list)


def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Calculate forecast benchmarks")
    parser.add_argument(
        "--forecast", type=str, required=True, help="Path to forecast zarr dataset"
    )
    parser.add_argument(
        "--stations",
        type=str,
        default="https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr",
        help="Path to ground truth zarr dataset",
    )
    parser.add_argument(
        "--start_date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="global",
        help="Region to benchmark (default: global)",
    )
    parser.add_argument(
        "--name_10m_wind_speed",
        type=str,
        default=None,
        help="Name of 10m wind speed variable",
    )
    parser.add_argument(
        "--name_2m_temperature",
        type=str,
        default=None,
        help="Name of 2m temperature variable",
    )
    parser.add_argument(
        "--use_dask",
        action="store_true",
        default=False,
        help="Use Dask for parallel computation",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of Dask workers to use (default: 4, only used if --use_dask is set and no client exists)",
    )
    return parser


def main(args=None) -> xr.Dataset:
    """Main function that can be called programmatically or via CLI.

    Args:
        args: Either an argparse.Namespace object or a list of command line arguments.
            If None, arguments will be parsed from sys.argv.

    Returns:
        xr.Dataset: The computed benchmarks dataset
    """
    init_logging()

    # Only create and parse arguments if we don't have a Namespace
    if not isinstance(args, argparse.Namespace):
        parser = get_parser()
        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(args)

    if args.use_dask:
        # Check if a client already exists
        try:
            client = get_client()
            logger.info("Using existing Dask client: %s", client.dashboard_link)
        except (ImportError, ValueError):
            # No client exists, create a new one
            cluster = LocalCluster(n_workers=args.n_workers)
            client = Client(cluster)
            logger.info(
                "Created new Dask client with %d workers: %s",
                args.n_workers,
                client.dashboard_link,
            )

    # Process stations
    stations = prepare_stations(args.stations, args.region)

    # Process forecast
    forecast = prepare_forecast(
        args.forecast,
        args.region,
        args.start_date,
        args.end_date,
        args.name_10m_wind_speed,
        args.name_2m_temperature,
    )

    # Either match stations or interpolate based on forecast type
    is_point_based = "station_id" in forecast.dims
    if is_point_based:
        forecast = intersect_stations(forecast, stations)
    else:
        forecast = interpolate_to_stations(forecast, stations)

    # Calculate benchmarks
    benchmarks_ds = generate_benchmarks(
        forecast=forecast,
        stations=stations,
    )

    # Save if output location specified
    if args.output:
        for var in benchmarks_ds.variables:
            benchmarks_ds[var].encoding.clear()
        logger.info("Writing benchmarks to %s", args.output)
        logger.info("Dataset size: %s MB", benchmarks_ds.nbytes / 1e6)

        # Explicitly rechunk all data variables and coordinates
        chunks = {}
        for dim in benchmarks_ds.dims:
            chunks[dim] = -1  # -1 means one chunk for the whole dimension
        benchmarks_ds = benchmarks_ds.chunk(chunks)
        benchmarks_ds.to_zarr(args.output, mode="w")
        logger.info("Finished writing benchmarks to %s", args.output)

    return benchmarks_ds
