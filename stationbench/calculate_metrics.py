import argparse
import logging
from datetime import date
from enum import Enum, auto

import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster

from stationbench.utils.regions import region_dict, select_region_for_stations
from stationbench.utils.logging import init_logging
from stationbench.utils.metrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)


class DataType(Enum):
    FORECAST = auto()
    GROUND_TRUTH = auto()


def preprocess_data(
    dataset_loc: str,
    start_date: date,
    end_date: date,
    region_name: str,
    wind_speed_name: str | None,
    temperature_name: str | None,
    data_type: DataType,
) -> xr.Dataset:
    logger.info("preprocessing dataset %s", dataset_loc)

    region = region_dict[region_name]

    chunks: dict[str, str | int] = {"latitude": "auto", "longitude": "auto"}

    if data_type == DataType.FORECAST:
        chunks["time"] = "auto"
        chunks["prediction_timedelta"] = -1
        ds = xr.open_zarr(dataset_loc, chunks=chunks)

        ds = ds.sel(time=slice(start_date, end_date))
        # align forecast time with lead times so that it's directly comparable against gt
        ds = ds.rename({"time": "init_time"})
        ds = ds.rename({"prediction_timedelta": "lead_time"})

        logger.info("creating valid time...")
        ds.coords["valid_time"] = ds.init_time + ds.lead_time

        # renaming variables to match ground truth
        if wind_speed_name:
            ds = ds.rename({wind_speed_name: "10m_wind_speed"})
        if temperature_name:
            ds = ds.rename({temperature_name: "2m_temperature"})
    elif data_type == DataType.GROUND_TRUTH:
        chunks["time"] = "auto"
        chunks["station_id"] = -1
        ds = xr.open_zarr(dataset_loc, chunks=chunks)

        # remove unused variables
        variables_to_keep = ["latitude", "longitude", "time", "station_id"]
        if wind_speed_name:
            variables_to_keep.append("10m_wind_speed")
        if temperature_name:
            variables_to_keep.append("2m_temperature")
        ds = ds[variables_to_keep]

    # shift to -180 to 180 if there are longitudes greater than 180
    if ds.longitude.max() > 180:
        logger.info("shifting longitude to -180 to 180...")
        ds["longitude"] = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.sortby("longitude")

    ds = ds.sortby("latitude")

    # select region
    lat_slice = slice(region.lat_slice[0], region.lat_slice[1])
    lon_slice = slice(region.lon_slice[0], region.lon_slice[1])

    logger.info(
        "Selecting region: https://linestrings.com/bbox/#%s,%s,%s,%s",
        lon_slice.start,
        lat_slice.start,
        lon_slice.stop,
        lat_slice.stop,
    )
    if data_type == DataType.FORECAST:
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
    elif data_type == DataType.GROUND_TRUTH:
        # drop all station_ids outside of the region
        original_stations = ds.sizes["station_id"]
        ds = select_region_for_stations(ds, lat_slice, lon_slice)
        remaining_stations = ds.sizes["station_id"]
        logger.info(
            "Filtered ground truth stations: %s -> %s",
            original_stations,
            remaining_stations,
        )
    logger.info("Finished processing of %s: %s", dataset_loc, ds)
    return ds


def generate_benchmarks(
    *,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
) -> xr.Dataset:
    """Generate benchmarks by comparing forecast against ground truth.

    Computes the following metrics:
    - RMSE: Root Mean Square Error
    - MBE: Mean Bias Error

    Args:
        forecast: Forecast dataset
        ground_truth: Ground truth dataset

    Returns:
        xr.Dataset with metrics for each variable
    """
    logger.info("Aligning GT with valid time")
    ground_truth = ground_truth.sel(time=forecast.valid_time)

    # align the fc to the gt points (this will take a few mins):
    logger.info("Interpolating forecast to ground truth points")
    fc_like_gt = forecast.interp(
        latitude=ground_truth.latitude,
        longitude=ground_truth.longitude,
        method="linear",
    )

    logger.info("Calculating metrics")
    metrics_list = []

    # Calculate each metric
    for metric in AVAILABLE_METRICS.values():
        metrics_list.append(metric.compute(fc_like_gt, ground_truth))

    # Merge all metrics into one dataset
    return xr.merge(metrics_list)


def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Compute benchmarks")
    parser.add_argument(
        "--forecast_loc", type=str, required=True, help="Location of input forecast"
    )
    parser.add_argument(
        "--ground_truth_loc",
        type=str,
        default="https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr",
        help="Location of ground truth data",
    )
    parser.add_argument(
        "--start_date",
        type=pd.Timestamp,
        required=True,
        help="Start date for benchmarking (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=pd.Timestamp,
        required=True,
        help="End date for benchmarking (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output_loc", type=str, required=True, help="Output path for benchmarks"
    )
    parser.add_argument("--region", type=str, required=True, help="Region to benchmark")
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
    return parser


def main(args=None):
    """Main function that can be called programmatically or via CLI.

    Args:
        args: Either an argparse.Namespace object or a list of command line arguments.
            If None, arguments will be parsed from sys.argv.
    """
    init_logging()

    if not isinstance(args, argparse.Namespace):
        parser = get_parser()
        args = parser.parse_args(args)

    cluster = LocalCluster(n_workers=22)
    client = Client(cluster)
    logging.info("Dask dashboard %s", client.dashboard_link)

    forecast = preprocess_data(
        dataset_loc=args.forecast_loc,
        start_date=args.start_date,
        end_date=args.end_date,
        region_name=args.region,
        wind_speed_name=args.name_10m_wind_speed,
        temperature_name=args.name_2m_temperature,
        data_type=DataType.FORECAST,
    )

    ground_truth = preprocess_data(
        dataset_loc=args.ground_truth_loc,
        start_date=args.start_date,
        end_date=args.end_date,
        region_name=args.region,
        wind_speed_name=args.name_10m_wind_speed,
        temperature_name=args.name_2m_temperature,
        data_type=DataType.GROUND_TRUTH,
    )

    benchmarks_ds = generate_benchmarks(
        forecast=forecast,
        ground_truth=ground_truth,
    )

    # Clear potential encoding
    for var in benchmarks_ds.variables:
        benchmarks_ds[var].encoding.clear()
    logger.info("Writing benchmarks to %s", args.output_loc)
    logger.info("Dataset size: %s MB", benchmarks_ds.nbytes / 1e6)
    logger.info(benchmarks_ds)
    benchmarks_ds.to_zarr(args.output_loc, mode="w")
    logger.info("Finished writing benchmarks to %s", args.output_loc)
    return benchmarks_ds
