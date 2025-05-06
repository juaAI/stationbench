# %%
import logging

import numpy as np
import xarray as xr

from stationbench.utils.logging import init_logging
from stationbench.calculate_metrics import (
    prepare_stations,
    prepare_forecast,
    intersect_stations,
    interpolate_to_stations,
)

logger = logging.getLogger(__name__)


def preprocess_datasets(datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    # Find common time
    common_time = None
    for ds in datasets.values():
        if common_time is None:
            common_time = ds["init_time"]
        else:
            common_time = np.intersect1d(common_time, ds["init_time"])

    logging.info(f"Number of common timesteps: {len(common_time)}")

    # Slice and interpolate datasets
    processed = {}
    reference_grid = datasets["aifs"]

    for name, ds in datasets.items():
        # Select common times
        ds = ds.sel(init_time=common_time).sel(
            lead_time=datasets["ept15"].lead_time
        )

        # Convert longitude if needed (for ept15)
        if name == "ept15":
            ds = ds.assign_coords(
                longitude=(((ds.longitude + 180) % 360) - 180)
            ).sortby("longitude")

        # Interpolate to reference grid
        ds = ds.interp(
            latitude=reference_grid.latitude,
            longitude=reference_grid.longitude,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        processed[name] = ds

    # Validate processed datasets have matching coordinates
    ds_names = list(processed.keys())
    for i in range(len(ds_names) - 1):
        current = ds_names[i]
        next_ds = ds_names[i + 1]

        # Check latitude coordinates
        assert np.allclose(
            processed[current].latitude, processed[next_ds].latitude
        ), f"Latitude mismatch between {current} and {next_ds}"

        # Check longitude coordinates
        assert np.allclose(
            processed[current].longitude, processed[next_ds].longitude
        ), f"Longitude mismatch between {current} and {next_ds}"

        # Check time coordinates
        assert np.array_equal(
            processed[current]["init_time"], processed[next_ds]["init_time"]
        ), f"Time mismatch between {current} and {next_ds}"

        # Check prediction_timedelta coordinates
        assert np.array_equal(
            processed[current].lead_time,
            processed[next_ds].lead_time,
        ), f"Prediction timedelta mismatch between {current} and {next_ds}"

    return processed


def linear_regression(
    forecasts: list[xr.Dataset],
    stations: xr.Dataset,
):
    """Perform linear regression on the forecast data against the station data.

    Args:
        forecasts: List of forecast datasets.
        stations: Station dataset.

    Returns:
        xr.Dataset: Dataset containing the results of the linear regression.
    """
    weights = np.array([1.0] * len(forecasts))


# Format: name: str, hindcast_path: str | list[str], var_names: tuple[str, str]
FORECASTS = [
    # (
    #     "ept2_v2",
    #     "/home/niall/mnt/ept-2/v2/global/2023-01-01-to-2024-12-28.zarr/",
    #     ("wind_speed_10m", "air_temperature_2m"),
    # ),
    # (
    #     "ept2_v1",
    #     [
    #         "/mnt/jua-shared-1/jua-hindcasts/EPT2-global-from-2023-01-01-to-2023-09-17.zarr",
    #         "/mnt/jua-shared-1/jua-hindcasts/EPT2-global-from-2023-09-18-to-2023-09-25.zarr",
    #         "/mnt/jua-shared-1/jua-hindcasts/EPT2-global-from-2023-09-26-to-2024-12-28.zarr",
    #     ],
    #     ("wind_speed_10m", "air_temperature_2m"),
    # ),
    # (
    #     "ept2_15_mixed",
    #     "/mnt/jua-shared-1/jua-hindcasts/test_ens.zarr",
    #     ("wind_speed_10m", "air_temperature_2m"),
    # ),
    (
        "ept15",
        "/mnt/jua-mount-2/jua-hindcasts/2024-12-05-sharing-sloth-1_20250426-101631_epoch16_max_lead_time12.zarr",
        ("wind_speed_10m", "air_temperature_2m"),
    ),
    # (
    #     "ec_hres_forecast",
    #     "gs://jua-benchmarking/forecasts/third_party/ifs-fc-2018-2023-0012-6hr-1440x721-mslp-z500.zarr",
    #     ("10m_wind_speed", "2m_temperature"),
    # ),
    # (
    #     "aurora",
    #     [
    #         "/home/niall/mnt/aurora-global-from-2023-01-02-to-2023-07-05.zarr",
    #         "/home/niall/mnt/aurora-global-from-2023-07-06-to-2024-12-28.zarr",
    #     ],
    #     ("wind_speed_10m", "air_temperature_2m"),
    # ),
    (
        "aifs",
        "/mnt/jua-mount-2/jua-hindcasts/AIFS-global-from-2023-01-02-to-2024-12-27.zarr",
        ("wind_speed_at_height_level_10m", "air_temperature_at_height_level_2m"),
    ),
    # (
    #     "ec_ens_mean",
    #     "gs://jua-data-sandbox/ifs/ensemble/ifs_em_2018-2023_1440x721.zarr",
    #     ("10m_wind_speed", "2m_temperature"),
    # ),
]

stations = xr.open_zarr("/home/niall/stationbench/WeatherReal-ISD-2023.zarr")

# %%
# Process stations
stations = prepare_stations(stations, region_name="global")

start_date = "2023-01-02"
end_date = "2023-02-10"

# Process forecast
forecasts = {}

for forecast_name, forecast_path, (
    name_10m_wind_speed,
    name_2m_temperature,
) in FORECASTS:
    forecast = prepare_forecast(
        forecast_path,
        "global",
        start_date,
        end_date,
        name_10m_wind_speed,
        name_2m_temperature,
    )
    forecasts[forecast_name] = forecast

forecasts = preprocess_datasets(forecasts)

# %%
# Either match stations or interpolate based on forecast type
for key in forecasts.keys():
    is_point_based = "station_id" in forecasts[key].dims
    if is_point_based:
        forecasts[key] = intersect_stations(forecasts[key], stations)
    else:
        forecasts[key] = interpolate_to_stations(forecasts[key], stations)

# %%
