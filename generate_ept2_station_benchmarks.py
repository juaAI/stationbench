# %%

import xarray as xr
import pandas as pd
import stationbench
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

REGIONS = ["global"]
FORECASTS = [
    # (
    #     "ept2_v2",
    #     "/mnt/jua-shared-1/jua-hindcasts/EPT2-global-from-2023-01-01-to-2024-12-28-v2.zarr",
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
    (
        "aurora",
        [
            "/mnt/jua-shared-1/jua-hindcasts/aurora-global-from-2023-01-02-to-2023-07-05.zarr",
            "/mnt/jua-shared-1/jua-hindcasts/aurora-global-from-2023-07-06-to-2024-12-28.zarr",
        ],
        ("wind_speed_10m", "air_temperature_2m"),
    ),
    # (
    #     "ec_hres_forecast",
    #     "gs://jua-benchmarking/forecasts/third_party/ifs-fc-2018-2023-0012-6hr-1440x721-mslp-z500.zarr",
    #     ("10m_wind_speed", "2m_temperature")
    # ),
    # ("ec_ens_mean", "gs://jua-data-sandbox/ifs/ensemble/ifs_em_2018-2023_1440x721.zarr", ("10m_wind_speed", "2m_temperature"))
]

stations = xr.open_zarr("https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr")

# %%
for forecast_name, forecast_path, (
    name_10m_wind_speed,
    name_2m_temperature,
) in FORECASTS:
    for region in REGIONS:
        logging.info(
            f"Starting station benchmark for region {region} and forecast {forecast_name}"
        )

        if isinstance(forecast_path, str):
            forecast = xr.open_zarr(forecast_path)
        elif isinstance(forecast_path, list):
            forecast = xr.open_mfdataset(forecast_path, consolidated=False)
        else:
            raise RuntimeError("Error reading forecast paths")

        forecast = forecast[[name_10m_wind_speed, name_2m_temperature]]

        start_date = "2023-01-01"
        end_date = "2023-12-31"

        # Define your custom range
        start = pd.Timestamp(start_date)  # could be before available data
        end = pd.Timestamp(end_date)  # could be after available data

        # Get the dataset's available time range
        available_times = forecast.time.to_index()
        data_start = available_times[0]
        data_end = available_times[-1]

        # Clamp your range to the data range
        clamped_start = max(start, data_start)
        clamped_end = min(end, data_end)

        # Generate 4-day intervals at 00:00, 06:00, 12:00, and 18:00
        dates_00 = pd.date_range(start=clamped_start, end=clamped_end, freq="4D")
        dates_06 = pd.date_range(
            start=clamped_start + pd.Timedelta(hours=6), end=clamped_end, freq="4D"
        )
        dates_12 = pd.date_range(
            start=clamped_start + pd.Timedelta(hours=12), end=clamped_end, freq="4D"
        )
        dates_18 = pd.date_range(
            start=clamped_start + pd.Timedelta(hours=18), end=clamped_end, freq="4D"
        )

        # Combine and sort
        target_times = (
            dates_00.append(dates_06).append(dates_12).append(dates_18).sort_values()
        )

        # Keep only times that exist in the dataset
        target_times = target_times.intersection(available_times)

        # Select from dataset
        forecast = forecast.sel(time=target_times)

        output_forecast = f"./data/{forecast_name}_{region}.zarr"

        stationbench.calculate_metrics(
            forecast=forecast,
            stations=stations,
            start_date=start_date,
            end_date=end_date,
            output=output_forecast,
            region=region,
            name_10m_wind_speed=name_10m_wind_speed,
            name_2m_temperature=name_2m_temperature,
            use_dask=False,
            # n_workers=4,
        )

# %%
