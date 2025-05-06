# %%
import xarray as xr

dataset = xr.open_dataset("/home/niall/stationbench/WeatherReal-ISD-2023.nc")

dataset = dataset[["t", "ws"]]

dataset = dataset.rename(
    {
        "t": "2m_temperature",
        "ws": "10m_wind_speed",
        "elev": "elevation",
        "lat": "latitude",
        "lon": "longitude",
        "station": "station_id",
        "time": "time",
    }
)

dataset["2m_temperature"] = dataset["2m_temperature"] + 273.15

dataset = dataset.transpose()

dataset.to_zarr(
    "/home/niall/stationbench/WeatherReal-ISD-2023.zarr",
    mode="w",
    consolidated=True,
    compute=True,
)
# %%
