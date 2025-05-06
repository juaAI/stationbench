# %%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def find_nearby_stations(ds1, ds2, distance_threshold_km=1.0):
    # Load coordinates into memory (small data)
    coords1 = np.stack([ds1.latitude.values, ds1.longitude.values], axis=1)
    coords2 = np.stack([ds2.latitude.values, ds2.longitude.values], axis=1)

    # Build KDTree for efficient spatial lookup
    tree = cKDTree(coords2)

    # Query all stations in ds1 against ds2
    distances, indices = tree.query(
        coords1, distance_upper_bound=distance_threshold_km / 111
    )  # rough conversion

    matches = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx < len(ds2.station_id):  # filter out "no match"
            # Compute accurate haversine distance
            lat1, lon1 = coords1[i]
            lat2, lon2 = coords2[idx]
            accurate_distance = haversine_distance(lat1, lon1, lat2, lon2)
            if accurate_distance <= distance_threshold_km:
                matches.append(
                    {
                        "station_id_ds1": ds1.station_id.values[i],
                        "station_id_ds2": ds2.station_id.values[idx],
                        "distance_km": accurate_distance,
                    }
                )

    return pd.DataFrame(matches)


# Compare dataset_ours and dataset_isd based on proximity
dataset_ours = xr.open_zarr(
    "https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr"
)
dataset_isd = xr.open_zarr("/home/niall/stationbench/WeatherReal-ISD-2023.zarr")

matches_df = find_nearby_stations(dataset_ours, dataset_isd, distance_threshold_km=0.1)
# print(matches_df)


# %%
# %%


def find_stations_in_ds2_not_ds1(ds1, ds2, distance_threshold_km=1.0):
    # Load coordinates into memory (small data)
    coords1 = np.stack([ds1.latitude.values, ds1.longitude.values], axis=1)
    coords2 = np.stack([ds2.latitude.values, ds2.longitude.values], axis=1)

    # Build KDTree for efficient spatial lookup
    tree = cKDTree(coords1)

    # Query all stations in ds2 against ds1
    distances, indices = tree.query(
        coords2, distance_upper_bound=distance_threshold_km / 111
    )  # rough conversion

    unmatched_stations = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx >= len(ds1.station_id):  # No match in ds1 (upper bound exceeded)
            # Compute accurate haversine distance
            lat2, lon2 = coords2[i]
            # Adding this station to the unmatched list since it has no corresponding match
            unmatched_stations.append(
                {
                    "station_id_ds2": ds2.station_id.values[i],
                    "latitude": lat2,
                    "longitude": lon2,
                    "distance_km": dist * 111,  # approximate km conversion
                }
            )

    return pd.DataFrame(unmatched_stations)


matches_in_ds2_not_ds1 = find_stations_in_ds2_not_ds1(
    dataset_ours, dataset_isd, distance_threshold_km=0.1
)

# %%
import plotly.graph_objects as go


def plot_station_matches_interactive(ds1, ds2, matches_df):
    # Get matched station coordinates
    ds1_coords = ds1[["latitude", "longitude"]].sel(
        station_id=xr.DataArray(matches_df["station_id_ds1"].values, dims="station_id")
    )
    ds2_coords = ds2[["latitude", "longitude"]].sel(
        station_id=xr.DataArray(matches_df["station_id_ds2"].values, dims="station_id")
    )

    lats1 = ds1_coords.latitude.values
    lons1 = ds1_coords.longitude.values
    lats2 = ds2_coords.latitude.values
    lons2 = ds2_coords.longitude.values

    fig = go.Figure()

    # Add station markers
    fig.add_trace(
        go.Scattergeo(
            lon=lons1,
            lat=lats1,
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="DS1 Matched Stations",
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=lons2,
            lat=lats2,
            mode="markers",
            marker=dict(size=5, color="green"),
            name="DS2 Matched Stations",
        )
    )

    # Add lines connecting matched pairs
    for lon1, lat1, lon2, lat2 in zip(lons1, lats1, lons2, lats2):
        fig.add_trace(
            go.Scattergeo(
                lon=[lon1, lon2],
                lat=[lat1, lat2],
                mode="lines",
                line=dict(width=1, color="red"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Nearby Station Matches Between Datasets",
        geo=dict(
            scope="world",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(230, 230, 230)",
            showocean=True,
            oceancolor="rgb(200, 220, 255)",
        ),
        height=800,
    )

    fig.show()
    fig.write_html("station_matches_map.html")


# # Count how many times each station appears
# ds1_counts = matches_df["station_id_ds1"].value_counts()
# ds2_counts = matches_df["station_id_ds2"].value_counts()

# # Find station_ids that appear more than once
# non_unique_ds1 = ds1_counts[ds1_counts > 1].index
# non_unique_ds2 = ds2_counts[ds2_counts > 1].index

# # Filter matches_df to keep rows where either station_id is non-unique
# non_unique_matches = matches_df[
#     matches_df["station_id_ds1"].isin(non_unique_ds1)
#     | matches_df["station_id_ds2"].isin(non_unique_ds2)
# ]


# %%
def plot_unmatched_stations_interactive(ds2, unmatched_df):
    # Get unmatched station coordinates from ds2
    ds2_coords = ds2[["latitude", "longitude"]].sel(
        station_id=xr.DataArray(
            unmatched_df["station_id_ds2"].values, dims="station_id"
        )
    )

    lats2 = ds2_coords.latitude.values
    lons2 = ds2_coords.longitude.values

    fig = go.Figure()

    # Add unmatched station markers from ds2
    fig.add_trace(
        go.Scattergeo(
            lon=lons2,
            lat=lats2,
            mode="markers",
            marker=dict(size=8, color="orange", symbol="circle"),
            name="Unmatched Stations (DS2)",
        )
    )

    fig.update_layout(
        title="Unmatched Stations in DS2 (No Corresponding Matches in DS1)",
        geo=dict(
            scope="world",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(230, 230, 230)",
            showocean=True,
            oceancolor="rgb(200, 220, 255)",
        ),
        height=800,
    )

    fig.show()
    fig.write_html("unmatched_stations_map.html")


# Example usage
# Assuming you have a DataFrame `unmatched_df` with unmatched stations:
# unmatched_df = find_stations_in_ds2_not_ds1(ds1, ds2)

plot_unmatched_stations_interactive(dataset_isd, matches_in_ds2_not_ds1)

# %%
from meteostat import Hourly
from datetime import datetime


def fetch_meteostat_timeseries(station_id, variable, start_year=2023, end_year=2024):
    # Search for Meteostat station by ID
    # Fetch hourly data
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 1, 1)
    data = Hourly(station_id, start, end)
    data = data.fetch()

    if data.empty:
        print(f"No Meteostat hourly data for station {station_id}")
        return None

    # Convert variable names to match if needed
    var_map = {
        "2m_temperature": "temp",  # Meteostat uses temp for temperature (°C)
        "10m_wind_speed": "wspd",  # Meteostat uses wspd for wind speed (m/s)
    }

    if variable not in var_map:
        print(f"No Meteostat mapping for variable {variable}")
        return None

    data = data[[var_map[variable]]].rename(columns={var_map[variable]: variable})

    if variable == "2m_temperature":
        data["2m_temperature"] = data["2m_temperature"] + 273.15
    elif variable == "10m_wind_speed":
        data["10m_wind_speed"] = data["10m_wind_speed"] / 3.6

    return data


# %%
import requests

# NOAA API URL for GHCN data
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_ghcn_data(
    station_id,
    start_date="2023-01-01",
    end_date="2023-12-31",
    variables=["2m_temperature", "10m_wind_speed"],
):
    variables_map = {
        "2m_temperature": "TEMP",
        "10m_wind_speed": "WIND",
    }

    params = {
        "dataset": "GHCND",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(variables_map[v] for v in variables),
        "format": "json",
        "includeStationName": "true",
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None

    # Parse the data into a pandas DataFrame
    data = response.json()
    df = pd.DataFrame(data)

    if df.empty:
        print(f"No data found for station {station_id}")
        return None

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Set the datetime as the index
    df.set_index("date", inplace=True)

    # Ensure the required variables are present and clean
    df = df[["station", "datatype", "value"]]
    df = df.pivot(columns="datatype", values="value")

    # Rename columns for easier understanding (e.g., TEMP to temperature, WIND to wind_speed)
    df.rename(columns={"TEMP": "temperature", "WIND": "wind_speed"}, inplace=True)

    # Filter out rows with NaNs if they exist
    df.dropna(subset=["temperature", "wind_speed"], how="any", inplace=True)

    # Ensure the date range is consistent with the request (this should already be true)
    df = df.loc[start_date:end_date]

    return df


# %%
import matplotlib.pyplot as plt
import seaborn as sns


import xarray as xr


def compare_matched_stations(ds1, ds2, matches_df, year=2023):
    # Time range
    time_start = np.datetime64(f"{year}-01-01")
    time_end = np.datetime64(f"{year+1}-01-01")

    # Select stations
    ids_ds1 = matches_df["station_id_ds1"].values
    ids_ds2 = matches_df["station_id_ds2"].values

    # Select and rename for alignment
    ds1_sel = (
        ds1.sel(station_id=xr.DataArray(ids_ds1, dims="pair"))
        .sel(time=slice(time_start, time_end))
        .rename({"station_id": "pair"})
    )
    ds2_sel = (
        ds2.sel(station_id=xr.DataArray(ids_ds2, dims="pair"))
        .sel(time=slice(time_start, time_end))
        .rename({"station_id": "pair"})
    )

    # Align datasets
    ds1_sel, ds2_sel = xr.align(ds1_sel, ds2_sel, join="inner")
    ds1_sel = ds1_sel.chunk({"pair": -1, "time": -1})
    ds2_sel = ds2_sel.chunk({"pair": -1, "time": -1})

    # Calculate squared differences and RMSE
    diff = ds1_sel - ds2_sel
    mean_in_time = (diff**2).mean(dim="time", skipna=True).compute()
    rmse = xr.ufuncs.sqrt(mean_in_time.mean(dim="pair", skipna=True)).compute()
    errors = xr.ufuncs.sqrt(mean_in_time).compute()

    # Plot histograms
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for i, var in enumerate(["10m_wind_speed", "2m_temperature"]):
        ax = axs[i]
        data = errors[var].dropna(dim="pair").values
        data = np.clip(data, 2e-4, 1e1)
        bins = np.logspace(np.log10(1e-4), np.log10(1e1), 200)

        sns.histplot(
            data, bins=bins, ax=ax, color="royalblue", edgecolor=None, alpha=0.75
        )
        ax.axvline(
            rmse.get(var, np.nan),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="RMSE",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1e1)
        ax.set_title(
            f"{var.replace('_', ' ').title()} Error Histogram (2023)\nRMSE = {rmse.get(var, float('nan')):.3f}",
            fontsize=14,
        )
        ax.set_xlabel("Absolute Error (|DS1 - DS2|)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

    # Distance histogram
    ax_dist = axs[2]
    distances = matches_df["distance_km"].values
    distances = np.clip(distances, 2e-4, 1e1)  # Clip values to avoid extreme outliers
    bins = np.logspace(np.log10(1e-4), np.log10(1e1), 200)

    sns.histplot(
        distances, bins=bins, ax=ax_dist, color="green", edgecolor="black", alpha=0.75
    )
    ax_dist.set_xscale("log")
    ax_dist.set_yscale("log")
    ax_dist.set_xlim(1e-4, 1e1)
    ax_dist.set_title("Histogram of Distances Between Matched Stations", fontsize=14)
    ax_dist.set_xlabel("Distance (km)", fontsize=12)
    ax_dist.set_ylabel("Frequency", fontsize=12)

    plt.suptitle(
        "Error Distribution and Distance Between DS1 and DS2",
        fontsize=16,
        weight="bold",
    )
    plt.show()

    # ------------------------------
    # Plot time series for worst, median, and best RMSE
    # ------------------------------
    for var in ["10m_wind_speed", "2m_temperature"]:
        error_values = errors[var].values
        valid_indices = np.where(
            np.logical_and(~np.isnan(error_values), error_values > 1e-2)
        )[0]
        if len(valid_indices) == 0:
            continue

        sorted_indices = valid_indices[np.argsort(error_values[valid_indices])]
        plot_kinds = {
            "Worst": sorted_indices[-1],
            "Median": sorted_indices[len(sorted_indices) // 2],
            "Best": sorted_indices[0],
        }

        for kind, idx in plot_kinds.items():
            id_ds1 = matches_df.iloc[idx]["station_id_ds1"]
            id_ds2 = matches_df.iloc[idx]["station_id_ds2"]

            ts_ds1 = ds1.sel(station_id=id_ds1).sel(time=slice(time_start, time_end))[
                var
            ]
            ts_ds2 = ds2.sel(station_id=id_ds2).sel(time=slice(time_start, time_end))[
                var
            ]

            lat1 = ds1.sel(station_id=id_ds1)["latitude"].values.item()
            lon1 = ds1.sel(station_id=id_ds1)["longitude"].values.item()
            elevation_ds1 = ds1.sel(station_id=id_ds1)["elevation"].values.item()

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
            )

            # Timeseries comparison
            ax1.scatter(
                ts_ds1.time,
                ts_ds1,
                label=f"{ds1.encoding['source'].split('/')[-1]}: {id_ds1}",
                alpha=0.3,
            )
            ax1.scatter(
                ts_ds2.time,
                ts_ds2,
                label=f"{ds2.encoding['source'].split('/')[-1]}: {id_ds2}",
                alpha=0.8,
                marker="x",
            )

            ax1.set_ylabel(var.replace("_", " ").title())
            ax1.set_title(
                f"{var.replace('_', ' ').title()} - {kind} Match\n"
                f"RMSE = {error_values[idx]:.3f}\nLoc: ({lat1:.2f}, {lon1:.2f}), Elevation: {elevation_ds1:.2f} m, Distance: {matches_df.iloc[idx]['distance_km']:.2f} km",
                fontsize=14,
            )
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

            # Compute absolute difference and mask NaNs
            abs_diff = np.abs(ts_ds1 - ts_ds2)
            valid_mask = ~np.isnan(abs_diff.values)

            # Plot only where both datasets have data
            ax2.scatter(
                abs_diff.time.values[valid_mask],
                abs_diff.values[valid_mask],
                label="|DS1 - DS2|",
                color="orange",
                alpha=0.3,
            )
            ax2.set_ylabel("Absolute Error")
            ax2.set_xlabel("Time")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()

            plt.show()

    return rmse


rmse_results = compare_matched_stations(
    dataset_ours, dataset_isd, matches_df, year=2023
)

# %%

# Assuming matches_df is your DataFrame
# 1. Filter such that station_id_ds2 is unique, keeping the row with the largest difference
filtered_df = matches_df.loc[
    matches_df.groupby("station_id_ds2")["distance_km"].idxmax()
]

# 2. Sort by distance_km in descending order
sorted_df = filtered_df.sort_values(by="distance_km", ascending=False).reset_index(
    drop=True
)

# Display the result
print(sorted_df)

# %%
from wetterdienst import Settings
from wetterdienst.provider.noaa.ghcn import NoaaGhcnRequest

settings = Settings(
    ts_unit_targets={"temperature": "degree_kelvin", "speed": "meter_per_second"},
)
request = NoaaGhcnRequest(
    parameters=("hourly", "data", "temperature_air_mean_2m"),
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    settings=settings,
)
request = request.filter_by_station_id("52754")  # %%
df = request.values.all().df

# %%

# Load the CSV file
csv_file = "/home/niall/stationbench/89573099999.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(csv_file)

# Ensure the DATE column is in datetime format
data["DATE"] = pd.to_datetime(data["DATE"])

# Extract temperature (TMP) and convert it to a numeric value
# TMP is typically in the format "value,quality_flag", so we split and convert
data["TMP"] = (
    data["TMP"].str.split(",").str[0].astype(float) / 10.0 + 273.15
)  # Convert to Kelvin

# Extract temperature (TMP) and convert it to a numeric value
# TMP is typically in the format "value,quality_flag", so we split and convert
data["WND"] = (
    data["WND"].str.split(",").str[3].astype(float) / 10.0
)  # Convert to Kelvin

data_meteostat = fetch_meteostat_timeseries("89573", "2m_temperature")

# Plot TEMP against DATE
plt.figure(figsize=(12, 6))
plt.scatter(data["DATE"], data["TMP"], label="Raw ISD", color="blue", alpha=0.7)
plt.scatter(
    data_meteostat.index,
    data_meteostat["2m_temperature"],
    label="Raw Meteostat",
    color="orange",
    alpha=0.7,
)
plt.ylabel("Temperature (K)")
plt.title("Station no: 89573")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# %%
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "/home/niall/stationbench/89573099999.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(csv_file)

# Ensure the DATE column is in datetime format
data["DATE"] = pd.to_datetime(data["DATE"])

# Extract wind speed (WND) and convert it to a numeric value
# WND is typically in the format "direction,speed,quality_flag", so we split and convert
data["WND_proc"] = (
    data["WND"].str.split(",").str[3].astype(float) / 10.0
)  # Convert to m/s

data_meteostat = fetch_meteostat_timeseries("89573", "10m_wind_speed")

# Plot TEMP against DATE
plt.figure(figsize=(12, 6))
plt.plot(data["DATE"], data["WND_proc"], label="Raw ISD", color="blue", alpha=0.7)
plt.plot(
    data_meteostat.index,
    data_meteostat["10m_wind_speed"],
    label="Raw Meteostat",
    color="orange",
    alpha=0.7,
)
plt.ylabel("Wind speed (m/s)")
plt.title("Station no: 89573")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# %%
import xarray as xr

dataset = xr.open_zarr("https://opendata.jua.sh/stationbench/meteostat_benchmark.zarr")

# %%
