# %%
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt

stations = xr.open_dataset("/home/niall/stationbench/WeatherReal-ISD-2023.zarr")

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np

# Extract latitude and longitude
lats = stations.latitude.values
lons = stations.longitude.values

# Create a map
fig, ax = plt.subplots(
    subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 8), dpi=300
)

# Limit the map to Europe
ax.set_extent([-15, 45, 36, 72], crs=ccrs.PlateCarree())

# Add a background color for the map
ax.stock_img()

# Add coastlines and borders
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

# Add gridlines
gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()

# Plot station locations
ax.scatter(lons, lats, color="red", s=10, transform=ccrs.PlateCarree())

plt.show()
# %%
import numpy as np
from stationbench.compare_forecasts import LEAD_RANGES
from matplotlib.cm import get_cmap
import seaborn as sns
import matplotlib.pyplot as plt

# List of zarr files with labels
zarr_files = [
    # (
    #     "EC IFS",
    #     "/home/roberto/stationbench/data/ec_hres_forecast_global_weatherreal.zarr",
    # ),
    (
        "EC ENS Mean",
        "/home/roberto/stationbench/data/ec_ens_mean_global_weatherreal.zarr",
    ),
    # ("EC AIFS", "/home/roberto/stationbench/data/aifs_global_weatherreal.zarr"),
    # ("Aurora", "/home/roberto/stationbench/data/aurora_global_weatherreal.zarr"),
    # ("Jua EPT-1.5", "/home/roberto/stationbench/data/ept1.5_global_weatherreal.zarr"),
    # ("Jua EPT-2", "/home/roberto/stationbench/data/ept2_v2_global_weatherreal.zarr"),
    (
        "Jua EPT-2e",
        "/home/roberto/stationbench/data/model_ens_global_weatherreal.zarr",
    ),
]

# Get color maps
accent_cmap = get_cmap("Pastel1")
set1_cmap = get_cmap("Set1")

# # Define colors for each model
# model_colors = {
#     "EC IFS": set1_cmap(0),  # First color from Accent
#     "EC AIFS": set1_cmap(1),  # Second color from Accent
#     "Aurora": set1_cmap(2),  # Third color from Accent
#     "EPT-1.5": set1_cmap(3),  # Fourth color from Accent
#     "EPT-2 v2": set1_cmap(4),  # Fifth color from Accent
# }

# Define colors for each model
model_colors = {
    "EC IFS": accent_cmap(0),  # First color from Accent
    "EC ENS Mean": set1_cmap(0),  # First color from Set1
    "EC AIFS": accent_cmap(1),  # Second color from Accent
    "Aurora": accent_cmap(2),  # Third color from Accent
    "Jua EPT-1.5": accent_cmap(3),  # Fourth color from Accent
    "Jua EPT-2": accent_cmap(4),  # Fifth color from Accent
    "Jua EPT-2e": set1_cmap(1),  # Second color from Set1
}

# Variables to analyze
variables = [("2m Temperature", "2m_temperature"), ("10m Wind Speed", "10m_wind_speed")]

# Initialize arrays to store RMSE values and best model indices
num_stations = len(stations.station_id)

for variable_label, variable in variables:
    for lead_range_label, lead_range_slice in LEAD_RANGES.items():
        rmse_values = np.full((len(zarr_files), num_stations), np.inf)
        best_model_indices = np.full(num_stations, -1, dtype=int)

        # Loop through each zarr file and calculate mean RMSE for the lead range
        for model_idx, (label, zarr_file) in enumerate(zarr_files):
            ds = xr.open_dataset(zarr_file)
            lead_time_mask = (
                ds.lead_time
                >= np.timedelta64(int(lead_range_slice.start.split(":")[0]), "h")
            ) & (
                ds.lead_time
                <= np.timedelta64(int(lead_range_slice.stop.split(":")[0]), "h")
            )

            if lead_time_mask.sum() == 0:
                continue  # Skip this model if no valid lead times are available

            rmse = (
                ds[variable]
                .sel(metric="rmse")
                .isel(lead_time=lead_time_mask)
                .mean(dim="lead_time")
                .values
            )
            rmse_values[model_idx, :] = rmse

        # Determine the best model for each station
        valid_stations_mask = np.any(rmse_values != np.inf, axis=0)
        best_model_indices[valid_stations_mask] = np.argmin(
            rmse_values[:, valid_stations_mask], axis=0
        )

        # Assign colors based on the best model
        station_colors = [
            model_colors[zarr_files[idx][0]] if idx != -1 else "gray"
            for idx in best_model_indices
        ]

        # Plot the stations with colors indicating the best model
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 8), dpi=300
        )
        ax.set_extent([-15, 45, 36, 72], crs=ccrs.PlateCarree())
        ax.stock_img()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        # Plot station locations with colors
        scatter = ax.scatter(
            lons[valid_stations_mask],
            lats[valid_stations_mask],
            color=np.array(station_colors)[valid_stations_mask],
            s=3,
            transform=ccrs.PlateCarree(),
        )

        # Add a legend
        for label, _ in zarr_files:
            ax.scatter(
                [], [], color=model_colors[label], label=label, s=1
            )  # Invisible points for legend
        ax.legend(loc="lower left", fontsize="small")

        ax.set_title(
            f"Best Model for {variable_label} RMSE ({lead_range_label})", fontsize=16
        )

        plt.show()
# %%
# Plot RMSE vs. lead times for each variable with error bars

for variable_label, variable in variables:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for label, zarr_file in zarr_files:
        ds = xr.open_dataset(zarr_file)
        lead_times = ds.lead_time
        ds = ds.where(
            (ds.latitude >= 36)
            & (ds.latitude <= 72)
            & (ds.longitude >= -15)
            & (ds.longitude <= 45)
        )
        rmse = ds[variable].sel(metric="rmse").mean(dim="station_id")
        rmse_std = ds[variable].sel(metric="rmse").std(dim="station_id")

        # Select lead times at intervals of 6 hours using xarray
        lead_times_6h = lead_times.where(
            (lead_times / np.timedelta64(1, "h")) % 6 == 0, drop=True
        )
        rmse_6h = rmse.sel(lead_time=lead_times_6h)
        rmse_std_6h = rmse_std.sel(lead_time=lead_times_6h)

        # Use seaborn to add error bars
        sns.lineplot(
            x=lead_times_6h / np.timedelta64(1, "D"),
            y=rmse_6h,
            label=label,
            color=model_colors[label],
            ax=ax,
            err_style="band",
            ci=None,
            err_kws={"alpha": 0.2, "color": model_colors[label]},
        )
        # ax.fill_between(
        #     lead_times_6h / np.timedelta64(1, "D"),
        #     rmse_6h - rmse_std_6h,
        #     rmse_6h + rmse_std_6h,
        #     color=model_colors[label],
        #     alpha=0.2,
        # )

    ax.set_title(f"RMSE vs. Lead Time for {variable_label} (Europe)", fontsize=16)
    ax.set_xlabel("Lead Time (days)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(title="Model", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.show()
    plt.close()
# %%
from scipy.spatial import cKDTree
import seaborn as sns

# Plot error distribution for different lead times at a single station

# Function to find the nearest station
def find_nearest_station(lat, lon, station_lats, station_lons):
    tree = cKDTree(np.column_stack((station_lats, station_lons)))
    _, idx = tree.query([lat, lon])
    return idx

# Latitude and longitude of the target location
target_lat = 47.37  # Example latitude
target_lon = 8.54  # Example longitude

# Find the nearest station
nearest_station_idx = find_nearest_station(target_lat, target_lon, lats, lons)
nearest_station_id = stations.station_id.values[nearest_station_idx]

# Plot error distribution for each variable and lead time range
for variable_label, variable in variables:
    for lead_range_label, lead_range_slice in LEAD_RANGES.items():
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        for label, zarr_file in zarr_files:
            ds = xr.open_dataset(zarr_file)
            ds = ds.sel(station_id=nearest_station_id)
            lead_time_mask = (
                ds.lead_time
                >= np.timedelta64(int(lead_range_slice.start.split(":")[0]), "h")
            ) & (
                ds.lead_time
                <= np.timedelta64(int(lead_range_slice.stop.split(":")[0]), "h")
            )

            if lead_time_mask.sum() == 0:
                continue  # Skip if no valid lead times are available

            errors = (
                ds[variable]
                .sel(metric="rmse")
                .isel(lead_time=lead_time_mask)
                .values.flatten()
            )

            # Use seaborn to plot a KDE with individual observations as ticks
            sns.kdeplot(
                errors,
                fill=True,
                alpha=0.6,
                label=label,
                color=model_colors[label],
                ax=ax,
            )
            sns.rugplot(
                errors,
                height=0.1,
                color=model_colors[label],
                ax=ax,
            )

        ax.set_title(
            f"Error Distribution for {variable_label} at Station Zurich-Fluntern ({lead_range_label})",
            fontsize=16,
        )
        ax.set_xlabel("RMSE", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(title="Model", fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.show()
        plt.close()
# %%
