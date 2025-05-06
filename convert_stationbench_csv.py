# %%
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

BLOBS = [
    "/home/niall/stationbench/stationbench-results/temporal_metrics.csv",
]

# %%
big_df = []

for blob in BLOBS:
    df = pd.read_csv(blob)
    df = df.reset_index(drop=True)  # Ensure unique indices
    big_df.append(df)

big_df = pd.concat(big_df, ignore_index=True)  # Concatenate with unique indices

# Remove rows with lead_time > 240
big_df = big_df[big_df["lead_time"] <= 240]

# %%
rename_dict = {
    "region": {
        "global": "Global",
        "europe": "Europe",
        "north-america": "North America",
    },
    "metric": {
        "rmse": "RMSE",
        "mbe": "MBE",
    },
}

# Rename entries in the corresponding columns of big_df using rename_dict
for column, mapping in rename_dict.items():
    big_df[column] = big_df[column].replace(mapping)

# Define the mapping for renaming columns
rename_columns = {
    "region": "Region",
    "metric": "Metric",
    "lead_time": "Prediction timedelta",
    "2m_temperature": "Air Temperature 2m [°C]",
    "10m_wind_speed": "Wind Speed 10m [m s⁻¹]",
    "model": "Model",
}

# Select only the desired columns
desired_columns = list(rename_columns.keys())
big_df = big_df[desired_columns]

# Rename the columns
big_df = big_df.rename(columns=rename_columns)

# %%
big_df.to_csv(
    "/home/niall/jua-core/live/services/dashboard/public/assets/benchmark-data/station-benchmarks-ept-2.csv",
    index=False,
    sep=",",
)

# %%
ept_1_5 = pd.read_csv(
    "/home/niall/jua-core/live/services/dashboard/public/assets/benchmark-data/benchmarks-ept-1-5.csv"
)
big_df = pd.concat(
    [big_df, ept_1_5[ept_1_5["Model"] == "EPT-1.5"]],
)

# Filter the data to only include rows where Metric is RMSE
rmse_df = big_df[big_df["Metric"] == "RMSE"]

# List of variables to plot
variables = [
    "Mean Sea-level Pressure [Pa]",
    "Air Temperature 2m [°C]",
    "Geopotential [m² s⁻²]",
    "Wind Speed 100m [m s⁻¹]",
    "Wind Speed 10m [m s⁻¹]",
]

# Create separate plots for each region
regions = rmse_df["Region"].unique()
for region in regions:
    region_df = rmse_df[rmse_df["Region"] == region]
    for variable in variables:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=region_df,
            x="Prediction timedelta",
            y=variable,
            hue="Model",
            marker="o",
        )
        plt.title(f"RMSE of {variable} vs Prediction Timedelta ({region})")
        plt.xlabel("Prediction Timedelta (hours)")
        plt.ylabel(f"RMSE of {variable}")
        plt.legend(title="Model")
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

# %%
