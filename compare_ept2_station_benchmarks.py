# %%
import stationbench
from datetime import datetime

# %%

regions = ["global", "europe", "north-america"]

benchmark_datasets_locs = {
    "EC IFS": "/home/niall/stationbench/data/ec_hres_forecast_global_first_of_month.zarr",
    # "EPT-2 v1": "/home/niall/stationbench/data/ept2_v1_global_first_of_month.zarr",
    "EPT-2 v2": "/home/niall/stationbench/data/ept2_v2_global_first_of_month.zarr",
    "EPT-2 1.5 mixed": "/home/niall/stationbench/data/ept2_15_mixed_global_first_of_month.zarr",
    "EPT-1.5": "/home/niall/stationbench/data/ept1.5_global_first_of_month.zarr",
    "Aurora": "/home/niall/stationbench/data/aurora_global_first_of_month.zarr",
    # "EC ENS Mean": "/home/niall/stationbench/data/ec_ens_mean_global_weatherreal_00.zarr",
    "AIFS": "/home/niall/stationbench/data/aifs_global_first_of_month.zarr",
}

stationbench.compare_forecasts(
    benchmark_datasets_locs=benchmark_datasets_locs,
    regions=regions,
    wandb_run_name=f"weatherreal-run-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
)

# # %%
# import pandas as pd
# import plotly.graph_objects as go

# # Read the CSV file
# df = pd.read_csv("stationbench-results/temporal_metrics.csv")

# # filters to plot
# metric = "rmse"
# region = "global"
# # variable = "2m_temperature"
# variable = "10m_wind_speed"

# # Filter for RMSE metric and global region for 2m temperature
# mask = (df["metric"] == metric) & (df["region"] == region)

# # %%
# # Create the plot
# fig = go.Figure()

# unique_models = df["model"].unique()
# raw_models = unique_models[~pd.Series(unique_models).str.contains("-vs-")]

# for model in raw_models:
#     data = df[mask & (df["model"] == model)][["lead_time", variable]]

#     fig.add_trace(
#         go.Scatter(x=data["lead_time"], y=data[variable], name=model, mode="lines")
#     )


# fig.update_layout(
#     title="RMSE Evolution Over Forecast Lead Time",
#     xaxis_title="Lead Time (hours)",
#     yaxis_title=f"RMSE ({variable})",
#     showlegend=True,
#     template="plotly_white",
# )

# fig.show()
# # %%

# import xarray as xr

# xr.open_zarr(
#     "/mnt/jua-shared-1/jua-hindcasts/EPT2-global-from-2023-01-01-to-2024-12-28-v2.zarr"
# )
# # %%

# %%
