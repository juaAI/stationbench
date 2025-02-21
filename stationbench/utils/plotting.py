from stationbench.utils.formatting import format_variable_name
import xarray as xr
import plotly.express as px
from plotly.graph_objects import Figure

RMSE_THRESHOLD = 20
GEO_SCATTER_CONFIGS = {
    "rmse-ss": {
        "title_template": "{var}, Skill Score [%] at lead time {lead_time_title}",
        "cmin": -35,
        "cmax": 35,
        "cmap": "RdBu",
        "label": "skill_score",
    },
    "rmse": {
        "title_template": "{var}, RMSE at lead time {lead_time_title}. Global RMSE: {global_metric:.2f}",
        "cmin": 0,
        "cmax": 7,
        "cmap": "Reds",
        "label": "RMSE",
    },
    "mbe": {
        "title_template": "{var}, MBE at lead time {lead_time_title}. Global MBE: {global_metric:.2f}",
        "cmin": -5,
        "cmax": 5,
        "cmap": "RdBu",
        "label": "MBE",
    },
}


def get_geo_scatter_config(mode: str, var: str, lead_title: str, global_metric: float):
    var_str = format_variable_name(var)
    config = GEO_SCATTER_CONFIGS[mode]
    title_template = str(config["title_template"])
    title = title_template.format(
        var=var_str, lead_time_title=lead_title, global_metric=global_metric
    )
    return {
        "title": title,
        "cmin": config["cmin"],
        "cmax": config["cmax"],
        "cmap": config["cmap"],
        "label": config["label"],
    }


def geo_scatter(
    metric_ds: xr.Dataset,
    var: str,
    lead_range_slice: slice,
    mode: str,
    lead_title: str,
) -> dict[str, Figure]:
    """
    Generate a scatter plot of the metric values on a map

    Args:
        metric_ds: xarray Dataset with metrics (RMSE, MBE, skill score)
        var: variable to plot
        lead_range: lead range to plot (slice object)
        lead_title: lead range title to plot
        mode: "rmse", "mbe", or "rmse-ss" to plot the RMSE, MBE, or RMSE skill score
    """
    # Select the appropriate metric
    metric_ds = metric_ds[var].sel(metric=mode)

    # Apply threshold only to RMSE
    metrics_clean = metric_ds.sel(lead_time=lead_range_slice)
    if mode == "rmse":
        metrics_clean = metrics_clean.where(metrics_clean < RMSE_THRESHOLD)

    global_metric = float(metrics_clean.mean(skipna=True).compute().values)
    metrics_averaged = metrics_clean.mean(dim=["lead_time"], skipna=True).dropna(
        dim="station_id"
    )

    # Convert to dataframe for plotting
    metrics_df = metrics_averaged.reset_coords()[
        ["latitude", "longitude", var]
    ].to_dataframe()

    geo_scatter_config = get_geo_scatter_config(
        mode=mode,
        var=var,
        global_metric=global_metric,
        lead_title=lead_title,
    )
    if "level" in metrics_averaged.dims:
        metrics_df = metrics_df.sel(level=500)

    fig = px.scatter_mapbox(
        metrics_df,
        lat="latitude",
        lon="longitude",
        color=var,
        width=1200,
        height=1200,
        zoom=1,
        title=geo_scatter_config["title"],
        color_continuous_scale=geo_scatter_config["cmap"],
        range_color=(geo_scatter_config["cmin"], geo_scatter_config["cmax"]),
    )
    fig.update_layout(mapbox_style="carto-positron")
    return {f"{geo_scatter_config['label']}_{var}_{lead_title.replace(' ', '_')}": fig}
