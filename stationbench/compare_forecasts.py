import argparse
import logging
from typing import cast
import json

import plotly.express as px
import wandb
import xarray as xr
from wandb.errors import CommError

from stationbench.utils.regions import Region
from stationbench.utils.formatting import format_variable_name
from stationbench.utils.regions import (
    get_lat_slice,
    get_lon_slice,
    region_dict,
    select_region_for_stations,
)
from stationbench.utils.logging import init_logging

RMSE_THRESHOLD = 20

LEAD_RANGES = {
    "Short term (6-48 hours)": slice("06:00:00", "48:00:00"),
    "Mid term (3-7 days)": slice("72:00:00", "168:00:00"),
}

GEO_SCATTER_CONFIGS = {
    "rmse-ss": {
        "title_template": "{var}, Skill Score [%] at lead time {lead_time_title}",
        "cmin": -35,
        "cmax": 35,
        "cmap": "RdBu",
        "wandb_label": "skill_score",
    },
    "rmse": {
        "title_template": "{var}, RMSE at lead time {lead_time_title}. Global RMSE: {global_metric:.2f}",
        "cmin": 0,
        "cmax": 7,
        "cmap": "Reds",
        "wandb_label": "RMSE",
    },
    "mbe": {
        "title_template": "{var}, MBE at lead time {lead_time_title}. Global MBE: {global_metric:.2f}",
        "cmin": -5,
        "cmax": 5,
        "cmap": "RdBu",
        "wandb_label": "MBE",
    },
}

LINE_PLOT_CONFIGS = {
    "rmse-ss": {
        "title_template": "{var}, Skill Score ({region})",
        "ylabel": "Skill Score [%]",
        "wandb_label": "skill_score",
    },
    "rmse": {
        "title_template": "{var}, RMSE ({region})",
        "ylabel": "RMSE",
        "wandb_label": "RMSE",
    },
    "mbe": {
        "title_template": "{var}, MBE ({region})",
        "ylabel": "Mean Bias Error",
        "wandb_label": "MBE",
    },
}

logger = logging.getLogger(__name__)


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
        "wandb_label": config["wandb_label"],
    }


def get_line_plot_config(mode, var, region):
    var_str = format_variable_name(var)
    config = LINE_PLOT_CONFIGS[mode]
    title_template = str(config["title_template"])
    title = title_template.format(var=var_str, region=region)
    return {
        "title": title,
        "ylabel": config["ylabel"],
        "wandb_label": config["wandb_label"],
    }


class PointBasedBenchmarking:
    def __init__(self, wandb_run: wandb.sdk.wandb_run.Run):
        self.wandb_run = wandb_run
        artifact_name = f"{wandb_run.id}-temporal_plots"
        self.incumbent_wandb_artifact = None
        try:
            self.incumbent_wandb_artifact = self.wandb_run.use_artifact(
                f"{artifact_name}:latest"
            )
            logger.info("Incumbent artifact %s found...", artifact_name)
        except CommError:
            logger.info(
                "Artifact %s not found, will creating new artifact", artifact_name
            )

        self.wandb_artifact = wandb.Artifact(artifact_name, type="dataset")

    def generate_metrics(
        self,
        evaluation_benchmarks: xr.Dataset,
        reference_benchmark_locs: dict[str, str],
        region_names: list[str],
    ):
        regions = {
            region_name: region_dict[region_name] for region_name in region_names
        }
        # open and align reference benchmarks
        reference_metrics = {
            k: xr.open_zarr(v) for k, v in reference_benchmark_locs.items()
        }

        # align the metrics datasets so we can plot together
        metrics_ds, *ref_metrics = xr.align(
            evaluation_benchmarks, *reference_metrics.values(), join="left"
        )
        reference_metrics = dict(
            zip(reference_metrics.keys(), ref_metrics, strict=True)
        )

        logger.info(
            "Point based benchmarks computed, generating plots and writing to wandb..."
        )
        stats: dict[str, wandb.Plotly] = {}

        # Add debugging
        logger.info(f"Initial metrics_ds dimensions: {metrics_ds.dims}")
        logger.info(f"Initial metrics: {metrics_ds.metric.values}")

        # Add skill score to metrics
        ref = reference_metrics[next(iter(reference_benchmark_locs))]
        metrics_ds = metrics_ds.copy()

        # Create skill score dataset with same structure
        skill_score_ds = metrics_ds.sel(metric="rmse").copy()
        for var in metrics_ds.data_vars:
            skill_score = (
                1 - metrics_ds[var].sel(metric="rmse") / ref[var].sel(metric="rmse")
            ) * 100
            skill_score_ds[var] = skill_score

        # Add rmse-ss to metric coordinate
        skill_score_ds = skill_score_ds.expand_dims({"metric": ["rmse-ss"]})

        # Merge original and skill score datasets
        metrics_ds = xr.merge([metrics_ds, skill_score_ds])

        logger.info(f"After merge - dims: {metrics_ds.dims}")
        logger.info(f"After merge - metrics: {metrics_ds.metric.values}")

        # Generate plots
        for var in metrics_ds.data_vars:
            for lead_range_name, lead_range_slice in LEAD_RANGES.items():
                for mode in ["rmse", "mbe", "rmse-ss"]:
                    stats |= self._geo_scatter(
                        metric_ds=metrics_ds,
                        var=cast(str, var),
                        lead_range=lead_range_slice,
                        lead_title=lead_range_name,
                        mode=mode,
                    )

            # Generate temporal plots
            for mode in ["rmse", "mbe", "rmse-ss"]:
                stats |= self._plot_lines(
                    metric_ds=metrics_ds,
                    reference_metrics=reference_metrics,
                    var=cast(str, var),
                    regions=regions,
                    mode=mode,
                )

        self.wandb_run.log_artifact(self.wandb_artifact).wait()
        return stats

    def _geo_scatter(
        self,
        metric_ds: xr.Dataset,
        var: str,
        lead_range: slice,
        mode: str,
        lead_title: str,
    ) -> dict[str, wandb.Plotly]:
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
        metrics_clean = metric_ds.sel(lead_time=lead_range)
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
            logger.info("***** Selecting level 500")
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
        return {
            f"stations_spatial_metrics/{geo_scatter_config['wandb_label']}/"
            f"{var} {lead_title}": wandb.Plotly(fig)
        }

    def _plot_lines(
        self,
        metric_ds: xr.Dataset,
        reference_metrics: dict[str, xr.Dataset],
        var: str,
        regions: dict[str, Region],
        mode: str,
    ) -> dict[str, wandb.Plotly]:
        """Generate line plots of the metrics over time.

        Args:
            metric_ds: Dataset containing the metrics
            reference_metrics: Dictionary of reference metric datasets
            var: Variable to plot
            regions: Dictionary of regions to plot
            mode: "rmse", "mbe", or "rmse-ss" to plot RMSE, MBE, or skill score
        """
        # Make copies before filtering
        metric_ds = metric_ds.copy()
        reference_metrics = {k: v.copy() for k, v in reference_metrics.items()}

        # Filter data
        if mode == "rmse":
            metric_ds = metric_ds.where(
                metric_ds[var].sel(metric="rmse") < RMSE_THRESHOLD
            ).compute()
            for k, v in reference_metrics.items():
                reference_metrics[k] = v.where(
                    v[var].sel(metric="rmse") < RMSE_THRESHOLD
                ).compute()

        ret: dict[str, wandb.Plotly] = {}
        for region_name, region in regions.items():
            ret |= self._plot_line_for_region(
                region=region_name,
                var=var,
                metric_ds=self._select_region(ds=metric_ds, region=region),
                reference_metrics={
                    k: self._select_region(ds=v, region=region)
                    for k, v in reference_metrics.items()
                },
                mode=mode,
            )
        return ret

    def _plot_line_for_region(
        self,
        region: str,
        var: str,
        metric_ds: xr.Dataset,
        mode: str,
        reference_metrics: dict[str, xr.Dataset],
    ) -> dict[str, wandb.Plotly]:
        config = get_line_plot_config(mode=mode, var=var, region=region)
        x = metric_ds.lead_time.values.astype("timedelta64[h]").astype(int)
        line_label = "Forecast"
        skill_score_reference = next(iter(reference_metrics))

        # Prepare data for Forecast and reference models
        if mode == "rmse-ss":
            # Use the skill score we already calculated
            plot_data = {
                f"{line_label} vs {skill_score_reference}": metric_ds[var]
                .sel(metric="rmse-ss")
                .values.tolist()
            }
        else:
            # Use RMSE or MBE directly
            plot_data = {
                line_label: metric_ds[var].sel(metric=mode).values.tolist(),
                **{
                    ref_name: ref_metric[var].sel(metric=mode).values.tolist()
                    for ref_name, ref_metric in reference_metrics.items()
                },
            }

        # Update or create wandb Table
        table_name = f"temporal_{var}_{region}"
        table_data = [
            (model, lead_time, value)
            for model, values in plot_data.items()
            for lead_time, value in zip(x, values, strict=False)
        ]

        columns = ["model", "lead_time", "value"]
        if self.incumbent_wandb_artifact:
            incumbent_table = self.incumbent_wandb_artifact.get(table_name)
            if incumbent_table:
                existing_data = [
                    row
                    for row in incumbent_table.data
                    if row[0]
                    != (
                        f"{line_label} vs {skill_score_reference}"
                        if mode == "rmse-ss"
                        else line_label
                    )
                ]
                table_data = existing_data + table_data
                columns = incumbent_table.columns

        table = wandb.Table(data=table_data, columns=columns)
        self.wandb_artifact.add(table, table_name)

        # Create and configure the plot
        fig = px.line(
            plot_data,
            x=x,
            y=list(plot_data.keys()),
            title=config["title"],
            labels={"value": config["ylabel"], "x": "Lead time (hours)"},
            height=800,
        )
        for trace in fig.data:
            trace.connectgaps = True

        return {
            f"stations_temporal_metrics/{config['wandb_label']}/"
            f"{var}/{region}_line_plot": wandb.Plotly(fig)
        }

    def _select_region(self, ds: xr.Dataset, region: Region) -> xr.Dataset:
        lat_slice = get_lat_slice(region)
        lon_slice = get_lon_slice(region)
        ds = select_region_for_stations(ds, lat_slice, lon_slice)

        return ds.mean(dim=["station_id"], skipna=True)


def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Compare forecast benchmarks")
    parser.add_argument(
        "--evaluation_benchmarks_loc",
        type=str,
        required=True,
        help="Path to evaluation benchmarks",
    )
    parser.add_argument(
        "--reference_benchmark_locs",
        type=str,
        required=True,
        help="Dictionary of reference benchmark locations",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="W&B run name",
    )
    parser.add_argument(
        "--regions",
        type=str,
        required=True,
        help="Comma-separated list of regions",
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
        # Convert string arguments if needed
        if isinstance(args.reference_benchmark_locs, str):
            args.reference_benchmark_locs = json.loads(args.reference_benchmark_locs)
        if isinstance(args.regions, str):
            args.regions = [r.strip() for r in args.regions.split(",")]

    # Initialize wandb
    try:
        wandb_run = wandb.init(
            project="stationbench",
            name=args.run_name,
            id=args.run_name,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        wandb_run = None

    logger.info(
        "Logging metrics to WandB: %s",
        wandb_run.url if wandb_run else "WandB not available",
    )

    evaluation_benchmarks = xr.open_zarr(args.evaluation_benchmarks_loc)

    metrics = PointBasedBenchmarking(
        wandb_run=wandb_run,
    ).generate_metrics(
        evaluation_benchmarks=evaluation_benchmarks,
        reference_benchmark_locs=args.reference_benchmark_locs,
        region_names=args.regions,
    )
    if wandb_run is not None:
        wandb_run.log(metrics)
