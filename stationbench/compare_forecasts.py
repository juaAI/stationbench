import argparse
import ast
import logging
from typing import cast
import json

import plotly.express as px
import wandb
import xarray as xr
from wandb.errors import CommError

from stationbench.utils.regions import Region
from stationbench.utils.metrics import format_variable_name
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
    "ss": {
        "title_template": "{var}, Skill Score [%] at lead time {lead_time_title}",
        "cmin": -35,
        "cmax": 35,
        "cmap": "RdBu",
        "wandb_label": "skill_score",
    },
    "rmse": {
        "title_template": "{var}, RMSE at lead time {lead_time_title}. Global RMSE: {global_rmse:.2f}",
        "cmin": 0,
        "cmax": 7,
        "cmap": "Reds",
        "wandb_label": "RMSE",
    },
}

LINE_PLOT_CONFIGS = {
    "ss": {
        "title_template": "{var}, Skill Score ({region})",
        "ylabel": "Skill Score [%]",
        "wandb_label": "skill_score",
    },
    "rmse": {
        "title_template": "{var}, RMSE ({region})",
        "ylabel": "RMSE",
        "wandb_label": "RMSE",
    },
}

logger = logging.getLogger(__name__)


def get_geo_scatter_config(mode: str, var: str, lead_title: str, global_rmse: float):
    var_str = format_variable_name(var)
    config = GEO_SCATTER_CONFIGS[mode]
    title_template = str(config["title_template"])
    title = title_template.format(
        var=var_str, lead_time_title=lead_title, global_rmse=global_rmse
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
        # open reference benchmarks:
        reference_rmses = {
            k: xr.open_zarr(v) for k, v in reference_benchmark_locs.items()
        }

        # move grid to -180 to 180 if needed:
        if evaluation_benchmarks.longitude.min() < 0:
            evaluation_benchmarks["longitude"] = xr.where(
                evaluation_benchmarks["longitude"] > 180,
                evaluation_benchmarks["longitude"] - 360,
                evaluation_benchmarks["longitude"],
            )
            for k, _ in reference_rmses.items():
                reference_rmses[k]["longitude"] = xr.where(
                    reference_rmses[k]["longitude"] > 180,
                    reference_rmses[k]["longitude"] - 360,
                    reference_rmses[k]["longitude"],
                )

        # align the reference rmses to the rmse so that we can plot together:
        rmse, *new_rmses = xr.align(
            evaluation_benchmarks, *reference_rmses.values(), join="left"
        )
        reference_rmses = dict(zip(reference_rmses.keys(), new_rmses, strict=True))

        # generate plots and write to wandb
        logger.info(
            "Point based benchmarks computed, generating plots and writing to wandb..."
        )
        stats: dict[str, wandb.Plotly] = {}
        variables = evaluation_benchmarks.data_vars
        skill_score_reference = next(iter(reference_benchmark_locs))
        for var in variables:
            for (
                lead_range_name,
                lead_range_slice,
            ) in LEAD_RANGES.items():
                stats |= self._geo_scatter(
                    rmse=rmse,
                    var=cast(str, var),
                    lead_range=lead_range_slice,
                    lead_title=lead_range_name,
                    mode="rmse",
                )
                stats |= self._geo_scatter(
                    rmse=(1 - rmse / reference_rmses[skill_score_reference]) * 100,
                    var=cast(str, var),
                    lead_range=lead_range_slice,
                    lead_title=lead_range_name,
                    mode="ss",
                )

            stats |= self._plot_lines(
                rmse=rmse,
                reference_rmses=reference_rmses,
                var=cast(str, var),
                regions=regions,
                mode="rmse",
            )

            stats |= self._plot_lines(
                rmse=rmse,
                reference_rmses=reference_rmses,
                var=cast(str, var),
                regions=regions,
                mode="ss",
            )

        self.wandb_run.log_artifact(self.wandb_artifact).wait()
        return stats

    def _geo_scatter(
        self,
        rmse: xr.Dataset,
        var: str,
        lead_range: slice,
        mode: str,
        lead_title: str,
    ) -> dict[str, wandb.Plotly]:
        """
        Generate a scatter plot of the RMSE values on a map
        :param rmse: xarray dataset with the RMSE values
        :param var: variable to plot
        :param lead_range: lead range to plot (slice object)
        :param lead_title: lead range title to plot
        :param mode: "rmse" or "ss" to plot the RMSE or skill score
        """
        rmse = rmse.where(rmse[var] < RMSE_THRESHOLD)
        rmse_clean = rmse.sel(lead_time=lead_range)

        global_rmse = float(rmse_clean[var].mean(skipna=True).compute().values)
        rmse_averaged = rmse_clean.mean(dim=["lead_time"], skipna=True).dropna(
            dim="station_id"
        )
        geo_scatter_config = get_geo_scatter_config(
            mode=mode,
            var=var,
            global_rmse=global_rmse,
            lead_title=lead_title,
        )
        if "level" in rmse_averaged.dims:
            logger.info("***** Selecting level 500")
            rmse_averaged = rmse_averaged.sel(level=500)

        fig = px.scatter_mapbox(
            rmse_averaged,
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
            f"point-based-benchmarking/spatial_error/{geo_scatter_config['wandb_label']}/"
            f"{var} {lead_title}": wandb.Plotly(fig)
        }

    def _plot_lines(
        self,
        rmse: xr.Dataset,
        reference_rmses: dict[str, xr.Dataset],
        var: str,
        regions: dict[str, Region],
        mode: str,
    ) -> dict[str, wandb.Plotly]:
        """
        Generate a line plot of the RMSE values over time
        :param rmse: xarray dataset with the RMSE values
        :param rerference_rmses: xarray datasets with the reference RMSE values
        :param var: variable to plot
        :param mode: "rmse" or "ss" to plot the RMSE or skill score
        """

        rmse = rmse.where(rmse[var] < RMSE_THRESHOLD).compute()
        for k, v in reference_rmses.items():
            reference_rmses[k] = v.where(v[var] < RMSE_THRESHOLD).compute()

        ret: dict[str, wandb.Plotly] = {}
        for region_name, region in regions.items():
            ret |= self._plot_line_for_region(
                region=region_name,
                var=var,
                rmse=self._select_region(ds=rmse, region=region),
                reference_rmses={
                    k: self._select_region(ds=v, region=region)
                    for k, v in reference_rmses.items()
                },
                mode=mode,
            )
        return ret

    def _plot_line_for_region(
        self,
        region: str,
        var: str,
        rmse: xr.Dataset,
        mode: str,
        reference_rmses: dict[str, xr.Dataset],
    ) -> dict[str, wandb.Plotly]:
        config = get_line_plot_config(mode=mode, var=var, region=region)
        x = rmse.lead_time.values.astype("timedelta64[h]").astype(int)
        line_label = "Jua"
        skill_score_reference = next(iter(reference_rmses))

        # Prepare data for Jua and reference models
        if mode == "ss":
            ss_rmse = reference_rmses[skill_score_reference]
            plot_data = {
                f"{line_label} vs {skill_score_reference}": (
                    (1 - rmse[var].values / ss_rmse[var].values) * 100
                ).tolist()
            }
        else:
            plot_data = {
                line_label: rmse[var].values.tolist(),
                **{
                    ref_name: ref_rmse[var].values.tolist()
                    for ref_name, ref_rmse in reference_rmses.items()
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
                        if mode == "ss"
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
            f"point-based-benchmarking/temporal_error/{config['wandb_label']}/"
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
    wandb_run = wandb.init(id=args.run_name, project="stationbench")
    if wandb_run is None:
        raise RuntimeError("Failed to initialize wandb run")

    evaluation_benchmarks = xr.open_zarr(args.evaluation_benchmarks_loc)
    
    metrics = PointBasedBenchmarking(
        wandb_run=wandb_run,
    ).generate_metrics(
        evaluation_benchmarks=evaluation_benchmarks,
        reference_benchmark_locs=args.reference_benchmark_locs,
        region_names=args.regions,
    )
    wandb_run.log(metrics)
