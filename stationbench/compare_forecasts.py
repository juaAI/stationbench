import argparse
import logging
import json
from typing import cast
import wandb
import xarray as xr
from wandb.errors import CommError
import pandas as pd
import numpy as np

from stationbench.utils.regions import Region
from stationbench.utils.regions import (
    get_lat_slice,
    get_lon_slice,
    region_dict,
    select_region_for_stations,
)
from stationbench.utils.logging import init_logging
from stationbench.utils.plotting import geo_scatter

LEAD_RANGES = {
    "Short term (6-48 hours)": slice("06:00:00", "48:00:00"),
    "Mid term (3-7 days)": slice("72:00:00", "168:00:00"),
}

logger = logging.getLogger(__name__)


def convert_dataset_to_table(
    dataset: xr.Dataset, model_name: str
) -> pd.DataFrame:
    """Convert xarray dataset to pandas DataFrame that is compatible with wandb."""
    df = dataset.to_dataframe().reset_index()
    df["lead_time"] = df["lead_time"] / np.timedelta64(1, "h")
    df["model"] = model_name
    return df

def calculate_metric_skill_score(
    forecast: xr.Dataset, reference: xr.Dataset, metric: str
) -> xr.Dataset:
    """
    Calculate the skill score between a forecast dataset and a reference dataset.

    Parameters:
        forecast (xr.Dataset): The forecast dataset.
        reference (xr.Dataset): The reference dataset.
        metric (str): The metric to calculate the skill score for.

    Returns:
        xr.Dataset: The skill score dataset
    """
    skill_score = 1 - forecast / reference
    # Properly set up the metric dimension
    skill_score = skill_score.expand_dims({"metric": [f"{metric}-ss"]})
    return skill_score

def calculate_skill_scores(
        temporal_metrics: list[xr.Dataset],
        spatial_metrics: list[xr.Dataset],
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Calculate temporal and spatial skill scores."""
        base_rmse = temporal_metrics[0].sel(metric="rmse")
        reference_rmse = temporal_metrics[1].sel(metric="rmse")
        temporal_ss = calculate_metric_skill_score(
            base_rmse, reference_rmse, metric="rmse"
        )

        base_spatial_rmse = spatial_metrics[0].sel(metric="rmse")
        reference_spatial_rmse = spatial_metrics[1].sel(metric="rmse")
        spatial_ss = calculate_metric_skill_score(
            base_spatial_rmse, reference_spatial_rmse, metric="rmse"
        )
        return temporal_ss, spatial_ss


class PointBasedBenchmarking:
    def __init__(self, wandb_run: wandb.sdk.wandb_run.Run, region_names: list[str]):
        self.wandb_run = wandb_run
        self.regions = {
            region_name: region_dict[region_name] for region_name in region_names
        }

    def process_temporal_and_spatial_metrics(
        self,
        benchmark_datasets: dict[str, xr.Dataset],
    ) -> tuple[xr.Dataset, xr.Dataset]:
        temporal_metric_datasets = []
        spatial_metric_datasets = []
        for dataset in benchmark_datasets.values():
            temporal_metrics = []
            spatial_metrics = []
            
            for metric in dataset.metric.values:
                temporal_regions = []
                temp_dataset = self.calculate_spatial_metric(
                        dataset, metric,
                    )
                spatial_metrics.append(temp_dataset)
                
                for region in self.regions.values():
                    temp_dataset = self.calculate_temporal_metrics(
                        dataset, region, metric
                    )
                    temporal_regions.append(temp_dataset)

                # Concatenate datasets for each region
                combined_temporal_regions = xr.concat(temporal_regions, dim="region")
                temporal_metrics.append(combined_temporal_regions)

            # Concatenate datasets for each metric
            combined_temporal_metric = xr.concat(temporal_metrics, dim="metric")
            temporal_metric_datasets.append(combined_temporal_metric)
            combined_spatial_metric = xr.concat(spatial_metrics, dim="metric")
            spatial_metric_datasets.append(combined_spatial_metric)
        logger.info("Length of temporal metric datasets: %s", len(temporal_metric_datasets))
        logger.info("Length of spatial metric datasets: %s", len(spatial_metric_datasets))
        return temporal_metric_datasets, spatial_metric_datasets

    def calculate_temporal_metrics(
        self, dataset: xr.Dataset, region: Region, metric: str
    ) -> xr.Dataset:
        """Calculate temporal evolution of metrics across regions.

        Args:
            dataset: Input dataset containing variables
            regions: List of regions to calculate metrics for

        Returns:
            Dataset containing metrics for each region
        """
        metric_ds = dataset.sel(metric=metric)
        metric_ds = self._select_region(metric_ds, region)
        metric_ds = metric_ds.mean(dim="station_id", skipna=True)
        metric_ds = metric_ds.expand_dims(region=[region.name])
        return metric_ds
    
    def calculate_spatial_metric(
        self, dataset: xr.Dataset, metric: str,
    ) -> xr.Dataset:
        """Calculate spatial evolution of metrics across lead times."""
        metric_ds = dataset.sel(metric=metric)
        # averaging is done in plotting function because we first need to compute the skill score
        return metric_ds

    def _select_region(self, ds: xr.Dataset, region: Region) -> xr.Dataset:
        lat_slice = get_lat_slice(region)
        lon_slice = get_lon_slice(region)
        ds = select_region_for_stations(ds, lat_slice, lon_slice)
        return ds

def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Compare forecast benchmarks")
    parser.add_argument(
        "--benchmark_datasets_locs",
        type=str,
        required=True,
        help="Dictionary of benchmark datasets locations",
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
        if isinstance(args.benchmark_datasets_locs, str):
            args.benchmark_datasets_locs = json.loads(args.benchmark_datasets_locs)
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

    benchmark_datasets = {
        model_name: xr.open_zarr(benchmark_dataset_loc)
        for model_name, benchmark_dataset_loc in args.benchmark_datasets_locs.items()
    }
    model_names = list(benchmark_datasets.keys())
    benchmark_datasets = list(xr.align(*benchmark_datasets.values(), join="left"))
    benchmark_datasets = dict(zip(model_names, benchmark_datasets))


    metrics = PointBasedBenchmarking(
        wandb_run=wandb_run,
        region_names=args.regions,
    )
    
    temporal_metrics_datasets, spatial_metrics_datasets = metrics.process_temporal_and_spatial_metrics(
        benchmark_datasets=benchmark_datasets,
    )

    temporal_ss, spatial_ss = calculate_skill_scores(
        temporal_metrics_datasets, spatial_metrics_datasets
    )

    # convert temporal metrics to tables
    temporal_metrics_tables = [convert_dataset_to_table(metric, model_name) for metric, model_name in zip(temporal_metrics_datasets, model_names)]
    temporal_ss_table = convert_dataset_to_table(temporal_ss, "temporal_ss")
    temporal_metrics_tables.append(temporal_ss_table)

    # plot spatial metrics
    spatial_metrics_plots = {}
    for spatial_metric_dataset in spatial_metrics_datasets:
        for lead_range_name in LEAD_RANGES:
            lead_range_slice = LEAD_RANGES[lead_range_name]
            for var in spatial_metric_dataset.data_vars:
                for metric in spatial_metric_dataset.metric.values:
                    fig = geo_scatter(
                        metric_ds = spatial_metric_dataset,
                        var = var,
                        lead_range_slice = lead_range_slice,
                        mode = metric,
                        lead_title = lead_range_name,
                    )
                    spatial_metrics_plots.update(fig)
    
    for lead_range_name in LEAD_RANGES:
        lead_range_slice = LEAD_RANGES[lead_range_name]
        for var in spatial_ss.data_vars:
            fig = geo_scatter(
                metric_ds = spatial_ss,
                var = var,
                lead_range_slice = lead_range_slice,
                mode = 'rmse-ss',
                lead_title = lead_range_name,
            )
            spatial_metrics_plots.update(fig)


    if wandb_run is not None:
        combined_df = pd.concat(temporal_metrics_tables)
        logger.info("Combined dataframe shape: %s", combined_df.shape)
        logger.info("Combined dataframe columns: %s", combined_df.columns.tolist())
        logger.info("Combined dataframe head:\n%s", combined_df.head().to_string())
        logger.info("Combined dataframe tail:\n%s", combined_df.tail().to_string())
        stats = {'temporal_metrics_table': wandb.Table(dataframe=combined_df, columns=combined_df.columns.tolist())}

        stats.update({
            key: wandb.Plotly(value)
            for key, value in spatial_metrics_plots.items()
        })
        wandb_run.log(stats)

