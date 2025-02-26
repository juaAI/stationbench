from typing import Union, Optional, List, Dict, Any
import xarray as xr


def load_dataset(
    dataset: Union[str, xr.Dataset, xr.DataArray],
    variables: Optional[List[str]] = None,
    chunks: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """Load dataset from path or return dataset directly.

    Args:
        dataset: Dataset, DataArray, or path to zarr store
        variables: Optional list of variables to load
        chunks: Optional chunks specification for dask

    Returns:
        Loaded xarray Dataset
    """
    if isinstance(dataset, str):
        kwargs = {}
        if variables is not None:
            kwargs["variables"] = variables
        if chunks is not None:
            kwargs["chunks"] = chunks
        return xr.open_zarr(dataset, **kwargs)
    elif isinstance(dataset, xr.DataArray):
        return dataset.to_dataset(name=dataset.name or "data")
    return dataset
