# Working with Custom Regions

StationBench comes with several predefined regions (`global`, `europe`, `north-america`), but you can also define your own custom regions for analysis.

## Adding a Custom Region

You can add a custom region using the `add_region` function:

```python
from stationbench.utils.regions import add_region

# Add a custom region for Australia
add_region(
    name="australia", 
    lat_slice=(-45, -10),  # (min_latitude, max_latitude)
    lon_slice=(110, 155)   # (min_longitude, max_longitude)
)
```

## Region Parameters

- `name`: A string identifier for your region
- `lat_slice`: A tuple of (min_latitude, max_latitude) in degrees
  - Values must be between -90 and 90
  - min_latitude must be less than max_latitude
- `lon_slice`: A tuple of (min_longitude, max_longitude) in degrees
  - Values must be between -180 and 180
  - min_longitude must be less than max_longitude

## Using Custom Regions

Once defined, you can use your custom region just like the built-in ones:

```python
# Calculate metrics for your custom region
stationbench.calculate_metrics(
    forecast="path/to/forecast.zarr",
    start_date="2023-01-01",
    end_date="2023-12-31",
    output="path/to/australia_metrics.zarr",
    region="australia",  # Your custom region
    name_10m_wind_speed="10si",
    name_2m_temperature="2t"
)

# Compare forecasts across multiple regions including your custom one
stationbench.compare_forecasts(
    benchmark_datasets_locs={
        "HRES": "path/to/hres_metrics.zarr", 
        "ENS": "path/to/ens_metrics.zarr"
    },
    regions=["europe", "australia"]  # Mix of built-in and custom regions
)
```

## Visualizing Custom Regions

When you use a custom region, all visualizations will automatically be bounded to your specified latitude and longitude ranges.

## Example: Creating Multiple Regional Analyses

```python
from stationbench.utils.regions import add_region

# Define several custom regions
regions = {
    "western-europe": ((40, 60), (-10, 15)),
    "eastern-europe": ((40, 60), (15, 40)),
    "scandinavia": ((55, 72), (5, 30)),
    "mediterranean": ((36, 45), (-5, 30))
}

# Add all regions
for name, (lat_range, lon_range) in regions.items():
    add_region(name, lat_range, lon_range)

# Now you can use any of these regions in your analyses
```

## Tips for Defining Regions

- Use meaningful names that clearly identify the geographical area
- Check your latitude/longitude bounds with a mapping tool
- For large regions, consider whether Dask parallelization might be beneficial
- Remember that regions are stored in memory, so they persist only for the duration of your Python session

