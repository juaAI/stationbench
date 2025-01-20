def format_variable_name(var: str) -> str:
    """Format variable names for display in plots."""
    name_mapping = {
        "10m_wind_speed": "10m Wind Speed",
        "2m_temperature": "2m Temperature",
    }
    return name_mapping.get(var, var)
