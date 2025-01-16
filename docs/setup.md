# Setup Guide

This guide will walk you through setting up StationBench on your local machine.

## Prerequisites

- Python 3.11 or higher

## Installation
The simplest way to install StationBench is via pip:

```bash
pip install stationbench
```

## Verifying Installation

You can verify the installation by running:

```bash
# Using CLI
stationbench-calculate --help
stationbench-compare --help

# Or in Python
python -c "import stationbench; print(stationbench.__version__)"
```

## Troubleshooting

### Common Issues

1. **Installation fails**
   - Ensure you have Python 3.11+ installed
   - Try upgrading pip: `pip install --upgrade pip`

2. **Development installation issues**
   - Update Poetry: `poetry self update`
   - Clear Poetry's cache: `poetry cache clear . --all`
   - Remove the `.venv` directory and reinstall: 
     ```bash
     rm -rf .venv
     poetry install
     ```

### Getting Help

If you encounter any issues:
1. Check the [project issues](https://github.com/juaAI/stationbench/issues)
2. Create a new issue with:
   - Your system information
   - The command you ran
   - The full error message
   - Steps to reproduce the problem

## Next Steps

Once you have completed the setup:
1. Read through the [tutorial](tutorial.ipynb)
2. Start benchmarking your weather forecasts!