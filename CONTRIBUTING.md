# Contributing to StationBench

We love your input! We want to make contributing to StationBench as easy and transparent as possible.

## Development Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:juaAI/stationbench.git
   cd stationbench
   ```

2. Create a new branch for your changes:
   ```bash
   git checkout -b branch-name
   ```

3. Install Poetry if you haven't already by following the instructions at:
   https://python-poetry.org/docs/

4. Install dependencies:
   ```bash
   poetry install --with dev
   ```

5. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Development Workflow

### Code Formatting and Linting

We use `ruff` for both formatting and linting. Format your code before committing:

```bash
poetry exec format
```

### Running Tests

Run the test suite:
```bash
poetry run pytest
```

## Pull Request Process

1. Ensure your code follows our style guide (run `poetry run format`)
2. Add tests for any new functionality
3. Update documentation if you're changing functionality
4. Create a Pull Request with a clear description:
   - What changes were made
   - Why the changes were made
   - Any notable implementation details

## Documentation

When adding new features, please:
1. Add docstrings to new functions/classes following this format:
   ```python
   def function_name(param1: type, param2: type) -> return_type:
       """Short description.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value
       """
   ```

2. Update relevant files in docs/
3. Update README.md if adding major features

## Code Style Guide

- Use type hints for all function arguments and return values
- Follow PEP 8 guidelines (enforced by ruff)
- Use descriptive variable names
- Add comments for complex logic

## Questions or Problems?

- Open an issue for bugs or feature requests
- Contact maintainers for security issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License. 