# cisTEM Testing Tools

This directory contains test scripts and utilities for testing cisTEM functionality.

## Running Tests

Each test can be run individually from their respective directories. For example:

```bash
# Run template matching reproducibility test
python /path/to/cisTEM/scripts/testing/programs/match_template/test_template_reproducibility.py --binary-path /path/to/binaries
```

## Temporary Directory Management

Test scripts that create temporary files use a centralized tracking system to make cleanup easier.
You can list and clean up temporary directories using the following options:

- `--list-temp-dirs`: List all tracked temporary directories
- `--rm-temp-dir INDEX`: Remove a specific temporary directory by index
- `--rm-all-temp-dirs`: Remove all tracked temporary directories