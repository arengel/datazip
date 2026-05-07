# Installation

## Requirements

- Python ≥ 3.12

## Basic Installation

```bash
pip install datazip
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add datazip
```

## Core Dependencies

These are installed automatically:

| Package | Purpose |
|---|---|
| `orjson` | Fast JSON encoding/decoding |

## Optional Dependencies

Some types require additional packages:

| Package       | Purpose                                   |
|---------------|-------------------------------------------|
| `pandas` and `pyarrow` | Pandas DataFrame and Series serialization |
| `polars`      | Polars DataFrame/Series serialization     |
| `numpy`       | Array serialization                       |


## Development Installation

To install with all development dependencies (including test tools):

```bash
pip install datazip[dev]
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/arengel/datazip.git
cd datazip
pip install -e ".[dev]"
```
