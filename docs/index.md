# DataZip

**DataZip** is a Python library that extends [`zipfile.ZipFile`](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile) to provide seamless serialization and deserialization of complex Python objects — a more portable and readable alternative to pickle for data science workflows.

## Why DataZip?

- **Human-inspectable archives**: DataZip files are standard `.zip` files. You can open them with any archive tool and inspect the contents.
- **Broad type support**: Works out of the box with pandas DataFrames/Series, NumPy arrays, Polars DataFrames, datetimes, paths, sets, frozensets, complex numbers, and custom classes.
- **Efficient storage**: Tabular data is stored as Parquet; arrays as `.npy`. JSON is used for metadata and simple types.
- **Lazy loading**: Objects and data are only deserialized when they are accessed, allowing efficient loading of objects within huge files. Nested access avoids deserialzing unnecessary enclosing objects.
- **No pickle by default**: Most types are serialized without pickle, making files safer and more portable.
- **Custom class integration**: Any class that implements `__getstate__`/`__setstate__` (the standard pickle protocol) works automatically. The `IOMixin` makes it even simpler.
- **Pluggable type support**: Teach DataZip how to handle any third-party or stdlib type by registering encoder/decoder pairs with [`DataZip.register_coders`][datazip.core.DataZip.register_coders]. The bundled NumPy, pandas, Polars, and Plotly integrations are themselves built on this hook — see the [User Guide](user-guide.md#extending-datazip-with-custom-coders).

## Quick Example

```python
from io import BytesIO
import pandas as pd
from datazip import DataZip

# Write
buffer = BytesIO()
with DataZip(buffer, "w") as z:
    z["df"] = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    z["config"] = {"threshold": 0.5, "labels": ["a", "b"]}
    z["values"] = {1, 2, frozenset([3, 4])}

# Read
with DataZip(buffer, "r") as z:
    df = z["df"]
    config = z["config"]
```

## Supported Types

| Category | Types                                                                                                                                           |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Primitives | `str`, `int`, `float`, `bool`, `None`, `complex`                                                                                                |
| Collections | `dict`, `list`, `tuple`, `set`, `frozenset`, `deque`, `defaultdict`                                                                             |
| Date/Time | `datetime`, `pandas.Timestamp`                                                                                                                  |
| Paths | `pathlib.Path`                                                                                                                                  | |
| Custom | Any class with `__getstate__`/`__setstate__`                                                                                                    |
| Optional | `numpy.ndarray`, `pandas.DataFrame`, `pandas.Series`, `polars.DataFrame`, `polars.LazyFrame`, `polars.Series`, `xarray.Dataset`, Plotly figures |

## Installation

See the [Installation](installation.md) page for full details including optional dependencies.
