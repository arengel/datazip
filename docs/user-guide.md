# User Guide

## How DataZip Works

A DataZip file is a standard `.zip` archive with a specific internal layout:

- **Data files**: Common data objects are stored as `.parquet` (DataFrames, Series), `.npy` (NumPy arrays), or `.pkl` (pickled objects like Plotly figures).
- **`__attributes__.json`**: References to all stored objects and their types.
- **`__metadata__.json`**: Version information, creation timestamp, and username.

This makes DataZip archives human-inspectable: you can open them with any zip tool and read the JSON files directly.

## Supported Types

### Primitives

All standard Python primitives are supported:

```python
with DataZip(buffer, "w") as z:
    z["s"] = "hello"
    z["i"] = 42
    z["f"] = 3.14
    z["b"] = True
    z["n"] = None
    z["c"] = 1 + 2j         # complex numbers
```

### Collections

```python
with DataZip(buffer, "w") as z:
    z["d"] = {"key": "value", "nested": {"a": 1}}
    z["l"] = [1, 2, 3]
    z["t"] = (1, "two", 3.0)    # tuples are preserved (not converted to list)
    z["s"] = {1, 2, 3}          # sets
    z["fs"] = frozenset({1, 2}) # frozensets
```

### Date and Time

```python
from datetime import datetime
with DataZip(buffer, "w") as z:
    z["dt"] = datetime(2024, 1, 15, 12, 0, 0)
```

### Paths

```python
from pathlib import Path
with DataZip(buffer, "w") as z:
    z["path"] = Path("/usr/local/data")
```

### NumPy Arrays

Arrays are stored in `.npy` format, preserving dtype and shape:

```python
import numpy as np
with DataZip(buffer, "w") as z:
    z["arr"] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
```

### Pandas DataFrames

DataFrames are stored as Parquet.

```python
import pandas as pd
with DataZip(buffer, "w") as z:
    # Regular DataFrame
    z["df"] = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    # MultiIndex columns
    z["multi"] = pd.DataFrame(
        {(0, "x"): [1, 2], (0, "y"): [3, 4], (1, "x"): [5, 6]}
    )
```

### Pandas Series

Series are stored as Parquet and the Series name is preserved:

```python
with DataZip(buffer, "w") as z:
    z["series"] = pd.Series([1, 2, 3], name="my_series")
```

### Polars

Polars DataFrames, LazyFrames, and Series are stored as Parquet:

```python
import polars as pl
with DataZip(buffer, "w") as z:
    z["pl_df"] = pl.DataFrame({"a": [1, 2, 3]})
    z["pl_lazy"] = pl.LazyFrame({"b": [4, 5, 6]})
    z["pl_series"] = pl.Series("c", [7, 8, 9])
```

### NamedTuples

NamedTuples are reconstructed if the class is importable. If not, they fall back to regular tuples:

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

with DataZip(buffer, "w") as z:
    z["pt"] = Point(1.0, 2.0)
```

## Custom Classes

### Automatic Serialization

Any class will be serialized automatically — no configuration needed:

```python
class Config:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

cfg = Config(0.01, 100)
with DataZip(buffer, "w") as z:
    z["cfg"] = cfg
```

#### Classes with `__slots__`

Classes using `__slots__` are also handled automatically:

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### Custom State Methods

For finer control, implement the standard pickle protocol:

```python
class MyClass:
    def __getstate__(self) -> dict:
        return {"data": self.data, "name": self.name}

    def __setstate__(self, state: dict) -> None:
        self.data = state["data"]
        self.name = state["name"]
```

### DataZip-specific State Methods

Use `_dzgetstate_` and `_dzsetstate_` when you need different behavior for DataZip vs. pickle. These take priority over `__getstate__`/`__setstate__` when DataZip is serializing:

```python
class MyClass:
    def _dzgetstate_(self) -> dict:
        # Exclude 'cache' attribute only for DataZip
        return {k: v for k, v in self.__dict__.items() if k != "cache"}

    def _dzsetstate_(self, state: dict) -> None:
        self.__dict__ = state
        self.cache = {}  # Reinitialize cache on load
```

### Priority Order

When serializing, DataZip checks for state methods in this order:

1. `_dzgetstate_` / `_dzsetstate_` (DataZip-specific)
2. `__getstate__` / `__setstate__` (standard pickle protocol)
3. Automatic `__dict__` / `__slots__` inspection

## Extending DataZip with Custom Coders

For types that DataZip doesn't support out of the box — or that don't round-trip
cleanly through the default `__getstate__`/`__setstate__` protocol — you can
register a pair of encoder/decoder functions with
[`DataZip.register_coders`][datazip.core.DataZip.register_coders]. This is the
same mechanism DataZip uses internally to add support for NumPy, pandas, Polars,
and Plotly when those libraries are installed.

### The Coder Contract

- **Encoder** signature: `encoder(self, name, item) -> JSONABLE`
    - `self` is the open `DataZip` instance (useful if you need to write a
      side-car file via `self._encode_loc_helper`).
    - `name` is the key the object is being stored under — handy when naming
      side-car files.
    - `item` is the object to encode.
    - Returns a JSON-able value. For non-primitive types, return a `dict` that
      contains a `"__type__"` key matching the `name` you registered the
      decoder under.
- **Decoder** signature: `decoder(self, obj) -> Any`
    - `obj` is the dict produced by the encoder.
    - Returns the reconstructed object.

### Registering Coders for a New Type

The simplest case is a type whose state can be captured as a string or other
JSON-able value. For example, [`decimal.Decimal`][decimal.Decimal] doesn't
round-trip with the default state protocol, but it has a trivial string
representation:

```python
from decimal import Decimal
from io import BytesIO
from datazip import DataZip


def encode_decimal(z, name, item):
    return {"__type__": "Decimal", "items": str(item)}


def decode_decimal(z, obj):
    return Decimal(obj["items"])


DataZip.register_coders(Decimal, "Decimal", encode_decimal, decode_decimal)

buffer = BytesIO()
with DataZip(buffer, "w") as z:
    z["pi"] = Decimal("3.14159")

with DataZip(buffer, "r") as z:
    assert z["pi"] == Decimal("3.14159")
```

### Storing Larger Payloads as Side-Car Files

When the object is bulky (a tensor, an image, a parquet-friendly table, ...),
it's usually better to write the payload to its own file inside the archive
rather than inlining it in the JSON attributes. Use
`self._encode_loc_helper(filename, data, bytes_)` from the encoder; it picks a
unique name inside the zip, writes the bytes, and returns the resolved name to
store under `"__loc__"`. The decoder then reads it back with `self.read(...)`.

```python
import numpy as np
from io import BytesIO


def encode_ndarray(z, name, data):
    np.save(buf := BytesIO(), data, allow_pickle=False)
    return {
        "__type__": "ndarray",
        "__loc__": z.encode_loc_helper(f"{name}.npy", data, buf.getvalue()),
    }


def decode_ndarray(z, obj):
    return np.load(BytesIO(z.read(obj["__loc__"])))


DataZip.register_coders(np.ndarray, "ndarray", encode_ndarray, decode_ndarray)
```

This is exactly how the bundled NumPy support is implemented.

### Backwards-Compatible `__type__` Names

If you ever change the `name` that an encoder writes, pass the old identifier
as `alt_name` so existing archives still decode:

```python
DataZip.register_coders(
    MyType,
    "MyType",                  # new identifier the encoder writes today
    encode_mytype,
    decode_mytype,
    alt_name="my_legacy_name", # also dispatches archives written under the old name
)
```

`alt_name` accepts either a single string or a tuple — useful when migrating
across multiple historical names.

### When to Reach for `register_coders`

Prefer the [custom-class options above](#custom-classes) (automatic state,
`__getstate__`/`__setstate__`, or `_dzgetstate_`/`_dzsetstate_`) when you own
the class. Reach for `register_coders` when:

- The type is owned by a third-party library you can't modify.
- The default state protocol doesn't preserve enough information (e.g.
  `Decimal`).
- You want to store the payload in a format other than JSON — for example,
  Parquet, `.npy`, or a pickled binary blob in its own side-car file.

## Object Deduplication

By default, DataZip tracks object identities to avoid storing the same object multiple times. This means multiple references to the same object are deduplicated:

```python
shared = [1, 2, 3]
with DataZip(buffer, "w") as z:
    z["a"] = shared
    z["b"] = shared   # stored only once; on read, a and b will be the same list
```

!!! warning "Deduplication and object lifetime"
    Python reuses memory addresses for objects with non-overlapping lifetimes. If you create an object, store it, delete it, then create a new object that happens to get the same memory address, DataZip may incorrectly skip storing the new object.

    Use `z.reset_ids()` to clear the deduplication cache between such operations, or disable deduplication entirely with `ids_for_dedup=False`:

    ```python
    DataZip(buffer, "w", ids_for_dedup=False)
    ```

## Updating Archives

DataZip is write-once by design (a zip file constraint). To update an existing archive, use `DataZip.replace()`:

```python
# Replace values for specific keys; all other keys are copied unchanged
with DataZip.replace("data.zip", threshold=0.8) as z:
    z["new_feature"] = [1, 2, 3]
```

To keep the original file as a backup:

```python
with DataZip.replace("data.zip", save_old=True, threshold=0.8) as z:
    pass  # "data_old.zip" will be kept alongside the new "data.zip"
```

## Deep Key Access

For nested DataZip structures (e.g. DataZips containing dicts of dicts), pass all the keys for nested access:

```python
with DataZip(buffer, "r") as z:
    value = z["outer_key", "inner_key"]
```
