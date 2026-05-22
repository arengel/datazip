"""Datazip."""

import logging
from functools import partial
from io import BytesIO

# Create a root logger for use anywhere within the package.
logger = logging.getLogger("datazip")

try:
    from datazip._version import version as __version__
except ImportError:
    logger.warning("Version unknown because package is not installed.")
    __version__ = "unknown"

from datazip.core import (  # noqa: E402
    DataZip,
    _decode_cache_helper,
    _encode_ignore,
    _quote_strip,
)
from datazip.mixin import IOMixin  # noqa: E402

try:
    import numpy as np

    def _encode_ndarray(z: DataZip, name: str, data: np.ndarray, **kwargs) -> dict:
        if z._ids_for_dedup and (loc := z._ids.get((id(data), type(data)), None)):
            return {"__type__": "ndarray", "__loc__": loc}
        np.save(temp := BytesIO(), data, allow_pickle=False)
        return {
            "__type__": "ndarray",
            "__loc__": z.encode_loc_helper(f"{name}.npy", data, temp.getvalue()),
        }

    def _decode_ndarray(z: DataZip, obj) -> np.ndarray:
        return np.load(BytesIO(z.read(obj["__loc__"])))

    DataZip.register_coders(
        np.ndarray,
        "ndarray",
        _encode_ndarray,
        partial(_decode_cache_helper, func=_decode_ndarray),
        ("numpy", "ndarray", None),
    )
    DataZip.register_coders(np.float64, "float64", lambda _, __, item: float(item))
    DataZip.register_coders(np.float64, "int64", lambda _, __, item: int(item))


except (ModuleNotFoundError, ImportError):
    pass


try:
    import pandas as pd

    if pd.__version__ < "2.0.0":
        raise ImportError("pandas < 2.0.0")

    def _encode_pd_df(z: DataZip, name: str, df: pd.DataFrame, **kwargs) -> dict:
        """Write a df in the ZIP as parquet."""
        if z._ids_for_dedup and (loc := z._ids.get((id(df), type(df)), None)):
            return {"__type__": "pdDataFrame", "__loc__": loc}
        try:
            return {
                "__type__": "pdDataFrame",
                "__loc__": z.encode_loc_helper(f"{name}.parquet", df, df.to_parquet()),
            }
        except Exception as exc:
            dt = df.dtypes.to_string().replace("\n", "\n\t")
            raise TypeError(
                f"Unable to write {type(df)} '{name}' as parquet with types\n {dt}"
            ) from exc

    def _encode_pd_series(z: DataZip, name: str, df: pd.Series, **kwargs) -> dict:
        if z._ids_for_dedup and (loc := z._ids.get((id(df), type(df)), None)):
            return {"__type__": "pdSeries", "__loc__": loc}
        return {
            "__type__": "pdSeries",
            "__loc__": z.encode_loc_helper(
                f"{name}.parquet", df, df.to_frame().to_parquet()
            ),
            "no_pqt_cols": [
                list(df.name) if isinstance(df.name, tuple) else df.name,
                None,
            ],
        }

    def _decode_pd_df(z: DataZip, obj: dict) -> pd.DataFrame:
        return pd.read_parquet(BytesIO(z.read(obj["__loc__"])))

    def _decode_pd_series(z: DataZip, obj: dict) -> pd.Series:
        out: pd.Series = pd.Series(
            pd.read_parquet(BytesIO(z.read(obj["__loc__"]))).squeeze()
        )
        cols, _names = obj.get("no_pqt_cols", (None, None))
        out.name = tuple(cols) if isinstance(cols, list) else cols
        return out

    DataZip.register_coders(
        pd.DataFrame,
        "pdDataFrame",
        _encode_pd_df,
        partial(_decode_cache_helper, func=_decode_pd_df),
        ("pandas.core.frame", "DataFrame", None),
    )
    DataZip.register_coders(
        pd.Series,
        "pdSeries",
        _encode_pd_series,
        partial(_decode_cache_helper, func=_decode_pd_series),
        ("pandas.core.series", "Series", None),
    )
    DataZip.register_coders(
        pd.Timestamp,
        "pdTimestamp",
        lambda _, __, item: {"__type__": "pdTimestamp", "items": str(item)},
        lambda _, obj: pd.Timestamp(_quote_strip(obj["items"])),
    )


except (ModuleNotFoundError, ImportError):
    pass


try:
    import polars as pl

    def _encode_pl_df(z: DataZip, name: str, df: pl.DataFrame, **kwargs) -> dict:
        """Write a polars df in the ZIP as parquet."""
        if z._ids_for_dedup and (loc := z._ids.get((id(df), type(df)), None)):
            return {"__type__": "plDataFrame", "__loc__": loc}
        df.write_parquet(temp := BytesIO())
        return {
            "__type__": "plDataFrame",
            "__loc__": z.encode_loc_helper(f"{name}.parquet", df, temp.getvalue()),
        }

    def _encode_pl_ldf(z: DataZip, name: str, df: pl.LazyFrame, **kwargs) -> dict:
        """Write a polars df in the ZIP as parquet."""
        if z._ids_for_dedup and (loc := z._ids.get((id(df), type(df)), None)):
            return {"__type__": "plLazyFrame", "__loc__": loc}
        df.collect().write_parquet(temp := BytesIO())  # ty:ignore[unresolved-attribute]
        return {
            "__type__": "plLazyFrame",
            "__loc__": z.encode_loc_helper(f"{name}.parquet", df, temp.getvalue()),
        }

    def _encode_pl_series(z: DataZip, name: str, df: pl.Series, **kwargs) -> dict:
        """Write a polars series in the ZIP as parquet."""
        if z._ids_for_dedup and (loc := z._ids.get((id(df), type(df)), None)):
            return {"__type__": "plSeries", "__loc__": loc}
        df.to_frame("IGNORE").write_parquet(temp := BytesIO())
        return {
            "__type__": "plSeries",
            "__loc__": z.encode_loc_helper(f"{name}.parquet", df, temp.getvalue()),
            "col_name": df.name,
        }

    def _decode_pl_df(z: DataZip, obj: dict) -> pl.DataFrame:
        return pl.read_parquet(BytesIO(z.read(obj["__loc__"])), use_pyarrow=True)

    def _decode_pl_ldf(z: DataZip, obj) -> pl.LazyFrame:
        return pl.read_parquet(BytesIO(z.read(obj["__loc__"])), use_pyarrow=True).lazy()

    def _decode_pl_series(z: DataZip, obj: dict) -> pl.Series:
        return (
            pl.read_parquet(BytesIO(z.read(obj["__loc__"])), use_pyarrow=True)
            .to_series()
            .alias(obj["col_name"])
        )

    DataZip.register_coders(
        pl.DataFrame,
        "plDataFrame",
        _encode_pl_df,
        partial(_decode_cache_helper, func=_decode_pl_df),
    )
    DataZip.register_coders(
        pl.LazyFrame,
        "plLazyFrame",
        _encode_pl_ldf,
        partial(_decode_cache_helper, func=_decode_pl_ldf),
    )
    DataZip.register_coders(
        pl.Series,
        "plSeries",
        _encode_pl_series,
        partial(_decode_cache_helper, func=_decode_pl_series),
    )

except (ModuleNotFoundError, ImportError):
    pass

try:
    import pickle

    from plotly import graph_objects as go

    def _encode_plotly(z: DataZip, name: str, item) -> dict:
        return {
            "__type__": "pgoFigure",
            "__loc__": z.encode_loc_helper(f"{name}.pkl", item, pickle.dumps(item)),
        }

    def _decode_plotly(z: DataZip, obj: dict) -> go.Figure:
        return pickle.load(BytesIO(z.read(obj["__loc__"])))  # noqa: S301

    DataZip.register_coders(go.Figure, "pgoFigure", _encode_plotly, _decode_plotly)

except (ModuleNotFoundError, ImportError):
    pass


try:
    import sqlalchemy as sa  # ty:ignore[unresolved-import]

    DataZip.register_coders(
        sa.engine.Engine,
        "saEngine",
        _encode_ignore,
        lambda _, obj: sa.create_engine(obj["items"]["url"]),
    )

except (ModuleNotFoundError, ImportError):
    pass


try:
    import xarray as xr

    def _encode_xrdataset(z: DataZip, name: str, data: xr.Dataset, **kwargs) -> dict:
        if z._ids_for_dedup and (loc := z._ids.get((id(data), type(data)), None)):
            return {"__type__": "xrDataset", "__loc__": loc}
        return {
            "__type__": "xrDataset",
            "__loc__": z.encode_loc_helper(f"{name}.nc", data, data.to_netcdf()),
        }

    def _decode_xrdataset(z: DataZip, obj) -> xr.Dataset:
        return xr.open_dataset(BytesIO(z.read(obj["__loc__"]))).load()

    DataZip.register_coders(
        xr.Dataset,
        "xrDataset",
        _encode_xrdataset,
        partial(_decode_cache_helper, func=_decode_xrdataset),
    )


except (ModuleNotFoundError, ImportError):
    pass

__all__ = ["DataZip", "IOMixin", "__version__"]
