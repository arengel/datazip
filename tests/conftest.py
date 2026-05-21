"""PyTest configuration module.

Defines useful fixtures, command line args.
"""

import logging
import shutil
from pathlib import Path

import pytest

from tests._optional import numpy as np
from tests._optional import pandas as pd
from tests._optional import polars as pl

logger = logging.getLogger(__name__)


def idfn(val):
    """ID function for pytest parameterization."""
    if isinstance(val, float):
        return None
    return str(val)


def assert_equal(left, right, check_pd_dtype=True) -> None:  # noqa: FBT002
    """Recursively check that left and right objects are equal."""
    if type(left) is not type(right):
        raise AssertionError(f"{type(left)=} is not {type(right)=}")
    if isinstance(right, pd.Series):
        pd.testing.assert_series_equal(left, right, check_dtype=check_pd_dtype)
    elif isinstance(right, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_dtype=check_pd_dtype)
    elif isinstance(right, pl.Series):
        import polars.testing as pltesting

        pltesting.assert_series_equal(left, right, check_dtypes=check_pd_dtype)
    elif isinstance(right, pl.DataFrame | pl.LazyFrame):
        import polars.testing as pltesting

        pltesting.assert_frame_equal(left, right, check_dtypes=check_pd_dtype)
    elif isinstance(right, list | tuple):
        for v0, v1 in zip(left, right, strict=True):
            assert_equal(v0, v1)
    elif isinstance(right, dict):
        for k0v0, k1v1 in zip(left.items(), right.items(), strict=True):
            assert_equal(k0v0, k1v1)
    else:
        try:
            if not np.all(left == right):
                raise AssertionError(f"{type(left)=} is not {type(right)=}")
        except AttributeError:
            try:
                if not all(l == r for l, r in zip(left, right, strict=True)):  # noqa: E741
                    raise AssertionError(f"{type(left)=} is not {type(right)=}")
            except TypeError:
                if left != right:
                    raise AssertionError(f"{type(left)=} is not {type(right)=}")  # noqa: B904


@pytest.fixture
def df_dict() -> dict:
    """Dictionary of dfs."""
    return {
        "a": pd.DataFrame(
            [[0, 1], [2, 3]],
            columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
        ),
        "b": pd.Series([1, 2, 3, 4]),
    }


# @pytest.fixture
# def klass_w_slot(df_dict):
#     """Generic class that uses slots."""
#     obj = _KlassSlots()
#     obj.foo = df_dict["a"]
#     obj.tup = (1, 2)
#     obj.lis = (3, 4)
#     obj._dfs = df_dict
#     return obj
#
#
# @pytest.fixture
# def klass_wo_slot(df_dict):
#     """Generic class that does not use slots."""
#     obj = _TestKlass()
#     obj.foo = df_dict["a"]
#     obj._dfs = df_dict
#     return obj


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests
    directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def temp_dir(test_dir) -> Path:
    """Return the path to a temp directory that gets deleted on teardown."""
    out = test_dir / "temp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(exist_ok=True)
    yield out
    shutil.rmtree(out)
