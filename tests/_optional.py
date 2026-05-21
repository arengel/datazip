"""Dummy modules and classes when optional packages are not installed.

Could do something more like this:
https://github.com/pola-rs/polars/blob/master/py-polars/polars/dependencies.py
"""


class Meta(type):
    def __getattr__(cls, name):
        return MT


class MT(metaclass=Meta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):

        return MT


try:
    import sqlalchemy
except (ModuleNotFoundError, ImportError):

    class sqlalchemy(MT):  # noqa: N801
        """Dummy for :mod:`sqlalchemy` when not installed."""

        pass


try:
    import plotly
except (ModuleNotFoundError, ImportError):

    class plotly(MT):  # noqa: N801
        """Dummy for :mod:`plotly` when not installed."""

        pass


try:
    import numpy

except (ModuleNotFoundError, ImportError):

    class numpy(MT):  # noqa: N801
        """Dummy for :mod:`numpy` when not installed."""

        pass


try:
    import pandas

    if pandas.__version__ < "2.0.0":
        raise ImportError("pandas < 2.0.0")

except (ModuleNotFoundError, ImportError):

    class pandas(MT):  # noqa: N801
        """Dummy for when pandas is not installed."""

        pass


try:
    import polars

except (ModuleNotFoundError, ImportError):

    class polars(MT):  # noqa: N801
        """Dummy for when polars is not installed."""

        pass
