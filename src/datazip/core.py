"""Code for [`DataZip`][datazip.core.DataZip]."""

from __future__ import annotations

import getpass
import logging
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import KeysView
from datetime import datetime
from functools import partial
from importlib import import_module
from io import BytesIO
from pathlib import Path, PosixPath, WindowsPath
from types import NoneType
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import orjson as json

from datazip import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from os import PathLike

    JSONABLE = float | int | dict | list | str | bool | None

LOGGER = logging.getLogger("datazip")


def _quote_strip(string: str) -> str:
    return string.replace("'", "").replace('"', "")


def _get_version(obj: Any) -> str:
    mod = import_module(obj.__class__.__module__.partition(".")[0])
    for v_attr in ("__version__", "version", "release"):
        if hasattr(mod, v_attr):
            return getattr(mod, v_attr)
    return "unknown"


def _get_username():
    try:
        return getpass.getuser()
    except (ModuleNotFoundError, OSError) as exc0:
        import os

        try:
            return os.getlogin()
        except Exception as exc1:
            LOGGER.error("No username %r from %r", exc1, exc0)
            return "unknown"


def _objinfo(obj: Any) -> str:
    return obj.__class__.__module__ + "|" + obj.__class__.__qualname__


def _get_klass(mod_klass: str | list | tuple):

    if isinstance(mod_klass, str):
        mod_klass = mod_klass.split("|")
    try:
        mod, qname, *_ = mod_klass
        klass: type = getattr(import_module(mod), qname)
    except (AttributeError, ModuleNotFoundError) as exc:
        raise ImportError(f"Unable to import {qname} from {mod}.") from exc
    else:
        return klass


def default_setstate(obj, state):
    """Called if no `__setstate__` implementation."""
    if state is None:
        pass
    elif isinstance(state, dict):
        obj.__dict__ = state
    elif isinstance(state, tuple):
        d_state, s_state = state
        if d_state is not None:
            obj.__dict__ = d_state
        for k, v in s_state.items():
            setattr(obj, k, v)


def default_getstate(obj):
    """Called if no `__getstate__` implementation."""

    def slots_dict(_slots):
        sout = {}
        for k in _slots:
            if k != "__dict__":
                try:  # noqa: SIM105
                    sout.update({k: getattr(obj, k)})
                except AttributeError:
                    pass
        return sout

    match obj:
        case object(__dict__=d_state, __slots__=slots):
            return d_state.copy(), slots_dict(slots)
        case object(__dict__=d_state):
            return d_state.copy()
        case object(__slots__=slots):
            return None, slots_dict(slots)
        case _:
            return None


def _decode_cache_helper(z: DataZip, obj: dict, func: Callable, **kwargs) -> Any:
    if obj["__loc__"] in z._red:
        return z._red[obj["__loc__"]]
    out = func(z, obj, **kwargs)
    z._red[obj["__loc__"]] = out
    return out


def _encode_ignore(z: DataZip, name: str, item):
    LOGGER.warning("%s of type %s will not be encoded", name, type(item))
    return "__IGNORE__"


class DataZip(ZipFile):
    """A `zipfile.ZipFile` with methods for easier use with Python objects."""

    suffixes: ClassVar[dict[str | tuple, str]] = {
        "pdDataFrame": ".parquet",
        "pdSeries": ".parquet",
        "ndarray": ".npy",
        # LEGACY type encoding
        ("pandas.core.frame", "DataFrame", None): ".parquet",
        ("pandas.core.series", "Series", None): ".parquet",
        ("numpy", "ndarray", None): ".npy",
    }

    def __init__(
        self,
        file: str | PathLike | BytesIO,
        mode: Literal["r", "w"] = "r",
        *args,
        ids_for_dedup=True,
        ignore_pd_dtypes=False,
        **kwargs,
    ):
        """Create a DataZip.

        Args:
            file: Either the path to the file, or a file-like object.
                If it is a path, the file will be opened and closed by DataZip.
            mode: The mode can be either read 'r', or write 'w'.
            recipes: Deprecated.
            compression: ZIP_STORED (no compression), ZIP_DEFLATED (requires zlib),
                ZIP_BZIP2 (requires bz2) or ZIP_LZMA (requires lzma).
            args: additional positional will be passed to
                `zipfile.ZipFile.__init__`.
            ids_for_dedup: If True, multiple references to the same object will not
                cause the object to be stored multiple times. If False, the object
                will be stored as many times as it has references. True can save space
                but because ids are not unique for objects with non-overlapping
                lifetimes, setting to True can result in subsequent new objects NOT
                being stored because they share an id with an earlier object.
            ignore_pd_dtypes: Ignored.
            kwargs: keyword arguments will be passed to
                `zipfile.ZipFile.__init__`.

        Examples:
            First we can create a [DataZip][datazip.core.DataZip]. In this case we are
            using a buffer (`io.BytesIO`) for convenience. In most cases though, `file`
            would be a `pathlib.Path` or `str` that represents a file. In these cases
            a `.zip` extension will be added if it is not there already.

            >>> buffer = BytesIO()  # can also be a file-like object
            >>> with DataZip(file=buffer, mode="w") as z0:
            ...     z0["foo"] = {
            ...         "a": (1, (2, {3})),
            ...         "b": frozenset({1.5, 3}),
            ...         "c": 0.9 + 0.2j,
            ...     }

            Getting items from [DataZip][datazip.core.DataZip], like setting them, uses
            standard Python subscripting.

            While always preferable to use a context manager as above, here it's more
            convenient to keep the object open. Even more unusual types that can't
            normally be stored in json should work.

            >>> z1 = DataZip(buffer, "r")
            >>> z1["foo"]
            {'a': (1, (2, {3})), 'b': frozenset({1.5, 3}), 'c': (0.9+0.2j)}

            Checking to see if an item is in a [DataZip][datazip.core.DataZip] uses
            standard Python syntax.

            >>> "foo" in z1
            True

            >>> len(z1)
            1

            When not used with a context manager, [DataZip][datazip.core.DataZip] should
            close itself automatically but it's not a bad idea to make sure.

            >>> z1.close()

            A [DataZip][datazip.core.DataZip] is a write-once, read-many affair because
            of the way `zip` files work. Appending to a
            [DataZip][datazip.core.DataZip] can be done with the
            [DataZip.replace][datazip.core.DataZip.replace] method.

            >>> buffer1 = BytesIO()
            >>> with DataZip.replace(buffer1, buffer, foo=5, bar=6) as z:
            ...     z["new"] = "foo"
            ...     z["foo"]
            5
        """
        if mode in ("a", "x"):
            raise ValueError("DataZip does not support modes 'a' or 'x'")

        # if isinstance(file, str):
        #     file = Path(file)

        clobber = kwargs.pop("clobber", False)
        if isinstance(file, Path):
            file = file.with_suffix(".zip")
            if file.exists() and mode == "w":
                if clobber:
                    file.unlink()
                else:
                    raise FileExistsError(
                        f"{file} exists, you cannot write or append to an "
                        f"existing DataZip."
                    )

        super().__init__(file, mode, *args, **kwargs)
        self._ignore_pd_dtypes = ignore_pd_dtypes
        self._attributes, self._metadata = {"__state__": {}}, {"__rev__": 2}
        self._ids_for_dedup = ids_for_dedup
        self._ids, self._red = {}, {}
        if mode == "r":
            self._attributes = self._json_get(
                "__attributes__", "attributes", "other_attrs"
            )
            self._metadata = self._json_get("__metadata__", "metadata")
            if self._metadata.get("__rev__", 1) != 2:
                warnings.warn(
                    f"{file} was created with an older version of DataZip, "
                    "all data might not be accessible, consider using v0.1.0.",
                    UserWarning,
                    stacklevel=2,
                )
                self._attributes = self._attributes | self._load_legacy_helper()

        self._delete_on_close = None

    @staticmethod
    def dump(obj: Any, file: Path | str | BytesIO, **kwargs) -> None:
        """Write the DataZip representation of `obj` to `file`.

        Args:
            obj: A Python object, it must implement `__getstate__` and
                `__setstate__`. There are other restrictions, especially if it
                contains instances of other custom Python objects, it may be enough
                for all of them to implement `__getstate__` and `__setstate__`.
            file: a file-like object, or a buffer where the
                [DataZip][datazip.core.DataZip] will be saved.
            kwargs: keyword arguments will be passed to
                [DataZip][datazip.core.DataZip].

        Returns:
            None

        Examples:
            Create an object that you would like to save as a
            [DataZip][datazip.core.DataZip].

            >>> class Foo:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            ...
            ...     def __repr__(self):
            ...         return f"Foo(a={self.a}, b={self.b})"
            >>> obj = Foo(a=5, b={"c": [2, 3.5]})
            >>> obj
            Foo(a=5, b={'c': [2, 3.5]})

            Save the object as a [DataZip][datazip.core.DataZip].

            >>> buffer = BytesIO()
            >>> DataZip.dump(obj, buffer)
            >>> del obj

            Get it back.

            >>> obj = DataZip.load(buffer, Foo)  # doctest: +SKIP
            >>> obj  # doctest: +SKIP
            Foo(a=5, b={'c': [2, 3.5]})
        """
        with DataZip(file, "w", **kwargs) as self:
            self["state"] = obj

    @staticmethod
    def load(file: Path | str | BytesIO, klass: type | None = None) -> Any:
        """Return the reconstituted object specified in the file.

        Args:
            file: a file-like object, or a buffer from which the
                [DataZip][datazip.core.DataZip] will be read.
            klass: (Optional) allows passing the class when it is known, this
                is handy when it is not possible to import the module that defines
                the class that `file` represents.

        Returns:
            Object from [DataZip][datazip.core.DataZip].

        Examples:
            See [DataZip.dump][datazip.core.DataZip.dump] for examples.
        """
        with DataZip(file, "r") as self:
            try:
                return DataZip._decode_obj(self, self._attributes["state"], klass)
            except KeyError:
                if len(self.keys()) == 1:
                    return self[next(iter(self.keys()))]
                return dict(self.items())

    @classmethod
    def replace(
        cls,
        file_or_new_buffer: str | PathLike | BytesIO,
        old_buffer: BytesIO | None = None,
        save_old=False,  # noqa: FBT002
        iterwrap=None,
        **kwargs,
    ):
        """Replace an old [DataZip][datazip.core.DataZip] with an editable new one.

        Note: Data and keys that are copied over by this function cannot be reliably
        mutated. `kwargs` must be used to replace the data associated with keys that
        exist in the old [DataZip][datazip.core.DataZip].

        Args:
            file_or_new_buffer: Either the path to the file to be replaced
                or the new buffer.
            old_buffer: only required if `file_or_new_buffer` is a buffer.
            save_old: if True, the old [DataZip][datazip.core.DataZip] will be
                saved with "_old" appended, if False it will be
                deleted when the new [DataZip][datazip.core.DataZip] is closed.
            iterwrap: this will be used to wrap the iterator that handles
                copying data to the new [DataZip][datazip.core.DataZip] to enable
                a progress bar, i.e. `tqdm`.
            kwargs: data that should be written into the new
                [DataZip][datazip.core.DataZip], for any keys that were in the old
                [DataZip][datazip.core.DataZip], the new value provided here will
                be used instead.

        Returns:
            New editable [DataZip][datazip.core.DataZip] with old data copied into it.

        Examples:
            Create a new test file object and put a datazip in it.

            >>> file = Path.home() / "test.zip"
            >>> with DataZip(file=file, mode="w") as z0:
            ...     z0["series"] = [1, 2, 4]

            Create a replacement DataZip.

            >>> z1 = DataZip.replace(file, save_old=False)

            The replacement has the old content.

            >>> z1["series"]
            [1, 2, 4]

            We can also now add to it.

            >>> z1["foo"] = "bar"

            While the replacement is open, the old verion still exists.

            >>> (Path.home() / "test_old.zip").exists()
            True

            Now we close the replacement which deletes the old file.

            >>> z1.close()
            >>> (Path.home() / "test_old.zip").exists()
            False

            Reopening the replacement, we see it contains all the objects.

            >>> z2 = DataZip(file, "r")

            >>> z2["series"]
            [1, 2, 4]

            >>> z1["foo"]
            'bar'

            And now some final test cleanup.

            >>> z2.close()
            >>> file.unlink()
        """
        if isinstance(file_or_new_buffer, BytesIO) and not isinstance(
            old_buffer, BytesIO
        ):
            raise TypeError(
                "If file_or_new_buffer is BytesIO, then old_buffer must be as well."
            )

        _to_delete = None
        if isinstance(file_or_new_buffer, str):
            file_or_new_buffer = Path(file_or_new_buffer)

        if isinstance(file_or_new_buffer, Path):
            file_or_new_buffer = file_or_new_buffer.with_suffix(".zip")
            old_buffer = Path(
                str(file_or_new_buffer).removesuffix(".zip") + "_old"
            ).with_suffix(".zip")
            file_or_new_buffer.rename(old_buffer)
            if not save_old:
                _to_delete = old_buffer

        self = cls(file_or_new_buffer, "w")
        if iterwrap is None:
            iterwrap = iter
        with DataZip(old_buffer, "r") as z:
            if z._metadata.get("__rev__") != 2:
                _to_delete = None
                LOGGER.warning(
                    "%s uses an old version of DataZip so not all data may be properly "
                    "copied over. For this reason, %s will not be deleted.",
                    file_or_new_buffer,
                    old_buffer,
                )
            for k, v in iterwrap(z.items()):
                if k in kwargs:
                    self[k] = kwargs.pop(k)
                else:
                    self[k] = v
            for k, v in iterwrap(kwargs.items()):
                self[k] = v

        self._delete_on_close = _to_delete

        return self

    def close(self) -> None:
        """Close the file, and for mode 'w' write attributes and metadata."""
        if self.fp is None:
            return

        if self.mode == "w":
            self.writestr(
                "__attributes__.json",
                json.dumps(self._attributes, option=json.OPT_NON_STR_KEYS),
            )
            self.writestr("__metadata__.json", json.dumps(self._metadata))
        self._red = {}
        super().close()
        if isinstance(self._delete_on_close, Path):
            self._delete_on_close.unlink()

    def __contains__(self, item) -> bool:
        """Provide `in` check."""
        return item.partition(".")[0] in self._attributes

    def __len__(self) -> int:
        """Provide for use of `len` builtin."""
        return len(self._attributes) - (1 if "__state__" in self._attributes else 0)

    def __getitem__(self, key: str | tuple) -> Any:
        """Retrieve an item from a [DataZip][datazip.core.DataZip].

        Args:
            key: name of item to retrieve. If multiple keys are provided,
                they are looked up recursively.

        Returns:
            Data associated with key

        Examples:
            >>> with DataZip(BytesIO(), mode="w") as z0:
            ...     z0["foo"] = {"a": [{"c": 5}]}
            ...     z0["foo", "a", 0, "c"]
            5

        """
        if isinstance(key, str | int):
            return self._decode(self._attributes[key])
        out = self._attributes
        for k in key:
            out = out[k]
            if isinstance(out, dict):
                # all the real keys of __state__ are strings
                out = self._attributes["__state__"].get(out.get("__loc__", -9999), out)
        return self._decode(out)

    def __setitem__(self, key: str, value: Any) -> None:
        """Write an item to a [DataZip][datazip.core.DataZip]."""
        if self.mode == "r":
            raise ValueError("Writing to DataZip requires mode 'w'")
        if key in ("__metadata__", "__attributes__", "__state__"):
            raise KeyError(f"{key=} is reserved, please use a different name")
        if key in self._attributes:
            raise KeyError(f"{key=} already in {self.filename}")
        if not isinstance(key, str):
            raise TypeError(f"{key=} is invalid, key must be a string.")
        if (for_attributes := self._encode(key, value)) != "__IGNORE__":
            self._attributes.update({key: for_attributes})

    def get(self, key: str, default=None) -> Any:
        """Retrieve an item if it is there otherwise return default."""
        return self[key] if key in self else default  # noqa: SIM401

    def reset_ids(self) -> None:
        """Reset the internal record of stored ids.

        Because 'two objects with non-overlapping lifetimes may have the same
        `id` value', it can be useful to reset the set of seen ids when
        you are adding objects with non-overlapping lifetimes.

        See `id`.
        """
        self._ids = {}

    def items(self) -> Generator[tuple[str, Any]]:
        """Lazily read name/key value pairs from a [DataZip][datazip.core.DataZip]."""
        for k in self._attributes.keys():  # noqa: SIM118
            if k == "__state__":
                continue
            yield k, self[k]

    def keys(self) -> KeysView:
        """Set of names in [DataZip][datazip.core.DataZip] as if it was a `dict`."""
        return KeysView(set(self._attributes) - {"__state__"})

    @classmethod
    def register_coders(
        cls,
        type_,
        name,
        encoder: Callable,
        decoder: Callable | None = None,
        alt_name: str | tuple | None = None,
    ):
        """Register custom encoder and decoder functions for a specific type.

        This class method allows registration of custom encoding and decoding
        functions that will be used to serialize and deserialize objects of a
        specified type. The encoder is mapped to the type itself, while the
        decoder is mapped to one or more string names that identify the encoded
        format.

        Encoders are called with `(self, name, item)` and must return a JSON-able
        value. For non-primitive types this is typically a `dict` containing a
        `"__type__"` key whose value matches `name`. Decoders are called with
        `(self, obj)`, where `obj` is the dict produced by the encoder, and must
        return the reconstructed object.

        Args:
            type_: The type/class for which the encoder should be registered.
            name: Primary name identifier for the decoder. The encoder must set
                `__type__` to this value so the matching decoder can be found.
            encoder: Function to encode objects of the specified type.
            decoder: Function to decode data back to the original type.
            alt_name: Alternative name(s) for the decoder, can be a single
                string or tuple of strings. Useful for backwards compatibility
                with older `__type__` identifiers.

        Returns:
            None

        Examples:
            By default, [`decimal.Decimal`][decimal.Decimal] does not round-trip
            cleanly through [DataZip][datazip.core.DataZip] because its default
            state representation is not preserved. We can teach
            [DataZip][datazip.core.DataZip] how to handle it by registering a pair
            of coders.

            >>> from decimal import Decimal
            >>> def encode_decimal(z, name, item):
            ...     return {"__type__": "Decimal", "items": str(item)}
            >>> def decode_decimal(z, obj):
            ...     return Decimal(obj["items"])
            >>> DataZip.register_coders(
            ...     Decimal, "Decimal", encode_decimal, decode_decimal
            ... )

            Once registered, [`Decimal`][decimal.Decimal] values can be stored and
            retrieved like any built-in type.

            >>> buffer = BytesIO()
            >>> with DataZip(buffer, "w") as z:
            ...     z["pi"] = Decimal("3.14159")
            >>> with DataZip(buffer, "r") as z:
            ...     z["pi"]
            Decimal('3.14159')
        """
        cls.ENCODERS[type_] = encoder
        if decoder is not None:
            cls.DECODERS[name] = decoder
            if alt_name is not None:
                cls.DECODERS[alt_name] = decoder

    def encode_loc_helper(self, name: str, data: Any, to_write: Any) -> str:
        """Write raw bytes to the zip under a unique name and record dedup info.

        This is the low-level helper that custom encoders call to attach a
        side-car file (e.g. `.parquet`, `.npy`, `.pkl`) to a
        [DataZip][datazip.core.DataZip] archive. If `name` is already taken
        inside the zip, a numeric prefix is added until a free name is found.
        The returned name should be stored in the encoded `dict` under
        `"__loc__"` so the matching decoder can call `self.read(...)` to
        load the payload back.

        The `data` argument's `id()` is recorded so that repeated references to
        the same Python object (across multiple keys) reuse a single side-car
        file — this is what powers DataZip's object deduplication for non-JSON
        types.

        Args:
            name: Desired filename inside the zip (e.g. `"my_array.npy"`).
            data: The original Python object being encoded. Its identity is
                used for deduplication and is not written to the archive.
            to_write: The bytes (or buffer) to write into the archive.

        Returns:
            The actual name used inside the zip. Equal to `name` if it was
            unused, or a prefixed variant such as `"0_my_array.npy"` if
            `name` was already taken.

        Examples:
            Open a DataZip in write mode and stash some bytes under a chosen
            filename. The first write keeps the name as-is.

            >>> buffer = BytesIO()
            >>> z = DataZip(buffer, "w")
            >>> z.encode_loc_helper("payload.bin", object(), b"hello")
            'payload.bin'

            Writing again under the same name yields a uniquely prefixed
            variant instead of overwriting the existing entry, so both
            payloads coexist in the archive.

            >>> z.encode_loc_helper("payload.bin", object(), b"world")
            '0_payload.bin'
            >>> sorted(z.namelist())
            ['0_payload.bin', 'payload.bin']
            >>> z.close()

            A custom encoder typically uses this helper like so, storing the
            returned name under ``"__loc__"`` for the decoder to find:

            >>> def encode_bytes(z, name, item):
            ...     return {
            ...         "__type__": "raw_bytes",
            ...         "__loc__": z.encode_loc_helper(f"{name}.bin", item, item),
            ...     }
        """
        i = 0
        new_name = name
        while new_name in self.namelist():
            new_name = f"{i}_{name}"
            i += 1
        self.writestr(new_name, to_write)
        self._ids[(id(data), type(data))] = new_name
        return new_name

    def _decode(self, obj: Any) -> Any:
        """Entry point for decoding anything."""
        if decoder := self.DECODERS.get(type(obj), None):
            return decoder(self, obj)
        raise TypeError(f"no decoder for {type(obj)} {obj}")

    def _decode_dict(self, obj: dict) -> Any:
        if "__type__" in obj:
            return self.DECODERS.get(obj["__type__"], DataZip._decode_obj)(self, obj)
        return {k: self._decode(v) for k, v in obj.items()}

    @staticmethod
    def _decode_namedtuple(_, obj):
        try:
            return _get_klass(obj["objinfo"])(**obj["items"])
        except Exception as exc:
            LOGGER.error("Namedtuple will be returned as a normal tuple, %r", exc)
            return tuple(obj["items"].values())

    def _decode_obj(self, obj, klass=None) -> Any:
        if obj["__loc__"] in self._red:
            return self._red[obj["__loc__"]]
        if klass is None:
            klass = _get_klass(obj["__type__"].split("|"))
        out_obj = klass.__new__(klass)
        state = self._decode(self._attributes["__state__"][str(obj["__loc__"])])
        # we cannot use hasattr here in case __getattr__ is defined and it throws
        # non-AttributeErrors if the object is not yet fully initialized
        if hasattr(klass, "_dzsetstate_"):
            out_obj._dzsetstate_(state)
        elif hasattr(klass, "__setstate__"):
            out_obj.__setstate__(state)
        else:
            default_setstate(out_obj, state)
        self._red[str(obj["__loc__"])] = out_obj
        return out_obj

    DECODERS: ClassVar[dict[type | str | tuple, Callable]] = {
        str: lambda _, item: item,
        int: lambda _, item: item,
        bool: lambda _, item: item,
        float: lambda _, item: item,
        NoneType: lambda _, item: item,
        dict: _decode_dict,
        list: lambda self, obj: [self._decode(v) for v in obj],
        "tuple": lambda self, obj: tuple(self._decode(v) for v in obj["items"]),
        "set": lambda _, obj: set(obj["items"]),
        "frozenset": lambda _, obj: frozenset(obj["items"]),
        "complex": lambda _, obj: complex(*obj["items"]),
        "type": lambda _, obj: _get_klass(obj["items"]),
        "defaultdict": lambda self, obj: defaultdict(
            self._decode(obj["default_factory"]), self._decode_dict(obj["items"])
        ),
        "Counter": lambda self, obj: Counter(self._decode_dict(obj["items"])),
        "dict_aslist": lambda self, obj: dict(
            self._decode(item) for item in obj["items"]
        ),
        "deque": lambda self, obj: deque([self._decode(v) for v in obj["items"]]),
        "OrderedDict": lambda self, obj: OrderedDict(self._decode_dict(obj["items"])),
        "datetime": lambda _, obj: datetime.fromisoformat(_quote_strip(obj["items"])),
        "Path": lambda _, obj: Path(_quote_strip(obj["items"])),
        "namedtuple": _decode_namedtuple,
        # LEGACY type encoding
        ("builtins", "tuple", None): lambda self, obj: tuple(
            self._decode(v) for v in obj["items"]
        ),
        ("builtins", "set", None): lambda _, obj: set(obj["items"]),
        ("builtins", "frozenset", None): lambda _, obj: frozenset(obj["items"]),
        ("builtins", "complex", None): lambda _, obj: complex(*obj["items"]),
    }

    def _encode(self, name, item) -> JSONABLE:
        """Entry point for encoding anything."""
        if encoder := self.ENCODERS.get(type(item), None):
            return encoder(self, name, item)
        if isinstance(item, tuple) and hasattr(item, "_asdict"):
            return {
                "__type__": "namedtuple",
                "items": {k: self._encode(k, v) for k, v in item._asdict().items()},
                "objinfo": _objinfo(item),
            }
        if self._ids_for_dedup and (loc := self._ids.get((id(item), type(item)), None)):
            return {"__type__": _objinfo(item), "__loc__": loc}
        return self._encode_obj(name, item)

    def _encode_dict(self, _, data: dict) -> dict:
        # we need to encode the dict differently if any keys are not int | str
        if set(map(type, data.keys())) - {str}:
            return {
                "__type__": "dict_aslist",
                "items": [self._encode(_, item) for _, item in enumerate(data.items())],
            }
        # ecode then filter
        return {
            k: v
            for k, v in {k_: self._encode(k_, v_) for k_, v_ in data.items()}.items()
            if v != "__IGNORE__"
        }

    def _encode_obj(self, name: str, item: Any) -> dict:
        klass = item.__class__
        if hasattr(klass, "_dzgetstate_"):
            state = item._dzgetstate_()
        elif hasattr(klass, "__getstate__"):
            state = item.__getstate__()
        elif hasattr(item, "__dict__") or hasattr(item, "__slots__"):
            state = default_getstate(item)
        else:
            raise TypeError(f"no encoder for {type(item)}")

        if name in self._attributes["__state__"]:
            name = f"{id(item)}_{name}"

        self._ids[(id(item), klass)] = name
        self._attributes["__state__"][name] = self._encode("state", state)
        return {
            "__type__": _objinfo(item),
            "__loc__": name,
            "__obj_version__": _get_version(item),
            "__io_version__": __version__,
            "__created_by__": _get_username(),
            "__file_created__": str(datetime.now(tz=ZoneInfo("UTC"))),
        }

    ENCODERS: ClassVar[dict[type, Callable]] = {
        str: lambda _, __, item: item,
        int: lambda _, __, item: item,
        bool: lambda _, __, item: item,
        float: lambda _, __, item: item,
        NoneType: lambda _, __, item: item,
        list: lambda self, _, item: [self._encode(i, e) for i, e in enumerate(item)],
        tuple: lambda self, _, item: {
            "__type__": "tuple",
            "items": [self._encode(i, e) for i, e in enumerate(item)],
        },
        dict: _encode_dict,
        set: lambda self, _, item: {
            "__type__": "set",
            "items": [self._encode(i, e) for i, e in enumerate(item)],
        },
        frozenset: lambda self, _, item: {
            "__type__": "frozenset",
            "items": [self._encode(i, e) for i, e in enumerate(item)],
        },
        complex: lambda _, __, item: {
            "__type__": "complex",
            "items": [item.real, item.imag],
        },
        type: lambda self, __, item: {
            "__type__": "type",
            "items": [item.__module__, item.__qualname__],
        },
        defaultdict: lambda self, __, item: {
            "__type__": "defaultdict",
            "items": self._encode_dict(__, item),
            "default_factory": self._encode(__, item.default_factory),
        },
        Counter: lambda self, __, item: {
            "__type__": "Counter",
            "items": self._encode_dict(__, item),
        },
        deque: lambda self, __, item: {
            "__type__": "deque",
            "items": [self._encode(_, e) for _, e in enumerate(item)],
        },
        OrderedDict: lambda self, __, item: {
            "__type__": "OrderedDict",
            "items": self._encode_dict(__, item),
        },
        datetime: lambda _, __, item: {"__type__": "datetime", "items": str(item)},
        Path: lambda _, __, item: {"__type__": "Path", "items": str(item)},
        PosixPath: lambda _, __, item: {"__type__": "Path", "items": str(item)},
        WindowsPath: lambda _, __, item: {"__type__": "Path", "items": str(item)},
        # things to ignore
        partial: _encode_ignore,
    }

    def _load_legacy_helper(self) -> dict:
        obj_meta = self._metadata.get("obj_meta", self._json_get("obj_meta"))
        locs = []

        def _make_attr_entry(k_, locs_):
            _attr = {}
            if (objinfo := tuple(obj_meta.get(k_, ""))) in self.DECODERS:
                _attr.update({"__type__": objinfo})
                if (
                    file_ := "".join((k_, self.suffixes.get(objinfo, "")))
                ) in self.namelist():
                    _attr.update({"__loc__": file_})
                    locs_.append(file_)
                elif k_ in self._attributes:
                    _attr.update({"items": self._attributes[k_]})
                if k_ in _no_pqt_cols:
                    _attr.update({"no_pqt_cols": _no_pqt_cols[k_]})
            else:
                _attr = self._attributes.get(k_, {})
            return _attr

        _no_pqt_cols = self._metadata.get(
            "no_pqt_cols", self._metadata.get("bad_cols", self._json_get("bad_cols"))
        )
        contents = self._metadata.get("contents", self._json_get("contents"))
        attrs = {}
        for k, sub_k in contents.items():
            if sub_k:
                LOGGER.warning(
                    "Unable to load nested structures, %s and its children will "
                    "not be accessible",
                    k,
                )

            attr = _make_attr_entry(k, locs)
            if attr:
                attrs.update({k: attr})

        for file in self.namelist():
            stem, suffix = file.split(".")
            if file not in locs and suffix == "parquet":
                bc = {"no_pqt_cols": _no_pqt_cols[stem]} if stem in _no_pqt_cols else {}
                attrs.update({stem: {"__type__": "pdDataFrame", "__loc__": file} | bc})
            if file not in locs and suffix == "npy":
                attrs.update({stem: {"__type__": "ndarray", "__loc__": file}})

        return attrs

    def _json_get(self, *args):
        for arg in args:
            try:
                return json.loads(self.read(f"{arg}.json"))
            except Exception:  # noqa: S110
                pass
        return {}

    def __repr__(self):
        return self.__class__.__qualname__ + f"(file={self.filename}, mode={self.mode})"
