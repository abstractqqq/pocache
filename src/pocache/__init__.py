from __future__ import annotations

# import numpy as np # No need to import but must be installed
import polars as pl
import pickle
import logging
import time
import xxhash
import hashlib
import tempfile as tmp
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class MODE(Enum):
    MEM = auto()
    TEMP = auto()

    @staticmethod
    def from_str(mode: str):
        _mode = mode.lower()
        if _mode in ("mem", "memory"):
            return MODE.MEM
        elif _mode in ("temp", "tempfile"):
            return MODE.TEMP
        else:
            raise ValueError(f"Input `{mode}` is not supported yet.")

class Session():

    __slots__ = ("_mode", "_verbose", "_lookup", "_secure", "_temp_dir")

    def __init__(self, mode: str | MODE = MODE.MEM, verbose: bool = False, secure: bool = False):
        self._mode : MODE = MODE.from_str(mode) if isinstance(mode, str) else MODE
        self._verbose : bool = verbose
        self._lookup : Dict[str : Any] = {}
        self._secure : bool = secure

        # Resolve optional variables
        self._temp_dir : tmp.TemporaryDirectory | None = None
        if self._mode == MODE.TEMP:
            self._temp_dir = tmp.TemporaryDirectory(prefix=".pocache-")

    def __del__(self):
        self._lookup.clear()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def _get_hash(self):
        if self._secure:
            return hashlib.sha256()
        else:
            return xxhash.xxh3_64()

    def cache_count(self) -> int:
        return len(self._lookup)

    def remove_first(self) -> None:
        """Removes the first item in the cache.
        """
        if len(self._lookup) > 0:
            first_key = next(iter(self._lookup))
            _ = self._lookup.pop(first_key)

    def remove_last(self) -> None:
        """Removes the last item in the cache.
        """
        if len(self._lookup) > 0:
            _ = self._lookup.popitem()

    def pocache(
        self
        , _func = None
        , *
        , use_col_names: bool = True
        , serializer: Callable | None = None
        , deserializer: Callable | None = None
    ) -> Any:
        """
        Caches the functions which takes Polars dataframe inputs.

        Parameters
        ----------
        use_col_names
            Whether to use column names as part of the hash.
        serializer
            If mode != MODE.MEM, then the serializer will write the result at the given path.
            The serializer callable must accept (result, path) as its two arguments.
        deserializer
            If mode != MODE.MEM, then the deserializer will load the cached result from the cache location.
            The deserializer callable must accept (path) as its input argument.
        """
        if self._mode != MODE.MEM and (serializer is None or deserializer is None):
            raise ValueError("When mode != MODE.MEM, you must pass `serializer` and `deserializer`.")

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):

                if any(not isinstance(df, (pl.LazyFrame, pl.DataFrame)) for df in args):
                    raise TypeError("Positional arguments must be polars dataframes, lazy or eager.")

                # Start hashing
                hash_start = time.perf_counter()
                h = self._get_hash()
                # Hash bytecode of func
                h.update(func.__code__.co_code)
                # Params hash
                h.update(pickle.dumps(sorted(kwargs.items()), fix_imports=False))
                # Hash dfs. Different strategy by size
                for df in args:
                    length = df.lazy().select(pl.len()).collect().item(0,0)
                    if length <= 2000: # tiny
                        h.update(df.serialize(format="binary"))
                    elif length < 2**32: # medium - large
                        # columns
                        cols = sorted(df.collect_schema().names())
                        # Column order should not matter
                        # Two dfs can have the same values, but different column names.
                        # If you care about that, use_col_names = True
                        if use_col_names:
                            h.update(len(cols).to_bytes(length=16)) # lol you won't have this many columns
                            h.update(pickle.dumps(cols))
        
                        # Calculate row hash
                        h.update(
                            df.lazy().select(cols).collect()
                            .hash_rows(seed=996).to_numpy()
                            .data
                        ) # use memoryview instead of id_hash.tobytes()
                    else:
                        raise ValueError("Data has too many rows (>= 2^32).")

                hash_key = h.hexdigest()
                hash_time = time.perf_counter() - hash_start

                # Check if cache exists
                deser_start = time.perf_counter()
                cached_result = None
                match self._mode:
                    case MODE.MEM:
                        cached_result = self._lookup.get(hash_key, None)
                    case MODE.TEMP:
                        cache_path = self._lookup.get(hash_key, None)
                        if cache_path:
                            cached_result = deserializer(cache_path)
                    case _:
                        raise NotImplementedError

                if cached_result is not None:
                    deser_time = time.perf_counter() - deser_start
                    if self._verbose:
                        logger.info(f"Cache hit. Hashing took: {hash_time:4f}s. Retrieval took: {deser_time:4f}s.")
                    return cached_result

                # Reach here if cache does not exist
                func_start = time.perf_counter()
                result = func(*args, **kwargs)
                func_time = time.perf_counter() - func_start

                # Save cache
                save_start = time.perf_counter()
                match self._mode:
                    case MODE.MEM:
                        self._lookup[hash_key] = result
                    case MODE.TEMP:
                        path = Path(self._temp_dir.name) / hash_key
                        serializer(result, path)
                        self._lookup[hash_key] = path
                    case _:
                        raise NotImplementedError

                if self._verbose:
                    save_time = time.perf_counter() - save_start
                    if func_time < hash_time:
                        logger.warning(
                            f"Hashing took {hash_time:4f}s but running the function "
                            f"only took {func_time:4f}s. Please reconsider."
                        )
                    else:
                        logger.info(
                            f"Hashing took: {hash_time:4f}s. The function took: {func_time:4f}s. Saving cache took: {save_time:4f}s."
                        )

                return result

            return wrapper

        # If _func is None, the decorator was called with parentheses
        if _func is None:
            return decorator
        return decorator(_func)