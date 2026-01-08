
from __future__ import annotations

# import numpy as np # No need to import but must be installed
import polars as pl
import pickle
import logging
import time
import xxhash
import hashlib
from enum import Enum, auto
from typing import Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class MODE(Enum):
    MEM = auto()

    @staticmethod
    def from_str(mode: str):
        if mode.lower() in ("mem", "memory"):
            return MODE.MEM
        else:
            raise ValueError(f"Input `{mode}` is not supported yet.")

class Session():

    __slots__ = ("mode", "verbose", "storage", "secure")

    def __init__(self, mode: str | MODE = MODE.MEM, verbose: bool = False, secure: bool = False):
        self.mode = MODE.from_str(mode) if isinstance(mode, str) else MODE
        self.verbose = verbose
        self.storage = {}
        self.secure = secure

    def _get_hash(self):
        if self.secure:
            return hashlib.sha256()
        else:
            return xxhash.xxh3_64()

    def cache_count(self) -> int:
        return len(self.storage)

    def remove_first(self) -> None:
        """Removes the first item in the cache.
        """
        if len(self.storage) > 0:
            first_key = next(iter(self.storage))
            _ = self.storage.pop(first_key)

    def remove_last(self) -> None:
        """Removes the last item in the cache.
        """
        if len(self.storage) > 0:
            _ = self.storage.popitem()

    def pocache(
        self
        , _func = None
        , *
        , prefix: str = ""
        , use_col_names: bool = True
        , cache_serializer: Callable | None = None
        , cache_deserializer: Callable | None = None
    ) -> Any:
        """
        Caches the functions which takes Polars dataframe inputs.

        Parameters
        ----------
        prefix
            Not yet implemented
        use_col_names
            Whether to use column names as part of the hash.
        cache_serializer
            Not yet implemented
        cache_deserializer
            Not yet implemented
        """

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
                        # Two dfs can be the same values, but different column names.
                        # If you care about that, use_col_names = True
                        if use_col_names:
                            h.update(len(cols).to_bytes(length=16)) # lol you won't have this many columns
                            h.update(("".join(cols)).encode("utf-8"))
        
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
                cached_result = None
                match self.mode:
                    case MODE.MEM:
                        cached_result = self.storage.get(hash_key, None)
                    case _:
                        raise NotImplementedError

                if cached_result is not None:
                    if self.verbose:
                        logger.info(f"Cache hit. Hashing took: {hash_time:4f}s.")
                    return cached_result

                # Reach here if cache does not exist
                func_start = time.perf_counter()
                result = func(*args, **kwargs)
                func_time = time.perf_counter() - func_start

                # Save cache
                match self.mode:
                    case MODE.MEM:
                        self.storage[hash_key] = result
                    case _:
                        raise NotImplementedError

                if self.verbose:
                    if func_time < hash_time:
                        logger.warning(
                            f"Hashing took {hash_time:4f}s but running the function "
                            f"only took {func_time:4f}s. Please reconsider."
                        )
                    else:
                        logger.info(f"Hashing took: {hash_time:4f}s. The function took: {func_time:4f}s.")

                return result

            return wrapper

        # If _func is None, the decorator was called with parentheses
        if _func is None:
            return decorator
        return decorator(_func)