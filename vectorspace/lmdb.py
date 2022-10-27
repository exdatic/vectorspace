from collections.abc import Mapping as MappingABC, MutableMapping as MutableMappingABC
import os
import sys
from typing import Union

import lmdb

__all__ = ['LMDB']


class LMDB(MutableMappingABC):

    def __init__(self, path: Union[str, os.PathLike], subdir=False, **kwargs):
        # Maximum size database may grow to; used to size the memory mapping.
        # If database grows larger than map_size, an exception will be raised
        # and the user must close and reopen Environment. On 64-bit there is
        # no penalty for making this huge (say 1TB). Must be <2GB on 32-bit.
        map_size = kwargs.pop('map_size', 1 << 40 if sys.maxsize >= 1 << 32 else 1 << 28)

        # Disable locking to prevent BadRslotError
        lock = kwargs.pop('lock', False)

        self.env = lmdb.open(os.fspath(path), subdir=subdir, lock=lock, map_size=map_size, **kwargs)
        self.max_key_size = self.env.max_key_size()

    def __getitem__(self, key):
        with self.env.begin() as txn:
            # handle list of keys, like db['foo', 'bar']
            if isinstance(key, (tuple, list)):
                def _iter_keys(keys):
                    with txn.cursor() as cursor:
                        for key in keys:
                            if cursor.set_range(key) and cursor.key() == key:
                                yield cursor.value()
                            else:
                                raise KeyError(key)
                return list(_iter_keys(key))

            # handle slice of keys, like db['foo':'bar']
            elif isinstance(key, (slice, range)):
                def _iter_slice(slice_):
                    assert slice_.step is None, 'slice step must be None'
                    with txn.cursor() as cursor:
                        if slice_.start is not None:
                            cursor.set_range(slice_.start)
                        for key, value in cursor.iternext(keys=True, values=True):
                            if slice_.stop is None or key < slice_.stop:
                                yield value
                            else:
                                break
                return list(_iter_slice(key))

            else:
                value = txn.get(key)
                if value is not None:
                    return value
                else:
                    raise KeyError(key)

    def __contains__(self, key):
        with self.env.begin() as txn:
            return txn.cursor().set_key(key)

    def keys(self):
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                yield from cursor.iternext(keys=True, values=False)

    def items(self, keys=None):
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                yield from cursor.iternext(keys=True, values=True)

    def values(self, keys=None):
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                yield from cursor.iternext(keys=False, values=True)

    def __iter__(self):
        return self.keys()

    def _ensure_max_key_size(self, key):
        return key[:self.max_key_size]

    def __setitem__(self, key, value):
        with self.env.begin(write=True) as txn:
            txn.put(self._ensure_max_key_size(key), value)

    def __delitem__(self, key):
        with self.env.begin(write=True) as txn:
            if not txn.delete(key):
                raise KeyError(key)

    def update(self, other=()):
        if isinstance(other, MappingABC):
            items = other.items()
        elif hasattr(other, "keys"):
            items = ((k, other[k]) for k in other.keys())
        else:
            items = other
        with self.env.begin(write=True) as txn:
            with txn.cursor() as cursor:
                return cursor.putmulti((self._ensure_max_key_size(k), v) for k, v in items)

    def copy(self, path: Union[str, os.PathLike], compact=True):
        self.env.copy(path, compact=compact)

    def __len__(self):
        return self.env.stat()["entries"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.env.close()
