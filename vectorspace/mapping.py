import codecs
from collections.abc import Mapping as MappingABC, MutableMapping as MutableMappingABC, Sequence as SequenceABC
from typing import Callable, MutableMapping, Optional, Tuple

import cloudpickle
import orjson
import zstandard

from vectorspace.lmdb import LMDB

__all__ = ['KeyValueMapping', 'LMDBDict', 'StringDict', 'JsonDict', 'PickleDict', 'PickleList']


def zstdio(compress: bool, **kwargs):
    # zstd is used over brotli because it decompresses twice as fast
    compress_level = kwargs.pop('compress_level', 6)
    zdump = zstandard.ZstdCompressor(level=compress_level).compress
    zload = zstandard.ZstdDecompressor().decompress
    return (
        lambda v: zdump(v) if compress else v,
        lambda b: zload(b) if b.startswith(zstandard.FRAME_HEADER) else b  # auto decompress
    )


class KeyValueMapping(MutableMappingABC):

    def __init__(self, wrapped: MutableMapping,
                 key_fn: Optional[Tuple[Callable, Callable]] = None,
                 val_fn: Optional[Tuple[Callable, Callable]] = None):
        self._wrapped = wrapped
        self._dump_key, self._load_key = key_fn if key_fn else (lambda v: v, lambda v: v)
        self._dump_val, self._load_val = val_fn if val_fn else (lambda v: v, lambda v: v)

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            # handle list of keys
            key = [self._dump_key(k) for k in key]
            return [self._load_val(v) for v in self._wrapped[key]]

        elif isinstance(key, (slice, range)):
            # handle slice of keys
            key = slice(
                self._dump_key(key.start) if key.start is not None else None,
                self._dump_key(key.stop) if key.stop is not None else None, key.step)
            return [self._load_val(v) for v in self._wrapped[key]]

        else:
            # handle rest (i.e. key only)
            return self._load_val(self._wrapped[self._dump_key(key)])

    def __contains__(self, key):
        return self._dump_key(key) in self._wrapped

    def keys(self):
        return (self._load_key(k) for k in self._wrapped.keys())

    def items(self):
        return ((self._load_key(k), self._load_val(v)) for k, v in self._wrapped.items())

    def values(self):
        return (self._load_val(v) for v in self._wrapped.values())

    def __iter__(self):
        return self.keys()

    def __setitem__(self, key, value):
        self._wrapped[self._dump_key(key)] = self._dump_val(value)

    def __delitem__(self, key):
        del self._wrapped[self._dump_key(key)]

    def update(self, other=()):
        if isinstance(other, MappingABC):
            items = other.items()
        elif hasattr(other, "keys"):
            items = ((k, other[k]) for k in other.keys())
        else:
            items = other
        return self._wrapped.update((self._dump_key(k), self._dump_val(v)) for k, v in items)

    def __len__(self):
        return len(self._wrapped)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if hasattr(self._wrapped, 'close'):
            self._wrapped.close()  # type: ignore


class LMDBDict(KeyValueMapping):
    """
    LMDB based dictionary with optional compression (Zstandard).
    Allows bytes as value.
    """

    def __init__(self, path, subdir=False, compress=False, **kwargs):
        dump, load = zstdio(compress, **kwargs)
        super().__init__(LMDB(path, subdir=subdir, **kwargs),
                         key_fn=(codecs.encode, codecs.decode),
                         val_fn=(lambda v: dump(v),
                                 lambda b: load(b)))


class StringDict(KeyValueMapping):
    """
    LMDB based dictionary with optional compression (Zstandard).
    Allows strings as value.
    """

    def __init__(self, path, subdir=False, compress=False, **kwargs):
        dump, load = zstdio(compress, **kwargs)
        super().__init__(LMDB(path, subdir=subdir, **kwargs),
                         key_fn=(codecs.encode, codecs.decode),
                         val_fn=(lambda v: dump(v.encode()),
                                 lambda b: load(b).decode()))


class JsonDict(KeyValueMapping):
    """
    LMDB based dictionary with optional compression (Zstandard).
    Allows any JSON serializable object as value.
    """

    def __init__(self, path, subdir=False, compress=False, **kwargs):
        dump, load = zstdio(compress, **kwargs)
        super().__init__(LMDB(path, subdir=subdir, **kwargs),
                         key_fn=(codecs.encode, codecs.decode),
                         val_fn=(lambda v: dump(orjson.dumps(v)),
                                 lambda b: orjson.loads(load(b))))


class PickleDict(KeyValueMapping):
    """
    LMDB based dictionary with optional compression (Zstandard).
    Allows any pickable object as value.
    """

    def __init__(self, path, subdir=False, compress=False, **kwargs):
        dump, load = zstdio(compress, **kwargs)
        super().__init__(LMDB(path, subdir=subdir, **kwargs),
                         key_fn=(codecs.encode, codecs.decode),
                         val_fn=(lambda v: dump(cloudpickle.dumps(v)),
                                 lambda b: cloudpickle.loads(load(b))))


class PickleList(KeyValueMapping, SequenceABC):
    """
    LMDB based list with optional compression (Zstandard).
    Allows any pickable object as value.
    """

    def __init__(self, path, subdir=False, compress=False, **kwargs):
        dump, load = zstdio(compress, **kwargs)
        super().__init__(LMDB(path, subdir=subdir, **kwargs),
                         key_fn=(lambda v: int.to_bytes(int(v), 4, 'big'),
                                 lambda v: int.from_bytes(v, 'big')),
                         val_fn=(lambda v: dump(cloudpickle.dumps(v)),
                                 lambda b: cloudpickle.loads(load(b))))

    def __getitem__(self, index):
        def _neg2pos(i):
            return len(self) + i if i < 0 else i

        if isinstance(index, (slice, range)):
            # handle slice: convert negative indices
            index = slice(
                _neg2pos(index.start) if index.start is not None else None,
                _neg2pos(index.stop) if index.stop is not None else None, index.step)
            return super().__getitem__(index)

        elif isinstance(index, int):
            # handle index: convert negative indices
            return super().__getitem__(_neg2pos(index))

        else:
            raise TypeError(f'list indices must be integers, tuples or slices, not {type(index)}')

    def __iter__(self):
        yield from self.values()
