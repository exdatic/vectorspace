import itertools
import json
import mmap
import os
import pickle
from random import shuffle
from typing import AsyncIterable, Dict, Iterable, List, TypeVar, Union

import orjson

from vectorspace.mapping import PickleList

__all__ = ['json_load', 'json_dump', 'readlines', 'writelines', 'randomize']


def expand_path(path: str):
    return os.path.expandvars(os.path.expanduser(path))


def readlines(path, binary=False, skip_rows=0):
    path = expand_path(path)
    with open(path, 'rb' if binary else 'r', encoding=None if binary else 'utf-8') as f:
        for i, line in enumerate(f):
            if i >= skip_rows:
                yield line.rstrip()


def readlines_backwards(path: str):
    with open(path, mode='r') as file:
        mmap_file = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
        mmap_file.seek(0, os.SEEK_END)

        end = mmap_file.tell()
        pos = end
        while pos > 0:
            pos = mmap_file.rfind(b'\n', 0, end - 1)
            if end - pos:
                mmap_file.seek(pos + 1)
                line = mmap_file.read(end - pos - 1)
                yield line
            end = pos


def writelines(lines, path, binary=False):
    path = expand_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb' if binary else 'w', encoding=None if binary else 'utf-8') as f:
        for i, line in enumerate(lines):
            if (i > 0):
                f.write(b'\n' if binary else '\n')
            f.write(line)


async def async_enumerate(asequence, start=0):
    """Asynchronously enumerate an async iterator from a given start value"""
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1


async def async_readlines(path, binary=False):
    with open(path, 'rb' if binary else 'r', encoding=None if binary else 'utf-8') as f:
        for line in f:
            yield line.rstrip()


async def async_writelines(lines: AsyncIterable[Union[str, bytes]], path, binary=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb' if binary else 'w', encoding=None if binary else 'utf-8') as f:
        async for i, line in async_enumerate(lines):
            if (i > 0):
                f.write(b'\n' if binary else '\n')
            f.write(line)


def json_load(path):
    path = expand_path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_dump(obj, path, **kwargs):
    path = expand_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, **kwargs)


def jsonl_load(path):
    """Load objects from line separated file."""
    yield from map(orjson.loads, readlines(path, binary=True))


def jsonl_dump(objs, path):
    """Dump objects to line separated file."""
    writelines(map(orjson.dumps, objs), path, binary=True)


def jsonl_load_items(path):
    """Load items of an object from line separated file.

    Example
    -------
    The following::
        {"foo": 1}
        {"bar": 2}

    will be read as::
        {"foo": 1, "bar": 2}

    Returns
    -------
    dict
        the object
    """
    return dict(item for obj in jsonl_load(path) for item in obj.items())


def jsonl_dump_items(obj, path):
    """Dump items of an object to line separated file.

    Example
    -------
    The following::
        {"foo": 1, "bar": 2}

    will be written as::
        {"foo": 1}
        {"bar": 2}

    Parameters
    ----------
    obj : dict
        an object
    """
    jsonl_dump(itertools.starmap(lambda k, v: {k: v}, obj.items()), path)


def iter_jsonl_backwards(path: str):
    for line in readlines_backwards(path):
        yield orjson.loads(line)


async def async_jsonl_load(path):
    async for line in async_readlines(path):
        yield orjson.loads(line)


async def async_jsonl_dump(objs: AsyncIterable[Dict], path):
    """Dump objects to line separated file."""
    await async_writelines((orjson.dumps(o) async for o in objs), path, binary=True)


def pickle_load(path):
    path = expand_path(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_dump(obj, path, **kwargs):
    path = expand_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        return pickle.dump(obj, f, **kwargs)


def pickle_dump_iter(path: Union[str, os.PathLike], iterable: Iterable, subdir=False, compress=False):
    with PickleList(path, subdir=subdir, compress=compress) as db:
        db.update((i, v) for i, v in enumerate(iterable))


def pickle_load_iter(path: Union[str, os.PathLike], subdir=False):
    with PickleList(path, subdir=subdir) as db:
        yield from db.values()


T = TypeVar('T')


def take(n: int, iterable: Iterable[T]) -> List[T]:
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def chunked(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    "Break iterable into chunks of length n"
    it = iter(iterable)
    chunk = take(n, it)
    while chunk != []:
        yield chunk
        chunk = take(n, it)


def randomize(iterable: Iterable[T], bufsize: int = 1000) -> Iterable[T]:
    "Naïve shuffle algorithm"
    for chunk in chunked(iterable, bufsize):
        shuffle(chunk)
        yield from chunk


def humantime(t: float) -> str:
    """Formats time into a compact human readable format

    Parameters
    ----------
    t : float
        number of seconds
    """
    times = {}
    units = {'y': 31536000, 'w': 604800, 'd': 86400, 'h': 3600, 'm': 60, 's': 1}
    for i, (unit, seconds) in enumerate(units.items()):
        if t // seconds > 0:
            times[unit] = int(t//seconds)
            t -= t//seconds * seconds
    if not times:
        if int(t * 1000) > 0:
            times['ms'] = int(t * 1000)
        else:
            return '0s'
    return ''.join(f'{v}{u}' for u, v in times.items())


def humansize(s: int) -> str:
    """Formats byte size into a compact human readable format

    Parameters
    ----------
    s : int
        number of bytes
    """
    sizes = {}
    units = {'T': 0x10000000000, 'G': 0x40000000, 'M': 0x100000, 'k': 0x400, 'b': 1}
    for i, (unit, size) in enumerate(units.items()):
        if s // size > 0:
            sizes[unit] = s/size
            s -= s//size * size
    if 'b' not in sizes:
        return '0b'
    u, v = max(sizes.items(), key=lambda i: i[1]*units[i[0]])
    return f'{int(v)}{u}' if int(v) == v else f'{v:.2f}{u}'
