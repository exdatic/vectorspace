import asyncio
from threading import Lock, RLock
from typing import Any, MutableMapping, TypeVar
from weakref import WeakValueDictionary

__all__ = ['KeyedLocks', 'AsyncKeyedLocks']

T = TypeVar("T")


class KeyedLocks:

    _keys: MutableMapping[Any, RLock]
    _lock: Lock

    def __init__(self):
        self._keys = WeakValueDictionary()
        self._lock = Lock()

    def __getitem__(self, key: Any) -> RLock:
        return self.get(key)

    def get(self, key: Any) -> RLock:
        with self._lock:
            if key in self._keys:
                lock = self._keys[key]
            else:
                lock = RLock()
                self._keys[key] = lock
            return lock


class AsyncKeyedLocks:

    _keys: MutableMapping[Any, asyncio.Lock]
    _lock: asyncio.Lock

    def __init__(self):
        self._keys = WeakValueDictionary()
        self._lock = asyncio.Lock()

    async def get(self, key: Any) -> asyncio.Lock:  # type: ignore
        async with self._lock:
            if key in self._keys:
                lock = self._keys[key]
            else:
                lock = asyncio.Lock()
                self._keys[key] = lock
            return lock
