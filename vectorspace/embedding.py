import codecs
from collections.abc import Sequence as SequenceABC
import inspect
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator

from vectorspace.lmdb import LMDB
from vectorspace.mapping import KeyValueMapping
from vectorspace.math import knn, norm, unitvec

__all__ = ['Embedding', 'load_embedding']


class IntToByteStrDict(KeyValueMapping):

    def __init__(self, path: Union[str, Path]):
        super().__init__(LMDB(path, subdir=False),
                         key_fn=(lambda v: int.to_bytes(int(v), 4, 'big'),
                                 lambda v: int.from_bytes(v, 'big')))


class ByteStrToIntDict(KeyValueMapping):

    def __init__(self, path: Union[str, Path]):
        super().__init__(LMDB(path, subdir=False),
                         val_fn=(lambda v: int.to_bytes(int(v), 4, 'big'),
                                 lambda v: int.from_bytes(v, 'big')))


class IntToStrDict(KeyValueMapping, SequenceABC):

    def __init__(self, path: Union[str, Path]):
        super().__init__(LMDB(path, subdir=False, readonly=True),
                         key_fn=(lambda v: int.to_bytes(int(v), 4, 'big'),
                                 lambda v: int.from_bytes(v, 'big')),
                         val_fn=(codecs.encode, codecs.decode))

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


class StrToIntDict(KeyValueMapping):

    def __init__(self, path: Union[str, Path]):
        super().__init__(LMDB(path, subdir=False, readonly=True),
                         key_fn=(codecs.encode, codecs.decode),
                         val_fn=(lambda v: int.to_bytes(int(v), 4, 'big'),
                                 lambda v: int.from_bytes(v, 'big')))


def dump_vocab(words: Iterable[bytes],
               output_dir: Union[str, Path],
               words_file='words.mdb',
               index_file='index.mdb'):
    output_dir = Path(output_dir)
    with IntToByteStrDict(output_dir / words_file) as i2w, ByteStrToIntDict(output_dir / index_file) as w2i:
        i2w.update((i, w) for i, w in enumerate(words))
        w2i.update((w, i) for i, w in enumerate(i2w.values()))


def load_vocab(output_dir: Union[str, Path],
               words_file='words.mdb',
               index_file='index.mdb'
               ) -> Tuple[Sequence[str], Mapping[str, int]]:
    output_dir = Path(output_dir)
    return IntToStrDict(output_dir / words_file), StrToIntDict(output_dir / index_file)


def load_embedding(input_dir: Union[str, Path]) -> 'Embedding':
    return Embedding.load(input_dir)


class WordView(Sequence[str]):
    """List of words with fancy indexing"""

    def __init__(self, words: Sequence[str]):
        self._words = words

    def __getitem__(self, index):
        if isinstance(index, (tuple, list, np.ndarray)):
            # handle list of indices or mask
            idcs = cast(np.ndarray, np.asarray(index))
            if np.issubdtype(idcs.dtype, np.bool_):
                idcs, *_ = np.where(idcs)

            # opt: convert indices to slice if possible
            diff = np.diff(idcs)
            if len(diff) and np.all(diff == 1):
                return self._words[slice(idcs[0], idcs[-1] + 1)]

            return [self._words[int(i)] for i in idcs]

        else:
            # delegate rest (i.e. index and slice)
            return self._words[index]

    def __iter__(self):
        yield from self._words

    def __len__(self):
        return len(self._words)


class IndexView(Mapping[str, int]):
    """Dictionary of words and indices with fancy indexing"""

    def __init__(self, index: Mapping[str, int]):
        self._index = index

    def __getitem__(self, key):
        if isinstance(key, (tuple, list, np.ndarray)):
            # handle list of keys
            return [self._index[k] for k in key]

        elif isinstance(key, (slice, range)):
            # handle slice of keys
            def _iter_slice(slice_):
                assert slice_.step is None, 'slice step must be None'
                for key in self._index.keys():
                    if slice_.start is not None and key < slice_.start:
                        continue
                    if slice_.stop is None or key < slice_.stop:
                        yield self._index[key]
                    else:
                        break
            if isinstance(self._index, KeyValueMapping):
                return self._index[key]
            else:
                return list(_iter_slice(key))

        elif inspect.isfunction(key):
            # handle lambda expressions
            return [self._index[k] for k in self._index.keys() if key(k)]

        else:
            # delegate rest (i.e. key only)
            return self._index[key]

    def __iter__(self):
        yield from self._index

    def __len__(self):
        return len(self._index)


class Embedding:

    def __init__(self,
                 vecs: np.ndarray,
                 norms: Optional[np.ndarray],
                 words: Sequence[str],
                 index: Mapping[str, int],
                 counts: Optional[np.ndarray] = None):
        self._vecs = vecs
        self._norms = norms
        self._words = WordView(words) if not isinstance(words, WordView) else words
        self._index = IndexView(index) if not isinstance(index, IndexView) else index
        self._counts = counts

    @property
    def vecs(self) -> np.ndarray:
        return self._vecs

    @property
    def norms(self) -> Optional[np.ndarray]:
        return self._norms

    @property
    def index(self) -> IndexView:
        return self._index

    @property
    def words(self) -> WordView:
        return self._words

    @property
    def counts(self) -> Optional[np.ndarray]:
        return self._counts

    @property
    def shape(self):
        return self.vecs.shape

    @property
    def unitvecs(self) -> np.ndarray:
        return (self._vecs / self._norms[:, None])  # type: ignore

    def unitvec(self, word) -> np.ndarray:
        assert self.norms is not None, 'norms must not be None'
        return self[word] / self.norms[self.index[word]]

    def __getitem__(self, word) -> np.ndarray:
        return self.vecs[self.index[word]]

    def __contains__(self, word):
        return word in self.index

    def __len__(self):
        return len(self.vecs)

    def dump(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dump_vocab((w.encode() for w in self.words), output_dir)
        np.save(output_dir / 'vecs', self.vecs)
        if self.norms is not None:
            np.save(output_dir / 'norms', self.norms)
        if self.counts is not None:
            np.save(output_dir / 'counts', self.counts)

    @classmethod
    def load(cls, input_dir: Union[str, Path]) -> 'Embedding':
        input_dir = Path(input_dir)
        vecs = np.load(input_dir / 'vecs.npy', mmap_mode='r')

        if (input_dir / 'norms.npy').is_file():
            norms = np.load(input_dir / 'norms.npy', mmap_mode='r')
        else:
            norms = None
        if (input_dir / 'counts.npy').is_file():
            counts = np.load(input_dir / 'counts.npy', mmap_mode='r')
        else:
            counts = None

        words, index = load_vocab(input_dir)
        return cls(vecs, norms, words, index, counts=counts)

    @classmethod
    def create(cls,
               vecs: np.ndarray,
               words: Sequence[str],
               counts: Optional[np.ndarray] = None
               ) -> 'Embedding':
        norms = cast(np.ndarray, norm(vecs))
        index = dict((w, i) for i, w in enumerate(words))
        return cls(vecs, norms, words, index, counts=counts)

    def reduce(self, reducer: BaseEstimator) -> 'Embedding':
        assert hasattr(reducer, 'fit_transform'), 'reducer has no fit_transform member'
        # always normalize vectors before dimensionality reduction
        if self.norms is not None:
            vecs = self.vecs / self.norms[..., np.newaxis]

        vecs = reducer.fit_transform(vecs)  # type: ignore

        # always normalize vectors after dimensionality reduction
        norms = cast(np.ndarray, norm(vecs))
        vecs /= norms[..., np.newaxis]

        # set norm to 1.0
        norms = np.ones(len(self), dtype=np.float32)
        return Embedding(vecs, norms, self.words, self.index, counts=self.counts)

    def filter(self, indices: Sequence[Union[int, bool]]) -> 'Embedding':
        # handle list of indices or mask
        idcs = np.asarray(indices)
        if np.issubdtype(idcs.dtype, np.bool_):
            idcs, = np.where(idcs)

        diff = np.diff(idcs)
        assert len(diff) <= 1 or np.any(diff > 0), 'indices must be sequential'

        # opt: convert indices to slice if possible
        if np.all(diff == 1):
            idcs = slice(idcs[0], idcs[-1] + 1)

        vecs = self.vecs[idcs]
        norms = self.norms[idcs] if self.norms is not None else None
        words = self.words[idcs]
        index = dict((w, i) for i, w in enumerate(words))
        counts = self.counts[idcs] if self.counts is not None else None

        return Embedding(vecs, norms, words, index, counts=counts)

    def avg(self,
            input: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
            normalize=False
            ) -> np.ndarray:
        # TODO we should split this function into a member method that takes words as input
        # and a function for averaging vectors
        """
        Computes the average of the input vectors. If normalize is True vectors are normalized
        before and after averaging
        """
        if isinstance(input, list):
            if all(map(lambda w: isinstance(w, np.ndarray), input)):
                vecs = np.array(input)
            else:
                vecs = self[input]
        elif isinstance(input, np.ndarray):
            vecs = input
        else:
            vecs = self[input]

        if len(vecs.shape) == 1:
            return unitvec(vecs) if normalize else vecs
        else:
            if normalize:
                norms = cast(np.ndarray, norm(vecs))
                return unitvec(np.mean(vecs / norms[..., np.newaxis], axis=0))
            else:
                return np.mean(vecs, axis=0)

    def knn(self, input: Union[str, np.ndarray, List[Union[str, np.ndarray]]], k=10
            ) -> Tuple[List[str], List[float]]:
        """

        The famous “King – Man + Woman = Queen” example:

        >>> emb = Embedding.load('wiki_en')
        >>> emb.knn([emb['king'], -emb['man'], emb['woman']], k=2)
        (['king', 'queen'], [0.8580309152603149, 0.8030596971511841])

        """
        assert self.norms is not None, 'norms must not be None'
        idcs, sims = knn(self.vecs, self.norms, self.avg(input, normalize=True), k)
        return list(self.words[idcs]), sims.tolist()
