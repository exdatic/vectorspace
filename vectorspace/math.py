import mmap
from typing import List, Tuple, Union, cast

import numba as nb
import numpy as np

__all__ = ['argtopk', 'avg', 'cossim', 'knn', 'norm', 'softmax', 'unitvec', 'unitvecs']


def _dot(matrix: np.ndarray, vector: np.ndarray):
    if type(matrix) is np.memmap:
        return _dot_chunked(matrix, vector)
    else:
        return np.dot(matrix, vector)


@nb.njit
def _dot_chunked(matrix: np.ndarray, vector: np.ndarray, n=mmap.PAGESIZE) -> np.ndarray:
    """Don't ask why, it's really faster on large memory-mapped arrays."""
    size = matrix.shape[0]
    sims = np.empty((size,), dtype=np.float32)
    for i in range(0, size, n):
        j = i + n
        sims[i:j] = np.dot(matrix[i:j], vector)
    return sims


def knn(matrix: np.ndarray,
        norms: np.ndarray,
        vector: np.ndarray,
        k: int
        ) -> Tuple[np.ndarray, np.ndarray]:
    sims = _dot(matrix, unitvec(vector))
    sims /= norms  # inplace
    if k is None or k == -1 or k >= len(sims):
        idcs = sims.argsort()[::-1][:k]
    else:
        # NOTE: argtopk sorts the other way around and returns an unsorted list
        idcs = np.array(sorted(argtopk(sims, k), key=lambda i: -sims[i]))
    return idcs, sims[idcs]


def unitvecs(x: np.ndarray) -> np.ndarray:
    norms = cast(np.ndarray, norm(x))
    return x / norms[..., np.newaxis]


def unitvec(x: np.ndarray) -> np.ndarray:
    # better precision when using MKL: return (np.array([x]) / norm([x])[:, None])[0]
    return x / norm(x)


def cossim(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.dot(unitvec(x1), unitvec(x2))


def norm(x: np.ndarray) -> Union[np.ndarray, float]:
    norms = np.linalg.norm(x, axis=-1)
    # avoid division by zero
    if norms.shape:
        norms[norms == 0] = 1e-8
    elif norms == 0:
        norms = 1e-8
    return norms


def avg(vecs: Union[List, np.ndarray], normalize: bool = False) -> np.ndarray:
    """
    Computes the average of the input vectors.
    If normalize is True vectors are normalized before and after averaging
    """
    if isinstance(vecs, list):
        vecs = np.array(vecs)
    if len(vecs.shape) == 1:
        return unitvec(vecs) if normalize else vecs
    else:
        if normalize:
            norms = cast(np.ndarray, norm(vecs))
            return unitvec(np.mean(vecs / norms[..., np.newaxis], axis=0))
        else:
            return np.mean(vecs, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x)
    return e / np.sum(e, axis=0)


# This license applies to parts of the code originating from the
# https://github.com/apache/lucene-solr repository:
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

@nb.njit
def argtopk(x: np.ndarray, k: int) -> List[int]:
    """
    Returns the indexes of the k largest elements of the given input.
    The implementation is based on Lucene's priority queue from the
    https://github.com/apache/lucene-solr repository.

    Parameters
    ----------
    x : array
        the input array
    k : int
        the k in "top-k"

    Returns
    -------
    list
        the unsorted list of indexes of the k largest elements
    """
    if (0 == k):
        # We allocate 1 extra to avoid if statement in top()
        heap_size = 2
    else:
        # NOTE: we add +1 because all access to heap is
        # 1-based not 0-based. heap[0] is unused.
        heap_size = k + 1

    size = 0
    heap = list(range(heap_size))

    for i in range(len(x)):
        if size < k:
            # --- add ---
            # Adds an Object to a PriorityQueue in log(size) time.
            # If one tries to add more objects than maxSize from initialize an
            size += 1
            heap[size] = i
            _up_heap(x, heap, size)
        elif size > 0 and not x[i] < x[heap[1]]:
            # --- insert_with_overflow ---
            # Adds an Object to a PriorityQueue in log(size) time.
            # It returns the object (if any) that was
            # dropped off the heap because it was full. This can be
            # the given parameter (in case it is smaller than the
            # full heap's minimum, and couldn't be added), or another
            # object that was previously the smallest value in the
            # heap and now has been replaced by a larger one, or null
            # if the queue wasn't yet full with maxSize elements.
            heap[1] = i
            # --- update_top ---
            # Should be called when the Object at top changes values.
            # Still log(n) worst case, but it's at least twice as fast.
            _down_heap(x, heap, size, 1)

    return heap[1:size+1]


@nb.njit
def _up_heap(x, heap, orig_pos):
    i = orig_pos
    node = heap[i]            # save bottom node
    j = i >> 1
    while j > 0 and x[node] < x[heap[j]]:
        heap[i] = heap[j]     # shift parents down
        i = j
        j = j >> 1
    heap[i] = node            # install saved node
    return i != orig_pos


@nb.njit
def _down_heap(x, heap, size, i):
    node = heap[i]            # save top node
    j = i << 1                # find smaller child
    k = j + 1
    if k <= size and x[heap[k]] < x[heap[j]]:
        j = k
    while j <= size and x[heap[j]] < x[node]:
        heap[i] = heap[j]     # shift up child
        i = j
        j = i << 1
        k = j + 1
        if k <= size and x[heap[k]] < x[heap[j]]:
            j = k
    heap[i] = node            # install saved node
