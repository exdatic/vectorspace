import numpy as np
import pytest
from typing import cast

from vectorspace.embedding import Embedding, WordView, IndexView
from vectorspace.math import norm


@pytest.fixture
def words():
    return WordView(['dog', 'cat', 'dog'])


@pytest.fixture
def index(words):
    return IndexView(dict((w, i) for i, w in enumerate(words)))


@pytest.fixture
def vecs(words, dim=10):
    return np.random.rand((len(words)), dim).astype(np.float32)


@pytest.fixture
def embedding(vecs, words, index):
    return Embedding(vecs, cast(np.ndarray, norm(vecs)), words, index)


def test_word_duplicates_size_unchanged(embedding):
    assert len(embedding) == 3


def test_word_duplicates_last_wins(embedding, index):
    assert (embedding['dog'] == embedding.vecs[2]).all()
    assert index['dog'] == 2


def test_embedding_knn_single_word(embedding):
    words, scores = embedding.knn('dog', 1)
    assert list(zip(words, np.around(scores, 4))) == [('dog', 1.0)]


def test_embedding_knn_multi_words(embedding):
    words, scores = embedding.knn(['dog', 'dog'], 1)
    assert list(zip(words, np.around(scores, 4))) == [('dog', 1.0)]


def test_embedding_knn_single_vector(embedding):
    words, scores = embedding.knn(embedding['dog'], 1)
    assert list(zip(words, np.around(scores, 4))) == [('dog', 1.0)]


def test_embedding_knn_multi_vector(embedding):
    words, scores = embedding.knn([embedding['dog'], embedding['dog']], 1)
    assert list(zip(words, np.around(scores, 4))) == [('dog', 1.0)]
