import numpy as np
import pytest

from vectorspace.embedding import WordView


@pytest.fixture
def words():
    return WordView(['dog', 'cat', 'pig'])


def test_words_get_single_item(words):
    assert words[0] == 'dog'


def test_words_get_multi_item_implicit(words):
    assert words[0, 1] == ['dog', 'cat']


def test_words_get_multi_item_explicit(words):
    assert words[[0, 1]] == ['dog', 'cat']


def test_words_get_multi_item_negative(words):
    assert words[[0, -1]] == ['dog', 'pig']


def test_words_get_multi_item_ndarray(words):
    assert words[np.arange(len(words))] == ['dog', 'cat', 'pig']


def test_words_get_slice_all(words):
    assert words[:] == ['dog', 'cat', 'pig']


def test_words_get_slice_all_reverse(words):
    assert words[::-1] == ['pig', 'cat', 'dog']


def test_words_get_slice_negative(words):
    assert words[-1:] == ['pig']


def test_words_get_mask(words):
    assert words[False, True, False] == ['cat']


def test_words_get_np_item(words):
    assert words[np.array([0, 1])] == ['dog', 'cat']


def test_words_get_np_mask(words):
    assert words[np.array([False, True])] == ['cat']
