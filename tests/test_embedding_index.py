import pytest

from vectorspace.embedding import WordView, IndexView


@pytest.fixture
def words():
    return WordView(['dog', 'cat', 'pig'])


@pytest.fixture
def index(words):
    return IndexView(dict((w, i) for i, w in enumerate(words)))


def test_index_get_single_item(index):
    assert index['dog'] == 0


def test_index_get_multi_item_implicit(index):
    assert index['dog', 'cat'] == [0, 1]


def test_index_get_multi_item_explicit(index):
    assert index[['dog', 'cat']] == [0, 1]


def test_index_get_slice_all(index):
    assert index[:] == [0, 1, 2]


def test_index_get_slice_all_reverse(index):
    with pytest.raises(AssertionError):
        index[::-1]


def test_index_get_slice(index):
    assert index['catfish':'dogfish'] == [0]


def test_index_get_lambda(index):
    assert index[lambda w: 'cat' in w] == [1]
