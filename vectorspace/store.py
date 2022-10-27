from collections.abc import MutableMapping as MutableMappingABC
import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union, cast

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk, scan
from elasticsearch_dsl import Index, Document
from elasticsearch_dsl.response import Hit

from vectorspace.utils import chunked

__all__ = ['Store', 'DEFAULT_SETTINGS']


DEFAULT_SETTINGS = {
    'index': {
        'number_of_shards': 5,
        'number_of_replicas': 0
    }
}


class Store(MutableMappingABC, Index):
    """
    Dict wrapper for an Elasticsearch index
    """
    def __init__(self,
                 es: Elasticsearch,
                 index: str,
                 mappings: Optional[Union[Dict, Type[Document]]] = None,
                 settings: Optional[Dict] = DEFAULT_SETTINGS,
                 meta: bool = False):
        super().__init__(using=es, name=index)  # type: ignore

        self._es = es
        self._index = index
        self._meta = meta

        if mappings:
            if isinstance(mappings, Type) and issubclass(mappings, Document):
                mappings = getattr(mappings, '_doc_type').mapping.to_dict()
            self.get_or_create_mapping()._update_from_dict(mappings)
        if settings:
            self.settings(**settings)

    @property
    def index(self):
        return self._index

    def _to_dict(self, res: Any) -> Dict:
        hit = Hit(res)
        doc = hit.to_dict()
        assert isinstance(doc, dict)
        if self._meta and hit.meta:
            meta = hit.meta.to_dict()
            assert isinstance(meta, dict)
            doc.update(meta)
        return cast(Dict, doc)

    def __getitem__(self, key: str) -> Dict:
        try:
            return self._to_dict(self._es.get(index=self._index, id=key))
        except NotFoundError:
            raise KeyError(key) from None

    def get(self, key: str, default: Optional[Dict] = None, **kwargs) -> Optional[Dict]:
        try:
            return self._to_dict(self._es.get(index=self._index, id=key, **kwargs))
        except NotFoundError:
            return default

    def __setitem__(self, key: str, doc: Dict):
        # create index using mappings & settings
        if not self.exists():
            self.create()
        self._es.index(index=self._index, body=doc, id=key)  # type: ignore (FIXME es8)

    def __delitem__(self, key: str):
        try:
            self._es.delete(index=self._index, id=key)
        except NotFoundError:
            raise KeyError(key) from None

    def keys(self):
        for doc in scan(self._es, index=self._index, _source=False, ignore=(404)):
            yield doc['_id']

    def items(self):
        for doc in scan(self._es, index=self._index, ignore=(404)):
            yield (doc['_id'], self._to_dict(doc))

    def values(self):
        for doc in scan(self._es, index=self._index, ignore=(404)):
            yield self._to_dict(doc)

    def mget(self, keys, chunk_size: int = 500, ignore=(404), **kwargs):
        for chunk in chunked(keys, chunk_size):
            chunk = [key for key in chunk if key is not None]
            res = self._es.mget(index=self._index, body={'ids': chunk}, ignore=ignore, **kwargs)
            for doc in res['docs']:
                yield (doc['_id'], self._to_dict(doc))

    def scan(self, query: Optional[Any] = None, ignore=(404), **kwargs):
        if query and isinstance(query, str):
            query = dict(query=dict(query_string=dict(query=query)))
        for doc in scan(self._es, index=self._index, ignore=ignore, query=query, **kwargs):
            yield (doc['_id'], self._to_dict(doc))

    def _batch_index(
        self, items: Iterable[Tuple[str, Dict]], upsert: bool = False, **kwargs
    ) -> Tuple[int, Union[int, List[Any]]]:
        def iter_actions():
            for (key, doc) in items:
                action = dict(_index=self.index, _id=key)
                if upsert:
                    action.update(_op_type='update', doc=doc, doc_as_upsert=True)  # type: ignore (FIXME es8)
                else:
                    action.update(_op_type='index', _source=doc)  # type: ignore (FIXME es8)
                yield action

        return bulk(self._es, iter_actions(), **kwargs)

    def upsert(self, other=(), **kwds):
        # create index using mappings and settings
        if not self.exists():
            self.create()

        if isinstance(other, Mapping):
            other = other.items()
        elif hasattr(other, "keys"):
            other = ((key, other[key]) for key in other.keys())

        # bulk upsert
        return self._batch_index(itertools.chain(other, kwds.items()), upsert=True)

    def update(self, other=(), **kwds):
        # create index using mappings and settings
        if not self.exists():
            self.create()

        if isinstance(other, Mapping):
            other = other.items()
        elif hasattr(other, "keys"):
            other = ((key, other[key]) for key in other.keys())

        # bulk update
        return self._batch_index(itertools.chain(other, kwds.items()), upsert=False)

    def upsert_by_query(self, query: Any, **kwds):
        return self._es.update_by_query(
            index=self.index,
            body={
                **query,
                'script': {
                    'source': 'params.forEach((k, v) -> ctx._source[k] = v)',
                    'params': kwds
                }
            },
            params=dict(conflicts='proceed'))

    def apply_script(self, keys: Iterable[str], source: str, **params) -> Tuple[int, Union[int, List[Any]]]:
        script = dict(source=source, params=params)

        def iter_actions():
            for key in keys:
                yield dict(_index=self.index, _id=key, _op_type='update', script=script)

        return bulk(self._es, iter_actions())

    def delete_all(self):
        return self.delete(ignore=(404))

    def delete_by_query(self, query: Any):
        return self._es.delete_by_query(index=self.index, body=query, ignore=(404))

    def __iter__(self):
        return self.keys()

    def __len__(self, query: Optional[Any] = None):
        return self._es.count(index=self._index, body=query, ignore=(404)).get('count', 0)

    def __contains__(self, key: str):
        try:
            self._es.get(index=self._index, id=key, _source=False)
            return True
        except NotFoundError:
            return False
