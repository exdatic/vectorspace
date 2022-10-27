import re
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from vectorspace.fasttext import SPACE_REPLACEMENT_CHAR

__all__ = ['iter_lines', 'iter_lines_fasttext', 'iter_texts', 'iter_texts_fasttext']


def _get_nested_value(val: Any, field: str) -> Any:
    if not field:
        return val
    else:
        field_list = field.split('.')
        key = field_list[0]
        if isinstance(val, dict):
            return _get_nested_value(val.get(key), '.'.join(field_list[1:]))
        elif isinstance(val, list):
            return [_get_nested_value(v[key], '.'.join(field_list[1:])) for v in val if key in v]


def _doc_to_text(doc: Dict, fields: List[str]) -> str:
    def get_value(doc, field):
        if '.' in field:
            value = _get_nested_value(doc, field)
        else:
            value = doc[field]
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            return '\n\n'.join(value)
        else:
            raise ValueError(f'type of field value not supported: {type(value)}')
    return '\n\n'.join([get_value(doc, f) for f in fields if f in doc])


def _get_labels(doc: Dict, label_fields: List[str], typed_labels=False) -> List[str]:
    labels: List[str] = []
    for field in label_fields:
        if '.' in field:
            value = _get_nested_value(doc, field)
        else:
            value = doc.get(field, None)
        if value:
            if isinstance(value, list):
                if typed_labels:
                    labels.extend([f'__{field}__{v}' for v in value if v])
                elif value:
                    labels.extend(value)
            else:
                if typed_labels:
                    labels.append(f'__{field}__{value}')
                else:
                    labels.append(value)
    return labels


def iter_texts(
    store: Mapping,
    text_fields: List[Union[str, List[str]]],
    label_fields: Optional[List[str]] = None,
    analyzer: Optional[Callable[[str], List[str]]] = None,
    typed_labels: bool = False,
    **kwargs
) -> Iterator[Union[Union[List[str], str], Tuple[Union[List[str], str], List[str]]]]:
    """Iterates over texts created from store documents.
       Yields text or tuples of text and labels"""

    def iter_fields(any_fields: List[Union[str, List[str]]]) -> Iterator[List[str]]:
        fields = []
        for field in any_fields:
            if isinstance(field, list):
                if fields:
                    yield fields
                    fields = []
                yield field
            else:
                fields.append(field)
        if fields:
            yield fields

    source_fields = []
    if label_fields:
        source_fields.extend(f for f in label_fields)
    if text_fields:
        for f in text_fields:
            if isinstance(f, list):
                source_fields.extend(f)
            else:
                source_fields.append(f)

    query = kwargs.get("query")
    # optimization
    if hasattr(store, 'scan'):
        docs = store.scan(_source=source_fields, query=query)  # type: ignore
    else:
        docs = store.items()

    for doc_id, doc in docs:
        doc['_id'] = doc_id  # inject _id
        for fields in iter_fields(text_fields):
            text = _doc_to_text(doc, fields)
            if analyzer is not None:
                text = analyzer(text)
            if label_fields:
                labels = _get_labels(doc, label_fields, typed_labels)
                yield text, labels
            else:
                yield text


def iter_lines(
    store: Mapping,
    text_fields: List[Union[str, List[str]]],
    label_fields: Optional[List[str]] = None,
    analyzer: Optional[Callable[[str], List[str]]] = None,
    sentencizer: Optional[Callable[[str], Iterator[str]]] = None,
    typed_labels: bool = False,
    **kwargs
) -> Iterator[Union[Union[List[str], str], Tuple[Union[List[str], str], List[str]]]]:
    """Iterates over each line of texts created from store documents. Splits on sentences if sentencizer is passed,
       else splits on line breaks.
       Yields lines or tuples of lines and labels"""
    for text_or_tuple in iter_texts(
        store,
        text_fields,
        label_fields=label_fields,
        analyzer=None,
        typed_labels=typed_labels,
        **kwargs
    ):
        if label_fields:
            assert isinstance(text_or_tuple, tuple)
            text, labels = text_or_tuple
        else:
            assert not isinstance(text_or_tuple, tuple)
            text, labels = text_or_tuple, []

        assert isinstance(text, str)

        if sentencizer is not None:
            lines = sentencizer(text)
        else:
            lines = text.split('\n')

        for line in lines:
            if analyzer is not None:
                line = analyzer(line)
            if label_fields:
                yield line, labels
            else:
                yield line


def _to_ft_line(text_or_tuple: Union[Union[List[str], str], Tuple[Union[str, List[str]], List[str]]]):
    if isinstance(text_or_tuple, tuple):
        tokens, labels = text_or_tuple
        if tokens and labels:
            text = ' '.join(tokens)
            labels_cleaned = [re.sub(r'\s', SPACE_REPLACEMENT_CHAR, label) for label in labels]
            label_tokens = ' '.join([f'__label__{label}' for label in labels_cleaned])
            return ' '.join([label_tokens, text])
    else:
        tokens = text_or_tuple
        if tokens:
            return ' '.join(tokens)
    return None


def iter_texts_fasttext(
    store: Mapping,
    text_fields: List[Union[str, List[str]]],
    analyzer: Callable[[str], List[str]],
    label_fields: Optional[List[str]] = None,
    typed_labels: bool = False,
    **kwargs
) -> Iterator[str]:
    """Iterates over lines in fasttext format of texts created from store documents.
       Removes lines that contain no text or no labels."""
    for text_or_tuple in iter_texts(
        store,
        text_fields,
        label_fields=label_fields,
        analyzer=analyzer,
        typed_labels=typed_labels,
        **kwargs
    ):
        line = _to_ft_line(text_or_tuple)
        if line is not None:
            yield line


def iter_lines_fasttext(
    store: Mapping,
    text_fields: List[Union[str, List[str]]],
    analyzer: Callable[[str], List[str]],
    label_fields: Optional[List[str]] = None,
    sentencizer: Optional[Callable[[str], Iterator[str]]] = None,
    typed_labels=False,
    **kwargs
) -> Iterator[str]:
    """Iterates over lines in fasttext format of texts created from store documents.
       Splits on sentences if sentencizer is passed, else splits on line breaks.
       Removes lines that contain no text or no labels."""
    for text_or_tuple in iter_lines(
        store,
        text_fields,
        label_fields=label_fields,
        analyzer=analyzer,
        sentencizer=sentencizer,
        typed_labels=typed_labels,
        **kwargs
    ):
        line = _to_ft_line(text_or_tuple)
        if line is not None:
            yield line
