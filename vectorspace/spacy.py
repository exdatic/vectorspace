from typing import Callable, Iterator, List, Optional, Union

from anyascii import anyascii

__all__ = ['Analyzer', 'Sentencizer']

trans_table = str.maketrans({
    'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
    'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
})


def nlp(lang, disable=['ner', 'tagger', 'parser']):
    try:
        import spacy
    except ImportError:
        raise ImportError("spaCy not installed: pip install spacy")

    # download matching spacy models on-the-fly
    try:
        # flake8: noqa F401
        import en_core_web_sm
    except ModuleNotFoundError:
        from spacy import cli
        cli.download('en_core_web_sm')  # type: ignore

    try:
        # flake8: noqa F401
        import de_core_news_sm
    except ModuleNotFoundError:
        from spacy import cli
        cli.download('de_core_news_sm')  # type: ignore

    if lang == "en":
        nlp_ = spacy.load("en_core_web_sm", disable=disable)
    elif lang == "de":
        nlp_ = spacy.load("de_core_news_sm", disable=disable)
    else:
        raise Exception('language not supported')

    # for sentence segmentation
    nlp_.add_pipe('sentencizer')
    return nlp_


class Analyzer:

    def __init__(
        self,
        lang: str,
        space: bool = False,
        punct: bool = False,
        num: bool = True,
        stop: bool = True,
        lower: bool = False,
        ascii: bool = False,
        min_len: int = 1,
        transform: Optional[Callable[[str], Optional[str]]] = None
    ):
        self.nlp = nlp(lang)
        self.space = space
        self.punct = punct
        self.num = num
        self.stop = stop
        self.lower = lower
        self.ascii = ascii
        self.min_len = min_len
        self.transform = transform

    def __call__(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens = []
        for tok in doc:
            if not self.space and tok.is_space:
                continue
            if not self.punct and tok.is_punct:
                continue
            if not self.num and tok.like_num:
                continue
            if not self.stop and tok.is_stop:
                continue
            if len(tok) < self.min_len:
                continue
            token = tok.text
            if self.lower:
                token = token.lower()
            if self.ascii:
                token = token.translate(trans_table)
                token = anyascii(token)
            if self.transform:
                token = self.transform(token)
                if not token:
                    continue
            tokens.append(token)
        return tokens


class Sentencizer:

    def __init__(self, lang: str):
        self.nlp = nlp(lang)

    def __call__(self, text: str) -> Iterator[str]:
        for span in self.nlp(text).sents:
            yield span.text
