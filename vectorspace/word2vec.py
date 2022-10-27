import gzip
import os
from pathlib import Path
from typing import Union

import numpy as np
from numpy.lib.format import open_memmap

from vectorspace.embedding import Embedding, dump_vocab, load_embedding
from vectorspace.math import norm
from vectorspace.utils import json_dump

__all__ = ['load_word2vec', 'save_word2vec_format']


def load_word2vec(filename: Union[str, Path], work_dir: Union[str, Path]) -> 'Embedding':
    """Load word embedding from text file"""
    filename = Path(filename)
    work_dir = Path(work_dir)

    if not (work_dir / 'meta.json').is_file():
        os.makedirs(work_dir, exist_ok=True)
        with gzip.open(filename) if filename.suffix in ('.gz', '.gzip') else open(filename, 'rb') as file:
            def iter_words():
                for i, line in enumerate(file):
                    word, line = line.split(maxsplit=1)
                    vecs = np.fromstring(line, sep=' ', dtype=np.float32)
                    vecs_file[i] = vecs
                    norms_file[i] = norm(vecs)
                    yield word

            n, m = np.fromstring(file.readline(), sep=' ', dtype=np.int32)
            vecs_file = open_memmap(work_dir / 'vecs.npy', dtype=np.float32, mode='w+', shape=(n, m))
            norms_file = open_memmap(work_dir / 'norms.npy', dtype=np.float32, mode='w+', shape=(n,))
            dump_vocab(iter_words(), work_dir)
            del vecs_file, norms_file

        json_dump({'filename': os.path.basename(filename)}, work_dir / 'meta.json', indent=4)

    return load_embedding(work_dir)


def save_word2vec_format(emb: Embedding, filename: str = 'model.vec', binary=False, write_header=True):
    n, m = emb.shape
    with open(filename, 'wb') as fout:
        if write_header:
            fout.write(f'{n} {m}\n'.encode('utf-8'))
        for word, vec in zip(emb.words, emb.vecs):
            if binary:
                fout.write(word.encode('utf-8') + b' ' + vec.tobytes())
            else:
                fout.write(f'{word} {" ".join(repr(v) for v in vec)}\n'.encode('utf8'))
