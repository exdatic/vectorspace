import os
from pathlib import Path
from typing import Union
import gzip

from anyascii import anyascii
import numpy as np
from numpy.lib.format import open_memmap

from vectorspace.embedding import dump_vocab, load_embedding, Embedding
from vectorspace.math import norm
from vectorspace.utils import json_dump

__all__ = ['load_numberbatch']


def load_numberbatch(filename: Union[str, Path],
                     work_dir: Union[str, Path],
                     lang: str,
                     size: int,
                     ascii: bool = False) -> 'Embedding':

    """Load word embedding from text file"""
    filename = Path(filename)
    work_dir = Path(work_dir)

    if not (work_dir / 'meta.json').is_file():
        os.makedirs(work_dir, exist_ok=True)

        with gzip.open(filename) if filename.suffix in ('.gz', '.gzip') else open(filename, 'rb') as file:
            def iter_lines():
                prefix = f'/c/{lang}/'.encode()
                match = False
                for line in file:
                    if line.startswith(prefix):
                        match = True
                        yield line[len(prefix):]
                    elif match:
                        break

            def iter_words():
                for i, line in enumerate(iter_lines()):
                    word, line = line.split(maxsplit=1)
                    vecs = np.fromstring(line, sep=' ', dtype=np.float32)
                    vecs_file[i] = vecs
                    norms_file[i] = norm(vecs)
                    if ascii:
                        word = anyascii(word.decode()).encode()
                    yield word

            n = size
            _, m = np.fromstring(file.readline(), sep=' ', dtype=np.int32)
            vecs_file = open_memmap(work_dir / 'vecs.npy', dtype=np.float32, mode='w+', shape=(n, m))
            norms_file = open_memmap(work_dir / 'norms.npy', dtype=np.float32, mode='w+', shape=(n,))
            dump_vocab(iter_words(), work_dir)
            del vecs_file, norms_file

        json_dump({'filename': os.path.basename(filename)}, work_dir / 'meta.json', indent=4)

    return load_embedding(work_dir)
