import mmap
import os
from pathlib import Path
import struct
from typing import Callable, Dict, List, Optional, Union, cast

import numba as nb
import numpy as np
from numpy.lib.format import open_memmap

from vectorspace.embedding import Embedding, dump_vocab, load_vocab
from vectorspace.math import norm, softmax
from vectorspace.utils import json_load, json_dump

__all__ = ['FastTextModel', 'load_fasttext', 'save_fasttext_format']

# Non-breaking space
SPACE_REPLACEMENT_CHAR = "\u00A0"


class FastTextModel:

    def __init__(self,
                 meta: Dict,
                 input: Embedding,
                 input_labels: Optional[Embedding] = None,
                 output: Optional[Embedding] = None,
                 output_labels: Optional[Embedding] = None,
                 ngrams: Optional[np.ndarray] = None):
        self._meta = meta
        self._input = input
        self._input_labels = input_labels
        self._output = output
        self._output_labels = output_labels
        self._ngrams = ngrams

    @property
    def meta(self):
        return self._meta

    @property
    def input(self) -> Embedding:
        return self._input

    @property
    def input_labels(self) -> Optional[Embedding]:
        return self._input_labels

    @property
    def output(self) -> Optional[Embedding]:
        return self._output

    @property
    def output_labels(self) -> Optional[Embedding]:
        return self._output_labels

    @property
    def labels(self) -> Optional[Embedding]:
        return self._input_labels or self.output_labels or None

    @property
    def ngrams(self) -> Optional[np.ndarray]:
        return self._ngrams

    def __len__(self):
        return self.meta['size']

    def merge(self, weights=((0.75, 0.25))) -> 'FastTextModel':
        assert self.output is not None and self.output.vecs is not None, 'output vectors must not be None'
        assert self.input.vecs.shape == self.output.vecs.shape, 'shape of input and output vectors must match'
        merged_vecs = np.average([
            self.input.vecs,
            self.output.vecs
        ], axis=0, weights=weights).astype(np.float32)
        merged_emb = Embedding(merged_vecs, cast(np.ndarray, norm(merged_vecs)),
                               self.input.words,
                               self.input.index,
                               counts=self.input.counts)
        return FastTextModel(
            self.meta,
            input=merged_emb,
            input_labels=self.input_labels,
            output=self.output,
            output_labels=self.output_labels,
            ngrams=self.ngrams)

    def sent_vector(self, sent: List[str], native_sent2vec=False):
        return sent_vector(self, sent, native_sent2vec)

    def predict(self, words: List[str], k=1):
        assert self.output_labels is not None, 'Predict requires output labels'
        vector = self.sent_vector(words)
        probs = softmax(np.dot(vector, self.output_labels.vecs.T))
        topk = np.argsort(-probs)[:k]
        return list(zip(self.output_labels.words[topk], probs[topk]))


def load_fasttext(filename: str, work_dir: Union[str, Path], chunk_size=4096) -> 'FastTextModel':
    """Load a fastText binary model"""

    work_dir = Path(work_dir)

    # analyze binary model and create support structures
    if not (work_dir / 'meta.json').is_file() or not (work_dir / 'labels.mdb').is_file():
        os.makedirs(work_dir, exist_ok=True)

        with open(filename, 'r+b') as file:
            mmap_file = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)

            keys = ('magic', 'version', 'dim', 'ws', 'epoch', 'min_count', 'neg', 'word_ngrams',
                    'loss', 'model', 'bucket', 'minn', 'maxn', 'lr_update_rate', 't',
                    'size', 'nwords', 'nlabels', 'ntokens', 'pruneidx')

            # read args and vocab props
            meta = dict(zip(keys, struct.unpack('<2I12i1d3i2q', mmap_file.read(92))))

            # resolve model and loss name
            meta.update(model=(None, 'cbow', 'skipgram', 'supervised', 'sent2vec', 'pvdm')[meta['model']],
                        loss=(None, 'hs', 'ns', 'softmax')[meta['loss']])

            assert meta['pruneidx'] == -1, 'pruned models not supported'
            size, bucket, nwords, nlabels = meta['size'], meta['bucket'], meta['nwords'], meta['nlabels']

            def iter_words(size: int, counts, word_transform: Optional[Callable] = None):
                strct = struct.Struct('<1q1b')
                start = mmap_file.tell()

                for i in range(size):
                    # read word
                    wlen = mmap_file.find(b'\0') - start
                    word = mmap_file.read(wlen)
                    start += wlen + 10

                    # read count/type (1 means is label)
                    mmap_file.seek(1, os.SEEK_CUR)
                    count, type = strct.unpack(mmap_file.read(9))

                    # write counts
                    counts[i] = count
                    if word_transform is not None:
                        yield word_transform(word)
                    else:
                        yield word

            # dump words
            counts = open_memmap(work_dir / 'counts.npy', dtype=np.int32, mode='w+', shape=(nwords,))
            dump_vocab(iter_words(nwords, counts), work_dir)
            del counts

            # dump labels
            label_counts = open_memmap(work_dir / 'label_counts.npy', dtype=np.int32, mode='w+', shape=(nlabels,))

            def label_transform(label: bytes):
                label_orig = label.decode().replace(SPACE_REPLACEMENT_CHAR, ' ').encode()
                return label_orig[len('__label__'):]

            dump_vocab(iter_words(nlabels, label_counts, word_transform=label_transform),
                       work_dir, words_file='labels.mdb', index_file='label_index.mdb')
            del label_counts

            if meta['version'] >= 11:
                quant, = struct.unpack('1?', mmap_file.read(1))
                assert not quant, 'quantized models not supported'

            # input vectors
            m, n = struct.unpack('2q', mmap_file.read(16))
            offset = mmap_file.tell()
            meta.update(input_offset=offset, input_m=m - bucket, input_n=n)

            # write input norms to mmap file
            input_vecs = np.memmap(file, dtype=np.float32, mode='r', offset=offset, shape=(m - bucket, n))
            input_norms = open_memmap(work_dir / 'input_norms.npy', dtype=np.float32, mode='w+', shape=(m - bucket,))
            for i in range(0, size, chunk_size):
                chunk = input_vecs[i:i+chunk_size]
                input_norms[i:i+len(chunk)] = norm(chunk)
            del input_norms, input_vecs

            # ngram vectors
            offset = offset + (m - bucket) * n * 4
            meta.update(bucket_offset=offset, bucket_m=bucket, bucket_n=n)

            mmap_file.seek(m * n * 4, os.SEEK_CUR)
            if meta['version'] >= 11:
                quant, = struct.unpack('1?', mmap_file.read(1))
                assert not quant, 'quantized models not supported'

            # output vectors
            m, n = struct.unpack('2q', mmap_file.read(16))
            offset = mmap_file.tell()
            meta.update(output_offset=offset, output_m=m, output_n=n)

            # write output norms to mmap file
            output_vecs = np.memmap(file, dtype=np.float32, mode='r', offset=offset, shape=(m, n))
            output_norms = open_memmap(work_dir / 'output_norms.npy', dtype=np.float32, mode='w+', shape=(m,))
            for i in range(0, size, chunk_size):
                chunk = output_vecs[i:i+chunk_size]
                output_norms[i:i+len(chunk)] = norm(chunk)
            del output_norms, output_vecs

            # dump metadata
            json_dump(meta, work_dir / 'meta.json', indent=4)

            del mmap_file

    # load metadata
    meta = json_load(work_dir / 'meta.json')
    nwords, nlabels = meta['nwords'], meta['nlabels']

    # open words
    counts = np.load(work_dir / 'counts.npy', mmap_mode='r')
    words, index = load_vocab(work_dir)

    # open labels
    label_counts = np.load(work_dir / 'label_counts.npy', mmap_mode='r')
    labels, label_index = load_vocab(work_dir, words_file='labels.mdb', index_file='label_index.mdb')

    # open input vectors
    offset = meta['input_offset']
    shape = (meta['input_m'], meta['input_n'])

    input_vecs = np.memmap(filename, dtype=np.float32, mode='r', offset=offset, shape=shape)
    input_norms = np.load(work_dir / 'input_norms.npy', mmap_mode='r')

    # input embedding
    input = Embedding(input_vecs[:nwords], input_norms[:nwords], words, index, counts)

    # input label embedding
    if meta['model'] == 'pvdm':
        input_labels = Embedding(input_vecs[nwords:], input_norms[nwords:], labels, label_index, label_counts)
    else:
        input_labels = None

    # open output vectors
    offset = meta['output_offset']
    shape = (meta['output_m'], meta['output_n'])

    output_vecs = np.memmap(filename, dtype=np.float32, mode='r', offset=offset, shape=shape)
    output_norms = np.load(work_dir / 'output_norms.npy', mmap_mode='r')

    # output embedding
    if meta['model'] != 'supervised':
        output = Embedding(output_vecs[:nwords], output_norms[:nwords], words, index, counts)
    else:
        output = None

    # output label embedding
    if meta['model'] == 'supervised':
        output_labels = Embedding(output_vecs, output_norms, labels, label_index, label_counts)
    elif meta['model'] == 'pvdm':
        output_labels = Embedding(output_vecs[nwords:], output_norms[nwords:], labels, label_index, label_counts)
    else:
        output_labels = None

    # open bucket vectors
    offset = meta['bucket_offset']
    shape = (meta['bucket_m'], meta['bucket_n'])
    ngrams = np.memmap(filename, dtype=np.float32, mode='r', offset=offset, shape=shape)

    return FastTextModel(meta, input, input_labels, output, output_labels, ngrams)


def save_fasttext_format(emb: Embedding, filename: str = 'model.bin', fallback_count=1):
    shape = emb.shape
    # use cbow defaults for unknown parameters
    meta = {
        "magic": 793712314,
        "version": 12,
        "dim": shape[1],
        "ws": 5,                # default
        "epoch": 5,             # default
        "min_count": 1,         # default
        "neg": 5,               # default
        "word_ngrams": 1,       # default
        "loss": 2,              # default (ns)
        "model": 1,             # default (cbow)
        "bucket": 0,            # off
        "minn": 0,              # off
        "maxn": 0,              # off
        "lr_update_rate": 100,  # default
        "t": 0.0001,            # default
        "size": shape[0],
        "nwords": shape[0],
        "nlabels": 0,
        "ntokens": shape[0],
        "pruneidx": -1
    }

    with open(filename, 'wb') as fout:
        keys = ('magic', 'version', 'dim', 'ws', 'epoch', 'min_count', 'neg', 'word_ngrams',
                'loss', 'model', 'bucket', 'minn', 'maxn', 'lr_update_rate', 't',
                'size', 'nwords', 'nlabels', 'ntokens', 'pruneidx')

        # write args and vocab props
        fout.write(struct.pack('<2I12i1d3i2q', *[meta[k] for k in keys]))

        for word, count in zip(emb.words, emb.counts if emb.counts else [fallback_count] * shape[0]):
            # write word
            fout.write(word.encode())
            fout.write(b'\0')

            # write count/type (1 means is label)
            fout.write(struct.pack('<1q1b', count, 0))

        # quantized models not supported
        fout.write(struct.pack('1?', False))

        # write input vectors
        fout.write(struct.pack('2q', *shape))
        emb.vecs.tofile(fout)

        # quantized models not supported
        fout.write(struct.pack('1?', False))

        # fill output vectors with zeros
        fout.write(struct.pack('2q', *shape))
        fout.seek(shape[0] * shape[1] * 4, 1)
        fout.truncate()


@nb.njit(locals=dict(h=nb.uint32, c=nb.uint8))
def hash_word(word: bytes):
    h = 2166136261
    for c in word:
        h ^= c
        h *= 16777619
    return h


def sent_vector(model: 'FastTextModel', sent: List[str], native_sent2vec=False):
    assert isinstance(sent, (tuple, list)), f'sent must be of type list and not {type(sent)}'
    assert model.ngrams is not None, 'ngram vectors must not be None'

    minn = model.meta['minn']
    maxn = model.meta['maxn']
    word_ngrams = model.meta['word_ngrams']
    bucket = model.meta['bucket']

    ngrams = []
    if word_ngrams > 1:
        # original sent2vec uses word indices and no hashes as fastText does
        if native_sent2vec:
            hashes = cast(List[int], model.input.index[sent])
        else:
            hashes = [hash_word(w.encode()) for w in sent]

        # addWordNgrams
        for i in range(len(hashes)):
            h = hashes[i]
            for j in range(i + 1, min(len(hashes), i + word_ngrams)):
                h = h * 116049371 + hashes[j]
                ngrams.append(h % bucket)

    if maxn > 1 and maxn >= minn:
        # addSubwords
        for w in sent:
            w = '<' + w + '>'
            for n in range(minn, min(len(w), maxn) + 1):
                for i in range(0, len(w) - n + 1):
                    ngrams.append(hash_word(w[i:i + n].encode()) % bucket)

    return np.mean(np.concatenate((
        model.input[[w for w in sent if w in model.input]],
        model.ngrams[ngrams])), axis=0)
