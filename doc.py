import os
from preprocess_nlp.vocab_utils.common import UNK, SOS, EOS, EOS_ID
import numpy as np
import collections
import preprocess_nlp.doc_format.txt as dtxt
import preprocess_nlp.doc_format.tfrecords as dtfrecords
import preprocess_nlp.doc_token as dt
import pdb


class Document(object):

    @classmethod 
    def create_from_file(cls, document_path, token_type, vocab=None):
        assert token_type==dt.WORD_TYPE or token_type==dt.ID_TYPE
        if not os.path.exists(document_path):
            raise IOError("file not found")
        if document_path.endswith(".txt"):
            doc_gen_f = dtxt.doc_gen_f(document_path, token_type)
        elif document_path.endswith(".tfrecords"):
            pass
        elif document_path.endswith(".npy"):
            pass
        else:
            raise ValueError("Not valid document state type")
        return cls(doc_gen_f, token_type, vocab)

    @classmethod
    def create_from_iter(cls, document_iter, token_type, vocab=None):
        assert token_type==dt.WORD_TYPE or token_type==dt.ID_TYPE
        def doc_gen_f():
            for token in document_iter:
                yield token
        return cls(doc_gen_f, token_type, vocab)

    def __init__(self, doc_gen_f, token_type, vocab=None):
        self._vocab = vocab
        self._iter_gen_func = doc_gen_f
        self._token_type = token_type
        self._label_dict = {}

    def __iter__(self):
        return self._iter_gen_func()

    def get_sequenced_iter(self, seq_len):
        doc_gen = iter(self)
        while True:
            seq_list = []
            try:
                for _ in range(seq_len):
                    seq_list.append(next(doc_gen))
            except:
                print("doc sequenced iter finished")
                break
                # raise ValueError("document already empty")
            yield tuple(seq_list)

    ####################
    # Client Interface #
    ####################
    @property
    def token_type(self):
        return self._token_type

    def set_vocab(self, vocab):
        self._vocab = vocab

    def set_label(self, key, val):
        self._label_dict[key] = val

    def get_label(self, key):
        return self._label_dict.get(key, None)

    ##########################
    # State Changing methods #
    ##########################
    def toggle_word_id(self):
        assert self._vocab is not None
        if self._token_type == dt.WORD_TYPE:
            self._iter_gen_func = dt.word2id_gen_f(self, self._vocab)
            self._token_type = dt.ID_TYPE
        else:
            self._iter_gen_func = dt.id2word_gen_f(self, self._vocab)
            self._token_type = dt.WORD_TYPE

    def mask_unk(self):
        assert self._vocab is not None
        iter_gen = self._iter_gen_func()
        def mask_unk_f():
            if self.token_type == str:
                unk_signature = UNK
                unk_check_f = self._vocab.check_word_exist
            else:
                raise NotImplementedError("Not implemented yet")
            for token in iter_gen:
                if not unk_check_f(token):
                    yield unk_signature
                else:
                    yield token
        self._iter_gen_func = mask_unk_f


