import os
import collections
from preprocess_nlp.vocab_utils.common import UNK, SOS, EOS, EOS_ID, PAD, PAD_ID
import numpy as np
import collections
import preprocess_nlp.doc_format.txt as dtxt
import preprocess_nlp.doc_format.tfrecords as dtfrecords
import preprocess_nlp.doc_token as dt
import pdb


class Document(object):
    @classmethod
    def create_from_txt(cls, txt_path, token_type, eol_type=dtxt.YIELD_EOL, vocab=None):
        dt.assert_type_valid(token_type)
        if not os.path.exists(txt_path):
            raise IOError("file not found")
        if eol_type == dtxt.YIELD_EOL:
            doc_gen_f = dtxt.doc_gen_f_yield_eol(txt_path, token_type)
        elif eol_type == dtxt.IGNORE_EOL:
            doc_gen_f = dtxt.doc_gen_f_ignore_eol(txt_path, token_type)
        elif eol_type == dtxt.KEEP_EOL_NL:
            doc_gen_f = dtxt.doc_gen_f_keep_eol_nl(txt_path, token_type)
        else:
            raise ValueError("Non existing end of line type")
        return cls(doc_gen_f, token_type, vocab)

    @classmethod
    def create_from_iter(cls, document_iter, token_type, vocab=None):
        dt.assert_type_valid(token_type)
        def doc_gen_f():
            for token in document_iter:
                yield token
        return cls(doc_gen_f, token_type, vocab)

    @classmethod
    def create_from_docs(cls, *docs):
        if len(docs) == 0:
            return None
        token_type = docs[0].token_type
        def merged_iter_f():
            for doc_ in docs:
                assert token_type==doc_.token_type
                for item in iter(doc_):
                    yield item
        return cls(merged_iter_f, token_type)  # TODO handle vocab 


    def __init__(self, doc_gen_f, token_type, vocab=None):
        self._vocab = vocab
        self._token_type = token_type
        self._label_dict = {}
        self._doc_len = None
        self._generator_fs = []
        self._iter_gen_func = doc_gen_f

    def __iter__(self):
        gen_f = self._iter_gen_func
        for generator_f in self._generator_fs:
            gen_f = generator_f(gen_f)
        return gen_f()

    def __len__(self):
        if not self._doc_len:
            self._doc_len = sum(1 for _ in iter(self))
        return self._doc_len

    def get_fixed_len_sequenced_iter(self, seq_len):
        doc_gen = iter(self)
        while True:
            seq_list = []
            try:
                for _ in range(seq_len):
                    seq_list.append(next(doc_gen))
            except:
                # print("doc sequenced iter finished")
                break
                # raise ValueError("document already empty")
            yield tuple(seq_list)
        pad_token = PAD if self._token_type == dt.WORD_TYPE else PAD_ID
        yield tuple(seq_list + [pad_token]*(seq_len - len(seq_list)))

    def get_stop_token_sequenced_iter(self, stop_token, include_stop=False):
        doc_gen = iter(self)
        seq_list = []
        while True:
            try:
                item = next(doc_gen)
                if item == stop_token:
                    if include_stop:
                        seq_list.append(item)
                    yield seq_list
                    seq_list = []
                else:
                    seq_list.append(item)
            except StopIteration:
                break
        if len(seq_list) > 0:
            yield seq_list


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
            # self._iter_gen_func = dt.word2id_gen_f(self, self._vocab)
            self._generator_fs.append(dt.word2id_gen_f(self._vocab))
            self._token_type = dt.ID_TYPE
        elif self._token_type == dt.ID_TYPE:
            # self._iter_gen_func = dt.id2word_gen_f(self, self._vocab)
            self._generator_fs.append(dt.id2word_gen_f(self._vocab))
            self._token_type = dt.WORD_TYPE
        else:
            raise ValueError("This type of token does not support toggle word/id")

    def mask_unk(self):
        assert self._vocab is not None
        def mask_unk_f(gen_f):
            def gen():
                iter_gen = gen_f()
                if self.token_type == dt.WORD_TYPE:
                    unk_signature = UNK
                    unk_check_f = self._vocab.check_word_exist
                else:
                    raise NotImplementedError("Not implemented yet")
                for token in iter_gen:
                    if not unk_check_f(token):
                        yield unk_signature
                    else:
                        yield token
            return gen
        self._generator_fs.append(mask_unk_f)


class DocumentTransformState(collections.namedtuple(
    "DocumentTransformState",
    ("docs", "transformer", "size")
    )):
    pass


