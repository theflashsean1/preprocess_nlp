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
    def create_from_txt(cls, txt_path, token_type, eol_type=dtxt.YIELD_EOL, vocab_reader=None):
        dt.assert_type_valid(token_type)
        if not os.path.exists(txt_path):
            raise IOError("file not found")
        flag_tokens = []
        if eol_type == dtxt.YIELD_EOL:
            doc_gen_f = dtxt.doc_gen_f_yield_eol(txt_path, token_type)
        elif eol_type == dtxt.IGNORE_EOL:
            doc_gen_f = dtxt.doc_gen_f_ignore_eol(txt_path, token_type)
        elif eol_type == dtxt.KEEP_EOL_NL:
            doc_gen_f = dtxt.doc_gen_f_keep_eol_nl(txt_path, token_type)
            flag_tokens.append("\n")
        else:
            raise ValueError("Non existing end of line type")
        return cls(doc_gen_f, token_type, flag_tokens, vocab)

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

    def __init__(self, src_gen_f, token_type,
                 flag_tokens=None, vocab_reader=None):
        self._vocab_reader = vocab_reader
        self._token_type = token_type
        self._label_dict = {}
        self._doc_len = None
        self._f_gen_fs = []
        self._src_gen_f = src_gen_f
        self._applied_flag_tokens = flag_tokens

    def __iter__(self):
        gen_f = self._src_gen_f
        for f_gen_f in self._f_gen_fs:
            gen_f = f_gen_f(gen_f)
        return gen_f()

    def __len__(self):
        if not self._doc_len:
            self._doc_len = sum(1 for _ in iter(self))
        return self._doc_len

    @property
    def token_type(self):
        return self._token_type

    @property
    def applied_flag_tokens(self):
        return self._applied_flag_tokens

    @property
    def is_flag_token_applied(self):
        return len(self.applied_flag_tokens) > 0

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
        assert self._vocab_reader is not None
        if self._token_type == dt.WORD_TYPE:
            self._f_gen_fs.append(dt.create_word2id_f_gen_f(
                self._vocab, self._applied_flag_tokens))
            self._token_type = dt.ID_TYPE
        elif self._token_type == dt.ID_TYPE:
            self._f_gen_fs.append(dt.create_id2word_f_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = dt.WORD_TYPE
        else:
            raise ValueError("This type of token does not support toggle word/id")

    def convert_embed(self):
        assert self._vocab_reader is not None
        if self._token_type == dt.WORD_TYPE:
            self._f_gen_fs.append(dt.create_word2embed_f_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = dt.EMBED_TYPE
        elif self._token_type == dt.ID_TYPE:
            self._f_gen_fs.append(dt.create_id2embed_f_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = dt.EMBED_TYPE
        else:
            raise ValueError("This type of token does not support toggle word/id")

    def skip_tokens(self, bool_token_transformers):
        left_len, right_len = dt.get_max_context_lens(
            bool_token_transformers
        )

        def skip_tokens_f_gen_f(gen_f):
            def gen():
                left = []
                right = []
                token_gen = gen_f()
                try:
                    center = next(token_gen)
                except StopIteration:
                    return
                while center is not None:
                    skip_flag = False
                    if len(left) == left_len and len(right) == right_len:
                        for transformer in bool_token_transformers:
                            skip_flag = transformer[left, center, right]
                            if skip_flag:
                                break
                        if not skip_flag:
                            yield center
                        center = dt.shift_context_center_tokens(
                            (left, center, right),
                            token_gen, left_len, right_len)
                    else:
                        for transformer in bool_token_transformers:
                            if transformer.is_applicable(len(left), len(right)):
                                skip_flag = transformer[left, center, right]
                                if skip_flag:
                                    break
                        if not skip_flag:
                            if len(right) < right_len:
                                try:
                                    right.append(next(token_gen))
                                    continue
                                except StopIteration:
                                    pass
                            yield center
                            center = dt.shift_context_center_tokens(
                                (left, center, right),
                                token_gen, left_len, right_len)
                        else:
                            center = dt.shift_context_center_tokens(
                                (left, center, right),
                                token_gen, left_len, right_len)
            return gen
        self._f_gen_fs.append(skip_tokens_f_gen_f)

    def replace_tokens(self, token_transformers):
        pass

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


class SeqDocument(object):
    @classmethod
    def create_len_separated_seq_doc(cls, doc, fixed_len):
        assert not doc.is_flag_token_applied

        def fixed_len_gen(doc_):
            doc_gen = iter(doc_)
            while True:
                seq_list = []
                try:
                    for _ in range(fixed_len):
                        seq_list.append(next(doc_gen))
                except:
                    break
                yield tuple(seq_list), "<len=" + str(fixed_len) + ">"
            pad_token = PAD if doc.token_type == dt.WORD_TYPE else PAD_ID
            final_seq = tuple(seq_list + [pad_token]*(fixed_len - len(seq_list)))
            yield final_seq, "<len=" + str(fixed_len) + ">"
        return cls(doc, fixed_len_gen)

    @classmethod
    def create_flag_separated_seq_doc(cls, doc):
        assert doc.is_flag_token_applied

        def flag_gen(doc_):
            doc_gen = iter(doc_)
            seq_list = []
            while True:
                try:
                    item = next(doc_gen)
                    if item in doc.applied_flag_tokens:
                        yield seq_list, item
                        seq_list = []
                    else:
                        seq_list.append(item)
                except StopIteration:
                    break
            if len(seq_list):
                yield seq_list, "<final>"
        return cls(doc, flag_gen)

    def __init__(self, doc, seq_gen_f):
        self._doc = doc
        self._seq_gen_f = seq_gen_f

    def __iter__(self):
        for seq_list, flag in self._seq_gen_f(self._doc):
            yield seq_list, flag

    def update_flag_tokens(self, flag_token_transformers):
        left_len, right_len = dt.get_max_context_lens(
            flag_token_transformers
        )

        def flag_gen(doc_):
            left = []
            right = []
            token_gen = self._seq_gen_f(doc_)
            try:
                seq, center = next(token_gen)
            except StopIteration:
                return
            while center is not None:
                new_center = center
                if len(left) == left_len and len(right) == right_len:
                    for transformer in flag_token_transformers:
                        new_center = transformer[left, center, right]
                        if new_center != center:
                            break
                    yield seq, new_center
                    center = dt.shift_context_center_tokens(
                        (left, center, right),
                        token_gen, left_len, right_len)
                else:
                    for transformer in flag_token_transformers:
                        if transformer.is_applicable(len(left), len(right)):
                            new_center = transformer[left, center, right]
                            if new_center != center:
                                break
                    if new_center == center:
                        if len(right) < right_len:
                            try:
                                right.append(next(token_gen))
                                continue
                            except StopIteration:
                                pass
                        yield seq, new_center
                        center = dt.shift_context_center_tokens(
                            (left, center, right),
                            token_gen, left_len, right_len)
                    else:
                        yield seq, new_center
                        center = dt.shift_context_center_tokens(
                            (left, center, right),
                            token_gen, left_len, right_len)
        self._seq_gen_f = flag_gen


class DocumentTransformState(collections.namedtuple(
    "DocumentTransformState",
    ("docs", "transformer", "size")
    )):
    pass
