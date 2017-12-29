import os
from preprocess_nlp.corpus_utils.tfrecords_corpus import TfrecordsDocumentState
from preprocess_nlp.corpus_utils.interface import DocumentState, TokenState
from preprocess_nlp.vocab_utils.common import UNK, SOS, EOS, EOS_ID
import numpy as np
import collections


class Document(object):
    seq_func_table = {
        "docs_labels": (doc_labels_gen, ["batch_size", "seq_len", "merging_docs", "label_keys", "num_examples"])        
    }

    @classmethod 
    def create_from_file(cls, document_path, token_state_type, vocab=None):
        if not os.path.exists(document_state_path):
            raise IOError("file not found")
        token_state = cls.create_token_state(token_state_type)
        if document_state_path.endswith(".txt"):
            document_state = TxtDocumentState(token_state.token_type)
        elif document_path.endswith(".tfrecords"):
            document_state = TfrecordsDocumentState(token_state.token_type)
        elif document_path.endswith(".npy"):
            document_state = None
        else:
            raise ValueError("Not valid document state type")
        doc_gen_f = document_state.doc_gen_func()
        return cls(doc_gen_f, document_state, token_state, vocab)

    @classmethod
    def create_from_iter(cls, document_iter, document_state_type, token_state_type, vocab=None):
        def doc_gen_f():
            for token in document_iter:
                yield token
        token_state = Document.create_token_state(token_state_type)
        if document_state_type == "txt":
            document_state = TxtDocumentState(token_state.token_type)
        elif document_state_type == "tfrecords":
            document_state = TfrecordsDocumentState(token_state.token_type)
        elif document_state_type == "npy":
            pass
        else:
            raise ValueError("Not valid document state type")
        token_state = cls.create_token_state(token_state_type)
        return cls(doc_gen_f, document_state, token_state, vocab)

    @staticmethod
    def create_token_state(token_state_type):
        if token_state_type == "word":
            token_state = WordTokenState()
        elif token_state_type == "id":
            token_state = IdTokenState()
        else:
            raise ValueError("Not valid token state type")
        return token_state

    def __init__(self, doc_gen_f, document_state, token_state, vocab=None):
        self._vocab = vocab
        self._iter_gen_func = doc_gen_f
        self._document_state = document_state
        self._token_state = token_state
        self._label_dict = {}

    def __str__(self):
        return str(self._document_state) 

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
        return self._token_state.token_type

    @property
    def doc_format(self):
        return self._document_state.doc_format

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
        self._iter_gen_func = self._token_state.toggle_word_id_gen_func(self._iter_gen_func(), self._vocab)
        new_token_state = WordTokenState() if isinstance(self._token_state, IdTokenState) else IdTokenState()
        self._token_state = new_token_state

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

    def convert2txt(self):
        self._document_state = TxtDocumentState(self.token_type)

    def convert2tfrecords(self):
        self._document_state = TfrecordsDocumentState(self.token_type)

    def convert2np_array(self):
        pass

    ##################
    # Output methods #
    ##################
    def save_seq(self, seq_config, new_doc_path):
        pass

    def iter_seq(self, seq_config):
        pass
    
    # TODO delete the following methods
    def iter_seq(self, seq_type, **kwargs):
        seq_func, seq_args = self.seq_func_table[seq_type]
        for key in seq_args:
            assert key in kwargs
        return seq_func(self._iter_gen_func(), **kwargs)

    def save_seq(self, seq_type, new_doc_path, **kwargs):
        doc_gen = self.iter_seq(seq_type, **kwargs)
        self._document_state.doc_save(doc_gen, new_doc_path, new_doc_path_sub)


class WordTokenState(TokenState):
    token_type = str

    @staticmethod
    def toggle_word_id_gen_func(document_iter, vocabulary):
        def toggle_word_id_gen():
            for word_token in document_iter:
                id_token = vocabulary.word2id_lookup(word_token)
                yield id_token  # Transformed
        return toggle_word_id_gen


class IdTokenState(TokenState):
    token_type = int

    @staticmethod
    def toggle_word_id_gen_func(document_iter, vocabulary):
        def toggle_word_id_gen():
            for id_token in document_iter:
                word_token = vocabulary.id2word_lookup(int(id_token))
                yield word_token  # Transformed
        return toggle_word_id_gen


class TxtDocumentState(DocumentState):
    """state that this class wil contain has to do with the token state dict
    and seq_state"""
    def __str__(self):
        return "Format: txt & Token type: " + str(self._token_type)

    @property
    def doc_format(self):
        return "txt"

    def doc_gen_func(self, doc_path):
        def doc_gen():
            with open(doc_path) as f:
                for line in f:
                    tokens = line.split()
                    for token in tokens:
                        yield token
                    if self._token_type == str:
                        yield EOS
                    else:
                        yield EOS_ID
        return doc_gen

    def doc_save(self, iter_dict, iter_path):
        pass

    """
    def doc_save(self, doc_line_gen, doc_path, doc_path_sub=None):
        if not doc_path_sub:
            self._doc_save_src(doc_line_gen, doc_path)
        else: self._doc_save_src_tgt(doc_line_gen, doc_path, doc_path_sub)
    
    def _doc_save_src(self, doc_line_gen, doc_path):
        with open(doc_path, "w") as f:
            for line_list in doc_line_gen:
                f.write(" ".join(line_list) + "\n")

    def _doc_save_src_tgt(self, doc_line_gen, doc_path, doc_path_sub):
        with open(doc_path, "w") as src_f, open(doc_path_sub, "w") as tgt_f:
            for src_line_list, tgt_line_list in doc_line_gen:
                src_f.write(" ".join(src_line_list) + "\n")
                tgt_f.write(" ".join(tgt_line_list) + "\n")
    """



