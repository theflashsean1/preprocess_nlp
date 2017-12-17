import os
from corpus_utils.tfrecords_corpus import TfrecordsDocumentState
from corpus_utils.interface import DocumentState, TokenState

def raw_gen(doc_gen):
    return doc_gen


def word2vec_center_context_gen(doc_gen, window_size):
    pass


def rnn_lang_model_gen(doc_gen, batch_size, seq_len):
    pass
    # Traverse the entire doc to get all the items in memory (array)
    # Then compute nb_batches = doc_len/(batch_size * seq_len)
    # slice through and get src:tgt array
    # Finally cast it down to src: context format


class Document(object):
    seq_func_table = {
        "raw": (raw_gen, []),
        "word2vec": (word2vec_center_context_gen, ["window_size"]),
        "rnn_lang_model": (rnn_lang_model_gen, ["batch_size", "seq_len"]),
    }

    def __init__(self, document_state_path, token_state_type, vocab=None):
        if not os.path.exists(document_state_path):
            raise IOError("file not found")
        self._vocab = vocab
        if token_state_type == "word":
            self._token_state = WordTokenState()
        elif token_state_type == "id":
            self._token_state = IdTokenState()
        else:
            raise ValueError("Not valid token state type")

        if document_state_path.endswith(".txt"):
            self._document_state = TxtDocumentState(self.token_type)
        elif document_state_path.endswith(".tfrecords"):
            self._document_state = TfrecordsDocumentState(self.token_type)
        elif document_state_path.endswith(".npy"):
            pass
        else:
            raise ValueError("Not valid document state type")
        self._iter_gen_func = self._document_state.doc_gen_func(document_state_path)
        self._gen_base_path = document_state_path

    def __str__(self):
        return str(self._document_state) + " & BasePath: " + self._gen_base_path

    ####################
    # Client Interface #
    ####################
    @property
    def token_type(self):
        return self._token_state.token_type

    @property
    def doc_format(self):
        return self._document_state.doc_format


    def set_vocab(self):
        self._vocab = vocab

    ##########################
    # State Changing methods #
    ##########################
    def toggle_word_id(self):
        assert self._vocab is not None
        self._iter_gen_func = self._token_state.toggle_word_id_gen_func(self._iter_gen_func(), self._vocab)
        new_token_state = WordTokenState() if isinstance(self._token_state, IdTokenState) else IdTokenState()
        self._token_state = new_token_state

    def convert2txt(self):
        self._document_state = TxtDocumentState(self.token_type)

    def convert2tfrecords(self):
        self._document_state = TfrecordsDocumentState(self.token_type)

    def convert2np_array(self):
        pass

    ##################
    # Output methods #
    ##################
    def iter_seq(self, seq_type, **kwargs):
        seq_func, seq_args = self.seq_func_table[seq_type]
        for key in seq_args:
            assert key in kwargs
        return seq_func(self._iter_gen_func(), **kwargs)

    def save_seq(self, seq_type, new_doc_path, new_doc_path_sub=None, **kwargs):
        seq_func, seq_args = self.seq_func_table[seq_type]
        for key in seq_args:
            assert key in kwargs
        doc_gen = seq_func(self._iter_gen_func(), **kwargs)
        self._document_state.doc_save(doc_gen, new_doc_path, new_doc_path_sub)


class WordTokenState(TokenState):

    @property
    def token_type(self):
        return str

    @staticmethod
    def toggle_word_id_gen_func(document_iter, vocabulary):
        def toggle_word_id_gen():
            for word_token in document_iter:
                id_token = self._vocabulary.word2id_lookup(word_token)
                yield id_token  # Transformed
        return toggle_word_id_gen


class IdTokenState(TokenState):

    @property
    def token_type(self):
        return int

    @staticmethod
    def toggle_word_id_gen_func(document_iter, vocabulary):
        def toggle_word_id_gen():
            for id_token in document_iter:
                word_token = self._vocabulary.id2word_lookup(int(id_token))
                yield word_token  # Transformed
        return toggle_word_id_gen


class TxtDocumentState(DocumentState):
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
        return doc_gen

    def doc_save(self, doc_gen, doc_path, doc_path_sub=None):
        if not doc_path_sub:
            self._doc_save_src(doc_gen, doc_path)
        else:
            self._doc_save_src_tgt(doc_gen, doc_path, doc_path_sub)
    
    def _doc_save_src(self, doc_gen, doc_path):
        with open(doc_path, "w") as f:
            for token in doc_gen:
                f.write(token)

    def _doc_save_src_tgt(self, doc_gen, doc_path, doc_path_sub):
        with open(doc_path, "w") as src_f, open(doc_path_sub, "w") as tgt_f:
            for src_token, tgt_token in zip(src_f, tgt_f):
                src_f.write(src_token)
                tgt_f.write(tgt_token)




