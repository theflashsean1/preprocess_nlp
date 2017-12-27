import os
from preprocess_nlp.corpus_utils.tfrecords_corpus import TfrecordsDocumentState
from preprocess_nlp.corpus_utils.interface import DocumentState, TokenState
from preprocess_nlp.vocab_utils.common import UNK, SOS, EOS, EOS_ID
import numpy as np

"""
When the following seq gen functions are called, assume that doc_gen only has
standarlized tokens, and "\n", for example, has been replaced with "</s>"
i.e. document_state has the responsibility for how to represent </s> but not
the seq functions

How to handle new line ?
"""
def raw_gen(doc_gen):
    return doc_gen


def word2vec_center_context_gen(doc, window_size, max_num_examples):
    doc_gen = iter(doc)
    num_example = 0
    center_word = next(doc_gen)
    forward_context_words = []
    backward_context_words = []
    for _ in range(window_size):
        forward_context_words.append(next(doc_gen))
    while len(forward_context_words)!=0 and num_example<max_num_examples:
        for context_word in backward_context_words + forward_context_words:
            yield [center_word], [context_word]
        num_example += 1
        backward_context_words.append(center_word)
        center_word = forward_context_words.pop(0)
        try:
            next_word = next(doc_gen)
            forward_context_words.append(next_word)
        except StopIteration:
            pass
        if len(backward_context_words) > window_size:
            backward_context_words.pop(0)
        

def rnn_lang_model_gen(doc, batch_size, seq_len, nb_epochs=1):
    doc_gen = iter(doc)
    data = np.array([token for token in doc_gen])
    data_len = data.shape[0]
    nb_batches = (data_len - 1)//(batch_size*seq_len)
    if nb_batches == 0:
        raise ValueError("Not enough data for even a single batch")
    rounded_data_len = nb_batches*batch_size*seq_len
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches*seq_len])
    ydata = np.reshape(data[1:rounded_data_len+1], [batch_size, nb_batches*seq_len])

    for epoch_id in range(nb_epochs):
        for batch_id in range(nb_batches):
            x = xdata[:, batch_id*seq_len:(batch_id+1)*seq_len]
            y = ydata[:, batch_id*seq_len:(batch_id+1)*seq_len]
            x = np.roll(x, -epoch_id, axis=0)
            y = np.roll(y, -epoch_id, axis=0)
            for x_one_seq, y_one_seq in zip(x, y):
                yield x_one_seq, y_one_seq


def doc_labels_gen(doc, batch_size, seq_len, merging_docs, label_keys, num_examples): 
    for doc_ in merging_docs:
        assert doc.token_type == doc_.token_type
    doc_seq_iters = [doc_.get_sequenced_iter(seq_len) for doc_ in [doc]+merging_docs]
    doc_label_dict = [{k: doc.get_label(k) for k in label_keys} for doc_ in [doc] + merging_docs]
    num_docs = len(docs)
    count = 0
    while count < num_examples:
        index = count%num_docs
        yield next(doc_seq_iters[index]), doc_label_dict[index]
        count += 1


class Document(object):
    seq_func_table = {
        "raw": (raw_gen, []),
        "word2vec": (word2vec_center_context_gen, ["window_size", "max_num_examples"]),
        "rnn_lang_model": (rnn_lang_model_gen, ["batch_size", "seq_len", "nb_epochs"]),
    }
    seq_label_func_table = {
        "docs_labels": (doc_labels_gen, ["batch_size", "seq_len", "merging_docs", "label_keys", "num_examples"])        
    }

    def __init__(self, document_state_path, token_state_type, vocab=None):
        if not os.path.exists(document_state_path):
            raise IOError("file not found")
        self._vocab = vocab
        self._label_dict = {}
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
    def iter_seq(self, seq_type, **kwargs):
        seq_func, seq_args = self.seq_func_table[seq_type]
        for key in seq_args:
            assert key in kwargs
        return seq_func(self._iter_gen_func(), **kwargs)

    def save_seq(self, seq_type, new_doc_path, new_doc_path_sub=None, **kwargs):
        doc_gen = self.iter_seq(seq_type, **kwargs)
        self._document_state.doc_save(doc_gen, new_doc_path, new_doc_path_sub)

    def iter_seq_with_labels(self, seq_type, **kwargs):
        seq_func, seq_args = self.seq_label_func_table[seq_type]
        for key in seq_args:
            assert key in kwargs
        return seq_func(self._iter_gen_func(), **kwargs)

    def save_seq_with_labels(self, seq_type, new_doc_path, new_doc_path_sub=None, **kwargs):
        doc_gen = self.iter_seq_with_labels(seq_type, **kwargs)
        self._document_state.doc_save_with_label(doc_gen, new_doc_path, new_doc_path_sub)


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




