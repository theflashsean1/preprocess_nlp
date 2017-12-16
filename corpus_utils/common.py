from corpus_utils.tfrecords_corpus import TfrecordsDocumentState
from corpus_utils.interface import DocumentState, TokenState, SeqState

class Document():

    def __init__(self, document_state_input, token_state_type,
                 document_state_type, seq_state_type, vocab=None):
        self._vocab = vocab
        if token_state_type == "word":
            self._token_state = WordTokenState()
        elif token_state_type == "id":
            self._token_state = IdTokenState()
        else:
            raise ValueError("Not valid token state type")

        if document_state_type == "txt":
            self._document_state = TxtDocumentState(document_state_input)
        elif document_state_type == "tfrecords":
            self._document_state = TfrecordsDocumentState(document_state_input)
        elif document_state_input == "numpy":
            pass
        else:
            raise ValueError("Not valid document state type")

        if seq_state_type == "raw_seq":
            pass
        elif seq_state_type == "batched_seq":
            pass
        elif seq_state_type == "word2vec":
            self._seq_state = Word2vecCenterContextSeqState()
        else:
            raise ValueError("Not valid Seq state type")

    def __str__(self):
        pass

    def __iter__(self):
        return self._seq_state.gen(iter(self._document_state))

    ####################
    # Client Interface #
    ####################
    @property
    def token_type(self):
        return self._token_state.token_type

    # Word <--> Id
    def toggle_word_id(self, new_doc_path):
        toggled_doc = self._token_state.toggle_word_id_gen(iter(self), self._vocab)
        new_token_state = WordTokenState() if isinstance(self._token_state, IdTokenState) else IdTokenState()
        self._token_state = new_token_state
        self._document_state.save_transformed_doc(toggled_doc, new_doc_path)


    # Txt, Tfrecords, Numpy array
    def convert2txt(self, new_doc_path):
        pass

    def convert2tfrecords(self, new_doc_path):
        self._document_state = TfrecordsDocumentState.create_tfrecords_document_state(iter(self),
                                                                                      new_doc_path,
                                                                                      self.token_type)
    def convert2np_array(self, new_doc_path):
        pass

    # Raw_sequence, Batched_seq
    def convert2raw_seq(self, new_doc_path):
        pass

    def convert2batched_seq(self, new_doc_path):
        pass


class WordTokenState(TokenState):

    @property
    def token_type(self):
        return str

    def toggle_word_id_gen(self, document_iter, vocabulary):
        for word in document_iter:
            # Use vocabulary to get id
            # return id list
            yield word  # Transformed


class IdTokenState(TokenState):

    @property
    def token_type(self):
        return int

    def toggle_word_id(self, document_iter, vocabulary):
        for word in document_iter:
            # Use vocabulary to get id
            # return id list
            yield word  # Transformed




class TxtDocumentState(DocumentState):
    def __init__(self, txt_path):
        self._txt_path = txt_path

    def __iter__(self):
        with open(self._txt_path) as f:
            for line in f:
                tokens = line.split()
                for token in tokens:
                    yield token

    @staticmethod
    def create_txt_document_state(prev_doc_gen, new_doc_path):
        with open(new_doc_path, "w") as f:
            for token in prev_doc_gen:
                f.write(token)
        return TxtDocumentState(new_doc_path)

    def save_transformed_doc(self, new_doc_gen, new_doc_path):
        with open(new_doc_path, "w") as f:
            for token in new_doc_gen:
                f.write(token)
        self._txt_path = new_doc_path


class RawSeqState():
    def gen(self, document_state_gen):
        pass

class Word2vecCenterContextSeqState():
    def __init__(self, window_size):
        self._window_size = window_size

    def gen(self, document_state_gen):
        pass


class RnnLanguageModelSeqState():
    def __init__(self, batch_size, seq_len):
        self._batch_size = batch_size
        self._seq_len = seq_len

    def gen(self, document_state_gen):
        pass
        # Traverse the entire doc to get all the items in memory (array)
        # Then compute nb_batches = doc_len/(batch_size * seq_len)
        # slice through and get src:tgt array
        # Finally cast it down to src: context format


