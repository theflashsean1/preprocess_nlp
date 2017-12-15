import abc

class Document():

    def __init__(self, document_state):
        self._token_state = None
        self._document_state = None
        self._seq_state = None

    def __str__(self):
        pass

    def __iter__(self):
        return iter(self._document_state)


    @property
    def token_state(self):
        return self._token_state

    @property
    def document_state(self):
        return self._document_state

    @property
    def seq_state(self):
        return self._seq_state

    def change_token_state(self, token_state):
        self._token_state = token_state

    def change_document_state(self, document_state):
        self._document_state = document_state

    def change_seq_state(self, seq_state):
        self._seq_state = seq_state

    def toggle_word_id(self, new_doc_name):
        transformed_doc = self._token_state.toggle_word_id()
        self._document_state.save_transformed_doc(transformed_doc)

    def convert2txt(self, new_doc_name):
        pass

    def convert2tfrecords(self, new_doc_name):
        pass

    def convert2np_array(self, new_doc_name):
        pass

    def convert2raw_seq(self, new_doc_name):
        pass

    def convert2batched_seq(self, new_doc_name):
        pass

class TokenState():
    def __init__(self, document):
        self._document = document

    @abc.abstractmethod
    def toggle_word_id(self):
        """
        :return: list that contains transformed tokens
        """
        pass

class WordTokenState(TokenState):

    @property
    def token_type(self):
        return str

    def toggle_word_id(self):
        for word in self._document:
            # Use vocabulary to get id
            # return id list
            yield word  # Transformed



class DocumentState():

    @abc.abstractmethod
    def convert2txt(self):
        pass

    @abc.abstractmethod
    def convert2tfrecords(self):
        pass

    @abc.abstractmethod
    def convert2np_array(self):
        pass

    @abc.abstractmethod
    def save_transformed_doc(self, new_doc):
        pass


class TxtDocumentState(DocumentState):
    def __init__(self, txt_path):
        self._txt_path = txt_path

    def __iter__(self):
        with open(self._txt_path) as f:
            tokens = f.split()
            for token in tokens:
                yield token

    def convert2txt(self):
        print("Already in txt format")

    def convert2tfrecords(self):
        pass
        
    def convert2np_array(self):
        pass

    def save_transformed_doc(self, new_doc):
        # Just write new_doc array in txt file
        pass

class TfrecordsDocumentState(DocumentState):
    def __init__(self, txt_path):
        self._txt_path = txt_path

    def __iter__(self):
        # generator yield
        pass

    def convert2txt(self):
        pass

    def convert2tfrecords(self):
        pass

    def convert2np_array(self):
        pass

    def save_transformed_doc(self, new_doc):
        # data type needs to be known, which
        # shall be obtained from self.token_state.token_type
        # Use it to decide "Feature" for tfrecords
        pass


class SeqState():
    def convert2raw_seq(self):
        pass

    def convert2batched_seq(self):
        pass
