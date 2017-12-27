import abc


class DocumentState(object):
    def __init__(self, token_type):
        self._token_type = token_type

    def update_token_type(self, token_type):
        self._token_type = token_type

    @abc.abstractmethod
    def doc_gen_func(self, doc_path):
        pass

    @abc.abstractmethod
    def doc_save(self, doc_gen, doc_path, doc_path_sub=None):
        pass

    @abc.abstractmethod
    def doc_save_with_label(self, doc_gen, labels_dict, doc_path, doc_path_sub=None):
        pass


class TokenState(object):
    @property
    @abc.abstractmethod
    def token_type(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def toggle_word_id_gen_func(document, vocabulary):
        """
        :return: list that contains transformed tokens
        """
        pass

