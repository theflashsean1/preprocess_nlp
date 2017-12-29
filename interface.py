import abc


class DocumentState(object):
    def __init__(self):
        self._config_dict = {}
    
    def get_all_iter_keys():
        return self._config_dict.keys()

    def set_iter_config(self, iter_key, token_type, seq_len, creation_path):
        self._config_dict[iter_key] = (token_type, seq_len, creation_path)

    def clone_config(self, doc_state):
        for key in doc_state.get_all_iter_keys():
            self.set_iter_config(key, doc_state.get_token_type[key], doc_state.get_seq_len[key])
    
    def get_token_type(self, iter_key):
        return self._config_dict.get(iter_key)[0]

    def get_seq_len(self, iter_key):
        return self._config_dict.get(iter_key)[1]

    def get_creation_path(self, iter_key):
        return self._config_dict.get(iter_key)[2]

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

