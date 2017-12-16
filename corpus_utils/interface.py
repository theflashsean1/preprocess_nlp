import abc

class DocumentState():
    """Please implement __iter__, and static factory method for creating itself"""
    @abc.abstractmethod
    def save_transformed_doc(self, new_doc_gen, new_doc_name):
        pass


class TokenState():
    @abc.abstractmethod
    def toggle_word_id_gen(self, document, vocabulary):
        """
        :return: list that contains transformed tokens
        """
        pass


class SeqState():
    @abc.abstractmethod
    def gen(self, doc_gen):
        pass

