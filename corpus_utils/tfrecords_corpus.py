import tensorflow as tf
from corpus_utils.interface import DocumentState

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_features(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))



class TfrecordsDocumentState(DocumentState):
    def __init__(self, tfrecords_path):
        self._tfrecords_path = tfrecords_path

    def __iter__(self):
        pass

    @staticmethod
    def create_tfrecords_document_state(prev_doc_gen, new_doc_name, token_type):
        pass

    def save_transformed_doc(self, new_doc_gen, new_doc_name):
        # data type needs to be known, which
        # shall be obtained from self.token_state.token_type
        # Use it to decide "Feature" for tfrecords
        pass



