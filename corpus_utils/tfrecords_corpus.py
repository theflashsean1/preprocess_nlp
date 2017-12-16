import tensorflow as tf
from corpus_utils.interface import DocumentState

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_features(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))



class TfrecordsDocumentState(DocumentState):
    def __init__(self, tfrecords_path, token_type):
        self._tfrecords_path = tfrecords_path
        self._token_type = token_type

    def __iter__(self):
        pass

    def save_transformed_doc(self, new_doc_gen, new_doc_path, new_doc_path_sub=None):
        pass



