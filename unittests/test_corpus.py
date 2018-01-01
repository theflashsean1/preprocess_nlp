import unittest
import pdb
from preprocess_nlp.doc import Document
from preprocess_nlp.doc_transformer import IdentityTransform, Word2VecTransform
import preprocess_nlp.doc_token as dt



class TestDocument(unittest.TestCase):

    def setUp(self):
        self._doc1 = Document.create_from_file("ptb/ptb.train.txt", dt.WORD_TYPE)

    def test_basic_doc_info(self):
        self.assertEqual(self._doc1.token_type, dt.WORD_TYPE)

    def test_raw_gen(self):
        identity_transformer = IdentityTransform(dt.WORD_TYPE)
        raw_iter = identity_transformer.get_iters(self._doc1)
        self.assertEqual(next(raw_iter), "aer")
        self.assertEqual(next(raw_iter), "banknote")
        self.assertEqual(next(raw_iter), "berlitz")

    def test_word2vec_gen(self):
        word2vec_transformer = Word2VecTransform(2, 1000, dt.WORD_TYPE)
        w2v_iter = word2vec_transformer.get_iters(self._doc1)
        center, context = next(w2v_iter)
        self.assertEqual(center, "aer")
        self.assertEqual(context, "banknote")

        center, context = next(w2v_iter)
        self.assertEqual(center, "aer")
        self.assertEqual(context, "berlitz")

        center, context = next(w2v_iter)
        self.assertEqual(center, "banknote")
        self.assertEqual(context, "aer")


if __name__ == '__main__':
    unittest.main()

