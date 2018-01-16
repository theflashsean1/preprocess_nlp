import unittest
import pdb
from preprocess_nlp.doc import Document
from preprocess_nlp.doc_transformer import IdentityTransform, Word2VecTransform
import preprocess_nlp.doc_token as dt
import preprocess_nlp.doc_format.txt as dtxt

class TestDocument(unittest.TestCase):

    def setUp(self):
        self._doc_test1 = Document.create_from_txt("mock_files/short_doc.txt", dt.WORD_TYPE)
        self._doc_test2 = Document.create_from_txt("mock_files/short_doc.txt", dt.WORD_TYPE, eol_type=dtxt.IGNORE_EOL)
        self._doc1 = Document.create_from_txt("ptb/ptb.train.txt", dt.WORD_TYPE)
        self._doc2 = Document.create_from_txt("ptb/ptb.valid.txt", dt.WORD_TYPE)
        self._doc3 = Document.create_from_txt("ptb/ptb.test.txt", dt.WORD_TYPE)

    def test_basic_doc_info(self):
        self.assertEqual(self._doc1.token_type, dt.WORD_TYPE)
        self.assertEqual(len(self._doc_test1), 14)
        self.assertEqual(len(self._doc_test2), 11)

    def test_merge_doc(self):
        merged_doc = Document.create_from_docs(self._doc1, self._doc2, self._doc3)
        merged_doc_gen = iter(merged_doc)
        for token in self._doc1:
            self.assertEqual(token, next(merged_doc_gen))
        for token in self._doc2:
            self.assertEqual(token, next(merged_doc_gen))
        for token in self._doc3:
            self.assertEqual(token, next(merged_doc_gen))

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

