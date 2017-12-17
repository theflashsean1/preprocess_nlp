import unittest
import pdb
import sys
sys.path.append('..')
from corpus_utils.src import Document


class TestDocument(unittest.TestCase):

    def setUp(self):
        self._doc1 = Document("ptb/ptb.train.txt", "word")
        self._doc2 = Document("ptb/ptb.train.txt", "word")

    def test_basic_doc_info(self):
        self.assertEqual(self._doc1.token_type, str)
        self.assertEqual(self._doc1.doc_format, "txt")

    def test_raw_gen(self):
        raw_iter = self._doc1.iter_seq("raw")
        next(raw_iter) == "aer"
        next(raw_iter) == "banknote"
        next(raw_iter) == "berlitz"

    def test_modify_doc_info(self):
        pass

if __name__ == '__main__':
    unittest.main()

