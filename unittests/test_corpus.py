import unittest
import os
import pdb
from preprocess_nlp.doc import Document
from preprocess_nlp.doc_transformer import IdentityTransform, Word2VecTransform, DocLabelsTransform, DocLabelsPadTransform
import preprocess_nlp.doc_token as dt
import preprocess_nlp.doc_format.txt as dtxt
from preprocess_nlp.vocab_utils.common import VocabReader, VocabCreator


class TestDocument(unittest.TestCase):

    def setUp(self):
        self._doc_test1 = Document.create_from_txt("mock_files/short_doc.txt", dt.WORD_TYPE)
        self._doc_test1.set_label("label", 0)
        self._doc_test2 = Document.create_from_txt("mock_files/short_doc.txt", dt.WORD_TYPE, eol_type=dtxt.IGNORE_EOL)
        self._doc_test2.set_label("label", 1)
        self._doc1 = Document.create_from_txt("ptb/ptb.train.txt", dt.WORD_TYPE)
        self._doc1.set_label("label", 1)
        self._doc2 = Document.create_from_txt("ptb/ptb.valid.txt", dt.WORD_TYPE)
        self._doc2.set_label("label", 0)
        self._doc3 = Document.create_from_txt("ptb/ptb.test.txt", dt.WORD_TYPE)
        self._doc3.set_label("label", 1)
        if not os.path.exists("unittests/mock_files/short_doc_vocab.txt"):
            with VocabCreator(10000, "mock_files/short_doc_vocab.txt",
                                     "mock_files/short_doc_vocab_counts.txt") as v_creator:
                with open("mock_files/short_doc.txt") as f:
                    for line in f:
                        v_creator.update_vocab_from_sentence(line)
        self._test_doc_vocab = VocabReader("mock_files/short_doc_vocab.txt",
                                           "mock_files/short_doc_vocab_counts.txt")
        self._doc_toggle = Document.create_from_txt("mock_files/short_doc.txt", dt.WORD_TYPE)
        self._doc_toggle_word = Document.create_from_txt("mock_files/short_doc.txt", dt.WORD_TYPE)


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
        word2vec_transformer = Word2VecTransform(2, dt.WORD_TYPE)
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

        w2v_iter2 = word2vec_transformer.get_iters(self._doc1, self._doc2)
        center, context = next(w2v_iter2)
        self.assertEqual(center, "aer")
        self.assertEqual(context, "banknote")

        center, context = next(w2v_iter2)
        self.assertEqual(center, "consumers")
        self.assertEqual(context, "may")

        center, context = next(w2v_iter2)
        self.assertEqual(center, "aer")
        self.assertEqual(context, "berlitz")

        center, context = next(w2v_iter2)
        self.assertEqual(center, "consumers")
        self.assertEqual(context, "want")

    def test_doc_labels_gen(self):
        doc_labels_transformer = DocLabelsPadTransform(batch_size=3, seq_len=2, token_type=dt.WORD_TYPE)
        text_classify_iter = doc_labels_transformer.get_iters(self._doc_test1, self._doc1,
                                                              self._doc_test2, self._doc2, self._doc3)
        seq, label, eod_flag = next(text_classify_iter)
        self.assertEqual(seq[0], "hello")
        self.assertEqual(seq[1], "why")
        self.assertEqual(label, 0)
        self.assertEqual(eod_flag, 0)

        seq, label, eod_flag = next(text_classify_iter)
        self.assertEqual(seq[0], "aer")
        self.assertEqual(seq[1], "banknote")
        self.assertEqual(label, 1)
        self.assertEqual(eod_flag, 0)

        seq, label, eod_flag = next(text_classify_iter)
        self.assertEqual(seq[0], "hello")
        self.assertEqual(seq[1], "why")
        self.assertEqual(label, 1)
        self.assertEqual(eod_flag, 0)

        seq, label, eod_flag = next(text_classify_iter)
        self.assertEqual(seq[0], "when")
        self.assertEqual(seq[1], "then")
        self.assertEqual(label, 0)
        self.assertEqual(eod_flag, 0)

    def test_vocab_gen(self):
        self.assertEqual(self._test_doc_vocab.vocab_size, (9+4))
        self.assertEqual(self._test_doc_vocab.word2id_lookup("<unk>"), 0)
        self.assertEqual(self._test_doc_vocab.word2id_lookup("<s>"), 1)
        self.assertEqual(self._test_doc_vocab.word2id_lookup("</s>"), 2)
        self.assertEqual(self._test_doc_vocab.word2id_lookup("<pad>"), 3)
        self.assertEqual(self._test_doc_vocab.id2word_lookup(0), "<unk>")
        self.assertEqual(self._test_doc_vocab.id2word_lookup(1), "<s>")
        self.assertEqual(self._test_doc_vocab.id2word_lookup(2), "</s>")
        self.assertEqual(self._test_doc_vocab.id2word_lookup(3), "<pad>")

    def test_token_toggle(self):
        self._doc_toggle.set_vocab(self._test_doc_vocab)
        self._doc_toggle.toggle_word_id()
        iterator = iter(self._doc_toggle)
        self.assertEqual(next(iterator), 6)
        self.assertEqual(next(iterator), 5)

        self._doc_toggle.toggle_word_id()
        iterator1 = iter(self._doc_toggle)
        iterator2 = iter(self._doc_toggle_word)
        for _ in range(len(self._doc_toggle)):
            self.assertEqual(next(iterator1), next(iterator2))
    

if __name__ == '__main__':
    unittest.main()

