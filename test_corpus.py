from corpus_utils.src import Document
from vocab_utils.common import VocabCreator


doc = Document("unittests/ptb/ptb.train.txt", "word")
""" 
raw_iter = doc.iter_seq("raw")
with VocabCreator(100, "vocab.txt", "vocab_counts.txt") as vocab_creator:
    for token in raw_iter:
        vocab_creator.update_vocab_from_word(token)
"""


#doc.save_seq("rnn_lang_model", "rnn_src.txt", "rnn_tgt.txt",
#             batch_size=64, seq_len=15, nb_epochs=2)

#doc.save_seq("word2vec", "word2vec_src.txt", "word2vec_tgt.txt",
#             window_size=2, max_num_examples=1000)
