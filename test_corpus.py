from corpus_utils.src import Document
from vocab_utils.common import VocabCreator


doc = Document("unittests/ptb/ptb.train.txt", "word")
raw_iter = doc.iter_seq("raw")
with VocabCreator(100, "vocab.txt", "vocab_counts.txt") as vocab_creator:
    for token in raw_iter:
        vocab_creator.update_vocab_from_word(token)
