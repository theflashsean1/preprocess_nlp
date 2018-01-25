UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2
PAD_ID = 3


class VocabCreator:

    def __init__(self, max_vocab_size, out_vocab_f_path, out_count_f_path, 
                                   prepend_special_tokens=[UNK, SOS, EOS, PAD]):
        self._opened_vocab_file = open(out_vocab_f_path, "w")
        self._vocab = {}
        self._max_vocab_size = max_vocab_size
        if out_count_f_path:
            self._opened_count_file = open(out_count_f_path, "w") 
        else:
            self._opened_count_file = None
        self._special_tokens = prepend_special_tokens
        

    def __enter__(self):
        return self

    def update_vocab_from_word(self, word):
        if not (word in self._special_tokens):
            self._vocab[word] = self._vocab[word]+1 if word in self._vocab else 1

    def update_vocab_from_sentence(self, sentence):
        for word in sentence.split():
            self.update_vocab_from_word(word)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for special_token in self._special_tokens:
            self._opened_vocab_file.write(special_token + "\n")
            self._opened_count_file.write("0\n")

        for word, count in sorted(self._vocab.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:self._max_vocab_size]:
            self._opened_vocab_file.write(word + "\n")
            if self._opened_count_file is not None:
                self._opened_count_file.write(str(count) + "\n")

        self._opened_vocab_file.close()
        if self._opened_count_file:
            self._opened_count_file.close()


class VocabReader:
    def __init__(self, vocab_f_path, count_f_path=None):
        with open(vocab_f_path) as f:
            words = f.read().strip().split()
            ids = range(len(words))
            self._vocab_size = len(words)
            self._id2word_table = words
            self._word2id_table = dict(zip(words, ids))
            self._vocab_counts = None
            if count_f_path is not None:
                with open(count_f_path) as count_f:
                    counts = count_f.read().strip().split()
                    self._vocab_counts = [int(count) for count in counts]
    
    def id2word_lookup(self, id_token):
        if id_token >= self.vocab_size:
            return UNK
        return self._id2word_table[id_token] 

    def word2id_lookup(self, word_token):
        return self._word2id_table.get(word_token, UNK_ID)

    def check_word_exist(self, word_token):
        return (word_token in self._word2id_table)

    def check_id_exist(self, id_token):
        return (id_token >= 0) and (id_token < len(self._id2word_table))

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab_counts_list(self):
        return self._vocab_counts

