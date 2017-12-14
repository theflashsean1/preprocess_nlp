UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class VocabCreator:

    def __init__(self, vocab_size, out_vocab_f_path, out_count_f_path, 
                                   prepend_special_tokens=[UNK, SOS, EOS]):
        self._opened_vocab_file = open(out_vocab_f_path, "w")
        self._vocab = {}
        self._vocab_size = vocab_size
        if out_count_f_path:
            self._opened_count_file = open(out_count_f_path, "w") 
        else:
            self._opened_count_file = None
        self._special_tokens = prepend_special_tokens
        

    def __enter__(self):
        return self


    def update_vocab_from_sentence(self, sentence):
        for word in sentence.split():
            if word in self._special_tokens:
                continue
            self._vocab[word] = self._vocab[word]+1 if word in self._vocab else 1


    def __exit__(self, exc_type, exc_val, exc_tb):
        for special_token in self._special_tokens:
            self._opened_vocab_file.write(special_token + "\n")
            self._opened_count_file.write("0\n")

        for word, count in sorted(self._vocabulary.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:self._vocab_size]:
            self._opened_vocab_file.write(word + "\n")
            if self._opened_count_file:
                self._opened_count_file(str(count) + "\n")

        self._opened_vocab_file.close()
        if self._opened_count_file:
            self._opened_count_file.close()
