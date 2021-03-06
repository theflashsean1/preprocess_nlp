import nltk
from preprocess_nlp.file_utils import common


def report_tokenize(report_path, mode="word2vec"):
    write_f_path = common.extend_path_basename(report_path, "tokenized_"+mode)
    with open(report_path) as read_f:
        with open(write_f_path, "w") as write_f:
            for line in read_f:
                words = line.split()
                if len(words)!=0:
                    write_f.write(" ".join([word.lower() for word in words])+"\n")
                
    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokens = nltk.tokenize.word_tokenize(line)
            for token in tokens:
                #if any(char.isdigit() for char in token):
                if token.lstrip('-').replace('.', '', 1).isdigit() and mode=="word2vec":
                    # int, neg, float
                    new_tokens.append("num")
                elif token == ".":
                    new_tokens.append("eos")
                else:
                    new_tokens.append(token)
                if token[-1] == ",":
                    new_tokens.append("comma")

            f.write(" ".join(new_tokens) + "\n")

    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(line)
            for i, token in enumerate(tokens):
                if token == "num" and mode=="word2vec":
                    tokens[i] = "<num>"
                elif token == "eos":
                    tokens[i] = "</s>"
                elif token == "comma":
                    tokens[i] = ","
            f.write(" ".join(tokens) + "\n")
    return write_f_path



