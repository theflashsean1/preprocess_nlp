import nltk
from preprocess_nlp import file_utils


def report_tokenize(report_path):
    write_f_path = file_utils.common.extend_path_basename(report_path, "tokenized")
    with open(report_path) as read_f:
        with open(write_f_path, "w") as write_f:
            for line in read_f:
                words = line.split()
                if len(words)!=0:
                    write_f.write(" ".join([word.lower() for word in words])+"\n")
                
    with file_utils.common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokens = nltk.tokenize.word_tokenize(line)
            for token in tokens:
                #if any(char.isdigit() for char in token):
                if token.lstrip('-').replace('.', '', 1).isdigit():
                    # int, neg, float
                    new_tokens.append("num")
                elif token == ".":
                    new_tokens.append("eos")
                else:
                    new_tokens.append(token)
                if token[-1] == ",":
                    new_tokens.append("comma")

            f.write(" ".join(new_tokens) + "\n")

    with file_utils.common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(line)
            for i, token in enumerate(tokens):
                if token == "num":
                    tokens[i] = "<num>"
                elif token == "eos":
                    tokens[i] = "</s>"
                elif token == "comma":
                    tokens[i] = ","
            f.write(" ".join(tokens) + "\n")
    return write_f_path



