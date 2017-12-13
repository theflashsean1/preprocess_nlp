import nltk
import file_utils


def notes_tokenize(data_path):
    write_f_path = file_utils.common.extend_path_basename(data_path, "preprocessed")
    with open(data_path) as read_f:
        with open(write_f_path, "w") as write_f:
            for line in read_f:
                words = line.split()
                new_words = []
                for i, word in enumerate(words):
                    special_eos = False
                    if "\\n" in word and i<(len(words) - 1):
                        if ":" in words[i+1]:
                            word = word.replace("\\n", ".")
                        word = word.replace("\\n", " ")
                    new_words.append(word.lower())
                write_f.write(" ".join(new_words) + "\n")

    with file_utils.common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokens = nltk.tokenize.word_tokenize(line)
            for token in tokens:
                if ":" in token:
                    new_tokens.append("<colon>")
                elif any(char.isdigit() for char in token):
                    new_tokens.append("<num>")
                elif token == ".":
                    new_tokens.append("eos")
                else:
                    new_tokens.append(token)
            f.write(" ".join(new_tokens) + "\n")
    
    with file_utils.common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(line)
            for i, token in enumerate(tokens):
                if token == "num":
                    tokens[i] = "<num>"
                elif token == "colon":
                    tokens[i] = ":"
                elif token == "eos":
                    tokens[i] = "</s>"
                elif "_______" in token:
                    tokens[i] = ""
            f.write(" ".join(tokens) + "\n")
    return write_f_path


