import nltk
import os
from preprocess_nlp.file_utils import common


def notes_tokenize_full(data_path, res_dir=None):
    if not res_dir:
        write_f_path = common.extend_path_basename(data_path, "fully_preprocessed")
    else:
        f_name = common.extend_file_basename(data_path, "fully_preprocessed")
        write_f_path = os.path.join(os.path.dirname(data_path), f_name)

    with open(data_path) as read_f:
        with open(write_f_path, "w") as write_f:
            for line in read_f:
                words = line.split()
                new_words = []
                i = 0
                while i < len(words):
                    word = words[i]
                    if "\\n" in word and i<(len(words) - 4):
                        # ... if the next word is "<word>: ", this new line is
                        # the actual end of sentence, we use . or </s> as used
                        # later
                        if ":" in words[i+1] or ":" in words[i+2] or ":" in words[i+3] or ":" in words[i+4]:
                            word = word.replace("\\n", " .")
                        word = word.replace("\\n", " ")
                    new_words.append(word.lower())
                    i+=1
                write_f.write(" ".join(new_words) + "\n")
    # STEP1 Replace all the new lines with space excpet ...
    # If [ found, skip everything until ]
    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            words = line.split()
            new_words = []
            i = 0
            while i < len(words):
                word = words[i]
                if "[" in word:
                    while i < len(words) and (not ("]" in words[i])):
                        i+=1
                    i+=1
                    continue
                if "(" in word:
                    while i < len(words) and (not (")" in words[i])):
                        i+=1
                    i+=1
                    continue
                new_words.append(word.lower())
                i+=1
            f.write(" ".join(new_words) + "\n")
    # STEP2 Tokenize all words, and replace some punctuations with words
    # so that they could be preserved when we use nltk to remove all other punc 
    with common.ReadReplaceOpen(write_f_path) as f:
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
    
    # STEP3 Remove puncs & reverse some reserved punctuations to original form
    # Also, remove/replace some unwanted tokens
    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(line)
            i = 0
            new_tokens = []
            while i < len(tokens):
                token = tokens[i]
                if i<len(tokens)-2:
                    if token == "num" and (tokens[i+1] == "am" or tokens[i+1] == "pm"):
                        new_tokens.append("<time>")
                        i+=2
                        continue
                    elif token == "num" and tokens[i+1] == "colon" and (tokens[i+2] == "am" or tokens[i+2] == "pm"):
                        new_tokens.append("<time>")
                        i+=3
                        continue
                    elif token == "colon" and (tokens[i+1] == "am" or tokens[i+1] == "pm"):
                        new_tokens.append("<time>")
                        i+=2
                        continue

                if token == "clip":
                    if i<len(tokens)-1:
                        if tokens[i+1] == "number":
                            i+=2
                            continue
                    else:
                        i+=1
                        continue
                    """
                    while tokens[i]!="</s>" or i < len(tokens):
                        i+=1
                    i+=1
                    continue
                    """

                if token == "num":
                    new_tokens.append("<num>")
                elif token == "colon":
                    new_tokens.append(":")
                elif token == "eos":
                    new_tokens.append("</s>")
                elif "_______" in token:
                    pass
                else:
                    new_tokens.append(token)

                i+=1

            f.write(" ".join(new_tokens) + "\n")
    return write_f_path


def digit_exists(str_token):
    return any(char.isdigit() for char in str_token)   


def non_digit_exists(str_token):
    return any((not char.isdigit()) for char in str_token)

def is_valid_num(str_token):
    try:
        num = int(float(str_token))
        return True
    except:
        return False


def notes_tokenize_keep_sca(data_path, res_dir=None):
    if not res_dir:
        write_f_path = common.extend_path_basename(data_path, "fully_preprocessed")
    else:
        f_name = common.extend_file_basename(data_path, "fully_preprocessed")
        write_f_path = os.path.join(os.path.dirname(data_path), f_name)

    with open(data_path) as read_f:
        with open(write_f_path, "w") as write_f:
            for line in read_f:
                words = line.split()
                new_words = []
                i = 0
                while i < len(words):
                    word = words[i]
                    if "\\n" in word and i<(len(words) - 4):
                        # ... if the next word is "<word>: ", this new line is
                        # the actual end of sentence, we use . or </s> as used
                        # later
                        if ":" in words[i+1] or ":" in words[i+2] or ":" in words[i+3] or ":" in words[i+4]:
                            word = word.replace("\\n", " .")
                        word = word.replace("\\n", " ")
                    new_words.append(word.lower())
                    i+=1
                write_f.write(" ".join(new_words) + "\n")
    # STEP1 Replace all the new lines with space excpet ...
    # If [ found, skip everything until ]
    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            words = line.split()
            new_words = []
            i = 0
            while i < len(words):
                word = words[i]
                if "[" in word:
                    while i < len(words) and (not ("]" in words[i])):
                        i+=1
                    i+=1
                    continue
                if "(" in word:
                    while i < len(words) and (not (")" in words[i])):
                        i+=1
                    i+=1
                    continue
                new_words.append(word.lower())
                i+=1
            f.write(" ".join(new_words) + "\n")
    # STEP2 Tokenize all words, and replace some punctuations with words
    # so that they could be preserved when we use nltk to remove all other punc 
    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            new_tokens = []
            tokens = nltk.tokenize.word_tokenize(line)
            for token in tokens:
                if ":" in token:
                    new_tokens.append("<colon>")
                elif digit_exists(token) and (not is_valid_num(token)):
                    new_tokens.append("<invalidnum>")
                elif token == ".":
                    new_tokens.append("eos")
                else:
                    new_tokens.append(token)
            f.write(" ".join(new_tokens) + "\n")
    
    # STEP3 Remove puncs & reverse some reserved punctuations to original form
    # Also, remove/replace some unwanted tokens
    with common.ReadReplaceOpen(write_f_path) as f:
        for line in f:
            tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(line)
            i = 0
            new_tokens = []
            while i < len(tokens):
                token = tokens[i]
                if i<len(tokens)-2:
                    if token == "invalidnum" and (tokens[i+1] == "am" or tokens[i+1] == "pm"):
                        new_tokens.append("<time>")
                        i+=2
                        continue
                    elif token == "invalidnum" and tokens[i+1] == "colon" and (tokens[i+2] == "am" or tokens[i+2] == "pm"):
                        new_tokens.append("<time>")
                        i+=3
                        continue
                    elif token == "colon" and (tokens[i+1] == "am" or tokens[i+1] == "pm"):
                        new_tokens.append("<time>")
                        i+=2
                        continue

                if token == "clip":
                    if i<len(tokens)-1:
                        if tokens[i+1] == "number":
                            i+=2
                            continue
                    else:
                        i+=1
                        continue
                    """
                    while tokens[i]!="</s>" or i < len(tokens):
                        i+=1
                    i+=1
                    continue
                    """

                if is_num:
                    pass
                    # new_tokens.append("<num>")
                elif token == "colon":
                    new_tokens.append(":")
                elif token == "eos":
                    new_tokens.append("</s>")
                elif "_______" in token:
                    pass
                else:
                    new_tokens.append(token)

                i+=1

            f.write(" ".join(new_tokens) + "\n")
    return write_f_path

