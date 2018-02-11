from preprocess_nlp.file_utils.common import MultiWriteOpen
from preprocess_nlp.vocab_utils.common import EOS, EOS_ID
import preprocess_nlp.doc_token as dt
import pdb


YIELD_EOL = "yield_eol"
IGNORE_EOL = "ignore_eol"
KEEP_EOL_NL = "keep_eol_nl"


def doc_line_gen_f(doc_path, token_type):
    convert_f = get_convert_f(token_type)
    with open(doc_path) as f:
        for line in f:
            yield [convert_f(token) for token in line.strip().split(" ")]


def get_convert_f(token_type):
    convert_f = lambda x: x
    if token_type == dt.VALUE_INT_TYPE or token_type == dt.ID_TYPE:
        convert_f = lambda x: int(float(x))
    elif token_type == dt.VALUE_FLOAT_TYPE:
        convert_f = lambda x: float(x)
    return convert_f


def _doc_gen_f(doc_path, token_type, eol_gen_f=None):
    def doc_gen():
        dt.assert_type_valid(token_type)
        convert_f = get_convert_f(token_type)
        with open(doc_path) as f:
            for line in f:
                tokens = line.strip().split(" ")
                for token in tokens:
                    yield convert_f(token)
                # Handle the end of line if the document is language based
                if eol_gen_f:
                    for eol in eol_gen_f(len(tokens)):
                        yield eol
    return doc_gen


def doc_gen_f_yield_eol(doc_path, token_type):
    def eol_gen_f(line_len):
        if line_len > 0:
            if token_type == dt.WORD_TYPE:
                yield EOS
            elif token_type == dt.ID_TYPE:
                yield EOS_ID
    gen_f = _doc_gen_f(doc_path, token_type, eol_gen_f)
    return gen_f


def doc_gen_f_keep_eol_nl(doc_path, token_type):
    def eol_gen_f(line_len):
        if line_len > 0:
            yield "\n"
    gen_f = _doc_gen_f(doc_path, token_type, eol_gen_f)
    return gen_f


def doc_gen_f_ignore_eol(doc_path, token_type):
    gen_f = _doc_gen_f(doc_path, token_type)
    return gen_f


def doc_save(doc, doc_transformer, *txt_save_paths):
    assert len(txt_save_paths) == len(doc_transformer)
    with MultiWriteOpen(*txt_save_paths) as opened_files:
        raw_stringfy_f = lambda x: x+"\n" 
        list_stringfy_f = lambda x: " ".join(x)+"\n"
        stringfy_fs = [list_stringfy_f if s_len>1 else raw_stringfy_f
                       for s_len in doc_transformer.seq_lens]
        for seqs in doc_transformer.get_iters(doc):
            for i, seq_f in enumerate(zip(seqs, opened_files)):
                seq, f = seq_f
                f.write(stringfy_fs[i](seq))    


def word2vec_iter(src_path, tgt_path):
    for src, tgt in zip(open(src_path), open(tgt_path)):
        yield src.split()[0], tgt.split()[0]



