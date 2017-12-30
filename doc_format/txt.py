from preprocess_nlp.file_utils.common import MultiWriteOpen

def doc_gen_f(self, doc_path):
    def doc_gen():
        with open(doc_path) as f:
            for line in f:
                tokens = line.split()
                for token in tokens:
                    yield token
                if self._token_type == str:
                    yield EOS
                else:
                    yield EOS_ID
    return doc_gen

def doc_save(self, doc, doc_transformer, *txt_save_paths):
    assert len(save_paths) == len(transformed_doc)
    opened_files = [open(save_path, "w") for save_path in save_paths]
    with MultiWriteOpen(save_paths) as opened_files:
        raw_stringfy_f = lambda x: x+"\n" 
        list_stringfy_f = lambda x: " ".join(x)+"\n"
        stringfy_fs = [list_stringfy_f if s_len>1 else raw_stringfy_f
                       for s_len in doc_transformer.seq_lens]
        for seqs in doc_transformer.get_iters(doc):
            for i, seq, f in enumerate(zip(seqs, opened_files)):
                f.write(stringfy_fs[i](seq))    


