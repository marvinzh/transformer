import spacy

tokens=["<UNK>","<PAD>","<EOS>","<SOS>"]

class Lang:
    def __init__(self, corpus, reserved_tokens=tokens, n_vocab=-1):
        vocab_list = reserved_tokens + self._build_vocab(corpus)
        vocab_list = vocab_list[:n_vocab] if n_vocab > 0 else vocab_list

        self.vocab_list = vocab_list
        self.vocab = dict()
        for i, token in enumerate(vocab_list):
            self.vocab[token] = i

    def __len__(self):
        return len(self.vocab)

    def idx2str(self, idx_seq):
        return list(map(
            lambda i: self.vocab_list[i], idx_seq
        ))

    def str2idx(self, str_seq):
        return list(map(
            lambda k: self.vocab[k], str_seq
        ))

    def _build_vocab(self, corpus):
        spacy.prefer_gpu()
        nlp = spacy.load('en_core_web_sm')

        vocab = dict()
        doc = nlp(corpus)
        for token in doc:
            if token not in vocab:
                vocab[token.text] = 1
            else:
                vocab[token.text] += 1

        vocab_list = [(key, vocab[key]) for key in vocab.keys()]
        vocab_list = sorted(vocab_list, key=lambda x: x[1])

        return [v[0] for v in vocab_list]
