import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<blank>", "<s>", "</s>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def train_tokenizer(sentences, vocab_size=8000):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    return tokenizer


def build_tokenizers(train_split, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, "tokenizer_de.json")
    tgt_path = os.path.join(cache_dir, "tokenizer_en.json")

    if os.path.exists(src_path) and os.path.exists(tgt_path):
        return Tokenizer.from_file(src_path), Tokenizer.from_file(tgt_path)

    src_tokenizer = train_tokenizer(train_split["de"])
    tgt_tokenizer = train_tokenizer(train_split["en"])
    src_tokenizer.save(src_path)
    tgt_tokenizer.save(tgt_path)
    return src_tokenizer, tgt_tokenizer


def encode(tokenizer, text, max_len):
    ids = tokenizer.encode(text).ids
    ids = ids[: max_len - 2]
    return [BOS_IDX] + ids + [EOS_IDX]


def decode(tokenizer, ids):
    ids = [i for i in ids if i not in (PAD_IDX, BOS_IDX, EOS_IDX)]
    return tokenizer.decode(ids)
