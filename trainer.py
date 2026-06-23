import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from translater import make_model, subsequent_mask
from tokenizer import BOS_IDX, EOS_IDX, PAD_IDX, build_tokenizers, decode, encode

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MAX_LEN = 64


class Batch:
    "Хранит src/tgt вместе с масками для одного шага обучения."

    def __init__(self, src, tgt, pad=PAD_IDX, device="cpu"):
        self.src = src.to(device)
        self.src_mask = (src != pad).unsqueeze(-2).to(device)
        self.tgt = tgt[:, :-1].to(device)
        self.tgt_y = tgt[:, 1:].to(device)
        self.tgt_mask = self.make_std_mask(self.tgt, pad).to(device)
        self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def make_collate_fn(src_tokenizer, tgt_tokenizer, device="cpu"):
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for example in batch:
            src_ids = encode(src_tokenizer, example["de"], MAX_LEN)
            tgt_ids = encode(tgt_tokenizer, example["en"], MAX_LEN)
            src_batch.append(torch.tensor(src_ids, dtype=torch.long))
            tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))

        src = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
        return Batch(src, tgt, device=device)

    return collate_fn


class LabelSmoothing(nn.Module):
    "Сглаживание меток через KL-дивергенцию."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    "Считает loss с учётом генератора (последний линейный слой + log_softmax)."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(
            x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
        ) / norm
        return sloss.data * norm, sloss


def rate(step, model_size, factor, warmup):
    "Скорость обучения по формуле из 'Attention is All You Need' (warm-up + decay)."
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train", desc=""):
    "Прогоняет одну эпоху (train или eval) и возвращает средний loss на токен."
    is_train = mode == "train"
    model.train(is_train)

    total_loss = 0
    total_tokens = 0

    progress = tqdm(data_iter, desc=desc, leave=False)
    for batch in progress:
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if is_train:
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()
        total_tokens += batch.ntokens
        progress.set_postfix(loss=f"{loss.item() / batch.ntokens:.4f}")

    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    "Жадная генерация: на каждом шаге выбирает наиболее вероятный токен."
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if next_word == EOS_IDX:
            break
    return ys


def train_translation(
    n_layers=4,
    d_model=256,
    d_ff=1024,
    h=8,
    n_epochs=8,
    batch_size=64,
    vocab_size=8000,
    train_subset=None,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    dataset = load_dataset("bentrevett/multi30k")
    train_split = dataset["train"]
    if train_subset is not None:
        train_split = train_split.select(range(train_subset))
    valid_split = dataset["validation"]

    src_tokenizer, tgt_tokenizer = build_tokenizers(dataset["train"], CACHE_DIR)
    src_vocab = src_tokenizer.get_vocab_size()
    tgt_vocab = tgt_tokenizer.get_vocab_size()

    collate_fn = make_collate_fn(src_tokenizer, tgt_tokenizer, device=device)
    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    criterion = LabelSmoothing(size=tgt_vocab, padding_idx=PAD_IDX, smoothing=0.1)
    criterion.to(device)
    model = make_model(src_vocab, tgt_vocab, N=n_layers, d_model=d_model, d_ff=d_ff, h=h)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    for epoch in range(n_epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            scheduler,
            mode="train",
            desc=f"epoch {epoch:02d} [train]",
        )

        model.eval()
        with torch.no_grad():
            eval_loss = run_epoch(
                valid_loader,
                model,
                SimpleLossCompute(model.generator, criterion),
                optimizer,
                scheduler,
                mode="eval",
                desc=f"epoch {epoch:02d} [eval]",
            )

        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | eval_loss {eval_loss:.4f}")

    translate_examples(model, src_tokenizer, tgt_tokenizer, valid_split, device=device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nмодель сохранена в {checkpoint_path}")

    return model


def translate_examples(model, src_tokenizer, tgt_tokenizer, examples, n=5, device="cpu"):
    "Прогоняет несколько примеров через обученную модель и печатает перевод."
    model.eval()
    print("\n--- примеры перевода ---")
    for example in tqdm(examples.select(range(n)), desc="translate", leave=False):
        src_ids = torch.tensor([encode(src_tokenizer, example["de"], MAX_LEN)], device=device)
        src_mask = torch.ones(1, 1, src_ids.size(-1), device=device)
        with torch.no_grad():
            out_ids = greedy_decode(
                model, src_ids, src_mask, max_len=MAX_LEN, start_symbol=BOS_IDX
            )
        translation = decode(tgt_tokenizer, out_ids[0].tolist())
        print(f"DE : {example['de']}")
        print(f"EN*: {translation}")
        print(f"EN : {example['en']}\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    train_translation()
