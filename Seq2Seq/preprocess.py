import os
import sys
import re
import gc
from collections import Counter

import kagglehub

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference_data import stream_csv_pairs
from config import Config

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# uses reference_data.py to help

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for word, _ in counter.most_common(vocab_size - 4):
        vocab[word] = len(vocab)
    return vocab


def build_inv_vocab(vocab):
    return {idx: word for word, idx in vocab.items()}


class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=50, reverse_source=True):
        N = len(src_texts)
        src_seqs_np = np.zeros((N, max_len), dtype=np.int32)
        tgt_seqs_np = np.zeros((N, max_len), dtype=np.int32)
        src_lens_np = np.zeros(N, dtype=np.int32)
        valid_mask = np.ones(N, dtype=bool)

        for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
            src_tokens = tokenize(src)
            tgt_tokens = tokenize(tgt)

            if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                valid_mask[i] = False
                continue
            if len(src_tokens) + 1 > max_len or len(tgt_tokens) + 2 > max_len:
                valid_mask[i] = False
                continue

            if reverse_source:
                src_tokens = src_tokens[::-1]

            src_ids = [src_vocab.get(t, UNK_IDX) for t in src_tokens] + [EOS_IDX]
            tgt_ids = [SOS_IDX] + [tgt_vocab.get(t, UNK_IDX) for t in tgt_tokens] + [EOS_IDX]

            src_seqs_np[i, :len(src_ids)] = src_ids
            tgt_seqs_np[i, :len(tgt_ids)] = tgt_ids
            src_lens_np[i] = len(src_ids)

        dropped = N - valid_mask.sum()
        if dropped > 0:
            print(f"  Dropped {dropped} pairs")

        self.src_seqs = torch.from_numpy(src_seqs_np[valid_mask]).long()
        self.tgt_seqs = torch.from_numpy(tgt_seqs_np[valid_mask]).long()
        self.src_lens = torch.from_numpy(src_lens_np[valid_mask]).long()
        del src_seqs_np, tgt_seqs_np, src_lens_np, valid_mask
        gc.collect()

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return self.src_seqs[idx], self.tgt_seqs[idx], self.src_lens[idx]


def collate_fn(batch):
    src_seqs, tgt_seqs, src_lens = zip(*batch)
    src_batch = torch.stack(src_seqs)
    tgt_batch = torch.stack(tgt_seqs)
    src_lengths = torch.stack(src_lens)
    max_src = src_lengths.max().item()
    max_tgt = (tgt_batch != 0).sum(dim=1).max().item()
    return src_batch[:, :max_src], tgt_batch[:, :max_tgt], src_lengths

# load kaggle
def load_and_preprocess(config, force_rebuild=False):
    if config.cache_path and os.path.exists(config.cache_path) and not force_rebuild:
        print(f"Loading cached data from {config.cache_path}...")
        cache = torch.load(config.cache_path)
        return cache["train_loader"], cache["val_loader"], cache["test_loader"], cache["src_vocab"], cache["tgt_vocab"]

    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
    csv_path = os.path.join(dataset_path, "en-fr.csv")

    n_samples = config.get_effective_dataset_size()
    print(f"Loading {n_samples:,} samples...")

    kept_en, kept_fr = stream_csv_pairs(csv_path, config.max_len, n_samples)

    # Train/val/test split (81/9/10)
    split1 = int(0.9 * len(kept_en))
    split2 = int(0.9 * split1)

    train_en, train_fr = kept_en[:split2], kept_fr[:split2]
    val_en, val_fr = kept_en[split2:split1], kept_fr[split2:split1]
    test_en, test_fr = kept_en[split1:], kept_fr[split1:]
    del kept_en, kept_fr
    gc.collect()

    print(f"Train: {len(train_en):,} | Val: {len(val_en):,} | Test: {len(test_en):,}")

    src_vocab = build_vocab(train_en, config.src_vocab_size)
    tgt_vocab = build_vocab(train_fr, config.tgt_vocab_size)
    print(f"Vocab sizes: {len(src_vocab):,} / {len(tgt_vocab):,}")

    train_dataset = TranslationDataset(train_en, train_fr, src_vocab, tgt_vocab, config.max_len, config.reverse_source)
    del train_en, train_fr; gc.collect()

    val_dataset = TranslationDataset(val_en, val_fr, src_vocab, tgt_vocab, config.max_len, config.reverse_source)
    del val_en, val_fr; gc.collect()

    test_dataset = TranslationDataset(test_en, test_fr, src_vocab, tgt_vocab, config.max_len, config.reverse_source)
    del test_en, test_fr; gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)

    if config.cache_path:
        torch.save({"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader, "src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, config.cache_path)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
