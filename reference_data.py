import re, os, random, csv
import numpy as np
import pandas as pd

csv.field_size_limit(10_000_000)  # allow very long fields

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, word_dict, max_len=50):
        SOS = word_dict["SOS"]
        EOS = word_dict["EOS"]
        SEP = word_dict["SEP"]
        N = len(src_texts)
        # Pre-allocate contiguous arrays — avoids 11M individual Python objects
        seqs_np = np.zeros((N, max_len), dtype=np.int32)
        sep_np  = np.zeros(N, dtype=np.int32)
        for i, (s, t) in enumerate(zip(src_texts, tgt_texts)):
            eng_ids = [word_dict[w] for w in s.lower().split() if w in word_dict]
            fr_ids  = [word_dict[w] for w in t.lower().split() if w in word_dict]
            seq = [SOS] + eng_ids + [SEP] + fr_ids + [EOS]
            seq = seq[:max_len]
            sep_pos = min(1 + len(eng_ids), max_len - 1)
            seqs_np[i, :len(seq)] = seq
            sep_np[i] = sep_pos
        # Keep as int32 to halve RAM; convert to long in collate
        self.seqs = torch.from_numpy(seqs_np)
        self.sep_positions = torch.from_numpy(sep_np)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx): # returns IDs of idx-th single sample and its sep position
        return self.seqs[idx], self.sep_positions[idx]

def translation_collate(batch):
    seqs, sep_positions = zip(*batch)
    seq_pad = pad_sequence(seqs, batch_first=True, padding_value=0).long()  # (B, L)
    sep_positions = torch.stack(sep_positions).long()                        # (B,)
    return seq_pad, sep_positions
                                                                                                                            
def build_combined_vocab(en_texts, fr_texts, vocab_size):
    word_count = {}
    for sent in list(en_texts) + list(fr_texts):
        for word in sent.lower().split():
            word_count[word] = word_count.get(word, 0) + 1
    # 4 slots for PAD, SOS, EOS, SEP
    top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:vocab_size - 4]
    word_dict = {"PAD": 0, "SOS": 1, "EOS": 2, "SEP": 3}
    for i, (word, _) in enumerate(top_words):
        word_dict[word] = i + 4
    return word_dict

def get_translation_dataloader(
    csv_path,
    vocab_size=10000,
    max_len=50,
    batch_size=64,
    n_samples=25_000_000,  # >22.5M dataset size -> keeps all pairs
    chunk_size=500_000,    # larger chunks = fewer pandas overhead calls
    num_workers=4,         # increase for AWS (match to vCPU count)
    cache_path=None,       # e.g. "data_cache.pt" — skip CSV reprocessing on reruns
):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading preprocessed data from {cache_path} ...")
        cache = torch.load(cache_path, weights_only=False)
        train_set = cache['train_set']
        test_set  = cache['test_set']
        word_dict = cache['word_dict']
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            collate_fn=translation_collate, drop_last=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            collate_fn=translation_collate, drop_last=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
        )
        return train_loader, test_loader, word_dict

    # ---- 1. Stream the CSV in chunks, length-filter, and reservoir-sample ----
    kept_en, kept_fr = [], []
    rng = random.Random(10701)
    total_seen = 0
    total_rows = 0
    filtered_length = 0

    skipped = 0
    with open(csv_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                skipped += 1
                continue
            total_rows += 1
            en = row.get('en')
            fr = row.get('fr')
            if not en or not fr:
                continue
            en_len = len(en.split())
            fr_len = len(fr.split())
            if en_len == 0 or fr_len == 0:
                continue
            if en_len + fr_len + 3 > max_len:
                filtered_length += 1
                continue

            total_seen += 1
            if len(kept_en) < n_samples:
                kept_en.append(en)
                kept_fr.append(fr)
            else:
                j = rng.randint(0, total_seen - 1)
                if j < n_samples:
                    kept_en[j] = en
                    kept_fr[j] = fr
    print(f"Skipped {skipped} malformed rows.")
    print(f"Total rows: {total_rows} | Filtered out (too long): {filtered_length} | Passed filter: {total_seen} | Kept: {len(kept_en)}")

    # ---- 2. Build vocab on the SUBSET ----
    word_dict = build_combined_vocab(kept_en, kept_fr, vocab_size)

    # ---- 3. Train/test split ----
    idx = list(range(len(kept_en)))
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    train_en = [kept_en[i] for i in train_idx]
    train_fr = [kept_fr[i] for i in train_idx]
    test_en  = [kept_en[i] for i in test_idx]
    test_fr  = [kept_fr[i] for i in test_idx]

    # Free raw text lists before building tensors (saves ~2-3 GB peak RAM)
    del kept_en, kept_fr, idx, train_idx, test_idx

    train_set = TranslationDataset(train_en, train_fr, word_dict, max_len)
    test_set  = TranslationDataset(test_en,  test_fr,  word_dict, max_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=translation_collate, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=translation_collate, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    if cache_path:
        print(f"Saving preprocessed data to {cache_path} ...")
        torch.save({'train_set': train_set, 'test_set': test_set,
                    'word_dict': word_dict}, cache_path)

    return train_loader, test_loader, word_dict