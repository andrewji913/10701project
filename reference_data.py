import re, os, random
import numpy as np
import pandas as pd

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
        self.seqs = []
        self.sep_positions = []
        for s, t in zip(src_texts, tgt_texts):
            eng_ids = [word_dict[w] for w in s.lower().split() if w in word_dict]
            fr_ids  = [word_dict[w] for w in t.lower().split() if w in word_dict]
            seq = [SOS] + eng_ids + [SEP] + fr_ids + [EOS]
            seq = seq[:max_len]
            sep_pos = 1 + len(eng_ids)  # index of SEP in seq
            self.seqs.append(torch.tensor(seq, dtype=torch.long))
            self.sep_positions.append(sep_pos)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx): # returns IDs of idx-th single sample and its sep position
        return self.seqs[idx], self.sep_positions[idx]

def translation_collate(batch):
    seqs, sep_positions = zip(*batch)
    seq_pad = pad_sequence(seqs, batch_first=True, padding_value=0)  # (B, L)
    sep_positions = torch.tensor(sep_positions, dtype=torch.long)    # (B,)
    return seq_pad, sep_positions
                                                                                                                            
def build_combined_vocab(en_texts, fr_texts, vocab_size):
    word_count = {}
    for sent in list(en_texts) + list(fr_texts):
        for word in sent.lower().split():
            word_count[word] = word_count.get(word, 0) + 1
    # reserve 4 slots for PAD, SOS, EOS, SEP
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
    n_samples=500_000,     # how many pairs to actually use
    chunk_size=200_000,    # rows per CSV chunk while streaming
):
    # ---- 1. Stream the CSV in chunks, length-filter, and reservoir-sample ----
    kept_en, kept_fr = [], []
    rng = random.Random(10701)

    reader = pd.read_csv(csv_path, chunksize=chunk_size, usecols=['en', 'fr'])
    # NOTE: column names in dhruvildave/en-fr-translation-dataset are 'en' and 'fr' (lowercase).
    # If your CSV has 'English' / 'French', change usecols and the references below.

    for chunk in reader:
        chunk = chunk.dropna()
        # cheap length filter on whitespace tokens — keep only short pairs
        en_lens = chunk['en'].str.split().str.len()
        fr_lens = chunk['fr'].str.split().str.len()
        # leave room for SOS, SEP, EOS  =>  en + fr + 3 <= max_len
        mask = (en_lens > 0) & (fr_lens > 0) & (en_lens + fr_lens + 3 <= max_len)
        chunk = chunk[mask]

        for en, fr in zip(chunk['en'].tolist(), chunk['fr'].tolist()):
            if len(kept_en) < n_samples:
                kept_en.append(en)
                kept_fr.append(fr)
            else:
                # reservoir sampling so we get a uniform sample across the full file
                j = rng.randint(0, len(kept_en))  # crude but fine here
                if j < n_samples:
                    kept_en[j] = en
                    kept_fr[j] = fr

        if len(kept_en) >= n_samples:
            # early exit: once full, we have a representative-enough subset
            # (remove this break if you want true reservoir sampling over the whole file)
            break

    print(f"Kept {len(kept_en)} pairs after filtering.")

    # ---- 2. Build vocab on the SUBSET, not the full 22M rows ----
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

    train_set = TranslationDataset(train_en, train_fr, word_dict, max_len)
    test_set  = TranslationDataset(test_en,  test_fr,  word_dict, max_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=translation_collate, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=translation_collate, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader, word_dict