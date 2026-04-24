import os
import time
import argparse

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import sacrebleu
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from preprocess import load_and_preprocess, build_inv_vocab, SOS_IDX, EOS_IDX, PAD_IDX
from model import build_model
from utils import set_seed, save_checkpoint, ids_to_text, count_parameters, format_time, LRScheduler, GradientClipper


def train_epoch(model, train_loader, optimizer, criterion, scaler, grad_clipper, device, epoch, use_amp):
    model.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

    for src, tgt, src_lengths in pbar:
        src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(src, src_lengths, tgt)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = grad_clipper(model, src.size(0))
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(src, src_lengths, tgt)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            grad_norm = grad_clipper(model, src.size(0))
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad_norm": f"{grad_norm:.2f}"})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, tgt_inv_vocab, max_decode_len, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_refs = []

    for src, tgt, src_lengths in tqdm(val_loader, desc="Validating", leave=False):
        src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
        outputs = model(src, src_lengths, tgt)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
        total_loss += loss.item()
        num_batches += 1

        preds = model.greedy_decode(src, src_lengths, max_decode_len, SOS_IDX, EOS_IDX)
        for pred, ref in zip(preds, tgt):
            all_preds.append(ids_to_text(pred.tolist(), tgt_inv_vocab))
            all_refs.append(ids_to_text(ref[1:].tolist(), tgt_inv_vocab))

    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score
    return total_loss / num_batches, bleu, all_preds[:5], all_refs[:5]


def train(config, force_rebuild=False):
    print("=" * 60)
    print("Seq2Seq English-to-French Translation")
    print("=" * 60)

    set_seed(config.seed)
    device = config.device
    print(f"Device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, _, src_vocab, tgt_vocab = load_and_preprocess(config, force_rebuild)
    tgt_inv_vocab = build_inv_vocab(tgt_vocab)

    print("\nBuilding model...")
    model = build_model(config, len(src_vocab), len(tgt_vocab)).to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    lr_scheduler = LRScheduler(optimizer, config.learning_rate, config.lr_decay_start_epoch, config.lr_decay_factor)
    grad_clipper = GradientClipper(config.grad_clip)
    use_amp = config.use_mixed_precision and device == "cuda"

    if use_amp:
        scaler = GradScaler('cuda')
    else:
        scaler = None

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\nTraining: {config.epochs} epochs, batch={config.batch_size}, lr={config.learning_rate}")
    print(f"Reverse source: {config.reverse_source}, Dataset: {config.dataset_fraction*100:.1f}%\n")

    best_val_loss = float("inf")
    best_bleu = 0.0
    history = {"train_loss": [], "val_loss": [], "val_bleu": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        lr = lr_scheduler.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, grad_clipper, device, epoch, use_amp)
        val_loss, val_bleu, preds, refs = validate(model, val_loader, criterion, tgt_inv_vocab, config.max_decode_len, device)

        print(f"Epoch {epoch}/{config.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | BLEU: {val_bleu:.2f} | LR: {lr:.6f} | {format_time(time.time()-t0)}")
        print(f"  Pred: {preds[0]}\n  Ref:  {refs[0]}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu"].append(val_bleu)
        history["lr"].append(lr)

        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_bleu, config, src_vocab, tgt_vocab,
                        os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_bleu, config, src_vocab, tgt_vocab,
                            os.path.join(config.checkpoint_dir, "best_model.pt"))
            print(f"  -> New best model!")
        print()

    print("=" * 60)
    print(f"Done! Best Val Loss: {best_val_loss:.4f}, BLEU: {best_bleu:.2f}")
    print("=" * 60)

    plot_curves(history, config)
    return model, history


def plot_curves(history, config):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "o-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["val_bleu"], "o-", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BLEU")
    axes[1].grid(True)

    axes[2].plot(epochs, history["lr"], "o-", color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    path = f"results/training_curves_frac{config.dataset_fraction}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved curves to {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dataset_fraction", type=float)
    parser.add_argument("--no_reverse", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    config = Config()

    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.dataset_fraction:
        config.dataset_fraction = args.dataset_fraction
    if args.no_reverse:
        config.reverse_source = False

    train(config, args.force_rebuild)


if __name__ == "__main__":
    main()
