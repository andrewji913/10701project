import os
import argparse
import gc

import kagglehub
import torch
import torch.nn as nn
import sacrebleu
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from preprocess import TranslationDataset, collate_fn, build_inv_vocab, SOS_IDX, EOS_IDX, PAD_IDX
from model import Seq2Seq
from utils import set_seed, ids_to_text

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference_data import stream_csv_pairs


def load_test_data_with_vocab(config, src_vocab, tgt_vocab):
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
    csv_path = os.path.join(dataset_path, "en-fr.csv")

    n_samples = config.get_effective_dataset_size()
    print(f"Loading {n_samples:,} samples...")

    kept_en, kept_fr = stream_csv_pairs(csv_path, config.max_len, n_samples)

    split1 = int(0.9 * len(kept_en))
    test_en, test_fr = kept_en[split1:], kept_fr[split1:]
    del kept_en, kept_fr
    gc.collect()

    print(f"Test set size: {len(test_en):,}")
    print(f"Using checkpoint vocab: src={len(src_vocab):,}, tgt={len(tgt_vocab):,}")

    # Create test dataset using CHECKPOINT vocab
    test_dataset = TranslationDataset(
        test_en, test_fr, src_vocab, tgt_vocab,
        config.max_len, config.reverse_source
    )
    del test_en, test_fr
    gc.collect()

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=1, pin_memory=True
    )

    return test_loader


@torch.no_grad()
def evaluate(model, test_loader, criterion, tgt_inv_vocab, max_decode_len, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_refs = []

    for src, tgt, src_lengths in tqdm(test_loader, desc="Testing", leave=False):
        src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)

        outputs = model(src, src_lengths, tgt)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
        total_loss += loss.item()
        num_batches += 1

        preds = model.greedy_decode(src, src_lengths, max_decode_len, SOS_IDX, EOS_IDX)
        for pred, ref in zip(preds, tgt):
            all_preds.append(ids_to_text(pred.tolist(), tgt_inv_vocab))
            all_refs.append(ids_to_text(ref[1:].tolist(), tgt_inv_vocab))

    avg_loss = total_loss / num_batches
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score

    return avg_loss, bleu, all_preds, all_refs


def evaluate_all_checkpoints(checkpoint_dir="checkpoints_0.25", output_path="results/test_curves.png"):
    print("=" * 60)
    print("Evaluating All Checkpoints on Test Set (10% held-out)")
    print("=" * 60)

    # Load first checkpoint to get config and vocab
    first_checkpoint = torch.load(
        os.path.join(checkpoint_dir, "checkpoint_epoch_1.pt"),
        map_location="cpu", weights_only=False
    )
    config = first_checkpoint["config"]
    src_vocab = first_checkpoint["src_vocab"]
    tgt_vocab = first_checkpoint["tgt_vocab"]

    set_seed(config.seed)
    device = config.device
    print(f"Device: {device}")

    # Load test data using CHECKPOINT vocab (not a new vocab)
    print("\nLoading test data with checkpoint vocab...")
    test_loader = load_test_data_with_vocab(config, src_vocab, tgt_vocab)
    tgt_inv_vocab = build_inv_vocab(tgt_vocab)

    print("\nBuilding model...")
    model = Seq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        param_init_range=config.param_init_range,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epochs = []
    test_losses = []
    test_bleus = []

    for epoch in range(1, 11):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint for epoch {epoch} not found, skipping...")
            continue

        print(f"\nEvaluating epoch {epoch}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_bleu, _, _ = evaluate(
            model, test_loader, criterion, tgt_inv_vocab, config.max_decode_len, device
        )

        epochs.append(epoch)
        test_losses.append(test_loss)
        test_bleus.append(test_bleu)

        print(f"  Epoch {epoch}: Test Loss = {test_loss:.4f}, Test BLEU = {test_bleu:.2f}")

    # Plot results
    print("\nGenerating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left plot: Test Loss
    ax1.plot(epochs, test_losses, "o-", color="#1f77b4", label="Test Loss", markersize=6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Right plot: Test BLEU Score
    ax2.plot(epochs, test_bleus, "o-", color="#2ca02c", label="Test BLEU", markersize=6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BLEU Score")
    ax2.set_title("Test BLEU Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close()

    # Save test metrics for plot_combined.py
    metrics_path = os.path.join(os.path.dirname(output_path), "test_metrics.pt")
    torch.save({
        "epochs": epochs,
        "test_losses": test_losses,
        "test_bleus": test_bleus
    }, metrics_path)
    print(f"Saved test metrics to {metrics_path}")

    # Print summary
    best_idx = test_bleus.index(max(test_bleus))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best Test BLEU: {test_bleus[best_idx]:.2f} at epoch {epochs[best_idx]}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Final Test BLEU: {test_bleus[-1]:.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Seq2Seq model on held-out test set")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_0.25",
                        help="Directory containing epoch checkpoints")
    parser.add_argument("--output", type=str, default="results/test_curves.png",
                        help="Output path for the plot")
    args = parser.parse_args()

    evaluate_all_checkpoints(args.checkpoint_dir, args.output)


if __name__ == "__main__":
    main()
