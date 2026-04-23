import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_bleu, config, src_vocab, tgt_vocab, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_bleu": val_bleu,
        "config": config,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }
    torch.save(checkpoint, filepath)


def detokenize(text):
    import re
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:\'\"\)\]])', r'\1', text)
    # Remove space after opening brackets & qoutes
    text = re.sub(r'([\(\[\"\'])\s+', r'\1', text)
    return text


def ids_to_text(ids, inv_vocab, skip_special=True, special_ids={0, 1, 2, 3}):
    tokens = []
    for idx in ids:
        if skip_special and idx in special_ids:
            continue
        if idx in inv_vocab:
            tokens.append(inv_vocab[idx])
        else:
            tokens.append("<UNK>")
    return detokenize(" ".join(tokens))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"


class LRScheduler:
    def __init__(self, optimizer, initial_lr, decay_start=5, decay_factor=0.5, decay_every=0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_start = decay_start
        self.decay_factor = decay_factor
        self.decay_every = decay_every
        self.current_lr = initial_lr

    def step(self, epoch):
        if epoch < self.decay_start:
            new_lr = self.initial_lr
        else:
            # Number of decay steps since decay started
            decay_steps = (epoch - self.decay_start) / self.decay_every
            new_lr = self.initial_lr * (self.decay_factor ** decay_steps)

        self.current_lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        return new_lr


class GradientClipper:
    # ensure nrom doesn't exceed threshold
    def __init__(self, max_norm=5.0):
        self.max_norm = max_norm

    def __call__(self, model, batch_size):
        # Compute total norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Normalize by batch size
        normalized_norm = total_norm / batch_size

        # Clip if necessary
        if normalized_norm > self.max_norm:
            clip_coef = (self.max_norm * batch_size) / total_norm
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        return normalized_norm
