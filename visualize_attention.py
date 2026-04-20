"""
Generate an attention heatmap from best_model.pt.

Loads the trained decoder, runs a single English sentence through it,
captures attention weights from one (layer, head), and saves a heatmap
showing how each query token attends to earlier tokens (causal).
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from rope import Decoder, RoPEAttentionLayer

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
CMAP = "Blues"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- load checkpoint --------------------------------------------------------
ckpt = torch.load("best_model.pt", map_location=device, weights_only=False)
vocab = ckpt["vocab"]
inv_vocab = {i: w for w, i in vocab.items()}
SOS, SEP, EOS = vocab["SOS"], vocab["SEP"], vocab["EOS"]

# Match the hyperparameters used at training time (from rope.py main())
model = Decoder(n_heads=8, d_model=512, d_k=64, d_v=64,
                d_lin=2048, n_layers=6, vocab_size=32000).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# -- patch attention to capture softmax scores ------------------------------
captured = {}

original_forward = RoPEAttentionLayer.forward
def forward_with_capture(self, x, cos, sin, mask=None):
    out = original_forward(self, x, cos, sin, mask)
    # Recompute scores (cheap, single sentence) so we can store them
    from rope import apply_rope
    Q = apply_rope(self.W_q(x), cos, sin)
    K = apply_rope(self.W_k(x), cos, sin)
    s = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
    if mask is not None:
        s = s.masked_fill(mask == 0, float("-inf"))
    captured.setdefault("layers", []).append(torch.softmax(s, dim=-1).detach().cpu())
    return out
RoPEAttentionLayer.forward = forward_with_capture

# -- pick an English sentence ----------------------------------------------
english = "the government of canada is committed to protecting the environment"

def encode(text):
    ids = [SOS] + [vocab[w] for w in text.lower().split() if w in vocab] + [SEP]
    return torch.tensor([ids], device=device, dtype=torch.long)

# Greedy generate French continuation so the heatmap also shows the FR side
@torch.no_grad()
def greedy(prefix, max_new=25):
    seq = prefix
    for _ in range(max_new):
        logits = model(seq)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        if nxt.item() == EOS:
            break
    return seq

prefix = encode(english)
captured.clear(); captured["layers"] = []
full = greedy(prefix)
# Drop the layers captured during greedy (each step appended); keep only the final pass
captured["layers"] = []
with torch.no_grad():
    _ = model(full)

token_ids = full[0].tolist()
labels = [inv_vocab.get(i, "?") for i in token_ids]
# Pretty special tokens
pretty = {"SOS": "<s>", "SEP": "<sep>", "EOS": "</s>"}
labels = [pretty.get(l, l) for l in labels]
n = len(labels)

# captured["layers"] has one entry per attention head (n_layers * n_heads heads total,
# but each layer has n_heads heads in a ModuleList — patched forward fires per head)
# Heads are appended in order: layer0_head0, layer0_head1, ..., layer5_head7
n_layers, n_heads = 6, 8
all_heads = captured["layers"]
assert len(all_heads) == n_layers * n_heads, f"got {len(all_heads)} heads"

# -- plot a 2x3 grid: one panel per layer, averaged over heads ---------------
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for layer_idx in range(n_layers):
    ax = axes[layer_idx // 3, layer_idx % 3]
    layer_heads = all_heads[layer_idx * n_heads:(layer_idx + 1) * n_heads]
    avg = torch.stack([h[0] for h in layer_heads]).mean(0).numpy()
    im = ax.imshow(avg, cmap=CMAP, aspect="auto")
    ax.set_title(f"Layer {layer_idx} (avg over {n_heads} heads)")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key (attended-to)"); ax.set_ylabel("Query")
    # Mark the SEP boundary
    sep_idx = labels.index("<sep>")
    ax.axhline(sep_idx + 0.5, color="red", linewidth=0.8, alpha=0.7)
    ax.axvline(sep_idx + 0.5, color="red", linewidth=0.8, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f'Attention weights — "{english}"', fontsize=13)
plt.tight_layout()
plt.savefig("attention_layers.png", dpi=150, bbox_inches="tight")
print("saved attention_layers.png")

# -- single best panel: pick a clean later layer -----------------------------
layer_idx = 4
layer_heads = all_heads[layer_idx * n_heads:(layer_idx + 1) * n_heads]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for h in range(n_heads):
    ax = axes[h // 4, h % 4]
    m = layer_heads[h][0].numpy()
    im = ax.imshow(m, cmap=CMAP, aspect="auto")
    ax.set_title(f"L{layer_idx} H{h}")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    sep_idx = labels.index("<sep>")
    ax.axhline(sep_idx + 0.5, color="red", linewidth=0.6, alpha=0.6)
    ax.axvline(sep_idx + 0.5, color="red", linewidth=0.6, alpha=0.6)
plt.suptitle(f'Layer {layer_idx} heads — "{english}"', fontsize=13)
plt.tight_layout()
plt.savefig("attention_heads.png", dpi=150, bbox_inches="tight")
print("saved attention_heads.png")
print("french output:", " ".join(labels[labels.index("<sep>") + 1:]))
