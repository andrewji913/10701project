# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import sacrebleu
import pickle
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from reference_data import get_translation_dataloader
import os
import kagglehub

dataset_path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
file_path = os.path.join(dataset_path, "en-fr.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_k, max_len=500):
        super(RotaryPositionalEncoding, self).__init__()
        # theta_i = rotation speed for each pair (consectuvie x's turned into a pair), later pairs rotate slowly
        theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
        # positions: [0, 1, ..., max_len-1]
        pos = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len, 1)
        # freqs: (max_len, d_k/2)
        freqs = pos * theta.unsqueeze(0) # angle to rotate for each position in a sentence based on the pos of a word in the embedding 
        # and the pairs are embedding dimensions
        self.register_buffer("cos_cached", freqs.cos())  # (max_len, d_k/2)
        self.register_buffer("sin_cached", freqs.sin())  # (max_len, d_k/2)

    def forward(self, seq_len, device):
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def apply_rope(x, cos, sin):
    # x: (batch, seq_len, d_k), cos/sin: (seq_len, d_k/2)
    d_half = x.shape[-1] // 2
    x0 = x[..., :d_half]
    x1 = x[..., d_half:]
    return torch.cat([
        x0 * cos - x1 * sin,
        x0 * sin + x1 * cos
    ], dim=-1)
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=500, drop_prob=0.1):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        return self.drop_out(tok_emb)
    
class RoPEAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cos, sin, mask=None):
        if mask is None:
            seq_len = x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        Q = self.W_q(x)
        K = self.W_k(x)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)


        D_q = self.d_k
        scores = (Q @ K.transpose(-2, -1))/(D_q **0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = self.softmax(scores)
        V = self.W_v(x)
        z = scores @ V
        
        return z
    
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super().__init__()
        self.multiLayers = nn.ModuleList([RoPEAttentionLayer(d_model, d_k, d_v) for i in range(n_heads)])
        self.linear = nn.Linear(n_heads * d_v, d_model)

    def forward(self, x, cos, sin, mask=None):
        multi_outputs = [head(x, cos, sin, mask) for head in self.multiLayers]
        concat = torch.cat(multi_outputs, dim=-1)

        output = self.linear(concat)
        return output
    
class Residual(nn.Module):
    def __init__(self, module, d_model, drop_p=0.1):
        super().__init__()
        self.module = module
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, *inp): # inp is a concatenated tensor of all input arguments (the tensor x (the data), cos, sin, mask)
        # TODO: Implement forward
        x = inp[0]
        normalized_x = self.layerNorm(x)
        rest = inp[1:]
        out = self.module(normalized_x, *rest)
        return x + self.dropout(out)

class Decoder(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_lin, n_layers, vocab_size):
        super().__init__()
        # TODO: Initialize TransformerEmbedding and n_layers of EncoderLayer
        # Hint: nn.ModuleList()
        self.transformEmbed = TransformerEmbedding(vocab_size, d_model)
        self.multiLayers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_k, d_v, d_lin) for i in range(n_layers)])
        self.rope = RotaryPositionalEncoding(d_k, max_len=512)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, inp): # inp (batch, seq_len): tells number of sentences in batch, and length of each sentence of batch
        # TODO: Implement forward by embedding the input and passing it through all layers
        seq_len = inp.shape[1]
        cos, sin = self.rope(seq_len, inp.device)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=inp.device)).bool()

        embedded = self.transformEmbed(inp)
        for layer in self.multiLayers:
            embedded = layer(embedded, cos, sin, mask) #inputs of decoderLayer forward

        return self.fc(embedded) #returns logits (B, L, V) for each of B sentences, for each of L positions, a score over the V vocab tokens
  

class FeedForward(nn.Module):
    def __init__(self, d_model, d_lin):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(d_model, d_lin)
        self.lin2 = nn.Linear(d_lin, d_model)

    def forward(self, inp):
        return self.lin2(self.relu(self.lin1(inp)))
    

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_lin):
        super().__init__()
        # TODO: Initialize MultiHeadAttention and FeedForward modules with Residual layers for both

        self.multi = RoPEMultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.multiRes = Residual(self.multi, d_model)
        self.ff = FeedForward(d_model, d_lin)
        self.ffRes = Residual(self.ff, d_model)
        
    def forward(self, inp, cos, sin, mask):
        seq_len = inp.shape[1]
        attention_out = self.multiRes(inp, cos, sin, mask)
        ff_out = self.ffRes(attention_out)
        return ff_out
def build_inv_vocab(word_dict):
    return {i: w for w, i in word_dict.items()}

def ids_to_words(ids, inv_vocab, specials={0, 1, 2, 3}):
    return " ".join(inv_vocab[i] for i in ids if i not in specials and i in inv_vocab)

@torch.no_grad()
def greedy_decode(model, prefix, eos_idx, max_new_tokens=50):
    # prefix: (1, L) containing [SOS] eng [SEP]
    model.eval()
    generated = prefix
    for _ in range(max_new_tokens):
        logits = model(generated)                     # (1, L, V)
        next_tok = logits[:, -1, :].argmax(dim=-1)    # (1,)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
        if next_tok.item() == eos_idx:
            break
    return generated[0].tolist()

@torch.no_grad()
def evaluate_bleu(model, test_loader, word_dict, device, max_new_tokens=50, max_batches=None):
    model.eval()
    inv_vocab = build_inv_vocab(word_dict)
    EOS = word_dict["EOS"]

    hyps, refs = [], []
    for b, (seq, sep_pos) in enumerate(test_loader):
        if max_batches is not None and b >= max_batches:
            break
        seq, sep_pos = seq.to(device), sep_pos.to(device)
        B = seq.size(0)
        for i in range(B):
            sp = sep_pos[i].item()
            prefix = seq[i, :sp+1].unsqueeze(0)         # [SOS] eng [SEP]
            ref_ids = seq[i, sp+1:].tolist()            # fr [EOS] pads
            if EOS in ref_ids:
                ref_ids = ref_ids[:ref_ids.index(EOS)]
            out_ids = greedy_decode(model, prefix, EOS, max_new_tokens)
            pred_fr = out_ids[sp+1:]                    # strip prefix
            if EOS in pred_fr:
                pred_fr = pred_fr[:pred_fr.index(EOS)]
            hyps.append(ids_to_words(pred_fr, inv_vocab))
            refs.append(ids_to_words(ref_ids, inv_vocab))

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    model.train()
    return bleu.score, hyps[:5], refs[:5]
    
def train(model, train_loader, epoch, optimizer, criterion):
    model.train()
    train_avg_loss = 0
    num_correct = 0
    num_tokens = 0

    batch_bar = tqdm.tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Train epoch {epoch}")
    for batch_idx, (seq, sep_pos) in enumerate(train_loader):
        seq = seq.to(device)
        sep_pos = sep_pos.to(device)

        inp    = seq[:, :-1]
        target = seq[:, 1:]

        logits = model(inp)                                       # (B, L-1, V)
        B, Lm1, V = logits.shape

        # French-only mask: target index j maps to original seq index j+1,
        # which is French iff j+1 > sep_pos  =>  j >= sep_pos
        positions = torch.arange(Lm1, device=device).unsqueeze(0)            # (1, L-1)
        french_mask = (positions >= sep_pos.unsqueeze(1)) & (target != 0)    # (B, L-1)

        # masked cross-entropy (criterion must be CrossEntropyLoss(reduction='none'))
        per_token_loss = criterion(logits.reshape(-1, V), target.reshape(-1)).reshape(B, Lm1)
        loss = (per_token_loss * french_mask).sum() / french_mask.sum().clamp(min=1)

        predictions = logits.argmax(dim=-1)                       # (B, L-1)
        num_correct += ((predictions == target) & french_mask).sum().item()
        num_tokens  += french_mask.sum().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_avg_loss += loss.item()
        running = train_avg_loss / (batch_idx + 1)
        batch_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running:.4f}")
        batch_bar.update()

    train_avg_loss /= len(train_loader)
    train_accuracy = num_correct / max(num_tokens, 1)
    return train_avg_loss, train_accuracy

def test(model, test_loader, epoch, criterion):
    model.eval()
    test_avg_loss = 0
    num_correct = 0
    num_tokens = 0

    with torch.no_grad():
        for seq, sep_pos in test_loader:
            seq = seq.to(device)
            sep_pos = sep_pos.to(device)

            inp    = seq[:, :-1]
            target = seq[:, 1:]

            logits = model(inp)
            B, Lm1, V = logits.shape

            positions = torch.arange(Lm1, device=device).unsqueeze(0)
            french_mask = (positions >= sep_pos.unsqueeze(1)) & (target != 0)

            per_token_loss = criterion(logits.reshape(-1, V), target.reshape(-1)).reshape(B, Lm1)
            loss = (per_token_loss * french_mask).sum() / french_mask.sum().clamp(min=1)

            predictions = logits.argmax(dim=-1)
            num_correct += ((predictions == target) & french_mask).sum().item()
            num_tokens  += french_mask.sum().item()

            test_avg_loss += loss.item()

    test_avg_loss /= len(test_loader)
    test_accuracy = num_correct / max(num_tokens, 1)
    return test_avg_loss, test_accuracy

def run(num_epochs, model, train_loader, test_loader, optimizer, criterion, vocab, max_len):
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, bleu_hist = [], [], [], [], []
    best_bleu = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_loss, train_acc = train(model, train_loader, epoch, optimizer, criterion)

        print(f"Train loss: {train_loss:.06f} | Accuracy: {train_acc*100:.04f}%")

        test_loss, test_acc = test(model, test_loader, epoch, criterion)
        print(f"Test loss: {test_loss:.06f} | Accuracy: {test_acc*100:.04f}%")

        bleu_score, sample_hyps, sample_refs = evaluate_bleu(
            model, test_loader, vocab, device, max_new_tokens=max_len, max_batches=20
        )
        print(f"BLEU: {bleu_score:.2f}")
        for h, r in zip(sample_hyps, sample_refs):
            print(f"  hyp: {h}")
            print(f"  ref: {r}")

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        bleu_hist.append(bleu_score)

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bleu': bleu_score,
            'vocab': vocab,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(checkpoint, 'best_model.pt')
            print(f"  New best BLEU: {best_bleu:.2f} — saved to best_model.pt")

    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, bleu_hist


def main(
    #Hyperparameters
    vocabulary_size = 32000,
    batch_size = 64,
    max_length = 100,
    lr = 1e-3,
    num_epochs = 10,
    n_heads = 8,
    d_model = 512,
    d_k = 64,
    d_v = 64,
    d_lin = 2048,
    n_layers = 6
):

    # TODO: Initialize Transformer model, criterion and optimizer
    model = Decoder(n_heads, d_model, d_k, d_v, d_lin, n_layers, vocab_size=vocabulary_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

   
    train_dataloader, test_dataloader, vocab = get_translation_dataloader(
        file_path, vocab_size=vocabulary_size, max_len=max_length, batch_size=batch_size)

    train_loss_list, train_acc_list, test_loss_list, test_acc_list, bleu_list = run(
      num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion,
      vocab, max_length
  )

    return train_loss_list, test_loss_list, bleu_list


if __name__ == '__main__':
    main()
    