import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
import matplotlib.pyplot as plt
import sacrebleu

class DualLogger(object):
    def __init__(self, filename="base_rnn_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("base_rnn_output.txt")

from reference_data import get_translation_dataloader

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
SEP_IDX = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        output, hidden = self.rnn(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocab_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_vocab_size = output_vocab_size
        
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_token, hidden, encoder_outputs=None):
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.out(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.output_vocab_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(source)
        
        decoder_input = target[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output.squeeze(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2) 
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs

def split_batch(seqs, sep_positions):
    batch_size = seqs.size(0)
    
    max_src_len = sep_positions.max().item() + 1
    max_tgt_len = seqs.size(1) - sep_positions.min().item()
    
    sources = torch.full((batch_size, max_src_len), PAD_IDX, dtype=torch.long)
    targets = torch.full((batch_size, max_tgt_len), PAD_IDX, dtype=torch.long)
    
    for i in range(batch_size):
        sep = sep_positions[i].item()
        src = seqs[i, :sep+1] 
        sources[i, :len(src)] = src
        
        tgt_len = (seqs[i, sep:] != PAD_IDX).sum().item()
        tgt = seqs[i, sep : sep + tgt_len]
        tgt[0] = SOS_IDX 
        targets[i, :tgt_len] = tgt
        
    return sources, targets

def train(model, train_loader, epoch, optimizer, criterion):
    model.train()
    train_avg_loss = 0
    
    batch_bar = tqdm.tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Train Epoch {epoch}")
    
    for seqs, sep_positions in train_loader:
        source, target = split_batch(seqs, sep_positions)
        source, target = source.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(source, target)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_avg_loss += loss.item()
        batch_bar.set_postfix(loss="{:.06f}".format(loss.item()))
        batch_bar.update()
        
    batch_bar.close()
    return train_avg_loss / len(train_loader)

def evaluate(model, test_loader, epoch, criterion, id_to_word):
    model.eval()
    test_avg_loss = 0
    
    all_cand_strings = []
    all_ref_strings = []
    
    batch_bar = tqdm.tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Eval Epoch {epoch}")
    
    with torch.no_grad():
        for seqs, sep_positions in test_loader:
            source, target = split_batch(seqs, sep_positions)
            source, target = source.to(device), target.to(device)

            output = model(source, target, teacher_forcing_ratio=0.0)
            
            predictions = output.argmax(dim=-1).cpu().numpy()
            targets = target.cpu().numpy()
            
            for i in range(predictions.shape[0]):
                cand_words = []
                for val in predictions[i]:
                    if val == EOS_IDX: break           
                    if val not in (PAD_IDX, SOS_IDX, SEP_IDX): 
                        cand_words.append(id_to_word.get(val, ""))
                        
                ref_words = []
                for val in targets[i]:
                    if val == EOS_IDX: break
                    if val not in (PAD_IDX, SOS_IDX, SEP_IDX):
                        ref_words.append(id_to_word.get(val, ""))
                        
                all_cand_strings.append(" ".join(cand_words))
                all_ref_strings.append(" ".join(ref_words))
            
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)

            loss = criterion(output_flat, target_flat)
            test_avg_loss += loss.item()
            
            batch_bar.set_postfix(loss="{:.06f}".format(loss.item()))
            batch_bar.update()

    batch_bar.close()
    
    epoch_bleu = sacrebleu.corpus_bleu(all_cand_strings, [all_ref_strings]).score
    return test_avg_loss / len(test_loader), epoch_bleu

def run_training(num_epochs, model, train_loader, test_loader, optimizer, criterion, id_to_word):
    train_loss_hist, test_loss_hist, bleu_hist = [], [], []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, epoch, optimizer, criterion)
        print(f"Train loss: {train_loss:.06f}")

        test_loss, test_bleu = evaluate(model, test_loader, epoch, criterion, id_to_word)
        print(f"Test loss: {test_loss:.06f} | Test BLEU: {test_bleu:.02f}")
    
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        bleu_hist.append(test_bleu)
        
    return train_loss_hist, test_loss_hist, bleu_hist

if __name__ == '__main__':
    csv_path = "en-fr.csv" 
    
    train_loader, test_loader, word_dict = get_translation_dataloader(
        csv_path=csv_path, 
        vocab_size=10000, 
        batch_size=128,
        n_samples=11_000_000,               
        cache_path="translation_cache_11M.pt" 
    )
    
    id_to_word = {v: k for k, v in word_dict.items()}
    
    VOCAB_SIZE = len(word_dict)
    HIDDEN_SIZE = 256
    NUM_EPOCHS = 10
    LR = 0.001

    encoder = EncoderRNN(input_vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(hidden_size=HIDDEN_SIZE, output_vocab_size=VOCAB_SIZE).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_loss_list, test_loss_list, bleu_list = run_training(
        NUM_EPOCHS, model, train_loader, test_loader, optimizer, criterion, id_to_word
    )

    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Baseline RNN: Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, bleu_list, label='Test BLEU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('Baseline RNN: BLEU vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rnn_training_curves.png')
    plt.show()