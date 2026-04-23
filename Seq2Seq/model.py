import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.0, param_init_range=0.08):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self._init_weights(param_init_range)

    def _init_weights(self, init_range):
        for _, param in self.named_parameters():
            nn.init.uniform_(param, -init_range, init_range)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, cell) = self.lstm(packed)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.0, param_init_range=0.08):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self._init_weights(param_init_range)

    def _init_weights(self, init_range):
        for _, param in self.named_parameters():
            nn.init.uniform_(param, -init_range, init_range)

    def forward(self, input_token, hidden, cell):
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc_out(output.squeeze(1))
        return output, hidden, cell

    def forward_sequence(self, tgt, hidden, cell):
        embedded = self.embedding(tgt)
        lstm_out, _ = self.lstm(embedded, (hidden, cell))
        outputs = self.fc_out(lstm_out)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.0, param_init_range=0.08):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            param_init_range=param_init_range,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            param_init_range=param_init_range,
        )

    def forward(self, src, src_lengths, tgt):
        hidden, cell = self.encoder(src, src_lengths)
        decoder_input = tgt[:, :-1]
        outputs = self.decoder.forward_sequence(decoder_input, hidden, cell)
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len, sos_idx, eos_idx):
        self.eval()
        batch_size = src.size(0)
        device = src.device

        hidden, cell = self.encoder(src, src_lengths)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        predictions = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            pred = output.argmax(dim=1)
            predictions.append(pred)

            finished = finished | (pred == eos_idx)
            if finished.all():
                break

            input_token = pred

        predictions = torch.stack(predictions, dim=1)
        return predictions


def build_model(config, src_vocab_size, tgt_vocab_size):
    return Seq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        param_init_range=config.param_init_range,
    )
