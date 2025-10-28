import torch
from torch import nn
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)

    def forward(self, src):
        src = src.unsqueeze(0)
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell
    
    
class WildfireSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_ratio = 0.5):
        
        batch_size = trg.size(0)
        output_dim = self.decoder.fc_out.out_features

        # decide trg_length
        if trg is not None:
            trg_length = trg.size(1)
        else:
            # default to same length as src
            trg_length = src.size(1)

        outputs = torch.zeros(batch_size, trg_length, output_dim, device=self.device, dtype = src.dtype)

        hidden, cell = self.encoder(src)

        input = trg[:, 0]  # (batch, feature_dim)

        for t in range(0, trg_length - 1):
            output, hidden, cell = self.decoder(input, hidden, cell)
                
            outputs[:, t] = output

            teacher_force = random.random() < teacher_ratio

            predicted_obs = output

            input = trg[:, t + 1].to(self.device) if teacher_force else predicted_obs.argmax(1)

        return outputs


if __name__ == "__main__":
    print()