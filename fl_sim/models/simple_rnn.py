"""
Reccurent Neural Network for Shakespeare Dataset
"""
import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, vocab_size=90, embedding_dim=8, hidden_dim=512,
                 num_layers=2):
        super(RNN, self).__init__()

        # set class variables
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_lstm_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers)

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=512,
            batch_first=True,
            num_layers=2)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embeds = self.embedding(input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc(lstm_out)
        # flatten the output
        out = out.reshape(-1, self.vocab_size)
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.num_lstm_layers, batch_size,
                              self.hidden_dim).to(device),
                  torch.zeros(self.num_lstm_layers, batch_size,
                              self.hidden_dim).to(device))
        return hidden


def simple_rnn(pretrained=False, num_classes=90):
    return RNN(vocab_size=num_classes)


def mini_simple_rnn(pretrained=False, num_classes=90):
    return RNN(vocab_size=num_classes, hidden_dim=128)
