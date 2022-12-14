import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    """Encodes the input context. Based on code from Lab 6""" 
    def __init__(self, pretrained_vectors, hidden_size=600, numlayers=2):
        """Initialize encoder.
        :param input_size: size of embedding
        :param hidden_size: size of hidden layers
        :param numlayers: number of layers
        """
        super().__init__()
        self.embed_size = pretrained_vectors.shape[1]
        # for bidirectional encoding, divide hidden size by 2
        self.hidden_size = hidden_size // 2

        # pretrained glove embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.lstm = nn.LSTM(
            self.embed_size, hidden_size, num_layers=numlayers, batch_first=True, bidirectional=True,
            dropout=0.3
        )

    def forward(self, input):
        """Return encoded state.
        :param input: (batchsize x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        """
        # print(f'===ENCODER FORWARD===') 
        # print(f'input shape {input.shape}') 
        embedded = self.embedding(input)
        # print(f'embedded shape {embedded.shape}') 
        output, (hidden, _) = self.lstm(embedded) 
        # print(f'output shape {output.shape}') 
        # print(f'hidden shape {hidden.shape}')
        return output, hidden