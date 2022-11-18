import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    """Encodes the input context. Based on code from Lab 6""" 
    def __init__(self, pretrained_vectors, embed_size=300, hidden_size=600, numlayers=2):
        """Initialize encoder.
        :param input_size: size of embedding
        :param hidden_size: size of hidden layers
        :param numlayers: number of layers
        """
        super().__init__()
        # for bidirectional encoding, divide hidden size by 2
        self.hidden_size = hidden_size // 2

        # pretrained glove embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=numlayers, batch_first=True, bidirectional=True
        )

    def forward(self, input, hidden=None):
        """Return encoded state.
        :param input: (batchsize x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        """
        # print(f'input in bilstm forward {input.shape}') 
        # print(f'embedding weight {self.embedding.weight}') 
        embedded = self.embedding(input)
        # print(f'embedded in bilstm forward {embedded.shape}') 
        output, hidden = self.lstm(embedded.float(), hidden) 
        return output, hidden