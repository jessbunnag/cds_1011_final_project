import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

PAD_ID = 0

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

class PositionalEncoding(nn.Module):
    '''
    Based off of https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    FROM MY UNDERSTANDING WE NEED AN INPUT CALLED src_max_len THAT REPRESENTS THE MAX LENGTH OF SRC
    '''
    def __init__(self, emb_size, dropout, src_max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(src_max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(src_max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Returns:
            3D tensor of size max SRC sentence length x batch size x SRC embedding dimension (300)
        '''
        print('POSITIONAL ENCODING SIZE IN', x.size())
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EncoderTransformer(nn.Module):
    '''
    FROM MY UNDERSTANDING WE NEED AN INPUT CALLED src_max_len THAT REPRESENTS THE MAX LENGTH OF SRC
    '''
    def __init__(self, pretrained_vectors, src_max_len, nhead=1, numlayers=2, dropout=0.3):
        super().__init__()
        emb_size = pretrained_vectors.shape[1]
        self.position_embed = PositionalEncoding(emb_size=emb_size, dropout=dropout, src_max_len=src_max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=numlayers)
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.embed_size = emb_size
        
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embed_size)
        x = self.position_embed(x)
        x = self.transformer(x)
        hidden = torch.mean(x, dim=1).unsqueeze(0)
        return x, hidden