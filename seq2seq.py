import torch.nn as nn

from encoder import BiLSTMEncoder
from AttnDecoder import *

class Seq2Seq(nn.Module):
    def __init__(self, pretrained_vectors, enc_embed_size, hidden_size, output_size, num_layers=2):
        super(Seq2Seq, self).__init__()

        self.enc = BiLSTMEncoder(pretrained_vectors['enc'], enc_embed_size, hidden_size, num_layers)
        self.dec = AttnDecoder(pretrained_vectors['dec'], output_size, hidden_size)

    def forward(self, answer, src_lens):    
        enc_output, enc_hidden = self.enc(answer)
        # calculate sentence encoder's output for init decoder hidden
        # concat last hidden state of forward and backward pass 
        enc_out_repr = torch.cat([enc_hidden[-2, :, :], enc_hidden[-1, :, :]], dim=1)
        print(f'enc_out_repr {enc_out_repr.shape}')
        # decoder_output, decoder_hidden, _, _ = self.dec(enc_hidden, enc_output, src_lens, context_vec=enc_output)
        
        # return decoder_output, decoder_hidden
        return