import torch.nn as nn
import torch 

from encoder import BiLSTMEncoder
from decoder import AttnLSTMDecoder

class Seq2Seq(nn.Module):
    def __init__(self, pretrained_vectors, hidden_size, output_size, num_layers=2):
        super(Seq2Seq, self).__init__()

        self.enc = BiLSTMEncoder(pretrained_vectors['enc'], hidden_size, num_layers)
        self.dec = AttnLSTMDecoder(pretrained_vectors['dec'], hidden_size, 2*hidden_size, output_size, num_layers)

    def forward(self, answer, question, src_lens, tgt_lens):    
        enc_output, enc_hidden = self.enc(answer)
        # calculate sentence encoder's output for init decoder hidden
        # concat last hidden state of forward and backward pass 
        enc_out_repr = torch.cat([
            enc_hidden[-2, :, :].unsqueeze(0), 
            enc_hidden[-1, :, :].unsqueeze(0)
        ], dim=0) # "s" in the paper
        print(f'enc_out_repr {enc_out_repr.shape}')
        
        _ = self.dec(input=question, encoder_outs=enc_output, hidden_init=enc_out_repr, targets_len=src_lens)
        
        # return decoder_output, decoder_hidden
        return