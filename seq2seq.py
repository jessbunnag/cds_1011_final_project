import torch.nn as nn
import torch

from encoder import BiLSTMEncoder, EncoderTransformer
from decoder import AttnLSTMDecoder

class Seq2Seq(nn.Module):
    '''
    I THINK WE NEED src_max_len INITIALIZED HERE
    '''
    def __init__(self, pretrained_vectors, src_max_len, hidden_size, output_size, num_layers=2, transformer_encoder=False, transformer_decoder=False):
        super(Seq2Seq, self).__init__()

        if transformer_encoder:
            self.enc = EncoderTransformer(pretrained_vectors['enc'], src_max_len, numlayers=num_layers)
        else:
            self.enc = BiLSTMEncoder(pretrained_vectors['enc'], hidden_size, num_layers)
        self.dec = AttnLSTMDecoder(pretrained_vectors['dec'], hidden_size, 2*hidden_size, output_size, num_layers)

    def forward(self, answer, question, src_lens, tgt_lens):    
        enc_output, enc_hidden = self.enc(answer)
        # calculate sentence encoder's output for init decoder hidden
        # concat last hidden state of forward and backward pass 
        enc_out_repr = torch.cat([
            enc_hidden[-2, :, :].unsqueeze(0),  # last layer of forward pass 
            enc_hidden[-1, :, :].unsqueeze(0)   # last layer of backward pass
        ], dim=0) # "s" in the paper
        
        dec_log_probs, dec_hidden, attn_scores_mat = self.dec(input=question, encoder_outs=enc_output, hidden_init=enc_out_repr)
        
        return dec_log_probs, dec_hidden, attn_scores_mat