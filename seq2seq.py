import torch.nn as nn
import torch

from encoder import BiLSTMEncoder, EncoderTransformer
from decoder import AttnLSTMDecoder

class Seq2Seq(nn.Module):
    def __init__(self, pretrained_vectors, src_max_len, hidden_size, output_size, num_layers=2, transformer_encoder=False, transformer_seq2seq=False, device=torch.device('cuda')):
        super(Seq2Seq, self).__init__()

        if transformer_encoder == True:
            self.enc = EncoderTransformer(pretrained_vectors['enc'], src_max_len, numlayers=num_layers)
            # since the transformer is not bidirectional, need to modify enc_out_size
            self.dec = AttnLSTMDecoder(pretrained_vectors['dec'], hidden_size, hidden_size, output_size, num_layers, transformer_encoder, transformer_seq2seq)
        else:
            self.enc = BiLSTMEncoder(pretrained_vectors['enc'], hidden_size, num_layers)
            self.dec = AttnLSTMDecoder(pretrained_vectors['dec'], hidden_size, 2*hidden_size, output_size, num_layers, transformer_encoder, transformer_seq2seq)
        self.transformer_encoder = transformer_encoder
        self.transformer_seq2seq = transformer_seq2seq
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

    def forward(self, answer, question, src_lens, tgt_lens):    
        enc_output, enc_hidden = self.enc(answer)
        # calculate sentence encoder's output for init decoder hidden
        # concat last hidden state of forward and backward pass 
        if (self.transformer_encoder == False) and (self.transformer_seq2seq == False):
            enc_out_repr = torch.cat([
                enc_hidden[-2, :, :].unsqueeze(0), # last layer of forward pass
                enc_hidden[-1, :, :].unsqueeze(0)   # last layer of backward pass
            ], dim=0) # "s" in the paper

        # if we're using a transformer encoder, initialize the hidden state of the decoder with zeros
        elif (self.transformer_encoder == True) and (self.transformer_seq2seq == False):
            enc_out_repr = torch.zeros(((self.num_layers, src_lens.size(0), self.hidden_size))).to(self.device)
        # print(f'enc_out_repr {enc_out_repr.shape}')
        
        dec_log_probs, dec_hidden, attn_scores_mat = self.dec(input=question, encoder_outs=enc_output, hidden_init=enc_out_repr)
        
        return dec_log_probs, dec_hidden, attn_scores_mat