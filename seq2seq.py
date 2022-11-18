import torch.nn as nn

from encoder import BiLSTMEncoder
from decoder import AttnDecoder

class Seq2Seq(nn.Module):
    def __init__(self, pretrained_vectors, output_size, input_size=300, hidden_size=600, num_layers=2):
        super(Seq2Seq, self).__init__()

        self.enc = BiLSTMEncoder(pretrained_vectors, input_size, hidden_size, num_layers)
        self.dec = AttnDecoder(pretrained_vectors, output_size, hidden_size)

    def forward(self, answer, question, src_lens):
        
        enc_output, enc_hidden = self.enc(answer)
        decoder_output, decoder_hidden, _, _ = self.dec(question, enc_hidden, enc_output, src_lens, context_vec=enc_output)

        return decoder_output, decoder_hidden