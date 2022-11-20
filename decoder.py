import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):

    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.l1 = nn.Linear(hidden_dim*2, hidden_dim, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        att_score = self.l1(encoder_outs) 
        att_score = torch.bmm(hidden, att_score.transpose(1,2)) 

        attn_scores = F.softmax(att_score.squeeze(), dim=1).unsqueeze(2)

        context = torch.bmm(encoder_outs.transpose(1, 2), attn_scores)
 
        return context.squeeze(), attn_scores.squeeze()

    def sequence_mask(self, sequence_length, max_len=None, device = torch.device('cuda')):
        if max_len is None:
            max_len = sequence_length.max().item()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size, 1])
        seq_range_expand = seq_range_expand.to(device)
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return (seq_range_expand < seq_length_expand).float()


class AttnLSTMDecoder(nn.Module):
    def __init__(self, pretrained_vectors, hidden_size, enc_out_size, out_vocab_size, num_layers):
        super(AttnLSTMDecoder, self).__init__()

        self.embed_size = pretrained_vectors.shape[1]
        self.hidden_size = hidden_size
        self.output_size = out_vocab_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        
        self.lstm_cell1 = nn.LSTMCell(self.hidden_size * 2 * 2, self.hidden_size, bias=True)
        self.lstm = nn.LSTM(
            self.embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )

        self.l1 = nn.Linear(hidden_size+enc_out_size, hidden_size, bias=False) # TODO: change l1 output size
        self.l2 = nn.Linear(hidden_size, out_vocab_size, bias=False)
        
        self.encoder_attention_module = AttentionModule(self.hidden_size)
        
    def forward(self, input, encoder_outs, hidden_init, targets_len):
        embedded = self.embedding(input)

        cell_init = torch.zeros_like(hidden_init)

        output, (h_out, _) = self.lstm(embedded, (hidden_init,cell_init)) 
        T = output.shape[1]
        log_probs = []

        for t in range(T):
            h_t = output[:, t, :]
            c_t, attn_scores = self.encoder_attention_module(h_t.unsqueeze(1), encoder_outs, targets_len)
            h_t_c_t = torch.cat([h_t, c_t], dim=1) 
        
            # pass concat of c_t;h_t into linear --> tanh --> linear --> softmax 
            fc_out = self.l2(torch.tanh(self.l1(h_t_c_t)))

            log_prob = F.log_softmax(fc_out,  dim=1) 

            log_probs.append(log_prob)

        log_probs_tensor = torch.stack(log_probs, dim=1)

        return log_probs_tensor, h_out