import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Attention_Module, self).__init__()
        self.l1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim + output_dim, output_dim, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        print(f"ATTENTION FORWARD")
        print(f"hidden shape {hidden.shape}")
        print(f"encoder_outs shape {encoder_outs.shape}")
        x = self.l1(hidden)
        print(f'attention module forward x unsqueeze {x.unsqueeze(-1).shape}')
        print(f'attention module forward encoder_outs {encoder_outs.shape}')
        att_score = torch.bmm(encoder_outs, x.unsqueeze(-1)); #this is bsz x seq x 1
        att_score = att_score.squeeze(-1); #this is bsz x seq
        att_score = att_score.transpose(0, 1)
        
        seq_mask = self.sequence_mask(src_lens, 
                                    max_len=max(src_lens).item(), 
                                    device = hidden.device).transpose(0, 1)


        masked_att = seq_mask * att_score
        masked_att[masked_att == 0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)
        x = (attn_scores.unsqueeze(2) * encoder_outs.transpose(0, 1)).sum(dim=0)
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        return x, attn_scores

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
    def __init__(self, pretrained_vectors, embed_size, hidden_size, output_size, num_layers):
        super(AttnLSTMDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size

        # pretrained glove embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        
        self.lstm_cell1 = nn.LSTMCell(self.hidden_size * 2 * 2, self.hidden_size, bias=True)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )

        self.l1 = nn.Linear(hidden_size*2, hidden_size, bias=False) # TODO: change l1 output size
        self.l2 = nn.Linear(hidden_size, output_size, bias=False)
        
        self.encoder_attention_module = AttentionModule(self.hidden_size, self.hidden_size)
        
    def forward(self, input, encoder_outs, hidden_init, targets_len):
        print(f'===DECODER FORWARD===')
        print(f'input shape {input.shape}') 
        embedded = self.embedding(input)

        print(f'embedded shape {embedded.shape}') 
        output, (h_out, _) = self.lstm(embedded, hidden_init) 
        print(f'output shape {output.shape}') 
        print(f'h_out shape {h_out.shape}')

        T = output.shape[1]

        log_probs = []

        # for each time step (of target) : this should be the target length of the batched targets 
        for t in range(T):
            # compute attention and c_t 
            # input = hidden, b_i enc_output 
            h_t = output[:, t, :].squeeze()
            print(f'before calling attention')
            c_t, attn_scores = self.encoder_attention_module(h_t, encoder_outs, targets_len)
            print(f'c_t shape {c_t.shape}')
            print(f'attn_scores {attn_scores.shape}')

            # concat c_t and h_t
            h_t_c_t = torch.cat([h_t, c_t], dim=1) 
        
            # pass concat of c_t;h_t into linear --> tanh --> linear --> softmax 
            fc_out = self.l2(F.tanh(self.l1(h_t_c_t)))
            log_prob = F.log_softmax(fc_out)

            log_probs.append(log_prob)

            # TODO: do beam search at each time step 

        return torch.stack(log_probs)
        # return return_scores.transpose(0, 1).contiguous(), memory.transpose(0,1), attn_wts_list, context_vec