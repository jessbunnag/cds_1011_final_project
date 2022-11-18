import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_Module(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(Attention_Module, self).__init__()
        self.l1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim + output_dim, output_dim, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        x = self.l1(hidden)
        att_score = torch.bmm(encoder_outs, x.unsqueeze(-1)); #this is bsz x seq x 1
        att_score = att_score.squeeze(-1); #this is bsz x seq
        att_score = att_score.transpose(0, 1);
        
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


class AttnDecoder(nn.Module):
    def __init__(self, pretrained_vectors, output_size, hidden_size):
        super(AttnDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        
        self.memory_lstm = nn.LSTMCell(self.hidden_size * 2, self.hidden_size, bias=True)

        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.encoder_attention_module = Attention_Module(self.hidden_size, self.hidden_size)
        
    def forward(self, memory, encoder_output, xs_len, context_vec = None):
        # context_vec is initialized with the last hidden layer of the encoder 
        memory = memory.transpose(0, 1)

        emb = self.embedding(encoder_output)
        emb = F.relu(emb) 
        
        emb = emb.transpose(0, 1)
        return_scores = torch.empty(emb.size(0), emb.size(1), self.output_size).to(encoder_output.device)
        
        if context_vec is None:
            context_vec = torch.zeros([emb.size(1), self.hidden_size]).to(emb.device)

        attn_wts_list = []

        for t in range(emb.size(0)):
            current_vec = emb[t]

            current_vec = torch.cat([current_vec, context_vec], dim = 1)
            selected_memory = memory[:, 0, :]

            mem_out = self.memory_lstm(current_vec, selected_memory)
    
            context_vec, attention0 = self.encoder_attention_module(mem_out, encoder_output, xs_len)

            scores = self.linear1(context_vec)
            scores = F.tanh(scores)
            scores = self.linear2(scores)

            attn_wts_list.append(attention0)

            scores = self.softmax(scores)
            return_scores[t] = scores

            memory = mem_out[:, None, :];
            
        return return_scores.transpose(0, 1).contiguous(), memory.transpose(0,1), attn_wts_list, context_vec