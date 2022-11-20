import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):

    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.l1 = nn.Linear(hidden_dim*2, hidden_dim, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        # print(f"===ATTENTION FORWARD===")
        # print(f"hidden shape {hidden.shape}")
        # print(f"encoder_outs shape {encoder_outs.shape}")

        att_score = self.l1(encoder_outs) #[128, 600, L]
        # print(f"l1(encoder_outs) transpose shape {att_score.transpose(1,2).shape}")
        att_score = torch.bmm(hidden, att_score.transpose(1,2)) #[128, 1, L]
        # print(f"att_score shape {att_score.shape}")

        attn_scores = F.softmax(att_score.squeeze()).unsqueeze(2)
        # print(f"att_score after softmax shape {attn_scores.shape}")

        # TODO: confirm whether we need sequence mask 

        # print(f"encoder_outs transpose shape {encoder_outs.transpose(1, 2).shape}")
        context = torch.bmm(encoder_outs.transpose(1, 2), attn_scores)
        # print(f'context shape {context.shape}')
        return context.squeeze(), attn_scores.squeeze()

        # OLD IMPLEMENTATION 
        # x = self.l1(hidden)
        # print(f'x unsqueeze {x.unsqueeze(-1).shape}')
        # att_score = torch.bmm(encoder_outs, x.unsqueeze(-1)); #this is bsz x seq x 1
        # att_score = att_score.squeeze(-1); #this is bsz x seq
        # att_score = att_score.transpose(0, 1)
        
        # seq_mask = self.sequence_mask(src_lens, 
        #                             max_len=max(src_lens).item(), 
        #                             device = hidden.device).transpose(0, 1)


        # masked_att = seq_mask * att_score
        # masked_att[masked_att == 0] = -1e10
        # attn_scores = F.softmax(masked_att, dim=0)
        # x = (attn_scores.unsqueeze(2) * encoder_outs.transpose(0, 1)).sum(dim=0)
        # x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        # return x, attn_scores

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

        # pretrained glove embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        
        self.lstm_cell1 = nn.LSTMCell(self.hidden_size * 2 * 2, self.hidden_size, bias=True)
        self.lstm = nn.LSTM(
            self.embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )

        self.l1 = nn.Linear(hidden_size+enc_out_size, hidden_size, bias=False) # TODO: change l1 output size
        self.l2 = nn.Linear(hidden_size, out_vocab_size, bias=False)
        
        self.encoder_attention_module = AttentionModule(self.hidden_size)

    def beamsearch(self, batch_sequences, batch_log_probs, k=3):
        cur_batch_sequences = []

        print('===beam search===')
        # loop through every example in the batch with its corresponding probabilities 
        for ex_log_probs in batch_log_probs:
            all_candidates = list()

            print('batch seq at b: ', batch_sequences)

            # expand on each candidate in the current example of the batch
            for b in range(len(batch_sequences)):
                print('batch sequence at b', batch_sequences[b])
                seq, score = batch_sequences[b][0], batch_sequences[b][1]
                for j in range(len(ex_log_probs)):
                    candidate = [seq + [j], score - ex_log_probs[j].item()]
                    all_candidates.append(candidate)

            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            print(f'top 3 sequences in batch{b}', ordered[:k])

            # select k best
            cur_batch_sequences.append(ordered[:k])

        return cur_batch_sequences
        
    def forward(self, input, encoder_outs, hidden_init, targets_len):
        # print(f'===DECODER FORWARD===')
        # print(f'input shape {input.shape}') 
        embedded = self.embedding(input)

        cell_init = torch.zeros_like(hidden_init)
        # print(f'embedded shape {embedded.shape}') 
        # print(f'hidden_init {hidden_init.shape}')
        # print(f'cell_init {cell_init.shape}')

        output, (h_out, _) = self.lstm(embedded, (hidden_init,cell_init)) 
        # print(f'output shape {output.shape}') 
        # print(f'h_out shape {h_out.shape}')

        T = output.shape[1]
        print('T:', T)

        batch_sequences = [[[list(), 0.0]]]

        # for each time step (of target) : this should be the target length of the batched targets 
        for t in range(T):
            # compute attention and c_t 
            # input = hidden, b_i enc_output 
            h_t = output[:, t, :]
            # print(f'h_t shape {h_t.shape}')
            c_t, attn_scores = self.encoder_attention_module(h_t.unsqueeze(1), encoder_outs, targets_len)
            # print(f'c_t shape {c_t.shape}')
            # print(f'attn_scores {attn_scores.shape}')

            # concat c_t and h_t
            h_t_c_t = torch.cat([h_t, c_t], dim=1) 
            # print(f'h_t_c_t {h_t_c_t.shape}')
        
            # pass concat of c_t;h_t into linear --> tanh --> linear --> softmax 
            fc_out = self.l2(F.tanh(self.l1(h_t_c_t)))
            # print(f'fc_out {fc_out.shape}')
            log_prob = F.log_softmax(fc_out) # --> [0, 1, .., 28K] --> [0.2, 0.8, ..., 0.5]
            print(f'log_prob {log_prob.shape}')
            # print(f'log_prob {log_prob}')

            # TODO: do beam search at each time step 
            # k = 3, take top k values (and indices) from the log_prob list 
            
            batch_sequences = self.beamsearch(batch_sequences, log_prob)

        return 
        # return return_scores.transpose(0, 1).contiguous(), memory.transpose(0,1), attn_wts_list, context_vec
