import torch
import heapq as hq

class BeamNode(object):
    def __init__(self, log_prob, prefix, dec_output, dec_hidden, attn_score_mat):
        # use neg log prob because we will use min heap when getting top k candiates during each search
        # max log prob = min neg log prob
        self.log_prob = log_prob
        self.prefix = prefix
        self.dec_output = dec_output
        self.dec_hidden = dec_hidden
        self.attn_score_mat = attn_score_mat

    def __repr__(self) -> str:
        s = (
            f"Prefix {self.prefix}\nSum log prob {self.log_prob}\n" +
            f"Decoder output {self.dec_output}\nDecoder hidden {self.dec_hidden}"
        )
        return s
        
    def __lt__(self, other):
        return self.log_prob < other.log_prob


class Beam(object):
    def __init__(self, beam_width, vocab, init_hidden, device, max_src_len, max_tgt_len):
        self.beam_width = beam_width
        self.vocab = vocab 
        self.bos_id = vocab.get_id('<bos>')
        self.eos_id = vocab.get_id('<eos>')
        self.device = device
        self.frontier = []
        self.candidates = []
        self.complete_paths = []
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.init_frontier(init_hidden)

    def init_frontier(self, init_hidden):
        start_prefix = [self.vocab.get_id('<bos>')]
        start_tensor = torch.tensor(start_prefix).unsqueeze(0).to(self.device) # batch size of 1
        start_state = BeamNode(
            log_prob=0, 
            prefix=start_prefix, 
            dec_output=start_tensor, 
            dec_hidden=init_hidden,
            attn_score_mat=torch.zeros((self.max_src_len, 1)).to(self.device)
        )
        self.frontier.append(start_state)

    def add_candidates(self, prev_node, candidates_logprob, dec_hidden, attn_score_mat):
        """Add possible candidate at each search to list of candidates
        Input: 
            prev_code: BeamNode object
            candidates: tensor shape [1, vocab_size]
        """
        # print(f'add candidates fn')
        for i in range(len(self.vocab)):
            if i == self.bos_id:
                continue
            word_logprob = candidates_logprob[0, i].item()

            new_state = BeamNode(
                log_prob=prev_node.log_prob + word_logprob,
                prefix=prev_node.prefix + [i],
                dec_output=torch.tensor([i]).unsqueeze(0).to(self.device),
                dec_hidden=dec_hidden,
                attn_score_mat=torch.cat((prev_node.attn_score_mat, attn_score_mat), dim=1)
            )

            self.candidates.append(new_state)

            # add state to candidate while mantaining heap variants
            if len(self.candidates) > self.beam_width:
                self.candidates = sorted(self.candidates, reverse=True)[:self.beam_width]

        # print(f'len candidates {len(self.candidates)}')
        # print(f'best in candidate {self.candidates[0].prefix[-1]}')
        # print(f'candidates {self.candidates}')

    def update_frontier(self):
        """Check if any node is the beam is complete. 
        If so, then add it to complete paths"""

        # self.candidates = sorted(self.candidates, reverse=True)[:self.beam_width]
        self.frontier = []
        for node in self.candidates:
            if node.prefix[-1] == self.eos_id or len(node.prefix) > self.max_tgt_len: # path is complete 
                self.complete_paths.append(node)
                self.beam_width -= 1
            else:
                self.frontier.append(node)

        # print(f'after updating frontier')
        # for node in self.frontier:
        #     print(node)
        #     print(f'prefix {self.vocab.decode_idx2token(node.prefix) }')
        # reset candidates for next search
        self.candidates = []

    def is_done(self):
        return self.beam_width == 0 or len(self.frontier) == 0

    def get_complete_paths(self):
        return self.complete_paths

    def get_best_k_paths(self, k):
        complete_paths = sorted(self.complete_paths, reverse=True)[:k]
        truncated_paths = []
        truncated_attn = []
        for node in complete_paths:
            if node.prefix[-1] == self.eos_id:
                truncated_paths.append(node.prefix[1:-1])
                truncated_attn.append(node.attn_score_mat[:, 1:-1])
            else:
                truncated_paths.append(node.prefix[1:])
                truncated_attn.append(node.attn_score_mat[:, 1:])
        return [self.vocab.decode_idx2token(path) for path in truncated_paths], truncated_attn


def beam_search(model, encoder_states, vocab, device, max_tgt_len, beam_width):
    """
    Parameters:
        model: trained seq2seq model 
        answer: answer sentence 
        beam_width: number of hypotheses to keep when searching through the beam

    Return: 
        best_paths
    """
    enc_output, init_hidden = encoder_states

    beam = Beam(beam_width, vocab, init_hidden, device, enc_output.shape[1], max_tgt_len)

    while not beam.is_done(): 
        for state in beam.frontier:
            dec_input = state.dec_output # [1, 1] (batch size = 1, sen len = 1)
            dec_hidden = state.dec_hidden.contiguous() #[2,1,600]
            # print(f'dec_input shape {dec_input.shape}')
            # print(f'dec_hidden shape {dec_hidden.shape}')
            dec_log_probs, dec_hidden, attn_scores_mat = model.dec(
                input=dec_input, encoder_outs=enc_output, hidden_init=dec_hidden
            )

            # print(f'attn_scores_mat {attn_scores_mat.shape}') #[1,src_sen_len,1]

            # print(f'dec_log_probs {dec_log_probs.shape}') #[1,1,vocab_size]
            # print(f'dec_hidden out {dec_hidden.shape}') # [2,1,600]

            beam.add_candidates(state, dec_log_probs.squeeze(1), dec_hidden, attn_scores_mat.squeeze(0))

        beam.update_frontier()

    # print(f'complete paths {beam.get_complete_paths()}' )
    best_paths, attn_mat = beam.get_best_k_paths(1)
    return best_paths[0], attn_mat[0]
            

def beam_search_batch(batch, model, vocab, device, beam_width=3,
        hidden_size=600, n_layers=2, transformer_encoder=False
    ):
    input = batch.source_vecs.to(device)  
    
    model.eval()

    # model encoder forward
    enc_output, enc_hidden = model.enc(input)
        # calculate sentence encoder's output for init decoder hidden
        # concat last hidden state of forward and backward pass 
    batch_size = enc_output.shape[0]

    if transformer_encoder == True:
        enc_out_repr = torch.zeros(((n_layers, batch_size, hidden_size))).to(device)
    else:
        enc_out_repr = torch.cat([
            enc_hidden[-2, :, :].unsqueeze(0),  # last layer of forward pass 
            enc_hidden[-1, :, :].unsqueeze(0)   # last layer of backward pass
        ], dim=0) # "s" in the paper


    # print(f'enc_output {enc_output.shape}') # [batch_size, seq_len, hidden_dim*2]
    # print(f'enc_out_repr {enc_out_repr.shape}') # [2, batch_size, hidden_dim]

    preds_list = []
    labels_list = []
    attn_score_list = []
    for i in range(batch_size):
        # print(f'target question: {batch.target_data[i]}')
        
        encoder_state = (enc_output[i].unsqueeze(0), enc_out_repr[:, i, :].unsqueeze(1))
        # print(f'enc_output i {encoder_state[0].shape}') # [1, seq_len, hidden_dim*2]
        # print(f'enc_out_repr i {encoder_state[1].shape}') 

        best_path, attn_score = beam_search(
            model, encoder_state, vocab, device, 
            max_tgt_len=batch.target_lens[i] + 5, 
            beam_width=beam_width
        )
        # print(f'predicted question: {best_path}')
   
        labels_list.append(batch.target_data[i])
        preds_list.append(best_path)
        attn_score_list.append(attn_score)
        # break


    return preds_list, labels_list, attn_score_list