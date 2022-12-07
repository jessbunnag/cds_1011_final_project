import torch
import heapq as hq

class BeamNode(object):
    def __init__(self, log_prob, prefix, dec_output, dec_hidden):
        # use neg log prob because we will use min heap when getting top k candiates during each search
        # max log prob = min neg log prob
        self.log_prob = log_prob
        self.prefix = prefix
        self.dec_output = dec_output
        self.dec_hidden = dec_hidden

    def __repr__(self) -> str:
        s = (
            f"Prefix {self.prefix}\nSum log prob {self.log_prob}\n" +
            f"Decoder output {self.dec_output}\nDecoder hidden {self.dec_hidden}"
        )
        return s
        
    def __lt__(self, other):
        return self.log_prob < other.log_prob


class Beam(object):
    def __init__(self, beam_width, vocab, init_hidden, device):
        self.beam_width = beam_width
        self.vocab = vocab 
        self.bos_id = vocab.get_id('<bos>')
        self.eos_id = vocab.get_id('<eos>')
        self.device = device
        self.frontier = []
        self.candidates = []
        self.complete_paths = []

        self.init_frontier(init_hidden)

    def init_frontier(self, init_hidden):
        start_prefix = [self.vocab.get_id('<bos>')]
        start_tensor = torch.tensor(start_prefix).unsqueeze(0).to(self.device) # batch size of 1
        start_state = BeamNode(
            log_prob=0, 
            prefix=start_prefix, 
            dec_output=start_tensor, 
            dec_hidden=init_hidden
        )
        self.frontier.append(start_state)

    def add_candidates(self, prev_node, candidates_logprob, dec_hidden):
        """Add possible candidate at each search to list of candidates
        Input: 
            prev_code: BeamNode object
            candidates: tensor shape [1, vocab_size]
        """
        # print(f'add candidates fn')
        print(f'argmax {candidates_logprob.argmax(dim=1)}')
        for i in range(len(self.vocab)):
            if i == self.bos_id:
                continue
            word_logprob = candidates_logprob[0, i].item()

            new_state = BeamNode(
                log_prob=prev_node.log_prob + word_logprob,
                prefix=prev_node.prefix + [i],
                dec_output=torch.tensor([i]).unsqueeze(0).to(self.device),
                dec_hidden=dec_hidden
            )

            # add state to candidate while mantaining heap variants
            if len(self.candidates) < self.beam_width:
                hq.heappush(self.candidates, new_state)
            else:
                if new_state > self.candidates[0]: # candidates is min heap so the top is always the lowest prob
                    hq.heappop(self.candidates) # remove the worst in the candidates 
                    hq.heappush(self.candidates, new_state)

        print(f'len candidates {len(self.candidates)}')
        print(f'best in candidate {self.candidates[0].prefix[-1]}')
        # print(f'candidates {self.candidates}')

    def update_frontier(self):
        """Check if any node is the beam is complete. 
        If so, then add it to complete paths"""

        self.frontier = []
        for node in self.candidates:
            if node.prefix[-1] == self.eos_id: # path is complete 
                print(f'path is complete ')
                print(f'{node.prefix}')
                self.complete_paths.append(node.prefix)
                self.beam_width -= 1
            else:
                self.frontier.append(node)

        print(f'after updating frontier')
        for node in self.frontier:
            print(node)
            print(f'prefix {self.vocab.decode_idx2token(node.prefix) }')
        # reset candidates for next search
        self.candidates = []

    def is_done(self):
        return self.beam_width == 0 or len(self.frontier) == 0

    def get_complete_paths(self):
        return self.complete_paths



def beam_search(model, encoder_states, vocab, device, beam_width):
    """
    Parameters:
        model: trained seq2seq model 
        answer: answer sentence 
        beam_width: number of hypotheses to keep when searching through the beam

    Return: 
        best_paths
    """
    enc_output, init_hidden = encoder_states

    beam = Beam(beam_width, vocab, init_hidden, device)

    while not beam.is_done(): 
        for state in beam.frontier:
            dec_input = state.dec_output # [1, 1] (batch size = 1, sen len = 1)
            dec_hidden = state.dec_hidden.contiguous() #[2,1,600]
            # print(f'dec_input shape {dec_input.shape}')
            # print(f'dec_hidden shape {dec_hidden.shape}')
            dec_log_probs, dec_hidden, attn_scores_mat = model.dec(
                input=dec_input, encoder_outs=enc_output, hidden_init=dec_hidden
            )

            # print(f'dec_log_probs {dec_log_probs.shape}') #[1,1,vocab_size]
            # print(f'dec_hidden out {dec_hidden.shape}') # [2,1,600]

            beam.add_candidates(state, dec_log_probs.squeeze(1), dec_hidden)

        beam.update_frontier()

    print(f'complete paths {beam.get_complete_paths()}' )
            

def beam_search_batch(batch, model, vocab, device, beam_width=3):
    input = batch.source_vecs.to(device)  
    
    model.eval()

    # model encoder forward
    enc_output, enc_hidden = model.enc(input)
        # calculate sentence encoder's output for init decoder hidden
        # concat last hidden state of forward and backward pass 
    enc_out_repr = torch.cat([
        enc_hidden[-2, :, :].unsqueeze(0),  # last layer of forward pass 
        enc_hidden[-1, :, :].unsqueeze(0)   # last layer of backward pass
    ], dim=0) # "s" in the paper

    encoder_states = (enc_output, enc_out_repr)

    # print(f'enc_output {enc_output.shape}') # [batch_size, seq_len, hidden_dim*2]
    # print(f'enc_out_repr {enc_out_repr.shape}') # [2, batch_size, hidden_dim]

    batch_size = enc_output.shape[0]

    for i in range(batch_size):
        print(f'target question: {batch.target_data[i]}')
        encoder_state = (enc_output[i].unsqueeze(0), enc_out_repr[:, i, :].unsqueeze(1))
        # print(f'enc_output i {encoder_state[0].shape}') # [1, seq_len, hidden_dim*2]
        # print(f'enc_out_repr i {encoder_state[1].shape}') 

        beam_search(model, encoder_state, vocab, device, beam_width)
        break


    return 