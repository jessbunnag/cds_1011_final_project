import torch
import heapq 

class BeamNode(object):
    def __init__(self, neg_log_prob, prefix, dec_output, dec_hidden):
        # use neg log prob because we will use min heap when getting top k candiates during each search
        # max log prob = min neg log prob
        self.neg_log_prob = neg_log_prob
        self.prefix = prefix
        self.dec_output = dec_output
        self.dec_hidden = dec_hidden

    def __repr__(self) -> str:
        s = (
            f"Prefix {self.prefix}\nSum log prob {-self.neg_log_prob}\n" +
            f"Decoder output {self.dec_output}\nDecoder hidden{self.dec_hidden}"
        )
        return s
        
    def __lt__(self, other):
        return self.neg_log_prob < other.neg_log_prob


class Beam(object):
    def __init__(self, beam_width, vocab, init_hidden) -> None:
        self.beam_width = beam_width
        self.vocab = vocab 
        self.bos_id = vocab.get_id('<bos>')
        self.eos_id = vocab.get_id('<eos>')
        self.frontier = []
        self.candidates = []

        self.init_frontier(init_hidden)

    def init_frontier(self, init_hidden):
        start_prefix = [self.vocab.get_id('<bos>')]
        start_tensor = torch.tensor(start_prefix)
        start_state = Beam(
            neg_log_prob=0, 
            prefix=start_prefix, 
            dec_output=start_tensor, 
            dec_hidden=init_hidden
        )
        self.frontier.append(start_state)

    def add_candidates(self, prev_node, candidates):
        """Add possible candidate at each search to candidates
        Input: 
            prev_code: BeamNode object
            candidates: tensor shape [1, vocab_size]
        """
        for i in range(len(self.vocab)):
            candidate = candidates[i]
            self.candidates.append(candidate)


    


def beam_search(model, encoder_states, vocab, beam_width=3):
    """
    Parameters:
        model: trained seq2seq model 
        answer: answer sentence 
        beam_width: number of hypotheses to keep when searching through the beam

    Return: 
        best_paths
    """
    enc_output, init_hidden = encoder_states

    beam = Beam(beam_width, vocab, init_hidden)

    while True: #TODO : add condition if frontier contains incomplete paths 
        for state in beam.frontier:
            print(state)
            dec_input = state.dec_output
            dec_hidden = state.dec_hidden
            dec_log_probs, dec_hidden, attn_scores_mat = model.dec(input=dec_input, encoder_outs=enc_output, hidden_init=dec_hidden)

        break
    