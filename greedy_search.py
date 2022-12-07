import torch 
from seq2seq_transformer import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    print(f'src {src.shape}')
    print(f'src_mask {src_mask.shape}')

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        print(f'ys {ys.shape}')
        print(f'memory {memory.shape}')
        print(f'tgt_mask {tgt_mask.shape}')
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def generate_question(model: torch.nn.Module, src_sentence, vocab):
    model.eval()
    src = torch.tensor(vocab['source'].encode_token2idx(src_sentence))
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return vocab['target'].decode_idx2token(tgt_tokens.tolist())
