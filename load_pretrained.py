import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm.notebook import tqdm

from build_dataset import build_train_vocab

def load_glove(file_path):
    with open(file_path, 'r') as f:
        glove_dict = {}
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embbeding = np.array(line[1:], dtype=np.float64)
            glove_dict[word] = torch.from_numpy(embbeding)
    return glove_dict

def make_glove_embeddings(glove_dict, vocab, hidden_size):
    pad_embedding = torch.zeros(hidden_size)
    bos_embedding = torch.rand(hidden_size)
    eos_embedding = torch.rand(hidden_size)
    unk_embedding = torch.rand(hidden_size)
    embedding_li = []

    for id, token in enumerate(vocab.tokens):
        if token in glove_dict:
            embedding = glove_dict[token]
        else:
            if token == '<bos>':
                embedding = bos_embedding
            elif token == '<eos>':
                embedding = eos_embedding
            elif token == '<pad>':
                embedding = pad_embedding
            else:
                embedding = unk_embedding
        embedding_li.append(embedding)

    return torch.stack(embedding_li)

emb_size = 300
glove_dict = load_glove(f'data/glove/glove.42B.{emb_size}d.txt')

train_file_path = {
    'source': f"data/processed/src-train.txt",
    'target': f"data/processed/tgt-train.txt"
}
vocab = build_train_vocab(train_file_path)
encoder_emb = make_glove_embeddings(glove_dict, vocab['source'], emb_size)
decoder_emb = make_glove_embeddings(glove_dict, vocab['target'], emb_size)

torch.save(encoder_emb, f'embeddings/encoder_emb_{emb_size}.pt')
torch.save(decoder_emb, f'embeddings/decoder_emb_{emb_size}.pt')