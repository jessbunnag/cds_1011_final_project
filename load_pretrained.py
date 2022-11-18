import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm.notebook import tqdm

from build_dataset import QAPair

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

glove_dict = load_glove('data/glove/glove.6B.50d.txt')

train_file_path = {
    'source': f"data/processed/src-train.txt",
    'target': f"data/processed/tgt-train.txt"
}
train_dataset = QAPair(train_file_path)
encoder_emb = make_glove_embeddings(glove_dict, train_dataset.answer_vocab, 50)
decoder_emb = make_glove_embeddings(glove_dict, train_dataset.question_vocab, 50)

torch.save(encoder_emb, 'embeddings/encoder_emb.pt')
torch.save(decoder_emb, 'embeddings/decoder_emb.pt')