import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm.notebook import tqdm
from collections import Counter
import pandas as pd
from functools import partial

class Vocabulary(object):
    def __init__(self, lil_tokens, max_vocab_size=10_000):
        self.lil_tokens = lil_tokens
        self.max_vocab_size = max_vocab_size
        self.tokens = []
        self.ids = {}

        # add special tokens 
        self.tokens.append('<bos>')
        self.tokens.append('<eos>')
        self.tokens.append('<pad>')
        self.tokens.append('<unk>')

        # add all the tokens 
        self.build_vocab()
    
    def build_vocab(self):
        all_tokens = [token for l_tokens in self.lil_tokens for token in l_tokens]
        counter = Counter(all_tokens)
        most_common = counter.most_common(self.max_vocab_size - len(self.tokens))
        self.tokens += [item[0] for item in most_common]
        self.ids = {token: id for id, token in enumerate(self.tokens)}

    def get_id(self, w):
        return self.ids[w]

    def get_token(self, id):
        return self.tokens[id]

    def decode_idx2token(self, list_id):
        return [self.tokens[i] for i in list_id]

    def encode_token2idx(self, list_token):
        return [self.ids[tok] if tok in self.ids else self.get_id('<unk>') for tok in list_token]

    def __len__(self):
        return len(self.tokens)


class QAPair(Dataset):
    def __init__(self, file_path):
        self.main_df, self.answer_vocab, self.question_vocab = load_qa_data(file_path)
        self.pad_idx = self.answer_vocab.get_id('<pad>')

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        return (self.main_df.iloc[idx]['source_indized'], self.main_df.iloc[idx]['target_indized'])

# Dataset utils function
def read_data(file_path):
    with open(file_path, 'r') as f:
        dataset = []
        for line in f:
            dataset.append(line.strip().split(' '))
    return dataset

def token2index_dataset(dataset_lil, vocab):
    index_lil = []
    for data in dataset_lil:
        data = ['<bos>'] + data + ['<eos>']
        index_data = vocab.encode_token2idx(data)
        index_lil.append(index_data)

    return index_lil

def load_qa_data(file_path):
    # read data 
    answer_lil = read_data(file_path['source'])
    question_lil = read_data(file_path['target'])

    # save list of words 
    main_df = pd.DataFrame()
    main_df['source_data'] = answer_lil
    main_df['target_data'] = question_lil

    # build dictionary for source and target 
    answer_vocab = Vocabulary(answer_lil, 45_000)
    question_vocab = Vocabulary(question_lil, 28_000)

    # convert words to idx for each dataset
    main_df['source_indized'] = token2index_dataset(answer_lil, answer_vocab)
    main_df['target_indized'] = token2index_dataset(question_lil, question_vocab)

    return main_df, answer_vocab, question_vocab

# Data loader util functions 
def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    
    for t in list_of_tensors:
        padded_tensor = torch.cat([t.unsqueeze(0), torch.tensor([[pad_token]*(max_length - t.size(-1))], dtype=torch.long)], dim = -1)
        padded_list.append(padded_tensor)
        
    padded_tensor = torch.cat(padded_list, dim=0)
    
    return padded_tensor

def pad_collate_fn(batch, pad_token):
    input_list = [torch.tensor(s[0]) for s in batch]
    target_list = [torch.tensor(s[1]) for s in batch]
        
    input_tensor = pad_list_of_tensors(input_list, pad_token)
    target_tensor = pad_list_of_tensors(target_list, pad_token)
    
    return input_tensor, target_tensor

if __name__ == "main":
    main_data_path = "data/processed"

    train_file_path = {
        'source': f"{main_data_path}/src-train.txt",
        'target': f"{main_data_path}/tgt-train.txt"
    }

    test_file_path = {
        'source': f"{main_data_path}/src-test.txt",
        'target': f"{main_data_path}/tgt-test.txt"
    }

    dev_file_path = {
        'source': f"{main_data_path}/src-dev.txt",
        'target': f"{main_data_path}/tgt-dev.txt"
    }

    dataset_dict = {
        'train': QAPair(train_file_path),
        'test': QAPair(test_file_path),
        'dev': QAPair(dev_file_path),
    }

    batch_size = 1024
    dataloader_dict = {}
    for split, dataset in dataset_dict.items():
        dataloader_dict[split] = DataLoader(
            dataset, 
            batch_size=1024, 
            shuffle=True, 
            collate_fn=partial(pad_collate_fn, pad_token=dataset.pad_idx)
        )