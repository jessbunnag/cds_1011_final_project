import torch
from torchtext.data.metrics import bleu_score
# from torchmetrics import BLEUScore
# from nltk.translate.bleu_score import modified_precision
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score
from torchmetrics.text.rouge import ROUGEScore
# from ignite.metrics import RougeL
import numpy as np


def unk_postprocessing(src, attn_scores_mat):
    '''
    Inputs:
        attn_scores_mat: 3D pytorch tensor of size (batch_size x max SRC sen len per batch x max TGT sen len per batch)
        src: 2D pytorch tensor of size (batch_size x max SRC sen len per batch)

    Returns:
        best_attn_labels: 2D pytorch tensor of size (batch_size x max TGT sen len per batch)
    '''
    # get the src token index with the max attention for this time step
    src_token_idx = torch.argmax(attn_scores_mat, dim=1)
    # print('src_token_idx SIZE', src_token_idx.size())

    # convert the index location in the src to the vocab class label for src
    best_attn_labels = torch.zeros(src_token_idx.size())
    for sen_idx in range(src_token_idx.size(0)):
        for tgt_idx in range(src_token_idx.size(1)):
            class_label = src[sen_idx][src_token_idx[sen_idx][tgt_idx]]
            best_attn_labels[sen_idx][tgt_idx] = class_label

    return best_attn_labels

def vanilla_inference(dec_log_probs, tgt_labels, tgt_vocab):
    '''
    Inputs:
        dec_log_probs: 3D pytorch tensor of size (batch_size x max TGT sen len per batch x TGT class vocab size)
        tgt_labels: 2D pytorch tensor of size (batch_size x max TGT sen len per batch)
        tgt_vocab: Vocabulary class object
    Returns:
        preds_list: list of lists of tokens with the predicted TGTs
        labels_list: list of lists of tokens with the label TGTs
    '''
    # find the argmax class for each token of each sentence in the batch
    pred_indices = torch.argmax(dec_log_probs, dim=2)

    preds_list = []
    labels_list = []
    # decode the indices of the predictions and targets to tokens
    for sen_idx in range(pred_indices.size(0)):
        tgt_pred_tokens = tgt_vocab.decode_idx2token(pred_indices[sen_idx].tolist())
        tgt_label_tokens = tgt_vocab.decode_idx2token(tgt_labels[sen_idx].tolist())

        clean_tgt_pred_tokens = []
        clean_tgt_label_tokens = []
        # clean the preds_list and labels_list by removing <bos>, <eos>, and all <pad> tokens after <eos>
        for idx, token in enumerate(tgt_label_tokens):
            if token == '<bos>':
                pass
            elif token == '<eos>':
                break
            else:
                clean_tgt_pred_tokens.append(tgt_pred_tokens[idx])
                clean_tgt_label_tokens.append(tgt_label_tokens[idx])

        preds_list.append(clean_tgt_pred_tokens)
        labels_list.append(clean_tgt_label_tokens)
        
    return preds_list, labels_list

def eval_metrics(preds_list, labels_list):
    '''
    Inputs:
        preds_list: list of lists of tokens with the predicted TGTs
        labels_list: list of lists of tokens with the label TGTs

    Returns:
        X
    '''
    
    # compute BLEU score:
    weights1 = [1.0/1.0]
    weights2 = [1.0/2.0, 1.0/2.0]
    weights3 = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    weights4 = [1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0]

    bleu_1 = bleu_score(preds_list, labels_list, max_n=1, weights=weights1)
    bleu_2 = bleu_score(preds_list, labels_list, max_n=2, weights=weights2)
    bleu_3 = bleu_score(preds_list, labels_list, max_n=3, weights=weights3)
    bleu_4 = bleu_score(preds_list, labels_list, max_n=4, weights=weights4)

    # # compute the meteor score:
    # all_meteor = []

    # for idx in range(len(labels_list)):
    #     all_meteor.append(meteor_score(labels_list[idx], preds_list[idx]))

    # compute ROUGE-L:


    ### from torchmetrics import BLEUScore:
    # all_bleu_1 = []
    # all_bleu_2 = []
    # all_bleu_3 = []
    # all_bleu_4 = []
    # metric1 = BLEUScore(n_gram=1)
    # metric2 = BLEUScore(n_gram=2)
    # metric3 = BLEUScore(n_gram=3)
    # metric4 = BLEUScore(n_gram=4)
    # for idx in range(len(labels_list)):
    #     all_bleu_1.append(metric1(preds_list[idx], labels_list[idx]).item())
    #     all_bleu_2.append(metric2(preds_list[idx], labels_list[idx]).item())
    #     all_bleu_3.append(metric3(preds_list[idx], labels_list[idx]).item())
    #     all_bleu_4.append(metric4(preds_list[idx], labels_list[idx]).item())

    return bleu_1, bleu_2, bleu_3, bleu_4