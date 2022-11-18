# 1011 Final Project - Question Generation 

## Resources 
- Learning to Ask: Neural Question Generation for Reading Comprehension by Xinya Du, Junru Shao, Claire Cardie: http://arxiv.org/abs/1705.00106
- [Singularity setup doc](https://docs.google.com/document/d/12D09OvptZ3OIMpjm3k_reLL4sipCftfiEMwLAl2tkm8/edit?usp=sharing)

## To-do list 
- [x] Set up singularity instance
- [x] Download GloVE word vectors and put it on Greene 
- [ ] Preprocessing 
  - [ ] Filter questions with is_impossible = False only
  - [x] Tokenization and sentence splitting 
  - [ ] Lowercase the entire dataset
  - [ ] Locate the sentence containing the answer --> input sentence 
  - [ ] If answer spans 2+ sentences, concatenate the sentences --> input sentence 
  - [ ] Prune training set so that input sentence and question has 1+ word in common 
  - [ ] Add <BOS> and <EOS>
- [x] Splitting data 80:10:10 train,val,test
- [x] Create vocab + dataloader (input = python list of list of tokens) 
  - [x] source side V (input sentence) : keep 45k most frequent tokens from train 
  - [x] target side V (question) : keep 28k most frequent tokens from train 
  - [x] for both vocabs, replace unknown with <UNK>
- [x] Define the model ENCODER (input = pairs of input sentence (list of tokens) and target (list of tokens))
  - [x] Embedding : d=300, GloVE pre-trained embeddings (NOT learned) 
  - [x] Bi directional LSTM, hidden size = 600, num layer = 2
  - [x] Concat last hidden state of forward and backward pass --> encoder's output (for decoder) 
- [x] Define the model decoder 
  - [x] LSTM, hidden size = 600, num layer = 1
  - [ ] Add 2 layers for LSTM 
  - [x] Attention based encoding 
- [ ] Define Seq2seq model (like in hw2) 
- [ ] Training 
  - [ ] Optimization: SGD, lr=1.0, halve lr at epoch 8, batch size = 64, Dropout = 0.3, Clip gradient at norm > 5
  - [ ] Implement train step 
  - [ ] Implement train loop 
