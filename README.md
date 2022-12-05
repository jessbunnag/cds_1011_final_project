# 1011 Final Project - Question Generation 

## How to run the model
1. Download glove.42B.300d.zip GloVe data from https://nlp.stanford.edu/projects/glove/ and put it in `data/glove/glove.42B.300d.txt`in your project dir
2. Run `python load_pretrained.py` to create parsed GloVe embeddings. Make sure you have empty dir `/embeddings` in your 
3. Run cells in `train.ipynb` to run the train pipeline.

## Resources 
- Learning to Ask: Neural Question Generation for Reading Comprehension by Xinya Du, Junru Shao, Claire Cardie: http://arxiv.org/abs/1705.00106
- [Singularity setup doc](https://docs.google.com/document/d/12D09OvptZ3OIMpjm3k_reLL4sipCftfiEMwLAl2tkm8/edit?usp=sharing)

## To-do list 
- [x] Set up singularity instance
- [x] Download GloVE word vectors and put it on Greene 
- [x] Preprocessing (Note: we have decided to move forward using the author's proprocessed dataset using Stanford CoreNLP)
  - [x] Filter questions with is_impossible = False only
  - [x] Tokenization and sentence splitting 
  - [x] Lowercase the entire dataset
  - [x] Locate the sentence containing the answer --> input sentence 
  - [x] If answer spans 2+ sentences, concatenate the sentences --> input sentence 
  - [x] Prune training set so that input sentence and question has 1+ word in common 
  - [x] Add <BOS> and <EOS>
- [x] Splitting data 80:10:10 train,val,test
- [x] Create vocab + dataloader (input = python list of list of tokens) 
  - [x] source side V (input sentence) : keep 45k most frequent tokens from train 
  - [x] target side V (question) : keep 28k most frequent tokens from train 
  - [x] for both vocabs, replace unknown with <UNK>
- [x] Define the model encoder
  - [x] Embedding : d=50, GloVE pre-trained embeddings (NOT learned) 
  - [x] Embedding : d=300, GloVE pre-trained embeddings (NOT learned) 
  - [x] Bi directional LSTM, hidden size = 600, num layer = 2
  - [x] Concat last hidden state of forward and backward pass --> encoder's output (for decoder) 
- [x] Define the model decoder 
  - [x] LSTM, hidden size = 600
  - [x] Add 2 layers for LSTM 
  - [x] Add dropout 0.3
  - [x] Attention based encoding 
  - [ ] Implement beam search
  - [x] Figure out if we need attention mask
- [x] Define Seq2seq model (like in hw2) 
- [x] Training 
  - [x] Optimization: SGD, lr=1.0, halve lr at epoch 8, batch size = 64, Dropout = 0.3
  - [x] Learning rate update
  - [x] Clip gradient at norm > 5
  - [x] Implement train step 
  - [x] Implement train loop  
- [ ] Inference 
  - [ ] Implement beam search at inference 
  - [ ] Implement evaluation metric 
  - [ ] Post-processing with the replacement of UNK
- [ ] Extensions 
  - [ ] Transformer Encoder 
  - [ ] Transformer Decoder 
  
## Notes 
### SQuAD Raw Data structure 
- `train_data` (loaded directly from json file) is a dictionary with 2 keys: "version" and "data". We only work with `train_data['data']`
- `train_data['data']` is a list of dictionaries. Each dictionary corresponds to a single article.
- `train_data['data'][i]` (each article) is a dictionary with 2 keys: "title" and "paragraphs". We are only interested in `train_data['data'][i]["paragraphs"]`
- `train_data['data'][i]["paragraphs"]` is a list of dictionaries. Each dictionary corresponds to a single paragraph in the article. 
- `train_data['data'][i]["paragraphs"][j]` is a dictionary with 2 keys: "qas" and "context". Let's call `train_data['data'][i]["paragraphs"][j]` a `paragraph`
  - `paragraph['context']` is the raw text of the paragraph
  - `paragraph['qas']` is a list of dictionaries. Each dictionary corresponds to a question-answer pair. 
    - `paragraph['qas'][i]` is a dictionary with 4 keys: "question", "id", "answers", "is_impossible" 
      - `paragraph['qas'][i]["question"]` is a raw string of the question. 
      - `paragraph['qas'][i]["id"]` is the question-answer id. 
      - `paragraph['qas'][i]["answers"]` is a list of dictionary. Each dictionary corresponds to an answer.
        - `paragraph['qas'][i]["answers"][j]` is a dictionary with 2 keys: 'text' and 'answer_start'. Let's call `paragraph['qas'][i]["answers"][j]` an 'answer'
          - `answer['text']` is the answer text 
          - `answer['answer_start']` is the character index in the context where the text answer starts 
