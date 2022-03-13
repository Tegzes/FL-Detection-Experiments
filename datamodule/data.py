
from sklearn.model_selection import validation_curve
import torch
from torchtext import data
from torchtext.vocab import Vectors, GloVe, FastText
import random

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

SEED = 97

def spacy_tokenizer(tweet):
    return [token.text for token in tokenizer(tweet)]

def load_dataset(batch_size, device):
    
    TEXT = data.Field(batch_first = True,
                    #   use_vocab = False,
                      tokenize = spacy_tokenizer,
                      lower=True,
                      include_lengths=True,
                      )

    LABEL = data.LabelField(sequential=False, batch_first=True, dtype=torch.float)

    fields = [(None, None), ('tweet', TEXT), ('sarcastic', LABEL)]

    full_data = data.TabularDataset(
                            path = '/home/tegzes/Desktop/Disertatie/Disertatie/main-project/sarcasm_1/datamodule/isarcasm2022_clean.csv',
                            format = 'csv',
                            fields = fields)

    train_data_temp, test_data = full_data.split(split_ratio=0.7, random_state=random.seed(SEED))
    train_data, validation_data = train_data_temp.split(split_ratio=0.7, random_state=random.seed(SEED))

    # TEXT.unk_token = '<unk>'
    # TEXT.pad_token = '<pad>'
    # TEXT.eos_token = '<eos>'

    TEXT.build_vocab(train_data, max_size = 10000, min_freq = 1, vectors=GloVe('6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors # builds embeddings to pass to model
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, validation_data, test_data), 
                                                                    batch_size=batch_size, 
                                                                    sort_key=lambda x: len(x.tweet), 
                                                                    sort = False,
                                                                    shuffle=True, 
                                                                    sort_within_batch = True, 
                                                                    device=device)
                                                                                                            
                                                                                                          
    vocab_size = len(TEXT.vocab)
    EMBEDDING_DIM = TEXT.vocab.vectors.shape[1]

    return TEXT, EMBEDDING_DIM, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
