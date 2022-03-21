
import torch
from torchtext import data
from torchtext.vocab import Vectors, GloVe, FastText
import random
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

SEED = 97
sarc_path = '/home/tegzes/Desktop/Disertatie/Disertatie/main-project/sarcasm_1/datamodule/isarcasm2022_clean.csv'

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
                            path = sarc_path,
                            format = 'csv',
                            fields = fields)

    train_data_temp, test_data = full_data.split(split_ratio=0.7, random_state=random.seed(SEED))
    train_data, validation_data = train_data_temp.split(split_ratio=0.7, random_state=random.seed(SEED))

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
                                                                                                            
    pad_token = '<pad>'
    pad_idx = TEXT.vocab[pad_token]
                                                                                                    
    vocab_size = len(TEXT.vocab)
    EMBEDDING_DIM = TEXT.vocab.vectors.shape[1]

    return TEXT, EMBEDDING_DIM, vocab_size, word_embeddings, train_iter, valid_iter, test_iter, pad_idx



# Roberta data loader

class SarcasmData(Dataset):
    def __init__(self, tweet, sarcastic, tokenizer, max_len):
        self.tokenizer = tokenizer
        # self.data = dataframe
        self.text = tweet
        self.targets = sarcastic
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }



def get_dataloader(file_path = sarc_path,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 1,
                    max_len = 256,
                    tokenizer_bert = None):


    dataset = pd.read_csv(file_path)

    # train_data, test_data = train_test_split(dataset, test_size = 0.35, random_state = SEED)
    # test_data, validation_data = train_test_split(test_data, test_size = 0.5, random_state = SEED)

    train_data_temp = dataset.sample(frac=0.8, random_state=SEED)
    test_data = dataset.drop(train_data_temp.index).reset_index(drop=True)
    train_data_temp = train_data_temp.reset_index(drop=True)

    train_data = train_data_temp.sample(frac=0.8, random_state=SEED)
    validation_data = train_data_temp.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    training_set = SarcasmData(train_data.tweet, train_data.sarcastic, tokenizer_bert, max_len)
    validation_set = SarcasmData(validation_data.tweet, validation_data.sarcastic, tokenizer_bert, max_len)
    testing_set = SarcasmData(test_data.tweet, test_data.sarcastic, tokenizer_bert, max_len)

    training_loader = DataLoader(training_set, batch_size, shuffle = shuffle, num_workers = num_workers)
    validation_loader = DataLoader(validation_set, batch_size, shuffle = shuffle, num_workers = num_workers)
    testing_loader = DataLoader(testing_set, batch_size, shuffle = shuffle, num_workers = num_workers)

    return training_loader, validation_loader, testing_loader