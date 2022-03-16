
import random, os
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm 

from transformers import AutoTokenizer, RobertaTokenizer
from datamodule import data
from models import LSTM, CNN, roberta, bertweet

bertweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + str(DEVICE))
print('Device name:', torch.cuda.get_device_name(0))


# global variables
SEED = 97
BATCH_SIZE = 4
HIDDEN_DIM = 64
OUTPUT_DIM = 1
EMBEDDING_LENGTH = 300
N_LAYERS = 2
LEARNING_RATE = 2e-5
BIDIRECTIONAL = False
DROPOUT = 0.25
N_EPOCHS = 3

# for the reproductibility if the experiments
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(SEED)

TEXT, EMBEDDING_DIM, VOCAB_SIZE, word_embeddings, train_iterator, valid_iterator, test_iterator, pad_idx = data.load_dataset(batch_size = BATCH_SIZE, device=DEVICE)


# BiLSTM model
# model = LSTM.LSTMSarcasm(OUTPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_LENGTH, N_LAYERS, BIDIRECTIONAL)


# attention LSTM model
# model = LSTM.LSTMSarcasmAttn(OUTPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_LENGTH, N_LAYERS)


# Roberta model
train_iterator, valid_iterator, test_iterator = data.get_dataloader(tokenizer_bert = roberta_tokenizer)
model = roberta.RobertaSarc()

# Bertweet model
# train_iterator, valid_iterator, test_iterator = data.get_dataloader(tokenizer_bert = bertweet_tokenizer)
# model = bertweet.BertweetClass()


loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

model.to(DEVICE)
loss_function = loss_function.to(DEVICE)

def calcuate_accuracy(preds, targets):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == targets).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# train routine
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()

    for _, batch in enumerate(iterator):

        ids = batch['ids'].to(DEVICE, dtype = torch.long)
        mask = batch['mask'].to(DEVICE, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(DEVICE, dtype = torch.long)
        labels = batch['targets'].to(DEVICE, dtype = torch.long)

        _, predictions = torch.max(model(ids, mask, token_type_ids).data, dim=1)

        optimizer.zero_grad()
        
        # predictions = model(tweet, tweet_len.to('cpu'))
        #predictions = model(tweet) #, tweet_len.to('cpu'))

        loss = criterion(predictions, labels)
        
        acc = calcuate_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# evaluation routine
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for _, batch in enumerate(iterator):

            ids = batch['ids'].to(DEVICE, dtype = torch.long)
            mask = batch['mask'].to(DEVICE, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(DEVICE, dtype = torch.long)
            labels = batch['targets'].to(DEVICE, dtype = torch.long)

            _, predictions = torch.max(model(ids, mask, token_type_ids).data, dim=1)

            # predictions = model(tweet, tweet_len.to('cpu'))
            #predictions = model(tweet) #, tweet_len.to('cpu'))

            loss = criterion(predictions, labels)
            
            acc = calcuate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# experiment loop
for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, loss_function)
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_function)
        
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')



test_loss, test_acc = evaluate(model, test_iterator, loss_function)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

