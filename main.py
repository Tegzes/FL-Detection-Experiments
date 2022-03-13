
import random, os
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm 

from datamodule import data
from models import LSTM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + str(device))

# global variables
SEED = 97
BATCH_SIZE = 4
HIDDEN_DIM = 64
OUTPUT_DIM = 1
EMBEDDING_LENGTH = 300
N_LAYERS = 2
LEARNING_RATE = 2e-5
BIDIRECTIONAL = True
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

TEXT, EMBEDDING_DIM, VOCAB_SIZE, word_embeddings, train_iterator, valid_iterator, test_iterator = data.load_dataset(batch_size = BATCH_SIZE, device=device) # add folder arg


model = LSTM.LSTMSarcasm(OUTPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_LENGTH, N_LAYERS, device)


loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

model.to(device)
loss_function = loss_function.to(device)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def binary_accuracy(preds, y):

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()

    for _, batch in enumerate(iterator):
        tweet, tweet_len = batch.tweet
        labels = batch.sarcastic

        optimizer.zero_grad()
        
        predictions = model(tweet, tweet_len.to('cpu'))
        loss = criterion(predictions, labels)
        
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

    
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for _, batch in enumerate(iterator):
            tweet, tweet_len = batch.tweet
            labels = batch.sarcastic
            predictions = model(tweet, tweet_len.to('cpu'))
            
            loss = criterion(predictions, labels)
            
            acc = binary_accuracy(predictions, labels)

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

