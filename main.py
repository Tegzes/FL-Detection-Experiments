
import torch
import torch.nn as nn
import torchmetrics

from tqdm import tqdm 
import hydra
# from omegaconf import DictConfig

from clearml import Task, Logger

from datamodule import data, utils
from models import LSTM, CNN, Bertweet, Roberta

from transformers import AutoTokenizer, RobertaTokenizer
bertweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
print("Device: " + str(DEVICE))
print('Device name:', torch.cuda.get_device_name(0))

# task = Task.init(project_name="FL Detection", task_name="Training")

# global variables
SEED = 97
sarc_path = '/home/tegzes/Desktop/FL-Detection-Experiments/datamodule/isarcasm2022.csv'
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 2
MAX_LEN = 256
HIDDEN_DIM = 64
OUTPUT_DIM = 1
EMBEDDING_LENGTH = 300
N_LAYERS = 2
LEARNING_RATE = 1e-5
BIDIRECTIONAL = False
DROPOUT = 0.25
N_EPOCHS = 3

# for the reproductibility if the experiments
utils.seed_everything(SEED)

#TEXT, EMBEDDING_DIM, VOCAB_SIZE, word_embeddings, train_iterator1, valid_iterator1, test_iterator1, pad_idx = data.load_dataset(sarc_path, BATCH_SIZE, DEVICE, SEED)


# BiLSTM model
# model = LSTM.LSTMSarcasm(OUTPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_LENGTH, N_LAYERS, BIDIRECTIONAL)


# attention LSTM model
# model = LSTM.LSTMSarcasmAttn(OUTPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_LENGTH, N_LAYERS)


# Roberta model
train_iterator, valid_iterator, test_iterator = data.roberta_data_loader(sarc_path, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, True, 0, MAX_LEN, roberta_tokenizer, SEED)
# model = Roberta.RobertaSarc()
# model = Roberta.RobertaLSTMSarc(N_LAYERS, BIDIRECTIONAL)

# Bert + LSTM
# model = Roberta.BertLSTM()

# Bertweet model
# train_iterator, valid_iterator, test_iterator = data.get_dataloader(tokenizer_bert = bertweet_tokenizer)
model = Bertweet.BertweetClass()

model.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


# train routine
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    no_of_iterations = 0
    no_of_examples = 0
    
    # torch metrics
    metric_acc = torchmetrics.Accuracy().to(torch.device("cuda", 0))
    metric_f1 = torchmetrics.F1(num_classes = 2, average="none").to(torch.device("cuda", 0))
    metric_f1_micro = torchmetrics.F1(num_classes = 2).to(torch.device("cuda", 0))
    metric_f1_macro = torchmetrics.F1(num_classes = 2, average='macro').to(torch.device("cuda", 0))
    metric_precision = torchmetrics.Precision(num_classes = 2, average="none").to(torch.device("cuda", 0))
    metric_recall = torchmetrics.Recall(num_classes = 2, average="none").to(torch.device("cuda", 0))
  
    
    model.train()

    for batch_idx, batch in tqdm(enumerate(iterator, 0)):
        
        ids = batch['ids'].to(DEVICE, dtype = torch.long)
        mask = batch['mask'].to(DEVICE, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(DEVICE, dtype = torch.long)
        targets = batch['targets'].to(DEVICE, dtype = torch.long)
        tweet_lens = batch['tweet_len']

        # outputs = model(ids, mask, token_type_ids, tweet_lens.to('cpu'))
        outputs = model(ids, mask, token_type_ids)

        # targets = targets.unsqueeze(1) # for BCEWithLogitsLoss criterion
        loss = criterion(outputs, targets)
        epoch_loss += loss.item()

        _, predictions = torch.max(outputs.data, dim = 1)
        acc = calcuate_accuracy(predictions, targets)

        no_of_iterations += 1
        no_of_examples += targets.size(0)
                
        metric_acc.update(predictions, targets)
        metric_f1.update(outputs, targets)
        metric_f1_micro.update(outputs, targets)
        metric_f1_macro.update(outputs, targets)
        metric_precision.update(predictions, targets)
        metric_recall.update(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        # for GPU
        optimizer.step()

        # Logger.current_logger().report_scalar(
        #     "train", "loss", iteration = (epoch * len(iterator) + batch_idx), value = loss.item())
#----
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(batch['ids']), len(iterator),
        #         100. * batch_idx / len(iterator), loss.item()))
        

    epoch_loss = epoch_loss/no_of_iterations
    epoch_acc = (acc*100)/no_of_examples
        
    acc_torch = metric_acc.compute()
    print(f"Training Accuracy: {acc_torch}")
    
    f1 = metric_f1.compute()
    print(f"Training F1: {f1}")
    
    f1_micro = metric_f1_micro.compute()
    print(f"Training F1 Micro: {f1_micro}")

    f1_macro = metric_f1_macro.compute()
    print(f"Training F1 Macro: {f1_macro}")
 
    precision = metric_precision.compute()
    print(f"Training Precision: {precision}")

    recall = metric_recall.compute()
    print(f"Training Recall: {recall}")
    
    print(f"Training Loss Epoch: {epoch_loss}")
  
    
    metric_acc.reset()
    metric_f1.reset()
    metric_f1_micro.reset()
    metric_f1_macro.reset()
    metric_precision.reset()
    metric_recall.reset()
    
    return epoch_loss, epoch_acc

# evaluation routine
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    no_of_iterations = 0
    no_of_examples = 0    
    
   # torch metrics
    metric_acc = torchmetrics.Accuracy().to(torch.device("cuda", 0))
    metric_f1 = torchmetrics.F1(num_classes = 2, average="none").to(torch.device("cuda", 0))
    metric_f1_micro = torchmetrics.F1(num_classes = 2).to(torch.device("cuda", 0))
    metric_f1_macro = torchmetrics.F1(num_classes = 2, average='macro').to(torch.device("cuda", 0))
    metric_precision = torchmetrics.Precision(num_classes = 2, average="none").to(torch.device("cuda", 0))
    metric_recall = torchmetrics.Recall(num_classes = 2, average="none").to(torch.device("cuda", 0))
  

    model.eval()
    
    with torch.no_grad():
    
        for _, batch in tqdm(enumerate(iterator, 0)):

            ids = batch['ids'].to(DEVICE, dtype = torch.long)
            mask = batch['mask'].to(DEVICE, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(DEVICE, dtype = torch.long)
            targets = batch['targets'].to(DEVICE, dtype = torch.long)

            outputs = model(ids, mask, token_type_ids)
        
            _, predictions = torch.max(outputs.data, dim = 1)

            loss = criterion(outputs, targets)
            
            acc = calcuate_accuracy(predictions, targets)

            epoch_loss += loss.item()
    
            metric_acc.update(predictions, targets)
            metric_f1.update(outputs, targets)
            metric_f1_micro.update(outputs, targets)
            metric_f1_macro.update(outputs, targets)
            metric_precision.update(predictions, targets)
            metric_recall.update(predictions, targets)

            no_of_iterations += 1
            no_of_examples += targets.size(0)
    
    epoch_loss = epoch_loss/no_of_iterations
    epoch_acc = (acc*100)/no_of_iterations #no_of_examples

    # clear ml
    # Logger.current_logger().report_scalar(
    #     "test", "loss", iteration=epoch, value=epoch_loss)
    # Logger.current_logger().report_scalar(
    #     "test", "accuracy", iteration=epoch, value=epoch_acc)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     epoch_loss, acc, len(iterator),
    #     100. * acc / len(iterator)))

    acc_torch = metric_acc.compute()
    print(f"Validation Accuracy: {acc_torch}")
    
    f1 = metric_f1.compute()
    print(f"Validation F1 Validation: {f1}")
    
    f1_micro = metric_f1_micro.compute()
    print(f"Validation F1 Micro: {f1_micro}")

    f1_macro = metric_f1_macro.compute()
    print(f"Validation F1 Macro: {f1_macro}")
 
    precision = metric_precision.compute()
    print(f"Validation Precision: {precision}")

    recall = metric_recall.compute()
    print(f"Validation Recall: {recall}")
    
    print(f"Validation Loss Epoch: {epoch_loss}")

    metric_acc.reset()
    metric_f1.reset()
    metric_f1_micro.reset()
    metric_f1_macro.reset()
    metric_precision.reset()
    metric_recall.reset()
             
    return epoch_loss, epoch_acc


# experiment loop
for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, loss_function)
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_function)
        
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


test_loss, test_acc = evaluate(model, test_iterator, loss_function)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

