import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import RobertaModel, RobertaConfig, BertModel


class RobertaSarc(torch.nn.Module):
    def __init__(self):
        super(RobertaSarc, self).__init__()
        self.roberta_layers = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.roberta_layers(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        # output = F.softmax(output, dim = 1)
        return output



class RobertaLSTMSarc(torch.nn.Module):
    def __init__(self,
                # hidden_size,
                num_layers,
                bidirectional):

        super(RobertaLSTMSarc, self).__init__()

        # self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.config = RobertaConfig.from_pretrained("roberta-base")
        self.config.output_hidden_states = True
        self.roberta_layers = RobertaModel.from_pretrained("roberta-base", config = self.config)
        self.hidden_size = self.roberta_layers.config.hidden_size

        # self.pre_classifier = torch.nn.Linear(768, 768)
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers, batch_first=True, bidirectional=True)
        # self.dropout = torch.nn.Dropout(0.25)
        self.classifier = torch.nn.Linear(self.hidden_size, 2)
        # self.classifier = torch.nn.Linear(2 * self.hidden_size if bidirectional else self.hidden_size, 2)
		

    def forward(self, input_ids, attention_mask, token_type_ids, tweet_len):
        # roberta part
        output = self.roberta_layers(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # last layer(can be used for token classif), pooler output (can be used for seq classif), hidden states
        hidden_states = output[2]


        last_hidden_layers = torch.stack([hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]])
        last_hidden_layers = torch.mean(last_hidden_layers, 0)
        # last_hidden_layers = torch.tensor(last_hidden_layers, dtype = torch.long)
        last_hidden_layers = last_hidden_layers.permute(1, 0, 2)

        # print(f"hidden_states shape: {len(hidden_states)}") # 13
        # print(f"Hidden_states -1 shape: {len(hidden_states[-1])}") # 4
        # print(f"last_hidden_layers shape: {len(last_hidden_layers)}") # 256
        # print(f"Expected hidden size: {self.hidden_size}") # 768 

        # lstm part
        _, (last_hidden, _) = self.lstm(last_hidden_layers)
        # output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        print(f"Last Hidden from LSTM: {len(last_hidden[0])}")
        # print(f"Last Hidden from LSTM: {len(last_hidden)}")
        
        output = self.classifier(output[1])

        return output

# Bert + LSTM
class BertLSTM(torch.nn.Module):
    def __init__(self,pre_trained='bert-base-uncased'):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pre_trained)
        self.hidden_size = self.bert.config.hidden_size
        self.LSTM = torch.nn.LSTM(self.hidden_size,self.hidden_size,bidirectional=True)
        self.clf = torch.nn.Linear(2 * self.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids, tweet_len):
        # tweet_len = tweet_len.to('cpu')
        encoded_layers, pooled_output = self.bert(input_ids, attention_mask)
        encoded_layers = encoded_layers.permute(1, 0, 2)

        _, (last_hidden, _) = self.LSTM(pack_padded_sequence(encoded_layers, tweet_len, batch_first=True, enforce_sorted=True))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = F.dropout(output_hidden,0.2)
        output = self.clf(output_hidden)
        
        return output