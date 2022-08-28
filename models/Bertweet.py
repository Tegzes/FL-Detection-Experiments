
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class BertweetClass(nn.Module):
    """
    BERTWEET
    """
    def __init__(self):
        super(BertweetClass, self).__init__()
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class BertweetLSTM(nn.Module):
    """
    BERTWEET + LSTM
    """
    def __init__(self):
        super(BertweetLSTM, self).__init__()

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.LSTM = nn.LSTM(768, 384, batch_first=True, bidirectional=True)
        self.out = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        _, (last_hidden, _) = self.LSTM(embedded)

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=-1)
        output_hidden = F.dropout(output_hidden,0.2)
        
        # output_hidden = F.dropout(embedded, 0.2)
        output = self.out(output_hidden)

        #output = [batch size, out dim]
        return output



class BertweetRCNN(torch.nn.Module):
    """
    Bertweet RCNN
    """
    def __init__(self, output_dim, dropout):
        super(BertweetRCNN, self).__init__()
        
        config = AutoConfig.from_pretrained("vinai/bertweet-base")
        config.output_hidden_states = True
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base", config)
        self.hidden_size = self.bertweet.config.hidden_size
        
        self.lstm = torch.nn.LSTM(self.hidden_size, 384, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = torch.nn.Linear(self.hidden_size + 2*384, 768) # basically the hidden state * 2
        self.tanh = torch.nn.Tanh()
        self.fc = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        output_bertweet = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # output_embedded = batch size, seq_len, embedding_dim
        output_lstm, _ = self.lstm(output_bertweet)
        # output_lstm = batch size, seq_len, 2*lstm_hidden_size
        output = torch.cat([output_lstm, output_bertweet], 2)
        # output = batch size, seq_len, embedding_dim + 2*hidden_size
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = batch size, seq_len, hidden_size_linear -> batch size, hidden_size_linear, seq_len
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = batch size, hidden_size_linear
        output = self.fc(output)
        # output = batch size, output_dim
        return output