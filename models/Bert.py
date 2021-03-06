import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel, BertConfig, BertForSequenceClassification


class BertClass(nn.Module):

    def __init__(self, dropout=0.25):

        super(BertClass, self).__init__()

        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('bert-base-uncased', config)
        

        self.linear1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(input_ids.shape)
        hidden_state = output[1]
        # pooler = hidden_state[:, 0]
        pooler = self.linear1(hidden_state)
        # pooler = torch.nn.ReLU()(pooler)

        dropout_output = self.dropout(pooler)
        linear_output = self.linear(dropout_output)

        return linear_output


class BertLSTM(nn.Module):
    """
    BERT + LSTM
    """
    def __init__(self,
                 bert,
                 bidirectional,
                 output_dim):
        super(BertLSTM, self).__init__()

        self.bert = bert
        # embedding_dim = bert.config.to_dict()['hidden_size']
        self.LSTM = nn.LSTM(768, 384, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(768, output_dim)

    def forward(self, text, mask):
        #text = [batch size, sent len]
        embedded = self.bert(text, mask)[0]

        _, (last_hidden, _) = self.LSTM(embedded)

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=-1)
        output_hidden = F.dropout(output_hidden,0.2)
        
        # output_hidden = F.dropout(embedded, 0.2)
        output = self.out(output_hidden)

        #output = [batch size, out dim]
        return output


class BertRCNN(torch.nn.Module):
    """
    Bert Recurrent CNN
    """
    def __init__(self, output_dim, dropout):
        super(BertRCNN, self).__init__()
        
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('bert-base-uncased', config)
        self.hidden_size = self.bert.config.hidden_size
        
        self.lstm = torch.nn.LSTM(self.hidden_size, 384, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = torch.nn.Linear(self.hidden_size + 2*384, 768) # basically the hidden state * 2
        self.tanh = torch.nn.Tanh()
        self.fc = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        output_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # output_embedded = batch size, seq_len, embedding_dim
        output_lstm, _ = self.lstm(output_bert)
        # output_lstm = batch size, seq_len, 2*lstm_hidden_size
        output = torch.cat([output_lstm, output_bert], 2)
        # output = batch size, seq_len, embedding_dim + 2*hidden_size
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = batch size, seq_len, hidden_size_linear -> batch size, hidden_size_linear, seq_len
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = batch size, hidden_size_linear
        output = self.fc(output)
        # output = batch size, output_dim
        return output