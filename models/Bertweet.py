
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

BERTWEET_MODEL = "vinai/bertweet-base"

class BertweetClass(torch.nn.Module):
    """
    The class that represents the original Bertweet model 
    """
    def __init__(self,
                dropout,
                output_dim):

        super(BertweetClass, self).__init__()

        self.dropout = dropout
        self.output_dim = output_dim

        self.config = AutoConfig.from_pretrained(BERTWEET_MODEL)
        self.config.output_hidden_states = True
        self.bertweet = AutoModel.from_pretrained(BERTWEET_MODEL, config = self.config)
        self.hidden_size = self.bertweet.config.hidden_size

        self.pre_classifier = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bertweet_output = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = bertweet_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class BertweetLSTM(torch.nn.Module):
    """
    The class that represents the original Bertweet model 
    combined with the LSTM layers
    """
    def __init__(self,
                num_layers,
                bidirectional,
                dropout,
                output_dim):

        super(BertweetLSTM, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.output_dim = output_dim

        self.config = AutoConfig.from_pretrained(BERTWEET_MODEL)
        self.config.output_hidden_states = True
        self.bertweet = AutoModel.from_pretrained(BERTWEET_MODEL, config = self.config)
        self.hidden_size = self.bertweet.config.hidden_size

        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.hidden_size if bidirectional else self.hidden_size//2, output_dim)
		

    def forward(self, input_ids, attention_mask, token_type_ids):
        bertweet_embedded = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        _, (last_hidden, _) = self.LSTM(bertweet_embedded)

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=-1)
        output_hidden = self.dropout(output_hidden,0.2)
        
        # output_hidden = F.dropout(embedded, 0.2)
        output = self.out(output_hidden)

        #output = [batch size, out dim]
        return output



class BertweetRCNN(torch.nn.Module):
    """
    The class that represents the original Bertweet model 
    combined with the RCNN layers
    """
    def __init__(self, output_dim, dropout):
        super(BertweetRCNN, self).__init__()
        
        config = AutoConfig.from_pretrained(BERTWEET_MODEL)
        config.output_hidden_states = True
        self.bertweet = AutoModel.from_pretrained(BERTWEET_MODEL, config)
        self.hidden_size = self.bertweet.config.hidden_size
        
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size//2, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc = torch.nn.Linear(self.hidden_size, output_dim)

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