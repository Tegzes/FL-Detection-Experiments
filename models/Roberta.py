import torch
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig

ROBERTA_MODEL = "roberta-base"

class RobertaSarc(torch.nn.Module):
    """
    The class that represents the original Roberta model 
    """
    def __init__(self,
                dropout,
                output_dim):

        super(RobertaSarc, self).__init__()

        self.dropout = dropout
        self.output_dim = output_dim

        self.config = RobertaConfig.from_pretrained(ROBERTA_MODEL)
        self.config.output_hidden_states = True
        self.roberta_layers = RobertaModel.from_pretrained(ROBERTA_MODEL, config = self.config)
        self.hidden_size = self.roberta_layers.config.hidden_size

        self.pre_classifier = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        roberta_output = self.roberta_layers(input_ids, attention_mask, token_type_ids)
        hidden_state = roberta_output[0] 
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class RobertaLSTMSarc(torch.nn.Module):
    """
    The class that represents the original Roberta model 
    combined with the LSTM layers
    """
    def __init__(self,
                num_layers,
                bidirectional,
                dropout,
                output_dim):

        super(RobertaLSTMSarc, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.output_dim = output_dim

        self.config = RobertaConfig.from_pretrained(ROBERTA_MODEL)
        self.config.output_hidden_states = True
        self.roberta_layers = RobertaModel.from_pretrained(ROBERTA_MODEL, config = self.config)
        self.hidden_size = self.roberta_layers.config.hidden_size

        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.hidden_size if bidirectional else self.hidden_size//2, output_dim)
		

    def forward(self, input_ids, attention_mask, token_type_ids):
        # roberta part
        roberta_embedded = self.roberta_layers(input_ids, attention_mask, token_type_ids)[0]

        # lstm part
        _, (last_hidden, _) = self.lstm(roberta_embedded)

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=-1)
        output_hidden = self.dropout(output_hidden)
        output = self.classifier(output_hidden)

        return output


class RobertaRCNN(torch.nn.Module):
    """
    The class that represents the original Roberta model 
    combined with the RCNN layers
    """
    def __init__(self, dropout, output_dim):
        super(RobertaRCNN, self).__init__()
        
        self.config = RobertaConfig.from_pretrained(ROBERTA_MODEL)
        self.config.output_hidden_states = True
        self.roberta_layers = RobertaModel.from_pretrained(ROBERTA_MODEL, config = self.config)
        self.hidden_size = self.roberta_layers.config.hidden_size
        
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size//2, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc = torch.nn.Linear(self.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        roberta_output = self.roberta_layers(input_ids, attention_mask, token_type_ids)[0]
        # output_embedded = batch size, seq_len, embedding_dim
        output_lstm, _ = self.lstm(roberta_output)
        # output_lstm = batch size, seq_len, 2*lstm_hidden_size
        output = torch.cat([output_lstm, roberta_output], 2)
        # output = batch size, seq_len, embedding_dim + 2*hidden_size
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = batch size, seq_len, hidden_size_linear -> batch size, hidden_size_linear, seq_len
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = batch size, hidden_size_linear
        output = self.fc(output)
        # output = batch size, output_dim
        return output