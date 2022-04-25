import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel


# Bert + LSTM
class BertLSTM(torch.nn.Module):
    def __init__(self,
                 bert,
                 output_dim):
        super(BertLSTM, self).__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.LSTM = torch.nn.LSTM(768, 384, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(768, output_dim)

    def forward(self, text, mask):
        #text = [batch size, sent len]
        embedded = self.bert(text, mask)[0]

        lstm_output, (last_hidden, _) = self.LSTM(embedded)

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=-1)
        output_hidden = F.dropout(output_hidden,0.2)
        
        # output_hidden = F.dropout(embedded, 0.2)
        output = self.out(output_hidden)

        #output = [batch size, out dim]
        return output