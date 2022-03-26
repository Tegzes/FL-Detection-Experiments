import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import RobertaModel, BertModel


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

        self.roberta_layers = RobertaModel.from_pretrained("roberta-base")
        self.hidden_size = self.roberta_layers.config.hidden_size

        # self.pre_classifier = torch.nn.Linear(768, 768)
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.dropout = torch.nn.Dropout(0.25)
        self.classifier = torch.nn.Linear(2 * self.hidden_size if bidirectional else self.hidden_size, 2)
		

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.roberta_layers(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        # pooler = hidden_state[:, 0]


        encoded_layers = hidden_state.permute(1, 0, 2)
        _, (last_hidden, _) = self.lstm(pack_padded_sequence(encoded_layers, token_type_ids))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)

        # pooler = torch.nn.ReLU()(pooler)

        # pooler = self.dropout(pooler)
        
        output = self.classifier(output_hidden)
        # output = torch.squeeze(output, 1)
        # output = F.softmax(output, dim = 1)
        return output