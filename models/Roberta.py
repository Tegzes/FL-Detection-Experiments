import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import RobertaModel, RobertaConfig


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
                num_layers,
                bidirectional,
                output_dim):

        super(RobertaLSTMSarc, self).__init__()

        # self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim

        self.config = RobertaConfig.from_pretrained("roberta-base")
        self.config.output_hidden_states = True
        self.roberta_layers = RobertaModel.from_pretrained("roberta-base", config = self.config)
        self.hidden_size = self.roberta_layers.config.hidden_size

        self.lstm = torch.nn.LSTM(self.hidden_size, 384, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(0.25)
        # self.classifier = torch.nn.Linear(self.hidden_size, output_dim)
        self.classifier = torch.nn.Linear(self.hidden_size if bidirectional else self.hidden_size//2, output_dim)
		

    def forward(self, input_ids, attention_mask, token_type_ids):
        # roberta part
        embedded = self.roberta_layers(input_ids, attention_mask, token_type_ids)[0]

        # lstm part
        _, (last_hidden, _) = self.lstm(embedded)

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=-1)
        output_hidden = self.dropout(output_hidden)
        output = self.classifier(output_hidden)

        return output

