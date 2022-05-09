
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

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