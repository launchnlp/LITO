import torch
import torch.nn as nn

class AnswerSelectionUni(nn.Module):
    # unidirectional lstm 
    def __init__(self, input_size, hidden_size):
        super(AnswerSelectionUni, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, self.hidden_size , num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_size , 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, qa_hss):
        rnn_out , _ = self.rnn(qa_hss)
        logits = self.fc2(rnn_out)
        logits = self.dropout(logits)
        logits = self.sigmoid(logits)
        return logits
    
class AnswerSelectionBi(nn.Module):
    # bidirectional lstm 
    def __init__(self, input_size, hidden_size):
        super(AnswerSelectionBi, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(2*hidden_size, 1) # bidirectional lstm
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, qa_hss):
        rnn_out , _ = self.rnn(qa_hss)
        logits = self.fc2(rnn_out)
        logits = self.dropout(logits)
        logits = self.sigmoid(logits)
        return logits

class AnswerSelectionMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AnswerSelectionMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size , 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = x.view(x.shape[0], -1)
        logits = self.fc2(x)
        logits = self.dropout(logits)
        logits = self.sigmoid(logits)
        return logits