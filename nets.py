import torch
from torch import nn
import torch.nn.functional as func

class Net1(nn.Module):
    def __init__(self, filter_sizes, filter_amount, dropout, classes):
        super(Net1, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=filter_amount, kernel_size=f) for f in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * filter_amount, classes)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, batch):
        reshaped = torch.reshape(batch, (len(batch), 1, -1))
        conv = [func.relu(con(reshaped)) for con in self.convs]
        pool = [func.max_pool1d(c, c.shape[2]).squeeze(2) for c in conv]
        concat = self.drop(torch.cat(pool, dim=1))
        return self.fc(concat)

class Net2(nn.Module): #53%
    def __init__(self, dropout, classes):
        super(Net2, self).__init__()
        
        content_dim = 256

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=750),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, classes)
        #self.drop = nn.Dropout(dropout)
    
    def forward(self, batch):
        out = self.conv(torch.reshape(batch, (len(batch), 1, 768)))
        #out = self.drop(out)
        #logits = self.fc(out.squeeze(2))
        out = self.fc1(out.squeeze(2))
        logits = self.fc2(out)
        return logits