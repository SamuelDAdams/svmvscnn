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

class Net2(nn.Module): #87% with k = [2, 4, 8, 16, 32, 64], k_amount = 64
    def __init__(self, dropout, channels, kernels, classes):
        super(Net2, self).__init__()
        
        self.conv1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        #self.norm = nn.BatchNorm1d()

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=750),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(dropout),
        # )
        self.lin = nn.Sequential(
            nn.Linear(channels * len(kernels), 32),
            nn.Linear(32,classes),
        )
    
    def forward(self, batch):
        x = torch.reshape(batch, (len(batch), 1, 768))
        x = [func.relu(conv(x)) for conv in self.conv1]
        x = [func.max_pool1d(l, l.size(2)) for l in x]
        x = torch.cat(x, dim=1).squeeze()
        x = self.dropout(x)
        #out = self.drop(out)
        #logits = self.fc(out.squeeze(2))
        logits = self.lin(x)
        return self.sigmoid(logits)