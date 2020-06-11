import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Network_pytorch(nn.Module):

    def __init__(self, input_dim, hideen_dim, output_dim):
        super (Network_pytorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hideen_dim)
        self.fc2 = nn.Linear(hideen_dim, output_dim)
        
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return (out)
        
    
