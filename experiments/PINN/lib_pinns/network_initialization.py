import torch.nn as nn
import torch
def xavier_init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, 1)
        m.bias.data.fill_(0.001)