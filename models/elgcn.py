import torch
import torch.nn as nn
import torch.nn.functional as F


class ELGCN(nn.Module):
    def __init__(self):
        super(ELGCN, super).__init__()