import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv


class GCNencoder(nn.Module):
    """Encoder Network"""
    def __init__(self, nfeat, nhid):
        super(GCNencoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid//2)
        # self.gc3 = GCNConv(nhid/2, nhid/4)
        # self.weights = Parameter(torch.randn(batch_size))

    def forward(self, x, edge_idx):
        x = F.leaky_relu(self.gc1(x, edge_idx))
        x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.gc2(x, edge_idx))
        # x = self.gc3(x, edge_idx, self.weights)
        return x


class GCNdecoder(nn.Module):
    """Decoder Network"""
    def __init__(self, nfeat, nhid):
        super(GCNdecoder, self).__init__()
        # self.gc1 = GCNConv(nhid/4, nhid/2)
        self.gc1 = GCNConv(nhid//2, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        # self.weights = Parameter(torch.randn(batch_size))

    def forward(self, x, edge_idx):
        x = F.leaky_relu(self.gc1(x, edge_idx))
        x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.gc2(x, edge_idx))
        # x = self.gc3(x, edge_idx, self.weights)
        return x


class ResGCNencoder(nn.Module):
    """Encoder Network"""
    def __init__(self, nfeat, nhid):
        super(ResGCNencoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid//2)
        self.downsample = nn.Linear(nfeat, nhid//2)  # transform the dimension of the input

    def forward(self, x, edge_idx):
        identity = self.downsample(x)  # transform the input
        out = F.leaky_relu(self.gc1(x, edge_idx))
        out = F.dropout(out, p=0.5)
        out = F.leaky_relu(self.gc2(out, edge_idx))
        out += identity  # add the transformed input to the output
        return out


class ResGCNdecoder(nn.Module):
    """Decoder Network"""

    def __init__(self, nfeat, nhid):
        super(ResGCNdecoder, self).__init__()
        self.gc1 = GCNConv(nhid//2, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.downsample = nn.Linear(nhid//2, nfeat)  # transform the dimension of the input

    def forward(self, x, edge_idx):
        identity = self.downsample(x)  # transform the input
        out = F.leaky_relu(self.gc1(x, edge_idx))
        out = F.dropout(out, p=0.5)
        out = F.leaky_relu(self.gc2(out, edge_idx))
        out += identity  # add the transformed input to the output
        return out


class Hospital(nn.Module):
    def __init__(self, encoder, decoders, views):
        super(Hospital, self).__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict({str(view): decoders[view - 1] for view in views})

    def forward(self, x, edge_idx):
        encoded = self.encoder(x, edge_idx)
        outputs = {view: decoder(encoded, edge_idx) for view, decoder in self.decoders.items()}
        return outputs


