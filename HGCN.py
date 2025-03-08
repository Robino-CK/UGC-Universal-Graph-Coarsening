import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Define a Heterogeneous Graph Neural Network
class HGCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, rel_names):
        super().__init__()
        
        # Create a separate GraphConv for each relation
        self.conv1 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(in_feats[rel[0]], h_feats)
            for rel in rel_names
        }, aggregate='sum')
        
        # Second layer - input is now h_feats for all node types after first conv
        self.conv2 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(h_feats, out_feats)
            for rel in rel_names
        }, aggregate='sum')
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h

class ImprovedRGCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, rel_names, dropout=0.2):
        super().__init__()
        
        # Feature transformation layers for each node type
        self.node_transforms = nn.ModuleDict({
            ntype: nn.Linear(in_dim, h_feats)
            for ntype, in_dim in in_feats.items()
        })
        
        # First RGCN layer
        self.conv1 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(h_feats, h_feats, norm='both', weight=True, bias=True)
            for rel in rel_names
        }, aggregate='mean')
        
        # Second RGCN layer
        self.conv2 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(h_feats, h_feats, norm='both', weight=True, bias=True)
            for rel in rel_names
        }, aggregate='mean')
        
        # Output layer for each node type
        self.outputs = nn.ModuleDict({
            ntype: nn.Linear(h_feats, out_feats)
            for ntype in in_feats.keys()
        })
        
        self.dropout = nn.Dropout(dropout)
        self.h_feats = h_feats
    
    def forward(self, g, inputs):
        # Transform node features to common dimension
        h = {ntype: F.relu(transform(features))
             for ntype, features in inputs.items()
             for transform in [self.node_transforms[ntype]]}
        
        # Store original transformed features for residual
        h0 = {k: v for k, v in h.items()}
        
        # First conv layer
        h = self.conv1(g, h)
        h = {k: F.relu(self.dropout(v)) for k, v in h.items()}
        
        # Second conv layer with residual
        h = self.conv2(g, h)
        h = {k: F.relu(v + h0[k]) for k, v in h.items()}  # Residual connection
        
        # Output projection
        return {k: self.outputs[k](v) for k, v in h.items()}