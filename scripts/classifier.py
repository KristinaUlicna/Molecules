import torch

from torch.nn import Linear
from torch.nn import functional as F 

from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(
        self, 
        num_features: int, 
        embedding_size: int, 
        num_attn_heads: int, 
        dropout_prob: float, 
        use_batch_norm: bool
    ) -> None:
        """TODO: This could be more flexible, with `hidden_channels` list and torch.nn.ModuleList"""
        # Init parent
        super(GAT, self).__init__()
        self.dropout_prob = dropout_prob
        
        # 3-layered GAT
        self.gat1 = GATv2Conv(num_features, embedding_size, heads=num_attn_heads)
        self.gat2 = GATv2Conv(embedding_size*num_attn_heads, embedding_size, heads=num_attn_heads//2)
        self.gat3 = GATv2Conv(embedding_size*num_attn_heads//2, embedding_size, heads=1)

        # 3-layered batch norm
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.gat1_bn = BatchNorm(embedding_size*num_attn_heads)
            self.gat2_bn = BatchNorm(embedding_size*num_attn_heads//2)
            self.gat3_bn = BatchNorm(embedding_size)
            
        # Output layer
        self.out = Linear(embedding_size*2, 2)
        

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch_index: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward call."""
        # Conv layers
        hidden = F.dropout(x, p=self.dropout_prob, training=self.training)
        hidden = self.gat1(hidden, edge_index)
        if self.use_batch_norm:
            hidden = self.gat1_bn(hidden)
        hidden = F.leaky_relu(hidden)
        
        hidden = F.dropout(hidden, p=self.dropout_prob, training=self.training)
        hidden = self.gat2(hidden, edge_index)
        if self.use_batch_norm:
            hidden = self.gat2_bn(hidden)
        hidden = F.leaky_relu(hidden)
          
        hidden = F.dropout(hidden, p=self.dropout_prob, training=self.training)
        hidden = self.gat3(hidden, edge_index)
        if self.use_batch_norm:
            hidden = self.gat3_bn(hidden)
        hidden = F.leaky_relu(hidden)
        
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([global_max_pool(hidden, batch_index), 
                            global_mean_pool(hidden, batch_index)], dim=1)
        
        # Apply a final linear for regression
        out = self.out(hidden)
        
        return out, hidden

