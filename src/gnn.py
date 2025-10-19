
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

max_colors = 10

class GNN_Model ( torch.nn.Module ):
    def __init__ ( self, max_colors ):
        super().__init__()
        
        k = max_colors + 1
        self.layer1 = GCNConv ( k, k )
        self.layer2 = GCNConv ( k, k )
        self.layer3 = GCNConv ( k, max_colors )
        
    def forward_pass ( self, X, A ):
        X = self.layer1 ( X, A )
        X = F.gelu ( X )
        X = self.layer2 ( X, A )
        X = F.gelu ( X )
        X = self.layer3 ( X, A )
        return F.log_softmax ( X, dim = 1 )

model = GNN_Model ( max_colors )
#model.load_state_dict ( torch.load ( "../data/gnn_model_weights.pth" ) )

optimizer = torch.optim.Adam ( model.parameters(), lr = 0.01, weight_decay = 5e-4 )

def forward_pass ( X, A ):
    return model.forward_pass ( X, A )



        