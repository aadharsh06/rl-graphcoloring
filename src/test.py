
import gnn
import torch
import helper
import numpy as np
from torch_geometric.utils import dense_to_sparse, to_dense_adj

A = np.array ( [
    [ 0, 1, 0 ],
    [ 1, 0, 1 ],
    [ 0, 1, 0 ]
] )

X, edge_index = helper.init_features ( A, 5 )

print ( gnn.forward_pass ( X, edge_index ) )