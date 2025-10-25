
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from itertools import product

# Test A
A = np.array ( [
    [ 0, 1, 0 ],
    [ 1, 0, 1 ],
    [ 0, 1, 0 ]
] )

def init_features ( A, max_colors ):
    degree = np.sum ( A, axis = 0 ).reshape ( -1, 1 )
    deg_max = np.max ( degree )
    degree = degree / ( deg_max + 1 )
    
    cur_colored = np.zeros ( ( degree.size, max_colors ) )
        
    A = torch.tensor ( A, dtype = torch.float )
    edge_index, edge_attr = dense_to_sparse ( A )
    
    X = np.hstack ( [ cur_colored, degree ] )
    
    return X, edge_index

def all_graphs ( n ):
    num_edges = n * (n - 1) // 2
    graphs = []
    
    for bits in product ( [0, 1], repeat = num_edges ):
        A = np.zeros ( ( n, n ), dtype = int )
        upper_indices = np.triu_indices ( n, 1 )
        A[upper_indices] = bits
        A = A + A.T
        graphs.append ( A )
    
    return graphs