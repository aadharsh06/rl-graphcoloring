
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

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
    #X = torch.tensor ( X, dtype = torch.float )
    
    return X, edge_index

def gen_random_graph():
    # Returning random adjecency matrix
    
    n = np.random.randint ( 2, 11 )
    
    A = np.triu ( np.random.randint ( 0, 2, size = ( n, n ) ), 1 )
    A = A + A.T

    return A
