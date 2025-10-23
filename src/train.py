
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import pickle as p
import mcts
import helper as h
import gnn
import os
import re

def gen_train_all ( n ):
    # Generate datasets for all graphs with n vertices
    graphs = h.all_graphs ( n )
    for A in graphs:
        mcts.run_episode ( A )

def train_all():
    max_vertices = 3
    
    min_file = int ( max ( os.listdir ( "../training_data" ), key = lambda f: int ( re.search ( r'\d+', f ).group() ) ).split('.')[0][1:] )
    for i in range ( 2, max_vertices + 1 ):
        gen_train_all ( i )
        
        print ( "Generated for all graphs with {} vertices".format ( i ))
        
    i = min_file + 1
    
    path = "../training_data/x{}.pkl".format ( str ( i ) )
    
    dataset = []
    
    while ( os.path.exists ( path ) ):
        with open ( path, 'rb' ) as f:
            ( ( X, A ), vertex, n_sa ) = p.load ( f )
            
            X = torch.tensor ( X, dtype = torch.float )
            A = torch.tensor ( A, dtype = torch.int )
            A, _ = dense_to_sparse ( A )
            n_sa = torch.tensor ( [ n_sa ], dtype = torch.float )
            n_sa = ( n_sa / ( n_sa.sum() + 1e-8 ) )
            
            data = Data ( x = X, edge_index = A, y = n_sa )
            data.vertex = torch.tensor ( [ vertex ] )
            dataset.append ( data )
            
        i += 1
        path = "../training_data/x{}.pkl".format ( str ( i ) )
        
    batch_size = 16
    train_data = DataLoader ( dataset, batch_size = batch_size, shuffle = True )
    
    model, optimizer = gnn.return_GNN()
    
    # Training loop
    
    epoch_num = 100
    total_loss = 0
    
    model.train()
    for epoch in range ( epoch_num ):   
        for batch in train_data:
            optimizer.zero_grad()
            pred = model.forward_pass ( batch.x, batch.edge_index )[batch.vertex]

            loss = F.cross_entropy ( pred, batch.y )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
            print ( "Epoch:", epoch, "Loss:", loss.item() )

    torch.save ( model.state_dict(), "../data/gnn_weights10.pth" )

train_all()