
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import pickle as p
import mcts
import helper as h
import gnn
import numpy as np
import os
import re
from time import time

def gen_train_all ( n ):
    # Generate datasets for all graphs with n vertices
    graphs = h.all_graphs ( n )
    for A in graphs:
        mcts.run_episode ( A )

def gen_random_graph ( max_vertices ):
    # Returning random adjecency matrix
    n = np.random.randint ( 2, max_vertices )
    
    A = np.triu ( np.random.randint ( 0, 2, size = ( n, n ) ), 1 )
    A = A + A.T

    return A

def gen_validation_set ( max_vertices ):
    # This function is only intended to be run once
    path = "../data/validation_set"
    if os.listdir ( path ):
        print ( "Validation set already generated" )
        return 0
    
    validation_size = 20
    
    for i in range ( validation_size ):
        A = gen_random_graph ( max_vertices )
        mcts.run_episode ( A, path )
        
def calc_val ( model, val_data ):
    total_loss = 0
    for batch in val_data:
        pred = model.forward_pass ( batch.x, batch.edge_index )[batch.vertex]
        
        pred = F.log_softmax ( pred, dim = -1 )
        loss = F.kl_div ( pred, batch.y, reduction = 'batchmean' )
        
        total_loss += loss.item()
    return total_loss / len ( val_data ) 

def gen_and_train ( model, optimizer, max_vertices, size ):
    
    org_path = "../training_data"
    
    try:
        min_file = int ( max ( os.listdir ( org_path ), key = lambda f: int ( re.search ( r'\d+', f ).group() ) ).split('.')[0][1:] )
    except:
        min_file = 0
        
    for i in range ( size ):
        A = gen_random_graph ( max_vertices )
        mcts.run_episode ( A, model, org_path )
        
    i = min_file + 1
    
    path = org_path + "/x{}.pkl".format ( str ( i ) )
    
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
        path = org_path + "/x{}.pkl".format ( str ( i ) )
        
    batch_size = 16

    train_data = DataLoader ( dataset, batch_size = batch_size, shuffle = True )
    
    # Training loop
    
    epoch_num = 100
    avg_loss = 0
    
    model.train()
    for epoch in range ( epoch_num ):   
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            pred = model.forward_pass ( batch.x, batch.edge_index )[batch.vertex]

            pred = F.log_softmax ( pred, dim = -1 )
            loss = F.kl_div ( pred, batch.y, reduction = 'batchmean' )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss += ( total_loss / len ( train_data ) )
        
    torch.save ( model.state_dict(), "../data/gnn_weights10.pth" )
    
    return avg_loss / epoch_num

def auto_train ( num_episodes, max_vertices, size ):
    
    print ( "Started" )
    
    model, optimizer = gnn.return_GNN()
    
    org_path = "../data/validation_set"
    i = 1
    
    path = org_path + "/x{}.pkl".format ( str ( i ) )
    
    val_dataset = []
    
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
            val_dataset.append ( data )
            
        i += 1
        path = org_path + "/x{}.pkl".format ( str ( i ) )
        
    batch_size = 16
    val_data = DataLoader ( val_dataset, batch_size = batch_size, shuffle = False )
    
    # Basic run
    for i in range ( num_episodes ):
        start = time()
        train_avg_loss = gen_and_train ( model, optimizer, max_vertices, size )
        val_avg_loss = calc_val ( model, val_data )
        end = time()
        print ( "Episode number: {}; Train avg loss: {:.6f}; Validation avg loss: {:.6f}; Time taken: {:.1f}s".format ( i+1, train_avg_loss, val_avg_loss, end - start ) )
        
    print ( "Ended" )

auto_train ( 100, 10, 100 )