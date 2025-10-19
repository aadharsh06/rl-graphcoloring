
# 0 - uncolored, 1 to max_colors are the colors
# Vertices are randomly permuted and their index is their identifier ( 0 to n - 1 )

import numpy as np
import gnn
import helper as h
import torch

# Test A
A = np.array ( [
    [ 0, 1, 0 ],
    [ 1, 0, 1 ],
    [ 0, 1, 0 ]
] )

max_colors = 10

class state_graph():
    def __init__ ( self, A ):
        np.random.shuffle ( A )
        self.graph = A
        self.colors = np.zeros ( ( 1, A.shape[0] ) )

class Node():
    def __init__ ( self, vertex, color, max_colors ):
        
        self.max_colors = max_colors
        self.vertex = vertex
        self.color = color
        
        self.n_sa = np.zeros ( ( 1, max_colors ) )
        self.w_sa = np.zeros ( ( 1, max_colors ) )
        self.q_sa = np.zeros ( ( 1, max_colors ) )
        
        self.children = []
        self.is_leaf = True
    
    def expand ( self, cur_graph ):
        if not self.is_leaf:
            return 0
        if self.vertex == len ( cur_graph.graph ):
            return 1
        
        for i in range ( 1, self.max_colors + 1 ):
            self.children.append ( Node ( self.vertex + 1, i, self.max_colors ) )
        self.is_leaf = False
    
    def print_details ( self ):
        print ( "\n------------------------------------")
        print ( "Vertex: ", self.vertex )
        print ( "Colors: ", self.color )
        
        print ( "N(s,a): ", self.n_sa )
        print ( "W(s,a): ", self.w_sa )
        print ( "Q(s,a): ", self.q_sa )
        
        print ( "Children nodes: ", self.children )
        print ( "Is leaf? : ", self.is_leaf )
        print ( "------------------------------------")
        
def reward ( H ):
    unique = []
    for i in H:
        if i[0].color not in unique:
            unique.append ( i[0].color )
    
    return - len ( unique )

def u_sa_calc ( H, cur_graph, cur_node ):
    c_puct = 2
    
    X, edge_index = h.init_features ( cur_graph.graph, max_colors )
    
    for i in range ( len ( H ) ):
        X[i][H[i][0].color-1] = 1
    
    X = torch.tensor ( X, dtype = torch.float )
    p_sa = gnn.forward_pass ( X, edge_index )
    p_sa = p_sa.detach().numpy()
    
    u_sa = c_puct * p_sa * np.sqrt ( cur_node.n_sa.sum() / ( 1 + cur_node.n_sa ) )
    return u_sa

def run_simulation ( cur_graph, root ):
    root.expand ( cur_graph )
    
    cur_node = root
    H = []
    while ( not cur_node.is_leaf ):
        q_sa = cur_node.q_sa
        u_sa = u_sa_calc ( H, cur_graph, cur_node )
        a_star = np.argmax ( (q_sa + u_sa)[cur_node.vertex] )
        
        H.append ( ( cur_node, a_star ) )
        cur_node = cur_node.children[a_star]
    cur_node.expand ( cur_graph )
    
    R = reward ( H )
    
    for i in range ( len ( H ) ):
        H[i][0].n_sa[0][H[i][1]] += 1
        H[i][0].w_sa[0][H[i][1]] += R
        H[i][0].q_sa[0][H[i][1]] = ( H[i][0].w_sa[0][H[i][1]] ) / ( H[i][0].n_sa[0][H[i][1]] )
        #print ( H[i][0].color, H[i][1], end = " | " )
    #print()
    
def run_episode ( A ):
    cur_graph = state_graph ( A )
    root = Node ( 0, 1, max_colors )
    
    for i in range ( 10 ):
        run_simulation ( cur_graph, root )
    root.print_details()

run_episode ( A )