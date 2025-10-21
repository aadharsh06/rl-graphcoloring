
# 0 - uncolored, 1 to max_colors are the colors
# Vertices are randomly permuted and their index is their identifier ( 0 to n - 1 )

import numpy as np
import gnn
import helper as h
import torch

# Test A
A = np.array ( [
    [ 0, 1, 1 ],
    [ 1, 0, 0 ],
    [ 1, 0, 0 ]
] )

max_colors = 10

class state_graph():
    def __init__ ( self, A ):
        self.graph = A
        self.colors = np.zeros ( A.shape[0] )

class Node():
    def __init__ ( self, vertex, color, max_colors ):
        
        self.max_colors = max_colors
        self.vertex = vertex
        self.color = color
        
        self.n_sa = np.zeros ( max_colors )
        self.w_sa = np.zeros ( max_colors )
        self.q_sa = np.zeros ( max_colors )
        
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
        print ( "Color: ", self.color )
        
        print ( "N(s,a): ", self.n_sa )
        print ( "W(s,a): ", self.w_sa )
        print ( "Q(s,a): ", self.q_sa )
        
        #print ( "Children nodes: ", self.children )
        print ( "Is leaf? : ", self.is_leaf )
        print ( "------------------------------------")
        
def reward ( H, root, cur_graph ):
    n = cur_graph.graph.shape[0]
    colors = np.zeros ( n )

    colors[root.vertex] = root.color

    for node, action in H:
        next_vertex = node.vertex + 1
        if next_vertex < n:
            colors[next_vertex] = action + 1  
            
    conflicts = 0
    for i in range ( n ):
        for j in range ( i + 1, n ):
            if cur_graph.graph[i][j] == 1 and colors[i] == colors[j] and colors[i] != 0:
                conflicts += 1

    used_colors = len ( set ( colors ) - {0} )

    R = ( -100 * conflicts ) + ( -7 * used_colors )
    
    return R

def u_sa_calc ( H, cur_graph, cur_node ):
    c_puct = 2
    
    X, edge_index = h.init_features ( cur_graph.graph, max_colors )
    
    for node, action in H:
        X[node.vertex][node.color - 1] = 1 
    
    X = torch.tensor ( X, dtype = torch.float )
    p_sa = gnn.forward_pass ( X, edge_index )
    p_sa = (p_sa.detach().numpy())[cur_node.vertex]
    
    u_sa = c_puct * p_sa * np.sqrt ( cur_node.n_sa.sum() / ( 1 + cur_node.n_sa ) )
    return u_sa

def run_simulation ( cur_graph, root ):
    root.expand ( cur_graph )
    
    cur_node = root
    H = []
    while ( not cur_node.is_leaf ): 
        
        q_sa = cur_node.q_sa
        u_sa = u_sa_calc ( H, cur_graph, cur_node )
        a_star = np.argmax ( (q_sa + u_sa) )

        H.append ( ( cur_node, a_star ) )
        cur_node = cur_node.children[a_star]
    cur_node.expand ( cur_graph )
    
    R = reward ( H, root, cur_graph )
    
    for i in range ( len ( H ) ):
        H[i][0].n_sa[H[i][1]] += 1
        H[i][0].w_sa[H[i][1]] += R
        H[i][0].q_sa[H[i][1]] = ( H[i][0].w_sa[H[i][1]] ) / ( H[i][0].n_sa[H[i][1]] )
    
def print_tree ( node ):
    if node.vertex > 1:
        return 1
    
    node.print_details()
    
    for child in node.children:
        print_tree ( child )

def run_episode ( A ):
    cur_graph = state_graph ( A )
    root = Node ( 0, 1, max_colors )
    
    cur_graph.colors[0] = 1
    
    for vertex in range ( len ( cur_graph.colors ) - 1 ):
        for i in range ( 150 ):
            run_simulation ( cur_graph, root )
        
        assign_color = np.argmax ( root.n_sa ) + 1
        cur_graph.colors[vertex+1] = assign_color
        root = root.children[assign_color - 1]
        
    print ( cur_graph.colors )

run_episode ( A )