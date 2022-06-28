from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import scipy.sparse as sp
# from utils import *
import random
import torch.nn.functional as F
from nocd.utils import to_sparse_tensor

# from utils import eliminate_symbol

# adj = adjacency matrix in csr format
# k = augmentation rate = k hop nearest neighbors are added directly as neighbor

# def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
#                      cuda: bool = True,
#                      ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:
#     """Convert a scipy sparse matrix to a torch sparse tensor.

#     Args:
#         matrix: Sparse matrix to convert.
#         cuda: Whether to move the resulting tensor to GPU.

#     Returns:
#         sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

#     """
#     if sp.issparse(matrix):
#         coo = matrix.tocoo()
#         indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
#         values = torch.FloatTensor(coo.data)
#         shape = torch.Size(coo.shape)
#         sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
#     elif torch.is_tensor(matrix):
#         row, col = matrix.nonzero().t()
#         indices = torch.stack([row, col])
#         values = matrix[row, col]
#         shape = torch.Size(matrix.shape)
#         sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
#     else:
#         raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
#     if cuda:
#         sparse_tensor = sparse_tensor.cuda()
#     return sparse_tensor.coalesce()



def dilate_graph( g, d=2):
    gn = nx.Graph()

    for i in range(len(g.nodes)):
        nbr = [n for n in g.neighbors(i)]
        gn.add_edge(i,i)
        
        p=1/d 
        # indices = np.random.randint(0,len(nbr),size=len(nbr)//d)
        prob = np.random.rand(len(nbr))
        # print(nbr, prob)
        for j in range(len(nbr)):
            if(prob[j]<p):
                gn.add_edge(i,nbr[j])
        # print(nbr, indices)

        # for j in indices:
        #     gn.add_edge(i,nbr[j])

    # nx.draw_spring(gn,with_labels= True)
    csr  = nx.to_scipy_sparse_matrix(gn,range(len(g.nodes)))
    # csr = nx.to_scipy_sparse_matrix(gn)
    # print(csr.toarray())
    return csr



def augment_graph_neighbors_with_degree(adj, k=2): #no significance of this number k=2 here
    #Here we are just considering second hop neighbors only.
    #for each node augment the edges or neigbors 
    # to the k-th multiple of that  node 
    adj= adj.tolil()
    adj.setdiag(1)
    adj=adj.tocsr()
    adj=adj.toarray()
    # print('before augment: \n',adj)
    rows, cols = np.where(adj == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges) # drop both node and edge attributes 
    # nx.draw_spring(g, with_labels = True)
#     print(g.nodes)
#     print(len(list(g.neighbors(20))), g.degree[20])
    edges = []
    g.remove_edges_from(nx.selfloop_edges(g))
    g_old = g.copy()

#     print('g.edges: ',g.edges)
 
    # apsp = dict(nx.shortest_path_length(g))
    # print(apsp)
    from tqdm import tqdm
    import random
    for i in tqdm(range(len(g.nodes))):
        # print('i: ',i)
        old_neighbors = list(g_old.neighbors(i))
        neighbors = set(old_neighbors)


        for n in old_neighbors:
            n_neighbors = set( list(g_old.neighbors(n))) - neighbors
            if(len(n_neighbors)>0):
                new_one = random.choice(list(n_neighbors))
                neighbors.add(new_one)

        new_neighbors = neighbors - set(old_neighbors)
        for j in new_neighbors:
            g.add_edge(i,j)

          

            
    # g.add_edges_from(edges)
    g.remove_edges_from(nx.selfloop_edges(g))

    # nx.draw_spring(g, with_labels = True)
    
    csr= nx.to_scipy_sparse_matrix(g,range(len(g.nodes)))
    # print(csr)
    return csr, g



def get_sorted_neighbors(node, G, originalG, num_nbrs):
    
    nbrs = list(G.neighbors(node))
    weighted_nbrs = []
    
    all_nbrs = set(nbrs)
    
    # we are considering just two hop, because more than two  hop increase number of neigbors much.
    
    for n in nbrs : # consider first hop neighbors
        e = (str(node+1), str(n+1))
        w = int(originalG.edges[e]['weight'])
        
        weighted_nbrs.append((n, w))
        
        second_nbrs = set(G.neighbors(n)) - set(all_nbrs)  # new neighbors
        
        all_nbrs.update(set(second_nbrs))
        
        for sn in second_nbrs: # consider second hop neigbors
            e2 = (str(n+1), str(sn+1))
            w2= int(originalG.edges[e2]['weight'])
            
            w2 = int(w*0.75 + w2*0.25)  # adapted weight for second hop neighbor
        
            weighted_nbrs.append((sn, w2))
            
    sorted_nbrs = sorted(weighted_nbrs, key=lambda x: x[1], reverse= True)
    
    if( len(sorted_nbrs)>num_nbrs):
        sorted_nbrs = sorted_nbrs[:num_nbrs]
        
    unpacked_nbrs = [n for n,_ in sorted_nbrs]
        
    return unpacked_nbrs
        
        
    
    
        
    
    
def augment_graph_weighted_choice_all(adj, k=2, G = None ):
    import numpy as np
    import networkx as nx
    #for each node augment the edges or neigbors 
    # to the k-th multiple of that  node 
    eliminate_symbol(G)
    adj= adj.tolil()
    adj.setdiag(1)
    adj=adj.tocsr()
    adj=adj.toarray()
    # print('before augment: \n',adj)
    rows, cols = np.where(adj == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges) # drop both node and edge attributes 
#     nx.draw_spring(g, with_labels = True)
#     print(g.nodes)
#     print(len(list(g.neighbors(20))), g.degree[20])
    edges = []
    g.remove_edges_from(nx.selfloop_edges(g))
    g_old = g.copy()
#     nx.draw_spring(g_old,with_labels= True)

#     print(get_sorted_neighbors(0, g_old, G, 3, k=2))
#     print('g.edges: ',g.edges)
 
    # apsp = dict(nx.shortest_path_length(g))
    # print(apsp)
    from tqdm import tqdm
    import random
    for i in tqdm(range(len(g.nodes))):
        # print('i: ',i)
        
        new_neighbors = get_sorted_neighbors(i, g_old, G, g_old.degree[i]*k)
        
#         print('node: ', i)
#         print('degree: ', g_old.degree[i])
#         print('neighbors: ', new_neighbors)
        g.remove_node(i)
        g.add_node(i)
        for j in new_neighbors:
#             print('j: ',j)
            g.add_edge(i,j)

          

            
    # g.add_edges_from(edges)
    g.remove_edges_from(nx.selfloop_edges(g))

    # nx.draw_spring(g, with_labels = True)
    
    csr= nx.to_scipy_sparse_matrix(g,range(len(g.nodes)))
    # print(csr)
    return csr, g
def get_sorted_differential_neighbors(node, G, originalG, split = 0.80 ):
    
    nbrs = list(G.neighbors(node))
    
    if(len(nbrs)==0):
        return []
    
    weighted_first_hop_nbrs = []
    weighted_second_hop_nbrs = []
    
    first_hop_nbrs = set(nbrs)
    second_hop_nbrs = set()
    all_nbrs = set()
    num_nbrs = len(first_hop_nbrs)
    
    for n in nbrs : # consider first hop neighbors
        e = (str(node+1), str(n+1))
        w = int(originalG.edges[e]['weight'])
        
        weighted_first_hop_nbrs.append((n, w))
        
        second_nbrs = set(G.neighbors(n)) - set(all_nbrs)  # new neighbors
        
        all_nbrs.update(set(second_nbrs))
        
        for sn in second_nbrs: # consider second hop neigbors
            e2 = (str(n+1), str(sn+1))
            w2= int(originalG.edges[e2]['weight'])
            
#             w2 = int(w*0.75 + w2*0.25)  # adapted weight for second hop neighbor
        
            weighted_second_hop_nbrs.append((sn, w2))
        
        
    num_of_first_hop_nbrs = round(num_nbrs*(split))
    num_of_second_hop_nbrs = round(num_nbrs*(1.0 - split))
    sorted_first_hop_nbrs = sorted(weighted_first_hop_nbrs, key=lambda x: x[1], reverse= True)[:num_of_first_hop_nbrs]
    sorted_second_hop_nbrs = sorted(weighted_second_hop_nbrs, key=lambda x: x[1], reverse= True)[:num_of_second_hop_nbrs]
    
    sorted_nbrs   = sorted_first_hop_nbrs + sorted_second_hop_nbrs 
    
    if( len(sorted_nbrs)>num_nbrs):
        sorted_nbrs = sorted_nbrs[:num_nbrs]
        
    unpacked_nbrs = [n for n,_ in sorted_nbrs]
        
    return unpacked_nbrs
    
def augment_graph_weighted_differential_choice(adj, k=2, G = None ):
    import numpy as np
    import networkx as nx
    #for each node augment the edges or neigbors 
    # to the k-th multiple of that  node 
    eliminate_symbol(G)
    adj= adj.tolil()
    adj.setdiag(1)
    adj=adj.tocsr()
    adj=adj.toarray()
    # print('before augment: \n',adj)
    rows, cols = np.where(adj == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges) # drop both node and edge attributes 
    nx.draw_spring(g, with_labels = True)
#     print(g.nodes)
#     print(len(list(g.neighbors(20))), g.degree[20])
    edges = []
    g.remove_edges_from(nx.selfloop_edges(g))
    g_old = g.copy()
#     nx.draw_spring(g_old,with_labels= True)

#     print(get_sorted_neighbors(0, g_old, G, 3, k=2))
#     print('g.edges: ',g.edges)
 
    # apsp = dict(nx.shortest_path_length(g))
    # print(apsp)
    from tqdm import tqdm
    import random
    for i in tqdm(range(len(g.nodes))):
        # print('i: ',i)
        
        new_neighbors = get_sorted_differential_neighbors(i, g_old, G, g_old.degree[i])
        
#         print('node: ', i)
#         print('degree: ', g_old.degree[i])
#         print('neighbors: ', new_neighbors)
        g.remove_node(i)
        g.add_node(i)
        for j in new_neighbors:
#             print('j: ',j)
            g.add_edge(i,j)

          

            
    # g.add_edges_from(edges)
    g.remove_edges_from(nx.selfloop_edges(g))

    # nx.draw_spring(g, with_labels = True)
    
    csr= nx.to_scipy_sparse_matrix(g,range(len(g.nodes)))
    # print(csr)
    return csr, g


def augment_graph_weighted_choice(adj, k=2, G = None ):
    import numpy as np
    import networkx as nx
    #for each node augment the edges or neigbors 
    # to the k-th multiple of that  node 
    eliminate_symbol(G)
    adj= adj.tolil()
    adj.setdiag(1)
    adj=adj.tocsr()
    adj=adj.toarray()
    # print('before augment: \n',adj)
    rows, cols = np.where(adj == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges) # drop both node and edge attributes 
    # nx.draw_spring(g, with_labels = True)
#     print(g.nodes)
#     print(len(list(g.neighbors(20))), g.degree[20])
    edges = []
    g.remove_edges_from(nx.selfloop_edges(g))
    g_old = g.copy()
#     nx.draw_spring(g_old,with_labels= True)


#     print('g.edges: ',g.edges)
 
    # apsp = dict(nx.shortest_path_length(g))
    # print(apsp)
    from tqdm import tqdm
    import random
    for i in tqdm(range(len(g.nodes))):
        # print('i: ',i)
        old_neighbors = list(g_old.neighbors(i))
        neighbors = set(old_neighbors)


        for n in old_neighbors:
            n_neighbors = set( list(g_old.neighbors(n))) - neighbors
            if(len(n_neighbors)>0):
                mx_w  = 0
                mx_n = -1
                
                for nxt in n_neighbors :    # get the neighbor with highest weighted edge
                    e = (str(n+1), str(nxt+1))
                    
                    if(int(G.edges[e]['weight'])>mx_w):
                        mx_n  = nxt 
                        mx_w = int(G.edges[e]['weight'])
                    
                if(mx_n>=0 and len(neighbors)<len(old_neighbors)*k):
                    new_one = mx_n
#                     print(mx_w)
                    neighbors.add(new_one)

        new_neighbors = neighbors - set(old_neighbors)
        for j in new_neighbors:
            g.add_edge(i,j)

          

            
    # g.add_edges_from(edges)
    g.remove_edges_from(nx.selfloop_edges(g))

    # nx.draw_spring(g, with_labels = True)
    
    csr= nx.to_scipy_sparse_matrix(g,range(len(g.nodes)))
    # print(csr)
    return csr, g




def normalize_adj(adj : sp.csr_matrix):
    """Normalize adjacency matrix and convert it to a sparse tensor."""
    if sp.isspmatrix(adj):
        adj = adj.tolil()
        adj.setdiag(1)
        adj = adj.tocsr()
        deg = np.ravel(adj.sum(1))
        deg_sqrt_inv = 1 / np.sqrt(deg)
        adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
    elif torch.is_tensor(adj):
        deg = adj.sum(1)
        deg_sqrt_inv = 1 / torch.sqrt(deg)
        adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
    return to_sparse_tensor(adj_norm)

# GCN module

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)


class GraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features, g, d=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # self.eye = torch.empty(in_features, out_features)
        self.reset_parameters()
        self.g = g
        self.adj = normalize_adj(dilate_graph(g, d))
        
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        # nn.init.eye_(self.eye)
        # print(self.eye)

    def forward(self, x):
        # if(self.in_features==self.out_features):
        #     return adj @ (x @ self.weight)+ adj @ x + self.bias   # residual connection 
        # else:
        #     return adj @ (x @ self.weight) + self.bias
        # adj = dilate_graph(g) # dilated convolution
        # adj = normalize_adj(adj)
        adj = self.adj
        # return adj @ (x @ self.weight) + adj @ x + self.bias
        return adj @ (x @ self.weight)  + self.bias


class GraphConvolutionRes(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features, g, d=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # self.eye = torch.empty(in_features, out_features)
        self.reset_parameters()
        self.g = g
        self.adj = normalize_adj(dilate_graph(g, d))
        
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        # nn.init.eye_(self.eye)
        # print(self.eye)

    def forward(self, x):
        # if(self.in_features==self.out_features):
        #     return adj @ (x @ self.weight)+ adj @ x + self.bias   # residual connection 
        # else:
        #     return adj @ (x @ self.weight) + self.bias
        # adj = dilate_graph(g) # dilated convolution
        # adj = normalize_adj(adj)
        adj = self.adj
        # return adj @ (x @ self.weight) + adj @ x + self.bias
        return adj @ (x @ self.weight)  + self.bias

class GCNRes(nn.Module):
    """
    Dynamic dilated aggregation in deep residual GCNs.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False, layer_num = 2, g=nx.Graph(), d =2):
        super().__init__()
        self.dropout = dropout
        self.g = g
        hidden_layers = [hidden_dims for _ in range(layer_num-1)]
        layer_dims = np.concatenate([*hidden_layers , [output_dim]]).astype(np.int32)
        # print(layer_dims)
        self.layers = nn.ModuleList([GraphConvolutionRes(input_dim, layer_dims[0], g, d)])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GraphConvolutionRes(layer_dims[idx], layer_dims[idx + 1], g, d))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_layers
            ]
        
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x):
        # g= self.g
        for idx, gcn in enumerate(self.layers):
            # print('idx: ',idx)
            # print('layers: ',self.layers)
            # print('gcn: ',gcn)
            # print('len(self.layers): ',len(self.layers))
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
              
            if(gcn.in_features == gcn.out_features):
                x1 = gcn.adj @ x # Skip connection 
            x = gcn(x)
            
            if idx != len(self.layers) - 1:
                if(gcn.in_features == gcn.out_features):
                    x = F.relu(x) + x1 #
                else:
                    x = F.relu(x)
                # x = F.relu(x)
                # print("idx: ",idx)
                # print("len(batch_norm) : ",len(self.batch_norm))
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]


class GCNResidualv2(nn.Module):
    """
    A slightly changed version for experimentation.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False, layer_num = 2, A=None, d =2):
        super().__init__()
        self.dropout = dropout
        self.A = A
        hidden_layers = [hidden_dims for _ in range(layer_num-1)]
        layer_dims = np.concatenate([*hidden_layers , [output_dim]]).astype(np.int32)
        # print(layer_dims)
        _ , G = augment_graph_neighbors_with_degree(A)
        self.layers = nn.ModuleList([GraphConvolutionRes(input_dim, layer_dims[0], G, d)])
        for idx in range(len(layer_dims) - 1):
            _ , G = augment_graph_neighbors_with_degree(A)
            self.layers.append(GraphConvolutionRes(layer_dims[idx], layer_dims[idx + 1], G, d))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_layers
            ]
        
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x):
        # g= self.g
        for idx, gcn in enumerate(self.layers):
            # print('idx: ',idx)
            # print('layers: ',self.layers)
            # print('gcn: ',gcn)
            # print('len(self.layers): ',len(self.layers))
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
              
            if(gcn.in_features == gcn.out_features):
                x1 = gcn.adj @ x # Skip connection 
            x = gcn(x)
            
            if idx != len(self.layers) - 1:
                if(gcn.in_features == gcn.out_features):
                    x = F.relu(x) + x1 #
                else:
                    x = F.relu(x)
                # x = F.relu(x)
                # print("idx: ",idx)
                # print("len(batch_norm) : ",len(self.batch_norm))
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]

    
    
class GCN(nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False, layer_num = 2, g=nx.Graph(), d =2):
        super().__init__()
        self.dropout = dropout
        self.g = g
        hidden_layers = [hidden_dims for _ in range(layer_num-1)]
        layer_dims = np.concatenate([*hidden_layers , [output_dim]]).astype(np.int32)
        # print(layer_dims)
        self.layers = nn.ModuleList([GraphConvolution(input_dim, layer_dims[0], g, d)])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GraphConvolution(layer_dims[idx], layer_dims[idx + 1], g, d))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_layers
            ]
        
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x):
        # g= self.g
        for idx, gcn in enumerate(self.layers):
            # print('idx: ',idx)
            # print('layers: ',self.layers)
            # print('gcn: ',gcn)
            # print('len(self.layers): ',len(self.layers))
            if self.dropout != 0:
                x= x.cuda()
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x)
            
            if idx != len(self.layers) - 1:
                x = F.relu(x) + x
                # x = F.relu(x)
                # print("idx: ",idx)
                # print("len(batch_norm) : ",len(self.batch_norm))
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]

# Improved GCN

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = [
    'ImprovedGCN',
    'ImpGraphConvolution',
]


class ImpGraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_own = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_nbr = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_own, gain=2.0)
        nn.init.xavier_uniform_(self.weight_nbr, gain=2.0)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight_nbr) + x @ self.weight_own + self.bias


class ImprovedGCN(nn.Module):
    """An improved GCN architecture.

    This version uses two weight matrices for self-propagation and aggregation,
    doesn't use batchnorm, and uses Tanh instead of ReLU nonlinearities.
    Has more stable training / faster convergence than standard GCN for overlapping
    community detection.

    This improved architecture was inspired by https://arxiv.org/abs/1906.12192
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, layer_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([ImpGraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(ImpGraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if layer_norm:
            self.layer_norm = [
                nn.LayerNorm([dim], elementwise_affine=False) for dim in hidden_dims
            ]
        else:
            self.layer_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(0)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = torch.tanh(x)
                if self.layer_norm is not None:
                    x = self.layer_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]

# print(triangles[0][0])

# Decoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


__all__ = [
    'BerpoDecoder',
]


class BernoulliDecoder(nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Base class for Bernoulli decoder.

        Args:
            num_nodes: Number of nodes in a graph.
            num_edges: Number of edges in a graph.
            balance_loss: Whether to balance contribution from edges and non-edges.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        raise NotImplementedError

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        raise NotImplementedError

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute loss for given edges and non-edges."""
        raise NotImplementedError

    def loss_full(self, emb, adj):
        """Compute loss for all edges and non-edges."""
        raise NotImplementedError

import networkx as nx
import itertools
def get_all_triples(edge_list):
    g=nx.Graph()
    # ones_idx=[[0,1],[5,6],[1,5],[1,6],[0,5],[0,6]]
    # print('edge_list',edge_list)
    g.add_edges_from(edge_list)
    # nx.draw_spring(g,with_labels="True")
    G=g
    triangles =  [list(set(triple)) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr in G[nbr2])]
    non_triangle = [list(set(triple)) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr not in G[nbr2])]
    # triangles = list(triangles)
    return torch.LongTensor(triangles),torch.LongTensor(non_triangle)

class BerpoDecoder(BernoulliDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super().__init__(num_nodes, num_edges, balance_loss)
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute BerPo loss for a batch of edges and non-edges."""
        # Loss for edges
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges

        # triples = get_all_triples(ones_idx.cpu().detach().numpy())
        # # print('trip:', triples)
        # n1,n2,n3 = triples[:,0], triples[:,1], triples[:,2]

        # sum1=torch.mean(torch.sum(emb[n1]*emb[n2], dim= 1))
        # sum2=torch.mean(torch.sum(emb[n1]*emb[n3], dim= 1))
        # sum3=torch.mean(torch.sum(emb[n2]*emb[n3], dim= 1))

        # loss_triples = np.max((sum1,sum2,sum3))
        # return loss_triples
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)
    def loss_batch_triangles(self, emb, ones_idx, zeros_idx):
        """Compute BerPo loss for a batch of edges and non-edges."""
        # Loss for triangles
        

        triples,non_triples = get_all_triples(ones_idx.cpu().detach().numpy())
        total = triples.shape[0]+ non_triples.shape[0]
        # print('trip:', triples)
        n1,n2,n3 = triples[:,0], triples[:,1], triples[:,2]

        sum1=-torch.mean(torch.log(-torch.expm1(-self.eps - torch.sum(emb[n1]*emb[n2], dim = 1 ))))
        sum2=-torch.mean(torch.log(-torch.expm1(-self.eps - torch.sum(emb[n3]*emb[n2], dim = 1 ))))
        sum3=-torch.mean(torch.log(-torch.expm1(-self.eps - torch.sum(emb[n1]*emb[n3], dim = 1 ))))
        

        loss_triples = (sum1+sum2+sum3)


        n1, n2, n3 = non_triples[:,0], non_triples[:,1], non_triples[:,2]

        loss_nontriples=torch.mean(torch.sum(emb[n2]*emb[n3], dim= 1))
        neg_scale = 1.0
        # loss_nontriples =  sum3
        # return loss_triples
        triangle_loss = (loss_triples + neg_scale * loss_nontriples) / (1 + neg_scale)

        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges

        # triples = get_all_triples(ones_idx.cpu().detach().numpy())
        # # print('trip:', triples)
        # n1,n2,n3 = triples[:,0], triples[:,1], triples[:,2]

        # sum1=torch.mean(torch.sum(emb[n1]*emb[n2], dim= 1))
        # sum2=torch.mean(torch.sum(emb[n1]*emb[n3], dim= 1))
        # sum3=torch.mean(torch.sum(emb[n2]*emb[n3], dim= 1))

        # loss_triples = np.max((sum1,sum2,sum3))
        # return loss_triples
        return 0.75*(loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale) + 0.25 * triangle_loss

    def loss_full(self, emb, adj):
        """Compute BerPo loss for all edges & non-edges in a graph."""
        e1, e2 = adj.nonzero()
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Correct for overcounting F_u * F_v for edges and nodes with themselves
        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)


