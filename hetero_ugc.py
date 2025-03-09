from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
import torch  
from UGC import hashed_values

from torch_geometric.transforms import AddMetaPaths
from torch_geometric.data import HeteroData

import time
import utils
from torch_geometric.data import Data
from UGC import plot_coarsened_graphs
import numpy as np
from UGC import partition
from UGC import allocate_list_bin_width
from torch_geometric.data import Data
import torch

def merge_nodes(data, summary_dict, list_bin_width, device):
    
    for bin_width in list_bin_width:
        time4 = time.time()
        current_bin_width_summary = summary_dict[bin_width]
        values = current_bin_width_summary.values()
        unique_values = set(values)
        rr = 1 - len(unique_values)/len(values)
        print(f'Graph reduced by: {rr*100} percent.\nWe now have {len(unique_values)} supernode, starting nodes were: {len(values)}')
        dict_blabla ={}
        C_diag = torch.zeros(len(unique_values))#, device= device)
        help_count = 0
        
        for v in unique_values:
            C_diag[help_count],dict_blabla[help_count] = utils.get_key(v, current_bin_width_summary)
            help_count += 1

        P_hat = torch.zeros((data.num_nodes, len(unique_values)))#, device= device)
        zero_list = torch.ones(len(unique_values), dtype=torch.bool)
        
        for x in dict_blabla:
            if len(dict_blabla[x]) == 0:
                print("zero element in this supernode",x)
            for y in dict_blabla[x]:
                P_hat[y,x] = 1
                if not data.y is None:
    
                    zero_list[x] = zero_list[x] and (not (data.train_mask)[y])
            
        P_hat = P_hat.to_sparse()
        #dividing by number of elements in each supernode to get average value 
        P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))
        
        features =  data.x.to(device = device).to_sparse()




        # cor_feat : features of supernodes by averaging out all the features values of child nodes
        cor_feat = (torch.sparse.mm((torch.t(P)), features.to_dense()))#.to_sparse()
        i = data.edge_index
        v = torch.ones(data.edge_index.shape[1])
        shape = torch.Size([data.x.shape[0],data.x.shape[0]])
        g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to(device = device)
        g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
        
        C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))
        #print("number of edges in the coarsened graph ",np.count_nonzero(g_coarse_adj.to_dense().to('cpu').numpy())/2)

        g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)
        
        
    
        edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
        edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
        edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
        edge_index_corsen = torch.stack((edges_src, edges_dst))
        edge_features = torch.from_numpy(edge_weight)

        #------------------
        ## Epsilion bounds
        
        # epsilion_bound = utils.get_smooth_features(data.edge_index, P_hat, data.x.numpy())
        # print("epsilion_bound ", epsilion_bound)
        # exit(1)
        #------------------
        if not data.y is None:
            num_classes = len(np.unique(data.y.numpy()))
    
            Y = np.array(data.y.cpu())
            Y = utils.one_hot(Y,num_classes)#.to(device)
            Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)])#.to(device)
            labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1)#.to(device)
    
            data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
        else:
            data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen)
            
        data_coarsen.edge_attr = edge_features
        projection = {}
        for key, value in dict_blabla.items():
            for v in value:
                projection[v] = key
        return data_coarsen, projection


def coarsen_graph(data):
    feature_size = data.num_features + data.num_nodes

    alpha = 0.19
    data.x = (1-alpha)*data.x
    g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
    g_adj = alpha*g_adj
    data.x = torch.cat((data.x, g_adj), dim = 1)
        
    no_of_hash = 500

    h_function = "dot"
    out_of_sample = 0
    projectors_distribution = 'uniform'
    A = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
    Bin_values = hashed_values(data, no_of_hash, feature_size, h_function, out_of_sample, projectors_distribution, A) 
    dataset_name = 'cora'
    ratio = 50
    scatter_alphabets = 'None'
    list_bin_width = allocate_list_bin_width(dataset_name,[ratio],h_function,scatter_alphabets)
    summary_dict = partition(list_bin_width, Bin_values, no_of_hash) # projection map
    device = "cpu"
    return merge_nodes(data, summary_dict, list_bin_width, device)


import torch
from torch_geometric.data import HeteroData

def reconstruct_heterogeneous_graph(original_hetero_data, merged_graphs, node_mappings):
    """
    Reconstruct a heterogeneous graph from separate merged homogeneous graphs using PyTorch.
    
    Parameters:
    -----------
    original_hetero_data : torch_geometric.data.HeteroData
        The original heterogeneous graph
    merged_graphs : dict
        Dictionary mapping node types to their merged homogeneous graphs
        e.g., {'user': user_merged_g, 'item': item_merged_g, 'tag': tag_merged_g}
    node_mappings : dict
        Dictionary mapping node types to their node mapping dictionaries
        e.g., {'user': user_node_mapping, 'item': item_node_mapping, 'tag': tag_node_mapping}
        where each mapping is {original_node_id: merged_node_id}
    
    Returns:
    --------
    tuple
        - torch_geometric.data.HeteroData: A new heterogeneous graph with the merged nodes
        - dict: Dictionary containing inverse mappings from merged node IDs back to original node IDs
              e.g., {'user': {merged_id: [original_ids]}, 'item': {merged_id: [original_ids]}, ...}
    """
    # Create a new HeteroData object for the reconstructed graph
    new_hetero_data = HeteroData()
    
    # Get all node types from the original graph (metadata stores node types)
    node_types = list(original_hetero_data.node_types)
    
    # Get all edge types from the original graph
    edge_types = list(original_hetero_data.edge_types)
    
    # First, copy node features from merged graphs to the new heterogeneous graph
    for ntype in node_types:
        # Set the number of nodes for this node type
        num_nodes = merged_graphs[ntype].x.size(0)
        
        # Copy node features
        new_hetero_data[ntype].x = merged_graphs[ntype].x
        
        # Copy any other node attributes that might exist
        for key, value in merged_graphs[ntype]:
            if key != 'x' and key != 'edge_index':
                new_hetero_data[ntype][key] = value
    
    # Process each edge type in the original heterogeneous graph
    for edge_type in edge_types:
        src_type, relation_type, dst_type = edge_type
        
        # Get all edges of this type from original graph
        orig_edge_index = original_hetero_data[edge_type].edge_index
        orig_src = orig_edge_index[0].numpy()
        orig_dst = orig_edge_index[1].numpy()
        
        # Map original node IDs to new merged node IDs
        src_mapping = node_mappings[src_type]
        dst_mapping = node_mappings[dst_type]
        
        # Create new edge lists
        new_src = []
        new_dst = []
        edge_weights = {}  # To track edge weights for multi-edges between same node pairs
        
        # Process each edge
        for i in range(len(orig_src)):
            # Get original source and destination node IDs
            o_src = orig_src[i].item()
            o_dst = orig_dst[i].item()
            
            # Map to merged node IDs
            m_src = src_mapping[o_src]
            m_dst = dst_mapping[o_dst]
            
            # Add edge to new edge lists (avoid duplicates by tracking edge weights)
            edge_key = (m_src, m_dst)
          #  print(edge_key)
            if edge_key not in edge_weights:
                edge_weights[edge_key] = 1
                new_src.append(m_src)
                new_dst.append(m_dst)
            else:
                edge_weights[edge_key] += 1
        
        # Create tensor edge lists
        new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        
        # Add to new heterogeneous graph
        new_hetero_data[src_type, relation_type, dst_type].edge_index = new_edge_index
        
        # Copy edge features if they exist in the original graph
        for key in original_hetero_data[edge_type].keys():
            if key != 'edge_index':
                # For simplicity, we're skipping edge features here since merging them
                # would require a more complex aggregation strategy based on specific needs
                pass
    
    return new_hetero_data

def hetero_coarsen(dataset_hetero):
    
    # Add meta-paths to your data
    data = dataset_hetero[0]
    data["conference"].x = torch.zeros((20, 1))
    metapaths = [
        [("paper", "conference"), ("conference", "paper")],
        [("author", "paper"), ("paper", "author")],
        [("conference", "paper"), ("paper", "conference")],
        [("term", "paper"), ("paper", "term")],
    ]
    data_with_metapaths = AddMetaPaths(metapaths)(data.clone())
    # Create a new HeteroData object
    data_only_metapaths = HeteroData()

    # Copy node features
    for node_type in data_with_metapaths.node_types:
        data_only_metapaths[node_type].update(data_with_metapaths[node_type])

    # Copy only the meta-path edges - they have the special naming convention with "__"
    for edge_type in data_with_metapaths.edge_types:
        if "_" in edge_type[1]:  # This identifies meta-path edges
            data_only_metapaths[edge_type].edge_index = data_with_metapaths[edge_type].edge_index
            # Copy any other edge attributes
            for key, value in data_with_metapaths[edge_type].items():
                if key != 'edge_index':
                    data_only_metapaths[edge_type][key] = value
                


    # First, we'll create the data_only_metapaths as we discussed before
    # ... (your existing code here) ...

    # Now let's create separate homogeneous graphs for each node type
    homogeneous_graphs = {}

    for node_type in data_only_metapaths.node_types:
        # Create a new homogeneous Data object
        homo_graph = Data()
        
        # Copy node features for this node type
        for key, value in data_only_metapaths[node_type].items():
            homo_graph[key] = value
        
        # Find all meta-path edges where this node type is both source and target
        relevant_edges = []
        
        for edge_type in data_only_metapaths.edge_types:
            src, edge_name, dst = edge_type
            
            # Only include edges where both source and target are the current node type
            if src == node_type and dst == node_type:
                edge_index = data_only_metapaths[edge_type].edge_index
                relevant_edges.append(edge_index)
        
        # If we found any relevant edges, combine them
        if relevant_edges:
            # Concatenate all edge indices
            combined_edge_index = torch.cat(relevant_edges, dim=1)
            homo_graph.edge_index = combined_edge_index
        else:
            # No edges found, create empty edge_index
            num_nodes = data_only_metapaths[node_type].num_nodes
            homo_graph.edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Store the homogeneous graph
        homogeneous_graphs[node_type] = homo_graph

    # Now homogeneous_graphs is a dictionary with a homogeneous graph for each node type
    
    coarsend_graphs = {}
    mappings = {}
    for ty, graph in homogeneous_graphs.items():
        if ty in ["paper", "term"]:
            g = graph
            map = {i: i for i in range(graph.num_nodes)}
        else:
            g, map =  coarsen_graph(graph)
        #print("map", map)
        coarsend_graphs[ty] = g
        mappings[ty] = map


    return reconstruct_heterogeneous_graph(data, coarsend_graphs, mappings), mappings