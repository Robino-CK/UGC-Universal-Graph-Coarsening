from locale import currency
import math
from pickle import FALSE
from re import L
from unicodedata import name
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric
from scatter_letters import sl

import seaborn as sns
from sklearn.manifold import TSNE


from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Yelp
from torch_geometric.datasets import AmazonProducts
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import AMiner
from torch_geometric.datasets import OGB_MAG
from sklearn.neighbors import NearestNeighbors

import json
import scipy as sp
from scipy.sparse import csr_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
#import tensorflow as tf
import argparse
import time

import pygsp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import utils
import GCN
import spectral_properties
import FACH_bin_widths
import GraphSage
import GAT

def parse_args():
    parser = argparse.ArgumentParser(description='Coarsened Graph Training')
    parser.add_argument('--full_dataset',type=bool,required=False,default=False,help="Checking accuracy on original dataset.")
    parser.add_argument('--dataset',type=str,required=False,default='cora',help="Dataset name")
    parser.add_argument('--edge_index_path',type=str,required=False,default='None',help="Give path of edge index file")
    parser.add_argument('--label_path',type=str,required=False,default='None',help="Give path of label file")
    parser.add_argument('--node_feat_path',type=str,required=False,default='None',help="Give path of node feature file")
    parser.add_argument('--add_adj_to_node_features',type=bool,required=False,default=False,help="Adding Adjacency matrix one hot vectors in node features")
    parser.add_argument('--epochs',type=int,required=False, default=500,help="Number of epochs to train the coarsened graph")
    parser.add_argument('--lr',type=float,required=False,default=0.003,help="Learning Rate")
    parser.add_argument('--decay',type=float,required=False,default=0.0005,help="Learning Rate Decay")
    parser.add_argument('--seed',type=int,required=False,default=42,help="Seed")
    parser.add_argument('--ratio',type=int,required=False,default=30,help='reduction ratio list, example (30,50,70)')
    parser.add_argument('--dataset_not_in_torch_geometric',type=bool,required=False,default=False,help='Turn true if your dataset is not in the torch geometric. We will create geometric dataset first')
    parser.add_argument('--num_classes',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--number_of_projectors',type=int,required=False,default=1000,help='Total number of projectors we want while Doing LSH.')
    parser.add_argument('--out_of_sample',type=int,required=False,default=0,help='FACH2.0 should be supporting this. out_of_sample in percent (from 0 to 1) of dataset')
    parser.add_argument('--feature_size',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--hash_function',type=str,required=False,default='dot',help='Hash Function choices 1). Dot 2). L1-norm 3). L2-norm')
    parser.add_argument('--projectors_distribution',type=str,required=False,default='uniform',help='1). uniform 2). normal. coming soon.... 3). VAEs in this case need to give learned mean and sigma also.')
    parser.add_argument('--random_coarsening',type=bool,required=False,default=False,help='True for random coarsening.')
    parser.add_argument('--visualize_graph',type=bool,required=False,default=False,help='True for graph visualization.')
    parser.add_argument('--induce_adverserial_edges',type=bool,required=False,default=False,help='True for adding noise in the graph edges.')
    parser.add_argument('--tsne_visualization',type=bool,required=False,default=False,help='tsne_visualization')
    parser.add_argument('--calculate_spectral_errors',type=bool,required=False,default=False,help='calculate_spectral_errors')
    parser.add_argument('--hidden_units',type=int,required=False,default=16,help='hidden_units of GCN')
    parser.add_argument('--gsp_graphs',type=bool,required=False,default=False,help='making graphs from Graph Signal Processing lib')
    parser.add_argument('--scatter_alphabets',type=str,required=False,default="None",help='making graphs from names and alphabets')
    
    args = parser.parse_args()
    return args

def hashed_values(data, no_of_hash,feature_size,function,out_of_sample,projectors_distribution):
  #import pdb;pdb.set_trace()
  if projectors_distribution == 'VAEs':
    print("some random intilization is given here for mean and sigma make sure these contain learned values")
    learned_mean = -0.0017
    learned_sigma = 0.29
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(learned_mean,learned_sigma)
  elif projectors_distribution == 'normal':
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0,1)
  else:
    #uniform
    Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1)
  
  if out_of_sample != 0:
    num_out_of_sample = (int)(data.num_nodes*(1 - out_of_sample))
    idx = np.random.randint(data.num_nodes, size=num_out_of_sample)
    out_of_sampled_data_x = data.x[idx,:]
  else:
    out_of_sampled_data_x = data.x

  if function == 'L2-norm':
    Bin_values = torch.cdist(out_of_sampled_data_x, Wl, p = 2)
  elif function == 'L1-norm':
    Bin_values = torch.cdist(out_of_sampled_data_x, Wl, p = 1)
  else:
    #dot
    Bin_values = torch.matmul(out_of_sampled_data_x, Wl.T)
    
  return Bin_values

def allocate_list_bin_width(dataset_name,ratio_list,hash_function):
  key = dataset_name + '_' + hash_function
  full_bin_width_list =  FACH_bin_widths.BIN_WIDTH_DICTONARY[key] 
  list_bin_width = []
  for ratio in ratio_list:
    key = (str)(ratio)
    list_bin_width.append(full_bin_width_list[key]) 
  return list_bin_width

def partition(list_bin_width,Bin_values,no_of_hash):
    summary_dict = {}
    for bin_width in list_bin_width:
        bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)])#.to(device)
        temp = torch.floor((1/bin_width)*(Bin_values + bias))#.to(device)
        cluster, _ = torch.max(temp, dim = 1)
        dict_hash_indices = {}
        no_nodes = Bin_values.shape[0]
        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i]) #.to('cpu')
        summary_dict[bin_width] = dict_hash_indices 
    return summary_dict

def val(model,data):
    data = data#.to(device)
    model.eval()
    pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

def split(data, num_classes,split_percent):
    indices = []
    num_test = (int)(data.num_nodes * split_percent / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().reshape(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    
    test_index = torch.cat([i[:num_test] for i in indices], dim=0)
    val_index = torch.cat([i[num_test:int(num_test*1.5)] for i in indices], dim=0)
    train_index = torch.cat([i[int(num_test*1.5):] for i in indices], dim=0)
    data.train_mask = utils.index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = utils.index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = utils.index_to_mask(test_index, size=data.num_nodes)
    return data


def train_on_original_dataset(data, num_classes, feature_size, hidden_units, learning_rate, decay, epochs):
  model = GCN.GCN_(feature_size, hidden_units, num_classes)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)
  test_split_percent = 0.2
  data = split(data,num_classes,test_split_percent)
  
  if data.edge_attr == None:
    edge_weight = torch.ones(data.edge_index.size(1))
    data.edge_attr = edge_weight
    
  for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index,data.edge_attr.float())
    pred = out.argmax(1)
    criterion = torch.nn.NLLLoss()
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    best_val_acc = 0
    
    val_acc = val(model,data)
    if best_val_acc < val_acc:
        torch.save(model, 'full_best_model.pt')
        best_val_acc = val_acc
  
    if epoch % 20 == 0:
        print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))

  model = torch.load('full_best_model.pt')
  model.eval()
  data = data#.to(device)
  pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  
  print('--------------------------')
  print('Accuracy on test data {:.3f}'.format(acc*100))



#################
def handling_gsp_graphs(G, with_labels):
  # print(G.W.shape)
  # print(G.labels)
  #print(G.W)
  adj_matrix = G.W.toarray()
  # print(adj_matrix)

  # node_degrees = np.sum(adj_matrix, axis=1)

  # print(adj_matrix.shape)
  # print(adj_matrix)

  
  # node_features = (node_degrees - np.min(node_degrees)) / (np.max(node_degrees) - np.min(node_degrees))
  #print(node_features.shape)


  edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
  #x = torch.tensor(node_features, dtype=torch.float).unsqueeze(-1)

  # feature generation
  b=np.ones(adj_matrix.shape[0])
  z=adj_matrix@b
  D=np.diag(z)
  L=D-adj_matrix
  feature_size = adj_matrix.shape[0]
  #node_features = torch.from_numpy(np.random.multivariate_normal(np.zeros(adj_matrix.shape[0]), np.linalg.pinv(L), feature_size).T.astype(np.float32))

  node_features = torch.from_numpy(adj_matrix).type(torch.float)


  if with_labels == False:
    num_classes = 1
    labels = torch.ones(adj_matrix.shape[0])
  else:
    num_classes = len(np.unique(G.labels))
    labels = torch.from_numpy(G.labels).type(torch.LongTensor)

  print(num_classes)

  data = Data(x = node_features, edge_index=edge_index, y = labels, num_nodes = adj_matrix.shape[0])

  G_nx = torch_geometric.utils.to_networkx(data, to_undirected=True)
  
  pos = {}
  for i, coord in enumerate(G.coords):
      pos[i] = coord
  #print(" pos ",len(pos))

  #G.plot(vertex_size=10)
  nx.draw(G_nx, pos=pos, node_size=10)
  #nx.draw_networkx_nodes(G_nx, pos=pos, node_size=10, node_color=G.labels)
  plt.show()

  return data, num_classes, feature_size, pos


def plot_coarsened_graphs(pos, P, adj_matrix, labels=False):
  new_pos = {}
  P = np.array(P)
  i = 0
  for row in P:
    non_zeros_indices = np.nonzero(row)
    values = [pos[key] for key in non_zeros_indices[0]]
    new_pos[i] = values[0]
    #new_pos[i] = np.sum(values,axis = 0)/len(values)
    i += 1
  print("total supernodes are ",i)

  edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
  #x = torch.tensor(node_features, dtype=torch.float).unsqueeze(-1)
  x = torch.from_numpy(adj_matrix).type(torch.float)
  data = Data(x=x, edge_index=edge_index, num_nodes = adj_matrix.shape[0])

  G_nx = torch_geometric.utils.to_networkx(data, to_undirected=True)

  #nx.draw(G_nx, pos=new_pos, node_size=10)
  if labels == False:
    nx.draw_networkx_nodes(G_nx, pos=new_pos, node_size=10)
  else:
    nx.draw_networkx_nodes(G_nx, pos=new_pos, node_size=10, node_color=labels)
  plt.show()


def handling_scatter_alphabets_graphs(name):
  coords = sl.text_to_data(name, repeat=True, intensity = 5, rand=True, in_path=None)
  
  my_dict = {}

  for i in range(len(coords) - 1):
      new_list = []
      diff = max(coords[i][0]) - min(coords[i][0])
      if i == 0:
          min_value_x = min(coords[i][0])
          new_list.append([x - min_value_x  for x in coords[i][0]])
          max_value_x = max(coords[i][0]) - min_value_x
          min_value_x = 0
      else:
          min_value_x = max_value_x + 70
          max_value_x = min_value_x + diff
          new_list.append([x + (min_value_x - min(coords[i][0])) for x in coords[i][0]])
      
      new_list.append(coords[i][1])
      my_dict[i] = new_list

  new_list_x = []
  new_list_y = []

  for i in range(len(my_dict)):
      new_list_x.append(my_dict[i][0])
      new_list_y.append(my_dict[i][1])

  one_d_list_x = []
  one_d_list_y = []

  for sublist in new_list_x:
      for element in sublist:
          one_d_list_x.append(element)
          
  for sublist in new_list_y:
      for element in sublist:
          one_d_list_y.append(element)

  # plt.scatter(one_d_list_x, one_d_list_y)
  # plt.show()

  points = np.array([one_d_list_x,one_d_list_y]).T

  # create a NearestNeighbors object
  k = 10
  nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)

  # get the indices of the nearest neighbors for each point
  _, indices = nn.kneighbors(points)

  # create an empty graph
  G = nx.Graph()

  # add the nodes to the graph
  for i in range(len(points)):
      G.add_node(i, pos=points[i])

  # add the edges to the graph
  # for i in range(len(points)):
  #     for j in indices[i][1:]:
  #         G.add_edge(i, j)

  # draw the graph
  pos = nx.get_node_attributes(G, 'pos')
  nx.draw(G, pos = points, node_size = 10)
  plt.show()


#################


if __name__ == "__main__":

  time1 = time.time()  
  args = parse_args()
  utils.fix_seeds(args.seed)
  device = torch.device("cpu")
  torch.cuda.empty_cache()

  if args.dataset_not_in_torch_geometric == True:
    '''Our dataset is not present on the torch_geometric datasets.
      Create Instance of the torch_geo from edge_index, label, node_feat.
    '''
    if args.edge_index_path == False or args.label_path == False or args.node_feat_path == False or args.num_classes == -1 or args.feature_size == -1:
      print("One or more required variable for creating Instance of Geometric dataset is missing. Please try again after giving information about following variable edge_index_path, label_path, node_feat_path, feature_size and num_classes")
      exit(1)
    
    new_dataset_hetro_node_feat = torch.load(args.node_feat_path)
    new_dataset_edge_index = torch.load(args.edge_index_path)
    new_dataset_hetro_label = torch.from_numpy(torch.load(args.label_path)).type(torch.LongTensor)

    data = Data(x=new_dataset_hetro_node_feat, edge_index = new_dataset_edge_index, y = new_dataset_hetro_label)
    num_classes = args.num_classes
    feature_size = args.feature_size
    print("done with new_dataset formation.")
  
  elif args.gsp_graphs == True:
    with_label = False
    
    if args.dataset == 'logo':
      G = pygsp.graphs.Logo()
    elif args.dataset == 'comet':
      G = pygsp.graphs.Comet()
    elif args.dataset == 'community':
       G = pygsp.graphs.Community()
    elif args.dataset == 'ring':
       G = pygsp.graphs.Ring()
    else:
      with_label = True
      G = pygsp.graphs.TwoMoons()


    data, num_classes, feature_size, pos = handling_gsp_graphs(G, with_label)
    print("done with fetching gsp_graphs")

  elif args.scatter_alphabets != "None":
    handling_scatter_alphabets_graphs(args.scatter_alphabets)
    print("done with handling_scatter_alphabets_graphs")
    exit(1)

  else:
    if args.dataset == 'karate':
      '''
      KarateClub nodes dont have features so we are generating its node features
      using its Laplacian's pseudo inverse see  karateClub_data_generation()
      for more details.
      '''
      dataset = KarateClub()
      karate_data_generation = 'deep_walk'

      if karate_data_generation == 'deep_walk':
        data, feature_size, num_classes = utils.karateClub_data_generation_deepwalk()
      else:
        data, feature_size, num_classes = utils.karateClub_data_generation()

    elif args.dataset == 'AMiner':
      # Heterogenous data
      dataset = AMiner(root = 'data/AMiner')
    
    elif args.dataset == 'OGB_MAG':
    # Heterogenous data
      dataset = OGB_MAG(root='./data', preprocess='metapath2vec')

    elif args.dataset == 'flickr':
      dataset = Flickr(root = 'data/Flickr')

    elif args.dataset == 'yelp':
      dataset = Yelp(root = 'data/Yelp')

    elif args.dataset == 'reddit':
      dataset = Reddit(root = 'data/Reddit')

    elif args.dataset == 'reddit2':
      dataset = Reddit2(root = 'data/Reddit')

    elif args.dataset == 'citeseer':
      dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer')

    elif args.dataset == 'cora':
      dataset = Planetoid(root = 'data/Cora', name = 'Cora')

    elif args.dataset == 'pubmed':
      dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')

    elif args.dataset == 'physics':
      dataset = Coauthor(root = 'data/Physics', name = 'Physics')
    
    elif args.dataset == 'dblp':
      dataset = CitationFull(root = 'newdata/DBLP', name = 'DBLP')

    elif args.dataset == 'cs':
      dataset = Coauthor(root = 'data/CS', name = 'CS')

    elif args.dataset == 'amazon':
      dataset = AmazonProducts(root = 'data/AmazonProducts')
    
    else:
      print("For now FACH don't support your mentioned dataset: ",args.dataset,". \nExiting......")
      exit(1)


    if args.dataset != 'karate':
      data = dataset[0]    
      num_classes = dataset.num_classes
      feature_size = dataset.num_features
       

  
  if args.add_adj_to_node_features == True:
    g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
    #adding self loops
    # g_adj.fill_diagonal_(1)
    
    #Add random noise to increase the uniqueness of supernodes range of randomness should be small such that similarity of nodes still exist also it should not be too 
    #small else we will not be able to induce the uniqueness.
    epsilon = 0.1
    random_numbers = np.random.uniform(-epsilon, epsilon, g_adj.shape)
    g_adj = g_adj.numpy()
    # Replace non-zero entries in the array with random numbers
    #g_adj[g_adj != 0] = random_numbers[g_adj != 0]
    g_adj = torch.from_numpy(g_adj)

    # alpha decides how much heterophly you want
    alpha = 0.77
    data.x = (1-alpha)*data.x
    g_adj = alpha*g_adj


    data.x = torch.cat((data.x, g_adj), dim = 1)

    feature_size = feature_size + data.num_nodes

  if args.full_dataset == True:
    train_on_original_dataset(data,num_classes,feature_size,args.hidden_units,args.lr,args.decay,args.epochs)
    exit(1)


  no_of_hash = args.number_of_projectors
  out_of_sample = args.out_of_sample
  hash_function = args.hash_function
  projectors_distribution = args.projectors_distribution

  test_split_percent = 0.2
  data = split(data,num_classes,test_split_percent) 
  time2 = time.time()
  
  Bin_values = hashed_values(data, no_of_hash, feature_size,hash_function,out_of_sample,projectors_distribution)  
  time3 = time.time()
  
  list_bin_width = allocate_list_bin_width(args.dataset,[args.ratio],args.hash_function)
  list_bin_width = [3]
  
  summary_dict = {}
  summary_dict = partition(list_bin_width,Bin_values,no_of_hash)
  temp_time4 = time.time()
  print("time taken in partition",temp_time4-time2)

  he_error_list = []
  ree_error_list = []
  dirichlet_energy_list = []

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
      
      # i thinnk this can be improved
      # does this have a time complexity if O(N*v) ? i.e for each unique value searching each node hash value
      for v in unique_values:
          C_diag[help_count],dict_blabla[help_count] = utils.get_key(v, current_bin_width_summary)
          help_count += 1

      # P_hat is bool 2D array which represent nodes contained in supernodes 
      P_hat = torch.zeros((data.num_nodes, len(unique_values)))#, device= device)
      zero_list = torch.ones(len(unique_values), dtype=torch.bool)
      
      if args.random_coarsening == False:
        for x in dict_blabla:
            if len(dict_blabla[x]) == 0:
              print("zero element in this supernode",x)
            for y in dict_blabla[x]:
                P_hat[y,x] = 1
                zero_list[x] = zero_list[x] and (not (data.train_mask)[y])
      else:
        #If we Randomly sample coarsened graph 
        for x in dict_blabla:
          if len(dict_blabla[x]) == 0:
            print("zero element in this supernode",x)
          num_nodes_in_this_supernode = 1#5*len(dict_blabla[x])#random.sample(range(0, data.num_nodes),1)[0]
          #num_nodes_in_this_supernode =  random.sample(range(0, (int)(data.num_nodes/len(unique_values))),1)
          random_array = random.sample(range(0, data.num_nodes), num_nodes_in_this_supernode)
          for y in random_array:
              P_hat[y,x] = 1
              zero_list[x] = zero_list[x] and (not (data.train_mask)[y])
      
      # for row in P_hat:
      #   print(row)
      
      print("check missclassified")
      missclassified = utils.detect_missclassified(P_hat,data.y)
      print(missclassified)

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

      # next line only for GCN training 
      g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)
      
      if args.induce_adverserial_edges == True:
        print("You have decided to induce adverserial edges into your graph\n")
        for i in range((int)(np.shape(g_coarse_dense)[0]*0.1)):
          for j in range((int)(np.shape(g_coarse_dense)[0]*0.1)):
            g_coarse_dense[i][j] = 0
     
      edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
      edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
      edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
      edge_index_corsen = torch.stack((edges_src, edges_dst))
      edge_features = torch.from_numpy(edge_weight)
            
      # -----------
      if args.gsp_graphs == True:
        plot_coarsened_graphs(pos, P.T, g_coarse_dense)#, labels=labels_coarse)
        print("plot_coarsened_graphs ")
        exit(1)
      #-------------


      if args.calculate_spectral_errors == True:
        if data.x.size(0) < 100:
          number_of_eigen_vectors = (int)(data.x.size(0)/2)
        else:
          number_of_eigen_vectors = 100
        
        spectral_properties.eigen_error(data.edge_index, edge_index_corsen, edge_features, number_of_eigen_vectors)

      Y = np.array(data.y.cpu())
      Y = utils.one_hot(Y,num_classes)#.to(device)
      Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)])#.to(device)
      labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1)#.to(device)

      # deleting unused variables
      del C_diag_matrix
      del g_coarse_adj
      del edge_weight
      del edges_dst
      del i
      del v

      data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
      data_coarsen.edge_attr = edge_features

      ##----------------
      # print("Training  GraphSage")
      # test_split_percent = 0.1
      # data_coarsen = split(data_coarsen,dataset.num_classes,test_split_percent)
      
      # GraphSage.train_graphSage(dataset.num_features, dataset.num_classes, data_coarsen)
      # # GAT.train_GAT(dataset.num_features, dataset.num_classes, dataset[0])
      # exit(1)
      ##----------------

      if args.tsne_visualization == True:
        original_tsne_graph_name = 'results_and_plots/tsne_original_' + args.dataset 
        utils.t_sne_visualize_graph(data.x,data.y,original_tsne_graph_name)
        coarsen_tsne_graph_name = 'results_and_plots/tsne_coarsen_' + args.dataset
        utils.t_sne_visualize_graph(data_coarsen.x.to_dense(),data_coarsen.y,coarsen_tsne_graph_name)

      # data.edge_index, edge_index_corsen, edge_features
      if args.calculate_spectral_errors == True:
        he_error = spectral_properties.hyperbolic_error(np.array(P_hat.to_dense()).T,data.edge_index,edge_index_corsen,edge_features,np.array(data.x))
        he_error_list.append(he_error)
        print("check hyperbolic error",he_error)
        
        eigen_plot_name = 'results_and_plots/' + args.dataset + '_' + (str)(math.floor(rr*100))
        spectral_properties.plot_most_significant_eigen_values(100,data.edge_index,edge_index_corsen,edge_features,eigen_plot_name)
        
        re_construct_error = spectral_properties.reconstruction_error(data.num_nodes,np.array(P_hat.to_dense()).T,data.edge_index,edge_index_corsen,edge_features)
        ree_error_list.append(re_construct_error)
        print("re_construction error ",re_construct_error)
        
        diri_energy = spectral_properties.dirichlet_energy(np.array(P_hat.to_dense()),data.edge_index,edge_index_corsen,edge_features,np.array(data.x),np.array(cor_feat.to_dense()))
        dirichlet_energy_list.append(diri_energy)
        print("dirichlet_energy error ",diri_energy)
      

      #this is main g_coarse_adj use it to visualize supernodes
      if args.visualize_graph == True:
        original_graph_name = 'results_and_plots/original_' + args.dataset
        pos = utils.visualize_graph(data.edge_index,data.num_nodes,data.y,original_graph_name)
        
        new_pos = {}
        i = 0
        for row in P.T:
          non_zeros_indices = np.array(np.nonzero(row))
          values = [pos[key] for key in non_zeros_indices[0]]
          new_pos[i] = values[0]
          #new_pos[i] = np.sum(values,axis = 0)/len(values)
          i += 1
        print("total supernodes are ",i)
        
        coarsen_graph_name = 'results_and_plots/coarsen_' + args.dataset + '_' + (str)(math.floor(rr*100))
        utils.visualize_graph(edge_index_corsen,len(unique_values),data_coarsen.y,coarsen_graph_name,new_pos)

      time5 = time.time()
      print('diff b/w t5 and t4 {}'.format(time5-time4))

      all_acc = []
      num_run = 1

      time_taken_to_train_gcn = []
      for i in range(num_run):
        global_best_val = 0
        global_best_test = 0
        best_val_acc = 0
        best_epoch = 0

        hidden_units = args.hidden_units
        learning_rate = args.lr
        decay = args.decay
        epochs = args.epochs
        
        model = GCN.GCN_(feature_size, hidden_units, num_classes)
        model = model#.to(device)
        data_coarsen = data_coarsen#.to(device)
        edge_weight = torch.ones(data_coarsen.edge_index.size(1))
        decay = decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data_coarsen.x, data_coarsen.edge_index,data_coarsen.edge_attr.float()) 
            pred = out.argmax(1)
            criterion = torch.nn.NLLLoss()
            loss = criterion(out[~zero_list], data_coarsen.y[~zero_list]) 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            val_acc = val(model,data)
            if best_val_acc < val_acc:
                torch.save(model, 'best_model.pt')
                best_val_acc = val_acc
                best_epoch = epoch
          
            if epoch % 20 == 0:
                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))

        time6 = time.time()
        print('diff b/w t6 and t5 {}'.format(time6-time5))
        time_taken_to_train_gcn.append(time6-time5)
        model = torch.load('best_model.pt')
        model.eval()
        data = data#.to(device)
        pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        
        acc = int(correct) / int(data.test_mask.sum())

        time7 = time.time()
        #print('diff b/w t7 and t5 {}'.format(time7-time5))
        all_acc.append(acc)
        
        # if t_sne and other visualizations take time it is better to use this limited visualization to get 
        # the gist of data

        # np.random.seed(432)
        # temp = random.sample(range(0, data.num_nodes), 2000)
        #print(temp)
        #t_sne_visualize_graph(data.x[temp],data.y[temp],"tsne_physics_limited")
        #t_sne_visualize_graph(data.x[temp],pred[temp],"tsne_physics_coarsened_50_limited")
        #t_sne_visualize_graph(data_coarsen.x.to_dense(),data_coarsen.y,"tsne_physics_only_fach_30")
      
      print("ratio ",rr)
      print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
      print('ave_time: {:.4f}'.format(np.mean(time_taken_to_train_gcn)), '+/- {:.4f}'.format(np.std(time_taken_to_train_gcn)))
      print("he_error_list ",he_error_list)
      print("ree_error_list ",ree_error_list)
      print("dirichlet_energy_list ",dirichlet_energy_list)