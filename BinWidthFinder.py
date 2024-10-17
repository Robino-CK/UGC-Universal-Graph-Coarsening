import numpy as np
import random

import scipy

from collections import Counter
import torch
from torch.nn import Linear
import argparse
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Yelp
from torch_geometric.datasets import NELL
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import OGB_MAG
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.datasets import AMiner
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian


from utils import karateClub_data_generation

import sys
import logging
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clustering(data, no_of_hash, bin_width, function, projectors_distribution,feature_size):
  data = data.to(device)
  
  if projectors_distribution == 'random':
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0,1)
  elif projectors_distribution == 'VAE':
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(-0.0017,0.29)
  else:
    Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1).to(device)

  no_nodes = data.x.shape[0]
  features = data.x
  bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)]).to(device)
  features.to(device)
  if function == 'dot':
    Bin_values = torch.floor((1/bin_width)*(torch.matmul(features, Wl.T) + bias)).to(device)
  elif function == 'l1':
    Bin_values = torch.floor((1/bin_width)*(torch.cdist(features, Wl, p = 1) + bias)).to(device)
  else:
    Bin_values = torch.floor((1/bin_width)*(torch.cdist(features, Wl, p = 2) + bias)).to(device)
  cluster, _ = torch.max(Bin_values, dim = 1)
  dict_hash_indices = {}
  for i in range(no_nodes):
    dict_hash_indices[i] = int(cluster[i]) #.to('cpu')
  return dict_hash_indices


def Find_Binwidth(feature_size, data,coarsening_ratio = 0.05, precision = 0.0005,hash_function='dot',projectors_distribution='uniform',number_of_projectors=1000):
  bw = 1
  ratio = 1
  counter = 0
  while(abs(ratio - coarsening_ratio) > precision):
    counter = counter + 1
    if(ratio > coarsening_ratio):
      bw = bw*0.5
    else:
      bw = bw*1.5

    g_coarsened = clustering(data.to(device),number_of_projectors, bw,hash_function,projectors_distribution,feature_size)
    values = g_coarsened.values() 
    unique_values = set(g_coarsened.values())
    ratio = (1 - (len(unique_values)/len(values)))
  print(counter)

  return bw, ratio

def parse_args():
    parser = argparse.ArgumentParser(description='Find Bin Width')
    parser.add_argument('--hash_function',type=str,required=False,default='dot',help='Hash Function choices 1). Dot 2). L1-norm 3). L2-norm')
    parser.add_argument('--feature_size',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--projectors_distribution',type=str,required=False,default='uniform',help='1). uniform 2). normal. coming soon.... 3). VAEs in this case need to give learned mean and sigma also.')
    parser.add_argument('--number_of_projectors',type=int,required=False,default=1000,help='Total number of projectors we want while Doing LSH.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('BinWidths.log', 'a'))
    print = logger.info
    args = parse_args()

    '''
    To get the appropriate bin-width values for different coarsening ratio's
    Copy and Paste any name from the below supported datasets to datasets_name list

    1.  Karate
    2.  AMiner
    3.  OGB_MAG
    4.  Flickr
    5.  Yelp
    6.  Reddit
    7.  Reddit2
    8.  Citeseer
    9.  Cora
    10. Pubmed
    11. Physics
    12. DBLP
    13. CS

    '''

    datasets_name = ['Cora','Citeseer']

    for d in datasets_name:
      print("Calculating Bin-widths for ",d," dataset")
      if d == 'Karate':
        '''
        KarateClub nodes dont have features so we are generating its node features
        using its Laplacian's pseudo inverse see  karateClub_data_generation()
        for more details.
        '''
        dataset = KarateClub()
        data, _ = karateClub_data_generation()
      
      elif d == 'AMiner':
        # Heterogenous data
        dataset = AMiner(root = 'data/AMiner')
        data = dataset[0]
      
      elif d == 'OGB_MAG':
      # Heterogenous data
        dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
        data = dataset[0]

      elif d == 'Flickr':
        dataset = Flickr(root = 'data/Flickr')
        data = dataset[0]

      elif d == 'Yelp':
        dataset = Yelp(root = 'data/Yelp')
        data = dataset[0]

      elif d == 'Reddit':
        dataset = Reddit(root = 'data/Reddit')
        data = dataset[0]

      elif d == 'Reddit2':
        dataset = Reddit2(root = 'data/Reddit')
        data = dataset[0]
   
      elif d == 'Citeseer':
        dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer')
        data = dataset[0]
   
      elif d == 'Cora':
        dataset = Planetoid(root = 'data/Cora', name = 'Cora')
        data = dataset[0]

      elif d == 'Pubmed':
        dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')
        data = dataset[0]
   
      elif d == 'Physics':
        dataset = Coauthor(root = 'data/Physics', name = 'Physics')
        data = dataset[0]
      
      elif d == 'DBLP':
        dataset = CitationFull(root = 'newdata/DBLP', name = 'DBLP')
        data = dataset[0]
   
      elif d == 'CS':
        dataset = Coauthor(root = 'data/CS', name = 'CS')
        data = dataset[0]


    hash_function = args.hash_function
    projectors_distribution = args.projectors_distribution
    if args.feature_size == -1:
      feature_size = dataset.num_features
    else:
      feature_size = args.feature_size
    number_of_projectors = args.number_of_projectors


    valid_ratios_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    # need to write code which can directly write bin-widths to the FACH_bin_widths file
    for ratio in valid_ratios_list:
      bw, ratio = Find_Binwidth(feature_size,data,coarsening_ratio = ratio, precision = 0.05,hash_function = hash_function,projectors_distribution = projectors_distribution,number_of_projectors = number_of_projectors)
      print(bw)
      print(f'{ratio*100} percent was asked we get {ratio}')