# UGC-Universal-Graph-Coarsening
This is the official repository of [UGC]([https://nips.cc/virtual/2024/poster/93695]): Universal Graph Coarsening accepted in 38 Conference on Neural Information Processing Systems ( [NeurIPS24]([https://www.genome.gov/](https://neurips.cc/Conferences/2024/CallForPapers)), held at the Vancouver Convention Center.


## Dependencies
Install required dependencies from requirement.txt file.

## Run
UGC support Vanilla GCN, GraphSage, GIN, GAT, APPNP GCN and 3WL gnn models. To run UGC use run.sh file.

### vanilla GCN
python UGC.py --dataset=cora --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.19

### GCN used in UGC 
python UGC.py --dataset=cora --model_type=ugc --ratio=50 --add_adj_to_node_features=True --alpha=0.19

### 3wl
python UGC.py --dataset=cora --model_type=3wl --ratio=50 --add_adj_to_node_features=True --alpha=0.19

### Graph Sage
python UGC.py --dataset=cora --model_type=sage --ratio=50 --add_adj_to_node_features=True --alpha=0.19

### GAT
python UGC.py --dataset=cora --model_type=gat --ratio=50 --add_adj_to_node_features=True --alpha=0.19

### GIN
python UGC.py --dataset=cora --model_type=gin --ratio=50 --add_adj_to_node_features=True --alpha=0.19
