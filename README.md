# UGC-Universal-Graph-Coarsening
This is the official repository of UGC: Universal Graph Coarsening accepted in 38 Annual Conference on Neural Information Processing Systems, held at the Vancouver Convention Center.


## Dependencies
Install required dependencies from requirement.txt file.

## Run
UGC support Vanilla GCN, GraphSage, GIN, GAT, APPNP GCN and 3WL gnn models. To run UGC use run.sh file.

## vanilla GCN
python UGC.py --dataset=cora --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.19

## used in UGC 
python UGC.py --dataset=cora --model_type=ugc --ratio=50 --add_adj_to_node_features=True --alpha=0.19

## Implemented for NeurIPS rebuttal 3wl
python UGC.py --dataset=cora --model_type=3wl --ratio=50 --add_adj_to_node_features=True --alpha=0.19

## Graph Sage
python UGC.py --dataset=cora --model_type=sage --ratio=50 --add_adj_to_node_features=True --alpha=0.19

## GAT
python UGC.py --dataset=cora --model_type=gat --ratio=50 --add_adj_to_node_features=True --alpha=0.19

## GIN
python UGC.py --dataset=cora --model_type=gin --ratio=50 --add_adj_to_node_features=True --alpha=0.19
