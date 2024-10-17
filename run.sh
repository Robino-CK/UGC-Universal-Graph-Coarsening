## vanilla GCN
python UGC.py --dataset=cora --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.19
python UGC.py --dataset=dblp --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.18
python UGC.py --dataset=pubmed --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.20
python UGC.py --dataset=physics --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.07
python UGC.py --dataset=squirrel --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.78
python UGC.py --dataset=chameleon --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.75
python UGC.py --dataset=texas --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.91
python UGC.py --dataset=film --model_type=gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.78


## used in UGC 
python UGC.py --dataset=cora --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.19
python UGC.py --dataset=dblp --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.18
python UGC.py --dataset=pubmed --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.20
python UGC.py --dataset=physics --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.07
python UGC.py --dataset=squirrel --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.78
python UGC.py --dataset=chameleon --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.75
python UGC.py --dataset=texas --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.91
python UGC.py --dataset=film --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.78

## Implemented for NeurIPS rebuttal
python UGC.py --dataset=cora --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.19
python UGC.py --dataset=dblp --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.18
python UGC.py --dataset=pubmed --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.20
python UGC.py --dataset=physics --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.07
python UGC.py --dataset=squirrel --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.78
python UGC.py --dataset=chameleon --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.75
python UGC.py --dataset=texas --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.91
python UGC.py --dataset=film --model_type=appnp_gcn --ratio=50 --add_adj_to_node_features=True --alpha=0.78

## Graph Sage
python UGC.py --dataset=cora --model_type=sage --ratio=50 --add_adj_to_node_features=True --alpha=0.19
python UGC.py --dataset=dblp --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.18
python UGC.py --dataset=pubmed --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.20
python UGC.py --dataset=physics --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.07
python UGC.py --dataset=squirrel --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.78
python UGC.py --dataset=chameleon --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.75
python UGC.py --dataset=texas --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.91
python UGC.py --dataset=film --model_type=sage--ratio=50 --add_adj_to_node_features=True --alpha=0.78

## GAT
python UGC.py --dataset=cora --model_type=gat --ratio=50 --add_adj_to_node_features=True --alpha=0.19
python UGC.py --dataset=dblp --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.18
python UGC.py --dataset=pubmed --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.20
python UGC.py --dataset=physics --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.07
python UGC.py --dataset=squirrel --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.78
python UGC.py --dataset=chameleon --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.75
python UGC.py --dataset=texas --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.91
python UGC.py --dataset=film --model_type=gat--ratio=50 --add_adj_to_node_features=True --alpha=0.78

## GIN
python UGC.py --dataset=cora --model_type=gin --ratio=50 --add_adj_to_node_features=True --alpha=0.19
python UGC.py --dataset=dblp --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.18
python UGC.py --dataset=pubmed --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.20
python UGC.py --dataset=physics --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.07
python UGC.py --dataset=squirrel --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.78
python UGC.py --dataset=chameleon --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.75
python UGC.py --dataset=texas --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.91
python UGC.py --dataset=film --model_type=gin--ratio=50 --add_adj_to_node_features=True --alpha=0.78