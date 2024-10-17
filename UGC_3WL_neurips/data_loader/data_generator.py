import data_loader.data_helper as helper
import utils.config
import torch
from torch_geometric.datasets import Planetoid
import numpy as np
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.hyperparams.batch_size
        self.is_qm9 = self.config.dataset_name == 'QM9'
        self.labels_dtype = torch.float32 if self.is_qm9 else torch.long

        if self.config.dataset_name in ['cora','pubmed','physics','dblp','cs','squirral','chameleon','texas','citeseer']:
            self.load_UGC_datasets(self.config.dataset_name, self.config)
        else:
            self.load_data()
            
    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        if self.is_qm9:
            self.load_qm9_data()
        else:
            self.load_data_benchmark()

        self.split_val_test_to_batches()


    def load_UGC_datasets(self, dataset_name, config):
        if dataset_name == 'cora':
            dataset = Planetoid(root = 'data/Cora', name = 'Cora')
        
        elif dataset_name == 'pubmed':
            dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')

        elif dataset_name == 'physics':
            dataset = Coauthor(root = 'data/Physics', name = 'Physics')
        
        elif dataset_name == 'dblp':
            dataset = CitationFull(root = 'newdata/DBLP', name = 'DBLP')

        elif dataset_name == 'cs':
            dataset = Coauthor(root = 'data/CS', name = 'CS')

        elif dataset_name == 'citeseer':
            dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer')

        train_rate = 0.6
        val_rate = 0.2
        
        data = dataset[0]
        percls_trn = int(round(train_rate*len(data.y)/config.num_classes))
        val_lb = int(round(val_rate*len(data.y)))

        def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
            index=[i for i in range(0,data.y.shape[0])]
            train_idx=[]
            rnd_state = np.random.RandomState(seed)
            for c in range(num_classes):
                class_idx = np.where(data.y.cpu() == c)[0]
                if len(class_idx)<percls_trn:
                    train_idx.extend(class_idx)
                else:
                    train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
            rest_index = [i for i in index if i not in train_idx]
            val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
            test_idx=[i for i in rest_index if i not in val_idx]

            def index_to_mask(index, size):
                mask = torch.zeros(size, dtype=torch.bool)
                mask[index] = True
                return mask

            data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
            data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
            data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
            
            return data
            
        data = random_splits(data, config.num_classes, percls_trn, val_lb)

        self.train_graphs, self.train_labels = data.x[data.train_mask], data.y[data.train_mask]
        self.test_graphs, self.test_labels = data.x[data.test_mask], data.y[data.test_mask]
        self.val_graphs, self.val_labels = data.x[data.val_mask], data.y[data.val_mask]

        self.train_size = self.train_graphs.shape[0]
        self.val_size = self.val_graphs.shape[0]
        self.test_size = self.test_graphs.shape[0]
        self.labels_std = 0

    # load QM9 data set
    def load_qm9_data(self):
        train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = \
            helper.load_qm9(self.config.target_param)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs, self.train_labels = train_graphs, train_labels
        self.val_graphs, self.val_labels = val_graphs, val_labels
        self.test_graphs, self.test_labels = test_graphs, test_labels

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std  # Needed for postprocess, multiply mean abs distance by this std

    # load data for a benchmark graph (COLLAB, NCI1, NCI109, MUTAG, PTC, IMDBBINARY, IMDBMULTI, PROTEINS)
    def load_data_benchmark(self):
        graphs, labels = helper.load_dataset(self.config.dataset_name)
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(graphs, labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
        # change validation graphs to the right shape
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        graphs, labels = next(self.iter)
        graphs, labels = torch.FloatTensor(graphs), torch.tensor(labels, device='cpu', dtype=self.labels_dtype)
        return graphs, labels

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches)
        elif what_set == 'test':
            self.iter = zip(self.test_graphs_batches, self.test_labels_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
        graphs, labels = helper.shuffle_same_size(graphs, labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = helper.shuffle(graphs, labels)
        self.iter = zip(graphs, labels)

    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels = helper.group_same_size(self.val_graphs, self.val_labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_labels_batches = graphs, labels

        if self.is_qm9:
            # Benchmark graphs have no test sets
            graphs, labels = helper.group_same_size(self.test_graphs, self.test_labels)
            graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
            self.num_iterations_test = len(graphs)
            self.test_graphs_batches, self.test_labels_batches = graphs, labels


if __name__ == '__main__':
    config = utils.config.process_config('../configs/10fold_config.json')
    data = DataGenerator(config)
    data.initialize('train')


