import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, f_oneway
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Planetoid

# # dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')

# dataset = Planetoid(root = 'data/Cora', name = 'Cora')


# feature_matrix = dataset[0].x
# label_matrix = dataset[0].y

# df = pd.DataFrame(feature_matrix)
# df['label'] = label_matrix

# # Calculate statistics
# grouped = df.groupby('label').mean()

# print("Mean values per label:\n", grouped)

# # # Visualize distributions
# for i in range(feature_matrix.shape[1]):
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data=df, x=i, hue='label')
#     plt.title(f'Distribution of Feature {i} by Label')
#     plt.show()

# # Perform statistical tests
# labels = df['label'].unique()
# # for i in range(feature_matrix.shape[1]):
# #     feature = df[i]
# #     print(f"Feature {i}:")
# #     for j in range(len(labels)):
# #         for k in range(j + 1, len(labels)):
# #             label1 = labels[j]
# #             label2 = labels[k]
# #             stat, p = ks_2samp(feature[df['label'] == label1], feature[df['label'] == label2])
# #             print(f"KS test between label {label1} and label {label2}: p-value = {p}")

# # ANOVA test
# for i in range(feature_matrix.shape[1]):
#     feature = df[i]
#     groups = [feature[df['label'] == label] for label in labels]
#     stat, p = f_oneway(*groups)
#     print(f"ANOVA for Feature {i}: p-value = {p}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

# Sample data
# np.random.seed(0)
# feature_matrix = np.random.rand(100, 10)  # 100 samples, 10 features
# label_matrix = np.random.randint(0, 3, 100)  # 3 labels

# dataset = Planetoid(root = 'data/Cora', name = 'Cora')


# feature_matrix = dataset[0].x
# label_matrix = dataset[0].y

# # Create a DataFrame for easier manipulation
# df = pd.DataFrame(feature_matrix)
# df['label'] = label_matrix

# def sample_subset(df, n_per_class):
#     sampled_df = df.groupby('label').apply(lambda x: x.sample(n_per_class)).reset_index(drop=True)
#     return sampled_df

# # Sample subset
# n_per_class = 100  # Number of points to sample per class
# df = sample_subset(df, n_per_class)

# # PCA for 2D visualization
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(feature_matrix)
# df['pca-one'] = pca_result[:, 0]
# df['pca-two'] = pca_result[:, 1]

# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="label",
#     palette=sns.color_palette("hsv", len(np.unique(label_matrix))),
#     data=df,
#     legend="full",
#     alpha=0.7
# )
# plt.title('PCA')
# plt.show()

# # t-SNE for 2D visualization
# tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
# tsne_result = tsne.fit_transform(feature_matrix)
# df['tsne-one'] = tsne_result[:, 0]
# df['tsne-two'] = tsne_result[:, 1]

# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="tsne-one", y="tsne-two",
#     hue="label",
#     palette=sns.color_palette("hsv", len(np.unique(label_matrix))),
#     data=df,
#     legend="full",
#     alpha=0.7
# )
# plt.title('t-SNE')
# plt.show()

# Calculate pairwise distances and compare intra-class and inter-class distances
# distance_matrix = pairwise_distances(df.iloc[:, :-1])
# labels = df['label'].unique()

# intra_class_distances = []
# inter_class_distances = []

# for label in labels:
#     same_class_mask = df['label'] == label
#     other_class_mask = ~same_class_mask

#     same_class_distances = distance_matrix[same_class_mask][:, same_class_mask]
#     other_class_distances = distance_matrix[same_class_mask][:, other_class_mask]

#     intra_class_distances.extend(same_class_distances[np.triu_indices(same_class_distances.shape[0], 1)])
#     inter_class_distances.extend(other_class_distances.flatten())

# # Plot histograms of intra-class and inter-class distances
# plt.figure(figsize=(10, 5))
# sns.histplot(intra_class_distances, color='blue', label='Intra-class', kde=True)
# sns.histplot(inter_class_distances, color='red', label='Inter-class', kde=True)
# plt.legend()
# plt.title('Intra-class vs. Inter-class Distances')
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import torch

# dataset = Planetoid(root = 'data/Cora', name = 'Cora')
# dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')
# data = dataset[0]

# file_edge_index = 'heterophlic_data/edge_index_squirrel.pt'
# file_label = 'heterophlic_data/label_squirrel.pt'
# node_feat = 'heterophlic_data/node_feat_squirrel.pt'

file_edge_index = 'heterophlic_data/edge_index_chameleon.pt'
file_label = 'heterophlic_data/label_chameleon.pt'
node_feat = r"heterophlic_data\node_feat_cameleon.pt"

hetro_edge_index = torch.load(file_edge_index)
hetro_label = torch.from_numpy(torch.load(file_label)).long()
hetro_node_feat = torch.load(node_feat)

data = Data(x=hetro_node_feat, edge_index = hetro_edge_index, y = hetro_label)


feature_matrix = data.x
label_matrix = data.y


# # Create a DataFrame for easier manipulation
# df = pd.DataFrame(feature_matrix)
# df['label'] = label_matrix

# # Function to sample subset of points from each class
# def sample_subset(df, n_per_class):
#     sampled_df = df.groupby('label').apply(lambda x: x.sample(n_per_class)).reset_index(drop=True)
#     return sampled_df

# # Sample subset
# n_per_class = 100  # Number of points to sample per class
# sampled_df = sample_subset(df, n_per_class)

# # Calculate pairwise distances
# distance_matrix = pairwise_distances(sampled_df.iloc[:, :-1])

# # Separate intra-class and inter-class distances
# labels = sampled_df['label'].values
# intra_class_distances = []
# inter_class_distances = []

# for label in np.unique(labels):
#     same_class_mask = labels == label
#     other_class_mask = ~same_class_mask

#     same_class_distances = distance_matrix[same_class_mask][:, same_class_mask]
#     other_class_distances = distance_matrix[same_class_mask][:, other_class_mask]

#     intra_class_distances.extend(same_class_distances[np.triu_indices(same_class_distances.shape[0], 1)])
#     inter_class_distances.extend(other_class_distances.flatten())

# # Plot histograms of intra-class and inter-class distances
# plt.figure(figsize=(10, 5))
# sns.histplot(intra_class_distances, color='blue', label='Intra-class', kde=True)
# sns.histplot(inter_class_distances, color='red', label='Inter-class', kde=True)
# plt.legend()
# plt.title('pubmed')
# plt.savefig("results_and_plots/intra_inter_distances_pubmed")
# plt.show()



# Create a DataFrame for easier manipulation
df = pd.DataFrame(feature_matrix)
df['label'] = label_matrix

# Function to sample subset of points from each class
def sample_subset(df, n_per_class):
    sampled_df = df.groupby('label').apply(lambda x: x.sample(n_per_class)).reset_index(drop=True)
    return sampled_df

# Sample subset
n_per_class = 100  # Number of points to sample per class
sampled_df = sample_subset(df, n_per_class)

# Calculate pairwise distances
distance_matrix = pairwise_distances(sampled_df.iloc[:, :-1])

# Separate intra-class and inter-class distances
labels = sampled_df['label'].values
intra_class_distances = []
inter_class_distances = []

for label in np.unique(labels):
    same_class_mask = labels == label
    other_class_mask = ~same_class_mask

    same_class_distances = distance_matrix[same_class_mask][:, same_class_mask]
    other_class_distances = distance_matrix[same_class_mask][:, other_class_mask]

    intra_class_distances.extend(same_class_distances[np.triu_indices(same_class_distances.shape[0], 1)])
    inter_class_distances.extend(other_class_distances.flatten())

# Plot distances
plt.figure(figsize=(12, 6))
plt.plot(intra_class_distances, label='Intra-class Distances', linestyle='none', marker='o', color='blue', alpha=0.5)
plt.plot(inter_class_distances, label='Inter-class Distances', linestyle='none', marker='x', color='red', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Distances for Intra-class and Inter-class')
plt.legend()
plt.show()

average_intra_class_distance = np.mean(intra_class_distances)
average_inter_class_distance = np.mean(inter_class_distances)

print(f"Average Intra-class Distance: {average_intra_class_distance:.4f}")
print(f"Average Inter-class Distance: {average_inter_class_distance:.4f}")
