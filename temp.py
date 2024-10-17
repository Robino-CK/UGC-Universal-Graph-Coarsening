import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx

# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Convert to NetworkX graph
G = to_networkx(data)

# Compute the Laplacian matrix
A = nx.adjacency_matrix(G)  # Get the adjacency matrix
D = np.diag(np.ravel(A.sum(axis=1)))  # Compute the degree matrix
L = D - A  # Compute the Laplacian matrix

print(L)

threshold = 1e-10
L[np.abs(L) < threshold] = 0


# Check if the Laplacian matrix is PSD
eigenvalues, _ = np.linalg.eig(L)
is_psd = np.all(eigenvalues >= 0)

if is_psd:
    print("Laplacian matrix is PSD")
else:
    print("Laplacian matrix is not PSD")

negative_eigenvalues = eigenvalues[eigenvalues < 0]
print("Eigenvalues less than 0:")
print(negative_eigenvalues)
