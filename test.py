import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import DataLoader
import numpy as np

# Load AIFB dataset
dataset = Entities(root='/tmp/Entities', name='AIFB')
#dataset = AIFB(root='/tmp/AIFB', transform=ToUndirected())
data = dataset[0]
num_nodes = data.num_nodes
if data.x is None:
    data.x = torch.eye(num_nodes)  # identity features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Define R-GCN model
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, use_activation=True):
        super(RGCN, self).__init__()
        self.use_activation = use_activation
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.classifier = torch.nn.Linear(out_channels, dataset.num_classes)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        if self.use_activation:
            x = F.relu(x)
        x = self.classifier(x)
        return x

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.cross_entropy(out[data.train_idx], data.y[data.train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type)
    pred = out.argmax(dim=1)

    accs = []
    for idx in [data.train_idx, data.test_idx]:
        correct = pred[idx].eq(data.y[idx]).sum().item()
        acc = correct / idx.size(0)
        accs.append(acc)
    return accs


# Run for both with and without activation
results = {}

for use_act in [True, False]:
    torch.manual_seed(42)
    model = RGCN(in_channels=data.x.size(1), out_channels=16,
                 num_relations=dataset.num_relations, use_activation=use_act).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        loss = train(model, optimizer, data)

    train_acc, test_acc = test(model, data)
    results["With Activation" if use_act else "Without Activation"] = {
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc
    }

# Print results
for key, value in results.items():
    print(f"\n=== {key} ===")
    print(f"Train Accuracy: {value['Train Accuracy']:.4f}")
    print(f"Test Accuracy : {value['Test Accuracy']:.4f}")
