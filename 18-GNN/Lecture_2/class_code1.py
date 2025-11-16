import torch
from torch_geometric.data import Data

# Define node features and edges
x = torch.tensor([[1], [2], [3], [4]],
                 dtype=torch.float)
edge_index = torch.tensor([[0, 1, 2, 3],
                           [1, 0, 3, 2]],
                          dtype=torch.long)
# Create a graph data object
data = Data(x=x, edge_index=edge_index)
print(data)

# ---------------------
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Is undirected: {data.is_undirected()}')

# ---------------------
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Get the first graph

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of classes: {dataset.num_classes}')

# ---------------------
edge_attr = torch.tensor([[0.5], [0.8], [0.3], [1.0]],
                         dtype=torch.float)
data.edge_attr = edge_attr
print(data)

# ---------------------
from torch_geometric.utils import to_undirected

edge_index = to_undirected(edge_index)
print(edge_index)

# ---------------------
from torch_geometric.data import DataLoader

loader = DataLoader(dataset,
                    batch_size=32,
                    shuffle=True)
for batch in loader:
    print(batch)

# ---------------------
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='/tmp/Cora', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]
print(data.x[0])  # Normalized node features

# ---------------------
from torch_geometric.nn import MessagePassing

class CustomGNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


# ---------------------
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

input_dim = dataset.num_features  # 1433 for Cora
hidden_dim = 16
output_dim = dataset.num_classes  # 7 for Cora
model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')