import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import pandas

'''Sample graph data & Visualization'''
edge_index = torch.tensor([[0, 1, 0],  # From nodes (A -> B, B -> C, A -> C)
                                [1, 2, 2]],     # To nodes
                          dtype=torch.long)
node_labels = ["A", "B", "C"]

graph_data = Data(edge_index=edge_index)

# Convert to NetworkX for visualization
G = nx.DiGraph()
G.add_edges_from(edge_index.t().tolist())

plt.figure(figsize=(4, 3))
pos = nx.circular_layout(G)  # Automatically adjust nodes location
nx.draw(G, pos,
        with_labels=True, labels={i: node_labels[i] for i in range(len(node_labels))},
        node_color='lavender',
        edge_color='black',
        node_size=2000,
        font_size=12, font_weight='bold', arrows=True)
plt.show()

#%%
# edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],  # Source nodes
#                            [1, 0, 2, 1, 3, 2]], dtype=torch.long)  # Target nodes
#
# # Node features (4 nodes, 3 features each)
# x = torch.tensor([[1, 0, 1],
#                   [0, 1, 1],
#                   [1, 1, 0],
#                   [0, 0, 1]], dtype=torch.float)
# # Define a simple GCN model
# class GNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)  # First message-passing layer
#         self.conv2 = GCNConv(hidden_channels, out_channels)  # Second layer
#
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)  # Activation function
#         x = self.conv2(x, edge_index)
#
#         # Node embeddings are the learned feature representations
#         node_embeddings = x.clone()
#
#         # Graph aggregation (mean pooling over all nodes)
#         graph_embedding = global_mean_pool(x, batch)
#
#         return node_embeddings, graph_embedding
#
#
# # Create a batch tensor (all nodes belong to the same graph)
# batch = torch.zeros(x.shape[0], dtype=torch.long)
#
# data = Data(x=x, edge_index=edge_index)  # Create a PyG graph data object
#
# # Initialize and run the model
# model = GNN(in_channels=3, hidden_channels=4, out_channels=2)
# node_embeddings, graph_embedding = model(data.x, data.edge_index, batch)
#
# print("Node Embeddings:")
# print(node_embeddings)
# print("\nGraph Embedding:")
# print(graph_embedding)
