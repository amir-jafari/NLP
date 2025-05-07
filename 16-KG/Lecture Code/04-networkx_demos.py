#%% --------------------------------------------------------------------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
#%% --------------------------------------------------------------------------------------------------------------------
G_simple = nx.Graph()
G_simple.add_node("A")
G_simple.add_edge("A", "B")
G_simple.nodes["A"]["role"] = "start"
G_simple.edges["A", "B"]["weight"] = 5

G_dir = nx.DiGraph()
G_dir.add_edge("X", "Y")
G_dir.add_edge("Y", "X")
G_dir.nodes["X"]["type"] = "source"

G_multi = nx.MultiGraph()
G_multi.add_edge("U", "V", key="first", label="road")
G_multi.add_edge("U", "V", key="second", label="rail")

print("Simple nodes:", G_simple.nodes(data=True))
print("Directed edges:", G_dir.edges(data=True))
print("MultiGraph edges:", G_multi.edges(keys=True, data=True))
#%% --------------------------------------------------------------------------------------------------------------------
# BFS and DFS trees from a source node
bfs_tree = nx.bfs_tree(G_simple, source="A")
dfs_tree = nx.dfs_tree(G_simple, source="A")

print("BFS tree edges:", list(bfs_tree.edges()))
print("DFS tree edges:", list(dfs_tree.edges()))

deg_cent = nx.degree_centrality(G_simple)
bet_cent = nx.betweenness_centrality(G_simple)
clo_cent = nx.closeness_centrality(G_simple)

print("Degree centrality:", deg_cent)
print("Betweenness centrality:", bet_cent)
print("Closeness centrality:", clo_cent)

# Shortest path (Dijkstra)
G_weighted = nx.Graph()
G_weighted.add_edge("A", "B", weight=2)
G_weighted.add_edge("B", "C", weight=1)
G_weighted.add_edge("A", "C", weight=5)
path = nx.shortest_path(G_weighted, source="A", target="C", weight="weight")
dist = nx.shortest_path_length(G_weighted, source="A", target="C", weight="weight")
print("Shortest path A→C:", path, "with distance", dist)
#%% --------------------------------------------------------------------------------------------------------------------
# Interactive Visualization
# a) matplotlib static
plt.figure(figsize=(4,4))
nx.draw(G_simple, with_labels=True, node_color="lightblue", edge_color="gray")
plt.title("Simple Graph")
plt.show()

# b) PyVis for browser interactivity
net = Network(height="400px", width="50%", bgcolor="#ffffff", font_color="black")
net.from_nx(G_simple)
net.write_html("networkx_simple.html")
print("Generated networkx_simple.html")