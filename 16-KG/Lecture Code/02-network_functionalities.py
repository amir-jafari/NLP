#%% --------------------------------------------------------------------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt

G_undirected = nx.Graph()
G_directed = nx.DiGraph()
G_multi = nx.MultiGraph()

print("Created Graph (undirected):", G_undirected)
print("Created Graph (directed):", G_directed)
print("Created Graph (multi):", G_multi)

#%% --------------------------------------------------------------------------------------------------------------------
G = nx.Graph()

G.add_node("Alice")
G.add_node("Bob")
G.add_nodes_from(["Carol", "Dave"])

G.add_edge("Alice", "Bob")
edges_to_add = [("Bob", "Carol"), ("Carol", "Dave")]
G.add_edges_from(edges_to_add)

print("\n--- After Adding Nodes & Edges ---")
print("Nodes:", G.nodes())
print("Edges:", G.edges())

G.nodes["Alice"]["role"] = "Researcher"
G.nodes["Bob"]["role"] = "Engineer"
G["Bob"]["Carol"]["relationship"] = "coworkers"

print("\nNode data:", G.nodes(data=True))
print("Edge data:", G.edges(data=True))

#%% --------------------------------------------------------------------------------------------------------------------
print("\n--- Graph Algorithms Demonstration ---")

path = nx.shortest_path(G, source="Alice", target="Dave")
print("Shortest path from Alice to Dave:", path)

degree_cen = nx.degree_centrality(G)
print("Degree centrality:", degree_cen)

components = list(nx.connected_components(G))
print("Connected components:", components)

#%% --------------------------------------------------------------------------------------------------------------------
sub_nodes = ["Bob", "Carol"]
subG = G.subgraph(sub_nodes)

print("\n--- Subgraphs & Components ---")
print("Subgraph nodes:", subG.nodes())
print("Subgraph edges:", subG.edges())

sub_components = list(nx.connected_components(subG))
print("Connected components in subG:", sub_components)

#%% --------------------------------------------------------------------------------------------------------------------
print("\n--- Visualization Demonstration ---")

plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G)

nx.draw(
    G, pos,
    with_labels=True,
    node_size=800,
    node_color="lightblue",
    edge_color="gray"
)

plt.title("Simple Graph Visualization")
plt.axis("off")
plt.show()

#%% --------------------------------------------------------------------------------------------------------------------
print("\n--- Advanced Topics Demonstration ---")

temp_filename = "temp_adjlist.adj"
nx.write_adjlist(G, temp_filename)
print(f"Graph written to {temp_filename}")

loaded_graph = nx.read_adjlist(temp_filename)
print("Graph loaded from file. Nodes:", loaded_graph.nodes())
print("Edges:", loaded_graph.edges())

multi_dig = nx.MultiDiGraph()
multi_dig.add_node("X")
multi_dig.add_node("Y")
multi_dig.add_edge("X", "Y", key="link1", weight=1.0)
multi_dig.add_edge("X", "Y", key="link2", weight=2.0)

print("MultiDiGraph edges (parallel):", multi_dig.edges(keys=True, data=True))
