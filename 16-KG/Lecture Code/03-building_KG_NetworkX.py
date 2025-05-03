#%% --------------------------------------------------------------------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt
#%% --------------------------------------------------------------------------------------------------------------------
G = nx.DiGraph()
#%% --------------------------------------------------------------------------------------------------------------------
triples = [
    ("Barack Obama", "born_in", "Honolulu"),
    ("Barack Obama", "spouse_of", "Michelle Obama"),
    ("Barack Obama", "member_of", "Democratic Party"),
    ("Barack Obama", "studied_at", "Harvard Law School"),
    ("Michelle Obama", "born_in", "Chicago"),
    ("Michelle Obama", "studied_at", "Princeton University"),
    ("Democratic Party", "headquartered_in", "Washington D.C."),
    ("Harvard Law School", "located_in", "Cambridge"),
    ("Princeton University", "located_in", "Princeton"),
    ("Honolulu", "located_in", "Hawaii"),
    ("Hawaii", "is_state_of", "United States"),
    ("Cambridge", "located_in", "Massachusetts"),
    ("Princeton", "located_in", "New Jersey"),
    ("United States", "has_capital", "Washington D.C."),
]

for subj, rel, obj in triples:
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, label=rel)
#%% --------------------------------------------------------------------------------------------------------------------
print("Nodes:", list(G.nodes()))
print("Edges with labels:")
for u, v, data in G.edges(data=True):
    print(f"  {u} -[{data['label']}]-> {v}")
#%% --------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw_networkx_nodes(G, pos, node_size=1800, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, arrowsize=20, arrowstyle='-|>')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=9)
plt.title("Knowledge Graph (NetworkX)", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

