#%% --------------------------------------------------------------------------------------------------------------------
import pandas as pd
import networkx as nx
#%% --------------------------------------------------------------------------------------------------------------------
# 1) Create sample data using inline DataFrame
data = [
    {'id': 1, 'name': 'Alice', 'type': 'Person'},
    {'id': 2, 'name': 'Bob', 'type': 'Person'},
    {'id': 3, 'name': 'Acme Corp', 'type': 'Company'}
]
df = pd.DataFrame(data)
print("Sample entities:")
print(df)
#%% --------------------------------------------------------------------------------------------------------------------
# 2) Build ontology-based KG using NetworkX
g = nx.DiGraph()
# Define classes
g.add_node("Entity", node_type="class")
g.add_node("Person", node_type="class")
g.add_node("Company", node_type="class")
# Add class hierarchy
g.add_edge("Person", "Entity", relation="subClassOf")
g.add_edge("Company", "Entity", relation="subClassOf")
#%% --------------------------------------------------------------------------------------------------------------------
# 3) Materialize instances
for _, row in df.iterrows():
    node_id = row['name'].replace(' ', '_')
    g.add_node(node_id, node_type="instance", instance_of=row['type'], label=row['name'])
    # Add type relationship
    g.add_edge(node_id, row['type'], relation="type")
#%% --------------------------------------------------------------------------------------------------------------------
# 4) Serialize to file
output = 'kg_data.txt'
with open(output, 'w') as f:
    f.write("# Knowledge Graph\n\n")
    f.write("## Classes\n")
    for node, attrs in g.nodes(data=True):
        if attrs.get('node_type') == 'class':
            f.write(f"{node}: {attrs}\n")
    f.write("\n## Instances\n")
    for node, attrs in g.nodes(data=True):
        if attrs.get('node_type') == 'instance':
            f.write(f"{node}: {attrs}\n")
    f.write("\n## Relationships\n")
    for u, v, attrs in g.edges(data=True):
        f.write(f"{u} --[{attrs.get('relation', '')}]--> {v}\n")
print(f"Knowledge graph written to {output}")