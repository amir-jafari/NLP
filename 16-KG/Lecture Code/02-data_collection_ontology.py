#%% --------------------------------------------------------------------------------------------------------------------
import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
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
# 2) Build ontology-based KG
EX = Namespace('http://example.org/')
g = Graph()
# Define classes
g.add((EX.Entity, RDF.type, RDFS.Class))
g.add((EX.Person, RDF.type, RDFS.Class))
g.add((EX.Company, RDF.type, RDFS.Class))
#%% --------------------------------------------------------------------------------------------------------------------
# 3) Materialize triples
for _, row in df.iterrows():
    uri = URIRef(EX[row['name'].replace(' ', '_')])
    g.add((uri, RDF.type, EX[row['type']]))
    g.add((uri, RDFS.label, Literal(row['name'])))
#%% --------------------------------------------------------------------------------------------------------------------
# 4) Serialize to Turtle
output = 'kg_data.ttl'
g.serialize(destination=output, format='turtle')
print(f"Knowledge graph written to {output}")