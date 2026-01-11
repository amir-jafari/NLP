#%% --------------------------------------------------------------------------------------------------------------------
import nltk
from nltk.corpus import wordnet as wn
import networkx as nx
import spacy
#%% --------------------------------------------------------------------------------------------------------------------
nltk.download('wordnet')
syns = wn.synsets("apple")
lemmas = set(lemma.name() for s in syns for lemma in s.lemmas())
print("1) WordNet lemmas for 'apple':", lemmas)
#%% --------------------------------------------------------------------------------------------------------------------
G = nx.DiGraph()
# add users
G.add_node("user1", type="user")
G.add_node("user2", type="user")
# add items
G.add_node("itemA", type="item")
G.add_node("itemB", type="item")
# add interactions
G.add_edge("user1", "itemA")
G.add_edge("user1", "itemB")
G.add_edge("user2", "itemB")
ranks = nx.pagerank(G, alpha=0.85)
print("2) PageRank scores:", ranks)
#%% --------------------------------------------------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
doc = nlp("IBM acquired Red Hat in 2019.")
spo = []
for tok in doc:
    if tok.dep_ == "nsubj":
        subj = tok.text
        pred = tok.head.text
        objs = [child.text for child in tok.head.children
                if child.dep_ in ("dobj", "pobj", "attr")]
        for obj in objs:
            spo.append((subj, pred, obj))
print("3) Extracted SPO:", spo)
#%% --------------------------------------------------------------------------------------------------------------------
# Simple knowledge graph using NetworkX
kg = nx.DiGraph()
# add classes and instances
kg.add_node("Person", node_type="class")
kg.add_node("Organization", node_type="class")
kg.add_node("Alice", node_type="instance", instance_of="Person")
kg.add_node("AcmeCorp", node_type="instance", instance_of="Organization", label="Acme Corporation")
# add relationships
kg.add_edge("Alice", "AcmeCorp", relation="worksAt")
# serialize to simple format
output_file = "kg_app.txt"
with open(output_file, "w") as f:
    f.write("# Nodes\n")
    for node, attrs in kg.nodes(data=True):
        f.write(f"{node}: {attrs}\n")
    f.write("\n# Edges\n")
    for u, v, attrs in kg.edges(data=True):
        f.write(f"{u} -> {v}: {attrs}\n")
print(f"4) Sample KG written to {output_file}")
