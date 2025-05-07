#%% --------------------------------------------------------------------------------------------------------------------
import nltk
from nltk.corpus import wordnet as wn
import networkx as nx
import spacy
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
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
EX = Namespace("http://example.org/")
g = Graph()
# define classes and properties
g.add((EX.Person, RDF.type, RDFS.Class))
g.add((EX.worksAt, RDF.type, RDF.Property))
# add an instance
alice = URIRef(EX.Alice)
g.add((alice, RDF.type, EX.Person))
org = URIRef(EX.AcmeCorp)
g.add((alice, EX.worksAt, org))
g.add((org, RDFS.label, Literal("Acme Corporation")))
# serialize to Turtle
output_file = "kg_app.ttl"
g.serialize(destination=output_file, format="turtle")
print(f"4) Sample KG written to {output_file}")
