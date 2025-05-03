#%% --------------------------------------------------------------------------------------------------------------------
triples = []
by_subject   = {}
by_predicate = {}
by_object    = {}
#%% --------------------------------------------------------------------------------------------------------------------
for subj, pred, obj in [
    ("Alice", "bornIn",  "London"),
    ("Alice", "knows",   "Bob"),
    ("Bob",   "bornIn",  "Paris"),
    ("Bob",   "worksAt", "Initech"),
]:
    t = (subj, pred, obj)
    if t in triples:
        continue
    triples.append(t)
    by_subject.setdefault(subj,   []).append(t)
    by_predicate.setdefault(pred, []).append(t)
    by_object.setdefault(obj,      []).append(t)
#%% --------------------------------------------------------------------------------------------------------------------
alice_facts = by_subject.get("Alice", [])
print("Alice →", alice_facts)
#%% --------------------------------------------------------------------------------------------------------------------
born_in_paris = [t for t in triples if t[1] == "bornIn" and t[2] == "Paris"]
print("Born in Paris →", born_in_paris)
#%% --------------------------------------------------------------------------------------------------------------------
print("All triples:")
for subj, pred, obj in triples:
    print(f"  ({subj}, {pred}, {obj})")
