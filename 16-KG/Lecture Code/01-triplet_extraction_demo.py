import spacy
nlp = spacy.load("en_core_web_sm")
#%% --------------------------------------------------------------------------------------------------------------------
examples = [
    "IBM is a multinational technology company.",
    "Alice, who lives in Seattle, works at Microsoft.",
    "Barack Obama was born in Honolulu.",
    "Google acquired YouTube in 2006.",
]
#%% --------------------------------------------------------------------------------------------------------------------
for sentence in examples:
    print(f"Sentence: {sentence}")
    doc = nlp(sentence)
    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ("VERB", "AUX"):
                subs = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objs = [w for w in token.rights if w.dep_ in ("dobj", "attr", "oprd")]
                for prep in [w for w in token.rights if w.dep_ == "prep"]:
                    objs.extend([w for w in prep.rights if w.dep_ == "pobj"]);
                for subj in subs:
                    subj_span = " ".join([t.text for t in subj.subtree])
                    for obj in objs:
                        obj_span = " ".join([t.text for t in obj.subtree])
                        pred = token.lemma_
                        triples.append((subj_span, pred, obj_span))
    if triples:
        for s, p, o in triples:
            print(f"  - ({s}, {p}, {o})")
    else:
        print("  - No triplets found.")
    print()