from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

tokenizer = TreebankWordTokenizer()

with open('kite.txt' , 'r') as f:
    kite_text = f.read()
with open('kite_history.txt' , 'r', encoding='utf-8') as f:
    kite_history = f.read()

kite_intro = kite_text.lower();intro_tokens = tokenizer.tokenize(kite_intro)
kite_history = kite_history.lower();history_tokens = tokenizer.tokenize(kite_history)
num_docs_containing_and ,num_docs_containing_kite = 0, 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1
    if 'kite' in doc:
        num_docs_containing_kite += 1
intro_total = len(intro_tokens) ;history_total = len(history_tokens)
intro_counts = Counter(intro_tokens);history_counts = Counter(history_tokens)
intro_tf = {};history_tf = {};intro_tfidf = {};history_tfidf = {}
intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
intro_tf['kite'] = intro_counts['kite'] / intro_total
history_tf['kite'] = history_counts['kite'] / history_total
num_docs = 2;intro_idf = {};history_idf = {};num_docs = 2
intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and
intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
