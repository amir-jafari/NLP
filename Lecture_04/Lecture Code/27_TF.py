from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
tokenizer = TreebankWordTokenizer()
with open('kite.txt' , 'r') as f:
    kite_text = f.read()
with open('kite_history.txt' , 'r', encoding='utf-8') as f:
    kite_history = f.read()

kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)

kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)

intro_total = len(intro_tokens) ; print(intro_total)
history_total = len(history_tokens) ; print(history_total)

intro_tf = {};history_tf = {}

intro_counts = Counter(intro_tokens)
history_counts = Counter(history_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
print('Term Frequency of "kite" in intro is: {:.4f}'.format(intro_tf['kite']))

history_tf['kite'] = history_counts['kite'] / history_total
print('Term Frequency of "kite" in history is: {:.4f}'.format(history_tf['kite']))

intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
print('Term Frequency of "and" in history is: {:.4f}'.format(history_tf['and']))
