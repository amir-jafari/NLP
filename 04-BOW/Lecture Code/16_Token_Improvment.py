import re
sentence ='Thomas Jefferson began building Monticello at theage of 26.'
tokens = re.split(r'[-\s.,;!?]+', sentence)
print(tokens)