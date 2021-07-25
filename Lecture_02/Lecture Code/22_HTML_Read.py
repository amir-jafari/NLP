from urllib import request
from bs4 import BeautifulSoup
from nltk import word_tokenize
import nltk

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
print(html[:60])


raw = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(raw); print(tokens)
print(tokens = tokens[110:390])
text = nltk.Text(tokens); print(text)
print(text.concordance('gene'))