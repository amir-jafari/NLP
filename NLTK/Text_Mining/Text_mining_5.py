# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Reading Data====> URL------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from bs4 import BeautifulSoup
from urllib2 import urlopen
import nltk
url = "http://google.com"

raw_html = urlopen(url).read()
raw_html[:250]


parsed_html = BeautifulSoup(raw_html, 'html.parser')
body = parsed_html.body

#==================================================================================
#%%
# Working with HTML

body('h1')
[h1.get_text() for h1 in body('h1')]

scripts = body('script')
scripts[2]
dummy= [s.extract() for s in body('script')] # remove the script tags from body

body_text = body.get_text()

body_tokens = nltk.word_tokenize(body_text)

body_tokens[0:100]


from nltk import word_tokenize
pars = [p.get_text() for p in body('p')]
par_tokens = word_tokenize('\n'.join(pars))


pars_nltk = nltk.Text(par_tokens)
pars_nltk.concordance('work')

pars_nltk[:10]
pars_nltk[-20:]

#==================================================================================
# from urllib import request     ## Python 3.5
from urllib2 import urlopen
from nltk.tokenize import word_tokenize
import nltk

url = "https://moz.com/robots.txt"
# response = request.urlopen(url)   ## Python 3.5
response = urlopen(url)
raw = response.read().decode('utf8')

print(type(raw))
print(len(raw))
print(raw)
print(raw.find("$"))


tokens = word_tokenize(raw)
print(tokens)
print(tokens[:15])

text = nltk.Text(tokens)
print(text[1:15])
print(text.count('$'))
print(text.collocations())

text.plot()

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Reading Data====> HTML-----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

url = "http://www.bbc.com/news/technology"
# html = request.urlopen(url).read().decode('utf8')  ## Python 3.5
html = urlopen(url).read().decode('utf8')  ## Python 3.5
html[:60]

from bs4 import BeautifulSoup
raw = BeautifulSoup(html, "html5lib").get_text()
tokens = word_tokenize(raw)
print(tokens)

tokens = tokens[:16]
text = nltk.Text(tokens)
text.concordance(',')

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Reading Data====> RSS Feeds-----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# sudo pip install pip feedparser
import feedparser

llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']

len(llog.entries)
post = llog.entries[2]

print(post.title)

content = post.content[0].value
content[:70]

raw = BeautifulSoup(content, "html5lib").get_text()
tokens =word_tokenize(raw)
print(tokens)

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Reading Data====> Local file-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import os
# os.chdir('D:')

f = open('document.txt', 'rU')
raw = f.read()



for line in f:
    print(line.strip())


'''
'r'       open for reading (default)
'w'       open for writing, truncating the file first
'x'       create a new file and open it for writing
'a'       open for writing, appending to the end of the file if it exists
'b'       binary mode
't'       text mode (default)
'+'       open a disk file for updating (reading and writing)
'U'       universal newline mode (deprecated, doesn't exist anymore)
'''

