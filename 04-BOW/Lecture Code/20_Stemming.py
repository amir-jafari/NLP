import re
def stem(phrase):
    return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)
                     [0][0].strip("'") for word in phrase.lower().split()])
print(stem("Doctor House's calls"))
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(' '.join([stemmer.stem(w).strip("'")
                for w in "dish washer's washed dishes".split()]))