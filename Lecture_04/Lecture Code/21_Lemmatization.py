import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("better", pos="n"))