import nltk
from nltk.corpus import stopwords

# Make sure you have downloaded the nltk stopwords corpus:
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

with open('stopwords.txt', 'w', encoding='utf-8') as f:
    for word in sorted(stop_words):
        f.write(word + '\n')

print("stopwords.txt created with nltk stopwords.")
