import spacy
import random
import re
import os
import pandas as pd

def cleaning(s):
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    return s

def create_dataset(df):
    nlp = spacy.load("en_core_web_sm")
    stopwords = nlp.Defaults.stop_words
    keywords = []
    # df_new = df.copy()  # Create copy of DataFrame
    for i in range(len(df)):
        k = 0
        title = df['product_title'][i]
        reviews = df['review_body'][i]
        doc = nlp(title)
        doc2 = nlp(reviews)
        text = cleaning(doc2)
        doc3 = nlp(text)
        # noun_title = ', '.join(nouns_list)
        nouns = doc.noun_chunks
        nouns_list = [str(x) for x in nouns]

        reviews_keywords = []
        for token in doc2:
            if (token.pos_ == 'ADJ' or token.pos_ == 'ADV' or token.pos_ == 'VER') and (str(token) not in stopwords):
                reviews_keywords.append(str(token))

        noun_adj_adv = ", ".join(nouns_list + reviews_keywords)
        keywords.append(noun_adj_adv)
        k += 1

    df['keywords'] = keywords

    return df


def create_dataset_random(df):

    nlp = spacy.load("en_core_web_sm")
    stopwords = nlp.Defaults.stop_words
    punctuation = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&', '-', '...', ' ',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']

    keywords = []
    df_new = df.copy()  # Create copy of DataFrame
    for i in range(len(df)):

        title = df['product_title'][i]
        reviews = df['review_body'][i]
        doc = nlp(title)
        doc2 = nlp(reviews)
        text = cleaning(doc2)
        doc3 = nlp(text)
        # noun_title = ', '.join(nouns_list)

        k = 0
        while k <= 42:
            if random.randint(0, 9) <= 3:
                nouns = doc.noun_chunks
                nouns_list = [str(x) for x in nouns]
            else:
                nouns_all = []
                for token1 in doc:
                    if token1.pos_ == 'PROPN':
                        nouns_all.append(str(token1))
                if len(nouns_all) <=1:
                    j = len(nouns_all)
                else:
                    j = random.randint(1, len(nouns_all))
                nouns_list = random.sample(nouns_all, j)

            reviews_keywords = []
            if k != 0:
                x1 = df.iloc[i, :]
                df_new.loc[42*i+k-0.5] = x1.to_list()
                df_new = df_new.sort_index().reset_index(drop=True)
            # n = random.randint(len(doc3) // 5, len(doc3) // 3)
            n = k//6 + 1
            samples = random.sample(list(doc3), n)
            for token in samples:
                if str(token) not in stopwords and str(token) not in punctuation:
                    reviews_keywords.append(str(token))

            noun_adj_adv = ", ".join(nouns_list + reviews_keywords)
            keywords.append(noun_adj_adv)
            k += 1
        if i % 10 == 0:
            print(f"Porgress:  {i/len(df)}")

    df_new['keywords'] = keywords
    df_fin = df_new.dropna(subset=['product_category', 'product_title', 'review_body', 'keywords'])

    return df_fin


if __name__=='__main__':
    PATH = os.getcwd()
    df = pd.read_csv(os.path.join(PATH, "../../../Data/Amazon_Review_200.csv"), encoding="ISO-8859-1")
    # df2 = create_dataset(df)
    df2 = create_dataset_random(df)
    df2.to_csv(os.path.join(PATH, "../../../Data/Amazon_Review_200_KW.csv"))
