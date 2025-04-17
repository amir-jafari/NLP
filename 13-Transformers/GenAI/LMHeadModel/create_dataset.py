
import pandas as pd
from utils import cleaning
import os
'''
https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners/notebook
'''


if __name__=='__main__':

    PATH = os.getcwd()
    df = pd.read_csv(os.path.join(PATH, "../../../../../Data/Amazon_Review_sm.csv"), encoding="ISO-8859-1")
    df = df.dropna()
    text_data = open(os.path.join(PATH, '../../../../../Data/Amazon_Review_sm.txt'), 'w')
    for idx, item in df.iterrows():
        article = cleaning(item["review_body"])
        text_data.write(article)
    text_data.close()