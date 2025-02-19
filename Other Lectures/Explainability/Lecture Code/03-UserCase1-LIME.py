# ==========================================================================
#
# User Case 1 - Sentiment Analysis
#
# ==========================================================================

# ==========================================================================
# The dataset 'IMDb' is used in this user case. Small Part of 'IMDb' is selected
# for sentiment Analysis, and the goal is to classify reviews as positive or
# negative.
# ==========================================================================

import pandas as pd

df_train = pd.read_csv('imdb_train.csv')
df_test = pd.read_csv('imdb_test.csv')


