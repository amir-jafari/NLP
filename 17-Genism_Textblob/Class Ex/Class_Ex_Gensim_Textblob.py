#=======================================================================================================================
# Q1: Gensim Dictionary & Corpus
#-----------------------------------------------------------------------------------------------------------------------
# Question:
# You have these three documents:
#   docs = [
#     "the cat sat on the mat",
#     "the dog chased the cat",
#     "the dog and cat are friends"
#   ]
# Explain how to build a Gensim Dictionary and a Bag-of-Words corpus from docs.
#
# Hint:
# 1. Lower-case and split each document into tokens.
# 2. Use corpora.Dictionary to map each unique token to an ID.
# 3. Convert each token list to a list of (token_id, count) pairs with doc2bow.
#
# Solution:
print('\n')
print('====================================================Q1 Solution================================================')



#===============================================================================
# Q2: Training & Querying an LDA Model
#-------------------------------------------------------------------------------
# Question:
# Using the same bow_corpus and dictionary from Q1, train a 2-topic LDA model,
# then infer the topic distribution of the sentence "dog and cat on mat".
#
# Hint:
# 1. Import and instantiate LdaModel with num_topics=2.
# 2. Call .get_document_topics() on the BoW of the new sentence.
#
# Solution:
print('\n')
print('====================================================Q2 Solution================================================')



#===============================================================================
# Q3: TextBlob Sentiment & Noun Phrases
#-------------------------------------------------------------------------------
# Question:
# Create a TextBlob for the sentence:
#   "The new smartphone release is stunning but the battery life is disappointing."
# and extract:
# 1. Its sentiment polarity and subjectivity
# 2. The list of noun phrases
#
# Hint:
# 1. Use TextBlob(text).sentiment
# 2. Use TextBlob(text).noun_phrases
#
# Solution:
print('\n')
print('====================================================Q3 Solution================================================')



#===============================================================================
# Q4: TextBlob Lemmatization & Spelling Correction
#-------------------------------------------------------------------------------
# Question:
# Given the misspelled text:
#   "Shee is runninng awsome feats in rennig trackk."
# 1. Correct the spelling
# 2. Lemmatize the word "runninng"
#
# Hint:
# 1. Wrap the string in a TextBlob and call .correct().
# 2. Use Word("runninng").lemmatize().
#
# Solution:
print('\n')
print('====================================================Q4 Solution================================================')



#===============================================================================
# Q5: Comparing Similarity Measures
#-------------------------------------------------------------------------------
# Question:
# Compute and compare:
# - Jaccard similarity between token sets of
#     "machine learning is fun" and "learning machines are fun"
# - Cosine similarity between their TF–IDF vectors using Gensim
#
# Hint:
# 1. For Jaccard: use set(TextBlob(...).words).
# 2. For TF-IDF: build dictionary, bow, TfidfModel, convert to dense, then cosine.
#
# Solution:
print('\n')
print('====================================================Q5 Solution================================================')

