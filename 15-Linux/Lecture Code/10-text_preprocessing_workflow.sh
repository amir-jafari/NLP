#!/usr/bin/env bash

# If you do not install snowball, please install it before running the below code
# pip install snowball


# ==============================================================================
# 10-text_preprocessing_workflow.sh
# ==============================================================================
# ------------------------------------------------------------------------------
# 1. Counting Tokens
# ------------------------------------------------------------------------------
# Purpose: tally frequency of each word after cleaning (pipeline style)
# ------------------------------------------------------------------------------
cat small_corpus.txt | sort | uniq -c | sort -nr | head

# ------------------------------------------------------------------------------
# 2. Stop-word Removal
# ------------------------------------------------------------------------------
# Purpose: discard high-frequency function words (the, and, of, …)
# ------------------------------------------------------------------------------
cat small_corpus.txt | grep -F -v -w -f stopwords.txt

# ------------------------------------------------------------------------------
# 3. Case Folding
# ------------------------------------------------------------------------------
# Purpose: merge "The" and "the"
# Usage Example:
#   cat small_corpus.txt | tr '[:upper:]' '[:lower:]'
# ------------------------------------------------------------------------------
cat small_corpus.txt| tr '[:upper:]' '[:lower:]'

# ------------------------------------------------------------------------------
# 4. Punctuation Stripping
# ------------------------------------------------------------------------------
# Purpose: keep only letters/digits; split on others
# ------------------------------------------------------------------------------
cat small_corpus.txt | tr -c '[:alnum:]' '\n'

# ------------------------------------------------------------------------------
# 5. Stemming / Lemmatization
# ------------------------------------------------------------------------------
# Purpose: reduce words to a root form (running → run)
# ------------------------------------------------------------------------------
cat small_corpus.txt | stemwords english > stemmed.txt

# ------------------------------------------------------------------------------
# 6. n-gram Generation (Bigrams)
# ------------------------------------------------------------------------------
# Purpose: capture local context by pairing tokens
# Usage Example:
#   cat small_corpus.txt | paste -d' ' - -
# This turns:
#   A\nB\nC\nD\n
# into:
#   "A B"\n"C D"
# ------------------------------------------------------------------------------
cat small_corpus.txt | paste -d' ' - -

# ------------------------------------------------------------------------------
# 7. TF–IDF Weighting (Concept Only)
# ------------------------------------------------------------------------------
# Purpose: Term-Frequency x Inverse-Document-Frequency re-weights common vs. rare
#          terms. Typically computed with a library (Python scikit-learn, Spark).
# Shell pipeline stops at raw counts; then you would export to Python for TF-IDF.
# ------------------------------------------------------------------------------
echo "TF-IDF weighting usually done in Python or another library."
