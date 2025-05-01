#!/usr/bin/env bash
# ==============================================================================
# 10-text_preprocessing_workflow.sh
# ==============================================================================
# -----------------------------------------------------------------------------
# 1. Counting Tokens
# ------------------------------------------------------------------------------
# Purpose: tally frequency of each word after cleaning (pipeline style)
# Usage Example:
#   cat tokens.txt | sort | uniq -c | sort -nr | head
# ------------------------------------------------------------------------------
sort | uniq -c | sort -nr | head

# ------------------------------------------------------------------------------
# 2. Stop-word Removal
# ------------------------------------------------------------------------------
# Purpose: discard high-frequency function words (the, and, of, …)
# Usage Example:
#   cat tokens.txt | grep -F -v -w -f stopwords.txt
# ------------------------------------------------------------------------------
grep -F -v -w -f stopwords.txt

# ------------------------------------------------------------------------------
# 3. Case Folding
# ------------------------------------------------------------------------------
# Purpose: merge "The" and "the"
# Usage Example:
#   cat input.txt | tr '[:upper:]' '[:lower:]'
# ------------------------------------------------------------------------------
tr '[:upper:]' '[:lower:]'

# ------------------------------------------------------------------------------
# 4. Punctuation Stripping
# ------------------------------------------------------------------------------
# Purpose: keep only letters/digits; split on others
# Usage Example:
#   cat input.txt | tr -c '[:alnum:]' '\n'
# ------------------------------------------------------------------------------
tr -c '[:alnum:]' '\n'

# ------------------------------------------------------------------------------
# 5. Stemming / Lemmatization
# ------------------------------------------------------------------------------
# Purpose: reduce words to a root form (running → run)
# Usage Example:
#   snowball english < tokens.txt > stemmed.txt
# ------------------------------------------------------------------------------
snowball english < tokens.txt > stemmed.txt

# ------------------------------------------------------------------------------
# 6. n-gram Generation (Bigrams)
# ------------------------------------------------------------------------------
# Purpose: capture local context by pairing tokens
# Usage Example:
#   cat tokens.txt | paste -d' ' - -
# This turns:
#   A\nB\nC\nD\n
# into:
#   "A B"\n"C D"
# ------------------------------------------------------------------------------
paste -d' ' - -

# ------------------------------------------------------------------------------
# 7. TF–IDF Weighting (Concept Only)
# ------------------------------------------------------------------------------
# Purpose: Term-Frequency x Inverse-Document-Frequency re-weights common vs. rare
#          terms. Typically computed with a library (Python scikit-learn, Spark).
# Shell pipeline stops at raw counts; then you would export to Python for TF-IDF.
# ------------------------------------------------------------------------------
echo "TF-IDF weighting usually done in Python or another library."
