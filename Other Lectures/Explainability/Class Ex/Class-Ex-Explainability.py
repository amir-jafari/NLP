import pandas as pd

# A sample data
amazon_data = [
    ("This product is amazing! I love it.", 1),
    ("Terrible experience, would not buy again.", 0),
    ("Highly recommend this item to everyone.", 1),
    ("Not worth the money, very disappointed.", 0),
    ("Fantastic quality and fast shipping!", 1),
    ("Waste of money, worst purchase ever.", 0),
]

df = pd.DataFrame(amazon_data, columns=["review", "label"])

# =================================================================
# Class_Ex1:
# Load the above sample dataset and split it to train:test = 7:3
# ----------------------------------------------------------------
print(20*'-' + ' Begin Q1 ' + 20*'-')





print(20*'-' + ' End Q1 ' + 20*'-', "\n")

# =================================================================
# Class_Ex2:
# Build a simple text classification pipeline using TF-IDF and Logistic Regression.
# ----------------------------------------------------------------
print(20*'-' + ' Begin Q2 ' + 20*'-')



print(20*'-' + ' End Q2 ' + 20*'-', "\n")

# =================================================================
# Class_Ex3:
# Analyze feature importance using SHAP.
# ----------------------------------------------------------------
print(20*'-' + ' Begin Q3 ' + 20*'-')



print(20*'-' + ' End Q3 ' + 20*'-', "\n")

# =================================================================
# Class_Ex4:
# Use LIME to explain a sample prediction.
# ----------------------------------------------------------------
print(20*'-' + ' Begin Q4 ' + 20*'-')



print(20*'-' + ' End Q4 ' + 20*'-')
