from nltk.tokenize.casual import casual_tokenize
message = "RT @TJMonticello Best day everrrrrrr at Monticello." \
          "Awesommmmmmeeeeeeee day :*)"
print(casual_tokenize(message))
print(casual_tokenize(message, reduce_len=True, strip_handles=True))