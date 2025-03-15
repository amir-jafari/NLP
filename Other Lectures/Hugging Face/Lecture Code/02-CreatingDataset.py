#%% --------------------------------------------------------------------------------------------------------------------
from datasets import Dataset
import pandas as pd

#%% --------------------------------------------------------------------------------------------------------------------
my_dict = {"text": ["I love apples", "This is so bad"], "label": [1, 0]}

#%% --------------------------------------------------------------------------------------------------------------------
dataset = Dataset.from_dict(my_dict)
df = dataset.to_pandas()
df.to_csv("my_dataset.txt", sep="\t", index=False)
print('Your dataset has been created and stored!')