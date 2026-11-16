from datasets import load_dataset
import pandas as pd
'''
download data: the full Dolly-15k instruction dataset (instruction, context, response, category)
'''
dataset = load_dataset('databricks/databricks-dolly-15k')
df = pd.DataFrame(dataset["train"])
df.to_csv("../Data/dolly-15k_train.csv", index=False)
print(df.shape, "-> ../Data/dolly-15k_train.csv   (the course uses the 200-row sample ../Data/dolly-200.csv)")
