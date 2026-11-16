from datasets import load_dataset
import pandas as pd
'''
download data: the full Dolly-15k instruction dataset (instruction, context, response, category)
'''
dataset = load_dataset('databricks/databricks-dolly-15k')
df = pd.DataFrame(dataset["train"])
df.to_csv("../Data/dolly-15k_train.csv", index=False)
print(df.shape, "-> ../Data/dolly-15k_train.csv   (the course uses the 200-row sample ../Data/dolly-200.csv)")
'''
save response as csv file
'''
# PATH = os.getcwd()
# df = pd.read_csv(os.path.join(PATH, "../Data/dolly-15k_train.csv"), encoding="ISO-8859-1")
#
# text_data = open(os.path.join(PATH, '../../../../../../Data/dolly-15k_train.txt'), 'w')
#
# for idx, item in df.iterrows():
#     article = item["response"]
#     text_data.write(article)
# text_data.close()

#
# import torch
# from transformers import pipeline
#
# generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
# res = generate_text("Explain to me the difference between nuclear fission and fusion.")
# print(res[0]["generated_text"])