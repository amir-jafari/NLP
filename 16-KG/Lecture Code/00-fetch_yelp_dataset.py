from datasets import load_dataset

dataset = load_dataset(
  "yelp_polarity",
  data_dir="./yelp_polarity"
)
train = dataset["train"]
test  = dataset["test"]
print(len(train), len(test))
