import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#%% --------------------------------------------------------------------------------------------------------------------
from transformers import AutoModel, AutoTokenizer
import torch

def quantize_model(model):
    model.half()
    return model
#%% --------------------------------------------------------------------------------------------------------------------
model_name = 'distilbert-base-uncased'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = quantize_model(model)

#%% --------------------------------------------------------------------------------------------------------------------
input_text = "This is a test data for the NLP LLM and Agent Lecture."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
print(outputs)