# ======================================================================================================================
# Shared Setup - dataset 'yelp_review_full'
# ----------------------------------------------------------------------------------------------------------------------
#%% --------------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BartTokenizer, BartForConditionalGeneration

yelp = load_dataset("yelp_review_full")
train_ds = yelp["train"].shuffle(seed=42).select(range(500))
test_ds = yelp["test"].shuffle(seed=42).select(range(100))

bert_tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
bert_model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=5)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def tokenize_fn(batch):
    return bert_tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

train_ds = train_ds.map(tokenize_fn, batched=True)
train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

#%%
# ======================================================================================================================
# Class_Ex1:
# How can you quickly fine-tune a tiny BERT model on AG_NEWS dataset?
# Step 1: Use 11-Hugging Face's Trainer API.
# Step 2: Train and evaluate the performance.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')



print(20*'-' + 'End Q1' + 20*'-')
# ======================================================================================================================
# Class_Ex2:
# How can you visualize the distribution of predicted classes using the fine-tuned BERT model?
# Step 1: Make predictions on test data
# Step 2: Plot a histogram of the predicted labels
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')



print(20*'-' + 'End Q2' + 20*'-')
# ======================================================================================================================
# Class_Ex3:
# How can you use BART model for text summarization on ag_news dataset?
# Step 1: Use the BartTokenizer and BartForConditionalGeneration
# Step 2: Generate summaries for some texts from the dataset
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')



print(20*'-' + 'End Q3' + 20*'-')
# ======================================================================================================================
# Class_Ex4:
# How to visualize attention weights for a BERT model to interpret its predictions?
# Step 1: Choose a single instance from test dataset.
# Step 2: Get attention weights from the model.
# Step 3: Visualize the attention matrix.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')



print(20*'-' + 'End Q4' + 20*'-')
# ======================================================================================================================
# Class_Ex5:
# How can you compare two pretrained tokenizers (e.g., BERT vs BART) in terms of their vocabulary overlap?
# Step 1: Load both tokenizer vocabularies
# Step 2: Compute the overlap of the tokens
# Step 3: Visualize or print out the percentage overlap
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')



print(20*'-' + 'End Q5' + 20*'-')