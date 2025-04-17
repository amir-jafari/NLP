#%% --------------------------------------------------------------------------------------------------------------------
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")
#%% --------------------------------------------------------------------------------------------------------------------
sentiment_pipeline = pipeline("sentiment-analysis")
sentiment_result = sentiment_pipeline("I love using 11-Hugging Face!")
print("Sentiment Analysis:", sentiment_result)

#%% --------------------------------------------------------------------------------------------------------------------
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zero_shot_result = zero_shot_pipeline("I enjoy painting portraits in my free time.", candidate_labels=["art", "sports", "politics"])
print("Zero-Shot Classification:", zero_shot_result)

#%% --------------------------------------------------------------------------------------------------------------------
generation_pipeline = pipeline("text-generation", model="gpt2")
generation_result = generation_pipeline("Once upon a time", max_length=30, num_return_sequences=1)
print("Text Generation:", generation_result)

#%% --------------------------------------------------------------------------------------------------------------------
qa_pipeline = pipeline("question-answering")
qa_result = qa_pipeline({
    "question": "Who wrote 'Pride and Prejudice'?",
    "context": "Jane Austen was an English novelist known primarily "
               "for her six major novels..."
})
print("Question Answering:", qa_result)

#%% --------------------------------------------------------------------------------------------------------------------
fill_mask_pipeline = pipeline("fill-mask", model="bert-base-uncased")
fill_mask_result = fill_mask_pipeline("Python is a [MASK] programming language.")
print("Masked LM (Fill-Mask):", fill_mask_result)