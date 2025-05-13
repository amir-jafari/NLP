#%% --------------------------------------------------------------------------------------------------------------------

from textblob import TextBlob, Word
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import logging

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#%% --------------------------------------------------------------------------------------------------------------------
def get_transformer_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].squeeze().numpy()
#%% --------------------------------------------------------------------------------------------------------------------
# 1. TextBlob Basic Tasks
print("\n#%% ================================TextBlob Basic NLP Tasks=================================================")
text = "TextBlob and Transformers together unlock powerful NLP workflows."
blob = TextBlob(text)
print("Original Text:", blob)
print("Words:", blob.words)
print("Sentences:", blob.sentences)
print("POS Tags:", blob.tags)
print("Noun Phrases:", blob.noun_phrases)
print("Sentiment (polarity, subjectivity):", blob.sentiment)
w = Word("geese")
print("Lemmatized 'geese':", w.lemmatize())
typo = TextBlob("I havv goood spelng!")
print("Corrected:", typo.correct())
#%% --------------------------------------------------------------------------------------------------------------------
# 2. Transformer Sentiment Pipeline
print("\n#%% ================================Transformer-based Sentiment Analysis=====================================")
sent_pipeline = pipeline("sentiment-analysis")
for sample in ["I love this library!", "This is so frustrating..."]:
    result = sent_pipeline(sample)[0]
    print(f"Text: {sample}\n  Label: {result['label']}, Score: {result['score']:.3f}")
#%% --------------------------------------------------------------------------------------------------------------------
# 3. Transformer Embedding Extraction
print("\n#%% ===================================Transformer Embedding Extraction======================================")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
sample_texts = [
    "Natural language processing is fascinating.",
    "Transformers produce state-of-the-art embeddings!"
]
embeddings = {t: get_transformer_embedding(t) for t in sample_texts}
for t, vec in embeddings.items():
    print(f"Text: {t}\n  Embedding vector (first 5 values): {vec[:5]}")
#%% --------------------------------------------------------------------------------------------------------------------
# 4. Combining Similarity: TextBlob vs Transformers
print("\n#%% Similarity Comparison")
print("================================ Outputs: Similarity Comparison (Jaccard vs Cosine) ===========================")
pairs = [(sample_texts[0], sample_texts[1])]
for a, b in pairs:
    blob_a, blob_b = TextBlob(a), TextBlob(b)
    tokens_a, tokens_b = set(blob_a.words), set(blob_b.words)
    jaccard = (len(tokens_a & tokens_b) / len(tokens_a | tokens_b)) if tokens_a | tokens_b else 0.0
    tr_sim = cosine_similarity(
        embeddings[a].reshape(1,-1), embeddings[b].reshape(1,-1)
    )[0,0]
    print(f"Similarity between:'{a}' '{b}'")
    print(f"  TextBlob (Jaccard): {jaccard:.3f}, Transformers (cosine): {tr_sim:.3f}")
