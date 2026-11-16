from sentence_transformers import SentenceTransformer

# Load pre-trained Sentence-BERT model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Sample sentences for embedding
sentences = [
    "This is a sample sentence to get embeddings.",
    "Each sentence will be converted into a vector."
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Print the shape and embeddings
print("Embedding shape:", embeddings.shape)
print("Sentence embeddings:")
for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embeddings[i]}")
