import torch
from transformers import AutoModel, AutoTokenizer

# Load pre-trained model and tokenizer from 11-Hugging Face
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Sample text for embedding
text = "This is a sample sentence to get embeddings."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate embeddings (output of the last hidden state)
with torch.no_grad():  # Disable gradient calculation
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

# Optionally, you can average the token embeddings to get a sentence embedding
sentence_embedding = torch.mean(embeddings, dim=1)

# Convert to numpy array for easier manipulation (optional)
sentence_embedding_np = sentence_embedding.numpy()

print("Embedding shape:", sentence_embedding_np.shape)
print("Sentence embedding:", sentence_embedding_np)
