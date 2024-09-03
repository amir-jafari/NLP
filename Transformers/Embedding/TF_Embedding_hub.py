import tensorflow as tf
import tensorflow_hub as hub


def get_sentence_embedding(text):
    # Load the Universal Sentence Encoder model from TensorFlow Hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Generate embedding
    embedding = embed([text])[0]

    return embedding.numpy()


# Example usage
if __name__ == "__main__":
    text = "This is an example sentence to embed."
    embedding = get_sentence_embedding(text)

    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")