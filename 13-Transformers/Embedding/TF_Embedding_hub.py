import os
import tarfile
import urllib.request
import tensorflow as tf

# Universal Sentence Encoder (Google) - a TensorFlow SavedModel.
# Downloaded once as a .tar.gz and loaded with plain TensorFlow (no tensorflow_hub package needed).
MODEL_URL = "https://www.kaggle.com/api/v1/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2/download"
MODEL_DIR = os.path.expanduser("~/.cache/universal-sentence-encoder")


def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        os.makedirs(MODEL_DIR, exist_ok=True)
        archive = os.path.join(MODEL_DIR, "model.tar.gz")
        print("Downloading Universal Sentence Encoder (~1GB, only once)...")
        urllib.request.urlretrieve(MODEL_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(MODEL_DIR, filter="data")
        os.remove(archive)
    return tf.saved_model.load(MODEL_DIR)


def get_sentence_embedding(embed, text):
    # Generate embedding (512 numbers per sentence)
    embedding = embed([text])[0]
    return embedding.numpy()


# Example usage
if __name__ == "__main__":
    embed = load_model()
    text = "This is an example sentence to embed."
    embedding = get_sentence_embedding(embed, text)

    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")
