import streamlit as st
import boto3
import json
import numpy as np
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation
from pypdf import PdfReader
import faiss
import os
import nltk

with st.sidebar:
    NoChunks =st.slider('No Chunks', value=10, step=1, min_value=1, max_value=100, format='%d')
    top_k = st.slider('Top K', value=10, step=1, min_value=1, max_value=100, format='%d')

# Load configuration
def load_configuration():
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")
    return config


# Create a Bedrock Runtime client
def create_bedrock_client(config):
    session = boto3.Session(
        aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
        aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
        aws_session_token=config['BedRock_LLM_API']['aws_session_token']
    )
    return session.client("bedrock-runtime", region_name="us-east-1")


# Get embedding vectors from Bedrock
def get_embedding_vectors(client, model_id, text):
    body_content = {
        "inputText": text
    }
    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body_content)
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Load and process the PDF
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Split text into chunks
def chunk_text(text, chunk_size=10):
    sentences = nltk.sent_tokenize(text)
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks


# Convert chunks to vectors using Bedrock embeddings
def vectorize_chunks(chunks, bedrock_client, model_id):
    vectors = []
    for chunk in chunks:
        embedding = get_embedding_vectors(bedrock_client, model_id, chunk)
        if embedding:
            vectors.append(embedding)
    vectors_array = np.array(vectors, dtype='float32')   # FAISS needs float32
    return vectors_array


# Store vectors in FAISS index
def create_faiss_index(vectors):
    n, d = vectors.shape
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index


# Query the index and retrieve relevant chunks
def query_index(index, query_vector, chunks, top_k=10):
    top_k = min(top_k, index.ntotal)
    distances, indices = index.search(np.array([query_vector], dtype='float32'), top_k)
    return [chunks[i] for i in indices[0]]


# Ask a question and generate an answer with a reference
def ask_question(query, chunks, bedrock_client, embedding_model_id, lm_model_id, index):
    query_vector = get_embedding_vectors(bedrock_client, embedding_model_id, query)
    if query_vector:
        relevant_chunks = query_index(index, query_vector, chunks, top_k=top_k)

        # Truncate relevant chunks if they exceed the allowed length
        prompt = "\n".join(relevant_chunks)
        if len(prompt) > 49000:  # Leave some space for additional prompt text
            prompt = prompt[:49000]  # Truncate the text to fit the limit

        prompt += "\nQuestion: " + query

        body_content = {
            "anthropic_version": "bedrock-2023-05-31",  # Required field
            "max_tokens": 1000,  # Limit the number of tokens in the response
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = bedrock_client.invoke_model(
                modelId=lm_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body_content)
            )
            response_body = json.loads(response['body'].read().decode())

            # Extract the text from the response
            if "content" in response_body:
                answer = "".join([content["text"] for content in response_body["content"]])
                return answer
            else:
                return "No content found in the response."
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    return None


@st.cache_resource(show_spinner=False)
def build_index(file_bytes, chunk_size):
    """Runs once per (PDF, chunk size); Streamlit re-runs the whole script on every click, so we cache this."""
    config = load_configuration()
    bedrock_client = create_bedrock_client(config)
    import io
    chunks = chunk_text(load_pdf(io.BytesIO(file_bytes)), chunk_size=chunk_size)
    vectors = vectorize_chunks(chunks, bedrock_client, EMBEDDING_MODEL_ID)
    return chunks, create_faiss_index(vectors), bedrock_client


EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"   # Claude Sonnet 4.5


def main():
    st.title("Knowledge Assistant Retrival Tool (KART) :shopping_trolley:")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Process the uploaded PDF (cached)
        with st.spinner("Processing the PDF..."):
            chunks, index, bedrock_client = build_index(uploaded_file.getvalue(), NoChunks)
            embedding_model_id, lm_model_id = EMBEDDING_MODEL_ID, LM_MODEL_ID

        st.success(f"PDF processed and indexed! ({len(chunks)} chunks)")

        # Ask a question
        question = st.text_input("Ask a question about the document:")

        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = ask_question(question, chunks, bedrock_client, embedding_model_id, lm_model_id, index)
                    if answer:
                        st.subheader("Answer:")
                        st.write(answer)
                    else:
                        st.error("Failed to generate an answer.")
            else:
                st.error("Please enter a question.")


if __name__ == "__main__":
    main()