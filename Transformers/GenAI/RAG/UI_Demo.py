import streamlit as st
import boto3
import json
import numpy as np
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation
from PyPDF2 import PdfReader
import faiss
import os
import nltk

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
        text += page.extract_text()
    return text


# Split text into chunks
def chunk_text(text, chunk_size=2):
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
    vectors_array = np.array(vectors)
    return vectors_array


# Store vectors in FAISS index
def create_faiss_index(vectors):
    n, d = vectors.shape
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index


# Query the index and retrieve relevant chunks
def query_index(index, query_vector, chunks, top_k=3):
    distances, indices = index.search(np.array([query_vector]), top_k)
    return [chunks[i] for i in indices[0]]


# Ask a question and generate an answer with a reference
def ask_question(query, chunks, bedrock_client, embedding_model_id, lm_model_id, index):
    query_vector = get_embedding_vectors(bedrock_client, embedding_model_id, query)
    if query_vector:
        relevant_chunks = query_index(index, query_vector, chunks)

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


def main():
    st.title("Document Question Answering App")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        config = load_configuration()
        bedrock_client = create_bedrock_client(config)

        # Process the uploaded PDF
        with st.spinner("Processing the PDF..."):
            pdf_text = load_pdf(uploaded_file)
            chunks = chunk_text(pdf_text)

            # Set the model ID for embedding and language models
            embedding_model_id = "amazon.titan-embed-text-v2:0"
            lm_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

            # Vectorize chunks and create FAISS index
            vectors = vectorize_chunks(chunks, bedrock_client, embedding_model_id)
            index = create_faiss_index(vectors)

        st.success("PDF processed and indexed!")

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