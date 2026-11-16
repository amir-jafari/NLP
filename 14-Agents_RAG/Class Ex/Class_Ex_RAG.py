#%% --------------------------------------------------------------------------------------------------------------------
#
# Class Ex - RAG lecture
#
#%% --------------------------------------------------------------------------------------------------------------------
#%% --------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
# Class_Ex1:
#
# You have a long text document. How can you split it into smaller chunks of a fixed size (for example 200
# characters per chunk), ensuring you don’t split words in the middle?
#
# Hint:
# Accumulate words until you reach approximately 200 characters, then start a new chunk
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')


print(20*'-' + 'End Q1' + 20*'-')

# ======================================================================================================================
# Class_Ex2:
# Creating and Using Embeddings
#
# How do you generate embeddings for a list of text chunks using a local Qwen embedding model served by Ollama
# (ollama pull qwen3-embedding:0.6b ; pip install ollama)?
#
# example_chunks = [
#     "RAG involves a retriever and a generator.",
#     "The retriever finds relevant documents.",
#     "The generator is typically an LLM that produces a final answer."
# ]
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')



print(20*'-' + 'End Q2' + 20*'-')

# ======================================================================================================================
# Class_Ex3:
# Building a Simple FAISS Vector Store
#
# Given a list of vector embeddings, how can you store them in a FAISS index and then query for the top similar vector?
#
# Hint:
# Use faiss.IndexFlatL2 for a simple L2 distance index
# Add embeddings to the index. Then perform a similarity search (k=2 or 3)
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')


print(20*'-' + 'End Q3' + 20*'-')

# ======================================================================================================================
# Class_Ex4:
# Simple Retrieval-Augmented Generation Pipeline
#
# How can you create a mini RAG pipeline that, given a user query, retrieves similar text chunks from FAISS and
# then concatenates them for a final “generated” answer?
#
# Hint:
# Use the same FAISS index from the previous exercise
# Embed the chunk_texts and the query with Qwen (Ex2), retrieve the top chunks, then let the Qwen LLM
# (ollama pull qwen3.5:9b) generate the final answer from them
#
# chunk_texts = [
#     "RAG stands for Retrieval-Augmented Generation.",
#     "Retriever fetches relevant information from a vector store.",
#     "Generator (LLM) takes retrieved chunks and user query to produce an answer.",
#     "LangChain and other libraries simplify building RAG pipelines.",
#     "Vector databases like FAISS power fast similarity searches."
# ]
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')


print(20*'-' + 'End Q4' + 20*'-')

# ======================================================================================================================
# Class_Ex5:
# Adding a Simple Chat-Style Prompt
#
# How can you build a basic prompt that inserts both a user query and retrieved chunks into a single text, emulating a
# chat interface?
#
# Hint:
# Format the prompt using f-strings, print it, and pass it to the Qwen LLM with ollama.generate
#
# chunk_texts = [
#     "RAG stands for Retrieval-Augmented Generation.",
#     "Retriever fetches relevant information from a vector store.",
#     "Generator (LLM) takes retrieved chunks and user query to produce an answer.",
#     "LangChain and other libraries simplify building RAG pipelines.",
#     "Vector databases like FAISS power fast similarity searches."
# ]
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')


print(20*'-' + 'End Q5' + 20*'-')