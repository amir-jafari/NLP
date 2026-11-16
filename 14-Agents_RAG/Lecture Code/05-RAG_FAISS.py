#%% --------------------------------------------------------------------------------------------------------------------
# RAG with a real vector index (FAISS) + Ollama Qwen
#   pip install ollama faiss-cpu numpy
#   ollama pull qwen3-embedding:0.6b ; ollama pull qwen3.5:9b
#
# Same 4 steps as 01-RAG_Scratch.py, but the "vector database" is now a FAISS index.
#%% --------------------------------------------------------------------------------------------------------------------
import numpy as np
import faiss
import ollama

EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
LANGUAGE_MODEL = 'qwen3.5:9b'

def embed(texts):
    vecs = np.array(ollama.embed(model=EMBEDDING_MODEL, input=texts)['embeddings'], dtype='float32')
    faiss.normalize_L2(vecs)            # unit vectors -> inner product == cosine similarity
    return vecs
#%% --------------------------------------------------------------------------------------------------------------------
# Step 1 - Documents (our knowledge base)
documents = [
    "LangChain is a framework for building LLM-powered applications.",
    "Ollama runs open-source LLMs such as Qwen locally on your own machine.",
    "FAISS is a library from Meta for fast similarity search over dense vectors.",
    "RAG retrieves relevant documents and gives them to the LLM as context.",
    "Qwen is a family of open-weight language models released by Alibaba.",
]
#%% --------------------------------------------------------------------------------------------------------------------
# Step 2 - Embed + index
doc_vecs = embed(documents)
index = faiss.IndexFlatIP(doc_vecs.shape[1])     # exact search, inner product
index.add(doc_vecs)
print(f'Indexed {index.ntotal} documents (dim={index.d})')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 3 - Retriever
def retrieve(question, k=2):
    scores, ids = index.search(embed([question]), k)
    return [documents[i] for i in ids[0]], scores[0]
#%% --------------------------------------------------------------------------------------------------------------------
# Step 4 - Prompt template + LLM
TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""

def rag(question):
    docs, scores = retrieve(question)
    for d, s in zip(docs, scores):
        print(f'  retrieved ({s:.2f}): {d}')
    prompt = TEMPLATE.format(context='\n'.join(docs), question=question)
    return ollama.generate(model=LANGUAGE_MODEL, prompt=prompt, think=False)['response']
#%% --------------------------------------------------------------------------------------------------------------------
question = "How can I run Qwen on my own computer?"
print('Q:', question)
print('A:', rag(question))
