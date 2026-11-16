#%% --------------------------------------------------------------------------------------------------------------------
# End-to-end RAG over a PDF:  PDF -> chunks -> Qwen embeddings -> vector search -> Qwen answer
#   pip install ollama pypdf numpy
#   ollama pull qwen3-embedding:0.6b ; ollama pull qwen3.5:9b
#%% --------------------------------------------------------------------------------------------------------------------
import numpy as np
import ollama
from pypdf import PdfReader

EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
LANGUAGE_MODEL = 'qwen3.5:9b'
PDF_PATH = 'The-Life-of-Abraham-Lincoln_page2.pdf'
#%% --------------------------------------------------------------------------------------------------------------------
# Step 1 - Load the PDF
text = '\n'.join(page.extract_text() for page in PdfReader(PDF_PATH).pages)
print(f'Loaded {len(text)} characters')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 2 - Chunk: ~500 characters, 50 characters overlap, never split a word
def chunk_text(text, size=500, overlap=50):
    words, chunks, current = text.split(), [], ''
    for w in words:
        if len(current) + len(w) + 1 > size:
            chunks.append(current)
            current = current[-overlap:].split(' ', 1)[-1]      # keep a small overlap
        current = (current + ' ' + w).strip()
    return chunks + [current]

chunks = chunk_text(text)
print(f'{len(chunks)} chunks')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 3 - Embed all chunks -> matrix (n_chunks x dim), normalised so dot product = cosine
def embed(texts):
    v = np.array(ollama.embed(model=EMBEDDING_MODEL, input=texts)['embeddings'])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

chunk_vecs = embed(chunks)
#%% --------------------------------------------------------------------------------------------------------------------
# Step 4 - Retrieve top-k chunks for a question
def retrieve(question, k=3):
    scores = chunk_vecs @ embed([question])[0]
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]
#%% --------------------------------------------------------------------------------------------------------------------
# Step 5 - Generate an answer grounded in the retrieved chunks
def rag_response(question):
    context = '\n\n'.join(retrieve(question))
    prompt = f"""Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""
    return ollama.generate(model=LANGUAGE_MODEL, prompt=prompt, think=False)['response']
#%% --------------------------------------------------------------------------------------------------------------------
for q in ['When was Lincoln born?', 'Where was Lincoln born?']:
    print('Q:', q)
    print('A:', rag_response(q), '\n')
