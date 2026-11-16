#%% --------------------------------------------------------------------------------------------------------------------
# Offline vector store with Deep Lake 4.x + real Qwen embeddings from Ollama
#   pip install deeplake ollama        (code below is for deeplake 4.x, the 3.x API is different)
#   ollama pull qwen3-embedding:0.6b
#%% --------------------------------------------------------------------------------------------------------------------
import os
import shutil
import deeplake
from deeplake import types
import ollama

EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
DATASET_PATH = 'my_local_vectorstore'

def embed(texts):
    return ollama.embed(model=EMBEDDING_MODEL, input=texts)['embeddings']
#%% --------------------------------------------------------------------------------------------------------------------
# 1) Create a local (offline) dataset with a text column and an embedding column
documents = [
    "Deep Lake is awesome for offline vector storage!",
    "Cats sleep about two thirds of every day.",
    "Abraham Lincoln was the 16th president of the United States.",
]
doc_embeddings = embed(documents)
dim = len(doc_embeddings[0])                       # 1024 for qwen3-embedding:0.6b

if os.path.exists(DATASET_PATH):
    shutil.rmtree(DATASET_PATH)
ds = deeplake.create(f"file://{DATASET_PATH}")
ds.add_column("text", types.Text())
ds.add_column("embedding", types.Embedding(dim))
#%% --------------------------------------------------------------------------------------------------------------------
# 2) Append text + embeddings and commit (Deep Lake versions data like git)
ds.append({"text": documents, "embedding": doc_embeddings})
ds.commit("add first documents")
#%% --------------------------------------------------------------------------------------------------------------------
# 3) Re-open the dataset and run a vector search with TQL (Tensor Query Language)
ds = deeplake.open_read_only(f"file://{DATASET_PATH}")
query = "Who was president of the US?"
q = ",".join(str(x) for x in embed(query)[0])

results = ds.query(f"""
    SELECT *, cosine_similarity(embedding, ARRAY[{q}]) AS score
    ORDER BY cosine_similarity(embedding, ARRAY[{q}]) DESC
    LIMIT 2
""")
print("Query:", query)
for row in results:
    print(f"[score={row['score']:.3f}] {row['text']}")
