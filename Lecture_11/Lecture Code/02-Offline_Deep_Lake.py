#%% --------------------------------------------------------------------------------------------------------------------
# pip install deeplake
# The below code must run deeplake 4.0, not deeplake 3.0. The function in two different versions.

# DO NOT 100% believe ChatGPT and the deeplake official website, they always do not update.
#%% --------------------------------------------------------------------------------------------------------------------
import deeplake
from deeplake import types
import numpy as np
#%% --------------------------------------------------------------------------------------------------------------------
# 1) Create an offline dataset locally
ds = deeplake.create("file://my_local_vectorstore")
ds.add_column("text", types.Text(index_type=types.BM25))
ds.add_column("embedding", types.Embedding(768))
ds.commit()
#%% --------------------------------------------------------------------------------------------------------------------
# 2) Append data (TEXT & EMBEDDINGS)
documents = [
    "Deep Lake is awesome for offline vector storage!",
    "Here is another document.",
    "Offline usage with create/open in Deep Lake 4.0!"
]
# Dummy embeddings
embeddings = [np.random.randn(768).tolist() for _ in documents]
ds.append({
    "text": documents,
    "embedding": embeddings
})
ds.commit()
#%% --------------------------------------------------------------------------------------------------------------------
# 3) Re-open the dataset offline, and do a quick vector query
ds_opened = deeplake.open("file://my_local_vectorstore")
query_emb = np.random.randn(768).tolist()
query_str = ",".join(str(x) for x in query_emb)

tql = f"""
SELECT * 
FROM (
    SELECT *, cosine_similarity(embedding, ARRAY[{query_str}]) as score
    FROM (
        SELECT *, ROW_NUMBER() AS row_id
    )
)
ORDER BY score DESC
LIMIT 2
"""
#%% --------------------------------------------------------------------------------------------------------------------
results = ds_opened.query(tql)
for row in results:
    print(f"[score={row['score']:.3f}] {row['text']}")
