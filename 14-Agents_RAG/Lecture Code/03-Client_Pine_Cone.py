#%% --------------------------------------------------------------------------------------------------------------------
# Cloud vector store (Pinecone) + local Qwen models (Ollama)
#   pip install pinecone ollama          (code below is for the pinecone 10.x client: keyword-only arguments)
#   ollama pull qwen3-embedding:0.6b ; ollama pull qwen3.5:9b
#   Sign up at pinecone.io, copy your API key into config.ini:
#       [Pinecone_API]
#       pinecone_api_key = <your key>
#
# Ollama/Qwen makes the embeddings and the answer; Pinecone only STORES and SEARCHES the vectors.
#%% --------------------------------------------------------------------------------------------------------------------
import time
import configparser
import ollama
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = 'qwen3-embedding:0.6b'      # 1024-dim vectors
LANGUAGE_MODEL = 'qwen3.5:9b'
INDEX_NAME = 'qwen-rag-index'
NAMESPACE = 'example-namespace'

def embed(texts):
    return ollama.embed(model=EMBEDDING_MODEL, input=texts)['embeddings']
#%% --------------------------------------------------------------------------------------------------------------------
# 1) Connect and create a serverless index (dimension must match the embedding model)
config = configparser.ConfigParser()
config.read('config.ini')
pc = Pinecone(api_key=config['Pinecone_API']['pinecone_api_key'])

if not pc.has_index(INDEX_NAME):
    pc.create_index(name=INDEX_NAME, dimension=1024, metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)
#%% --------------------------------------------------------------------------------------------------------------------
# 2) Upsert: each record = id + vector + metadata (the original text is kept in metadata)
records = [
    ("rec1", "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "history"),
    ("rec2", "Photosynthesis allows plants to convert sunlight into energy.", "science"),
    ("rec3", "Albert Einstein developed the theory of relativity.", "science"),
    ("rec4", "The mitochondrion is often called the powerhouse of the cell.", "biology"),
    ("rec5", "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "literature"),
    ("rec6", "The Great Wall of China was built to protect against invasions.", "history"),
    ("rec7", "Honey never spoils due to its low moisture content and acidity.", "food science"),
    ("rec8", "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "history"),
    ("rec9", "Leonardo da Vinci painted the Mona Lisa.", "art"),
    ("rec10", "The Statue of Liberty was a gift from France to the United States.", "history"),
    ("rec11", "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "history"),
    ("rec12", "Mount Everest is the tallest mountain in the world.", "geography"),
]
vectors = embed([text for _, text, _ in records])
index.upsert(namespace=NAMESPACE, vectors=[
    {"id": rid, "values": vec, "metadata": {"text": text, "category": cat}}
    for (rid, text, cat), vec in zip(records, vectors)
])

# Pinecone is eventually consistent: wait until the new vectors are searchable
while index.describe_index_stats().namespaces.get(NAMESPACE, None) is None or \
        index.describe_index_stats().namespaces[NAMESPACE].vector_count < len(records):
    time.sleep(1)
#%% --------------------------------------------------------------------------------------------------------------------
# 3) Retrieve: embed the query locally, search in Pinecone
query = "Famous historical structures and monuments"
results = index.query(namespace=NAMESPACE, vector=embed(query)[0], top_k=5, include_metadata=True)
for m in results.matches:
    print(f"id: {m.id:5s} score: {m.score:.2f}  [{m.metadata['category']}] {m.metadata['text']}")
#%% --------------------------------------------------------------------------------------------------------------------
# 4) (Optional) metadata filter: only search inside one category
results = index.query(namespace=NAMESPACE, vector=embed("Who made art?")[0], top_k=2,
                      include_metadata=True, filter={"category": {"$eq": "art"}})
print("\nFiltered (art):", [m.metadata['text'] for m in results.matches])
#%% --------------------------------------------------------------------------------------------------------------------
# 5) Generate: a full RAG answer with Qwen using the Pinecone hits as context
question = "Which monuments were connected to France?"
hits = index.query(namespace=NAMESPACE, vector=embed(question)[0], top_k=3, include_metadata=True).matches
context = "\n".join(m.metadata['text'] for m in hits)
prompt = f"Answer based only on this context:\n{context}\n\nQuestion: {question}\nAnswer:"
print("\nQ:", question)
print("A:", ollama.generate(model=LANGUAGE_MODEL, prompt=prompt, think=False)['response'])
