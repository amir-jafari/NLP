#%% --------------------------------------------------------------------------------------------------------------------
# RAG from scratch with Ollama + Qwen  (no frameworks: only the `ollama` python client)
#
# Setup (once):
#   1) Install Ollama: https://ollama.com/download      then start it:  ollama serve
#   2) Pull the two Qwen models:
#        ollama pull qwen3-embedding:0.6b     # embedding model  (text -> vector)
#        ollama pull qwen3.5:9b               # language model   (generates the answer)
#   3) pip install ollama
#
# RAG = 1) Load data  ->  2) Embed + store  ->  3) Retrieve  ->  4) Generate
#%% --------------------------------------------------------------------------------------------------------------------
import ollama

EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
LANGUAGE_MODEL = 'qwen3.5:9b'
#%% --------------------------------------------------------------------------------------------------------------------
# Step 1 - Load the dataset (each line = one chunk)
with open('cat-facts.txt', 'r') as file:
    dataset = [line.strip() for line in file if line.strip()]
print(f'Loaded {len(dataset)} chunks')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 2 - Build a "vector database": a plain list of (chunk, embedding)
embeddings = ollama.embed(model=EMBEDDING_MODEL, input=dataset)['embeddings']   # one call embeds all chunks
VECTOR_DB = list(zip(dataset, embeddings))
print(f'Vector DB has {len(VECTOR_DB)} entries, dim = {len(embeddings[0])}')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 3 - Retrieval: embed the query, rank chunks by cosine similarity
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)

def retrieve(query, top_n=3):
    q_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    scores = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in VECTOR_DB]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
#%% --------------------------------------------------------------------------------------------------------------------
# Step 4 - Generation: put the retrieved chunks into the prompt
question = input('Ask me a question about cats: ') or 'How long do cats sleep?'
retrieved = retrieve(question)

print('Retrieved knowledge:')
for chunk, score in retrieved:
    print(f'  ({score:.2f}) {chunk}')

context = '\n'.join(f'- {chunk}' for chunk, _ in retrieved)
system_prompt = f"""You are a helpful chatbot.
Use only the following context to answer the question. Don't make up any new information.
Context:
{context}"""

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[{'role': 'system', 'content': system_prompt},
              {'role': 'user', 'content': question}],
    think=False,      # Qwen3.5 is a "thinking" model; turn thinking off for a short, direct answer
    stream=True,
)
print('\nChatbot response:')
for part in stream:
    print(part['message']['content'], end='', flush=True)
print()
