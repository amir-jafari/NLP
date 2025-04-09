#%% --------------------------------------------------------------------------------------------------------------------
# Before using the below code, installing ollama from https://ollama.com/download and
"""
If you are using remote deployment on AWS, you run the below command in the terminal

sudo snap install ollama
"""
# follow the below steps to set up

# Step 1 - Open a terminal and run the following command
"""
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
"""

# If you see the following output, it mean it is successful
# pulling manifest
# ...
# verifying sha256 digest
# writing manifest
# success

# Step 2 - Install ollama package
"""
pip install ollama
"""

# If you cannot use ollama, you can create a new environment in the remote deployment.
# It is better to create a new envir for transformers, LLM, RAG usage, and set it up when you use remote deployment
#%% --------------------------------------------------------------------------------------------------------------------
import ollama
#%% --------------------------------------------------------------------------------------------------------------------
# Step 1 - Loading the dataset
dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 2 - Implement the vector database
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')
#%% --------------------------------------------------------------------------------------------------------------------
# Step 3 - Implement the retrieval function
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  similarities.sort(key=lambda x: x[1], reverse=True)
  return similarities[:top_n]
#%% --------------------------------------------------------------------------------------------------------------------
# Step 4 - Generation phrase
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
#%% --------------------------------------------------------------------------------------------------------------------
stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)