#%% --------------------------------------------------------------------------------------------------------------------
# Talking to a local LLM (Qwen) through Ollama  -  the "G" in RAG
#   ollama pull qwen3.5:9b
#   pip install ollama
#%% --------------------------------------------------------------------------------------------------------------------
import ollama

LANGUAGE_MODEL = 'qwen3.5:9b'
#%% --------------------------------------------------------------------------------------------------------------------
# 1) Single prompt -> single answer   (ollama.generate = plain text completion)
out = ollama.generate(model=LANGUAGE_MODEL, prompt='Explain RAG in one sentence.', think=False)
print('=== generate ===')
print(out['response'])
#%% --------------------------------------------------------------------------------------------------------------------
# 2) Chat with roles: system (instructions) + user (question)   <- this is the format RAG uses
messages = [
    {'role': 'system', 'content': 'You are a concise NLP teaching assistant.'},
    {'role': 'user', 'content': 'What is an embedding?'},
]
out = ollama.chat(model=LANGUAGE_MODEL, messages=messages, think=False)
print('\n=== chat ===')
print(out['message']['content'])
#%% --------------------------------------------------------------------------------------------------------------------
# 3) Streaming: print tokens as they are generated
print('\n=== stream ===')
for part in ollama.chat(model=LANGUAGE_MODEL, messages=messages, think=False, stream=True):
    print(part['message']['content'], end='', flush=True)
print()
#%% --------------------------------------------------------------------------------------------------------------------
# 4) "Thinking" mode: Qwen3.5 can reason first, then answer (Chain-of-Thought)
out = ollama.chat(model=LANGUAGE_MODEL, think=True,
                  messages=[{'role': 'user', 'content': 'Route A: 12+15 min. Route B: 30 min. Which is faster?'}])
print('\n=== thinking ===')
print(out['message']['thinking'][:500], '...')
print('ANSWER:', out['message']['content'])
