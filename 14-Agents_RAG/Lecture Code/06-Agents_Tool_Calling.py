#%% --------------------------------------------------------------------------------------------------------------------
# Agent with native tool calling: Qwen (via Ollama) decides WHEN to call a Python function
#   ollama pull qwen3.5:9b ; pip install ollama
#
# Loop:  user question -> LLM -> (tool call?) -> run tool -> send result back -> LLM final answer
#%% --------------------------------------------------------------------------------------------------------------------
import ollama

LANGUAGE_MODEL = 'qwen3.5:9b'
#%% --------------------------------------------------------------------------------------------------------------------
# 1) Tools are plain Python functions. Type hints + docstring tell the LLM how to use them.
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word.

    Args:
        word: The word to measure
    """
    return len(word)

def add(a: int, b: int) -> int:
    """Adds two integers.

    Args:
        a: First number
        b: Second number
    """
    return a + b

TOOLS = {'get_word_length': get_word_length, 'add': add}
#%% --------------------------------------------------------------------------------------------------------------------
# 2) The agent loop
def run_agent(question):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant. Use the tools when needed.'},
                {'role': 'user', 'content': question}]
    while True:
        response = ollama.chat(model=LANGUAGE_MODEL, messages=messages,
                               tools=list(TOOLS.values()), think=False)
        msg = response['message']
        messages.append(msg)
        if not msg.tool_calls:                          # no tool requested -> final answer
            return msg.content
        for call in msg.tool_calls:                     # run each requested tool
            name, args = call.function.name, call.function.arguments
            result = TOOLS[name](**args)
            print(f'  [tool] {name}({args}) -> {result}')
            messages.append({'role': 'tool', 'tool_name': name, 'content': str(result)})
#%% --------------------------------------------------------------------------------------------------------------------
print('Final Answer:', run_agent("What is the length of the word 'ChatGPT' plus the length of 'Qwen'?"))
