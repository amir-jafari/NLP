#%% --------------------------------------------------------------------------------------------------------------------
# Agent from scratch (ReAct style): no tool-calling API, just a prompt format + regex + a loop
#   ollama pull qwen3.5:9b ; pip install ollama
#
# The LLM writes text like "Tool: ... / Tool Input: ...". WE parse it, run the tool, and feed the result back.
#%% --------------------------------------------------------------------------------------------------------------------
import re
import ollama

LANGUAGE_MODEL = 'qwen3.5:9b'

def get_word_length(word: str) -> int:
    return len(word.strip("'\" "))

TOOLS = {'get_word_length': get_word_length}

SYSTEM_PROMPT = """You are an assistant who thinks step by step.
Available tools:
  get_word_length: returns the number of characters in a word.

To use a tool, reply with exactly these two lines and then STOP:
Tool: <tool_name>
Tool Input: <input>

When you know the answer, reply with:
Final Answer: <answer>"""
#%% --------------------------------------------------------------------------------------------------------------------
def run_agent(question, max_steps=5):
    conversation = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': question}]
    for step in range(max_steps):
        reply = ollama.chat(model=LANGUAGE_MODEL, messages=conversation, think=False)['message']['content']
        tool = re.search(r'Tool:\s*(.*)\nTool Input:\s*(.*)', reply)

        if tool and tool.group(1).strip() in TOOLS:              # 1) tool request -> run it
            reply = tool.group(0)                                #    drop anything the LLM "guessed" after it
            result = TOOLS[tool.group(1).strip()](tool.group(2))
            observation = f'Tool result: {result}'
        else:
            final = re.search(r'Final Answer:\s*(.*)', reply)    # 2) final answer -> done
            if final:
                print(f'--- step {step + 1} ---\n{reply}')
                return final.group(1).strip()
            observation = 'Please use a tool or give the Final Answer.'

        print(f'--- step {step + 1} ---\n{reply}\n{observation}')
        conversation.append({'role': 'assistant', 'content': reply})
        conversation.append({'role': 'user', 'content': observation})   # 3) observation back to the LLM
    return 'No answer within step limit.'
#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Agent final answer:', run_agent("Hello, can you give me the length of 'ChatGPT'?"))
