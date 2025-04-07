#%% --------------------------------------------------------------------------------------------------------------------
import configparser
import boto3
import re
from langchain_aws import ChatBedrock
from langchain.schema import SystemMessage, HumanMessage, AIMessage
#%% --------------------------------------------------------------------------------------------------------------------
def get_word_length(word: str) -> int:
    return len(word)
def dict_to_lc_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return AIMessage(content=content)
def convert_to_langchain_messages(conversation):
    return [dict_to_lc_message(m) for m in conversation]
def call_claude(messages):
    lc_messages = convert_to_langchain_messages(messages)
    chat_result = llm(lc_messages)
    return chat_result.content

def run_agent(query: str):
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who can think step by step. "
                "When you want to use a tool, follow this format:\n\n"
                "Tool: <tool_name>\nTool Input: <input>\n\n"
                "When you are ready to provide a final answer, write:\n\n"
                "Final Answer: <answer>\n\n"
                "Begin!"
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]
    while True:
        assistant_reply = call_claude(conversation)
        conversation.append({"role": "assistant", "content": assistant_reply})
        final_answer_match = re.search(r"Final Answer:\s*(.*)", assistant_reply)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return final_answer
        tool_call_match = re.search(r"Tool:\s*(.*)", assistant_reply)
        if tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            tool_input_match = re.search(r"Tool Input:\s*(.*)", assistant_reply)
            tool_input = tool_input_match.group(1).strip() if tool_input_match else ""

            if tool_name in TOOLS:
                tool_result = TOOLS[tool_name](tool_input)
                conversation.append({
                    "role": "assistant",
                    "content": f"Tool result: {tool_result}"
                })
            else:
                conversation.append({
                    "role": "assistant",
                    "content": f"Unknown tool: {tool_name}"
                })
        else:
            pass
#%% --------------------------------------------------------------------------------------------------------------------
parser = configparser.ConfigParser()
parser.read("config.ini")
bedrock_client = boto3.client(service_name="bedrock-runtime",region_name="us-east-1",
    aws_access_key_id=parser["BedRock_LLM_API"]["aws_access_key_id"],
    aws_secret_access_key=parser["BedRock_LLM_API"]["aws_secret_access_key"],
    aws_session_token=parser["BedRock_LLM_API"]["aws_session_token"])
#%% --------------------------------------------------------------------------------------------------------------------
llm = ChatBedrock(client=bedrock_client,model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",)
#%% --------------------------------------------------------------------------------------------------------------------
TOOLS = {"get_word_length": get_word_length}
#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    question = "Hello, can you give me the length of 'ChatGPT'?"
    answer = run_agent(question)
    print("Agent final answer:", answer)
