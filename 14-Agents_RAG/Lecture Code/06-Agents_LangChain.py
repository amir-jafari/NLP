#%% --------------------------------------------------------------------------------------------------------------------
import configparser
import boto3
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#%% --------------------------------------------------------------------------------------------------------------------
parser = configparser.ConfigParser()
parser.read("config.ini")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1",
    aws_access_key_id=parser["BedRock_LLM_API"]["aws_access_key_id"],
    aws_secret_access_key=parser["BedRock_LLM_API"]["aws_secret_access_key"],
    aws_session_token=parser["BedRock_LLM_API"]["aws_session_token"])
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
#%% --------------------------------------------------------------------------------------------------------------------
tools = [get_word_length]
llm = ChatBedrock(client=bedrock_client, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",)
#%% --------------------------------------------------------------------------------------------------------------------
# Bind tools to the LLM (enables tool calling)
llm_with_tools = llm.bind_tools(tools)
#%% --------------------------------------------------------------------------------------------------------------------
# Create a simple chain that uses tools
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the available tools when needed."),
        ("user", "{input}"),
    ]
)

chain = prompt | llm_with_tools
#%% --------------------------------------------------------------------------------------------------------------------
# Invoke the chain
response = chain.invoke({"input": "Hello! What is the length of 'ChatGPT'?"})
print("Agent Response:", response)

# If the model wants to use a tool, call it manually
if hasattr(response, 'tool_calls') and response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        if tool_name == 'get_word_length':
            result = get_word_length.invoke(tool_args)
            print(f"\nTool Result: The length of '{tool_args.get('word', '')}' is {result}")
else:
    print("\nFinal Answer:", response.content)
