#%% --------------------------------------------------------------------------------------------------------------------
import configparser
import boto3
from langchain_aws import ChatBedrock
from langchain.agents import initialize_agent, AgentType, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#%% --------------------------------------------------------------------------------------------------------------------
parser = configparser.ConfigParser()
parser.read("config.ini")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1",
    aws_access_key_id=parser["BedRock_LLM_API"]["aws_access_key_id"],
    aws_secret_access_key=parser["BedRock_LLM_API"]["aws_secret_access_key"],
    aws_session_token=parser["BedRock_LLM_API"]["aws_session_token"])
@tool
def get_word_length(word: str) -> int:
    return len(word)
#%% --------------------------------------------------------------------------------------------------------------------
tools = [get_word_length]
llm = ChatBedrock(client=bedrock_client, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",)
#%% --------------------------------------------------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Please chat with me!"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm_with_tools = llm.bind_tools(tools)
#%% --------------------------------------------------------------------------------------------------------------------
agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
response = agent.run("Hello! What is the length of 'ChatGPT'?")
print("Agent Response:", response)
