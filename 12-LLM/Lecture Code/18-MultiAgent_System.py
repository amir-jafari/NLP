# %% --------------------------------------------------------------------------------------------------------------------
from typing import Annotated, Optional, List, Any
import json
import boto3
from pydantic import Field, BaseModel
from langchain.llms.base import LLM
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import requests
import zlib
import base64
import configparser


# %% ******************************* Please revise the "API_KEY" to the file 'config.ini' **********************************
class AWSBedrockConfig:
    MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    REGION_NAME = "us-east-1"

    parser = configparser.ConfigParser()
    parser.read('config.ini')

    AWS_ACCESS_KEY_ID = parser["BedRock_LLM_API"]["aws_access_key_id"]
    AWS_SECRET_ACCESS_KEY = parser["BedRock_LLM_API"]["aws_secret_access_key"]
    AWS_SESSION_TOKEN = parser["BedRock_LLM_API"]["aws_session_token"]


# %% --------------------------------------------------------------------------------------------------------------------
class MyBedrockLLM(LLM):
    model_id: str = Field(...)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)
    region_name: str = Field(...)
    aws_access_key_id: str = Field(...)
    aws_secret_access_key: str = Field(...)
    aws_session_token: str = Field(...)
    session: Any = Field(default=None, exclude=True)
    client: Any = Field(default=None, exclude=True)

    def bind_tools(self, tool_classes):
        return self

    def __init__(
            __pydantic_self__,
            model_id: str,
            temperature: float,
            max_tokens: int,
            region_name: str,
            aws_access_key_id: str,
            aws_secret_access_key: str,
            aws_session_token: str,
            **kwargs
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            **kwargs
        )
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        object.__setattr__(__pydantic_self__, "session", session)
        client = session.client("bedrock-runtime", region_name=region_name)
        object.__setattr__(__pydantic_self__, "client", client)

    @property
    def _llm_type(self) -> str:
        return "aws-bedrock-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        body_content = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body_content))
        result_str = response["body"].read().decode()
        try:
            parsed = json.loads(result_str)
            if "completion" in parsed:
                return parsed["completion"]
            elif "messages" in parsed and len(parsed["messages"]) > 0:
                return parsed["messages"][0].get("content", "")
            else:
                return result_str
        except json.JSONDecodeError:
            return result_str

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> AIMessage:
        text_output = self._call(prompt, stop=stop)
        return AIMessage(content=text_output)


@tool
def some_custom_tool(input_string: str) -> str:
    """Executes some custom action and returns a success message."""
    return "SUCCESS"


repl = PythonREPL()


@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Executes Python code in a REPL environment"""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


class Agent:
    def __init__(self, llm, system_prompt, tools=None, name=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.name = name

    def format_messages(self, conversation_history):
        formatted_messages = [SystemMessage(content=self.system_prompt)]
        for message in conversation_history:
            if isinstance(message, tuple):
                role, content = message
                if role == "user":
                    formatted_messages.append(HumanMessage(content=content))
                else:
                    formatted_messages.append(HumanMessage(content=content, name=role))
            else:
                formatted_messages.append(message)
        return formatted_messages

    def invoke(self, conversation_history):
        messages = self.format_messages(conversation_history)
        formatted_tools = ""
        if self.tools:
            formatted_tools = "\n\nTools available:\n"
            for tool in self.tools:
                formatted_tools += f"- {tool.__ne__}: {tool.__doc__ or 'No description'}\n"
        prompt = f"{self.system_prompt}{formatted_tools}\n\n"
        for msg in messages[1:]:  # Skip system message
            sender = msg.name if hasattr(msg, 'name') and msg.name else "user"
            prompt += f"{sender}: {msg.content}\n"
        response = self.llm(prompt)
        if self.name:
            return HumanMessage(content=response.content, name=self.name)
        return response


def make_system_prompt(role_description: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants. "
        "Use the provided tools to progress towards answering the question. "
        "If you are unable to fully answer, that's OK; another assistant with different tools "
        "will help where you left off. Execute what you can to make progress. "
        "If you or any of the other assistants have the final answer or deliverable, "
        "prefix your response with FINAL ANSWER so the team knows to stop.\n"
        f"{role_description}"
    )


def build_multi_agent_system():
    bedrock_llm = MyBedrockLLM(
        model_id=AWSBedrockConfig.MODEL_ID,
        temperature=0.7,
        max_tokens=300,
        region_name=AWSBedrockConfig.REGION_NAME,
        aws_access_key_id=AWSBedrockConfig.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWSBedrockConfig.AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWSBedrockConfig.AWS_SESSION_TOKEN
    )
    research_agent = Agent(
        llm=bedrock_llm,
        system_prompt=make_system_prompt("You can only do research. You are working with a chart generator colleague."),
        tools=[],
        name="researcher"
    )
    chart_agent = Agent(
        llm=bedrock_llm,
        system_prompt=make_system_prompt("You can only generate charts. You are working with a researcher colleague."),
        tools=[python_repl_tool],
        name="chart_generator"
    )
    return research_agent, chart_agent


def run_multi_agent_workflow(user_input, max_turns=10):
    research_agent, chart_agent = build_multi_agent_system()
    conversation_history = [("user", user_input)]
    current_agent = research_agent
    other_agent = chart_agent
    print("Starting multi-agent workflow...")
    print(f"User: {user_input}")

    for turn in range(max_turns):
        response = current_agent.invoke(conversation_history)
        conversation_history.append(response)
        print(f"{response.name}: {response.content}")
        print("----")
        if "FINAL ANSWER" in response.content:
            print("Workflow complete: Final answer provided.")
            break
        current_agent, other_agent = other_agent, current_agent

    if turn >= max_turns - 1:
        print("Workflow ended: Maximum turns reached")

    return conversation_history


# %% --------------------------------------------------------------------------------------------------------------------
def to_kroki_url(diagram_type, diagram_text, output_format="svg"):
    """
    Compresses the diagram text with zlib, then base64 encodes it,
    then returns a Kroki URL that should render the diagram.
    """
    compressed = zlib.compress(diagram_text.encode("utf-8"), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
    return f"https://kroki.io/{diagram_type}/{output_format}/{encoded}"


def conversation_to_mermaid_flowchart(conversation):
    """
    Convert the conversation array into a simple Mermaid flowchart.
    Each message has a 'role' and 'content'.
    We'll link them in the order they appear, starting at __start__ and ending at __end__.
    """
    lines = []
    lines.append("flowchart TB")
    lines.append("    __start__ --> user_0")
    last_node = "user_0"
    user_count = 0
    assistant_count = 0
    for i, msg in enumerate(conversation, start=1):
        if isinstance(msg, tuple):
            role, content = msg
        else:
            role = msg.name or "user"
            content = msg.content

        if role == "user":
            user_count += 1
            node_name = f"user_{user_count}"
        else:
            node_name = role

        if node_name != last_node:
            lines.append(f"    {last_node} --> {node_name}")
        last_node = node_name
    lines.append(f"    {last_node} --> __end__")
    return "\n".join(lines)

# %% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_query = "First, get the UK's GDP over the past 5 years, then make a line chart of it. Once you make the chart, finish."
    conversation = run_multi_agent_workflow(user_query, max_turns=10)

    # %% --------------------------------------------------------------------------------------------------------------------
    mermaid_text = conversation_to_mermaid_flowchart(conversation)
    url = to_kroki_url("mermaid", mermaid_text, output_format="png")
    response = requests.get(url)
    if response.status_code == 200:
        with open("conversation_flow.png", "wb") as f:
            f.write(response.content)
        print("conversation_flow.png has been created from multi-agent conversation flow!")
    else:
        print("Error:", response.status_code, response.text)
