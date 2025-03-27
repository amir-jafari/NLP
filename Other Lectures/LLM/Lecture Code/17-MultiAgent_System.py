from typing import Annotated, Literal, Optional, List, Any
import json
import boto3
from pydantic import Field
from langchain.llms.base import LLM
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_experimental.utilities import PythonREPL


class AWSBedrockConfig:
    AWS_ACCESS_KEY_ID = "A"
    AWS_SECRET_ACCESS_KEY = "A"
    AWS_SESSION_TOKEN = "A"
    REGION_NAME = "us-east-1"
    MODEL_ID = "amazon.titan-embed-text-v2:0"


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

    # THIS is the key addition: a bind_tools method
    def bind_tools(self, tool_classes):
        # If you actually need the tools, store them. Otherwise, simply:
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


@tool(description="Executes some custom action and returns a success message.")
def some_custom_tool(input_string: str) -> str:
    return "SUCCESS"


repl = PythonREPL()


@tool(description="Executes Python code in a REPL environment")
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants. "
        "Use the provided tools to progress towards answering the question. "
        "If you are unable to fully answer, that's OK; another assistant with different tools "
        "will help where you left off. Execute what you can to make progress. "
        "If you or any of the other assistants have the final answer or deliverable, "
        "prefix your response with FINAL ANSWER so the team knows to stop.\n"
        f"{suffix}"
    )


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto


def build_workflow():
    bedrock_llm = MyBedrockLLM(
        model_id=AWSBedrockConfig.MODEL_ID,
        temperature=0.7,
        max_tokens=300,
        region_name=AWSBedrockConfig.REGION_NAME,
        aws_access_key_id=AWSBedrockConfig.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWSBedrockConfig.AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWSBedrockConfig.AWS_SESSION_TOKEN
    )

    research_agent = create_react_agent(
        bedrock_llm,
        tools=[],
        prompt=make_system_prompt("You can only do research. You are working with a chart generator colleague.")
    )

    def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
        result = research_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "chart_generator")
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="researcher")
        return Command(update={"messages": result["messages"]}, goto=goto)

    chart_agent = create_react_agent(
        bedrock_llm,
        tools=[python_repl_tool],
        prompt=make_system_prompt("You can only generate charts. You are working with a researcher colleague.")
    )

    def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
        result = chart_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "researcher")
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="chart_generator")
        return Command(update={"messages": result["messages"]}, goto=goto)

    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "chart_generator")
    workflow.add_edge("chart_generator", "researcher")

    return workflow.compile()


if __name__ == "__main__":
    graph = build_workflow()
    events = graph.stream(
        {
            "messages": [
                (
                    "user",
                    "First, get the UK's GDP over the past 5 years, then make a line chart of it. Once you make the chart, finish."
                )
            ],
        },
        {"recursion_limit": 10},
    )
    for step in events:
        print(step)
        print("----")
