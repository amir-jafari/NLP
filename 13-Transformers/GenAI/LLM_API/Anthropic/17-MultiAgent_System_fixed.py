import io
import os
import re
import json
import contextlib
import boto3
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

# Multi-agent system in plain Python: a "researcher" and a "chart_generator" (both Claude Sonnet 4.5 on Bedrock)
# take turns on a shared conversation until one of them writes FINAL ANSWER.

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


# ---------------------------------------------------------------------------------------------------------------------
# 1) LLM: one function that sends a prompt to Claude on Bedrock (AWS keys come ONLY from ../../config/config.ini)
def make_bedrock_llm(max_tokens=700):
    load_dotenv()
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../../config/{os.environ['CONFIG_FILE']}")
    session = boto3.Session(
        aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
        aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
        aws_session_token=config['BedRock_LLM_API']['aws_session_token'],
    )
    client = session.client("bedrock-runtime", region_name="us-east-1")

    def llm(system_prompt, prompt):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        return json.loads(response["body"].read())["content"][0]["text"]

    return llm


# ---------------------------------------------------------------------------------------------------------------------
# 2) Tool: run Python code and capture what it prints (a tiny "Python REPL")
def python_repl_tool(code):
    """Executes Python code and returns its printed output."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__name__": "__repl__"})
        return f"Successfully executed. Stdout:\n{buffer.getvalue()}"
    except BaseException as e:
        return f"Failed to execute. Error: {e!r}"


# ---------------------------------------------------------------------------------------------------------------------
# 3) Agent = name + system prompt + (optional) tool
class Agent:
    def __init__(self, llm, name, role_description, can_run_code=False):
        self.llm, self.name, self.can_run_code = llm, name, can_run_code
        self.system_prompt = (
            "You are a helpful AI assistant, collaborating with other assistants. "
            "If you are unable to fully answer, that's OK; another assistant with different tools "
            "will help where you left off. Execute what you can to make progress. "
            "If you or any of the other assistants have the final answer or deliverable, "
            "prefix your response with FINAL ANSWER so the team knows to stop.\n"
            f"{role_description}"
        )
        if can_run_code:
            self.system_prompt += ("\nTo run Python, put it in ONE ```python ... ``` block. Use matplotlib with "
                                   "plt.savefig('chart.png') (never plt.show()). The code is executed for you.")

    def invoke(self, history):
        transcript = "\n\n".join(f"{speaker}: {text}" for speaker, text in history)
        reply = self.llm(self.system_prompt, f"Conversation so far:\n{transcript}\n\nYour turn ({self.name}):")
        code = re.search(r"```python\n(.*?)```", reply, re.S)
        if self.can_run_code and code:                     # the agent asked to run code -> run it
            reply += "\n\n[tool output] " + python_repl_tool(code.group(1))
        return reply


# ---------------------------------------------------------------------------------------------------------------------
# 4) Workflow: agents take turns until FINAL ANSWER (or max_turns)
def run_multi_agent_workflow(user_input, max_turns=6):
    import matplotlib
    matplotlib.use("Agg")                                  # save charts to files, no pop-up windows
    llm = make_bedrock_llm()
    researcher = Agent(llm, "researcher", "You can only do research (use your knowledge; no internet). "
                                          "You are working with a chart generator colleague.")
    charter = Agent(llm, "chart_generator", "You can only generate charts with Python. "
                                            "You are working with a researcher colleague.", can_run_code=True)

    history = [("user", user_input)]
    current, other = researcher, charter
    print(f"User: {user_input}\n" + "-" * 80)
    for turn in range(max_turns):
        reply = current.invoke(history)
        history.append((current.name, reply))
        print(f"{current.name}: {reply}\n" + "-" * 80)
        if "FINAL ANSWER" in reply:
            print("Workflow complete: Final answer provided.")
            break
        current, other = other, current                    # hand over to the other agent
    else:
        print("Workflow ended: Maximum turns reached")
    return history


if __name__ == "__main__":
    user_query = ("First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                  "Once you make the chart, finish.")
    run_multi_agent_workflow(user_query)
