import boto3
import json
import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

# Read config.ini file (AWS keys live ONLY in ../../config/config.ini)
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../../config/{config_file}")

session = boto3.Session(
    aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
    aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
    aws_session_token=config['BedRock_LLM_API']['aws_session_token']
)
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

# Claude Sonnet 4.5 (inference profile)
model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Converse API: one request format for every Bedrock model, with a system prompt and multi-turn history
system = [{"text": "You are a concise NLP teaching assistant."}]
messages = [{"role": "user", "content": [{"text": "What is a transformer in NLP?"}]}]

response = bedrock_client.converse(modelId=model_id, system=system, messages=messages,
                                   inferenceConfig={"maxTokens": 300, "temperature": 0})
answer = response["output"]["message"]
print("Turn 1:", answer["content"][0]["text"])

# Multi-turn: append the assistant answer + a follow-up question (the model has no memory; we resend the history)
messages += [answer, {"role": "user", "content": [{"text": "And how is it different from an RNN? One sentence."}]}]
response = bedrock_client.converse(modelId=model_id, system=system, messages=messages,
                                   inferenceConfig={"maxTokens": 300, "temperature": 0})
print("\nTurn 2:", response["output"]["message"]["content"][0]["text"])
print("\nTokens used:", response["usage"])
