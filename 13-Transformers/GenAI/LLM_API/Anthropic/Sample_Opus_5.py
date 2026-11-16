import boto3
from botocore.exceptions import ClientError
import json


import os
import sys
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation


# Windows consoles default to cp1252 and crash on the Unicode Claude emits (e.g. checkmarks)
sys.stdout.reconfigure(encoding="utf-8")

# Read config.ini file
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../../config/{config_file}")


session = boto3.Session(
    aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
    aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
    aws_session_token=config['BedRock_LLM_API']['aws_session_token']
)


# Create a Bedrock Runtime client in the AWS Region you want to use.
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID using inference profile for Claude Opus 5
model_id = "us.anthropic.claude-opus-5"

content = "What is the capital of france?"

body_content = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4000,
    # NOTE: Opus 5 rejects temperature / top_p / top_k (HTTP 400) -- do not add them.
    # Thinking is ON by default on Opus 5. "adaptive" lets the model decide how much
    # to think; "summarized" returns a readable summary instead of an empty block.
    "thinking": {
      "type": "adaptive",
      "display": "summarized"
    },
    # effort controls how much reasoning / how many tokens Opus 5 spends:
    # low | medium | high (default) | xhigh | max
    "output_config": {
      "effort": "medium"
    },
    "messages": [
      {
        "role": "user",
        "content": content
      }
    ]
  }


try:
    response = bedrock_client.invoke_model(
        modelId=model_id,

        body=json.dumps(body_content)
    )
except ClientError as e:
    print(f"Bedrock call failed: {e}")
    raise

response_body = json.loads(response['body'].read().decode())

# Opus 5 safety classifiers can decline a request: HTTP 200 with stop_reason "refusal".
# Always check stop_reason before reading content.
if response_body.get("stop_reason") == "refusal":
    print("Request was refused:", response_body.get("stop_details"))
else:
    for block in response_body["content"]:
        if block["type"] == "thinking":
            print("--- thinking ---")
            print(block.get("thinking", ""))
        elif block["type"] == "text":
            print("--- answer ---")
            print(block["text"])

print("--- usage ---")
print(response_body["usage"])
