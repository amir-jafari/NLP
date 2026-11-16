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

# Streaming: print the answer token-by-token as Claude writes it (same request body as Sample_Sonnet_4_5.py)
body_content = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "Explain the attention mechanism in 3 sentences."}]
}

response = bedrock_client.invoke_model_with_response_stream(modelId=model_id, body=json.dumps(body_content))
for event in response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if chunk["type"] == "content_block_delta":
        print(chunk["delta"].get("text", ""), end="", flush=True)
print()
