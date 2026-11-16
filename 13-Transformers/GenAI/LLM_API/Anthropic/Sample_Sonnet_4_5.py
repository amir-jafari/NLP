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

# Create a Bedrock Runtime client in the AWS Region you want to use.
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

# Model ID = inference profile for Claude Sonnet 4.5 (run check_profiles.py to see all available IDs)
model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

content = "What is the capital of france?"

body_content = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "temperature": 0,
    "messages": [
        {
            "role": "user",
            "content": content
        }
    ]
}

response = bedrock_client.invoke_model(
    modelId=model_id,
    body=json.dumps(body_content)
)
response_body = json.loads(response['body'].read().decode())
print("Raw response:", json.dumps(response_body, indent=2))
print("\nAnswer:", response_body["content"][0]["text"])
