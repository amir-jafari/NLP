# %% --------------------------------------------------------------------------------------------------------------------
import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation

# %% --------------------------------------------------------------------------------------------------------------------
def load_configuration():
    load_dotenv()
    config_file = '../../12-LLM/Lecture Code/config.ini'
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f'{config_file}')
    return config

def create_bedrock_client(config):
    session = boto3.Session(
        aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
        aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
        aws_session_token=config['BedRock_LLM_API']['aws_session_token']
    )
    return session.client("bedrock-runtime", region_name="us-east-1")

def query_llama3_model(client, model_id, prompt):
    body_content = {
        "prompt": prompt
    }
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body_content)
        )
        response_body = response['body'].read().decode()
        return response_body
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None

# %% --------------------------------------------------------------------------------------------------------------------
def main():
    config = load_configuration()
    bedrock_client = create_bedrock_client(config)
    model_id = "meta.llama3-70b-instruct-v1:0"
    prompt = "What means NLP stand for in Computer Science Area?"
    response = query_llama3_model(bedrock_client, model_id, prompt)
    if response:
        print(f"Model Response: {response}")

# %% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()