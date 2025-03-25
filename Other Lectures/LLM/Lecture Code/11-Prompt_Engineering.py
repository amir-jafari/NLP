#%% --------------------------------------------------------------------------------------------------------------------
import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv  # pip install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

def load_configuration():
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../../config/{config_file}")
    return config

def create_bedrock_client(config):
    session = boto3.Session(
        aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
        aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
        aws_session_token=config['BedRock_LLM_API']['aws_session_token']
    )
    return session.client("bedrock-runtime", region_name="us-east-1")

def query_llama2_model(client, model_id, prompt):
    body_content = {
        "prompt": prompt
    }
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body_content),
            contentType="application/json",  # Adjust if needed
            accept="application/json"
        )
        response_body = response['body'].read().decode()
        return response_body
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None

#%% --------------------------------------------------------------------------------------------------------------------
def main():
    config = load_configuration()
    bedrock_client = create_bedrock_client(config)
    model_id = "meta.llama2-7b-chat-v1:0"
    naive_prompt = "Explain the Apollo moon missions."
    improved_prompt = (
        "You are a space historian. Explain in detail the significance of NASA’s Apollo moon "
        "missions, highlighting key achievements and their impact on science and society."
    )

    print("\n--- Naïve Prompt ---")
    naive_response = query_llama2_model(bedrock_client, model_id, naive_prompt)
    if naive_response:
        print("Naïve Response:\n", naive_response)

    print("\n--- Improved Prompt ---")
    improved_response = query_llama2_model(bedrock_client, model_id, improved_prompt)
    if improved_response:
        print("Improved Response:\n", improved_response)

#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
