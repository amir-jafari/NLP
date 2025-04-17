import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv
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


def get_embedding_vectors(client, model_id, text):
    # Body content structure for embedding API
    body_content = {
        "inputText": text
    }
    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body_content)
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None


def main():
    # Load configuration
    config = load_configuration()

    # Create a Bedrock Runtime client
    bedrock_client = create_bedrock_client(config)

    # Set the model ID for embedding
    model_id = "amazon.titan-embed-text-v2:0"

    # Text to get embedding for
    text = "What is the capital of France?"

    # Get embedding vectors
    embeddings = get_embedding_vectors(bedrock_client, model_id, text)

    if embeddings:
        print(f"Embedding Vectors: {embeddings}")


if __name__ == "__main__":
    main()