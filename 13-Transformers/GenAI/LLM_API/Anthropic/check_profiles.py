import boto3
import os
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation

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

client = session.client('bedrock', region_name='us-east-1')

try:
    profiles = client.list_inference_profiles()
    print('\nAvailable Inference Profiles:')
    for p in profiles.get('inferenceProfileSummaries', []):
        if 'anthropic' in p['inferenceProfileId'].lower():
            print(f"  - {p['inferenceProfileId']}")

    if not any('anthropic' in p['inferenceProfileId'].lower() for p in profiles.get('inferenceProfileSummaries', [])):
        print("  No Anthropic inference profiles found!")
except Exception as e:
    print(f'Error listing inference profiles: {e}')