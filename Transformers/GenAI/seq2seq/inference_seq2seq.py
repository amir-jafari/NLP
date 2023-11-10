from datasets import load_dataset
from utils import load_data, load_data_collator, load_tokenizer, load_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
import os
import torch
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def generate_text(model_path, keyword, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    pipeline = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
    final_outputs = pipeline(
        keyword,
        max_length=max_length,
        do_sample=True,
        top_k=0,
        top_p=0.9,
    )
    print(final_outputs[0]['generated_text'])

if __name__=='__main__':

    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")

    # you need to set parameters
    model_name = config['seq2seq']['model_name']
    # sequence = input("Please input the start of sentence:")
    model_path = config['seq2seq']['model_path']
    sequence = 'Waltzmart Mens 3D Creative Graffiti Print Hip Hop Style T-Shirts, seriously, kind, stretchy, impressed, definitely, lightweight, comfortable, thrilled, highly, actual, honest'
    # sequence = 'dreamcatcher, edging, small plastic, case, noticeable, annoying, $5'
    print("t5 without train:")
    generate_text(model_name, sequence, 100)
    print("t5 with train:")
    generate_text(model_path, sequence, 200)
