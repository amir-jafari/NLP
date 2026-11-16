from utils import load_tokenizer, load_model
import os
import torch
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

def generate_text(model_path, keyword, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    inputs = tokenizer(keyword, return_tensors="pt").to(model.device)
    final_outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=0,
        top_p=0.9,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    # Read config.ini file
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")

    model_name = config['condi_gen']['model_name']
    model_path = config['condi_gen']['model_path']
    sequence = 'Waltzmart Mens 3D Creative Graffiti Print Hip Hop Style T-Shirts, seriously, kind, stretchy, impressed, definitely, lightweight, comfortable, thrilled, highly, actual, honest'
    # sequence = 'dreamcatcher, edging, small plastic, case, noticeable, annoying, $5'
    print("t5 without train:")
    generate_text(model_name, sequence, 100)
    print("t5 with train:")
    generate_text(model_path, sequence, 200)

