import pandas as pd
import numpy as np
from utils import *

import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation


# Read config.ini file
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")

print(config['instance']['user'])

def generate_text(model_path, sequence, max_length):
    # model_path = os.path.join(PATH, '../../../NLG_results')
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    # print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


if __name__=='__main__':

    PATH = os.getcwd()
    model_name = config['casualLM']['model_name']
    model_path = config['casualLM']['model_path']
    test_file_path = config['casualLM']['file_path']
    df = pd.read_csv(os.path.join(test_file_path), encoding="ISO-8859-1")
    seq_index = [int(x) for x in np.linspace(0, 199, 10).tolist()]
    sentences = df[config['casualLM']['colunm_name']][seq_index]
    gpt2_output = []
    finetune_output = []
    for sentence in sentences:
        sequence = " ".join(sentence.split(" ")[:6])
        text_output1 = generate_text(model_name, sequence, 30)  # change it to model_path if using fine-tuning model
        text_output2 = generate_text(model_path, sequence, 30)
        gpt2_output.append(text_output1)
        finetune_output.append(text_output2)

    df_out = pd.DataFrame(
        dict(
            orig=sentences,
            gpt2=gpt2_output,
            finetune=finetune_output
        )
    )
    df_out.to_csv(config['casualLM']['out_file'])