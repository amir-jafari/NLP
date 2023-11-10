import re
import logging
from functools import partial
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
from utils import *

if __name__ == "__main__":
    logger = logging.getLogger("logger")
    from dotenv import load_dotenv  # pip3 install python-dotenv
    from configparser import ConfigParser, ExtendedInterpolation

    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")
    print(config['instance']['user'])

    ### name for model and tokenizer
    # INPUT_MODEL = "sangjeedondrub/tibetan-roberta-causal-base"
    INPUT_MODEL = config['casualLM_dolly']['model_name']
    # INPUT_MODEL = "EleutherAI/pythia-2.8b"

    ### text prompt
    model_path = config['casualLM_dolly']['model_path']
    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=model_path,
        gradient_checkpointing=True
    )

    text = "What is a polygon?"
    print(text)
    pre_process_result = preprocess(tokenizer, text)
    print("pre:")
    model_result = forward(model, tokenizer, pre_process_result)
    final_output = postprocess(tokenizer, model_result, False)
    print("post")
    print(final_output)
