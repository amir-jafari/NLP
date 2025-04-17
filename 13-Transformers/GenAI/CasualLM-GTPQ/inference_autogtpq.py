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

    ### name for model and tokenizer    # INPUT_MODEL = "sangjeedondrub/tibetan-roberta-causal-base"
    INPUT_MODEL = 'TheBloke/Llama-2-13B-GPTQ'
    model_basename = "gptq_model-4bit-128g"
    # INPUT_MODEL = "EleutherAI/pythia-2.8b"

    ### text prompt
    PATH = os.getcwd()
    model_path = os.path.join(PATH, '../../../../../../results_autogtpq/')
    model, tokenizer = get_model_tokenizer(
        model_basename=model_basename,
        pretrained_model_name_or_path=model_path,
        gradient_checkpointing=True

    )

    text = "What years are the data for?"
    print(text)
    pre_process_result = preprocess(tokenizer, text)
    print("pre:")
    model_result = forward(model, tokenizer, pre_process_result)
    final_output = postprocess(tokenizer, model_result, False)
    print("post")
    print(final_output)
