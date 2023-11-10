import re
import logging
from functools import partial
import numpy as np
import os

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_dataset
from utils import *
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

if __name__ == "__main__":

    logger = logging.getLogger("logger")
    # Read config.ini file
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")
    print(config['instance']['user'])

    ### name for model and tokenizer
    INPUT_MODEL = config['casualLM_dolly']['model_name']
    # INPUT_MODEL = 'bigscience/bloom-560m'
    # INPUT_MODEL = 'bigscience/bloom-1b1' # CUDA out of memory
    # INPUT_MODEL = 'EleutherAI/gpt-neo-1.3B' # CUDA out of memory
    # INPUT_MODEL = 'meta-llama/Llama-2-7b-chat-hf' # CUDA out of memory
    # INPUT_MODEL = "EleutherAI/pythia-2.8b"   # CUDA out of memory
    # INPUT_MODEL = "tiiuae/falcon-7b-instruct"   # Process finished with exit code 137




    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=INPUT_MODEL,
        gradient_checkpointing=True
    )
    # find max length in model configuration
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max length: {max_length}")
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = config['casualLM_dolly']['max_length']
        logger.info(f"Using default max length: {max_length}")
        print(f"Using default max length: {max_length}")

    data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        )

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length)

    split_dataset = processed_dataset.train_test_split(test_size=float(config['casualLM_dolly']['test_size']), seed=10086)
    local_output_dir = config['casualLM_dolly']['model_path']
    training_args = TrainingArguments(
            output_dir=local_output_dir,
            per_device_train_batch_size=int(config['casualLM_dolly']['batch_size']),
            per_device_eval_batch_size=int(config['casualLM_dolly']['batch_size']),
            fp16=False,
            bf16=False,
            learning_rate=float(config['casualLM_dolly']['learning_rate']),
            num_train_epochs=int(config['casualLM_dolly']['epochs']),
            deepspeed=None,
            gradient_checkpointing=True,
            logging_dir=f"{local_output_dir}/runs",
            logging_strategy="steps",
            logging_steps=int(config['casualLM_dolly']['save_steps']),
            evaluation_strategy="steps",
            eval_steps=int(config['casualLM_dolly']['save_steps']),
            save_strategy="steps",
            save_steps=int(config['casualLM_dolly']['save_steps']),
            save_total_limit=int(config['casualLM_dolly']['save_total_limit']),
            load_best_model_at_end=False,
            # report_to="tensorboard",
            disable_tqdm=True,
            remove_unused_columns=False,
            local_rank=2,
            warmup_steps=0,
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )


    trainer.train()
    trainer.save_model()


