from functools import partial
import numpy as np
import os
import logging
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
'''
cite: https://ai.plainenglish.io/exploring-dolly-2-0-a-guide-to-training-your-own-chatgpt-like-model-dd9b785ff1df
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")
print(config['instance']['user'])


### to be added as special tokens
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

logger = logging.getLogger("logger")
### Model Loading
def load_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    return tokenizer

def load_model(pretrained_model_name_or_path, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True
    )
    return model.to(device)

def get_model_tokenizer(pretrained_model_name_or_path, gradient_checkpointing):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    # tokenizer.model_max_length
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device), tokenizer

# training prompt that does not contain an input string.
PROMPT_NO_INPUT_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
{response}
{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# training prompt that contains an input string that serves as context
PROMPT_WITH_INPUT_FORMAT = """{intro}
{instruction_key}
{instruction}
{input_key}
{input}
{response_key}
{response}
{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

def load_training_dataset(path_or_dataset="databricks/databricks-dolly-15k"):
    # dataset = load_dataset(path_or_dataset)["train"]
    ### load csv file
    train_file_path = config['casualLM_dolly']['file_path']
    df = pd.read_csv(train_file_path)
    dataset = Dataset.from_pandas(df)


    # data_files = {"train": os.path.join(PATH, '../../../../../../Data')}  # load dataset from files
    # # the file needs to have following column: id, context, question, answers
    # dataset = load_dataset(path=os.path.join(PATH, '../../../../../../Data'))
    # 'DatasetDict' object has no attribute 'train_test_split'

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(
                instruction=instruction,
                response=response,
                input=context
            )
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(
                instruction=instruction,
                response=response
            )
        return rec
    dataset = dataset.map(_add_text)
    return dataset

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        # padding=True
    )


def preprocess_dataset(tokenizer, max_length):
    dataset = load_training_dataset()
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset2 = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    dataset2 = dataset2.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset2 = dataset2.shuffle()
    return dataset2
    # return dataset2.set_format('torch', columns=['input_ids', 'attention_mask'], device='cuda')

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)
        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[1])[0]:  # 1 for opt-350m model
                # config for the models
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

'''
Inference
'''
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)
def load_model_tokenizer_for_generate(
    pretrained_model_name_or_path: str,) :
    """Loads the model and tokenizer so that it can be used for generating responses.

    Args:
        pretrained_model_name_or_path (str): name or path for model

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return model.to(device), tokenizer

def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        ValueError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)[1]
    # if len(token_ids) > 1:
    #     raise ValueError(f"Expected only a single token for f'{key}' but found f{token_ids}")
    return token_ids
def preprocess(tokenizer, instruction_text):
    prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(
        instruction=instruction_text
    )
    inputs = tokenizer(prompt_text, return_tensors="pt",)
    inputs["prompt_text"] = prompt_text
    inputs["instruction_text"] = instruction_text
    return inputs

def postprocess(tokenizer, model_outputs, return_full_text=False):
    response_key_token_id = get_special_token_id(tokenizer, RESPONSE_KEY_NL)
    end_key_token_id = get_special_token_id(tokenizer, END_KEY)
    generated_sequence = model_outputs["generated_sequence"][0]
    instruction_text = model_outputs["instruction_text"]
    generated_sequence = generated_sequence.cpu().numpy().tolist()
    records = []

    # print(response_key_token_id, end_key_token_id)

    for sequence in generated_sequence:
        decoded = None

        try:
            response_pos = sequence.index(response_key_token_id)
        except ValueError:
            logger.warn(
                f"Could not find response key {response_key_token_id} in: {sequence}"
            )
            response_pos = None

        if response_pos:
            try:
                end_pos = sequence.index(end_key_token_id)
            except ValueError:
                logger.warning(
                    f"Could not find end key, the output is truncated!"
                )
                end_pos = None
            decoded = tokenizer.decode(
                sequence[response_pos + 1 : end_pos]).strip()

        # If True,append the decoded text to the original instruction.
        if return_full_text:
            decoded = f"{instruction_text}\n{decoded}"
        rec = {"generated_text": decoded}
        records.append(rec)
    # return records
    return decoded

def forward(model, tokenizer, model_inputs, max_length=200):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]

    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length
    )

    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(
        in_b, out_b // in_b, *generated_sequence.shape[1:]
    )
    instruction_text = model_inputs.get("instruction_text", None)

    return {
        "generated_sequence": generated_sequence,
        "input_ids": input_ids, "instruction_text": instruction_text
    }
