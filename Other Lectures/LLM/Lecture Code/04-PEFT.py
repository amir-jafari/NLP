#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model

#%% --------------------------------------------------------------------------------------------------------------------
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

#%% --------------------------------------------------------------------------------------------------------------------
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,device_map="auto")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "Explain: Natural Language Processing is"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=60)
print("Prompt:", input_text)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
