#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model

#%% --------------------------------------------------------------------------------------------------------------------
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

#%% --------------------------------------------------------------------------------------------------------------------
model_name = "google/flan-t5-small"

# Ensure model and inputs are on the same device to avoid CUDA/CPU mismatch during generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "Explain: Natural Language Processing is"
# Move tokenized inputs to the same device as the model
inputs = tokenizer(input_text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=60)
print("Prompt:", input_text)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
