#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import pipeline

#%% --------------------------------------------------------------------------------------------------------------------
model_id = "EleutherAI/gpt-neo-125M"
pipe = pipeline(task="text-generation",model=model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,)
prompt = (
    "You are a pirate chatbot who always responds in pirate speak!\n"
    "User: Who are you?\n"
    "PirateBot:"
)

#%% --------------------------------------------------------------------------------------------------------------------
outputs = pipe(prompt,max_new_tokens=50,do_sample=True,temperature=0.7,)
print("Model Response:")
print(outputs[0]["generated_text"])
