#%% --------------------------------------------------------------------------------------------------------------------
from transformers import pipeline

#%% --------------------------------------------------------------------------------------------------------------------
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
prompt = "Large Language Model is"
outputs = text_generator(prompt, max_length=50, num_return_sequences=1)

#%% --------------------------------------------------------------------------------------------------------------------
print("Prompt:", prompt)
print("Generated text:", outputs[0]["generated_text"])