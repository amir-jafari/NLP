#%% --------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import seaborn as sns

#%% --------------------------------------------------------------------------------------------------------------------
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#%% --------------------------------------------------------------------------------------------------------------------
inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
embeddings = model.embeddings.word_embeddings(inputs['input_ids'])
embeddings.retain_grad()

#%% --------------------------------------------------------------------------------------------------------------------
outputs = model(inputs_embeds=embeddings)
attention = outputs.attentions
attention_matrix = attention[0][0][0].detach().numpy()

#%% --------------------------------------------------------------------------------------------------------------------
sns.heatmap(attention_matrix, xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), cmap="viridis")
plt.title("Attention Weights")
plt.show()