#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

#%% --------------------------------------------------------------------------------------------------------------------
text = 'John went to the store. He bought some apples.'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer.encode_plus(text, return_tensors='pt')

#%% --------------------------------------------------------------------------------------------------------------------
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(**inputs)
attentions = outputs.attentions

#%% --------------------------------------------------------------------------------------------------------------------
attn_mat = attentions[-1][0, 0].detach().numpy()
plt.imshow(attn_mat, cmap='viridis')
plt.colorbar()
plt.title('Head 0 Attention (last layer)')
plt.show()
