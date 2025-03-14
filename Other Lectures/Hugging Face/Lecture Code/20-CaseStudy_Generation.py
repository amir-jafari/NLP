#%% --------------------------------------------------------------------------------------------------------------------
from transformers import pipeline

#%% --------------------------------------------------------------------------------------------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text1 = """
Hugging Face is an open-source provider of NLP technologies. 
They created the Transformers library, which provides 
thousands of pre-trained models for tasks like classification, 
summarization, translation, and more. It's widely adopted 
by both researchers and industry professionals.
"""

text2 = """
Encoder-decoder models (also called sequence-to-sequence models) 
use both parts of the Transformer architecture. At each stage, 
the attention layers of the encoder can access all the words in 
the initial sentence, whereas the attention layers of the decoder 
can only access the words positioned before a given word in the input.

The pretraining of these models can be done using the objectives 
of encoder or decoder models, but usually involves something a bit 
more complex. For instance, T5 is pretrained by replacing random 
spans of text (that can contain several words) with a single mask 
special word, and the objective is then to predict the text that 
this mask word replaces.

Sequence-to-sequence models are best suited for tasks revolving 
around generating new sentences depending on a given input, such 
as summarization, translation, or generative question answering.
"""

#%% --------------------------------------------------------------------------------------------------------------------
summary1 = summarizer(text1, max_length=50, min_length=15, do_sample=False)
print("Generated Summary for text1:\n", summary1[0]["summary_text"])
summary2 = summarizer(text2, max_length=50, min_length=15, do_sample=False)
print("Generated Summary for text2:\n", summary2[0]["summary_text"])

