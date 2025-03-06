#%% --------------------------------------------------------------------------------------------------------------------
import os
import requests
import sentencepiece as spm

#%% --------------------------------------------------------------------------------------------------------------------
model_url = "https://huggingface.co/t5-small/resolve/main/spiece.model"
model_path = "spiece.model"

if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

#%% --------------------------------------------------------------------------------------------------------------------
sp = spm.SentencePieceProcessor(model_file=model_path)
text = "Hello world!"
pieces = sp.EncodeAsPieces(text)
ids = sp.EncodeAsIds(text)
print("Text:        ", text)
print("Pieces:      ", pieces)
print("IDs:         ", ids)
print("Decoded Text:", sp.DecodeIds(ids))