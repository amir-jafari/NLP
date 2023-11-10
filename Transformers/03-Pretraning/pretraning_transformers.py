# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline
from transformers import Trainer, TrainingArguments
import os
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
# ----------------------------------------------------------------------------------------------------------------------
paths = [str(x) for x in Path(".").glob("**/*.txt")]
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

token_dir = os.makedirs(os.getcwd() +'/content/KantaiBERT',exist_ok=True)
tokenizer.save_model(os.getcwd() +'/content/KantaiBERT/')
tokenizer = ByteLevelBPETokenizer(os.getcwd() +'/content/KantaiBERT' +"/vocab.json",os.getcwd() +'/content/KantaiBERT'+"/merges.txt",)
tokenizer.encode("The Critique of Pure Reason.").tokens
tokenizer.encode("The Critique of Pure Reason.")
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
torch.cuda.is_available()
# ----------------------------------------------------------------------------------------------------------------------
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

print(config)
tokenizer = RobertaTokenizer.from_pretrained(os.getcwd() +'/content/KantaiBERT', max_length=512)
model = RobertaForMaskedLM(config=config)
print(model)
print(model.num_parameters())
# ----------------------------------------------------------------------------------------------------------------------

LP=list(model.parameters())
lp=len(LP)
print(lp)
for p in range(0,lp):
  print(LP[p])

np=0
for p in range(0,lp):
  PL2=True
  try:
    L2=len(LP[p][0])
  except:
    L2=1
    PL2=False
  L1=len(LP[p])
  L3=L1*L2
  np+=L3
  if PL2==True:
    print(p,L1,L2,L3)
  if PL2==False:
    print(p,L1,L3)

print(np)
# ----------------------------------------------------------------------------------------------------------------------
dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path="kant.txt", block_size=128,)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="./KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer( model=model, args=training_args, data_collator=data_collator, train_dataset=dataset,)
trainer.train()
trainer.save_model(os.getcwd() +'/content/KantaiBERT')
fill_mask = pipeline("fill-mask", model=os.getcwd() +'/content/KantaiBERT', tokenizer=os.getcwd() +'/content/KantaiBERT')
fill_mask("Human thinking involves<mask>.")

