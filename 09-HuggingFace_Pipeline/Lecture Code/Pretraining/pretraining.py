import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from pathlib import Path

PATH = os.getcwd()
SAVE_MODEL = os.getcwd()

#paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

#tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=["<s>","<pad>", "</s>","<unk>", "<mask>",])
tokenizer.train(files= "kant.txt", vocab_size=52_000, min_frequency=2, special_tokens=["<s>","<pad>", "</s>","<unk>", "<mask>",])


tokenizer.save_model(SAVE_MODEL )

tokenizer = ByteLevelBPETokenizer(SAVE_MODEL + "/vocab.json", SAVE_MODEL+ "/merges.txt",)

tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")),("<s>", tokenizer.token_to_id("<s>")),)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("For it is in reality vain to profess"))

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained(SAVE_MODEL, max_len=512)
model = RobertaForMaskedLM(config=config)

print(model.num_parameters())

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path= PATH + "/kant.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling( tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir= SAVE_MODEL,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(SAVE_MODEL)


