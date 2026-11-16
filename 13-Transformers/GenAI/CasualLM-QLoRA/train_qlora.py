from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from utils import cfg, load_tokenizer, load_4bit_model, add_lora_adapters, load_training_dataset

if __name__ == "__main__":
    model_name = cfg['model_name']
    output_dir = cfg['model_path']

    # 1) 4-bit base model + LoRA adapters
    tokenizer = load_tokenizer(model_name)
    model = load_4bit_model(model_name)
    model = add_lora_adapters(model)

    # 2) Dolly data -> tokens (prompt masked out of the loss), 70/30 train/test split
    dataset = load_training_dataset(tokenizer).train_test_split(test_size=float(cfg['test_size']), seed=10086)

    # 3) Train: only the LoRA weights receive gradients
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg['batch_size']),
        per_device_eval_batch_size=int(cfg['batch_size']),
        learning_rate=float(cfg['learning_rate']),      # LoRA uses a much higher LR than full fine-tuning
        num_train_epochs=int(cfg['epochs']),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=int(cfg['logging_steps']),
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # pads input_ids with the pad token and labels with -100
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    # 4) Save only the adapters (a few MB) - the 4-bit base model is re-downloaded/loaded at inference time
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved LoRA adapters to", output_dir)
