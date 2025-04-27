# ----------------------------------------------------------------------------------------------------------------------
import argparse, pathlib, datasets, transformers
ap = argparse.ArgumentParser()
ap.add_argument("--train_file"); ap.add_argument("--model_name")
ap.add_argument("--output_dir"); ap.add_argument("--epochs", type=int, default=3)
args = ap.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
ds = datasets.load_dataset("json", data_files=args.train_file, split="train")
tok = transformers.AutoTokenizer.from_pretrained(args.model_name)
def tok_and_label(batch):
    tok_out = tok(batch["text"], truncation=True, padding="max_length", max_length=64)
    tok_out["labels"] = [0] * len(batch["text"])
    return tok_out
ds_tok = ds.map(tok_and_label, batched=True)
# ----------------------------------------------------------------------------------------------------------------------
model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2)
trainer = transformers.Trainer(model=model, train_dataset=ds_tok)
trainer.train()
pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)