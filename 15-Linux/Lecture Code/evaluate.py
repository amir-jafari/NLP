# ----------------------------------------------------------------------------------------------------------------------
import argparse, json, datasets, transformers, torch, numpy as np, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--model_dir"); ap.add_argument("--test_file"); ap.add_argument("--metrics_out")
args = ap.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
tok = transformers.AutoTokenizer.from_pretrained(args.model_dir)
model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model_dir)
ds = datasets.load_dataset("json", data_files=args.test_file, split="train")
ds_tok = ds.map(lambda ex: tok(ex["text"], truncation=True, padding="max_length", max_length=64),
                batched=True)
ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
loader = torch.utils.data.DataLoader(ds_tok.remove_columns("text"), batch_size=8)
acc_n = acc_d = 0
model.eval()
for batch in loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        preds = model(**batch).logits.argmax(-1).cpu().numpy()
    labels = np.zeros_like(preds)           # dummy 0-label for demo
    acc_n += (preds == labels).sum(); acc_d += len(preds)

metrics = {"accuracy": float(acc_n) / acc_d if acc_d else 0.0}
pathlib.Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
print(metrics)
