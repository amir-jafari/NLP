
import scipy.stats as stats
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# 1) Load a small portion of STS-B to keep training quick and reduce errors
dataset = load_dataset("glue", "stsb")
train_dataset = dataset["train"].select(range(100))
eval_dataset = dataset["validation"].select(range(100))

# 2) Prepare tokenizer and model for regression (num_labels=1)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# 3) Tokenize
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 4) Define custom Pearson correlation metric to avoid extra libraries
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    pearsonr, _ = stats.pearsonr(labels, predictions)
    return {"pearson": pearsonr}

# 5) Training arguments (short run for simplicity)
training_args = TrainingArguments(
    output_dir="test-stsb",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    seed=42
)

# 6) Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 7) Train and evaluate
trainer.train()
metrics = trainer.evaluate(eval_dataset)
print("Validation metrics:", metrics)
